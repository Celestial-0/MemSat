"""Deep durable run lifecycle module."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from msat.memory.embed import Embedder
    from msat.providers.local import LocalReader

from msat.arms.negative import (
    ControlConfig,
    Verdict,
    go_no_go,
    render_control,
)
from msat.arms.negative import (
    evaluate as evaluate_arm_e,
)
from msat.arms.render import ArmRegistry, Memory, RenderedPrompt
from msat.gate import GateVerdict, gate
from msat.hardware.capacity import check_cell_capacity, preflight_capacity_report
from msat.hardware.vram import DEFAULT_HEADROOM_GB
from msat.metrics.normalisation import Normaliser, Step, token_f1
from msat.metrics.score import condition_key_str
from msat.planning import ExecutablePlan, verify_crossed_grid
from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
)
from msat.runner.executor import (
    Executor,
    Generator,
    RunConfig,
    RunResult,
)
from msat.runner.input import MissingInputError, RunInputs
from msat.runner.ladder import (
    LADDER,
    LadderState,
    Rung,
    RungName,
    check_order,
)
from msat.runner.ledger import EXCLUDED_FINISH_REASONS
from msat.runner.metadata import (
    ObservationMetadata,
    read_observation_metadata,
    write_bytes_artifact,
    write_json_artifact,
    write_observation_metadata,
)
from msat.runner.schedule import ScheduledUnit, build_schedule
from msat.runner.wal import WalState, read_wal
from msat.store import Condition, ResponseStore, StoredResponse, cache_key


class GateFailedError(RuntimeError):
    """Raised when an evaluated gate fails and halts execution."""


class CapacityRefusalError(RuntimeError):
    """Raised when cells exceed reader context limit and cannot be run."""


@dataclass(frozen=True, slots=True)
class CapacityVerdict:
    reader: str
    label: str
    prompt_tokens: int
    reader_capacity_tokens: int
    vram_capacity_tokens: int
    declared_context_window: int
    binding_constraint: str
    supported: bool
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "reader": self.reader,
            "label": self.label,
            "prompt_tokens": self.prompt_tokens,
            "reader_capacity_tokens": self.reader_capacity_tokens,
            "vram_capacity_tokens": self.vram_capacity_tokens,
            "declared_context_window": self.declared_context_window,
            "binding_constraint": self.binding_constraint,
            "supported": self.supported,
            "reason": self.reason,
            "overflow_by": max(0, self.prompt_tokens - self.reader_capacity_tokens),
        }


def default_normaliser() -> Normaliser:
    """Default fallback normaliser for token-f1 scoring."""
    return Normaliser(
        steps=(
            Step(id="lower", op="lowercase", params={}),
            Step(id="strip", op="strip", params={}),
            Step(id="ws", op="collapse_whitespace", params={"replacement": " "}),
        )
    )


def resolve_normaliser(bundle: Any) -> Normaliser:
    """Resolve explicit normalization; only dummy bundles may use the test default."""
    from msat.config import ConfigBundle

    raw = getattr(bundle, "raw", {}) or {}
    prompts = getattr(bundle, "prompts", {}) or {}
    spec = raw.get("prompts/normalisation.yaml") or raw.get("normalisation.yaml")
    if not spec:
        src = prompts.get("prompts/normalisation.yaml") or prompts.get("normalisation.yaml")
        if src:
            import yaml

            spec = yaml.safe_load(src)
    if spec is None and not isinstance(bundle, ConfigBundle):
        return default_normaliser()
    if not isinstance(spec, dict):
        raise ValueError("An explicit answer-normalisation configuration is required.")
    normaliser = Normaliser.from_config(spec)
    normaliser.normalise("")  # Validate operations and regex parameters before dispatch.
    return normaliser


def read_ladder(path: Path | str) -> LadderState:
    """Read ladder state from disk. Missing file returns empty state."""
    p = Path(path)
    if not p.is_file():
        return LadderState()
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return LadderState(passed=set(raw.get("passed", [])), notes=dict(raw.get("notes", {})))
    except (ValueError, OSError) as e:
        raise RuntimeError(f"{p} is unreadable ({e}); refusing to report ladder state") from e


def write_ladder(path: Path | str, state: LadderState) -> Path:
    """Write ladder state to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    write_json_artifact(p, state.as_dict())
    return p


def arm_e_thresholds(bundle: Any) -> dict[str, float] | None:
    """Extract measurable Arm E pass thresholds from bundle prompts."""
    source = (getattr(bundle, "prompts", {}) or {}).get("prompts/arm_e.j2") or (
        getattr(bundle, "prompts", {}) or {}
    ).get("arm_e.j2", "")
    if not source:
        return None
    cfg = ControlConfig.from_template(source)
    if not cfg.thresholds_measurable:
        return None
    return {
        "accuracy_below": float(cfg.pass_threshold_accuracy_below or 0.0),
        "recall_below": float(cfg.pass_threshold_recall_below or 0.0),
    }


def evaluate_gate(
    out_dir: Path,
    *,
    context: str,
    run_id: str,
    config_hash: str,
    notes: Sequence[str] = (),
    **inputs: Any,
) -> GateVerdict:
    """Evaluate GO/NO-GO gate and persist verdict."""
    verdict = gate(
        accuracy_by_cell=inputs.pop("accuracy_by_cell", {}),
        recall_by_k=inputs.pop("recall_by_k", {}),
        stratified=inputs.pop("stratified", None),
        degenerate_cells=inputs.pop("degenerate_cells", []),
        judge_floor_pp=inputs.pop("judge_floor_pp", 0.0),
        n_questions=inputs.pop("n_questions", 0),
        **inputs,
    )
    verdict.notes.extend(notes)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        **verdict.as_dict(),
        "context": context,
        "run_id": run_id,
        "config_hash": config_hash,
    }
    write_json_artifact(out_dir / "go_no_go.json", payload)
    return verdict


def check_capacity(reader: Any, prompt_tokens: int, *, label: str = "") -> CapacityVerdict:
    """Check whether a reader can serve the prompt under R3.2 capacity limit."""
    card_gb = float(getattr(reader, "card_gb", 8.0))
    headroom_gb = float(getattr(reader, "headroom_gb", DEFAULT_HEADROOM_GB))
    v = check_cell_capacity(reader, prompt_tokens, card_gb=card_gb, headroom_gb=headroom_gb)
    reason = v.reason
    if not v.supported and "cannot serve" not in reason:
        reason = (
            f"R3.2: reader {reader.id} cannot serve {prompt_tokens} prompt tokens: {v.reason}. "
            "Truncating would leave the declared condition inconsistent with its actual prompt."
        )
    if v.supported:
        reason = f"{prompt_tokens:,} prompt tokens fit the {v.reader_capacity_tokens:,}-token reader capacity"
    return CapacityVerdict(
        reader=reader.id,
        label=label,
        prompt_tokens=prompt_tokens,
        reader_capacity_tokens=v.reader_capacity_tokens,
        vram_capacity_tokens=v.vram_capacity_tokens,
        declared_context_window=v.declared_context_window,
        binding_constraint=v.binding_constraint,
        supported=v.supported,
        reason=reason,
    )


def crossing(plan: Any, bundle: Any) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    """Verify crossed (K, B) grid estimability and count estimable interaction cells."""
    design = bundle.design
    report = verify_crossed_grid(
        plan,
        k_grid=design.K_grid,
        b_grid=design.b_values,
        scales=design.scales,
        readers=design.readers,
    )
    per_block = len(design.K_grid) * len(design.b_values)
    n_interaction_cells = per_block * sum(
        1 for b in report["per_scale_reader"].values() if b["complete"]
    )
    degenerate: list[dict[str, Any]] = []
    for block, row in sorted(report["per_scale_reader"].items()):
        for k, b in row["missing"]:
            degenerate.append({"scale_reader": block, "K": k, "B": b, "reason": "missing cell"})
        for k, b in row["extra"]:
            degenerate.append({"scale_reader": block, "K": k, "B": b, "reason": "undeclared cell"})
    return report, n_interaction_cells, degenerate


def capacity_preflight(bundle: Any, plan: Any, panel: Sequence[LocalReader]) -> dict[str, Any]:
    """Preflight capacity check across all plan cells delegating to hardware.capacity."""
    plan_conditions = getattr(plan, "conditions", None) or getattr(plan, "cells", ())
    if not plan or not plan_conditions:
        return {
            "n_checked": 0,
            "n_supported": 0,
            "n_unsupported": 0,
            "all_fit": True,
            "per_reader": {},
            "unsupported": [],
        }

    report = preflight_capacity_report(bundle, plan, panel)
    report["n_checked"] = report["n_cells"]
    report["all_fit"] = report["n_unsupported"] == 0
    return report


def _generation_request(
    bundle: Any, reader: LocalReader, rendered: RenderedPrompt
) -> GenerationRequest:
    decoding = getattr(bundle, "decoding", None)
    max_tokens = int(getattr(decoding, "max_tokens", 512)) if decoding else 512
    temperature = float(getattr(decoding, "temperature", 0.7)) if decoding else 0.7
    top_p = float(getattr(decoding, "top_p", 0.95)) if decoding else 0.95
    stop = tuple(str(s) for s in (getattr(decoding, "stop_sequences", ()) or ()))
    seed = int(getattr(decoding, "provider_seed", 0)) if decoding else 0

    return GenerationRequest(
        model=reader.model,
        prompt=rendered.user,
        system=rendered.system,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=stop,
        seed=seed,
        request_id=rendered.arm,
    )


@dataclass
class RungReport:
    """Summary of a completed rung execution."""

    rung: RungName
    passed: bool
    run_result: RunResult
    gate_verdict: GateVerdict
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "rung": self.rung,
            "passed": self.passed,
            "run_result": self.run_result.as_dict(),
            "gate_verdict": self.gate_verdict.as_dict(),
            "details": self.details,
        }


class RunLifecycle:
    """Owns execution lifecycle: inputs, rendering, Executor adaptation, ladder & gates."""

    def __init__(
        self,
        *,
        run_dir: Path | str,
        bundle: Any,
        registry: ArmRegistry,
        panel: Sequence[LocalReader],
        inputs: RunInputs,
        run_id: str = "run",
        plan: ExecutablePlan | None = None,
        config: RunConfig | None = None,
        resume: bool = False,
        seed: int | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.bundle = bundle
        self.registry = registry
        self.panel = tuple(panel)
        self.inputs = inputs
        if self.inputs.pool is None and self.inputs.questions:
            self.inputs.build_control_pool(scale=LADDER[0].scales[0])
        self.run_id = run_id
        self.plan = plan
        self.resume = resume

        # Default seed from bundle.design.seed
        design_seed = getattr(getattr(bundle, "design", None), "seed", None)
        bundle_seed = getattr(bundle, "seed", 20261012)
        self.seed = (
            seed if seed is not None else (design_seed if design_seed is not None else bundle_seed)
        )

        self.config = config or RunConfig(
            run_dir=self.run_dir,
            seed=self.seed,
            runs_per_cell=getattr(bundle.design, "runs_per_cell", 1)
            if hasattr(bundle, "design")
            else 1,
        )
        self.readers_by_id: dict[str, LocalReader] = {r.id: r for r in self.panel}
        self.store = ResponseStore(self.config.responses_dir)
        self.ladder_path = self.run_dir / "ladder.json"
        self._rendered_cache: dict[ScheduledUnit, RenderedPrompt] = {}
        self._indices: dict[tuple[str, str], Any] = {}
        self._backend_failure: tuple[Any, Exception] | None = None
        self._generator_failures: list[tuple[ScheduledUnit, Any, Exception]] = []

        self.embedder = embedder
        if self.embedder is None and (self.inputs.is_synthetic or self.inputs.allow_test_adapters):
            from msat.memory.embed import HashEmbedder

            self.embedder = HashEmbedder()

        # Load existing observation metadata from run_dir to preserve across restarts/rungs
        self._observations: dict[str, ObservationMetadata] = {}
        for obs in read_observation_metadata(self.run_dir / "observations.parquet"):
            key = f"{obs.condition_key}|{obs.qid}|{obs.replicate}"
            self._observations[key] = obs

    def get_ladder_state(self) -> LadderState:
        return read_ladder(self.ladder_path)

    def _resolve_embedder(self) -> Embedder:
        if self.embedder is not None:
            return self.embedder
        retrieval = getattr(self.bundle, "retrieval", None)
        config = getattr(retrieval, "embedder", None) or getattr(self.bundle, "raw", {}).get(
            "retrieval.yaml", {}
        ).get("embedder")
        if config is None:
            raise MissingInputError("No configured embedder found; live retrieval is refused.")

        def value(name: str, default: Any = None) -> Any:
            return (
                config.get(name, default)
                if isinstance(config, dict)
                else getattr(config, name, default)
            )

        model_name, revision = value("id"), value("version")
        if not model_name or not revision or revision == "UNCONFIRMED":
            raise MissingInputError(
                "Live retrieval requires a pinned embedder identity and revision."
            )
        from msat.memory.embed import SentenceTransformerEmbedder

        try:
            self.embedder = SentenceTransformerEmbedder(
                model_name=str(model_name),
                revision=str(revision),
                device=str(value("device", "cuda")),
                batch_size=int(value("batch_size", 16)),
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            raise MissingInputError(
                f"Configured embedder {model_name!r} is unavailable: {exc}"
            ) from exc
        return self.embedder

    def _retrieval_unit(self) -> str:
        retrieval = getattr(self.bundle, "retrieval", None)
        return str(
            getattr(retrieval, "unit", None)
            or getattr(self.bundle, "raw", {}).get("retrieval.yaml", {}).get("unit", "session")
        )

    def _get_or_create_index(self, q: Any) -> Any:
        scale = q.scale or "S"
        key = (scale, q.qid)
        if key in self._indices:
            return self._indices[key]
        from msat.memory.write import write as memory_write_fn

        session_tuples = [
            (s.session_id, " ".join(t.content for t in s.turns if t.content).strip())
            for s in q.sessions
        ]
        index, _ = memory_write_fn(
            q.qid, session_tuples, embedder=self._resolve_embedder(), unit=self._retrieval_unit()
        )
        self._indices[key] = index
        return index

    def _prepare_schedule(self, schedule: Sequence[ScheduledUnit]) -> None:
        if not schedule:
            raise MissingInputError("No observations are scheduled for execution.")
        digest = hashlib.sha256()

        def stamp(payload: Any) -> None:
            digest.update(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8"))

        for (scale, qid), question in sorted(self.inputs.questions_by_scale.items()):
            stamp(
                [
                    scale,
                    qid,
                    question.question,
                    question.gold_answers,
                    question.gold_session_ids,
                    question.ability,
                ]
            )
            for session in question.sessions:
                stamp(
                    [
                        session.session_id,
                        session.date,
                        [(turn.role, turn.content, turn.has_answer) for turn in session.turns],
                    ]
                )
        stamp(
            [
                self.inputs.question_text,
                self.inputs.gold_answers,
                self.inputs.gold_sessions,
                self.inputs.filler_sessions,
            ]
        )
        if self.inputs.pool is not None:
            for qid, memories in sorted(self.inputs.pool.by_question.items()):
                for memory in memories:
                    stamp([qid, memory.memory_id, memory.session_id, memory.text, memory.n_tokens])
        identity = {
            "config_hash": str(getattr(self.bundle, "config_hash", "")),
            "seed": self.seed,
            "inputs_hash": digest.hexdigest(),
        }
        path = self.run_dir / "run_identity.json"
        if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != identity:
            raise ValueError(
                "Run configuration or question inputs changed; use the original inputs for resume."
            )
        for unit in schedule:
            rendered = self._render_unit(unit)
            reader = self.readers_by_id[unit.condition.reader]
            cpt = float(
                getattr(self.bundle, "pricing", {})
                .get("token_estimates", {})
                .get("chars_per_token", 4.0)
            )
            verdict = check_capacity(reader, int(len(rendered.prompt) / cpt) + 1)
            if not verdict.supported:
                raise CapacityRefusalError(verdict.reason)
        if not path.exists():
            write_json_artifact(path, identity)

    def _render_unit(self, unit: ScheduledUnit) -> RenderedPrompt:
        if unit in self._rendered_cache:
            return self._rendered_cache[unit]

        arm_id = unit.condition.arm
        cond = unit.condition
        qid = unit.qid
        repetition = unit.repetition
        scale = cond.scale

        question_text = self.inputs.get_question_text(qid, scale=scale)
        gold_sessions = self.inputs.get_gold_sessions(qid, scale=scale)

        if arm_id == "E":
            # Arm E: Negative Control
            pool = self.inputs.pool or self.inputs.build_control_pool()
            if not pool or not pool.by_question:
                raise MissingInputError("Arm E requires non-empty control pool.")
            control_arm = self.registry.get("E")
            raw_prompts = getattr(self.bundle, "prompts", {}) or {}
            template_src = raw_prompts.get("prompts/arm_e.j2") or raw_prompts.get("arm_e.j2", "")
            ctrl_cfg = (
                ControlConfig.from_template(template_src) if template_src else ControlConfig()
            )

            from dataclasses import replace

            seed_val = self.seed + repetition
            if seed_val != ctrl_cfg.seed:
                ctrl_cfg = replace(ctrl_cfg, seed=seed_val)

            gold_by_q = {
                q_id: frozenset(self.inputs.get_gold_sessions(q_id, scale=scale))
                for q_id in set(pool.by_question) | set(self.inputs.qids_by_scale().get(scale, ()))
            }
            rendered = render_control(
                control_arm,
                question_text,
                pool,
                qid,
                gold_sessions,
                cond,
                ctrl_cfg,
                budget_tokens=int(cond.b) if cond.b_mode == "per_memory_tokens" else None,
                qid=qid,
                gold_by_question=gold_by_q,
            )
            from dataclasses import replace

            target_count = len(pool.by_question.get(qid, ()))
            rendered = replace(
                rendered,
                corpus_fraction=min(cond.k, target_count) / target_count if target_count else 0.0,
            )
            self._rendered_cache[unit] = rendered
            return rendered

        # All baseline and full arms require actual question sessions; placeholder generation is forbidden
        q = self.inputs.get_question(qid, scale=scale)
        if q is None or not q.sessions:
            raise MissingInputError(
                f"Question {qid!r} on scale {scale!r} has no sessions available for arm {arm_id!r}. "
                "Refusing to manufacture placeholder sessions."
            )

        if arm_id in ("D", "D-max"):
            # Arm D: Full Context Baseline
            full_arm = self.registry.get(arm_id)
            reader = self.readers_by_id[cond.reader]
            memories = [
                Memory(
                    memory_id=s.session_id,
                    session_id=s.session_id,
                    source_question_id=qid,
                    text=" ".join(t.content for t in s.turns if t.content).strip(),
                    n_tokens=max(1, s.n_chars // 4),
                )
                for s in q.sessions
            ]
            corpus_tokens = sum(m.n_tokens for m in memories)
            ceiling = reader.reader_capacity_tokens
            if arm_id == "D-max":
                from msat.arms.max_context import fit_to_ceiling

                reserve = int(getattr(getattr(self.bundle, "decoding", None), "max_tokens", 512))
                reserve += int(
                    getattr(self.bundle, "pricing", {}).get("prompt_overhead_tokens", 500)
                )
                fit = fit_to_ceiling(
                    reader=reader.id,
                    ceiling=ceiling,
                    mean_session_tokens=corpus_tokens / len(memories),
                    n_sessions_available=len(memories),
                    reserve_tokens=reserve,
                )
                rendered = full_arm.render(
                    question_text, memories, cond, fit=fit, gold_ids=tuple(gold_sessions), qid=qid
                )
                if rendered.n_memories == 0:
                    raise CapacityRefusalError(
                        "D-max cannot fit any complete session in the available context budget."
                    )
            else:
                from msat.arms.baselines import FullContextTruncationRefusedError

                try:
                    rendered = full_arm.render(
                        question_text,
                        memories,
                        cond,
                        corpus_tokens=corpus_tokens,
                        ceiling=ceiling,
                        qid=qid,
                    )
                except FullContextTruncationRefusedError as exc:
                    raise CapacityRefusalError(str(exc)) from exc
        elif arm_id == "F":
            # Arm F: Oracle Gold Sessions
            if not gold_sessions:
                raise MissingInputError(
                    f"Question {qid!r} has no gold sessions declared for oracle arm F."
                )
            gold_s = [s for s in q.sessions if s.session_id in gold_sessions]
            if not gold_s:
                raise MissingInputError(
                    f"Question {qid!r} declares gold sessions {gold_sessions} but none exist in actual sessions."
                )
            memories = [
                Memory(
                    memory_id=s.session_id,
                    session_id=s.session_id,
                    source_question_id=qid,
                    text=" ".join(t.content for t in s.turns if t.content).strip(),
                    n_tokens=max(1, s.n_chars // 4),
                )
                for s in gold_s
            ]
            arm = self.registry.get("F")
            rendered = arm.render(question_text, memories, cond, budget_tokens=None, qid=qid)
        elif arm_id in ("A", "G", "A+G", "B", "C"):
            # Configured retrieval with pinned embedder & memory index
            from msat.memory.retrieve import retrieve

            index = self._get_or_create_index(q)
            query_vec = self._resolve_embedder().encode([question_text])[0]
            budget_tokens = int(cond.b) if cond.b_mode == "per_memory_tokens" else None
            retrieved = retrieve(
                index,
                question_text,
                query_vec,
                k=cond.k,
                unit=self._retrieval_unit(),
                budget_tokens=budget_tokens,
            )
            memories = [
                Memory(
                    memory_id=index.units[index.unit_index[uid]].unit_id,
                    session_id=index.units[index.unit_index[uid]].source_id,
                    source_question_id=qid,
                    text=index.units[index.unit_index[uid]].text,
                    n_tokens=index.units[index.unit_index[uid]].n_tokens,
                )
                for uid in retrieved.unit_ids
            ]
            if len(memories) != cond.k:
                raise MissingInputError(
                    f"Question {qid!r} on scale {scale!r} supplies {len(memories)} retrieval "
                    f"memories for declared K={cond.k}; the condition cannot be executed."
                )

            if arm_id == "A+G":
                treatment_arm = (
                    self.registry.get("G") if "G" in self.registry.arms else self.registry.get("A")
                )
                rendered = treatment_arm.render(
                    question_text,
                    memories,
                    cond,
                    budget_tokens=budget_tokens,
                    qid=qid,
                )
                from dataclasses import replace

                rendered = replace(rendered, arm="A+G")
            elif arm_id == "G":
                arm = self.registry.get("G")
                rendered = arm.render(
                    question_text,
                    memories,
                    cond,
                    budget_tokens=budget_tokens,
                    qid=qid,
                )
            elif arm_id == "A":
                arm = self.registry.get("A")
                rendered = arm.render(
                    question_text,
                    memories,
                    cond,
                    budget_tokens=budget_tokens,
                    qid=qid,
                )
            elif arm_id == "B":
                # Arm B: Filler padding with gold exclusion
                if not self.inputs.filler_sessions:
                    from msat.arms.padding import FillerMissingError

                    raise FillerMissingError("Arm B requires filler sessions but none are loaded.")
                from msat.arms.padding import FillerSource, PaddingBudget, pad_to_length

                # Filter out gold sessions to strictly guarantee no gold leakage
                corpus_tokens = sum(max(1, s.n_chars // 4) for s in q.sessions)
                if cond.b_mode != "corpus_fraction":
                    raise ValueError(
                        "Arm B requires an explicit corpus_fraction budget; no budget is substituted."
                    )
                fraction = cond.b
                budget = PaddingBudget(corpus_tokens=corpus_tokens, fraction=fraction)
                filler_src = FillerSource(
                    sessions=self.inputs.filler_sessions, path=Path("5_filler_sess.json")
                )
                padded_memories, _ = pad_to_length(
                    memories, budget, filler_src, gold_sessions=tuple(gold_sessions)
                )
                arm = self.registry.get("B")
                rendered = arm.render(question_text, padded_memories, cond, qid=qid)
            elif arm_id == "C":
                # Arm C: Balanced position order
                from msat.arms.position import VARIANTS

                variant = VARIANTS[repetition % len(VARIANTS)]
                shuffle_seed = self.seed + repetition
                arm = self.registry.get("C")
                rendered = arm.render(
                    question_text,
                    memories,
                    cond,
                    variant=variant,
                    shuffle_seed=shuffle_seed,
                    gold_sessions=tuple(gold_sessions),
                    qid=qid,
                )
            else:
                arm = self.registry.get(arm_id)
                rendered = arm.render(question_text, memories, cond, qid=qid)
        else:
            arm = self.registry.get(arm_id)
            memories = [
                Memory(
                    memory_id=s.session_id,
                    session_id=s.session_id,
                    source_question_id=qid,
                    text=" ".join(t.content for t in s.turns if t.content).strip(),
                    n_tokens=max(1, s.n_chars // 4),
                )
                for s in q.sessions[: max(1, cond.k)]
            ]
            rendered = arm.render(question_text, memories, cond, qid=qid)

        from dataclasses import replace

        rendered = replace(
            rendered, corpus_fraction=len(set(rendered.session_ids)) / len(q.sessions)
        )
        self._rendered_cache[unit] = rendered
        return rendered

    def _render_request_body(self, unit: ScheduledUnit) -> dict[str, Any]:
        """Produce the canonical request dictionary for cache-key stability."""
        rendered = self._render_unit(unit)
        reader = self.readers_by_id[unit.condition.reader]
        req = _generation_request(self.bundle, reader, rendered)
        return req.canonical_body()

    def _record_observation(
        self, unit: ScheduledUnit, stored: StoredResponse
    ) -> ObservationMetadata:
        reader = self.readers_by_id[unit.condition.reader]
        rendered = self._render_unit(unit)
        gold_set = self.inputs.get_gold_sessions(unit.qid, scale=unit.condition.scale)
        gold_ranks = list(rendered.gold_ranks(gold_set))
        scalar_gold_rank = gold_ranks[0] if gold_ranks else None
        q = self.inputs.get_question(unit.qid, scale=unit.condition.scale)
        ability = getattr(q, "question_type", None) or getattr(q, "ability", None) or "unknown"
        question_type = str(getattr(q, "question_type", ability))

        canonical_key = condition_key_str(stored)
        from msat.arms.conditions import prompt_digest

        real_prompt_hash = prompt_digest(rendered)
        cfg_hash = str(getattr(self.bundle, "config_hash", ""))
        canonical_cfg_hash = str(getattr(self.bundle, "canonical_config_hash", None) or cfg_hash)

        obs = ObservationMetadata(
            config_hash=cfg_hash,
            condition_key=canonical_key,
            arm=unit.condition.arm,
            k=unit.condition.k,
            b=unit.condition.b,
            b_mode=unit.condition.b_mode,
            scale=unit.condition.scale,
            reader=unit.condition.reader,
            qid=unit.qid,
            repetition=unit.repetition,
            memory_ranks=list(rendered.ranks),
            gold_ranks=gold_ranks,
            gold_rank=scalar_gold_rank,
            corpus_fraction=rendered.corpus_fraction,
            ability=ability,
            question_type=question_type,
            reader_strength=reader.reader_strength,
            prompt_sha256=real_prompt_hash,
            cache_key=stored.key,
            canonical_config_hash=canonical_cfg_hash,
        )
        uid = f"{canonical_key}|{unit.qid}|{unit.repetition}"
        self._observations[uid] = obs
        return obs

    def _make_generator(self) -> Generator:
        chars_per_token = float(
            (getattr(self.bundle, "pricing", {}) or {})
            .get("token_estimates", {})
            .get("chars_per_token", 4.0)
        )

        def generator(unit: ScheduledUnit, *, attempt: int) -> StoredResponse:
            reader = self.readers_by_id[unit.condition.reader]
            rendered = self._render_unit(unit)

            # R3.2 capacity check before generation
            prompt_tokens = int(len(rendered.prompt) / chars_per_token) + 1
            verdict = check_capacity(
                reader, prompt_tokens, label=f"{unit.condition.arm}/{unit.qid}"
            )
            if not verdict.supported:
                raise CapacityRefusalError(
                    f"preflight capacity failed for {unit}: {verdict.reason}"
                )

            gen_req = _generation_request(self.bundle, reader, rendered)
            try:
                gen_res = reader.generate(gen_req)
            except BackendUnavailableError as exc:
                self._backend_failure = (reader, exc)
                self._generator_failures.append((unit, reader, exc))
                raise
            except Exception as exc:
                self._generator_failures.append((unit, reader, exc))
                raise

            body = gen_req.canonical_body()
            key = cache_key(unit.condition, body)

            stored = StoredResponse(
                key=key,
                condition=unit.condition.as_dict(),
                request=body,
                text=gen_res.text,
                finish_reason=gen_res.finish_reason,
                model_version=gen_res.model_version or reader.version,
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
                latency_ms=gen_res.latency_ms,
                ttft_ms=gen_res.ttft_ms,
                queue_wait_ms=gen_res.queue_wait_ms,
                backend=gen_res.backend or reader.backend_name,
                raw=gen_res.raw,
                qid=unit.qid,
                repetitions=(unit.repetition,),
            )
            self._record_observation(unit, stored)
            return stored

        return generator

    def _execute_schedule(self, schedule: Sequence[ScheduledUnit]) -> RunResult:
        self._prepare_schedule(schedule)
        executor = Executor(
            self.config, self._make_generator(), self._render_request_body, resume=self.resume
        )
        try:
            result = executor.run(schedule)
        finally:
            executor.ledger.write(self.config.ledger_path)
            self._flush_conditions_and_metadata(schedule)
        if self._backend_failure is not None:
            reader, error = self._backend_failure
            self._write_backend_failure(reader, error, result.n_generated, len(schedule))
            raise BackendUnavailableError(str(error)) from error
        return result

    def _write_backend_failure(
        self, reader: Any, error: Exception, n_generated: int = 0, n_checked: int = 0
    ) -> None:
        e_thresh = arm_e_thresholds(self.bundle)
        thresholds = {
            "accuracy": e_thresh.get("accuracy_below") if e_thresh else None,
            "recall": e_thresh.get("recall_below") if e_thresh else None,
        }
        write_json_artifact(
            self.run_dir / "arm_e_control.json",
            {
                "arm": "E",
                "verdict": None,
                "proceed": False,
                "accuracy": None,
                "recall": None,
                "n_generated": n_generated,
                "capacity": {
                    "on_overflow": "reject_cell",
                    "n_checked": n_checked,
                    "n_supported": n_checked,
                    "n_accepted": n_checked,
                },
                "backend": {
                    "available": False,
                    "reader": reader.id,
                    "backend": reader.backend_name,
                    "error": type(error).__name__,
                    "detail": str(error),
                    "n_generated_before_failure": n_generated,
                },
                "thresholds": thresholds,
                "reason": f"Backend unavailable ({type(error).__name__}: {error}). Arm E cannot be evaluated.",
                "diagnostics": {
                    "backend_available": False,
                    "thresholds_measurable": e_thresh is not None,
                    "accuracy_is_none_because": f"backend unavailable: {error}",
                    "causal_detail": str(error),
                },
            },
        )

    def run_rung(self, rung_name: RungName) -> RungReport:
        """Execute a declared ladder rung through the durable lifecycle."""
        rung = next(r for r in LADDER if r.name == rung_name)
        state = self.get_ladder_state()
        check_order(state, rung_name)

        if rung_name == "negative":
            return self._run_negative_rung(rung, state)
        if rung_name == "baselines":
            return self._run_baselines_rung(rung, state)
        if rung_name == "crossed":
            return self._run_full_rung(rung, state)
        raise ValueError(f"unknown rung {rung_name}")

    def _run_negative_rung(self, rung: Rung, state: LadderState) -> RungReport:
        # Check backend readiness before dispatch
        for reader in self.panel:
            health = reader.health()
            if not health.get("ready", True):
                engine = health.get("engine", {})
                reason = engine.get("reason") or "backend is not ready"
                self._write_backend_failure(reader, BackendUnavailableError(reason))
                raise BackendUnavailableError(f"reader {reader.id!r} backend unavailable: {reason}")

        # Check thresholds
        e_thresh = arm_e_thresholds(self.bundle)
        if e_thresh is None and not self.inputs.is_synthetic:
            evaluate_gate(
                self.run_dir,
                context=f"{self.run_id}/arm_e_thresholds",
                run_id=self.run_id,
                config_hash=getattr(self.bundle, "config_hash", ""),
                notes=["Arm E pass thresholds null; negative control cannot be evaluated."],
            )
            raise GateFailedError("Arm E pass thresholds are null; run cannot proceed.")

        # Require inputs
        if not self.inputs.is_synthetic and not self.inputs.allow_test_adapters:
            missing_inputs: list[str] = []
            if not self.inputs.questions and not self.inputs.gold_sessions:
                missing_inputs.append("--questions")
            has_gold = bool(self.inputs.gold_answers) or any(
                self.inputs.get_gold_answers(qid) for qid in self.inputs.questions
            )
            if not has_gold:
                missing_inputs.append("--gold")
            pool = self.inputs.pool or self.inputs.build_control_pool()
            if not pool or not pool.by_question:
                missing_inputs.append("--arm-e-pool")
            if missing_inputs:
                raise MissingInputError(
                    f"{', '.join(missing_inputs)} is missing input; supply the control inputs or --data with benchmark files."
                )

        pool = self.inputs.pool or self.inputs.build_control_pool()
        if not pool or not pool.by_question:
            raise MissingInputError("Arm E requires non-empty control pool.")

        normaliser = resolve_normaliser(self.bundle)

        # Build schedule for Arm E
        scale = rung.scales[0]
        k0 = int(self.bundle.design.K_grid[0])
        b0 = self.bundle.design.B_grid[0]
        cells = [
            Condition("E", k0, float(b0.value), str(b0.mode), scale, reader.id)
            for reader in self.panel
        ]
        qids = sorted(
            self.inputs.qids_by_scale().get(scale, list(self.inputs.gold_sessions.keys()))
        )
        if rung.name == "negative":
            qids = [
                qid
                for qid in qids
                if self.inputs.get_gold_sessions(qid, scale) and qid in pool.by_question
            ]
        if rung.n_questions > 0:
            qids = qids[: rung.n_questions]
        if not qids:
            raise MissingInputError("no questions available for negative control rung")

        unscorable = [
            qid
            for qid in qids
            if not any(
                normaliser.normalise(gold).strip()
                for gold in self.inputs.get_gold_answers(qid, scale)
            )
        ]
        if unscorable:
            raw_prompts = getattr(self.bundle, "prompts", {}) or {}
            source = raw_prompts.get("prompts/arm_e.j2") or raw_prompts.get("arm_e.j2", "")
            control_config = ControlConfig.from_template(source) if source else ControlConfig()
            outcome = evaluate_arm_e(None, None, control_config, n_questions=0)
            report = {
                **go_no_go(outcome),
                "arm": "E",
                "n_cells": len(cells),
                "n_questions": len(qids),
                "accuracy": None,
                "recall": None,
                "coverage_complete": False,
                "n_generated": 0,
                "n_scored": 0,
                "unscorable_qids": unscorable,
                "diagnostics": {
                    "thresholds_measurable": control_config.thresholds_measurable,
                    "accuracy_is_none_because": "no usable control output was scored",
                    "n_unattributed": len(unscorable),
                    "reason": "Gold answers are unavailable; generation was refused before dispatch.",
                },
            }
            write_json_artifact(self.run_dir / "arm_e_control.json", report)
            evaluate_gate(
                self.run_dir,
                context=f"{self.run_id}/arm_e",
                run_id=self.run_id,
                config_hash=getattr(self.bundle, "config_hash", ""),
                n_interaction_cells=0,
                notes=["Arm E did NOT collapse: missing gold answers prevent evaluation."],
            )
            raise GateFailedError("Arm E is INCONCLUSIVE: gold answers are unavailable.")

        schedule = build_schedule(cells, qids, runs_per_cell=1, seed=self.seed)
        if not schedule:
            raise MissingInputError("empty schedule generated for negative control rung")

        # Capacity preflight before execution
        chars_per_token = float(
            (getattr(self.bundle, "pricing", {}) or {})
            .get("token_estimates", {})
            .get("chars_per_token", 4.0)
        )
        for unit in schedule:
            prompt = self._render_unit(unit)
            prompt_tokens = int(len(prompt.prompt) / chars_per_token) + 1
            verdict = check_capacity(self.readers_by_id[unit.condition.reader], prompt_tokens)
            if not verdict.supported:
                raise CapacityRefusalError(
                    f"preflight capacity failed for {unit}: {verdict.reason}"
                )

        # Execute via Executor: durable WAL + store + ledger
        run_res = self._execute_schedule(schedule)

        # Durable coverage verification: required complete observations with usable finish reasons
        scores: list[float] = []
        recalls: list[float] = []
        draws: list[dict[str, Any]] = []

        missing_units: list[str] = []
        unusable_finish: list[str] = []

        for unit in schedule:
            key = cache_key(unit.condition, self._render_request_body(unit))
            stored = self.store.get(key)
            if stored is None:
                missing_units.append(f"{unit.condition.key}|{unit.qid}")
                continue
            if stored.finish_reason in EXCLUDED_FINISH_REASONS or stored.finish_reason != "stop":
                unusable_finish.append(f"{unit.qid}:{stored.finish_reason}")

            rendered = self._render_unit(unit)
            gold_set = self.inputs.get_gold_sessions(unit.qid, scale=unit.condition.scale)
            hits = sorted(set(rendered.session_ids) & gold_set)
            draws.append(
                {
                    "label": f"E/{unit.condition.scale}/{unit.condition.reader}/{unit.qid}",
                    "n_drawn": rendered.n_memories,
                    "count_matched_to_K": rendered.n_memories == unit.condition.k,
                    "session_ids": list(rendered.session_ids),
                    "gold_sessions_drawn": hits,
                }
            )
            if gold_set:
                recalls.append(len(hits) / len(gold_set))

            golds = self.inputs.get_gold_answers(unit.qid, scale=unit.condition.scale)
            if golds:
                scores.append(max(token_f1(stored.text, g, normaliser) for g in golds))
            else:
                raise MissingInputError(f"Missing gold answers for question {unit.qid!r}")

        # Failure detection from actual required durable coverage
        coverage_complete = (
            len(missing_units) == 0
            and len(unusable_finish) == 0
            and run_res.n_failed == 0
            and len(scores) == len(schedule)
        )

        # Extract generator failure diagnostics if any
        generator_failed = run_res.n_failed > 0 or bool(self._generator_failures)
        causal_error: str | None = None
        causal_detail: str | None = None
        failed_reader: LocalReader | None = None

        if self._generator_failures:
            _, failed_reader, last_exc = self._generator_failures[-1]
            causal_error = type(last_exc).__name__
            causal_detail = str(last_exc)
        elif self.config.ledger_path.is_file():
            ledger_data = json.loads(self.config.ledger_path.read_text(encoding="utf-8"))
            exclusions = ledger_data.get("exclusions", [])
            if exclusions:
                causal_detail = exclusions[-1].get("detail")
                causal_error = "ExecutionError"
        elif self.config.wal_path.is_file():
            for entry in read_wal(self.config.wal_path):
                if entry.state is WalState.FAILED and entry.detail:
                    causal_detail = entry.detail
                    causal_error = "ExecutionError"

        accuracy = sum(scores) / len(scores) if (scores and coverage_complete) else None
        recall = sum(recalls) / len(recalls) if (recalls and coverage_complete) else None

        raw_prompts = getattr(self.bundle, "prompts", {}) or {}
        template_src = raw_prompts.get("prompts/arm_e.j2") or raw_prompts.get("arm_e.j2", "")
        ctrl_cfg = ControlConfig.from_template(template_src) if template_src else ControlConfig()

        outcome = evaluate_arm_e(accuracy, recall, ctrl_cfg, n_questions=len(scores))

        backend_info: dict[str, Any] = {
            "available": not generator_failed,
            "backend": (
                failed_reader.backend_name if failed_reader else self.panel[0].backend_name
            ),
            "reader": failed_reader.id if failed_reader else self.panel[0].id,
        }
        if generator_failed:
            if causal_error:
                backend_info["error"] = causal_error
            if causal_detail:
                backend_info["detail"] = causal_detail
            backend_info["n_failed"] = run_res.n_failed

        ctrl_report = {
            **go_no_go(outcome),
            "arm": "E",
            "n_cells": len(cells),
            "n_questions": len(qids),
            "accuracy": accuracy,
            "recall": recall,
            "coverage_complete": coverage_complete,
            "missing_units": missing_units,
            "unusable_finish": unusable_finish,
            "draws": draws,
            "n_generated": run_res.n_generated,
            "n_scored": len(scores),
            "n_refused": len(missing_units) + len(unusable_finish),
            "capacity": {
                "on_overflow": "reject_cell",
                "n_checked": len(schedule),
                "n_accepted": len(schedule),
            },
            "backend": backend_info,
            "diagnostics": {
                "thresholds_measurable": ctrl_cfg.thresholds_measurable,
                "accuracy_is_none_because": (
                    f"generator failure prevented response generation: {causal_detail}"
                    if generator_failed
                    else (
                        "incomplete measured output: missing or unusable responses"
                        if not coverage_complete
                        else ""
                    )
                ),
                "causal_detail": causal_detail,
                "n_failed": run_res.n_failed,
                "missing_units": missing_units,
                "unusable_finish": unusable_finish,
            },
        }
        if generator_failed and causal_detail:
            ctrl_report["reason"] = (
                f"generator failed to produce responses ({causal_detail}). {outcome.reason}"
            )
        write_json_artifact(self.run_dir / "arm_e_control.json", ctrl_report)

        passed = bool(coverage_complete and outcome.verdict == Verdict.PASS)
        _, n_interaction_cells, degenerate_cells = (
            crossing(self.plan, self.bundle) if self.plan else ({}, 1, [])
        )
        if passed:
            gate_notes = [f"Arm E negative control: {outcome.verdict} — {outcome.reason}"]
        elif generator_failed:
            gate_notes = [
                f"Arm E execution failure: generator failed to produce responses ({causal_detail}). "
                f"Incomplete measured output prevents negative control evaluation."
            ]
        elif not coverage_complete:
            gate_notes = [
                f"Arm E incomplete output: {outcome.reason}. "
                f"Incomplete measured output prevents negative control evaluation."
            ]
        else:
            gate_notes = [f"Arm E did NOT collapse: {outcome.verdict} — {outcome.reason}"]

        gate_verdict = evaluate_gate(
            self.run_dir,
            context=f"{self.run_id}/arm_e",
            run_id=self.run_id,
            config_hash=getattr(self.bundle, "config_hash", ""),
            notes=gate_notes,
            n_questions=len(qids),
            degenerate_cells=degenerate_cells,
            n_interaction_cells=n_interaction_cells if passed else 0,
        )

        # Write conditions and metadata sidecars BEFORE any ladder update
        self._flush_conditions_and_metadata(schedule)

        if passed and gate_verdict.proceed:
            state.pass_rung(
                "negative", note=f"Arm E passed with accuracy={accuracy}, recall={recall}"
            )
            write_ladder(self.ladder_path, state)
        else:
            if generator_failed:
                raise GateFailedError(
                    f"Arm E execution failed: generator failed to produce responses: {causal_detail}"
                )
            if not coverage_complete:
                raise GateFailedError(
                    f"Arm E output incomplete: {missing_units or unusable_finish}"
                )
            raise GateFailedError(f"Arm E negative control failed to collapse: {outcome.reason}")

        return RungReport(
            rung="negative",
            passed=passed,
            run_result=run_res,
            gate_verdict=gate_verdict,
            details=ctrl_report,
        )

    def _run_baselines_rung(self, rung: Rung, state: LadderState) -> RungReport:
        baseline_arms = tuple(
            "D-max"
            if arm == "D"
            and getattr(self.bundle, "mode", "full") == "fast"
            and "D-max" in self.registry.arms
            else arm
            for arm in rung.arms
        )
        for reader in self.panel:
            health = reader.health()
            if not health.get("ready", True):
                engine = health.get("engine", {})
                reason = engine.get("reason") or "backend is not ready"
                raise BackendUnavailableError(f"reader {reader.id!r} backend unavailable: {reason}")

        scale = rung.scales[0]
        k0 = int(self.bundle.design.K_grid[0])
        b0 = self.bundle.design.B_grid[0]
        cells = [
            Condition(arm, k0, float(b0.value), str(b0.mode), scale, reader.id)
            for arm in baseline_arms
            for reader in self.panel
        ]
        qids = sorted(
            self.inputs.qids_by_scale().get(scale, list(self.inputs.gold_sessions.keys()))
        )
        if rung.n_questions > 0:
            qids = qids[: rung.n_questions]
        if not qids:
            raise MissingInputError("no questions available for baselines rung")

        schedule = build_schedule(cells, qids, runs_per_cell=1, seed=self.seed)
        if not schedule:
            raise MissingInputError("empty schedule generated for baselines rung")

        run_res = self._execute_schedule(schedule)

        # Verify durable coverage
        coverage_complete = run_res.n_failed == 0 and len(schedule) > 0
        if coverage_complete:
            for unit in schedule:
                key = cache_key(unit.condition, self._render_request_body(unit))
                stored = self.store.get(key)
                if (
                    stored is None
                    or stored.finish_reason in EXCLUDED_FINISH_REASONS
                    or stored.finish_reason != "stop"
                ):
                    coverage_complete = False
                    break

        generator_failed = run_res.n_failed > 0 or bool(self._generator_failures)
        causal_detail: str | None = None
        if self._generator_failures:
            _, _, last_exc = self._generator_failures[-1]
            causal_detail = str(last_exc)
        elif self.config.ledger_path.is_file():
            ledger_data = json.loads(self.config.ledger_path.read_text(encoding="utf-8"))
            exclusions = ledger_data.get("exclusions", [])
            if exclusions:
                causal_detail = exclusions[-1].get("detail")
        elif self.config.wal_path.is_file():
            for entry in read_wal(self.config.wal_path):
                if entry.state is WalState.FAILED and entry.detail:
                    causal_detail = entry.detail

        passed = coverage_complete
        _, n_interaction_cells, degenerate_cells = (
            crossing(self.plan, self.bundle) if self.plan else ({}, 1, [])
        )
        if passed:
            gate_notes = ["Baselines rung completed."]
            if "D-max" in baseline_arms:
                gate_notes.append(
                    "Explicit pilot baseline D-max used; maximum feasible context is not full-context D."
                )
        elif generator_failed:
            gate_notes = [
                f"Baselines rung execution failure: generator failed to produce responses ({causal_detail})."
            ]
        else:
            gate_notes = ["Baselines rung failed durable coverage."]

        gate_verdict = evaluate_gate(
            self.run_dir,
            context=f"{self.run_id}/baselines",
            run_id=self.run_id,
            config_hash=getattr(self.bundle, "config_hash", ""),
            notes=gate_notes,
            n_questions=len(qids),
            degenerate_cells=degenerate_cells,
            n_interaction_cells=n_interaction_cells if passed else 0,
        )

        # Flush conditions and metadata sidecars BEFORE ladder write
        self._flush_conditions_and_metadata(schedule)

        if passed and gate_verdict.proceed:
            state.pass_rung("baselines", note="Baselines completed without failure")
            write_ladder(self.ladder_path, state)
        else:
            if generator_failed:
                raise GateFailedError(
                    f"Baselines rung failed: generator failed to produce responses: {causal_detail}"
                )
            raise GateFailedError("Baselines rung failed gate evaluation or coverage.")

        return RungReport(
            rung="baselines",
            passed=passed,
            run_result=run_res,
            gate_verdict=gate_verdict,
            details={"arms": list(baseline_arms)},
        )

    def _run_full_rung(self, rung: Rung, state: LadderState) -> RungReport:
        if self.plan is None:
            raise RuntimeError("ExecutablePlan is required for the full rung.")

        schedule = self.plan.schedule
        if not schedule:
            raise MissingInputError("ExecutablePlan schedule is empty for full rung.")

        run_res = self._execute_schedule(schedule)

        coverage_complete = run_res.n_failed == 0 and len(schedule) > 0
        if coverage_complete:
            for unit in schedule:
                key = cache_key(unit.condition, self._render_request_body(unit))
                stored = self.store.get(key)
                if (
                    stored is None
                    or stored.finish_reason in EXCLUDED_FINISH_REASONS
                    or stored.finish_reason != "stop"
                ):
                    coverage_complete = False
                    break

        generator_failed = run_res.n_failed > 0 or bool(self._generator_failures)
        causal_detail: str | None = None
        if self._generator_failures:
            _, _, last_exc = self._generator_failures[-1]
            causal_detail = str(last_exc)
        elif self.config.ledger_path.is_file():
            ledger_data = json.loads(self.config.ledger_path.read_text(encoding="utf-8"))
            exclusions = ledger_data.get("exclusions", [])
            if exclusions:
                causal_detail = exclusions[-1].get("detail")
        elif self.config.wal_path.is_file():
            for entry in read_wal(self.config.wal_path):
                if entry.state is WalState.FAILED and entry.detail:
                    causal_detail = entry.detail

        passed = coverage_complete
        _, n_interaction_cells, degenerate_cells = (
            crossing(self.plan, self.bundle) if self.plan else ({}, 1, [])
        )
        if passed:
            gate_notes = ["Full factorial run completed."]
        elif generator_failed:
            gate_notes = [
                f"Full rung execution failure: generator failed to produce responses ({causal_detail})."
            ]
        else:
            gate_notes = ["Full rung failed durable coverage."]

        gate_verdict = evaluate_gate(
            self.run_dir,
            context=f"{self.run_id}/full",
            run_id=self.run_id,
            config_hash=getattr(self.bundle, "config_hash", ""),
            notes=gate_notes,
            degenerate_cells=degenerate_cells,
            n_interaction_cells=n_interaction_cells if passed else 0,
        )

        # Flush conditions and metadata sidecars BEFORE ladder write
        self._flush_conditions_and_metadata(schedule)

        if passed and gate_verdict.proceed:
            state.pass_rung("crossed", note="Full grid completed")
            write_ladder(self.ladder_path, state)
        else:
            if generator_failed:
                raise GateFailedError(
                    f"Full rung failed: generator failed to produce responses: {causal_detail}"
                )
            raise GateFailedError("Full rung failed gate evaluation or coverage.")

        return RungReport(
            rung="crossed",
            passed=passed,
            run_result=run_res,
            gate_verdict=gate_verdict,
        )

    def _flush_conditions_and_metadata(self, schedule: Sequence[ScheduledUnit]) -> None:
        # Build observation metadata for all completed units in schedule
        for u in schedule:
            req_body = self._render_request_body(u)
            key = cache_key(u.condition, req_body)
            stored = self.store.get(key)
            if stored is not None:
                self._record_observation(u, stored)

        import pyarrow as pa  # type: ignore[import-untyped]
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        from msat.arms.conditions import arrow_schema, row_for

        destination = self.run_dir / "conditions.parquet"
        rows = pq.read_table(destination).to_pylist() if destination.is_file() else []
        keys = (
            "arm",
            "qid",
            "K",
            "B",
            "B_mode",
            "scale",
            "reader",
            "position_variant",
            "prompt_sha256",
        )
        indexed = {tuple(row.get(key) for key in keys): row for row in rows}
        for unit in schedule:
            row = row_for(self._render_unit(unit))
            indexed[tuple(row.get(key) for key in keys)] = row
        ordered = [
            indexed[key]
            for key in sorted(indexed, key=lambda key: tuple(str(value) for value in key))
        ]
        buffer = pa.BufferOutputStream()
        pq.write_table(
            pa.Table.from_pylist(ordered, schema=arrow_schema()), buffer, compression="zstd"
        )
        write_bytes_artifact(destination, buffer.getvalue().to_pybytes())

        if self._observations:
            write_observation_metadata(
                list(self._observations.values()), self.run_dir / "observations.parquet"
            )
