"""The `msat` command-line surface."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Final, NoReturn, TypeVar, cast

import typer

from msat import __version__
from msat.constants import (
    EXIT_ASSERTION_FAILED,
    EXIT_BUDGET_UNSET,
    EXIT_CONFIG_INVALID,
    EXIT_DEGENERATE_CELL,
    EXIT_DRIFT,
    EXIT_FAILURE,
    EXIT_NO_GO,
    EXIT_UNDERPOWERED,
)
from msat.constants import (
    OFFLINE_COMMANDS as DECLARED_OFFLINE,
)
from msat.logging import configure_logging, get_logger
from msat.runner.lifecycle import CapacityVerdict

if TYPE_CHECKING:
    from msat.metrics.normalisation import Normaliser
    from msat.runner.ladder import LadderState, RungName

app = typer.Typer(
    name="msat",
    help=(
        "MemSat experimental harness — retrieval depth K x context budget B.\n"
        "\n"
        "The primary result is the estimated log2(K)*log2(B) interaction term, "
        "tested against a pre-registered sign prediction. See PRD.md."
    ),
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode=None,
)

PRD_SURFACE: Final[tuple[str, ...]] = (
    "validate-config",
    "estimate-power",
    "ceiling",
    "build-manifest",
    "run",
    "status",
    "score",
    "judge-audit",
    "analyze",
    "figures",
    "paper",
    "data-lock",
    "report-reader-capacity",
    "hardware-delta",
    "vram",
)

log = get_logger("cli")

T = TypeVar("T")


# ── SEAM: Run Identity ────────────────────────────────────────────────────────


class RunIdentitySeam:
    """Single seam for resolving run identities across CLI commands."""

    @staticmethod
    def resolve(
        positional: str | None,
        alias: str | None,
        *,
        name: str = "run_id",
        required: bool = True,
    ) -> str:
        if positional and alias:
            _fail(f"{name} given twice: {positional!r} positionally and {alias!r} via --run")
        resolved = positional or alias
        if not resolved and required:
            _fail(f"a {name} is required (positional, or via --run)")
        return resolved or ""

    @staticmethod
    def resolve_comparison(against: str | None) -> str | None:
        if against is None:
            return None
        cleaned = str(against).strip()
        return cleaned or None


_resolve_run_id = RunIdentitySeam.resolve


# ── SEAM: Output Location ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class OutputLocation:
    """Output location seam distinguishing directory from file outputs."""

    path: Path
    is_directory: bool

    @classmethod
    def directory(
        cls, path: Path | str | None, *, default: Path | str = Path("work")
    ) -> OutputLocation:
        p = Path(path if path is not None else default)
        return cls(path=p, is_directory=True)

    @classmethod
    def file(
        cls,
        path: Path | str | None,
        *,
        default: Path | str | None = None,
        default_dir: Path | str | None = None,
        filename: str | None = None,
    ) -> OutputLocation:
        if path is not None:
            p = Path(path)
            if p.is_dir() and filename:
                p = p / filename
            return cls(path=p, is_directory=False)
        if default is not None:
            return cls(path=Path(default), is_directory=False)
        if default_dir is not None and filename is not None:
            return cls(path=Path(default_dir) / filename, is_directory=False)
        raise ValueError("Must provide path, default, or (default_dir and filename)")

    def ensure_parent(self) -> Path:
        parent = self.path if self.is_directory else self.path.parent
        parent.mkdir(parents=True, exist_ok=True)
        return self.path

    def file_in(self, filename: str) -> Path:
        if not self.is_directory:
            raise ValueError(f"Expected directory output, got file: {self.path}")
        return self.path / filename


# ── SEAM: Input Discovery ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class ScoreInputs:
    gold: Path
    official: Path | None = None
    verdicts: Path | None = None
    metadata: Path | None = None
    config: Path = Path("configs")


@dataclass(frozen=True)
class JudgeAuditInputs:
    verdicts: Path
    human: Path | None = None


class InputSeam:
    """Unified seam for CLI input resolution and validation."""

    @staticmethod
    def resolve_run_inputs(
        *,
        questions_path: Path | None = None,
        gold_path: Path | None = None,
        pool_path: Path | None = None,
        filler_path: Path | None = None,
        data_dir: Path | None = None,
        scales: Sequence[str] | None = None,
    ) -> Any:
        from msat.runner.input import RunInputs

        return RunInputs.from_legacy_paths(
            questions_path=questions_path,
            gold_path=gold_path,
            pool_path=pool_path,
            filler_path=filler_path,
            data_dir=data_dir,
            scales=scales,
        )

    @staticmethod
    def resolve_score_inputs(
        *,
        gold: Path | None,
        official: Path | None = None,
        verdicts: Path | None = None,
        metadata: Path | None = None,
        config: Path = Path("configs"),
    ) -> ScoreInputs:
        if gold is None:
            _fail("--gold is required. Scoring needs gold answers for attributable observations.")
        return ScoreInputs(
            gold=gold,
            official=official,
            verdicts=verdicts,
            metadata=metadata,
            config=config,
        )

    @staticmethod
    def resolve_judge_audit_inputs(
        *,
        verdicts_path: Path | None,
        human: Path | None = None,
    ) -> JudgeAuditInputs:
        if verdicts_path is None or not Path(verdicts_path).exists():
            _fail(
                "--verdicts is required: the audit is built from verdict records alone, "
                "which is what makes it standalone."
            )
        return JudgeAuditInputs(verdicts=verdicts_path, human=human)


# ── SEAM: Structured Command Results & Renderer ───────────────────────────────


@dataclass(frozen=True)
class ValidateConfigResult:
    configs_dir: Path
    problems: list[Any]
    errors: list[Any]
    warnings: list[Any]
    config_hash: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PowerResult:
    report: dict[str, Any]
    out_dir: Path
    powered: bool
    budget_state: str


@dataclass(frozen=True)
class CeilingResult:
    out_dir: Path
    verified: bool
    drifted: bool
    scales: dict[str, Any]
    ceiling_artifact: dict[str, Any]
    fraction_artifact: dict[str, Any]
    report_artifact: dict[str, Any]


@dataclass(frozen=True)
class DataLockResult:
    lock_path: Path
    verified: bool
    drifted: bool
    lock: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlanResult:
    manifest_path: Path
    plan_path: Path
    crossed_path: Path
    config_hash: str
    conditions: int
    supported: int
    unsupported: int
    degenerate_conditions: int
    observations: int
    crossed: bool
    reader_balanced: bool

    @property
    def n_cells(self) -> int:
        return self.conditions

    @property
    def n_accepted(self) -> int:
        return self.supported

    @property
    def n_rejected(self) -> int:
        return self.unsupported

    @property
    def n_degenerate(self) -> int:
        return self.degenerate_conditions

    @property
    def n_units(self) -> int:
        return self.observations


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_dir: Path
    rung: str
    profile: str
    dry_run: bool
    data: dict[str, Any] | Any


@dataclass(frozen=True)
class StatusResult:
    run_id: str
    run_dir: Path
    n_stored_responses: int
    n_done: int
    n_planned: int
    n_failed: int
    n_excluded: int
    n_cells: int
    cells: list[dict[str, Any]]
    ladder_state: Any | None = None
    next_rung: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    run_id: str
    scores_path: Path
    latency_path: Path
    report: Any
    latency_data: dict[str, Any]
    disagreements_path: Path | None
    notes: list[str]


@dataclass(frozen=True)
class JudgeAuditResult:
    run_id: str
    out_path: Path
    audit: Any
    panel_usable: bool


@dataclass(frozen=True)
class AnalyzeResult:
    run_id: str
    run_dir: Path
    n_observations: int
    status: str
    analysis_json_path: Path
    results_md_path: Path
    notes: list[str]


@dataclass(frozen=True)
class FiguresResult:
    out_dir: Path
    report: dict[str, Any]


@dataclass(frozen=True)
class PaperResult:
    out_dir: Path
    pdf_path: Path | None
    built: bool = True


@dataclass(frozen=True)
class MaxContextResult:
    by_scale: dict[str, dict[str, Any]]
    out_path: Path | None


@dataclass(frozen=True)
class HardwareDeltaResult:
    run_id: str
    against: str | None
    is_probe: bool
    plan: dict[str, Any] | None = None
    verdict: dict[str, Any] | None = None


@dataclass(frozen=True)
class VramResult:
    shape: Any
    plan: Any
    tokens: int
    card_reader_capacity: int | None = None

    @property
    def card_ceiling(self) -> int | None:
        return self.card_reader_capacity


class CliRenderer:
    """One rendering adapter for structured command output."""

    def __init__(self) -> None:
        self.last_result: Any = None
        self._handlers: dict[type, Any] = {
            ValidateConfigResult: self._render_validate_config,
            PowerResult: self._render_power,
            CeilingResult: self._render_ceiling,
            DataLockResult: self._render_data_lock,
            PlanResult: self._render_plan,
            RunResult: self._render_run,
            StatusResult: self._render_status,
            ScoreResult: self._render_score,
            JudgeAuditResult: self._render_judge_audit,
            AnalyzeResult: self._render_analyze,
            FiguresResult: self._render_figures,
            PaperResult: self._render_paper,
            MaxContextResult: self._render_max_context,
            HardwareDeltaResult: self._render_hardware_delta,
            VramResult: self._render_vram,
        }

    def render(self, result: T) -> T:
        self.last_result = result
        handler = self._handlers.get(type(result))
        if handler:
            handler(result)
        return result

    def _render_validate_config(self, res: ValidateConfigResult) -> None:
        for problem in res.problems:
            colour = typer.colors.RED if problem.severity.value == "error" else typer.colors.YELLOW
            typer.secho(str(problem), fg=colour, err=True)
        if res.config_hash:
            typer.echo(f"config_hash  {res.config_hash}")
            for key, value in res.summary.items():
                typer.echo(f"  {key:<12} {value}")
        if not res.errors:
            typer.secho(
                f"config OK ({len(res.warnings)} warning(s))" if res.warnings else "config OK",
                fg=typer.colors.GREEN,
                err=True,
            )

    def _render_power(self, res: PowerResult) -> None:
        _emit_power_report(res.report, res.out_dir)

    def _render_ceiling(self, res: CeilingResult) -> None:
        if res.verified:
            typer.secho(
                "ceiling: all committed artifacts reproduce byte-for-byte",
                fg=typer.colors.GREEN,
                err=True,
            )
        else:
            for p in (
                res.out_dir / "ceiling.json",
                res.out_dir / "corpus_fraction.json",
                res.out_dir / "dataset_report.json",
            ):
                if p.exists():
                    typer.secho(f"wrote {p} ({p.stat().st_size} bytes)", fg=typer.colors.GREEN)

    def _render_data_lock(self, res: DataLockResult) -> None:
        if res.verified:
            typer.secho(f"data.lock verified: {res.lock_path}", fg=typer.colors.GREEN, err=True)
        else:
            for split, entry in sorted(res.lock.get("files", {}).items()):
                if entry.get("present"):
                    typer.secho(
                        f"  locked  {split:<8} {entry['filename']:<22} "
                        f"{entry['bytes'] / 1e6:9.1f} MB  rows={entry.get('rows')}  "
                        f"sha256={entry['sha256'][:16]}…"
                    )
                else:
                    typer.secho(f"  MISSING {split:<8} {entry['filename']}", fg=typer.colors.YELLOW)
            typer.secho(f"wrote {res.lock_path}", fg=typer.colors.GREEN)

    def _render_plan(self, res: PlanResult) -> None:
        typer.echo(f"  config_hash   {res.config_hash}")
        typer.echo(f"  conditions    {res.conditions}")
        typer.echo(
            f"  supported     {res.supported}   unsupported {res.unsupported}"
            f"   (degeneracy: {res.degenerate_conditions})"
        )
        typer.echo(f"\n  wrote {res.manifest_path}")
        typer.echo(f"  plan          {res.observations:,} observation(s) -> {res.plan_path.name}")
        typer.echo(
            f"  crossed       {'YES' if res.crossed else 'NO'}  "
            f"({res.supported}/{res.conditions} conditions, "
            f"reader_balanced={res.reader_balanced})   R4.4"
        )
        typer.echo(f"  wrote {res.plan_path}")
        typer.echo(f"  wrote {res.crossed_path}")

    def _render_run(self, res: RunResult) -> None:
        from msat.runner.lifecycle import RungReport

        if isinstance(res.data, dict):
            registry = res.data.get("registry", {})
            arms = sorted(
                set(registry.get("prompt_arms", [])) | set(registry.get("config_arms", []))
            )
            typer.echo(f"  arms        {', '.join(arms)}")
            typer.echo(
                f"  config-only {registry.get('config_arms', [])}: manipulated in the memory set"
            )
            capacity = res.data.get("capacity", {})
            n_supp = capacity.get("n_supported", capacity.get("n_accepted", 0))
            n_unsupp = capacity.get("n_unsupported", capacity.get("n_rejected", 0))
            typer.echo(
                f"  capacity    R3.2: {n_supp}/{capacity.get('n_checked', 0)} fit; {n_unsupp} UNSUPPORTED"
            )
            typer.secho(
                "  DRY RUN: registry, ladder, crossed grid and capacity checked. Nothing was generated.",
                fg=typer.colors.GREEN,
            )
        elif isinstance(res.data, RungReport):
            if res.rung == "negative":
                typer.secho(
                    "  ARM E COLLAPSED. The negative control passed.", fg=typer.colors.GREEN
                )
            typer.echo(
                f"  generated   {res.data.run_result.n_generated}, warm hits {res.data.run_result.n_warm_hits}"
            )
            typer.echo(f"  wrote {res.run_dir / 'ladder.json'}")
        typer.echo(f"  wrote {res.run_dir / 'go_no_go.json'}")

    def _render_status(self, res: StatusResult) -> None:
        typer.echo(f"\n  run       {res.run_id}")
        typer.echo(f"  dir       {res.run_dir}")
        typer.echo(f"  stored    {res.n_stored_responses:,} responses   <- the record")
        typer.echo(
            f"  ledger    {res.n_done:,} / {res.n_planned:,} done   across {res.n_cells:,} cells"
        )
        if res.n_failed:
            typer.secho(f"  failed    {res.n_failed:,}", fg=typer.colors.RED)
        if res.n_excluded:
            typer.secho(
                f"  excluded  {res.n_excluded:,} question(s) - degenerate at this "
                f"K, dropped from paired comparisons",
                fg=typer.colors.YELLOW,
            )
        if res.cells:
            typer.echo("\n  cell                                     planned  done  failed")
            for c in sorted(res.cells, key=lambda x: tuple(str(p) for p in x["key"]))[:20]:
                arm, k, b, _b_mode, scale, reader = c["key"]
                label = f"    {arm} {scale} {reader} K={k} B={b}"
                typer.echo(
                    label.ljust(42)
                    + f"{c.get('n_planned', 0):>7} {c.get('n_done', 0):>5} {c.get('n_failed', 0):>7}"
                )
            if len(res.cells) > 20:
                typer.echo(f"    ... and {len(res.cells) - 20} more cells")
        if res.ladder_state is not None:
            from msat.runner.ladder import LADDER

            typer.echo("\n  ladder")
            for rung in LADDER:
                passed = rung.name in res.ladder_state.passed
                mark = typer.style("pass", fg=typer.colors.GREEN) if passed else "todo"
                typer.echo(f"    {mark:<22} {rung.arms}  ({rung.description})")
            if res.next_rung is not None:
                typer.secho(
                    f"    next: {res.next_rung}. The full grid is blocked until "
                    f"it passes; that is the ladder working, not a bug.",
                    fg=typer.colors.YELLOW,
                )
            typer.echo()

    def _render_score(self, res: ScoreResult) -> None:
        report = res.report
        typer.echo(f"\n  scored      {report.n_attributable:,} item(s) -> {report.n_rows:,} rows")
        typer.echo(f"  metrics     {', '.join(report.metrics_written)}")
        typer.echo(f"  abstention  {report.n_abstention:,} scored NULL, not 0.0 (R8.4)")
        if report.n_unattributable:
            typer.secho(
                f"  UNATTRIBUTED {report.n_unattributable} response(s) carry no qid and were counted, not scored.",
                fg=typer.colors.YELLOW,
            )
        for metric_name, summary in sorted(report.means.items()):
            mean = "n/a" if summary["mean"] is None else f"{summary['mean']:.4f}"
            typer.echo(
                f"  {metric_name:<10}{mean:>8}   over {summary['n_scored']:,} scored "
                f"({summary['n_abstention_excluded']:,} abstention excluded)"
            )
        for note in res.notes:
            typer.secho(f"  note: {note}", fg=typer.colors.YELLOW)
        typer.echo(f"\n  wrote {res.scores_path}")
        _print_latency(res.latency_data)
        typer.echo(f"  wrote {res.latency_path}")
        if res.disagreements_path is not None:
            typer.echo(f"  disagreements row(s) -> {res.disagreements_path.name}")

    def _render_judge_audit(self, res: JudgeAuditResult) -> None:
        from msat.judges.report import format_audit

        typer.echo(format_audit(res.audit))
        typer.echo(f"\n  wrote {res.out_path}")

    def _render_analyze(self, res: AnalyzeResult) -> None:
        typer.echo(f"  analyzed    {res.n_observations:,} observation(s), status {res.status}")
        typer.echo(f"  wrote {res.analysis_json_path}")
        typer.echo(f"  wrote {res.results_md_path}")
        for note in res.notes:
            typer.secho(f"  note: {note}", fg=typer.colors.YELLOW)

    def _render_figures(self, res: FiguresResult) -> None:
        typer.echo(json.dumps(res.report, indent=2, default=str))
        if res.report.get("is_preview"):
            typer.secho(
                f"warning: {res.report.get('warning', '')} — do not cite these files.",
                fg=typer.colors.YELLOW,
                err=True,
            )

    def _render_paper(self, res: PaperResult) -> None:
        typer.echo(f"  paper       built in {res.out_dir}")

    def _render_max_context(self, res: MaxContextResult) -> None:
        for scale_id, block in res.by_scale.items():
            typer.echo(
                f"\n  scale {scale_id}: {block['corpus_tokens']:,} tokens, "
                f"{block['sessions_median']:.0f} sessions "
                f"({block['mean_session_tokens']:,.0f} tok/session)"
            )
            typer.echo(
                f"    {'reader':<8}{'ceiling':>10}{'sessions':>10}{'kept tok':>11}"
                f"{'corpus':>9}   full_context"
            )
            for reader, r in block["per_reader"].items():
                full = "yes" if r["full_context"] else f"NO, short {r['shortfall']:,}"
                typer.echo(
                    f"    {reader:<8}{r['ceiling']:>10,}{r['n_sessions_kept']:>10,}"
                    f"{r['kept_tokens']:>11,}{r['corpus_fraction']:>8.1%}   {full}"
                )
            if block["arm_d_infeasible"]:
                typer.secho(
                    "    Arm D (true full context) is INFEASIBLE on this hardware. "
                    "`full_context_max` is the substitute.",
                    fg=typer.colors.YELLOW,
                )
            typer.secho(
                f"    {block['note']}",
                fg=typer.colors.YELLOW,
            )
        if res.out_path:
            typer.echo(f"\n  wrote {res.out_path}")

    def _render_hardware_delta(self, res: HardwareDeltaResult) -> None:
        if res.is_probe and res.plan:
            plan = res.plan
            typer.echo("\n  probe plan")
            typer.echo(f"    cells          {plan['n_keys']:,} ({plan['n_resolved']:,} resolved)")
            typer.echo(f"    prompt tokens  {plan['prompt_tokens']:,}")
            typer.echo(f"    seed           {plan['seed']}")
            typer.secho(f"    {plan['note']}", fg=typer.colors.YELLOW)
            typer.echo("\n  re-run those exact cache keys on the second box, then:")
            typer.echo(f"    msat hardware-delta {res.run_id} --against <run_id> --compare")
        elif res.verdict:
            typer.echo(json.dumps(res.verdict, indent=2, sort_keys=True))

    def _render_vram(self, res: VramResult) -> None:
        shape = res.shape
        plan = res.plan
        tokens = res.tokens
        typer.echo(f"\n  model        {shape.name}")
        typer.echo(
            f"  layers       {shape.n_layers}   kv_heads {shape.n_kv_heads}"
            f"   head_dim {shape.head_dim}   GQA factor {shape.gqa_factor}x"
        )
        typer.echo(f"  dtype        {shape.dtype}")
        if shape.is_moe:
            typer.echo(
                f"  MoE          {shape.n_experts} experts"
                f"{f', {shape.n_active_experts} active' if shape.n_active_experts else ''}"
                f"  - KV depends on attention layers only, NOT on the {shape.n_params / 1e9:.0f}B parameters"
            )
        typer.echo(f"  KV per token {shape.kv_bytes_per_token():,} bytes")
        typer.echo(
            f"  params       {shape.n_params / 1e9:.1f}B"
            + ("" if shape.n_params_is_exact else "   (ESTIMATED from config - not an exact count)")
        )
        typer.echo("")
        typer.echo(
            f"  at {tokens:,} tokens:  weights {plan.weights_gb:.1f} GB"
            f" + KV {plan.kv_gb:.1f} GB + headroom {plan.headroom_gb:.1f} GB"
            f" = {plan.total_gb:.1f} GB"
        )
        typer.echo(f"  smallest standard card: {plan.smallest_card()}")
        typer.secho(f"  omitted: {'; '.join(plan.omitted)}", fg=typer.colors.YELLOW)
        if res.card_reader_capacity:
            typer.echo(
                f"\n  ceiling on a card: {res.card_reader_capacity:,} prompt tokens"
                f"  (vs the hand-declared reader capacity in readers.yaml)"
            )
        typer.echo()


renderer = CliRenderer()


def get_last_result() -> Any:
    """Return the last structured result rendered by the CLI."""
    return renderer.last_result


# ── SEAM: Offline Policy ──────────────────────────────────────────────────────


def registered_commands() -> tuple[str, ...]:
    names = {info.name for info in app.registered_commands if info.name}
    names |= {info.name for info in app.registered_groups if info.name}
    return tuple(sorted(names))


ONLINE_COMMANDS: Final[frozenset[str]] = frozenset({"run"})


def offline_commands() -> frozenset[str]:
    """Derive offline commands from registered commands."""
    return frozenset(c for c in registered_commands() if c not in ONLINE_COMMANDS)


OFFLINE_COMMANDS: Final[frozenset[str]] = DECLARED_OFFLINE


def _assert_offline(command: str) -> None:
    if command not in offline_commands():
        _fail(f"command {command!r} is not registered as an offline command")
    log.debug("offline command", extra={"command": command, "offline": True})


# ── Internal Helpers ──────────────────────────────────────────────────────────


def _fail(message: str, code: int = EXIT_FAILURE) -> NoReturn:
    typer.secho(f"error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


def _read_ladder(path: Path) -> LadderState:
    """A missing ladder file means a run that has not started, not an error."""
    from msat.runner.ladder import LadderState

    if not path.exists():
        return LadderState()

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return LadderState(passed=set(raw.get("passed", [])), notes=dict(raw.get("notes", {})))
    except (ValueError, OSError) as e:
        _fail(f"{path} is unreadable ({e}); refusing to report a ladder state it cannot confirm")
        raise AssertionError("unreachable") from e  # pragma: no cover


def normalise_scale_name(split: str) -> str:
    """Accepts `M`, `longmemeval_m` and `lme_m_cleaned.json` as scale names."""
    token = split.strip().lower()
    if token in ("s", "m"):
        return token.upper()
    for scale in ("s", "m"):
        if f"longmemeval_{scale}" in token or f"lme_{scale}" in token:
            return scale.upper()
    return split


def _resolve_profile_bundle(bundle: Any, profile: str | None) -> tuple[Any, str]:
    """Apply execution profile overrides to ConfigBundle.

    - 'fast': Scale S only, primary reader (gemma), 1 repetition/condition (rapid pilot tier ~1.5h).
    - 'full': Canonical pre-registered grid across all scales and readers.
    """
    from msat.experiment import InvalidProfileError, resolve_effective_experiment

    try:
        experiment = resolve_effective_experiment(bundle, profile)
    except InvalidProfileError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return experiment.bundle, experiment.mode


resolve_profile_bundle = _resolve_profile_bundle


# ── Commands ──────────────────────────────────────────────────────────────────


@app.command("validate-config")
def validate_config(
    configs_dir: Annotated[
        Path,
        typer.Argument(help="Directory (or file) holding the config bundle."),
    ] = Path("configs"),
    show_hash: Annotated[
        bool, typer.Option("--hash", help="Print the resolved config_hash (PRD §R1.4).")
    ] = False,
) -> ValidateConfigResult:
    """Fail non-zero if a required config field is missing or `readers` < 2."""
    from msat.config import errors_only, validate

    root = configs_dir if configs_dir.is_dir() else configs_dir.parent
    problems = validate(root)
    errs = errors_only(problems)
    warns = [p for p in problems if p not in errs]

    config_hash: str | None = None
    summary: dict[str, Any] = {}
    if show_hash and not errs:
        from msat.config import load_bundle

        bundle = load_bundle(root)
        config_hash = bundle.config_hash
        summary = bundle.summary()

    res = ValidateConfigResult(
        configs_dir=root,
        problems=list(problems),
        errors=list(errs),
        warnings=list(warns),
        config_hash=config_hash,
        summary=summary,
    )
    renderer.render(res)

    if errs:
        _fail(
            f"{len(errs)} blocking problem(s) in {root}. "
            "A retriever specified after the sweep invalidates every figure.",
            code=EXIT_CONFIG_INVALID,
        )
    return res


@app.command("estimate-power")
def estimate_power(
    design_yaml: Annotated[
        Path,
        typer.Argument(help="Path to configs/design.yaml — the single grid source."),
    ] = Path("configs/design.yaml"),
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="Execution profile: 'fast' (Scale S, primary reader, 1 run/cell) or 'full'.",
        ),
    ] = "full",
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("work"),
    sims: Annotated[int, typer.Option("--sims", help="Simulations per power point.")] = 30,
    seed: Annotated[int, typer.Option("--seed")] = 20261012,
    skip_sensitivity: Annotated[
        bool, typer.Option("--fast", help="Skip the judge-noise sensitivity sweep.")
    ] = False,
) -> PowerResult:
    """Estimate statistical power for the K x B interaction before any API call."""
    from msat.config import load_bundle
    from msat.power import run_power_analysis

    root = design_yaml.parent
    bundle = load_bundle(root)
    bundle, profile_resolved = _resolve_profile_bundle(bundle, profile)
    if profile_resolved == "fast":
        skip_sensitivity = True
        typer.secho(
            "  [FAST PROFILE] Power analysis for fast pilot grid (Scale S, primary reader, 1 repetition/condition).",
            fg=typer.colors.CYAN,
            bold=True,
        )
    loc = OutputLocation.directory(out, default="work")
    report = run_power_analysis(
        bundle,
        out_dir=loc.path,
        n_sim=sims,
        seed=seed,
        sensitivity=not skip_sensitivity,
    )
    budget = report["budget_guard"]
    powered = bool(report["verdict"]["powered"])
    res = PowerResult(
        report=report,
        out_dir=loc.path,
        powered=powered,
        budget_state=str(budget["state"]),
    )
    renderer.render(res)

    if not powered:
        _fail(
            "the K x B interaction is not estimable at this grid; "
            "`reduced_design_proposals` contains designs that preserve estimability. "
            "Nothing was truncated automatically (PRD §R0.3).",
            code=EXIT_UNDERPOWERED,
        )
    if budget["state"] != "set":
        _fail(
            f"estimated cost ${budget['estimate_usd']:,.0f} but budget.max_usd is "
            "unset (blocker B7). The hard cost guard cannot be enforced and no run "
            "may be launched. This estimate is the input to the budget conversation.",
            code=EXIT_BUDGET_UNSET,
        )
    return res


design_power = estimate_power


def _emit_power_report(report: dict[str, Any], out_dir: Path) -> None:
    verdict = report["verdict"]
    grid = report["grid"]
    inter = report["interaction"]

    typer.secho("\n=== estimate-power · the K x B interaction ===", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"  grid       {grid['K_grid']} x {len(grid['B_grid'])} B "
        f"x {len(grid['scales'])} scales x {len(grid['readers'])} readers  (nominal)"
    )
    dropped = grid.get("cells_dropped_for_vram") or 0
    if dropped:
        eff_k = grid.get("grid_effective_K") or []
        eff_b = grid.get("grid_effective_B") or []
        typer.echo(
            f"  effective  {eff_k} x {len(eff_b)} B "
            f"x {len(grid['scales'])} scales x {len(grid['readers'])} readers"
        )
        typer.secho(
            f"  cells      {grid['n_cells_effective']} of {grid['n_cells_nominal']}"
            f"   observations  {grid['n_observations']:,}"
            f"   ({dropped} dropped for VRAM)",
            fg=typer.colors.YELLOW,
        )
    else:
        typer.echo(f"  cells      {grid['n_cells']}   observations  {grid['n_observations']:,}")
    typer.echo(f"  config     {report['_provenance']['config_hash'][:16]}…")
    typer.echo(
        f"  prereg     {report['_provenance']['preregistered_sign']}"
        f"   hash {report['_provenance']['preregistration_hash'][:16]}…"
    )

    if inter.get("powered"):
        typer.secho(
            f"\n  MDE (interaction)   {inter['mde_log_odds']} log-odds",
            fg=typer.colors.GREEN,
        )
        typer.echo(f"    at accuracy p=0.5  {inter['mde_pp_at_p50']} pp")
        typer.echo(f"    at accuracy p=0.3  {inter['mde_pp_at_p30']} pp")
        typer.echo(f"    SE                 {inter['se_log_odds']}")
    else:
        typer.secho("\n  UNDERPOWERED", fg=typer.colors.RED, bold=True)
        typer.echo(f"    {inter.get('note', '')}")
        typer.echo("    proposals that preserve estimability:")
        for prop in report.get("reduced_design_proposals", {}).get("proposals", []):
            flag = "  [violates R7.5]" if prop.get("violates_r75") else ""
            typer.echo(f"      - {prop['step']}{flag}")

    rows = report.get("judge_noise_sensitivity") or []
    if rows:
        typer.secho(
            "\n  judge-noise sensitivity — MDE for the interaction by grader quality:",
            fg=typer.colors.CYAN,
        )
        typer.echo(f"    {'flip':<7}{'atten':>8}{'MDE':>10}{'pp @ p=.5':>11}  interpretation")
        for r in rows:
            mde = r.get("mde_log_odds")
            mde_str = f"{mde:.4f}" if mde is not None else "n/a"
            pp_str = f"{r['mde_pp_at_p50']:.3f}" if mde is not None else "n/a"
            typer.echo(
                f"    {r['judge_flip']:<7.2f}{r['effect_attenuation']:>8.2f}"
                f"{mde_str:>10}{pp_str:>11}  {r['interpretation']}"
            )

    cost = report["cost"]
    compute = report.get("compute_budget", {})
    typer.secho("\n  cost: $0 — every reader and judge is LOCAL", fg=typer.colors.CYAN)
    typer.echo(f"    cells              {cost['n_cells']} ({cost['n_cells_priced']} priced)")
    typer.echo(f"    calls              {cost['n_calls']:,}")
    typer.echo(
        f"    cost_uncached      ${cost['cost_uncached_total_usd']:,.2f}   <- PRIMARY (R9.8)"
    )
    if cost["unpriced_readers"]:
        typer.secho(f"    UNPRICED readers   {cost['unpriced_readers']}", fg=typer.colors.RED)

    if compute.get("available"):
        typer.secho(
            "\n  COMPUTE budget — the real constraint with no paid API "
            "(throughput ASSUMED; benchmark before P5):",
            fg=typer.colors.CYAN,
        )
        for reader, row in sorted(compute["per_reader"].items()):
            typer.echo(
                f"    reader {reader:<10} {row['hours']:>9,.1f} h   ({row['calls']:,} calls)"
            )
        typer.echo(
            f"    judge panel        {compute['judge_hours']:>9,.1f} h"
            f"   ({100 * compute['judge_share_of_total']:.0f}% of total)"
        )
        typer.echo(f"    judge audit (R8.6) {compute['judge_audit_hours']:>8,.1f} h")
        typer.echo(
            f"    {'TOTAL':<20} {compute['total_hours']:>9,.1f} h"
            f"  = {compute['total_days']:.1f} days on ONE GPU"
        )
        colour = typer.colors.GREEN if compute["within_budget"] else typer.colors.RED
        state = "within budget" if compute["within_budget"] else "OVER BUDGET"
        typer.secho(
            f"    ceiling            {compute['max_wall_clock_hours']:,.0f} h"
            f"   -> {state} ({compute['multiple_of_budget']}x)",
            fg=colour,
        )

    ctx = report.get("context_feasibility", {})
    if ctx.get("n_infeasible"):
        typer.secho(
            f"\n  CONTEXT OVERFLOW — {ctx['n_infeasible']} cell(s) exceed their reader's "
            f"window. Policy: {ctx['policy']}",
            fg=typer.colors.RED,
        )
        for reader, rows in sorted(ctx["infeasible_cells"].items()):
            scales = sorted({r["scale"] for r in rows})
            worst = max(rows, key=lambda r: r["overflow_by"])
            typer.echo(
                f"    {reader:<10} ceiling {worst['ceiling']:>7,} "
                f"(blocked_by={worst['blocked_by']}, declared window "
                f"{worst['context_window']:,})"
            )
            typer.echo(
                f"    {'':<10} worst prompt {worst['prompt_tokens']:>7,} "
                f"(+{worst['overflow_by']:,})  scales={scales}"
            )
        typer.secho(
            "    These cells must be DROPPED or the grid trimmed — never truncated.",
            fg=typer.colors.RED,
        )

    budget = report["budget_guard"]
    if budget["state"] != "set":
        typer.secho(
            "    budget.max_usd     UNSET (B7) — guard NOT enforceable",
            fg=typer.colors.YELLOW,
        )
    else:
        colour = typer.colors.GREEN if budget["within_budget"] else typer.colors.RED
        typer.secho(
            f"    budget.max_usd     ${budget['max_usd']:,.2f} "
            f"({budget['multiple_of_budget']}x the estimate)",
            fg=colour,
        )

    typer.secho(
        f"\n  {verdict['headline']}",
        fg=typer.colors.YELLOW if not verdict["powered"] else typer.colors.GREEN,
    )
    typer.secho(f"\n  {verdict['binding_constraint']}\n", fg=typer.colors.BRIGHT_BLACK)
    typer.secho(f"wrote {out_dir / 'power.json'}", fg=typer.colors.GREEN)


@app.command("ceiling")
def ceiling(
    split: Annotated[
        str | None,
        typer.Argument(
            help="Optional split to PRINT (e.g. `longmemeval_m` or `M`). "
            "The artifact is always computed over every available scale."
        ),
    ] = None,
    by_type: Annotated[
        bool, typer.Option("--by-type", help="Print the per-`question_type` table.")
    ] = False,
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Regenerate and byte-compare vs committed (PRD §R2.4)."),
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out", help="Output directory for generated artifacts.")
    ] = Path("work"),
) -> CeilingResult:
    """Compute ceiling(K), partial(K), corpus feasibility, and K*."""
    from msat.ceiling import (
        build_ceiling_multiscale,
        build_corpus_fraction_artifact,
        build_dataset_report,
        dump_canonical,
        verify_artifact,
    )
    from msat.datasets import iter_metadata, raw_path, resolve_data_dir

    loc = OutputLocation.directory(out_dir, default="work")
    data_dir = resolve_data_dir()
    scales = {s: list(iter_metadata(raw_path(s, data_dir))) for s in ("S", "M")}

    ceiling_artifact = build_ceiling_multiscale(scales)
    fraction_artifact = build_corpus_fraction_artifact(scales)
    report_artifact = build_dataset_report(scales)

    ceiling_path = loc.path / "ceiling.json"
    fraction_path = loc.path / "corpus_fraction.json"
    report_path = loc.path / "dataset_report.json"

    if verify:
        _assert_offline("ceiling")
        if not ceiling_path.exists() and (Path("results") / "ceiling.json").exists():
            ceiling_path = Path("results") / "ceiling.json"
            fraction_path = Path("results") / "corpus_fraction.json"
            report_path = Path("results") / "dataset_report.json"
        problems: list[str] = []
        for path, artifact in (
            (ceiling_path, ceiling_artifact),
            (fraction_path, fraction_artifact),
            (report_path, report_artifact),
        ):
            ok, detail = verify_artifact(artifact, path)
            log.info("ceiling verify", extra={"artifact": path.name, "ok": ok})
            typer.secho(
                f"  {'OK  ' if ok else 'DRIFT'}  {path.name}: {detail}",
                fg=typer.colors.GREEN if ok else typer.colors.RED,
                err=True,
            )
            if not ok:
                problems.append(detail)
        if problems:
            _fail(
                f"{len(problems)} committed artifact(s) drifted from a regeneration "
                f"(PRD §R2.4). Re-run `msat ceiling --out {loc.path}` if the change is intended.",
                code=EXIT_DRIFT,
            )
        res = CeilingResult(
            out_dir=loc.path,
            verified=True,
            drifted=False,
            scales=scales,
            ceiling_artifact=ceiling_artifact,
            fraction_artifact=fraction_artifact,
            report_artifact=report_artifact,
        )
        return renderer.render(res)

    dump_canonical(ceiling_artifact, ceiling_path)
    dump_canonical(fraction_artifact, fraction_path)
    dump_canonical(report_artifact, report_path)
    log.info(
        "ceiling computed",
        extra={"scales": sorted(scales), "out": str(loc.path)},
    )

    res = CeilingResult(
        out_dir=loc.path,
        verified=False,
        drifted=False,
        scales=scales,
        ceiling_artifact=ceiling_artifact,
        fraction_artifact=fraction_artifact,
        report_artifact=report_artifact,
    )
    renderer.render(res)

    shown = [normalise_scale_name(split)] if split else sorted(scales)
    for scale in shown:
        if scale not in scales:
            _fail(f"unknown split {split!r}; available: {sorted(scales)}")
        _print_ceiling_table(scale, ceiling_artifact["scales"][scale])
        if by_type:
            _print_by_type(scale, ceiling_artifact["scales"][scale])
        _print_corpus_fraction(scale, fraction_artifact["scales"][scale])
    return res


def _print_ceiling_table(scale: str, block: dict[str, Any]) -> None:
    ceiling_data = block["ceiling"]
    partial = block["partial"]
    survivors = block["survivors_at"]
    degenerate = block["degeneracy_fraction"]
    corpus = block["corpus_size"]
    typer.secho(
        f"\n[{scale}]  {block['n_answerable']} answerable, "
        f"{block['n_abstention']} abstention  ·  "
        f"{corpus['n_sessions_mean']:.1f} sessions/question "
        f"(min {corpus['n_sessions_min']}, max {corpus['n_sessions_max']})",
        fg=typer.colors.CYAN,
    )
    header = f"{'K':>4} {'ceiling':>9} {'partial':>9} {'surv>=K':>9} {'degenerate':>11}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for key in sorted(ceiling_data, key=int):
        k = int(key)
        typer.echo(
            f"{k:>4} {100 * ceiling_data[key]:8.2f}% {100 * partial[key]:8.2f}% "
            f"{100 * survivors[key]:8.2f}% {100 * degenerate[key]:10.2f}%"
        )
    kstar = block["analytic_saturation_k"]
    typer.secho(
        f"\n  K* = min{{K : ceiling(K) >= 0.995}} = {kstar}"
        f"   ·   beyond it, no retriever can add evidence"
        if kstar is not None
        else "\n  K* not reached on this grid",
        fg=typer.colors.YELLOW,
    )


def _print_by_type(scale: str, block: dict[str, Any]) -> None:
    typer.secho(f"\n[{scale}] by question_type:", fg=typer.colors.CYAN)
    typer.echo(
        f"{'type':<28}{'n':>5}{'mean_gold':>10}{'c(1)':>8}{'c(2)':>8}{'c(4)':>8}  classification"
    )
    typer.echo("-" * 100)
    for ability, row in sorted(block["by_type"].items(), key=lambda kv: -kv[1]["n"]):
        probe = row["probe"]
        typer.echo(
            f"{ability:<28}{row['n']:>5}{row['mean_gold_sessions']:>10.2f}"
            f"{100 * probe.get('1', 0):7.1f}%{100 * probe.get('2', 0):7.1f}%"
            f"{100 * probe.get('4', 0):7.1f}%  {row['classification']}"
        )
    typer.secho(
        "  retriever-independent: ceiling(1)=100% — any K-effect is purely reader-side.\n"
        "  retriever-limited:     ceiling(1)=0%   — K=1 is structurally impossible.",
        fg=typer.colors.BRIGHT_BLACK,
    )


def _print_corpus_fraction(scale: str, block: dict[str, Any]) -> None:
    typer.secho(f"\n[{scale}] corpus fraction  K / |corpus|  (gap G12):", fg=typer.colors.CYAN)
    typer.echo(f"{'K':>4}{'p10':>9}{'p50':>9}{'p90':>9}{'full_ctx':>11}{'K>corpus':>11}")
    typer.echo("-" * 53)
    for key in sorted(block["per_k"], key=int):
        row = block["per_k"][key]
        q = row["quantiles"]
        typer.echo(
            f"{int(key):>4}{q['p10']:9.3f}{q['p50']:9.3f}{q['p90']:9.3f}"
            f"{100 * row['frac_full_context']:10.1f}%{100 * row['frac_k_gt_corpus']:10.1f}%"
        )
    typer.secho(
        "  full_ctx = fraction where K/|corpus| == 1.0, i.e. the condition has silently\n"
        "  become full-context. The same K is a different system at different scales.",
        fg=typer.colors.BRIGHT_BLACK,
    )


@app.command("data-lock")
def data_lock(
    verify: Annotated[
        bool,
        typer.Option("--verify", help="Check raw files against the committed data.lock."),
    ] = False,
    lock_path: Annotated[Path, typer.Option("--lock", help="Path to data.lock.")] = Path(
        "data.lock"
    ),
) -> DataLockResult:
    """Write SHA-256, byte size and row count for every raw benchmark file."""
    from msat.datalock import build_lock, dump_lock, verify_lock

    if verify:
        _assert_offline("data-lock")
        problems = verify_lock(lock_path)
        for problem in problems:
            typer.secho(f"  DRIFT  {problem}", fg=typer.colors.RED, err=True)
        if problems:
            _fail(
                f"{len(problems)} raw input(s) do not match data.lock. "
                "Every committed artifact derived from them is no longer reproducible.",
                code=EXIT_DRIFT,
            )
        res = DataLockResult(lock_path=lock_path, verified=True, drifted=False, problems=problems)
        return renderer.render(res)

    lock = build_lock()
    dump_lock(lock, lock_path)
    log.info("data lock written", extra={"lock": str(lock_path)})
    res = DataLockResult(lock_path=lock_path, verified=False, drifted=False, lock=lock)
    return renderer.render(res)


@app.command("build-manifest")
def build_manifest_command(
    run_spec: Annotated[
        Path | None,
        typer.Argument(help="Run-spec YAML; defaults to configs/design.yaml."),
    ] = None,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help="Execution profile: 'fast' (Scale S, primary reader, 1 run/cell) or 'full' (canonical pre-registered grid).",
        ),
    ] = "full",
    out: Annotated[Path | None, typer.Option("--out", help="Output file or directory.")] = None,
    corpus: Annotated[
        str, typer.Option("--corpus", help="synthetic | longmemeval")
    ] = "longmemeval",
    limit: Annotated[int | None, typer.Option("--limit", help="Cap questions per scale.")] = None,
) -> PlanResult:
    """Build the Condition manifest and execution plan."""
    from msat.config import load_bundle
    from msat.manifest import build_manifest, stats_from_corpus, write_manifest
    from msat.planning import plan_from_manifest, verify_crossed_grid, write_plan

    root = Path(run_spec).parent if run_spec else Path("configs")
    if not (root / "design.yaml").exists():
        raise typer.BadParameter(f"no design.yaml in {root}")
    bundle = load_bundle(root)
    bundle, profile_resolved = _resolve_profile_bundle(bundle, profile)
    if profile_resolved == "fast":
        typer.secho(
            "  [FAST PROFILE] scale: S | reader: gemma | repetitions: 1 (rapid pilot tier)",
            fg=typer.colors.CYAN,
            bold=True,
        )
    cpt = float(bundle.pricing.get("token_estimates", {}).get("chars_per_token", 4.0))

    corpus_map: dict[str, list[Any]] = {}
    raw_counts: dict[str, list[tuple[int, int]]] = {}

    if corpus == "synthetic":
        from msat.selftest import synthetic_corpus

        corpus_map, raw_counts = synthetic_corpus(cpt)
    else:
        from msat.datasets.longmemeval import iter_metadata, raw_path

        for scale in bundle.design.scales:
            split = scale.upper()
            stats: list[Any] = []
            counts: list[tuple[int, int]] = []
            for q in iter_metadata(raw_path(split)):
                stats.append(
                    stats_from_corpus(
                        q.qid, q.corpus.n_sessions, q.corpus.n_chars, chars_per_token=cpt
                    )
                )
                counts.append((q.corpus.n_sessions, q.corpus.n_chars))
                if limit and len(stats) >= limit:
                    break
            corpus_map[scale] = stats
            raw_counts[scale] = counts

    qids_by_scale = {
        scale: [str(s.question_id) for s in stats]
        for scale, stats in corpus_map.items()
        if scale in bundle.design.scales
    }
    manifest = build_manifest(
        bundle,
        corpus_map,
        chars_per_token=cpt,
        raw_counts=raw_counts,
        metadata={
            "qids_by_scale": qids_by_scale,
            "seed": int(bundle.design.seed),
            "runs_per_cell": int(bundle.design.runs_per_cell),
            "preregistration_hash": bundle.preregistration_hash,
        },
    )

    manifest_filename = "manifest_fast.json" if profile_resolved == "fast" else "manifest.json"
    loc = OutputLocation.file(out, default_dir="work", filename=manifest_filename)
    loc.ensure_parent()
    dest = loc.path
    write_manifest(manifest, dest)
    log.info(
        "manifest built",
        extra={"config_hash": manifest.config_hash, "n_conditions": len(manifest.conditions)},
    )

    executable = plan_from_manifest(
        dest,
        qids_by_scale,
        seed=int(bundle.design.seed),
        repetition=int(bundle.design.repetition),
    )
    plan_path = write_plan(executable, dest.with_name(f"{dest.stem}.plan.json"))
    crossed = verify_crossed_grid(
        executable,
        k_grid=bundle.design.K_grid,
        b_grid=bundle.design.b_values,
        scales=bundle.design.scales,
        readers=bundle.design.readers,
    )
    crossed_path = dest.with_name("crossed_grid.json")
    crossed_path.write_text(json.dumps(crossed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log.info(
        "crossed grid verified",
        extra={
            "config_hash": manifest.config_hash,
            "crossed": crossed["crossed"],
            "n_planned_cells": crossed["n_planned_cells"],
        },
    )

    res = PlanResult(
        manifest_path=dest,
        plan_path=plan_path,
        crossed_path=crossed_path,
        config_hash=manifest.config_hash,
        conditions=len(manifest.conditions),
        supported=len(manifest.supported),
        unsupported=len(manifest.unsupported),
        degenerate_conditions=len(manifest.fully_degenerate),
        observations=executable.n_observations,
        crossed=bool(crossed["crossed"]),
        reader_balanced=bool(crossed.get("reader_balanced", False)),
    )
    renderer.render(res)

    for cell in manifest.rejected:
        typer.secho(
            f"    REJECTED  {cell.condition.scale}/{cell.condition.reader} "
            f"K={cell.condition.k} B={int(cell.condition.b)}   {cell.reason}",
            fg=typer.colors.RED,
        )
        if cell.example_reason:
            typer.echo(f"      e.g. {cell.example_reason}")

    partial = [c for c in manifest.accepted if c.n_degenerate]
    if partial:
        worst = max(partial, key=lambda c: c.degeneracy_fraction)
        typer.secho(
            f"\n  {len(partial)} accepted cell(s) are degenerate for SOME questions. "
            f"Worst: {worst.condition.scale} K={worst.condition.k} at "
            f"{worst.degeneracy_fraction:.0%}. Those questions are excluded from "
            f"paired comparisons at P4.3.",
            fg=typer.colors.YELLOW,
        )

    for problem in crossed["problems"]:
        typer.secho(f"    NOT CROSSED  {problem}", fg=typer.colors.RED)

    if manifest.rejected:
        typer.secho(
            "  A rejected cell is not one that runs badly - it is a condition already "
            "satisfied before the manipulation is applied. Excluding it is the point.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_DEGENERATE_CELL)
    if not crossed["crossed"]:
        typer.secho(
            f"  {crossed['n_expected_cells'] - crossed['n_planned_cells']} declared "
            "(K, B) cell(s) are absent from the plan. Arms A and G must be ONE "
            "crossed grid (PRD §R4.4): a pair of marginal sweeps produces no "
            "interaction term, which is the paper's headline claim.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_DEGENERATE_CELL)
    return res


build_plan = build_manifest_command


@app.command("run")
def run_experiment(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    profile: Annotated[
        str, typer.Option("--profile", help="Execution profile: fast or full.")
    ] = "full",
    resume: Annotated[bool, typer.Option("--resume", help="Resume durable work.")] = False,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs"),
    plan: Annotated[Path | None, typer.Option("--plan", help="Built manifest.")] = None,
    results_dir: Annotated[
        Path | None, typer.Option("--results", help="Parent directory for run output.")
    ] = None,
    rung: Annotated[
        str | None, typer.Option("--rung", help="negative, baselines, or crossed.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Validate without generation.")
    ] = False,
    questions: Annotated[
        Path | None,
        typer.Option("--questions", help="Benchmark questions or gold-session mapping."),
    ] = None,
    gold: Annotated[Path | None, typer.Option("--gold", help="Gold answers.")] = None,
    arm_e_pool: Annotated[
        Path | None, typer.Option("--arm-e-pool", help="Cross-question control pool.")
    ] = None,
    reader_id: Annotated[
        str | None,
        typer.Option("--reader", help="Restrict a control or baseline rung to one reader."),
    ] = None,
    data: Annotated[
        Path | None, typer.Option("--data", help="Directory of local benchmark files.")
    ] = None,
    filler: Annotated[
        Path | None, typer.Option("--filler", help="Filler sessions for arm B.")
    ] = None,
) -> RunResult:
    """Validate and execute one rung through the durable run module."""
    from msat.config import ConfigError, load_bundle
    from msat.providers.base import BackendUnavailableError
    from msat.runner.input import MissingInputError
    from msat.runner.lifecycle import CapacityRefusalError, GateFailedError
    from msat.runner.run_workflow import run_lifecycle_workflow

    resolved = RunIdentitySeam.resolve(run_id, run)
    if results_dir is not None:
        base_dir = Path(results_dir)
        run_dir = base_dir / resolved
    else:
        base_dir = Path("work")
        run_dir = base_dir / resolved
        if not run_dir.is_dir() and (Path("results") / resolved).is_dir():
            run_dir = Path("results") / resolved
    try:
        bundle, resolved_profile = _resolve_profile_bundle(load_bundle(config), profile)
    except (ConfigError, ValueError) as exc:
        _fail(str(exc), EXIT_CONFIG_INVALID)
    state = _read_ladder(run_dir / "ladder.json")
    chosen = _resolve_rung(rung, state)
    manifest_name = "manifest_fast.json" if resolved_profile == "fast" else "manifest.json"
    manifest = plan or next(
        (
            candidate
            for candidate in (
                base_dir / manifest_name,
                Path("work") / manifest_name,
                Path("results") / manifest_name,
                config.parent / "results" / manifest_name,
                config / manifest_name,
            )
            if candidate.is_file()
        ),
        base_dir / manifest_name,
    )
    typer.echo(f"  profile     {resolved_profile}, readers {', '.join(bundle.design.readers)}")
    typer.echo(f"  config_hash {bundle.config_hash}")
    typer.echo(f"  ladder      entering rung {chosen!r}")

    workflow_rung = chosen

    inputs = InputSeam.resolve_run_inputs(
        questions_path=questions,
        gold_path=gold,
        pool_path=arm_e_pool,
        filler_path=filler,
        data_dir=data,
        scales=bundle.design.scales,
    )

    try:
        result = run_lifecycle_workflow(
            bundle,
            run_dir,
            rung=cast("RungName", workflow_rung),
            run_id=resolved,
            inputs=inputs,
            plan_path=manifest,
            reader_id=reader_id,
            panel=_panel(bundle),
            resume=resume,
            dry_run=dry_run,
        )
    except BackendUnavailableError as exc:
        _fail(
            f"The reader has no loadable backend: {exc}. Nothing was simulated and no scientific verdict is claimed."
        )
    except GateFailedError as exc:
        control = run_dir / "arm_e_control.json"
        if control.is_file():
            report = json.loads(control.read_text(encoding="utf-8"))
            if report.get("verdict") is not None:
                typer.secho(f"  arm E       {report['verdict']}", fg=typer.colors.RED)
                heading = (
                    "ARM E DID NOT COLLAPSE"
                    if report["verdict"] == "FAIL" and report.get("coverage_complete", True)
                    else "ARM E NOT EVALUATED"
                )
                typer.secho(f"  {heading}", fg=typer.colors.RED, bold=True)
        typer.secho(f"  NO-GO: {exc}", fg=typer.colors.RED, bold=True)
        _fail(str(exc), EXIT_NO_GO)
    except CapacityRefusalError as exc:
        _fail(str(exc), EXIT_DEGENERATE_CELL)
    except MissingInputError as exc:
        _fail(str(exc))
    except (ConfigError, ValueError, KeyError, OSError) as exc:
        _fail(str(exc), EXIT_CONFIG_INVALID)

    res = RunResult(
        run_id=resolved,
        run_dir=run_dir,
        rung=chosen,
        profile=resolved_profile,
        dry_run=dry_run,
        data=result,
    )
    renderer.render(res)
    return res


def _resolve_rung(rung: str | None, state: LadderState) -> str:
    valid_rungs = ("negative", "baselines", "crossed")
    if rung is not None:
        if rung not in valid_rungs:
            _fail(f"unknown rung {rung!r}; the ladder has {list(valid_rungs)}")
        return rung
    nxt = state.first_unpassed()
    if nxt:
        return nxt.name
    return "crossed"


def _evaluate_gate(
    out_dir: Path,
    *,
    context: str,
    run_id: str,
    config_hash: str,
    notes: Sequence[str] = (),
    **inputs: Any,
) -> Any:
    from msat.runner.lifecycle import evaluate_gate

    verdict = evaluate_gate(
        out_dir,
        context=context,
        run_id=run_id,
        config_hash=config_hash,
        notes=notes,
        **inputs,
    )
    log.info(
        "p5 gate evaluated",
        extra={
            "run_id": run_id,
            "config_hash": config_hash,
            "gate_context": context,
            "outcome": str(verdict.outcome),
            "passed_branches": verdict.passed_branches,
        },
    )
    typer.echo(f"  gate         {verdict.outcome}  ({context})")
    for branch in verdict.branches:
        mark = "PASS" if branch.passed else "fail"
        colour = typer.colors.GREEN if branch.passed else typer.colors.RED
        typer.secho(f"    {mark}  {branch.branch}", fg=colour)
    return verdict


def _gate_exit(verdict: Any, *, message: str) -> NoReturn:
    typer.secho(f"\n  NO-GO: {message}", fg=typer.colors.RED, bold=True)
    for branch in verdict.branches:
        typer.secho(f"    {branch.branch}: {branch.detail}", fg=typer.colors.RED)
    for note in verdict.notes:
        typer.secho(f"    note: {note}", fg=typer.colors.YELLOW)
    typer.secho(
        "  A BOUNDED_NEGATIVE is a result, not a failure — and it is decided with the "
        "guide IN ADVANCE,\n  not discovered after the budget is spent.",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=EXIT_NO_GO)


def _check_capacity(reader: Any, prompt_tokens: int, *, label: str = "") -> CapacityVerdict:
    from msat.runner.lifecycle import check_capacity

    return check_capacity(reader, prompt_tokens, label=label)


def capacity_report(
    verdicts: Sequence[CapacityVerdict],
    *,
    coverage: dict[str, Any] | None = None,
    headroom: dict[str, Any] | None = None,
) -> dict[str, Any]:
    per_reader: dict[str, dict[str, Any]] = {}
    for v in verdicts:
        row = per_reader.setdefault(
            v.reader,
            {
                "n_checked": 0,
                "n_supported": 0,
                "n_unsupported": 0,
                "reader_capacity_tokens": v.reader_capacity_tokens,
                "vram_capacity_tokens": v.vram_capacity_tokens,
                "declared_context_window": v.declared_context_window,
                "binding_constraint": v.binding_constraint,
            },
        )
        row["n_checked"] += 1
        row["n_supported" if v.supported else "n_unsupported"] += 1
    return {
        "requirement": "PRD §R3.2",
        "policy": "reject_cell — never truncate silently (retrieval.yaml context_policy)",
        "on_overflow": "reject_cell",
        "never_truncate_silently": True,
        "n_checked": len(verdicts),
        "n_supported": sum(1 for v in verdicts if v.supported),
        "n_unsupported": sum(1 for v in verdicts if not v.supported),
        "supported": [v.as_dict() for v in verdicts if v.supported],
        "unsupported": [v.as_dict() for v in verdicts if not v.supported],
        "per_reader": per_reader,
        "panel_coverage": coverage,
        "headroom_sensitivity": headroom,
        "note": (
            "An unsupported condition is reported, not repaired. Trimming the prompt leaves it "
            "able to contain its own answer while appearing not to, and the curve that "
            "follows measures nothing (PRD §R0.3 — propose, never truncate)."
        ),
    }


def _panel(bundle: Any) -> list[Any]:
    from msat.providers.local import readers_from_config

    try:
        readers = readers_from_config(
            bundle.raw["readers.yaml"], allow_single=bundle.mode == "fast"
        )
        return [r for r in readers if r.id in bundle.design.readers]
    except (ValueError, KeyError) as exc:
        _fail(f"the reader panel cannot be built: {exc}")
        raise AssertionError("unreachable") from exc  # pragma: no cover


def _capacity_preflight(bundle: Any, plan: Any, panel: Sequence[Any]) -> dict[str, Any]:
    from msat.hardware.capacity import preflight_capacity_report

    return preflight_capacity_report(bundle, plan, panel=list(panel))


def _plan_from(manifest_path: Path, bundle: Any, qids: dict[str, list[str]]) -> Any:
    from msat.planning import plan_from_manifest

    return plan_from_manifest(
        manifest_path,
        qids,
        seed=int(bundle.design.seed),
        runs_per_cell=int(bundle.design.runs_per_cell),
    )


def _crossing(plan: Any, bundle: Any) -> tuple[dict[str, Any], int, list[dict[str, Any]]]:
    from msat.runner.lifecycle import crossing

    return crossing(plan, bundle)


def _load_gold_sessions(path: Path) -> dict[str, list[str]]:
    """Keys are session ids, never chunk ids."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> [answer_session_ids].")
    return {str(k): [str(s) for s in v] for k, v in raw.items()}  # type: ignore[union-attr]


panel = _panel
capacity_preflight = _capacity_preflight
plan_from = _plan_from
crossing = _crossing
load_gold_sessions = _load_gold_sessions
evaluate_gate = _evaluate_gate
gate_exit = _gate_exit
check_capacity = _check_capacity


def load_control_pool(path: Path) -> tuple[Any, dict[str, str]]:
    from msat.arms.negative import PooledMemory
    from msat.arms.render import Memory

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> {{question, memories}}.")
    by_question: dict[str, list[Any]] = {}
    question_text: dict[str, str] = {}
    for qid, entry in raw.items():
        if not isinstance(entry, dict) or "memories" not in entry:
            _fail(f"{path}: entry {qid!r} must be an object with a `memories` list.")
        question_text[str(qid)] = str(entry.get("question", ""))
        memories = []
        for raw_memory in entry["memories"]:  # type: ignore[index]
            memories.append(
                Memory(
                    memory_id=str(raw_memory["memory_id"]),
                    session_id=str(raw_memory["session_id"]),
                    source_question_id=str(qid),
                    text=str(raw_memory.get("text", "")),
                    n_tokens=int(raw_memory.get("n_tokens", 0)),
                )
            )
        by_question[str(qid)] = memories
    return PooledMemory(by_question=by_question), question_text


def run_negative_control(
    *,
    bundle: Any,
    arm: Any,
    cells: Sequence[Any],
    readers: Sequence[Any],
    pool: Any,
    question_text: dict[str, str],
    gold_by_question: dict[str, list[str]],
    gold_answers: dict[str, str],
    generate: Callable[[Any, Any], Any],
    max_questions: int | None = None,
) -> dict[str, Any]:
    from msat.arms.negative import ControlConfig, evaluate, go_no_go, render_control
    from msat.metrics.normalisation import token_f1
    from msat.providers.base import BackendUnavailableError

    config = ControlConfig.from_template(
        (bundle.prompts or {}).get("prompts/arm_e.j2") or (bundle.prompts or {}).get("arm_e.j2", "")
    )
    chars_per_token = float(
        (bundle.pricing or {}).get("token_estimates", {}).get("chars_per_token", 4.0)
    )
    gold_sets = {q: frozenset(v) for q, v in gold_by_question.items()}
    qids = [q for q in sorted(gold_sets) if gold_sets[q] and q in pool.by_question]
    if max_questions is not None:
        qids = qids[:max_questions]
    if not qids:
        _fail(
            "no question has BOTH declared gold sessions and a pool entry, so Arm E "
            "cannot be count-matched against anything. --questions and --arm-e-pool "
            "must cover at least one common qid that declares `answer_session_ids`."
        )

    normaliser = _load_normaliser()
    verdicts: list[CapacityVerdict] = []
    refusals: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    scores: list[float] = []
    recalls: list[float] = []
    backend_error: dict[str, Any] | None = None
    n_generated = 0
    n_unattributed = 0

    for cell in cells:
        reader = next((r for r in readers if r.id == cell.reader), None)
        if reader is None:
            continue
        for qid in qids:
            gold = gold_sets[qid]
            label = f"E/{cell.scale}/{reader.id}/K={cell.k}/B={int(cell.b)}/{qid}"
            try:
                rendered = render_control(
                    arm,
                    question_text.get(qid, ""),
                    pool,
                    qid,
                    gold,
                    cell,
                    config,
                    budget_tokens=int(cell.b),
                    qid=qid,
                    gold_by_question=gold_sets,
                )
            except ValueError as exc:
                refusals.append(
                    {
                        "label": label,
                        "kind": "underfilled_control",
                        "reason": str(exc),
                        "prompt_tokens": None,
                        "reader_capacity_tokens": reader.reader_capacity_tokens,
                        "binding_constraint": reader.binding_constraint,
                    }
                )
                continue

            verdict = _check_capacity(
                reader,
                int(len(rendered.prompt) / chars_per_token) + 1,
                label=label,
            )
            verdicts.append(verdict)
            if not verdict.supported:
                refusals.append({**verdict.as_dict(), "kind": "context_overflow"})
                continue

            try:
                result = generate(reader, _generation_request(bundle, reader, rendered))
            except (BackendUnavailableError, ImportError, OSError) as exc:
                backend_error = {
                    "available": False,
                    "reader": reader.id,
                    "backend": str(getattr(reader, "backend_name", "transformers")),
                    "error": type(exc).__name__,
                    "detail": str(exc),
                    "n_generated_before_failure": n_generated,
                    "remedy": (
                        "Arm E is generated by a LOCAL engine; there is no API reader "
                        "in this design. Install `bitsandbytes` for the 4-bit path "
                        "(uv sync --extra bits) or run vLLM as a service "
                        "(make vllm), then re-run. `msat validate-config` reports "
                        "backend availability without loading weights."
                    ),
                }
                break

            n_generated += 1
            hits = sorted(set(rendered.session_ids) & gold)
            draws.append(
                {
                    "label": label,
                    "n_drawn": rendered.n_memories,
                    "count_matched_to_K": rendered.n_memories == cell.k,
                    "session_ids": list(rendered.session_ids),
                    "gold_sessions_drawn": hits,
                }
            )
            if gold:
                recalls.append(len(hits) / len(gold))
            answer = gold_answers.get(qid)
            if answer is None:
                n_unattributed += 1
                continue
            scores.append(token_f1(result.text, answer, normaliser))

    accuracy = sum(scores) / len(scores) if scores else None
    recall = sum(recalls) / len(recalls) if recalls else None
    outcome = evaluate(accuracy, recall, config, n_questions=len(scores))
    report: dict[str, Any] = {
        **go_no_go(outcome),
        "arm": "E",
        "n_cells": len(cells),
        "n_questions_considered": len(qids),
        "n_generated": n_generated,
        "n_refused": len(refusals),
        "n_scored": len(scores),
        "backend": backend_error
        or {
            "available": True,
            "backend": str(getattr(readers[0], "backend_name", "transformers"))
            if readers
            else None,
        },
        "diagnostics": {
            "accuracy_is_none_because": (
                "no usable control output was scored" if accuracy is None else None
            ),
            "recall_is_none_because": (
                "no drawn control carried a gold session, so recall is undefined"
                if recall is None
                else None
            ),
            "n_unattributed": n_unattributed,
            "refusal_kinds": sorted({str(r.get("kind")) for r in refusals}),
            "thresholds_measurable": config.thresholds_measurable,
        },
        "capacity": capacity_report(verdicts),
        "refusals": refusals,
        "control": {
            "count_matched": all(d["count_matched_to_K"] for d in draws) if draws else None,
            "mean_recall_of_gold": recall,
            "mean_token_f1": accuracy,
            "draws": draws,
        },
        "why_it_matters": (
            "Arm E draws memories from OTHER questions, count-matched to the Arm A "
            "condition so only the CONTENT differs. A correct reader scores near "
            "zero. If it does not, the pipeline is not respecting the condition and "
            "every other number in this project is suspect (PRD §R4.3 / §G5)."
        ),
    }
    return report


def _generation_request(bundle: Any, reader: Any, rendered: Any) -> Any:
    from msat.runner.lifecycle import _generation_request as build_request

    return build_request(bundle, reader, rendered)


def latency_artifact(store: Any) -> dict[str, Any]:
    from msat.metrics.scoring_workflow import build_latency_artifact

    return build_latency_artifact(store)


def _print_latency(artifact: dict[str, Any]) -> None:
    overall = artifact["total_latency"]["overall"]
    ttft = artifact["ttft"]
    typer.secho(
        "\n  latency (R8.5) — medians, IQR and quantiles; never a bootstrap CI",
        fg=typer.colors.CYAN,
    )
    typer.echo(f"    {'series':<10}{'n':>7}{'p50':>10}{'IQR':>10}{'p90':>10}{'p99':>11}   mean")
    for label, block in (("total", overall), ("ttft", ttft)):
        q = block["quantiles_ms"]
        typer.echo(
            f"    {label:<10}{block['n']:>7}{q.get('p50', 0):>10,.0f}"
            f"{block['iqr_ms']:>10,.0f}{q.get('p90', 0):>10,.0f}{q.get('p99', 0):>11,.0f}"
            f"{block['mean_ms']:>10,.0f}"
        )
    if ttft["n"] == 0:
        typer.secho(
            "    no TTFT was recorded on any response, so the reader's own latency to "
            "first token is unreported rather than inferred.",
            fg=typer.colors.YELLOW,
        )
    if not ttft["p99_reliable"]:
        typer.secho(
            f"    p99 rests on n={ttft['n']}; below ~100 calls it is one or two "
            "observations and reads as a precise number.",
            fg=typer.colors.YELLOW,
        )
    for reader, block in sorted(artifact["total_latency"].get("by_reader", {}).items()):
        typer.echo(
            f"      {reader:<8} p50 {block['quantiles_ms'].get('p50', 0):>9,.0f} ms   "
            f"IQR {block['iqr_ms']:>9,.0f} ms   n={block['n']}"
        )
    queue = artifact["total_latency"]["queue_wait"]
    if queue["recorded"] and not queue["recording_concurrency"]:
        typer.secho(
            f"    queue wait is {100 * queue['share_of_latency']:.0f}% of total latency, "
            f"so `total` describes THIS HARNESS, not the reader. Use `queue_excluded_*` "
            f"to compare readers.",
            fg=typer.colors.YELLOW,
        )
    if artifact["n_impossible_records"]:
        typer.secho(
            f"    {artifact['n_impossible_records']} record(s) declare a TTFT below "
            f"their queue wait and were EXCLUDED rather than repaired.",
            fg=typer.colors.YELLOW,
        )


@app.command("status")
def status(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    results_dir: Annotated[
        Path, typer.Option("--results", help="Parent of the per-run directory.")
    ] = Path("results"),
    ladder: Annotated[
        bool, typer.Option("--ladder/--no-ladder", help="Show the smoke ladder state.")
    ] = True,
) -> StatusResult:
    """Print the per-cell completion ledger and spend to date."""
    from msat.runner.executor import RunConfig, status_report

    resolved = RunIdentitySeam.resolve(run_id, run)
    _assert_offline("status")
    loc = OutputLocation.directory(results_dir, default="results")
    run_dir = loc.path / resolved
    # If not found in requested dir and default is results or work, check the other
    if not run_dir.is_dir() and (Path("results") / resolved).is_dir():
        run_dir = Path("results") / resolved
    elif not run_dir.is_dir() and (Path("work") / resolved).is_dir():
        run_dir = Path("work") / resolved

    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}. Build one with `msat build-manifest`.")

    report = status_report(RunConfig(run_dir=run_dir))

    if report["n_done"] > report["n_stored_responses"]:
        typer.secho(
            f"\n  INCONSISTENT: the ledger credits {report['n_done'] - report['n_stored_responses']:,} "
            f"more unit(s) than the store holds. The ledger is ahead of the record.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=EXIT_ASSERTION_FAILED)

    ladder_state = None
    next_rung_name = None
    if ladder:
        state_path = run_dir / "ladder.json"
        ladder_state = _read_ladder(state_path)
        nxt = ladder_state.first_unpassed()
        if nxt is not None:
            next_rung_name = nxt.name

    res = StatusResult(
        run_id=resolved,
        run_dir=run_dir,
        n_stored_responses=report["n_stored_responses"],
        n_done=report["n_done"],
        n_planned=report["n_planned"],
        n_failed=report["n_failed"],
        n_excluded=report["n_excluded"],
        n_cells=report["n_cells"],
        cells=report.get("cells", []),
        ladder_state=ladder_state,
        next_rung=next_rung_name,
    )
    renderer.render(res)
    return res


@app.command("score")
def score(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    profile: Annotated[
        str, typer.Option("--profile", help="Scoring profile: fast or full.")
    ] = "full",
    metric: Annotated[
        str, typer.Option("--metric", help="One of: em, f1, judge, official, all.")
    ] = "all",
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("results"),
    gold: Annotated[
        Path | None, typer.Option("--gold", help="Gold answers JSON. Required.")
    ] = None,
    config: Annotated[Path, typer.Option("--config")] = Path("configs"),
    official: Annotated[
        Path | None, typer.Option("--official", help="Cached official scores.")
    ] = None,
    verdicts: Annotated[
        Path | None, typer.Option("--verdicts", help="Cached judge verdicts.")
    ] = None,
    metadata: Annotated[
        Path | None, typer.Option("--metadata", help="Observation metadata.")
    ] = None,
) -> ScoreResult:
    """Stored scoring: re-score stored responses using the selected configuration; never calls a model."""
    from msat.metrics.scoring_workflow import (
        InvalidScoringConfigurationError,
        run_scoring_workflow,
        validate_mode_and_metrics,
    )

    resolved = RunIdentitySeam.resolve(run_id, run)
    _assert_offline("score")
    profile_clean = profile.lower().strip()
    if profile_clean not in ("fast", "full"):
        raise typer.BadParameter(f"profile must be 'fast' or 'full', got {profile!r}")
    try:
        validate_mode_and_metrics(profile_clean, metric)
    except InvalidScoringConfigurationError as exc:
        _fail(str(exc))

    loc = OutputLocation.directory(results_dir, default="results")
    run_dir = loc.path / resolved
    if not run_dir.is_dir() and (Path("results") / resolved).is_dir():
        run_dir = Path("results") / resolved
    elif not run_dir.is_dir() and (Path("work") / resolved).is_dir():
        run_dir = Path("work") / resolved

    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}.")

    inputs = InputSeam.resolve_score_inputs(
        gold=gold,
        official=official,
        verdicts=verdicts,
        metadata=metadata,
        config=config,
    )

    try:
        result = run_scoring_workflow(
            run_dir,
            gold=inputs.gold,
            config_dir=inputs.config,
            mode=profile_clean,
            metric=metric,
            official=inputs.official,
            verdicts_path=inputs.verdicts,
            metadata_path=inputs.metadata,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc))

    res = ScoreResult(
        run_id=resolved,
        scores_path=result.scores_path,
        latency_path=result.latency_path,
        report=result.report,
        latency_data=result.latency_data,
        disagreements_path=result.disagreements_path,
        notes=list(result.notes),
    )
    renderer.render(res)
    return res


@app.command("judge-audit")
def judge_audit(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    verdicts_path: Annotated[
        Path | None,
        typer.Option("--verdicts", help="JSON list of verdict records. Required."),
    ] = None,
    out: Annotated[Path, typer.Option("--out", help="Output file.")] = Path(
        "results/judge_audit.json"
    ),
    human: Annotated[Path | None, typer.Option("--human", help="Filled human_sample.csv.")] = None,
) -> JudgeAuditResult:
    """Two passes of the same deterministic system under the same judge."""
    from msat.judges.report import build_audit

    resolved = RunIdentitySeam.resolve(run_id, run)
    _assert_offline("judge-audit")
    inputs = InputSeam.resolve_judge_audit_inputs(verdicts_path=verdicts_path, human=human)
    records = json.loads(inputs.verdicts.read_text(encoding="utf-8"))
    human_data = _load_human_agreement(inputs.human) if inputs.human else None

    loc = OutputLocation.file(out, default="results/judge_audit.json")
    loc.ensure_parent()

    audit = build_audit(records, human=human_data)
    dest = audit.write(loc.path)

    floor = audit.noise_floor or 0.0
    gate_verdict = _evaluate_gate(
        dest.parent,
        context=f"{resolved}/judge_audit",
        run_id=resolved,
        config_hash="",
        notes=[
            f"Judge panel: kappa_min={audit.panel.get('kappa_min')} against a bar of "
            f"{audit.panel.get('cautionary_bar')}, {audit.n_judges} judge(s), "
            f"{audit.n_items:,} item(s).",
            "Branches (i)-(iii) are empty because this command reads verdict records "
            "only (R8.7): it is a STANDALONE artefact and does not need scores.",
        ],
        accuracy_by_cell={},
        recall_by_k={},
        stratified=None,
        degenerate_cells=(
            [] if audit.panel_usable else [{"judge_pair": p} for p in _unusable_pairs(audit)]
        ),
        n_interaction_cells=0,
        judge_floor_pp=100.0 * floor,
    )

    res = JudgeAuditResult(
        run_id=resolved,
        out_path=dest,
        audit=audit,
        panel_usable=bool(audit.panel_usable),
    )
    renderer.render(res)

    if not audit.panel_usable:
        _gate_exit(gate_verdict, message="the judge panel is not usable (PRD §R8.2)")
    return res


def _unusable_pairs(audit: Any) -> list[str]:
    panel = getattr(audit, "panel", {}) or {}
    return [
        str(entry.get("pair", "?"))
        for entry in panel.get("verdicts", []) or []
        if "BELOW" in str(entry.get("verdict", "")) or "UNDEFINED" in str(entry.get("verdict", ""))
    ]


def _load_gold(path: Path) -> dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        _fail(f"{path} must be a JSON object mapping qid -> gold answer.")
    return {str(k): str(v) for k, v in raw.items()}


def _load_official(results_dir: Path, run_id: str) -> dict[str, float] | None:
    p = results_dir / run_id / "official.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in raw.items()}


def _load_human_agreement(path: Path) -> dict[str, Any]:
    from msat.judges.human_sample import judge_vs_human, read_human_csv

    labelled = read_human_csv(Path(path))
    judge_side_path = Path(path).with_name("judge_side.csv")
    if not judge_side_path.exists():
        _fail(
            f"{judge_side_path} is missing. The labeller file and the judge's view are "
            f"written separately on purpose - showing a labeller the judge's verdicts "
            f"makes them agree with it more often, and the resulting agreement figure "
            f"measures the anchor rather than the judge."
        )
    judge_side = read_human_csv(judge_side_path)
    out = dict(judge_vs_human(labelled, judge_side))
    out["status"] = "RUN"
    return out


@app.command("analyze")
def analyze(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    offline: Annotated[bool, typer.Option("--offline", help="Assert no network access.")] = True,
    check: Annotated[
        bool, typer.Option("--check", help="Recompute scores from cache before analysis.")
    ] = False,
    results_dir: Annotated[Path, typer.Option("--results")] = Path("results"),
    config: Annotated[Path, typer.Option("--config")] = Path("configs"),
    profile: Annotated[
        str, typer.Option("--profile", help="fast pilot or full confirmatory profile.")
    ] = "full",
    outcome: Annotated[
        str, typer.Option("--outcome", help="Binary correctness metric: em or official.")
    ] = "em",
    metadata: Annotated[
        Path | None, typer.Option("--metadata", help="Observation metadata.")
    ] = None,
    gold: Annotated[
        Path | None, typer.Option("--gold", help="Gold answers required for --check.")
    ] = None,
    bootstrap: Annotated[int, typer.Option("--bootstrap", min=1)] = 200,
    seed: Annotated[int, typer.Option("--seed")] = 0,
) -> AnalyzeResult:
    """Stored-run analysis: validate stored observations, fit declared models, and derive results offline."""
    from msat.analysis.analysis_workflow import run_analysis_workflow
    from msat.metrics.scoring_workflow import run_scoring_workflow

    resolved = RunIdentitySeam.resolve(run_id, run)
    _assert_offline("analyze")
    profile_clean = profile.lower().strip()
    if profile_clean not in ("fast", "full"):
        raise typer.BadParameter(f"profile must be 'fast' or 'full', got {profile!r}")
    loc = OutputLocation.directory(results_dir, default="results")
    run_dir = loc.path / resolved
    if not run_dir.is_dir() and (Path("results") / resolved).is_dir():
        run_dir = Path("results") / resolved
    elif not run_dir.is_dir() and (Path("work") / resolved).is_dir():
        run_dir = Path("work") / resolved

    try:
        if check:
            if gold is None:
                _fail("--gold is required with --check to recompute metrics from cache.")
            run_scoring_workflow(
                run_dir,
                gold=gold,
                config_dir=config,
                mode=profile_clean,
                metric=outcome,
                metadata_path=metadata,
            )
        result = run_analysis_workflow(
            run_dir,
            config_root=config,
            selected_outcome=outcome,
            metadata_path=metadata,
            strict=profile_clean == "full",
            n_boot=bootstrap,
            seed=seed,
        )
    except (OSError, ValueError) as exc:
        _fail(str(exc), code=EXIT_ASSERTION_FAILED if isinstance(exc, ValueError) else EXIT_FAILURE)

    res = AnalyzeResult(
        run_id=resolved,
        run_dir=run_dir,
        n_observations=result.n_observations,
        status=result.status,
        analysis_json_path=result.analysis_json_path,
        results_md_path=result.results_md_path,
        notes=list(result.notes),
    )
    renderer.render(res)

    if result.status != "SUCCESS":
        _fail(
            "The declared primary model is unestimable; see the recorded diagnostics.",
            EXIT_UNDERPOWERED,
        )
    return res


@app.command("figures")
def figures(
    out_dir: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("figures"),
    scores: Annotated[
        Path,
        typer.Option("--scores", help="scores.parquet from `msat score`."),
    ] = Path("results/scores.parquet"),
    ceiling: Annotated[
        Path, typer.Option("--ceiling", help="ceiling.json from `msat ceiling`.")
    ] = Path("results/ceiling.json"),
    metric: Annotated[
        str,
        typer.Option(
            "--metric",
            help=(
                "Which score column the accuracy panel plots. Required, not "
                "defaulted: `em` and `official` differ (R8.1) and the axis says "
                "'accuracy', so the number behind it must be named."
            ),
        ),
    ] = "official",
    scale: Annotated[
        list[str] | None,
        typer.Option("--scale", help="Restrict to one or more scales. Default: all."),
    ] = None,
    preview: Annotated[
        bool,
        typer.Option(
            "--preview",
            help=(
                "Render the synthetic fixture, stamped 'NOT RESULTS' in the "
                "figure itself. For checking the layout, never for a claim."
            ),
        ),
    ] = False,
) -> FiguresResult:
    """Regenerate every figure from the artifacts a run wrote."""
    _assert_offline("figures")
    from msat.figures.centrepiece import (
        FigureInputMissing,
        build_from_disk,
        preview_report,
    )

    loc = OutputLocation.directory(out_dir, default="figures")
    loc.ensure_parent()
    try:
        if preview:
            report = preview_report(loc.path)
        else:
            report = build_from_disk(
                scores_path=scores,
                ceiling_path=ceiling,
                metric=metric,
                out_dir=loc.path,
                scales=scale,
            )
    except FigureInputMissing as exc:
        _fail(str(exc), EXIT_CONFIG_INVALID)

    res = FiguresResult(out_dir=loc.path, report=report)
    return renderer.render(res)


@app.command("paper")
def paper(
    out_dir: Annotated[
        Path, typer.Option("--out", help="Output directory for paper build.")
    ] = Path("../paper/build"),
) -> PaperResult:
    """Build manuscript artifacts from stored analysis results."""
    _assert_offline("paper")
    loc = OutputLocation.directory(out_dir, default="../paper/build")
    loc.ensure_parent()

    src_pdf = Path("../paper/build/MemSat/MemSat.pdf")
    dest_pdf = loc.path / "MemSat.pdf"
    if src_pdf.is_file() and src_pdf.resolve() != dest_pdf.resolve():
        shutil.copy2(src_pdf, dest_pdf)

    res = PaperResult(
        out_dir=loc.path, pdf_path=dest_pdf if dest_pdf.exists() else None, built=True
    )
    return renderer.render(res)


@app.command("report-reader-capacity")
def report_reader_capacity(
    config: Annotated[Path, typer.Option("--config", help="Config directory.")] = Path("configs"),
    out: Annotated[Path | None, typer.Option("--out", help="Write the plan as JSON.")] = None,
) -> MaxContextResult:
    """Report Reader capacity: how much of each conversation each reader can be shown."""
    from msat.arms.max_context import max_context_plan
    from msat.config import load_bundle
    from msat.hardware.capacity import compute_reader_limit
    from msat.hardware.vram import DEFAULT_HEADROOM_GB

    bundle = load_bundle(config)
    cpt = float(bundle.pricing.get("token_estimates", {}).get("chars_per_token", 4.0))
    reserve = int(bundle.decoding.max_tokens)

    by_scale: dict[str, dict[str, Any]] = {}
    for scale in bundle.design.scales:
        sc = bundle.scales.get(scale)
        limits = {
            f.id: compute_reader_limit(
                f,
                headroom_gb=float(
                    bundle.raw["readers.yaml"]
                    .get("context_policy", {})
                    .get("headroom_gb", DEFAULT_HEADROOM_GB)
                ),
            ).reader_capacity_tokens
            for f in bundle.readers.families
        }
        by_scale[scale] = max_context_plan(
            corpus_tokens=int(sc.chars_per_question / cpt),
            sessions_median=sc.sessions_median,
            limits=limits,
            reserve_tokens=reserve,
        )

    out_file = None
    if out:
        loc = OutputLocation.file(out)
        loc.ensure_parent()
        loc.path.write_text(json.dumps(by_scale, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        out_file = loc.path

    res = MaxContextResult(by_scale=by_scale, out_path=out_file)
    return renderer.render(res)


max_context = report_reader_capacity


@app.command("hardware-delta")
def hardware_delta(
    run_id: Annotated[str | None, typer.Argument(help="Run identifier.")] = None,
    run: Annotated[str | None, typer.Option("--run", help="Alias for run_id.")] = None,
    against: Annotated[
        str | None,
        typer.Option("--against", help="A second run's store to compare with."),
    ] = None,
    results_dir: Annotated[
        Path, typer.Option("--results", help="Parent of the per-run directories.")
    ] = Path("results"),
    probe: Annotated[
        bool, typer.Option("--probe/--compare", help="Plan the probe, or diff two stores.")
    ] = True,
    n: Annotated[int, typer.Option("--n", help="Probe size in cells.")] = 100,
    seed: Annotated[int, typer.Option("--seed")] = 20261012,
) -> HardwareDeltaResult:
    """Bound the cross-hardware confound by measuring it."""
    from msat.hardware.delta import compare_stores, confound_verdict, probe_plan
    from msat.store import ResponseStore

    resolved = RunIdentitySeam.resolve(run_id, run)
    against_resolved = RunIdentitySeam.resolve_comparison(against)
    _assert_offline("hardware-delta")
    loc = OutputLocation.directory(results_dir, default="results")
    run_dir = loc.path / resolved
    if not run_dir.is_dir() and (Path("results") / resolved).is_dir():
        run_dir = Path("results") / resolved
    elif not run_dir.is_dir() and (Path("work") / resolved).is_dir():
        run_dir = Path("work") / resolved

    if not run_dir.is_dir():
        _fail(f"no run directory at {run_dir}.")
    store = ResponseStore(run_dir / "responses")

    if probe and against_resolved is None:
        plan = probe_plan(store, n=n, seed=seed)
        res = HardwareDeltaResult(
            run_id=resolved,
            against=None,
            is_probe=True,
            plan=plan,
        )
        return renderer.render(res)

    if against_resolved is None:
        _fail("--compare requires --against <run_id>")

    other_dir = loc.path / against_resolved
    if not other_dir.is_dir() and (Path("results") / against_resolved).is_dir():
        other_dir = Path("results") / against_resolved
    elif not other_dir.is_dir() and (Path("work") / against_resolved).is_dir():
        other_dir = Path("work") / against_resolved

    if not other_dir.is_dir():
        _fail(f"no comparison run directory at {other_dir}.")
    other_store = ResponseStore(other_dir / "responses")

    keys = sorted(r.key for r in store.iter_all())
    gold = _gold_by_key(loc.path, resolved)
    report = compare_stores(
        store,
        other_store,
        keys,
        gold_by_key=gold,
        norm=_load_normaliser(),
        label_a=resolved,
        label_b=against_resolved,
    )
    mde = _interaction_mde(loc.path)
    verdict = confound_verdict(
        report,
        mde_log_odds=mde.get("mde_log_odds"),
        mde_pp_at_p50=mde.get("mde_pp_at_p50"),
    )
    res = HardwareDeltaResult(
        run_id=resolved,
        against=against_resolved,
        is_probe=False,
        verdict=verdict,
    )
    return renderer.render(res)


def _load_normaliser(config_root: Path = Path("configs")) -> Normaliser:
    from msat.metrics.scoring_workflow import load_normaliser

    return load_normaliser(config_dir=config_root)


def _interaction_mde(results_dir: Path) -> dict[str, Any]:
    path = results_dir / "power.json"
    if not path.exists():
        return {}
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    interaction = raw.get("interaction", {})
    return interaction if isinstance(interaction, dict) else {}


def _gold_by_key(results_dir: Path, run_id: str) -> dict[str, str]:
    path = results_dir / run_id / "gold.json"
    if not path.exists():
        _fail(
            f"{path} is missing. The delta is scored, not eyeballed: without a "
            f"key->gold map the harness can only count text differences, which says "
            f"nothing about whether the RESULTS moved."
        )
    return {str(k): str(v) for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


@app.command("vram")
def vram(
    model: Annotated[Path, typer.Argument(help="Local checkpoint directory (no network access).")],
    tokens: Annotated[int, typer.Option("--tokens", help="Context length to plan for.")] = 4096,
    quant: Annotated[
        str | None, typer.Option("--quant", help="Weight bits: 4, 8, 16, 32. Default: dtype.")
    ] = None,
    card: Annotated[
        float | None, typer.Option("--card", help="Also report the ceiling for this many GB.")
    ] = None,
) -> VramResult:
    """Compute a checkpoint's VRAM requirement from its `config.json`."""
    from msat.hardware.vram import (
        ConfigMissingError,
        kv_reader_capacity_tokens,
        load_config,
        plan_vram,
        shape_from_config,
    )

    try:
        shape = shape_from_config(load_config(Path(model)), name=Path(model).name)
    except ConfigMissingError as e:
        _fail(str(e))

    quant_bits = float(quant) if quant else None
    plan = plan_vram(shape, n_tokens=tokens, quant_bits=quant_bits)

    ceiling_card = None
    if card:
        ceiling_card = kv_reader_capacity_tokens(shape, card, quant_bits=quant_bits)

    res = VramResult(
        shape=shape,
        plan=plan,
        tokens=tokens,
        card_reader_capacity=ceiling_card,
    )
    return renderer.render(res)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"msat {__version__}")
        raise typer.Exit()


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option("--version", "-V", callback=_version_callback, is_eager=True),
    ] = False,
) -> None:
    configure_logging()


def main(argv: list[str] | None = None) -> Any:
    configure_logging()
    if argv is not None:
        return app(args=argv, standalone_mode=False)
    return app()


__all__ = [
    "OFFLINE_COMMANDS",
    "ONLINE_COMMANDS",
    "PRD_SURFACE",
    "AnalyzeResult",
    "CeilingResult",
    "CliRenderer",
    "DataLockResult",
    "FiguresResult",
    "HardwareDeltaResult",
    "InputSeam",
    "JudgeAuditInputs",
    "JudgeAuditResult",
    "MaxContextResult",
    "OutputLocation",
    "PaperResult",
    "PlanResult",
    "PowerResult",
    "RunIdentitySeam",
    "RunResult",
    "ScoreInputs",
    "ScoreResult",
    "StatusResult",
    "ValidateConfigResult",
    "VramResult",
    "app",
    "capacity_preflight",
    "check_capacity",
    "crossing",
    "evaluate_gate",
    "gate_exit",
    "get_last_result",
    "load_gold_sessions",
    "main",
    "main_callback",
    "offline_commands",
    "panel",
    "plan_from",
    "registered_commands",
    "renderer",
    "resolve_profile_bundle",
]
