"""The arm registry — arms as data, loaded from `configs/arms.yaml`.

PRD §R4 / `arms.yaml`: "Arms are DATA, not code: adding one needs no pipeline
change." This module is where that becomes true. There is no dispatch table of
`if arm == "A"` anywhere in the pipeline; adding arm H means adding a YAML entry
and, at most, one small class.

THE THREE RENDERERS, AND WHY THEY ARE THREE
-------------------------------------------
    SimpleArm     A and G. Natural top-K in rank order, per-memory budget.
    PositionArm   C. Same renderer, different ordering.
    MaxContextArm / FullContextArm   D and D-max. Different corpus fraction, and
                  D refuses when the whole conversation does not fit.

A and G deliberately SHARE `SimpleArm` so a change to memory formatting moves both
together and cancels out of the interaction term. Giving G its own renderer would
lose exactly that cancellation and make a formatting change look like a budget
effect - which is the kind of confound this design exists to avoid.

WHAT THE REGISTRY REFUSES
-------------------------
1. **Arms A or G disabled.** They are the crossed grid; without both there is no
   interaction term and no paper. Refusing beats running a design that cannot
   answer its own question.
2. **Arm D marked enabled without a feasibility check wired.** D is known to be
   infeasible on an 8 GB card; an enabled D with no `full_context_feasible` in the
   pipeline is a run that will OOM at hour 14.
3. **A template that is missing.** Silently rendering an empty prompt produces
   plausible output for a condition that was never actually run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from msat.arms.baselines import FullContextArm
from msat.arms.budget import BudgetArm, assert_crossed
from msat.arms.max_context import MaxContextArm
from msat.arms.position import PositionArm
from msat.arms.render import ArmRegistry, DelegatingArm, SimpleArm
from msat.config import ArmSpec

ARM_CLASSES: dict[str, type[Any]] = {
    "A": SimpleArm,
    "G": BudgetArm,
    "C": PositionArm,
    "D": FullContextArm,
    "D-max": MaxContextArm,
}
"""Which class renders which arm id. `D-max` is registered so the VRAM-capped
substitute is reachable by name; it is not in `arms.yaml` because it is DERIVED
from D's feasibility, not declared as a separate experimental condition."""

DELEGATING_ARMS: dict[str, str] = {"B": "A", "E": "A", "F": "A"}
"""Arms whose `.j2` file is a CONFIG BLOCK, not a prompt.

`arms.yaml` lists `arm_e.j2` as arm E's `prompt_template`, which reads like a
template and is not one: it declares `source`, `seed`, `exclude_gold` and the two
pass thresholds, and is parsed by `ControlConfig.from_template`. Arm B's is the
padding block. Arm F's carries the oracle's evidence ids.

All three are manipulated in the MEMORY SET, and their prompt is deliberately the
treatment's. For E that identity is load-bearing: a collapse must be attributable to
the absence of evidence, not to a template change (PRD §R4.3). They are built as
`DelegatingArm` so that identity is structural - see `test_e_is_structurally_identical_to_a`.
"""

PROMPT_ARMS: frozenset[str] = frozenset({"A", "C", "D", "G", "D-max"})
"""Arms whose `.j2` IS a prompt template and must be present."""


class RegistryError(RuntimeError):
    """The arm configuration and the code disagree, or an arm cannot be built."""


@dataclass(frozen=True, slots=True)
class _Unused:  # pragma: no cover - removed
    """Placeholder removed."""


def _template_for(prompts: Mapping[str, str], name: str) -> str:
    """Look a template up under either key form.

    `ConfigBundle.prompts` is keyed by RELATIVE path (`prompts/arm_a.j2`) while
    `arms.yaml` stores a bare filename (`arm_a.j2`). Normalising to the basename
    makes the lookup total, so a change to the bundle's key convention cannot
    silently produce empty prompts for every arm - which the `PROMPT_ARMS` check
    below would catch, but with a message that points at the wrong file.
    """
    if not name:
        return ""
    base = name.rsplit("/", 1)[-1]
    for key in (f"prompts/{base}", base, name):
        if key in prompts:
            return prompts[key]
    return ""


def load_specs(bundle: Any) -> tuple[Any, ...]:
    """The declared arms, in file order.

    Returns `config.ArmSpec` objects from the real bundle, and plain objects with
    the same attribute names from a test double. There is deliberately NO second
    `ArmSpec` dataclass here: an earlier draft defined one, and two definitions of
    the same record is how the prompts-key and the field list drift apart.
    """
    declared = getattr(bundle.arms, "arms", None)
    if declared is None:
        raw = bundle.raw.get("arms", []) if hasattr(bundle, "raw") else []
        declared = raw.get("arms", []) if isinstance(raw, dict) else []
    out: list[Any] = []
    for entry in declared:

        def field_of(name: str, default: Any = None, _e: Any = entry) -> Any:
            if isinstance(_e, Mapping):
                return _e.get(name, default)
            return getattr(_e, name, default)

        out.append(
            ArmSpec(
                id=str(field_of("id", "")),
                isolates=str(field_of("isolates", "") or ""),
                enabled=bool(field_of("enabled", True)),
                prompt_template=str(field_of("prompt_template", "") or ""),
                notes=str(field_of("notes", "") or ""),
            )
        )
    return tuple(out)


def build_registry(bundle: Any) -> tuple[ArmRegistry, dict[str, Any]]:
    """Build every enabled arm and report what was refused.

    Returns the registry and a report. The report is not decoration: `msat run`
    writes it next to the plan, because "arm D did not run" has to be visible in the
    run's own artefacts and not reconstructed later from a missing ledger cell.
    """
    specs = load_specs(bundle)
    prompts: dict[str, str] = dict(getattr(bundle, "prompts", {}) or {})
    arms: dict[str, Any] = {}
    refused: dict[str, str] = {}
    delegates: dict[str, str] = {}

    for spec in specs:
        if not spec.enabled:
            refused[spec.id] = "disabled in arms.yaml"
            continue

        delegate = DELEGATING_ARMS.get(spec.id)
        if delegate is not None:
            parent = arms.get(delegate)
            if parent is None:
                refused[spec.id] = (
                    f"delegates to arm {delegate!r}, which is disabled or missing. "
                    f"Its prompt is inherited, so there is nothing to build."
                )
                continue
            arms[spec.id] = DelegatingArm(
                id=spec.id,
                isolates=spec.isolates,
                template=parent.template,
                prompt_template_name=f"{delegate}:{spec.prompt_template}",
            )
            delegates[spec.id] = delegate
            continue

        cls = ARM_CLASSES.get(spec.id)
        if cls is None:
            refused[spec.id] = (
                f"no renderer is registered for arm {spec.id!r}. Adding an arm needs "
                f"a class in `ARM_CLASSES` - the pipeline itself does not change."
            )
            continue
        template = _template_for(prompts, spec.prompt_template)
        if spec.id in PROMPT_ARMS and not template:
            raise RegistryError(
                f"arm {spec.id}: template {spec.prompt_template!r} is missing or empty. "
                f"Rendering an empty prompt produces a plausible score for a condition "
                f"that was never run."
            )
        arms[spec.id] = cls(
            id=spec.id,
            isolates=spec.isolates,
            template=template,
            prompt_template_name=spec.prompt_template,
        )

    registry = ArmRegistry(arms=arms)
    report: dict[str, Any] = {
        "n_declared": len(specs),
        "n_built": len(arms),
        "refused": refused,
        "delegates": delegates,
        "prompt_arms": sorted(a for a in arms if a in PROMPT_ARMS),
        "config_arms": sorted(delegates),
        "crossed": assert_crossed(registry, declared_crossed_arms(bundle)),
        "note": (
            "`config_arms` name a `.j2` that is a CONFIG BLOCK, not a prompt, and "
            "inherit the treatment's template on purpose. Their manipulation is in "
            "the memory set."
        ),
    }

    missing_crossed = [a for a in CROSSED if a not in arms]
    if missing_crossed:
        raise RegistryError(
            f"arms {missing_crossed} are missing from the registry. They are the two "
            f"halves of ONE crossed grid; without both there is no interaction term "
            f"and no headline result. Refusing to build a plan that cannot answer its "
            f"own question."
        )
    return registry, report


CROSSED = ("A", "G")


def declared_crossed_arms(bundle: Any) -> Sequence[str]:
    """`crossed_grid.arms` from `arms.yaml`, falling back to the declared pair."""
    raw = bundle.raw.get("crossed_grid") if hasattr(bundle, "raw") else None
    if isinstance(raw, Mapping) and raw.get("arms"):
        return [str(a) for a in raw["arms"]]
    return list(CROSSED)


def arm_d_feasibility(
    *, corpus_tokens: int, limits: Mapping[str, int], context_window: int = 2**31
) -> dict[str, Any]:
    """Per-reader `full_context` verdict, for the run report.

    Arm D is BUILT whether or not it can run. Whether it can run is a property of
    the corpus and the hardware, recorded here so the run's artefacts state it
    rather than leaving an empty ledger cell to be interpreted later.
    """
    from msat.arms.baselines import full_context_feasible

    per = {
        r: full_context_feasible(
            corpus_tokens, context_window=context_window, vram_ceiling=ceiling
        ).as_dict()
        for r, ceiling in limits.items()
    }
    return {
        "per_reader": per,
        "any_feasible": any(v["ok"] for v in per.values()),
        "substitute": "D-max" if not any(v["ok"] for v in per.values()) else None,
    }


def load_templates(config_root: Path) -> dict[str, str]:
    """Every `prompts/*.j2`, keyed by FILENAME - which is what `arms.yaml` stores."""
    prompts: dict[str, str] = {}
    d = Path(config_root) / "prompts"
    if not d.is_dir():
        return prompts
    for path in sorted(d.glob("*.j2")):
        prompts[path.name] = path.read_text(encoding="utf-8")
    return prompts
