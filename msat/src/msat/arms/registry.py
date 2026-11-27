"""The arm registry: arms as data, loaded from `configs/arms.yaml`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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

DELEGATING_ARMS: dict[str, str] = {"B": "A", "E": "A", "F": "A"}

PROMPT_ARMS: frozenset[str] = frozenset({"A", "C", "D", "G", "D-max"})


class RegistryError(RuntimeError):
    """Raised when the arm configuration and the code disagree."""


def _template_for(prompts: Mapping[str, str], name: str) -> str:
    if not name:
        return ""
    base = name.rsplit("/", 1)[-1]
    for key in (f"prompts/{base}", base, name):
        if key in prompts:
            return prompts[key]
    return ""


def load_specs(bundle: Any) -> tuple[Any, ...]:
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
    raw = bundle.raw.get("crossed_grid") if hasattr(bundle, "raw") else None
    if isinstance(raw, Mapping) and raw.get("arms"):
        return [str(a) for a in raw["arms"]]
    return list(CROSSED)


def arm_d_feasibility(
    *, corpus_tokens: int, limits: Mapping[str, int], context_window: int = 2**31
) -> dict[str, Any]:
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
    prompts: dict[str, str] = {}
    d = Path(config_root) / "prompts"
    if not d.is_dir():
        return prompts
    for path in sorted(d.glob("*.j2")):
        prompts[path.name] = path.read_text(encoding="utf-8")
    return prompts
