"""Validated run preparation and durable ladder dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from msat.arms.registry import build_registry
from msat.config import ConfigBundle
from msat.experiment import validate_manifest_for_execution
from msat.planning import plan_from_manifest
from msat.runner.input import MissingInputError, RunInputs
from msat.runner.ladder import RungName, check_order
from msat.runner.lifecycle import (
    CapacityRefusalError,
    GateFailedError,
    RungReport,
    RunLifecycle,
    arm_e_thresholds,
    capacity_preflight,
    crossing,
    evaluate_gate,
    read_ladder,
)
from msat.runner.metadata import write_json_artifact

if TYPE_CHECKING:
    from msat.memory.embed import Embedder
    from msat.providers.local import LocalReader


def run_lifecycle_workflow(
    bundle: Any,
    run_dir: Path | str,
    *,
    rung: RungName = "negative",
    run_id: str = "run",
    inputs: RunInputs | None = None,
    questions_path: Path | str | None = None,
    gold_path: Path | str | None = None,
    pool_path: Path | str | None = None,
    filler_path: Path | str | None = None,
    data_dir: Path | str | None = None,
    plan_path: Path | str | None = None,
    reader_id: str | None = None,
    panel: Sequence[LocalReader] | None = None,
    resume: bool = False,
    dry_run: bool = False,
    is_synthetic: bool = False,
    seed: int | None = None,
    embedder: Embedder | None = None,
) -> RungReport | dict[str, Any]:
    """Validate the selected design, prepare inputs, and execute one durable rung."""
    destination = Path(run_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if panel is None:
        from msat.providers.local import readers_from_config

        panel = readers_from_config(
            bundle.raw["readers.yaml"], allow_single=getattr(bundle, "mode", "full") == "fast"
        )
        panel = [reader for reader in panel if reader.id in bundle.design.readers]
    declared_panel = list(panel)
    chosen_panel = [reader for reader in panel if reader_id is None or reader.id == reader_id]
    if not chosen_panel:
        raise ValueError(f"unknown reader {reader_id!r}")
    registry, registry_report = build_registry(bundle)
    state = read_ladder(destination / "ladder.json")
    try:
        check_order(state, rung)
    except ValueError:
        raise
    except RuntimeError as exc:
        evaluate_gate(
            destination,
            context=f"{run_id}/ladder",
            run_id=run_id,
            config_hash=bundle.config_hash,
            notes=[f"Ladder order violation; pre-run refusal without a measured result: {exc}"],
        )
        raise GateFailedError(str(exc)) from exc

    if not dry_run and arm_e_thresholds(bundle) is None:
        evaluate_gate(
            destination,
            context=f"{run_id}/arm_e_thresholds",
            run_id=run_id,
            config_hash=bundle.config_hash,
            notes=["Arm E pass thresholds are null; the negative control cannot be evaluated."],
        )
        raise GateFailedError("Arm E pass thresholds are null; no rung may proceed.")

    if inputs is None:
        if dry_run:
            inputs = RunInputs()
        else:
            if data_dir is None:
                missing = [
                    flag
                    for flag, value in (
                        ("--questions", questions_path),
                        ("--gold", gold_path),
                        ("--arm-e-pool", pool_path),
                    )
                    if value is None
                ]
                if missing:
                    raise MissingInputError(
                        f"{', '.join(missing)} is missing input; supply the control inputs or --data with benchmark files."
                    )
            inputs = RunInputs.from_legacy_paths(
                questions_path=questions_path,
                gold_path=gold_path,
                pool_path=pool_path,
                filler_path=filler_path,
                data_dir=data_dir,
                scales=bundle.design.scales,
                is_synthetic=is_synthetic,
            )
    resolved_seed = int(seed if seed is not None else getattr(bundle.design, "seed", 20261012))
    plan = None
    if plan_path is not None:
        path = Path(plan_path)
        if not path.is_file():
            raise MissingInputError(f"No manifest at {path}; run `msat build-manifest` first.")
        plan = plan_from_manifest(
            path,
            inputs.qids_by_scale(),
            seed=resolved_seed,
            runs_per_cell=bundle.design.runs_per_cell,
        )
        grid, interaction_cells, degenerate = crossing(plan, bundle)
        write_json_artifact(destination / "crossed_grid.json", grid)
        if not grid["crossed"]:
            evaluate_gate(
                destination,
                context=f"{run_id}/preflight",
                run_id=run_id,
                config_hash=bundle.config_hash,
                degenerate_cells=degenerate,
                n_interaction_cells=0,
                notes=grid["problems"],
            )
            raise GateFailedError("The declared grid is not a crossed grid.")
        if isinstance(bundle, ConfigBundle):
            validate_manifest_for_execution(
                path,
                bundle,
                seed=resolved_seed,
                runs_per_cell=bundle.design.runs_per_cell,
                qids_by_scale=inputs.qids_by_scale(),
                strict_questions=rung in ("crossed", "full") and not dry_run,
            )
    else:
        grid, interaction_cells, degenerate = {}, 0, []
        if rung == "crossed":
            raise MissingInputError("An executable manifest is required for the crossed rung.")

    capacity = (
        capacity_preflight(bundle, plan, declared_panel)
        if plan
        else {"n_checked": 0, "n_unsupported": 0, "all_fit": True}
    )
    write_json_artifact(destination / "capacity.json", capacity)
    gate = evaluate_gate(
        destination,
        context=f"{run_id}/preflight",
        run_id=run_id,
        config_hash=bundle.config_hash,
        degenerate_cells=degenerate,
        n_interaction_cells=interaction_cells,
        notes=["Pre-run design check; no measured accuracy or recall is claimed."],
    )
    if plan is not None and not gate.proceed:
        raise GateFailedError("The design failed its pre-run gate.")
    if dry_run:
        return {
            "dry_run": True,
            "rung": rung,
            "registry": registry_report,
            "capacity": capacity,
            "grid": grid,
            "ladder_state": state.as_dict(),
            "gate": gate.as_dict(),
        }
    if rung == "crossed" and capacity["n_unsupported"]:
        raise CapacityRefusalError(
            f"{capacity['n_unsupported']} declared conditions exceed reader capacity before dispatch."
        )
    if rung == "crossed" and {reader.id for reader in chosen_panel} != set(bundle.design.readers):
        raise ValueError(
            "A crossed grid requires its complete declared reader panel; use the fast profile for a primary-reader pilot."
        )
    lifecycle = RunLifecycle(
        run_dir=destination,
        bundle=bundle,
        registry=registry,
        panel=chosen_panel,
        inputs=inputs,
        run_id=run_id,
        plan=plan,
        resume=resume,
        seed=resolved_seed,
        embedder=embedder,
    )
    return lifecycle.run_rung(rung)
