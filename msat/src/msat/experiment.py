"""Deep effective experiment design module.

Provides unified mode resolution, dynamic primary reader selection, filtered
effective bundle and panel synchronization, and manifest validation before execution.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from msat.config import ConfigBundle, compute_config_hash
from msat.manifest import Manifest
from msat.providers.local import LocalReader, readers_from_config


class ExecutionProfile(StrEnum):
    """Supported experiment execution profiles."""

    FULL = "full"
    FAST = "fast"

    @classmethod
    def from_str(cls, val: str | ExecutionProfile | None) -> ExecutionProfile:
        if val is None:
            return cls.FULL
        if isinstance(val, cls):
            return val
        s = str(val).lower().strip()
        if s == "full":
            return cls.FULL
        if s == "fast":
            return cls.FAST
        raise InvalidProfileError(f"profile must be 'fast' or 'full', got {val!r}")


class InvalidProfileError(ValueError):
    """The execution profile is not recognised."""


InvalidModeError = InvalidProfileError
ExecutionMode = ExecutionProfile


class IncompatibleManifestError(ValueError):
    """The manifest is incompatible with the effective experiment configuration."""


@dataclass(frozen=True, slots=True)
class EffectiveExperiment:
    """The resolved effective experiment bundle and reader panel."""

    profile: str
    bundle: ConfigBundle
    canonical_config_hash: str
    effective_config_hash: str
    preregistration_hash: str
    primary_reader: str
    panel: tuple[LocalReader, ...]

    @property
    def mode(self) -> str:
        return self.profile

    @property
    def is_fast(self) -> bool:
        return self.profile == "fast"

    def summary(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "mode": self.profile,
            "canonical_config_hash": self.canonical_config_hash,
            "effective_config_hash": self.effective_config_hash,
            "preregistration_hash": self.preregistration_hash,
            "primary_reader": self.primary_reader,
            "scales": list(self.bundle.design.scales),
            "readers": list(self.bundle.design.readers),
            "repetition": self.bundle.design.repetition,
            "runs_per_cell": self.bundle.design.runs_per_cell,
            "panel": [r.id for r in self.panel],
        }


def resolve_effective_experiment(
    bundle: ConfigBundle,
    profile: str | ExecutionProfile | None = None,
    *,
    mode: str | ExecutionProfile | None = None,
    card_gb: float = 8.0,
) -> EffectiveExperiment:
    """Resolve an execution profile into a unified effective experiment.

    - Validates profile: invalid profiles fail with InvalidProfileError.
    - Dynamically selects declared primary reader (role == 'primary') rather than hardcoding.
    - Full profile validates canonical reader requirement (>= 2 families).
    - Fast profile produces distinct reproducible effective_config_hash while preserving
      canonical_config_hash and preregistration_hash.
    - In fast profile, filtered effective design, raw configs, and reader panel all agree.
    """
    target = profile if profile is not None else (mode if mode is not None else "full")
    exec_profile = ExecutionProfile.from_str(target)

    primaries = [f for f in bundle.readers.families if f.role == "primary"]
    if len(primaries) != 1:
        raise ValueError(
            f"Expected exactly one reader family with role 'primary', found {len(primaries)}: {[f.id for f in primaries]}"
        )
    primary_family = primaries[0]
    primary_reader_id = primary_family.id

    canonical_config_hash = bundle.canonical_config_hash or compute_config_hash(bundle.raw)
    preregistration_hash = getattr(bundle, "preregistration_hash", "")

    if bundle.profile == "fast" if hasattr(bundle, "profile") else bundle.mode == "fast":
        if exec_profile != ExecutionProfile.FAST:
            raise InvalidProfileError(
                "Load the canonical bundle before selecting the full profile."
            )
        return EffectiveExperiment(
            profile="fast",
            bundle=bundle,
            canonical_config_hash=canonical_config_hash,
            effective_config_hash=bundle.config_hash,
            preregistration_hash=preregistration_hash,
            primary_reader=primary_reader_id,
            panel=tuple(
                readers_from_config(bundle.raw["readers.yaml"], card_gb=card_gb, allow_single=True)
            ),
        )
    readers_from_config(bundle.raw["readers.yaml"], card_gb=card_gb, allow_single=False)

    if exec_profile == ExecutionProfile.FULL:
        # Canonical validation: >= 2 reader families required (PRD R7.5)
        raw_readers = bundle.raw.get("readers.yaml", {})
        panel_list = readers_from_config(raw_readers, card_gb=card_gb, allow_single=False)
        selected = set(bundle.design.readers)
        missing = selected - {r.id for r in panel_list}
        if missing:
            raise ValueError(f"Declared readers are missing from the registry: {sorted(missing)}")
        panel_list = [r for r in panel_list if r.id in selected]
        full_bundle = bundle.model_copy(
            update={
                "config_hash": canonical_config_hash,
                "canonical_config_hash": canonical_config_hash,
                "effective_config_hash": canonical_config_hash,
                "mode": "full",
            }
        )

        return EffectiveExperiment(
            profile=ExecutionProfile.FULL.value,
            bundle=full_bundle,
            canonical_config_hash=canonical_config_hash,
            effective_config_hash=canonical_config_hash,
            preregistration_hash=preregistration_hash,
            primary_reader=primary_reader_id,
            panel=tuple(panel_list),
        )

    # Fast pilot tier: Scale S, primary reader, 1 run/cell
    fast_design = bundle.design.model_copy(
        update={
            "scales": ["S"],
            "readers": [primary_reader_id],
            "repetition": 1,
        }
    )
    fast_readers = bundle.readers.model_copy(update={"families": [primary_family]})

    # Synchronize raw configurations so panel, hashing, and design agree
    effective_raw = copy.deepcopy(bundle.raw)
    raw_design = dict(effective_raw.get("design.yaml", {}))
    raw_design["scales"] = ["S"]
    raw_design["readers"] = [primary_reader_id]
    raw_design["runs_per_cell"] = 1
    raw_design["repetition"] = 1
    effective_raw["design.yaml"] = raw_design

    raw_readers = dict(effective_raw.get("readers.yaml", {}))
    raw_readers["families"] = [
        f for f in raw_readers.get("families", []) if f.get("id") == primary_reader_id
    ]
    effective_raw["readers.yaml"] = raw_readers

    effective_config_hash = compute_config_hash(effective_raw)

    effective_bundle = bundle.model_copy(
        update={
            "design": fast_design,
            "readers": fast_readers,
            "raw": effective_raw,
            "config_hash": effective_config_hash,
            "canonical_config_hash": canonical_config_hash,
            "effective_config_hash": effective_config_hash,
            "mode": "fast",
        }
    )

    # Single-reader pilot panel
    fast_panel = readers_from_config(
        effective_raw["readers.yaml"], card_gb=card_gb, allow_single=True
    )

    # Verify complete agreement
    assert list(fast_design.readers) == [primary_reader_id]
    assert [f.id for f in fast_readers.families] == [primary_reader_id]
    assert [f["id"] for f in raw_readers["families"]] == [primary_reader_id]
    assert [r.id for r in fast_panel] == [primary_reader_id]

    return EffectiveExperiment(
        profile=ExecutionProfile.FAST.value,
        bundle=effective_bundle,
        canonical_config_hash=canonical_config_hash,
        effective_config_hash=effective_config_hash,
        preregistration_hash=preregistration_hash,
        primary_reader=primary_reader_id,
        panel=tuple(fast_panel),
    )


def validate_manifest_for_execution(
    manifest_or_path: Manifest | Path | str | Mapping[str, Any],
    experiment: EffectiveExperiment | ConfigBundle,
    *,
    qids_by_scale: Mapping[str, Sequence[str]] | None = None,
    seed: int | None = None,
    runs_per_cell: int | None = None,
    repetition: int | None = None,
    strict_questions: bool = True,
) -> dict[str, Any]:
    """Validate that a manifest is compatible with an experiment before execution.

    Verifies mode, config_hash attribution, seed, runs_per_cell, and question availability.
    Fails explicitly with IncompatibleManifestError if validation fails.
    """
    if isinstance(manifest_or_path, (str, Path)):
        p = Path(manifest_or_path)
        if not p.is_file():
            raise FileNotFoundError(f"Manifest not found: {p}")
        raw = json.loads(p.read_text(encoding="utf-8"))
    elif isinstance(manifest_or_path, Manifest):
        raw = manifest_or_path.as_dict()
    elif isinstance(manifest_or_path, Mapping):
        raw = dict(manifest_or_path)
    else:
        raise TypeError(f"Unsupported manifest type: {type(manifest_or_path)}")

    if isinstance(experiment, EffectiveExperiment):
        exp_mode = experiment.mode
        exp_eff_hash = experiment.effective_config_hash
        exp_can_hash = experiment.canonical_config_hash
        exp_design = experiment.bundle.design
    else:
        exp_mode = experiment.mode
        exp_eff_hash = experiment.effective_config_hash or experiment.config_hash
        exp_can_hash = experiment.canonical_config_hash or experiment.config_hash
        exp_design = experiment.design

    manifest_hash = str(raw.get("config_hash", ""))
    manifest_effective = str(raw.get("effective_config_hash", manifest_hash))
    manifest_mode = str(raw.get("mode", "full"))

    problems: list[str] = []

    # 1. Mode check
    if manifest_mode != exp_mode:
        problems.append(
            f"Manifest execution mode {manifest_mode!r} does not match experiment mode {exp_mode!r}"
        )

    # 2. Config hash check (accepts either effective match or canonical match)
    if manifest_effective != exp_eff_hash or manifest_hash != exp_eff_hash:
        problems.append(
            f"Manifest config_hash {manifest_hash!r} (effective {manifest_effective!r}) "
            f"does not match experiment hash {exp_eff_hash!r}"
        )

    # 3. Grid metadata checks
    grid = raw.get("grid", {})
    expected_seed = int(exp_design.seed) if seed is None else int(seed)
    expected_repeats = (
        int(getattr(exp_design, "repetition", exp_design.runs_per_cell))
        if (repetition is None and runs_per_cell is None)
        else int(repetition if repetition is not None else runs_per_cell)  # type: ignore[arg-type]
    )
    metadata = raw.get("metadata", {})
    declared_seed = grid.get("seed", metadata.get("seed"))
    if declared_seed is not None and int(declared_seed) != expected_seed:
        problems.append(
            f"Manifest seed ({declared_seed}) does not match requested seed ({expected_seed})"
        )

    declared_repeats = grid.get(
        "repetition",
        grid.get("runs_per_cell", metadata.get("repetition", metadata.get("runs_per_cell"))),
    )
    if declared_repeats is not None and int(declared_repeats) != expected_repeats:
        problems.append(
            f"Manifest runs_per_cell ({declared_repeats}) does not match requested ({expected_repeats})"
        )
    if raw.get("canonical_config_hash", exp_can_hash) != exp_can_hash:
        problems.append("Manifest canonical_config_hash does not match the canonical design.")

    # 4. Filtered condition alignment
    if "cells" in raw and "conditions" in raw:
        cells = raw["cells"] if len(raw["cells"]) <= len(raw["conditions"]) else raw["conditions"]
    else:
        cells = raw.get("conditions") or raw.get("cells", [])
    expected_coordinates = {
        (int(k), round(float(b.value), 6), str(b.mode), scale, reader)
        for k in exp_design.K_grid
        for b in exp_design.B_grid
        for scale in exp_design.scales
        for reader in exp_design.readers
    }
    seen_coordinates: set[tuple[Any, ...]] = set()
    for cell in cells:
        coordinate = (
            int(cell["K"]),
            round(float(cell["B"]), 6),
            str(cell["B_mode"]),
            str(cell["scale"]),
            str(cell["reader"]),
        )
        if coordinate in seen_coordinates:
            problems.append(f"Manifest contains a duplicate condition coordinate: {coordinate!r}")
        seen_coordinates.add(coordinate)
        if coordinate not in expected_coordinates:
            problems.append(f"Manifest contains an undeclared condition coordinate: {coordinate!r}")
        if cell.get("arm") not in ("A+G", "A", "G"):
            problems.append(f"Manifest contains an undeclared crossed arm: {cell.get('arm')!r}")
        if cell.get("status") not in ("accepted", "rejected", "supported", "unsupported"):
            problems.append(
                "Manifest condition status must be accepted, rejected, supported, or unsupported."
            )
    missing_coordinates = expected_coordinates - seen_coordinates
    if missing_coordinates:
        problems.append(
            f"Manifest is missing {len(missing_coordinates)} declared condition coordinate(s)."
        )
    accepted = [c for c in cells if c.get("status") in ("accepted", "supported")]
    all_scales = {c.get("scale") for c in cells} or set(grid.get("scales", []))
    all_readers = {c.get("reader") for c in cells} or set(grid.get("readers", []))

    unexpected_scales = all_scales - set(exp_design.scales)
    if unexpected_scales:
        problems.append(
            f"Manifest contains cells with unexpected scale(s) {sorted(unexpected_scales)}; "
            f"declared scales are {list(exp_design.scales)}"
        )

    unexpected_readers = all_readers - set(exp_design.readers)
    if unexpected_readers:
        problems.append(
            f"Manifest contains cells with unexpected reader(s) {sorted(unexpected_readers)}; "
            f"declared readers are {list(exp_design.readers)}"
        )

    # 5. Strict questions check
    if strict_questions and qids_by_scale is None:
        problems.append("Runtime planning requires explicit question IDs by scale.")
    if strict_questions and qids_by_scale is not None:
        target_scales = {c.get("scale") for c in accepted} or all_scales
        for sc in target_scales:
            qs = qids_by_scale.get(sc, [])
            if not qs:
                problems.append(f"Scale {sc!r} has cells in manifest but no question IDs supplied")
            if len(qs) != len(set(qs)):
                problems.append(f"Scale {sc!r} contains duplicate question IDs.")
            declared_qids = metadata.get("qids_by_scale", {}).get(sc)
            if declared_qids is not None and set(qs) != set(declared_qids):
                problems.append(
                    f"Scale {sc!r} question IDs do not match the manifest question set."
                )
            for cell in accepted:
                if cell.get("scale") == sc and int(cell.get("n_questions", 0)) != len(qs):
                    problems.append(
                        f"Scale {sc!r} question count does not match its planned conditions."
                    )
                    break

    if problems:
        raise IncompatibleManifestError("; ".join(problems))

    return {
        "valid": True,
        "mode": exp_mode,
        "n_accepted_cells": len(accepted),
        "manifest_hash": manifest_hash,
        "effective_hash": exp_eff_hash,
    }
