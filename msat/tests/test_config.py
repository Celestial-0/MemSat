"""`validate-config` and the config bundle (PRD §R1, §R7.5).

Every test here mutates a COPY of the real config bundle and asserts the
validator catches the specific defect. A validator that has only ever seen a good
bundle is untested.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from msat.config import (
    BUNDLE_FILES,
    Severity,
    compute_config_hash,
    errors_only,
    load_bundle,
    validate,
)

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def bundle_copy(complete_configs: Path, tmp_path: Path) -> Path:
    """A writable copy of a COMPLETE bundle.

    Built from `complete_configs` (the fixture that fills in the reader
    checkpoints) rather than from `configs/` directly, so a test about, say, the
    null-field rule does not fail because the checkpoints are still UNCONFIRMED.
    Nested under its own directory because `complete_configs` also receives the
    same `tmp_path` and already claims `tmp_path/configs`.
    """
    dst = tmp_path / "bundle_under_test" / "configs"
    shutil.copytree(complete_configs, dst)
    return dst


def _codes(root: Path) -> set[str]:
    return {p.code for p in validate(root)}


def _edit(root: Path, name: str, mutate: Any) -> None:
    path = root / name
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ── the shipped bundle: deliberately incomplete ────────────────────────────────
def test_shipped_config_has_no_unexpected_errors(complete_configs: Path) -> None:
    """The shipped bundle differs from a completed one ONLY in the checkpoints.

    Run against `complete_configs` so this asserts the real thing — that the
    committed YAML is otherwise sound — rather than being permanently red.
    """
    problems = errors_only(validate(complete_configs))
    assert problems == [], "\n".join(str(p) for p in problems)


def test_shipped_config_has_no_unconfirmed_checkpoints() -> None:
    """PRD §R7.6: an unnamed reader makes every number unattributable.

    INVERTED 2026-10-13. This previously asserted the shipped bundle was BLOCKED,
    which was true while every family carried `UNCONFIRMED` - and all three repo
    ids were nonexistent besides. The shipped bundle now validates, and that is
    the regression worth pinning: the near-miss was real and a warning would have
    shipped straight past it.

    The rejection path is still covered, against the shipped bundle's own text, in
    `tests/test_cli.py::test_validate_config_rejects_an_unconfirmed_checkpoint`.
    """
    problems = errors_only(validate(CONFIGS))
    assert "CFG_INVALID" not in {p.code for p in problems}, " ".join(p.message for p in problems)
    blob = " ".join(p.message for p in problems)
    assert "UNCONFIRMED" not in blob


def test_an_unconfirmed_checkpoint_is_still_an_error(complete_configs: Path) -> None:
    """The guard survives the fix above. Unpinning one family must hard-fail."""
    import yaml

    path = complete_configs / "readers.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["families"][0]["exact_checkpoint_confirmed"] = False
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    problems = errors_only(validate(complete_configs))
    codes = {p.code for p in problems}
    assert "CFG_INVALID" in codes
    message = " ".join(p.message for p in problems)
    assert "UNCONFIRMED checkpoint" in message
    assert "R7.6" in message


def test_placeholder_identifiers_are_rejected_even_when_marked_confirmed(
    complete_configs: Path,
) -> None:
    """Setting the flag must not let `UNCONFIRMED` through as a version string."""
    import yaml

    path = complete_configs / "readers.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["families"][0]["version"] = "UNCONFIRMED"
    data["families"][0]["exact_checkpoint_confirmed"] = True
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    codes = {p.code for p in errors_only(validate(complete_configs))}
    assert "CFG_INVALID" in codes


def test_no_paid_provider_remains() -> None:
    """The design is all-local. A stray anthropic/openai reader is a regression."""
    import yaml

    data = yaml.safe_load((CONFIGS / "readers.yaml").read_text(encoding="utf-8"))
    providers = {f["provider"] for f in data["families"]}
    assert providers == {"local"}, f"non-local reader provider(s) present: {providers}"


def test_no_pricing_for_a_paid_model_remains() -> None:
    """A paid price in the snapshot is either a leftover or a plan that came back."""
    import yaml

    data = yaml.safe_load((CONFIGS / "pricing" / "latest.yaml").read_text(encoding="utf-8"))
    for name, entry in data["models"].items():
        assert entry["provider"] == "local", f"{name} is not local"
        assert entry["input"] == 0.0 and entry["output"] == 0.0, f"{name} has a price"


def test_config_hash_is_stable_under_key_reordering(bundle_copy: Path) -> None:
    """Reordering keys is not a configuration change."""
    before = load_bundle(bundle_copy).config_hash

    path = bundle_copy / "retrieval.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(
        yaml.safe_dump(data, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )
    assert load_bundle(bundle_copy).config_hash == before


def test_config_hash_changes_on_a_single_prompt_character(bundle_copy: Path) -> None:
    """A one-character prompt edit MUST change the hash (PRD §G2)."""
    before = load_bundle(bundle_copy).config_hash
    path = bundle_copy / "prompts" / "arm_a.j2"
    path.write_text(
        path.read_text(encoding="utf-8").replace("shortest span", "shortest spans"),
        encoding="utf-8",
    )
    assert load_bundle(bundle_copy).config_hash != before


def test_preregistration_hash_is_independent_of_other_files(bundle_copy: Path) -> None:
    """PRD §R0.4: "registered in advance" must stay verifiable after later edits."""
    before = load_bundle(bundle_copy).preregistration_hash
    _edit(bundle_copy, "design.yaml", lambda d: d.update(runs_per_cell=99))
    assert load_bundle(bundle_copy).preregistration_hash == before


# ── R1.1 — no field may be null ────────────────────────────────────────────────
def test_null_field_in_retrieval_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "retrieval.yaml", lambda d: d["embedder"].update(dimensionality=None))
    assert "CFG_NULL_FIELD" in _codes(bundle_copy)


def test_null_similarity_metric_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "retrieval.yaml", lambda d: d["similarity"].update(metric=None))
    codes = _codes(bundle_copy)
    assert "CFG_NULL_FIELD" in codes


# ── R7.5 — two reader families are MANDATORY ──────────────────────────────────
def test_single_reader_family_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "readers.yaml", lambda d: d.__setitem__("families", d["families"][:1]))
    assert "CFG_TOO_FEW_READERS" in _codes(bundle_copy)


def test_two_reader_families_pass(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert bundle.readers.n_families >= 2


def test_reader_strength_is_declared_for_every_family(bundle_copy: Path) -> None:
    """PRD §R7.6 — the covariate must exist before R9 can model it."""
    bundle = load_bundle(bundle_copy)
    for family in bundle.readers.families:
        assert 0.0 <= family.reader_strength <= 1.0
        assert family.version and family.released


def test_reader_strength_spread_is_adequate(complete_configs: Path) -> None:
    """The three shipped readers span 0.72-0.80, which is enough for R9 to model.

    2606.21807's finding is that compression gain SHRINKS as the reader gets
    stronger. A panel whose strengths were all ~0.78 could not detect that trend
    even while satisfying R7.5's "two families" letter, so the validator warns on
    a flat panel and this asserts the shipped panel is NOT flat.
    """
    bundle = load_bundle(complete_configs)
    strengths = [f.reader_strength for f in bundle.readers.families]
    assert max(strengths) - min(strengths) >= 0.05
    assert "CFG_READER_STRENGTH_FLAT" not in _codes(complete_configs)


def test_flat_reader_panel_is_flagged(bundle_copy: Path) -> None:
    """Three near-identical readers test seed noise, not reader-strength dependence."""

    def mutate(d: dict[str, Any]) -> None:
        for family in d["families"]:
            family["reader_strength"] = 0.78

    _edit(bundle_copy, "readers.yaml", mutate)
    assert "CFG_READER_STRENGTH_FLAT" in _codes(bundle_copy)


def test_three_local_reader_families(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert bundle.readers.n_families == 3
    assert {f.provider for f in bundle.readers.families} == {"local"}


# ── R7.1 — seeds must vary something ──────────────────────────────────────────
def test_zero_temperature_without_nonce_is_rejected(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["temperature"] = 0.0
        d["prompt_nonce"] = False

    _edit(bundle_copy, "decoding.yaml", mutate)
    assert "CFG_INVALID" in _codes(bundle_copy)


def test_zero_temperature_with_nonce_is_accepted(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["temperature"] = 0.0
        d["prompt_nonce"] = True

    _edit(bundle_copy, "decoding.yaml", mutate)
    assert "CFG_INVALID" not in _codes(bundle_copy)


# ── R1.7 — B's mode travels with its value ────────────────────────────────────
def test_mixed_b_modes_are_flagged(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["B_grid"][-1]["mode"] = "corpus_fraction"

    _edit(bundle_copy, "design.yaml", mutate)
    assert "CFG_MIXED_B_MODES" in _codes(bundle_copy)


def test_every_b_level_declares_a_mode(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert all(b.mode for b in bundle.design.B_grid)


# ── R0.1 — design.yaml is the ONLY grid ───────────────────────────────────────
def test_duplicate_grid_definition_is_rejected(bundle_copy: Path) -> None:
    (bundle_copy / "extra.yaml").write_text("K_grid: [1, 2, 4]\n", encoding="utf-8")
    assert "CFG_DUPLICATE_GRID" in _codes(bundle_copy)


def test_shipped_configs_have_no_duplicate_grid() -> None:
    assert "CFG_DUPLICATE_GRID" not in _codes(CONFIGS)


# ── R1.2 — scoring rules live in config, not code ─────────────────────────────
def test_missing_normalisation_is_rejected(bundle_copy: Path) -> None:
    path = bundle_copy / "prompts" / "normalisation.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["normalisation"]["steps"] = []
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert "CFG_NO_NORMALISATION" in _codes(bundle_copy)


def test_missing_abstention_rule_is_rejected(bundle_copy: Path) -> None:
    path = bundle_copy / "prompts" / "abstention.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["abstention"]["scoring"]["decline_markers"] = []
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert "CFG_NO_ABSTENTION_RULE" in _codes(bundle_copy)


def test_all_prompt_templates_exist() -> None:
    for name in BUNDLE_FILES:
        if name.endswith(".j2") or name.endswith(".yaml"):
            assert (CONFIGS / name).is_file(), f"{name} missing from the bundle"


# ── N4/N8 — an unset budget is not an infinite budget ─────────────────────────
def test_unset_budget_is_flagged_but_not_an_error(bundle_copy: Path) -> None:
    """B7 must be loud, not fatal: the estimate is the input to the conversation."""
    problems = validate(bundle_copy)
    assert "CFG_BUDGET_UNSET" in {p.code for p in problems}
    assert errors_only(problems) == []


def test_budget_status_and_value_must_agree(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["max_usd"] = 100.0  # set, but status still says pending

    _edit(bundle_copy, "budget_guard.yaml", mutate)
    assert "CFG_INVALID" in _codes(bundle_copy)


# ── R0.4 — the pre-registration ───────────────────────────────────────────────
def test_preregistration_records_a_sign_prediction(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    pre = bundle.preregistration
    assert pre.interaction_sign_prediction in ("approximately_zero", "negative", "positive")
    assert pre.interaction_sign_prediction in pre.sign_readings
    assert pre.secondary_endpoints, "R9.7 needs an ENUMERATED family"


def test_preregistration_is_dated_before_collection(bundle_copy: Path) -> None:
    pre = load_bundle(bundle_copy).preregistration
    assert pre.registered == "2026-10-12"
    assert pre.status in ("pending_guide_signoff", "signed")


def test_hash_helper_is_order_independent() -> None:
    assert compute_config_hash({"a": 1, "b": 2}) == compute_config_hash({"b": 2, "a": 1})


def test_severity_values() -> None:
    assert Severity.ERROR.value == "error"
    assert Severity.WARN.value == "warn"
