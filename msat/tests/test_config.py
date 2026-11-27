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


def test_shipped_config_has_no_unexpected_errors(complete_configs: Path) -> None:
    problems = errors_only(validate(complete_configs))
    assert problems == [], "\n".join(str(p) for p in problems)


def test_shipped_config_has_no_unconfirmed_checkpoints() -> None:
    problems = errors_only(validate(CONFIGS))
    assert "CFG_INVALID" not in {p.code for p in problems}, " ".join(p.message for p in problems)
    blob = " ".join(p.message for p in problems)
    assert "UNCONFIRMED" not in blob


def test_an_unconfirmed_checkpoint_is_still_an_error(complete_configs: Path) -> None:
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
    import yaml

    path = complete_configs / "readers.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["families"][0]["version"] = "UNCONFIRMED"
    data["families"][0]["exact_checkpoint_confirmed"] = True
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    codes = {p.code for p in errors_only(validate(complete_configs))}
    assert "CFG_INVALID" in codes


def test_no_paid_provider_remains() -> None:
    import yaml

    data = yaml.safe_load((CONFIGS / "readers.yaml").read_text(encoding="utf-8"))
    providers = {f["provider"] for f in data["families"]}
    assert providers == {"local"}, f"non-local reader provider(s) present: {providers}"


def test_no_pricing_for_a_paid_model_remains() -> None:
    import yaml

    data = yaml.safe_load((CONFIGS / "pricing" / "latest.yaml").read_text(encoding="utf-8"))
    for name, entry in data["models"].items():
        assert entry["provider"] == "local", f"{name} is not local"
        assert entry["input"] == 0.0 and entry["output"] == 0.0, f"{name} has a price"


def test_config_hash_is_stable_under_key_reordering(bundle_copy: Path) -> None:
    before = load_bundle(bundle_copy).config_hash

    path = bundle_copy / "retrieval.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    path.write_text(
        yaml.safe_dump(data, sort_keys=True, default_flow_style=False), encoding="utf-8"
    )
    assert load_bundle(bundle_copy).config_hash == before


def test_config_hash_changes_on_a_single_prompt_character(bundle_copy: Path) -> None:
    before = load_bundle(bundle_copy).config_hash
    path = bundle_copy / "prompts" / "arm_a.j2"
    path.write_text(
        path.read_text(encoding="utf-8").replace("shortest span", "shortest spans"),
        encoding="utf-8",
    )
    assert load_bundle(bundle_copy).config_hash != before


def test_preregistration_hash_is_independent_of_other_files(bundle_copy: Path) -> None:
    before = load_bundle(bundle_copy).preregistration_hash
    _edit(bundle_copy, "design.yaml", lambda d: d.update(runs_per_cell=99))
    assert load_bundle(bundle_copy).preregistration_hash == before


def test_null_field_in_retrieval_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "retrieval.yaml", lambda d: d["embedder"].update(dimensionality=None))
    assert "CFG_NULL_FIELD" in _codes(bundle_copy)


def test_null_similarity_metric_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "retrieval.yaml", lambda d: d["similarity"].update(metric=None))
    codes = _codes(bundle_copy)
    assert "CFG_NULL_FIELD" in codes


def test_single_reader_family_is_rejected(bundle_copy: Path) -> None:
    _edit(bundle_copy, "readers.yaml", lambda d: d.__setitem__("families", d["families"][:1]))
    assert "CFG_TOO_FEW_READERS" in _codes(bundle_copy)


def test_two_reader_families_pass(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert bundle.readers.n_families >= 2


def test_reader_strength_is_declared_for_every_family(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    for family in bundle.readers.families:
        assert 0.0 <= family.reader_strength <= 1.0
        assert family.version and family.released


def test_reader_strength_spread_is_adequate(complete_configs: Path) -> None:
    bundle = load_bundle(complete_configs)
    strengths = [f.reader_strength for f in bundle.readers.families]
    assert max(strengths) - min(strengths) >= 0.05
    assert "CFG_READER_STRENGTH_FLAT" not in _codes(complete_configs)


def test_flat_reader_panel_is_flagged(bundle_copy: Path) -> None:

    def mutate(d: dict[str, Any]) -> None:
        for family in d["families"]:
            family["reader_strength"] = 0.78

    _edit(bundle_copy, "readers.yaml", mutate)
    assert "CFG_READER_STRENGTH_FLAT" in _codes(bundle_copy)


def test_three_local_reader_families(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert bundle.readers.n_families == 3
    assert {f.provider for f in bundle.readers.families} == {"local"}


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


def test_mixed_b_modes_are_flagged(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["B_grid"][-1]["mode"] = "corpus_fraction"

    _edit(bundle_copy, "design.yaml", mutate)
    assert "CFG_MIXED_B_MODES" in _codes(bundle_copy)


def test_every_b_level_declares_a_mode(bundle_copy: Path) -> None:
    bundle = load_bundle(bundle_copy)
    assert all(b.mode for b in bundle.design.B_grid)


def test_duplicate_grid_definition_is_rejected(bundle_copy: Path) -> None:
    (bundle_copy / "extra.yaml").write_text("K_grid: [1, 2, 4]\n", encoding="utf-8")
    assert "CFG_DUPLICATE_GRID" in _codes(bundle_copy)


def test_shipped_configs_have_no_duplicate_grid() -> None:
    assert "CFG_DUPLICATE_GRID" not in _codes(CONFIGS)


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


def test_unset_budget_is_flagged_but_not_an_error(bundle_copy: Path) -> None:
    problems = validate(bundle_copy)
    assert "CFG_BUDGET_UNSET" in {p.code for p in problems}
    assert errors_only(problems) == []


def test_budget_status_and_value_must_agree(bundle_copy: Path) -> None:
    def mutate(d: dict[str, Any]) -> None:
        d["max_usd"] = 100.0

    _edit(bundle_copy, "budget_guard.yaml", mutate)
    assert "CFG_INVALID" in _codes(bundle_copy)


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
