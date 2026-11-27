from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msat.hardware.vram import DEFAULT_HEADROOM_GB, kv_reader_capacity_tokens
from msat.providers.base import GenerationRequest, GenerationResult
from msat.providers.local import (
    ContextOverflowError,
    LocalReader,
    ReaderVramSpec,
    load_readers,
    panel_coverage,
    reader_from_family,
    readers_from_config,
)

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "readers.yaml"

FAMILY = {
    "id": "test",
    "model": "org/test-7b",
    "version": "a" * 40,
    "released": "2026-01-01",
    "role": "primary",
    "reader_strength": 0.75,
    "context_window": 131072,
    "exact_checkpoint_confirmed": True,
    "vram": {
        "params_b": 7.0,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "bytes_per_param": 0.55,
    },
}


def _reader(**over: Any) -> LocalReader:
    backend = over.pop("backend", "transformers")
    fam = {**FAMILY, **over}
    return reader_from_family(fam, card_gb=24.0, backend=backend)


def test_the_bytes_per_param_conversion_is_not_applied_twice() -> None:
    r = _reader()
    assert r.effective_quant_bits == pytest.approx(4.4)
    assert r.weights_gb == pytest.approx(7.0e9 * 0.55 / 1024**3)
    assert r.weights_gb == pytest.approx(3.586, abs=0.01)
    assert r.weights_gb > 3.0, "a 7B model at int4 is ~3.6 GB, not 0.45 GB"


def test_the_ceiling_agrees_with_the_shared_vram_helper() -> None:
    r = _reader()
    assert r.vram_capacity_tokens() == kv_reader_capacity_tokens(
        r.shape, 24.0, quant_bits=r.effective_quant_bits, headroom_gb=r.headroom_gb
    )


def test_a_bigger_card_gives_a_bigger_ceiling() -> None:
    r = _reader()
    assert r.vram_capacity_tokens(card_gb=48.0) > r.vram_capacity_tokens(card_gb=8.0)


def test_a_card_too_small_for_the_weights_gives_zero_not_a_negative() -> None:
    r = _reader()
    assert r.vram_capacity_tokens(card_gb=1.0) == 0


def test_the_ceiling_is_computed_rather_than_read_from_config() -> None:
    r = _reader()
    assert r.vram_capacity_tokens() != r.declared_context_window
    assert r.declared_context_window == 131072


def test_every_shipped_family_is_vram_bound() -> None:
    for r in load_readers(CONFIG, card_gb=8.0):
        assert r.binding_constraint == "vram", r.id
        assert r.reader_capacity_tokens < r.declared_context_window, r.id


def test_the_serving_limit_is_the_smaller_of_the_two_bounds() -> None:
    r = _reader()
    assert r.reader_capacity_tokens == min(r.declared_context_window, r.vram_capacity_tokens())


def test_context_window_reports_the_servable_limit_not_the_declared_one() -> None:
    r = reader_from_family(FAMILY, card_gb=8.0)
    assert r.vram_capacity_tokens() < r.declared_context_window
    assert r.context_window() == r.vram_capacity_tokens()
    assert r.context_window() != 131072


def test_on_a_big_card_the_declared_window_binds_instead() -> None:
    r = reader_from_family(FAMILY, card_gb=24.0)
    assert r.vram_capacity_tokens() > r.declared_context_window
    assert r.binding_constraint == "declared_window"
    assert r.reader_capacity_tokens == r.declared_context_window


def test_the_headroom_decides_whether_qwen_is_usable() -> None:
    qwen = next(r for r in load_readers(CONFIG, card_gb=8.0) if r.id == "qwen")
    tight = qwen.vram_capacity_at_headroom(0.8)
    default = qwen.vram_capacity_tokens()
    assert default == DEFAULT_HEADROOM_GB or qwen.headroom_gb == DEFAULT_HEADROOM_GB
    assert tight > 5 * default, "the swing is large enough to change the experiment"
    assert default < 4096, "at the default allowance qwen cannot serve a real prompt"


def test_the_sensitivity_report_names_the_allowance_behind_each_number() -> None:
    qwen = next(r for r in load_readers(CONFIG, card_gb=8.0) if r.id == "qwen")
    s = qwen.capacity_sensitivity()
    assert set(s) == {"tight_0.8gb", "default", "loose_5.0gb"}
    assert s["tight_0.8gb"] >= s["default"] >= s["loose_5.0gb"]


def test_the_headroom_is_a_parameter_rather_than_a_buried_default() -> None:
    a = reader_from_family(FAMILY, card_gb=24.0, headroom_gb=0.8)
    b = reader_from_family(FAMILY, card_gb=24.0, headroom_gb=3.0)
    assert a.headroom_gb == 0.8
    assert b.headroom_gb == DEFAULT_HEADROOM_GB
    assert a.vram_capacity_tokens() > b.vram_capacity_tokens()


def test_context_policy_headroom_is_honoured_when_present() -> None:
    doc = {
        "families": [FAMILY, {**FAMILY, "id": "second"}],
        "context_policy": {"headroom_gb": 0.8},
    }
    got = readers_from_config(doc, card_gb=24.0)
    assert all(r.headroom_gb == 0.8 for r in got)


def test_context_policy_headroom_falls_back_to_the_default() -> None:
    doc = {"families": [FAMILY, {**FAMILY, "id": "second"}]}
    got = readers_from_config(doc, card_gb=24.0)
    assert all(r.headroom_gb == DEFAULT_HEADROOM_GB for r in got)


def test_a_prompt_over_the_ceiling_is_refused_with_the_numbers() -> None:
    r = reader_from_family(FAMILY, card_gb=8.0)
    assert r.binding_constraint == "vram"
    with pytest.raises(ContextOverflowError) as exc:
        r.check_capacity(r.reader_capacity_tokens + 1)
    msg = str(exc.value)
    assert f"{r.reader_capacity_tokens:,}" in msg, "the limit, formatted for reading"
    assert "vram" in msg, "which bound it was, so the reader knows what to change"
    assert "declared" in msg, "and the other number, which was not the binding one"


def test_a_prompt_exactly_at_the_ceiling_is_served() -> None:
    r = _reader()
    assert r.check_capacity(r.reader_capacity_tokens) == r.reader_capacity_tokens


def test_the_refusal_explains_why_truncation_is_wrong() -> None:
    r = _reader()
    with pytest.raises(ContextOverflowError, match="contain its own answer"):
        r.check_capacity(r.reader_capacity_tokens + 10_000)


def test_supports_never_raises() -> None:
    r = _reader()
    assert r.supports(0) is True
    assert r.supports(r.reader_capacity_tokens) is True
    assert r.supports(r.reader_capacity_tokens + 1) is False


def test_coverage_lists_the_unsupported_cells() -> None:
    r = _reader()
    cells = [(k, 128.0, "S") for k in (1, 2, 4, 8, 16, 32)]
    rep = r.coverage_over(cells, token_basis=lambda k, b, scale: int(k * b) + 500)
    assert rep["n_cells"] == 6
    assert rep["n_supported"] + rep["n_unsupported"] == 6, "every cell lands somewhere"
    assert rep["coverage"] == pytest.approx(rep["n_supported"] / 6)


def test_coverage_names_each_unsupported_cell() -> None:
    r = _reader()
    rep = r.coverage_over(
        [(k, 128.0, "S") for k in (1, 2, 4, 8)], token_basis=lambda k, b, scale: int(k * b) + 500
    )
    for entry in rep["unsupported"]:
        assert set(entry) >= {"k", "b", "scale", "reason"}


def test_panel_coverage_reports_non_uniformity() -> None:
    readers = [
        reader_from_family({**FAMILY, "id": "small", "vram": {**FAMILY["vram"], "params_b": 3.0}}),
        reader_from_family({**FAMILY, "id": "big", "vram": {**FAMILY["vram"], "params_b": 30.0}}),
    ]
    cells = [(k, 128.0, "S") for k in (1, 2, 4, 8, 16, 32)]
    cov = panel_coverage(readers, cells, token_basis=lambda k, b, scale: int(k * b) + 500)
    assert cov["uniform_coverage"] is False
    assert set(cov["supported_per_reader"]) == {"small", "big"}
    assert "R9.12" in cov["note"]


def test_panel_coverage_reports_uniformity_when_it_is_uniform() -> None:
    readers = [_reader(), reader_from_family({**FAMILY, "id": "twin"})]
    cov = panel_coverage(
        readers,
        [(k, 128.0, "S") for k in (1, 2, 4)],
        token_basis=lambda k, b, scale: int(k * b) + 500,
    )
    assert cov["uniform_coverage"] is True


def test_the_metadata_carries_everything_r7_6_requires() -> None:
    m = _reader().reader_metadata()
    for key in ("id", "model", "version", "released", "reader_strength"):
        assert m[key] not in (None, ""), key
    assert m["version"] == "a" * 40


def test_the_version_is_a_commit_sha_not_a_tag() -> None:
    for r in load_readers(CONFIG):
        assert len(r.version) == 40, f"{r.id}: {r.version!r} is not a commit SHA"
        assert all(c in "0123456789abcdef" for c in r.version), r.id


def test_reader_strength_travels_with_the_family() -> None:
    assert _reader().reader_metadata()["reader_strength"] == 0.75
    strengths = {r.id: r.reader_strength for r in load_readers(CONFIG)}
    assert len(set(strengths.values())) > 1, "a panel with one strength cannot test it"


def test_the_effective_quantisation_is_recorded_not_just_the_request() -> None:
    m = _reader().reader_metadata()
    assert m["quant_bits"] == 4.0, "what was asked for"
    assert m["bytes_per_param"] == 0.55, "what the config decided"


def test_an_unconfirmed_checkpoint_is_carried_through() -> None:
    r = reader_from_family({**FAMILY, "exact_checkpoint_confirmed": False})
    assert r.reader_metadata()["exact_checkpoint_confirmed"] is False


def test_a_missing_identity_field_is_refused_rather_than_defaulted() -> None:
    for missing in ("version", "released", "reader_strength", "model"):
        fam = {k: v for k, v in FAMILY.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            reader_from_family(fam)


def test_a_one_family_panel_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        readers_from_config({"families": [FAMILY]}, card_gb=24.0)


def test_a_two_family_panel_is_accepted() -> None:
    got = readers_from_config({"families": [FAMILY, {**FAMILY, "id": "second"}]}, card_gb=24.0)
    assert [r.id for r in got] == ["test", "second"]


def test_the_shipped_config_declares_three_families() -> None:
    assert len(load_readers(CONFIG)) == 3


def test_a_missing_family_block_is_an_error_not_an_empty_panel() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        readers_from_config({}, card_gb=24.0)


def test_building_a_reader_does_not_load_an_engine() -> None:
    r = _reader()
    assert r._backend is None


def test_context_window_needs_no_engine() -> None:
    r = _reader()
    assert r.context_window() > 0
    assert r._backend is None, "the feasibility check must not need a GPU"


def test_health_reports_the_family_even_when_the_engine_is_missing() -> None:
    h = _reader(backend="does-not-exist").health()
    assert h["id"] == "test"
    assert h["reader_capacity_tokens"] > 0
    assert h["ready"] is False


def test_generation_stamps_the_configured_version_onto_the_result() -> None:
    stamped = GenerationResult(
        text="hi", finish_reason="stop", prompt_tokens=1, completion_tokens=1, latency_ms=1.0
    )
    reader = reader_from_family(FAMILY, card_gb=24.0)
    object.__setattr__(reader, "_backend", _FakeBackend(stamped))
    got = reader.generate(GenerationRequest(model="org/test-7b", prompt="p"))
    assert got.model_version == FAMILY["version"]


def test_a_backend_supplied_version_is_not_overwritten() -> None:
    supplied = GenerationResult(
        text="hi",
        finish_reason="stop",
        prompt_tokens=1,
        completion_tokens=1,
        latency_ms=1.0,
        model_version="deadbeef",
    )
    reader = reader_from_family(FAMILY, card_gb=24.0)
    object.__setattr__(reader, "_backend", _FakeBackend(supplied))
    assert reader.generate(GenerationRequest(model="m", prompt="p")).model_version == "deadbeef"


class _FakeBackend:
    name = "fake"

    def __init__(self, result: GenerationResult) -> None:
        self._result = result

    def health(self) -> dict[str, Any]:
        return {"ready": True}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._result

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [self._result for _ in requests]

    def context_window(self, model: str) -> int:
        return 131072

    def close(self) -> None:
        pass


def test_the_shipped_config_loads_into_readers() -> None:
    readers = load_readers(CONFIG)
    assert [r.id for r in readers] == ["gemma", "qwen", "phi"]


def test_the_shipped_config_reports_a_positive_ceiling_for_every_family() -> None:
    for r in load_readers(CONFIG, card_gb=8.0):
        assert r.vram_capacity_tokens() > 0, r.id
        assert r.kv_bytes_per_token > 0, r.id


def test_the_kv_footprint_follows_the_declared_architecture() -> None:
    gemma = load_readers(CONFIG)[0]
    assert gemma.kv_bytes_per_token == 86_016


def test_the_reader_metadata_is_json_serialisable() -> None:
    json.dumps([r.reader_metadata() for r in load_readers(CONFIG)])


def test_a_reader_round_trips_through_its_dict() -> None:
    r = _reader()
    d = r.as_dict()
    assert d["id"] == r.id
    assert d["vram"]["params_b"] == r.vram.params_b
    assert "rationale" in d


def test_a_zero_shaped_vram_block_gives_no_ceiling_rather_than_a_crash() -> None:
    r = reader_from_family({k: v for k, v in FAMILY.items() if k != "vram"}, card_gb=24.0)
    assert r.vram_capacity_tokens() == 0
    assert r.supports(1) is False


def test_the_vram_spec_round_trips_into_a_shape() -> None:
    spec = ReaderVramSpec(params_b=7.0, layers=32, kv_heads=8, head_dim=128)
    shape = spec.to_shape(name="x")
    assert shape.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert shape.name == "x"
