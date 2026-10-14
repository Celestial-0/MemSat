"""`providers/local.py` - a reader FAMILY bound to a local backend (P8, R7.5/R7.6).

THE THREE THINGS THIS MODULE OWNS
---------------------------------
1. **The ceiling is computed.** Every family in `readers.yaml` declares a 131k-262k
   window and none can deliver it on 8 GB, because the weights are resident before
   the KV cache grows. A typed ceiling is a number that looks authoritative,
   survives a card swap, and is then wrong.
2. **Overflow is a rejected cell, not a truncated prompt.** A prompt cut to fit still
   contains its own answer while appearing not to, and the curve that follows
   measures nothing (R3.2, `never_truncate_silently`).
3. **Identity, version, date and `reader_strength` are recorded** (R7.6) so R9 can
   model reader strength as a covariate.

THE BUG THESE TESTS WERE WRITTEN AGAINST
----------------------------------------
`bytes_per_param: 0.55` in the config is BYTES (4.4 bits, int4 + overhead). An early
draft of `weights_gb` divided by 8 again as though it were bits, reporting gemma's
weights as 0.26 GB instead of 2.05 GB and inflating its ceiling from ~37k to ~59k
tokens - in the direction that marks deep cells servable when they are not.

`test_the_bytes_per_param_conversion_is_not_applied_twice` pins the unit, because
"nearly 1.6x, the wrong way" is the failure mode that hides until a run OOMs at hour
14.

THE HEADROOM IS NOT A CONSTANT
-----------------------------
`DEFAULT_HEADROOM_GB = 3.0` versus the ~0.8 GB implied by `readers.yaml`'s hand-typed
ceiling comments is a 7.8x swing on qwen (2,682 vs 20,704 tokens) - and qwen at
2,682 is unusable as the mandatory second reader. `test_the_headroom_decides_whether
_qwen_is_usable` makes that visible rather than picking a number quietly. Open
decision D11.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from msat.hardware.vram import DEFAULT_HEADROOM_GB, kv_ceiling_tokens
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

CONFIG = Path("configs/readers.yaml")

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
    fam = {**FAMILY, **over}
    return reader_from_family(fam, card_gb=24.0)


# ── the ceiling is computed, and the unit conversion happens exactly once ──────


def test_the_bytes_per_param_conversion_is_not_applied_twice() -> None:
    """`0.55` is BYTES. 7e9 params x 0.55 B = 3.85e9 B = 3.58 GiB.

    Dividing by 8 as well would report 0.45 GB, freeing ~3 GB of budget that does not
    exist and inflating the ceiling by 1.6x - in the direction that marks deep cells
    servable when they are not.
    """
    r = _reader()
    assert r.effective_quant_bits == pytest.approx(4.4)
    assert r.weights_gb == pytest.approx(7.0e9 * 0.55 / 1024**3)
    assert r.weights_gb == pytest.approx(3.586, abs=0.01)
    assert r.weights_gb > 3.0, "a 7B model at int4 is ~3.6 GB, not 0.45 GB"


def test_the_ceiling_agrees_with_the_shared_vram_helper() -> None:
    """One formula, one implementation.

    `LocalReader` must not re-derive the arithmetic: `msat vram` and this module
    reporting different ceilings for the same checkpoint is precisely how a card's
    feasibility check and a run's actual allocation come to disagree."""
    r = _reader()
    assert r.ceiling_tokens() == kv_ceiling_tokens(
        r.shape, 24.0, quant_bits=r.effective_quant_bits, headroom_gb=r.headroom_gb
    )


def test_a_bigger_card_gives_a_bigger_ceiling() -> None:
    r = _reader()
    assert r.ceiling_tokens(card_gb=48.0) > r.ceiling_tokens(card_gb=8.0)


def test_a_card_too_small_for_the_weights_gives_zero_not_a_negative() -> None:
    """A negative capacity would make `serves()` true for nothing and read as a
    rejection of the reader rather than of the card."""
    r = _reader()
    assert r.ceiling_tokens(card_gb=1.0) == 0


def test_the_ceiling_is_computed_rather_than_read_from_config() -> None:
    """Nothing in the family entry declares a ceiling, so there is nothing to read -
    and if someone adds one, it must not silently become the answer, because the
    declared window is a property of the CHECKPOINT and the ceiling of the CARD."""
    r = _reader()
    assert r.ceiling_tokens() != r.declared_context_window
    assert r.declared_context_window == 131072


# ── VRAM, not the declared window, is the binding constraint here ──────────────


def test_every_shipped_family_is_vram_bound() -> None:
    """`readers.yaml` says so in prose. Asserted so the claim cannot rot unnoticed."""
    for r in load_readers(CONFIG, card_gb=8.0):
        assert r.binding_constraint == "vram", r.id
        assert r.serving_limit_tokens < r.declared_context_window, r.id


def test_the_serving_limit_is_the_smaller_of_the_two_bounds() -> None:
    r = _reader()
    assert r.serving_limit_tokens == min(r.declared_context_window, r.ceiling_tokens())


def test_context_window_reports_the_servable_limit_not_the_declared_one() -> None:
    """`base.LocalBackend.context_window` feeds the context-feasibility check BEFORE
    a run (R3.2). Returning the declared window there marks every long cell feasible
    and then fails at allocation, fourteen hours in.

    On 8 GB this family's VRAM ceiling is below its declared window, so the two
    answers genuinely differ - which is the only case where the distinction is
    visible. On a 24 GB card they coincide and there is nothing to catch.
    """
    r = reader_from_family(FAMILY, card_gb=8.0)
    assert r.ceiling_tokens() < r.declared_context_window
    assert r.context_window() == r.ceiling_tokens()
    assert r.context_window() != 131072


def test_on_a_big_card_the_declared_window_binds_instead() -> None:
    """The limit is a MINIMUM of two bounds, and which one binds is reported. A test
    that only ever saw the VRAM case would not notice the window being ignored."""
    r = reader_from_family(FAMILY, card_gb=24.0)
    assert r.ceiling_tokens() > r.declared_context_window
    assert r.binding_constraint == "declared_window"
    assert r.serving_limit_tokens == r.declared_context_window


# ── the headroom allowance decides whether the panel is usable (D11) ───────────


def test_the_headroom_decides_whether_qwen_is_usable() -> None:
    """THE finding this phase produced.

    qwen is the secondary reader R7.5 requires. At `vram.py`'s 3.0 GB default it can
    hold 2,682 prompt tokens, which is not a reader; at the ~0.8 GB allowance implied
    by `readers.yaml`'s hand-typed comments it reaches ~20,700 and is fine.

    So the panel has two usable readers or three depending on a number that is a
    guess about allocator slack rather than a measurement. That is a decision, and
    it is not one to make silently inside a helper's default.
    """
    qwen = next(r for r in load_readers(CONFIG, card_gb=8.0) if r.id == "qwen")
    tight = qwen.ceiling_at_headroom(0.8)
    default = qwen.ceiling_tokens()
    assert default == DEFAULT_HEADROOM_GB or qwen.headroom_gb == DEFAULT_HEADROOM_GB
    assert tight > 5 * default, "the swing is large enough to change the experiment"
    assert default < 4096, "at the default allowance qwen cannot serve a real prompt"


def test_the_sensitivity_report_names_the_allowance_behind_each_number() -> None:
    """A single ceiling would hide a 7.8x swing behind a decision nobody made."""
    qwen = next(r for r in load_readers(CONFIG, card_gb=8.0) if r.id == "qwen")
    s = qwen.ceiling_sensitivity()
    assert set(s) == {"tight_0.8gb", "default", "loose_5.0gb"}
    assert s["tight_0.8gb"] >= s["default"] >= s["loose_5.0gb"]


def test_the_headroom_is_a_parameter_rather_than_a_buried_default() -> None:
    """So it is a policy value a reviewer can see and change, not a constant living
    in another module's signature."""
    a = reader_from_family(FAMILY, card_gb=24.0, headroom_gb=0.8)
    b = reader_from_family(FAMILY, card_gb=24.0, headroom_gb=3.0)
    assert a.headroom_gb == 0.8
    assert b.headroom_gb == DEFAULT_HEADROOM_GB
    assert a.ceiling_tokens() > b.ceiling_tokens()


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


# ── overflow is a rejected cell, never a truncated prompt ─────────────────────


def test_a_prompt_over_the_ceiling_is_refused_with_the_numbers() -> None:
    """On 8 GB, where this family's VRAM ceiling is the binding one."""
    r = reader_from_family(FAMILY, card_gb=8.0)
    assert r.binding_constraint == "vram"
    with pytest.raises(ContextOverflowError) as exc:
        r.check_capacity(r.serving_limit_tokens + 1)
    msg = str(exc.value)
    assert f"{r.serving_limit_tokens:,}" in msg, "the limit, formatted for reading"
    assert "vram" in msg, "which bound it was, so the reader knows what to change"
    assert "declared" in msg, "and the other number, which was not the binding one"


def test_a_prompt_exactly_at_the_ceiling_is_served() -> None:
    r = _reader()
    assert r.check_capacity(r.serving_limit_tokens) == r.serving_limit_tokens


def test_the_refusal_explains_why_truncation_is_wrong() -> None:
    """The message has to carry the reasoning, because the obvious fix - truncate -
    would look reasonable and be catastrophic."""
    r = _reader()
    with pytest.raises(ContextOverflowError, match="contain its own answer"):
        r.check_capacity(r.serving_limit_tokens + 10_000)


def test_serves_never_raises() -> None:
    r = _reader()
    assert r.serves(0) is True
    assert r.serves(r.serving_limit_tokens) is True
    assert r.serves(r.serving_limit_tokens + 1) is False


# ── coverage is reported, never assumed (R0.3 / N8) ──────────────────────────


def test_coverage_lists_the_unsupported_cells() -> None:
    """A hole nobody is told about is exactly the silently-truncated factorial that
    R0.3 and N8 forbid."""
    r = _reader()
    cells = [(k, 128.0, "S") for k in (1, 2, 4, 8, 16, 32)]
    rep = r.coverage_over(cells)
    assert rep["n_cells"] == 6
    assert rep["n_served"] + rep["n_unsupported"] == 6, "every cell lands somewhere"
    assert rep["coverage"] == pytest.approx(rep["n_served"] / 6)


def test_coverage_names_each_unsupported_cell() -> None:
    r = _reader()
    rep = r.coverage_over([(k, 128.0, "S") for k in (1, 2, 4, 8)])
    for entry in rep["unsupported"]:
        assert set(entry) >= {"k", "b", "scale", "reason"}


def test_panel_coverage_reports_non_uniformity() -> None:
    """On 8 GB the three families do NOT have the same ceiling, so they do not have
    the same support - which is why R9.12 refuses to compare them blindly."""
    readers = [
        reader_from_family({**FAMILY, "id": "small", "vram": {**FAMILY["vram"], "params_b": 3.0}}),
        reader_from_family({**FAMILY, "id": "big", "vram": {**FAMILY["vram"], "params_b": 30.0}}),
    ]
    cells = [(k, 128.0, "S") for k in (1, 2, 4, 8, 16, 32)]
    cov = panel_coverage(readers, cells)
    assert cov["uniform_coverage"] is False
    assert set(cov["served_per_reader"]) == {"small", "big"}
    assert "R9.12" in cov["note"]


def test_panel_coverage_reports_uniformity_when_it_is_uniform() -> None:
    readers = [_reader(), reader_from_family({**FAMILY, "id": "twin"})]
    cov = panel_coverage(readers, [(k, 128.0, "S") for k in (1, 2, 4)])
    assert cov["uniform_coverage"] is True


# ── R7.6: identity, version, date, strength ───────────────────────────────────


def test_the_metadata_carries_everything_r7_6_requires() -> None:
    m = _reader().reader_metadata()
    for key in ("id", "model", "version", "released", "reader_strength"):
        assert m[key] not in (None, ""), key
    assert m["version"] == "a" * 40


def test_the_version_is_a_commit_sha_not_a_tag() -> None:
    """Tags get repointed, which changes the weights under a fixed `config_hash` and
    makes two runs incomparable while every artifact claims the same identity."""
    for r in load_readers(CONFIG):
        assert len(r.version) == 40, f"{r.id}: {r.version!r} is not a commit SHA"
        assert all(c in "0123456789abcdef" for c in r.version), r.id


def test_reader_strength_travels_with_the_family() -> None:
    """R7.6 exists so R9 can model it as a covariate, and 2606.21807's finding is that
    it matters."""
    assert _reader().reader_metadata()["reader_strength"] == 0.75
    strengths = {r.id: r.reader_strength for r in load_readers(CONFIG)}
    assert len(set(strengths.values())) > 1, "a panel with one strength cannot test it"


def test_the_effective_quantisation_is_recorded_not_just_the_request() -> None:
    """R6.4: quantisation changes the weights and therefore the answers, and a silent
    int4 -> bf16 fallback is undetectable without both numbers."""
    m = _reader().reader_metadata()
    assert m["quant_bits"] == 4.0, "what was asked for"
    assert m["bytes_per_param"] == 0.55, "what the config decided"


def test_an_unconfirmed_checkpoint_is_carried_through() -> None:
    """`config.py` makes it an ERROR; this must not quietly promote it to confirmed."""
    r = reader_from_family({**FAMILY, "exact_checkpoint_confirmed": False})
    assert r.reader_metadata()["exact_checkpoint_confirmed"] is False


def test_a_missing_identity_field_is_refused_rather_than_defaulted() -> None:
    """The previous panel was three nonexistent models because ids were invented."""
    for missing in ("version", "released", "reader_strength", "model"):
        fam = {k: v for k, v in FAMILY.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            reader_from_family(fam)


# ── R7.5: at least two families ───────────────────────────────────────────────


def test_a_one_family_panel_is_refused() -> None:
    """A one-reader panel is not a smaller experiment - it cannot test whether the
    depth curve generalises across readers at all."""
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


# ── the engine is bound lazily and its identity is stamped ────────────────────


def test_building_a_reader_does_not_load_an_engine() -> None:
    """`design-power` and `validate-config` call this on a machine with no GPU. An
    engine imported in `__init__` would pay 30 s and 18 GB to answer "no"."""
    r = _reader()
    assert r._backend is None


def test_context_window_needs_no_engine() -> None:
    r = _reader()
    assert r.context_window() > 0
    assert r._backend is None, "the feasibility check must not need a GPU"


def test_health_reports_the_family_even_when_the_engine_is_missing() -> None:
    """A reader debugging a refused run needs the ceiling; an engine error alone does
    not say it."""
    h = _reader(backend="does-not-exist").health()
    assert h["id"] == "test"
    assert h["serving_limit"] > 0
    assert h["ready"] is False


def test_generation_stamps_the_configured_version_onto_the_result() -> None:
    """R6.2: prices and behaviour both drift, so a re-run a month later is not
    comparable without the version. A backend filling it from whatever it loaded
    records the engine's idea of the version - the exact drift being guarded against.

    Driven through a fake backend rather than a real one: this test is about the
    stamp, and no GPU is involved.
    """
    stamped = GenerationResult(
        text="hi", finish_reason="stop", prompt_tokens=1, completion_tokens=1, latency_ms=1.0
    )
    reader = reader_from_family(FAMILY, card_gb=24.0)
    object.__setattr__(reader, "_backend", _FakeBackend(stamped))
    got = reader.generate(GenerationRequest(model="org/test-7b", prompt="p"))
    assert got.model_version == FAMILY["version"]


def test_a_backend_supplied_version_is_not_overwritten() -> None:
    """If the engine genuinely knows the loaded revision, that is better information
    than the config's pin."""
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
    """Minimal `LocalBackend`. Health is deliberately irrelevant to these tests."""

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


# ── the shipped config actually loads ─────────────────────────────────────────


def test_the_shipped_config_loads_into_readers() -> None:
    readers = load_readers(CONFIG)
    assert [r.id for r in readers] == ["gemma", "qwen", "phi"]


def test_the_shipped_config_reports_a_positive_ceiling_for_every_family() -> None:
    for r in load_readers(CONFIG, card_gb=8.0):
        assert r.ceiling_tokens() > 0, r.id
        assert r.kv_bytes_per_token > 0, r.id


def test_the_kv_footprint_follows_the_declared_architecture() -> None:
    """2 x layers x kv_heads x head_dim x 2 bytes, by hand for gemma:
    2 x 42 x 2 x 256 x 2 = 86,016."""
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
    """A family with no `vram:` block must fail as INFEASIBLE, not as a
    ZeroDivisionError - and must not be reported as a reader that can serve
    everything."""
    r = reader_from_family({k: v for k, v in FAMILY.items() if k != "vram"}, card_gb=24.0)
    assert r.ceiling_tokens() == 0
    assert r.serves(1) is False


def test_the_vram_spec_round_trips_into_a_shape() -> None:
    spec = ReaderVramSpec(params_b=7.0, layers=32, kv_heads=8, head_dim=128)
    shape = spec.to_shape(name="x")
    assert shape.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert shape.name == "x"
