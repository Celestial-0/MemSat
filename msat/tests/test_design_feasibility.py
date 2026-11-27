from __future__ import annotations

from pathlib import Path

import pytest

from msat.design import (
    Cell,
    ReaderLimit,
    budget_tokens,
    build_cells,
    context_feasibility,
    duplicate_conditions,
    headroom_that_fits,
    prompt_tokens,
    vram_max_prompt_tokens,
)
from msat.hardware.vram import DEFAULT_HEADROOM_GB

PRICING = {
    "prompt_overhead_tokens": 500,
    "token_estimates": {
        "S": {
            "prompt_tokens_at_K1": 1000,
            "prompt_tokens_at_K64": 8000,
            "corpus_tokens": 8000,
        },
        "M": {
            "prompt_tokens_at_K1": 2000,
            "prompt_tokens_at_K64": 16000,
            "corpus_tokens": 16000,
        },
    },
}


def test_vram_capacity_is_plain_subtraction_and_division() -> None:
    free_mb = (8.0 - 2.0 - DEFAULT_HEADROOM_GB) * 1024.0
    expected = int(free_mb / 100.0 * 1000.0)
    assert vram_max_prompt_tokens(weight_gb=2.0, kv_mb_per_1k=100.0) == expected


def test_the_design_and_runtime_ceilings_are_the_same_number() -> None:
    from msat.hardware.vram import ModelShape, kv_reader_capacity_tokens

    shape = ModelShape(
        name="probe",
        n_params=9_000_000_000,
        n_layers=32,
        n_kv_heads=4,
        head_dim=256,
        n_attention_heads=32,
        dtype="bfloat16",
        n_params_is_exact=True,
    )
    quant_bits = 4.4
    kv_mb_per_1k = shape.kv_bytes_per_token() * 1000 / 1024 / 1024

    design_side = vram_max_prompt_tokens(
        weight_gb=shape.weights_gb(quant_bits=quant_bits),
        kv_mb_per_1k=kv_mb_per_1k,
    )
    runtime_side = kv_reader_capacity_tokens(shape, 8.0, quant_bits=quant_bits)

    assert 0 <= design_side - runtime_side <= 512


def test_headroom_that_fits_reports_the_threshold_not_a_preference() -> None:
    ok = headroom_that_fits(weight_gb=4.610, kv_mb_per_1k=125.0, required_prompt_tokens=16_884)
    assert 1.0 < ok < 1.5, f"threshold moved to {ok:.2f} GB; re-derive before trusting it"

    assert headroom_that_fits(weight_gb=7.5, kv_mb_per_1k=100.0, required_prompt_tokens=16_884) < 0


def test_shipped_readers_weight_gb_is_gibibytes_not_decimal_gigabytes() -> None:
    from msat.config import load_bundle

    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    bundle = load_bundle(configs_dir)
    expected = {"gemma": 2.049, "phi": 1.946, "qwen": 4.610}
    for family in bundle.readers.families:
        if family.vram is None:
            continue
        assert family.vram.weight_gb == pytest.approx(expected[family.id], abs=0.005), (
            f"{family.id}: {family.vram.weight_gb:.3f} GiB, expected {expected[family.id]}"
        )


def test_weights_alone_that_do_not_fit_yield_zero_not_a_huge_number() -> None:
    assert vram_max_prompt_tokens(weight_gb=30.0, kv_mb_per_1k=50.0) == 0


def test_vram_capacity_rejects_a_nonsense_kv_rate() -> None:
    with pytest.raises(ValueError, match="kv_mb_per_1k"):
        vram_max_prompt_tokens(weight_gb=2.0, kv_mb_per_1k=0.0)


def test_binding_ceiling_is_the_smaller_of_window_and_vram() -> None:
    free_mb = (8.0 - 2.0 - DEFAULT_HEADROOM_GB) * 1024.0
    wide = ReaderLimit.from_vram("wide", context_window=262_144, weight_gb=2.0, kv_mb_per_1k=100.0)
    assert wide.binding == "vram"
    assert wide.reader_capacity_tokens == int(free_mb / 100.0 * 1000.0)
    tight = ReaderLimit.from_window("tight", 8_192)
    assert tight.binding == "window"
    assert tight.reader_capacity_tokens == 8_192


def test_absolute_budget_is_taken_as_tokens() -> None:
    assert budget_tokens(4096.0, "absolute_tokens", None) == 4096.0


def test_corpus_fraction_is_resolved_against_the_corpus() -> None:
    assert budget_tokens(0.25, "corpus_fraction", 16_000.0) == 4_000.0


def test_corpus_fraction_without_a_corpus_size_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError, match="corpus_tokens"):
        budget_tokens(0.25, "corpus_fraction", None)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_corpus_fraction_outside_zero_to_one_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        budget_tokens(bad, "corpus_fraction", 16_000.0)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown b_mode"):
        budget_tokens(1.0, "furlongs", None)


def test_budget_caps_the_prompt_and_k_only_fills_it() -> None:
    anchors = PRICING["token_estimates"]["M"]
    at64 = prompt_tokens(64, 4096.0, anchors, 500.0)
    assert at64 == 4096.0 + 500.0

    at1 = prompt_tokens(1, 4096.0, anchors, 500.0)
    assert at1 == 2000.0 + 500.0


def test_raising_k_past_the_budget_does_not_raise_the_prompt() -> None:
    anchors = PRICING["token_estimates"]["M"]
    sizes = [prompt_tokens(k, 4096.0, anchors, 500.0) for k in (8, 16, 32, 64)]
    assert sizes == [4096.0 + 500.0] * 4


def test_overhead_is_added_even_when_the_budget_does_not_bind() -> None:
    anchors = PRICING["token_estimates"]["M"]
    assert prompt_tokens(1, 4096.0, anchors, 500.0) == 2500.0


def test_none_budget_returns_the_uncapped_content() -> None:
    anchors = PRICING["token_estimates"]["M"]
    assert prompt_tokens(1, None, anchors, 0.0) == 2000.0


def _grid(readers: tuple[str, ...] = ("a",)) -> list[Cell]:
    return build_cells(
        [1, 2, 4, 8],
        [(4096, "absolute_tokens"), (65536, "absolute_tokens")],
        ["M"],
        list(readers),
    )


def test_budget_inert_below_the_cutoff_collapses_b_levels() -> None:
    cells = _grid()
    report = duplicate_conditions(cells, PRICING)
    k1 = [d for d in report["duplicate_sets"] if all(m["K"] == 1 for m in d["members"])]
    assert k1, "K=1 should collapse its two B levels into one condition"


def test_budget_binds_and_all_k_past_the_cutoff_collapse() -> None:
    report = duplicate_conditions(_grid(), PRICING)
    big = [d for d in report["duplicate_sets"] if d["n_members"] >= 2]
    assert big
    assert report["n_redundant_cells"] > 0


def test_readers_are_not_collapsed_into_each_other() -> None:
    one = duplicate_conditions(_grid(("a",)), PRICING)
    two = duplicate_conditions(_grid(("a", "b")), PRICING)
    assert two["n_cells"] == 2 * one["n_cells"]
    assert two["n_unique_conditions"] == 2 * one["n_unique_conditions"]
    assert two["n_redundant_cells"] == 2 * one["n_redundant_cells"]


def test_a_grid_with_no_budget_effect_reports_no_duplicates() -> None:
    pricing = {
        **PRICING,
        "token_estimates": {
            s: {**v, "prompt_tokens_at_K64": 500.0, "prompt_tokens_at_K1": 100.0}
            for s, v in PRICING["token_estimates"].items()
        },
    }
    cells = build_cells([1, 2], [(10_000_000, "absolute_tokens")], ["M"], ["a"])
    report = duplicate_conditions(cells, pricing)
    assert report["n_redundant_cells"] == 0


MEASURED = {
    "prompt_overhead_tokens": 500,
    "token_estimates": {
        "S": {
            "sessions_mean": 48.0,
            "tokens_per_session": 2500.0,
            "corpus_tokens": 120_000.0,
            "prompt_tokens_at_K1": 2500.0,
            "prompt_tokens_at_K64": 120_000.0,
        },
        "M": {
            "sessions_mean": 480.0,
            "tokens_per_session": 2500.0,
            "corpus_tokens": 1_200_000.0,
            "prompt_tokens_at_K1": 2500.0,
            "prompt_tokens_at_K64": 160_000.0,
        },
    },
}


def test_per_memory_budget_scales_with_k() -> None:
    a = MEASURED["token_estimates"]["M"]
    assert prompt_tokens(1, 512.0, a, 0.0, b_mode="per_memory_tokens") == 512.0
    assert prompt_tokens(2, 512.0, a, 0.0, b_mode="per_memory_tokens") == 1024.0
    assert prompt_tokens(8, 512.0, a, 0.0, b_mode="per_memory_tokens") == 4096.0
    for k in (1, 2, 4, 8):
        assert prompt_tokens(2 * k, 512.0, a, 0.0, b_mode="per_memory_tokens") == pytest.approx(
            2 * prompt_tokens(k, 512.0, a, 0.0, b_mode="per_memory_tokens")
        )


def test_per_memory_budget_changes_density_not_count() -> None:
    a = MEASURED["token_estimates"]["M"]
    k = 8
    small = prompt_tokens(k, 512.0, a, 0.0, b_mode="per_memory_tokens")
    large = prompt_tokens(k, 1024.0, a, 0.0, b_mode="per_memory_tokens")
    assert small == 8 * 512
    assert large == 8 * 1024
    assert small != large


def test_per_memory_budget_is_capped_by_the_mean_memory_length() -> None:
    a = MEASURED["token_estimates"]["M"]
    assert prompt_tokens(4, 9_000.0, a, 0.0, b_mode="per_memory_tokens") == 4 * 2500.0


def test_per_memory_budget_cannot_retrieve_more_than_the_corpus_holds() -> None:
    a = MEASURED["token_estimates"]["S"]
    assert prompt_tokens(64, 512.0, a, 0.0, b_mode="per_memory_tokens") == 48 * 512
    assert prompt_tokens(32, 512.0, a, 0.0, b_mode="per_memory_tokens") == 32 * 512


def test_per_memory_budget_needs_measured_corpus_values() -> None:
    with pytest.raises(ValueError, match="sessions_mean"):
        prompt_tokens(4, 512.0, PRICING["token_estimates"]["S"], 0.0, b_mode="per_memory_tokens")


def test_budget_tokens_refuses_per_memory_mode() -> None:
    with pytest.raises(ValueError, match="per_memory"):
        budget_tokens(512.0, "per_memory_tokens", None)


def test_a_per_memory_grid_has_no_staircase() -> None:
    cells = build_cells(
        [1, 2, 4, 8, 16, 32, 64],
        [(512, "per_memory_tokens"), (1024, "per_memory_tokens"), (2048, "per_memory_tokens")],
        ["M"],
        ["a"],
    )
    report = duplicate_conditions(cells, MEASURED)
    assert report["n_redundant_cells"] == 0
    assert report["n_unique_conditions"] == len(cells)


def test_the_same_grid_under_absolute_tokens_is_degenerate() -> None:
    cells = build_cells(
        [1, 2, 4, 8, 16, 32, 64],
        [(512, "absolute_tokens"), (1024, "absolute_tokens"), (2048, "absolute_tokens")],
        ["M"],
        ["a"],
    )
    report = duplicate_conditions(cells, MEASURED)
    assert report["n_redundant_cells"] == 18
    assert report["n_unique_conditions"] == 3


def test_infeasible_cells_name_the_ceiling_that_rejected_them() -> None:
    cells = _grid()
    limits = {"a": ReaderLimit.from_window("a", 3_000)}
    report = context_feasibility(cells, limits, PRICING)

    assert report["n_infeasible"] > 0
    for rows in report["infeasible_cells"].values():
        for row in rows:
            assert row["blocked_by"] in {"vram", "window"}
            assert row["prompt_tokens"] > row["reader_capacity_tokens"]
            assert row["overflow_by"] == row["prompt_tokens"] - row["reader_capacity_tokens"]


def test_a_vram_limit_rejects_what_a_window_limit_would_have_allowed() -> None:
    cells = _grid()
    window_only = context_feasibility(cells, {"a": ReaderLimit.from_window("a", 262_144)}, PRICING)
    assert window_only["n_infeasible"] == 0

    memory_aware = context_feasibility(
        cells,
        {
            "a": ReaderLimit.from_vram(
                "a", context_window=262_144, weight_gb=2.0, kv_mb_per_1k=1000.0
            )
        },
        PRICING,
    )
    assert memory_aware["n_infeasible"] > 0
    assert memory_aware["readers"]["a"]["blocked_by"] == "vram"
