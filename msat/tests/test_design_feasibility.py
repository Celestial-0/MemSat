"""Tests for prompt sizing, feasibility, and grid degeneracy.

Every number here is hand-computable on purpose. These functions decide which
cells exist and what they cost, so a test that needed the implementation to
compute its own expected value would be a tautology.

The cases that matter are the ones that were WRONG before 2026-10-13:

  - `B` was ignored entirely, so every prompt was sized from `K` alone
  - `corpus_fraction` was declared in the schema and never implemented
  - `context_window` was checked, but not the VRAM ceiling that actually binds
  - two cells could render byte-identical prompts while `Cell.key` said otherwise
"""

from __future__ import annotations

import pytest

from msat.design import (
    Cell,
    ReaderLimit,
    _budget_tokens,
    _prompt_tokens,
    build_cells,
    context_feasibility,
    duplicate_conditions,
    vram_max_prompt_tokens,
)

# Two scales with deliberately simple anchors so the arithmetic is obvious.
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


# ── VRAM arithmetic ───────────────────────────────────────────────────────────
def test_vram_capacity_is_plain_subtraction_and_division() -> None:
    # 8 GB card, 2 GB weights, 0.5 GB overhead -> 5.5 GB = 5632 MB free.
    # At 100 MB per 1k tokens that is 56.32k tokens.
    cap = vram_max_prompt_tokens(weight_gb=2.0, kv_mb_per_1k=100.0)
    assert cap == int(5632.0 / 100.0 * 1000.0) == 56_320


def test_weights_alone_that_do_not_fit_yield_zero_not_a_huge_number() -> None:
    # A reader that cannot be resident is not "slow" - it cannot run. Returning a
    # large positive capacity here would silently mark its cells feasible.
    assert vram_max_prompt_tokens(weight_gb=30.0, kv_mb_per_1k=50.0) == 0


def test_vram_capacity_rejects_a_nonsense_kv_rate() -> None:
    with pytest.raises(ValueError, match="kv_mb_per_1k"):
        vram_max_prompt_tokens(weight_gb=2.0, kv_mb_per_1k=0.0)


def test_binding_ceiling_is_the_smaller_of_window_and_vram() -> None:
    wide = ReaderLimit.from_vram("wide", context_window=262_144, weight_gb=2.0, kv_mb_per_1k=100.0)
    assert wide.binding == "vram"
    assert wide.ceiling == 56_320

    # A small window on a huge card: the window binds, not the memory.
    tight = ReaderLimit.from_window("tight", 8_192)
    assert tight.binding == "window"
    assert tight.ceiling == 8_192


# ── the budget, and the mode it travels with ──────────────────────────────────
def test_absolute_budget_is_taken_as_tokens() -> None:
    assert _budget_tokens(4096.0, "absolute_tokens", None) == 4096.0


def test_corpus_fraction_is_resolved_against_the_corpus() -> None:
    # 0.25 of a 16,000-token corpus is 4,000 tokens.
    assert _budget_tokens(0.25, "corpus_fraction", 16_000.0) == 4_000.0


def test_corpus_fraction_without_a_corpus_size_is_an_error_not_a_guess() -> None:
    # It used to be read as a token count, so 0.02 became a 0.02-token prompt and
    # the cell rendered as the bare question. A declared-but-dead option is worse
    # than a missing one because the schema says it is valid.
    with pytest.raises(ValueError, match="corpus_tokens"):
        _budget_tokens(0.25, "corpus_fraction", None)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_corpus_fraction_outside_zero_to_one_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _budget_tokens(bad, "corpus_fraction", 16_000.0)


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown b_mode"):
        _budget_tokens(1.0, "furlongs", None)


# ── prompt sizing: the bug that was actually live ─────────────────────────────
def test_budget_caps_the_prompt_and_k_only_fills_it() -> None:
    anchors = PRICING["token_estimates"]["M"]
    # At K=64 the content is 16,000 tokens. A 4,096 budget truncates to 4,096 + 500.
    at64 = _prompt_tokens(64, 4096.0, anchors, 500.0)
    assert at64 == 4096.0 + 500.0

    # At K=1 the content is 2,000 tokens, BELOW the budget, so B is inert.
    at1 = _prompt_tokens(1, 4096.0, anchors, 500.0)
    assert at1 == 2000.0 + 500.0


def test_raising_k_past_the_budget_does_not_raise_the_prompt() -> None:
    # This is the whole point of the fix: before it, B was ignored and every K
    # produced a larger prompt, which made B look like it always did work.
    anchors = PRICING["token_estimates"]["M"]
    sizes = [_prompt_tokens(k, 4096.0, anchors, 500.0) for k in (8, 16, 32, 64)]
    assert sizes == [4096.0 + 500.0] * 4


def test_overhead_is_added_even_when_the_budget_does_not_bind() -> None:
    anchors = PRICING["token_estimates"]["M"]
    assert _prompt_tokens(1, 4096.0, anchors, 500.0) == 2500.0


def test_none_budget_returns_the_uncapped_content() -> None:
    anchors = PRICING["token_estimates"]["M"]
    assert _prompt_tokens(1, None, anchors, 0.0) == 2000.0


# ── degenerate conditions ─────────────────────────────────────────────────────
def _grid(readers: tuple[str, ...] = ("a",)) -> list[Cell]:
    return build_cells(
        [1, 2, 4, 8],
        [(4096, "absolute_tokens"), (65536, "absolute_tokens")],
        ["M"],
        list(readers),
    )


def test_budget_inert_below_the_cutoff_collapses_b_levels() -> None:
    # At K=1 the content is 2,000 tokens, under both budgets, so 4096 and 65536
    # render the same prompt. Two declared levels, one condition.
    cells = _grid()
    report = duplicate_conditions(cells, PRICING)
    k1 = [d for d in report["duplicate_sets"] if all(m["K"] == 1 for m in d["members"])]
    assert k1, "K=1 should collapse its two B levels into one condition"


def test_budget_binds_and_all_k_past_the_cutoff_collapse() -> None:
    # content(K) doubles per level from 2,000, so K=4..8 exceed 4,096 and every
    # one of them truncates to the same 4,096-token prefix.
    report = duplicate_conditions(_grid(), PRICING)
    big = [d for d in report["duplicate_sets"] if d["n_members"] >= 2]
    assert big
    assert report["n_redundant_cells"] > 0


def test_readers_are_not_collapsed_into_each_other() -> None:
    # Different readers render the same prompt but are DIFFERENT conditions - the
    # reader is the treatment. Collapsing them understates the design and makes
    # the degeneracy look worse than it is.
    one = duplicate_conditions(_grid(("a",)), PRICING)
    two = duplicate_conditions(_grid(("a", "b")), PRICING)
    assert two["n_cells"] == 2 * one["n_cells"]
    assert two["n_unique_conditions"] == 2 * one["n_unique_conditions"]
    assert two["n_redundant_cells"] == 2 * one["n_redundant_cells"]


def test_a_grid_with_no_budget_effect_reports_no_duplicates() -> None:
    # B far above every prompt, so B never truncates and never differs.
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


# ── per-memory budgets: the mode that makes the interaction estimable ─────────
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
    """The property the whole redesign rests on: prompt = K x B, so K and B are
    independent. Under a truncating budget, doubling K could not change the prompt
    once the budget bound."""
    a = MEASURED["token_estimates"]["M"]
    assert _prompt_tokens(1, 512.0, a, 0.0, b_mode="per_memory_tokens") == 512.0
    assert _prompt_tokens(2, 512.0, a, 0.0, b_mode="per_memory_tokens") == 1024.0
    assert _prompt_tokens(8, 512.0, a, 0.0, b_mode="per_memory_tokens") == 4096.0
    # Doubling K doubles the prompt at a fixed budget.
    for k in (1, 2, 4, 8):
        assert _prompt_tokens(2 * k, 512.0, a, 0.0, b_mode="per_memory_tokens") == pytest.approx(
            2 * _prompt_tokens(k, 512.0, a, 0.0, b_mode="per_memory_tokens")
        )


def test_per_memory_budget_changes_density_not_count() -> None:
    a = MEASURED["token_estimates"]["M"]
    k = 8
    small = _prompt_tokens(k, 512.0, a, 0.0, b_mode="per_memory_tokens")
    large = _prompt_tokens(k, 1024.0, a, 0.0, b_mode="per_memory_tokens")
    assert small == 8 * 512
    assert large == 8 * 1024
    assert small != large


def test_per_memory_budget_is_capped_by_the_mean_memory_length() -> None:
    """A budget at or above the mean memory length is a no-op - truncating a
    memory to its own length changes nothing. This is why the grid's B values sit
    below ~2,550."""
    a = MEASURED["token_estimates"]["M"]
    assert _prompt_tokens(4, 9_000.0, a, 0.0, b_mode="per_memory_tokens") == 4 * 2500.0


def test_per_memory_budget_cannot_retrieve_more_than_the_corpus_holds() -> None:
    """S has 48 sessions. Asking for K=64 gives 48 memories, not 64 - and that is
    a real duplicate condition, not a rounding detail."""
    a = MEASURED["token_estimates"]["S"]
    assert _prompt_tokens(64, 512.0, a, 0.0, b_mode="per_memory_tokens") == 48 * 512
    assert _prompt_tokens(32, 512.0, a, 0.0, b_mode="per_memory_tokens") == 32 * 512


def test_per_memory_budget_needs_measured_corpus_values() -> None:
    with pytest.raises(ValueError, match="sessions_mean"):
        _prompt_tokens(4, 512.0, PRICING["token_estimates"]["S"], 0.0, b_mode="per_memory_tokens")


def test_budget_tokens_refuses_per_memory_mode() -> None:
    """per_memory is resolved per memory, so asking for one total budget is a
    category error and says so rather than returning a plausible wrong number."""
    with pytest.raises(ValueError, match="per_memory"):
        _budget_tokens(512.0, "per_memory_tokens", None)


def test_a_per_memory_grid_has_no_staircase() -> None:
    """Every (K, B) pair must give a distinct prompt - the property the
    `absolute_tokens` grid provably cannot have at K=1 and K=2."""
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
    """The counterfactual, kept so the redesign cannot be quietly reverted.

    512/1024/2048 as ABSOLUTE totals all sit below content(K=1) = 2,500, so every
    cell truncates to its own budget and the budget alone decides the prompt.
    Retrieval depth K stops existing entirely: all 7 depths collapse into one
    condition at each B, giving 18 redundant cells out of 21.

    That is the degenerate extreme of the staircase. The shipped grid avoided it
    only by accident - its old values happened to straddle content(K) - and even
    then 78 of 126 cells were duplicates.
    """
    cells = build_cells(
        [1, 2, 4, 8, 16, 32, 64],
        [(512, "absolute_tokens"), (1024, "absolute_tokens"), (2048, "absolute_tokens")],
        ["M"],
        ["a"],
    )
    report = duplicate_conditions(cells, MEASURED)
    assert report["n_redundant_cells"] == 18  # 6 per B, across 3 B levels
    assert report["n_unique_conditions"] == 3  # one per budget, depth irrelevant


# ── feasibility reports which ceiling binds ───────────────────────────────────
def test_infeasible_cells_name_the_ceiling_that_rejected_them() -> None:
    cells = _grid()
    limits = {"a": ReaderLimit.from_window("a", 3_000)}
    report = context_feasibility(cells, limits, PRICING)

    assert report["n_infeasible"] > 0
    for rows in report["infeasible_cells"].values():
        for row in rows:
            assert row["blocked_by"] in {"vram", "window"}
            assert row["prompt_tokens"] > row["ceiling"]
            assert row["overflow_by"] == row["prompt_tokens"] - row["ceiling"]


def test_a_vram_limit_rejects_what_a_window_limit_would_have_allowed() -> None:
    """The failure mode that sent 15 cells to P5 before this was caught.

    The checkpoint declares a 262k window, so a window-only check calls every
    cell feasible. The 8 GB card then cannot hold the KV cache.
    """
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
