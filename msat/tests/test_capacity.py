"""Tests for Recommendation 02: Deep Reader Capacity Module.

Verifies:
- Token modeling (per_memory_tokens, absolute_tokens)
- Limits across cards and headroom
- Cell verdicts and support (equality accepted, limit + 1 rejected)
- Qwen on 8 GB rejects S/K32/B512 prompt estimate (16,884 > 2,682)
- Panel coverage agrees with preflight verdict basis
- check_capacity and coverage_over honour card_gb override
- Missing reader and token anchors fail explicitly
- Modeled vs rendered distinction preserved (no silent truncation)
- D11 headroom remains default 3.0 GB (no 0.5 GB rollback)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from msat.config import ConfigBundle, load_bundle
from msat.hardware.capacity import (
    MissingAnchorError,
    MissingReaderError,
    check_cell_capacity,
    evaluate_cell_capacity,
    evaluate_panel_capacity,
    model_prompt_tokens,
    preflight_capacity_report,
)
from msat.hardware.vram import DEFAULT_HEADROOM_GB
from msat.planning import PlanCondition
from msat.providers.local import (
    ContextOverflowError,
    LocalReader,
    load_readers,
)
from msat.store import Condition

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_coverage_requires_explicit_prompt_token_basis(readers: list[LocalReader]) -> None:
    with pytest.raises(MissingAnchorError, match="Coverage requires"):
        readers[0].coverage_over([(32, 512.0, "S")])
    with pytest.raises(MissingAnchorError, match="No prompt-token basis"):
        readers[0].coverage_over([(32, 512.0, "S")], token_basis={})


def test_panel_reports_card_and_headroom_override_consistently(
    bundle: ConfigBundle, readers: list[LocalReader]
) -> None:
    qwen = next(r for r in readers if r.id == "qwen")
    cell = PlanCondition(Condition("A+G", 32, 512.0, "per_memory_tokens", "S", "qwen"), 1, 0.0)

    class Plan:
        cells = (cell,)

    report = preflight_capacity_report(bundle, Plan(), panel=[qwen], card_gb=24.0, headroom_gb=0.8)
    assert report["n_unsupported"] == 0
    reader_report = report["per_reader"]["qwen"]
    coverage_report = report["panel_coverage"]["readers"][0]
    assert reader_report["reader_capacity_tokens"] == coverage_report["reader_capacity_tokens"]
    assert reader_report["vram_capacity_tokens"] > qwen.reader_capacity_tokens


READERS_YAML = CONFIGS / "readers.yaml"


@pytest.fixture
def bundle() -> ConfigBundle:
    return load_bundle(CONFIGS)


@pytest.fixture
def readers() -> list[LocalReader]:
    return load_readers(READERS_YAML, card_gb=8.0)


def test_d11_headroom_remains_configured_default(readers: list[LocalReader]) -> None:
    """Headroom remains configured default (3.0 GB), no 0.5 GB rollback."""
    assert DEFAULT_HEADROOM_GB == 3.0
    for r in readers:
        assert r.headroom_gb == DEFAULT_HEADROOM_GB


def test_token_modeling_and_qwen_rejection_at_s_k32_b512(
    bundle: ConfigBundle, readers: list[LocalReader]
) -> None:
    """S/K32/B512 prompt estimate is 16,884 tokens and exceeds Qwen's 2,682-token limit on 8 GB."""
    qwen = next(r for r in readers if r.id == "qwen")
    anchors = bundle.pricing["token_estimates"]["S"]

    # Model prompt tokens: 32 * min(2563, 512) + 500 = 16,884
    modeled_prompt = model_prompt_tokens(
        32,
        512.0,
        anchors,
        overhead=500.0,
        b_mode="per_memory_tokens",
    )
    assert modeled_prompt == 16_884

    # Qwen limit on 8 GB
    limit = qwen.reader_capacity_tokens
    assert limit < 3_000
    assert modeled_prompt > limit

    # Capacity check verdict
    verdict = check_cell_capacity(
        qwen,
        modeled_prompt,
        scale="S",
        k=32,
        b=512.0,
        b_mode="per_memory_tokens",
        card_gb=8.0,
    )
    assert verdict.supported is False
    assert verdict.status == "unsupported"
    assert verdict.reader_capacity_tokens == limit
    assert "exceeds" in verdict.reason.lower()


def test_equality_accepted_and_plus_one_rejected(readers: list[LocalReader]) -> None:
    """Equality (prompt == limit) is accepted; limit + 1 is rejected."""
    gemma = next(r for r in readers if r.id == "gemma")
    limit = gemma.reader_capacity_tokens

    # Exactly at limit: supported
    v_eq = check_cell_capacity(gemma, limit, card_gb=8.0)
    assert v_eq.supported is True
    assert v_eq.status == "supported"
    assert v_eq.reader_capacity_tokens == limit
    assert gemma.check_capacity(limit) == limit

    # Limit + 1: unsupported
    v_plus1 = check_cell_capacity(gemma, limit + 1, card_gb=8.0)
    assert v_plus1.supported is False
    assert v_plus1.status == "unsupported"
    assert v_plus1.reader_capacity_tokens == limit
    assert "exceeds" in v_plus1.reason.lower()
    with pytest.raises(ContextOverflowError):
        gemma.check_capacity(limit + 1)


def test_card_gb_override_in_check_capacity_and_reporting(readers: list[LocalReader]) -> None:
    """check_capacity card_gb override reports override AND compares against overridden limit."""
    gemma = next(r for r in readers if r.id == "gemma")
    limit_8gb = gemma.reader_capacity_tokens
    limit_24gb = gemma.vram_capacity_tokens(card_gb=24.0)

    assert limit_24gb > limit_8gb

    # A prompt exceeding 8 GB limit but within 24 GB limit succeeds with card_gb=24.0
    prompt_mid = limit_8gb + 1_000
    assert gemma.check_capacity(prompt_mid, card_gb=24.0) == gemma.declared_context_window

    # Exceeding the 24 GB limit raises error citing 24.0 GB and the overridden limit
    with pytest.raises(ContextOverflowError) as exc:
        gemma.check_capacity(gemma.declared_context_window + 10, card_gb=24.0)
    msg = str(exc.value)
    assert "24.0 GB" in msg
    assert f"{gemma.declared_context_window:,}" in msg


def test_coverage_over_honours_card_gb_and_token_basis(readers: list[LocalReader]) -> None:
    """LocalReader.coverage_over supports token_basis and card_gb override."""
    qwen = next(r for r in readers if r.id == "qwen")
    cells = [(32, 512.0, "S")]

    # With explicit token_basis of 16,884 tokens on 8 GB: unsupported
    rep_8gb = qwen.coverage_over(cells, card_gb=8.0, token_basis=lambda k, b, s: 16_884)
    assert rep_8gb["n_supported"] == 0
    assert rep_8gb["n_unsupported"] == 1

    # On a 24 GB card, Qwen ceiling is ~20k+ tokens; 16,884 tokens is supported!
    rep_24gb = qwen.coverage_over(cells, card_gb=24.0, token_basis=lambda k, b, s: 16_884)
    assert rep_24gb["n_supported"] == 1
    assert rep_24gb["n_unsupported"] == 0


def test_missing_anchors_fail_explicitly(bundle: ConfigBundle, readers: list[LocalReader]) -> None:
    """Missing reader or token anchors fail explicitly."""
    gemma = readers[0]

    # Missing scale in pricing
    with pytest.raises(MissingAnchorError):
        evaluate_cell_capacity(gemma, 1, 128.0, bundle.pricing, scale="NONEXISTENT")

    # Missing required anchors inside scale estimates
    bad_pricing = {"token_estimates": {"S": {"other_field": 123}}}
    with pytest.raises(MissingAnchorError):
        evaluate_cell_capacity(gemma, 1, 128.0, bad_pricing, scale="S")


def test_missing_reader_fails_explicitly(bundle: ConfigBundle, readers: list[LocalReader]) -> None:
    """Preflight check fails explicitly if a reader in the plan is not in the panel."""
    plan_cell = PlanCondition(
        Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "unknown_reader"), 1, 0.0
    )

    class DummyPlan:
        cells = (plan_cell,)

    with pytest.raises(MissingReaderError, match="unknown_reader"):
        preflight_capacity_report(bundle, DummyPlan(), panel=readers)


def test_modeled_vs_rendered_distinction(readers: list[LocalReader]) -> None:
    """CapacityVerdict preserves modeled vs rendered prompt token distinction."""
    gemma = readers[0]

    # Modeled verdict
    v_modeled = check_cell_capacity(gemma, 1000, is_rendered=False)
    assert v_modeled.modeled_prompt_tokens == 1000
    assert v_modeled.rendered_prompt_tokens is None

    # Rendered verdict
    v_rendered = check_cell_capacity(gemma, 1050, is_rendered=True)
    assert v_rendered.rendered_prompt_tokens == 1050
    assert v_rendered.modeled_prompt_tokens is None


def test_capacity_verdict_agreement(bundle: ConfigBundle, readers: list[LocalReader]) -> None:
    """Preflight capacity evaluation and panel coverage agree on verdicts."""
    cells = [
        PlanCondition(Condition("A+G", 1, 128.0, "per_memory_tokens", "S", "gemma"), 1, 0.0),
        PlanCondition(Condition("A+G", 32, 512.0, "per_memory_tokens", "S", "qwen"), 1, 0.0),
    ]

    class MockPlan:
        conditions = ()
        cells = ()

    mock_plan = MockPlan()
    mock_plan.conditions = tuple(cells)
    mock_plan.cells = tuple(cells)

    preflight = preflight_capacity_report(bundle, mock_plan, panel=readers, card_gb=8.0)
    cov = evaluate_panel_capacity(readers, mock_plan.cells, bundle.pricing, card_gb=8.0)

    # Preflight and coverage agree on verdicts
    assert preflight["n_supported"] == sum(cov["supported_per_reader"].values())
    assert preflight["n_unsupported"] == sum(row["n_unsupported"] for row in cov["readers"])
    assert preflight["n_supported"] == 1
    assert preflight["n_unsupported"] == 1
    assert preflight["unsupported"][0]["reader"] == "qwen"
    assert preflight["unsupported"][0]["K"] == 32


def test_capacity_verdict_distinguished_from_recall_ceiling() -> None:
    """Reader capacity hardware quantity is distinct from analytical recall ceiling."""
    from msat.ceiling import ceiling_curve
    from msat.hardware.capacity import CapacityVerdict, ReaderLimit

    # Analytical recall bound is a curve / probability bound in [0, 1]
    recall_bound = ceiling_curve([1, 2, 4], grid=(1, 2, 4))
    assert all(0.0 <= v <= 1.0 for v in recall_bound.values())

    # Reader capacity is a prompt-token count on hardware
    limit = ReaderLimit.from_window("gemma", 8192)
    assert limit.reader_capacity_tokens == 8192
    assert limit.reader_capacity_tokens > 1.0

    verdict = check_cell_capacity(limit, 4000)
    assert isinstance(verdict, CapacityVerdict)
    assert verdict.status == "supported"
    assert verdict.supported is True
    assert verdict.reader_capacity_tokens == 8192
