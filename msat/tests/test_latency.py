from __future__ import annotations

import pytest

from msat.metrics.latency import (
    LatencySample,
    latency_report,
    summarise_latency,
)


def _samples(values: list[float], *, queue: float = 0.0) -> list[LatencySample]:
    return [
        LatencySample(
            latency_ms=v,
            queue_wait_ms=queue,
            ttft_ms=queue + min(v, 200.0),
            server_time_ms=max(0.0, v - queue - min(v, 200.0)),
        )
        for v in values
    ]


def test_the_summary_reports_median_iqr_and_quantiles() -> None:
    s = summarise_latency(_samples([100, 200, 300, 400, 500]))
    assert s.quantiles["p50"] == pytest.approx(300)
    assert s.quantiles["p90"] == pytest.approx(460)
    assert s.iqr == pytest.approx(s.quantiles["p75"] - s.quantiles["p25"])


def test_every_quantile_r85_names_is_present() -> None:
    s = summarise_latency(_samples([100, 200, 300]))
    for q in ("p25", "p50", "p75", "p90", "p95", "p99"):
        assert q in s.quantiles, q


def test_there_is_no_bootstrap_ci_anywhere_in_the_summary() -> None:
    d = summarise_latency(_samples([100, 200, 300])).as_dict()
    text = " ".join(d).lower()
    for forbidden in ("ci", "confidence", "bootstrap", "stderr", "se_"):
        assert forbidden not in text, f"R8.5 forbids {forbidden!r} in a latency summary"


def test_no_bootstrap_interval_is_computed_anywhere_in_the_module() -> None:
    import inspect

    from msat.metrics import latency as mod

    src = inspect.getsource(mod).lower()
    assert (
        "bootstrap" not in src.replace("not a defensible", "").replace("a bootstrap ci", "")
        or "never" in src
        or "forbidden" in src
    )


def test_a_heavy_tail_is_declared_and_the_mean_is_flagged() -> None:
    values = [100.0] * 99 + [100_000.0]
    s = summarise_latency(_samples(values))
    assert s.skewed is True
    assert s.mean > 1000
    assert "not a representative latency" in s.caveat


def test_a_tight_distribution_is_not_declared_skewed() -> None:
    assert summarise_latency(_samples([100, 101, 99, 100])).skewed is False


def test_an_all_zero_run_is_reported_as_invalid_not_as_instant() -> None:
    s = summarise_latency([LatencySample(latency_ms=0.0), LatencySample(latency_ms=0.0)])
    assert s.n == 0
    assert s.n_invalid == 2
    assert "no valid latency samples" in s.caveat


def test_an_empty_sample_set_returns_placeholders_not_a_crash() -> None:
    s = summarise_latency([])
    assert s.n == 0
    assert s.as_dict()["quantiles_ms"] == {}


def test_a_p99_over_few_samples_is_declared_unreliable() -> None:
    assert summarise_latency(_samples([100.0] * 40)).p99_reliable is False
    assert summarise_latency(_samples([100.0] * 150)).p99_reliable is True


def test_ttft_cannot_be_below_the_queue_wait() -> None:
    with pytest.raises(ValueError, match="below queue_wait_ms"):
        LatencySample(latency_ms=500, queue_wait_ms=400, ttft_ms=100)


def test_queue_excluded_latency_is_the_readers_cost_not_our_scheduler() -> None:
    s = LatencySample(latency_ms=1000, queue_wait_ms=800, ttft_ms=900, server_time_ms=100)
    assert s.queue_excluded == pytest.approx(200)


def test_the_report_says_when_queue_wait_dominates() -> None:
    rep = latency_report(_samples([1000.0] * 10, queue=900.0))
    qw = rep["queue_wait"]
    assert qw["recorded"] is True
    assert qw["share_of_latency"] > 0.5
    assert qw["recording_concurrency"] is False
    assert "measures this harness" in qw["note"]


def test_the_report_says_when_concurrency_is_not_confounding() -> None:
    rep = latency_report(_samples([1000.0] * 10, queue=10.0))
    assert rep["queue_wait"]["recording_concurrency"] is True


def test_the_report_says_when_queue_wait_was_not_recorded() -> None:
    mixed = [
        LatencySample(latency_ms=100, queue_wait_ms=10, ttft_ms=50, server_time_ms=40),
        LatencySample(latency_ms=200),
    ]
    rep = latency_report(mixed)
    assert rep["queue_wait"]["recorded"] is False
    assert "cannot be separated" in rep["queue_wait"]["note"]


def test_the_report_carries_per_reader_summaries() -> None:
    rep = latency_report(
        _samples([100, 200]),
        by_label={"gemma": _samples([100, 200]), "qwen": _samples([900, 1000])},
    )
    assert set(rep["by_reader"]) == {"gemma", "qwen"}
    assert (
        rep["by_reader"]["qwen"]["quantiles_ms"]["p50"]
        > rep["by_reader"]["gemma"]["quantiles_ms"]["p50"]
    )


def test_the_quantile_interpolates_between_order_statistics() -> None:
    from msat.metrics.latency import quantile

    assert quantile(list(range(1, 11)), 0.5) == pytest.approx(5.5)
    assert quantile([7], 0.99) == 7
    assert quantile([], 0.5) == 0.0


def test_the_median_is_the_p50() -> None:
    values = [10, 20, 30, 40, 50, 60, 70]
    assert summarise_latency(_samples(values)).quantiles["p50"] == pytest.approx(40)


def test_negative_latency_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="negative latency"):
        LatencySample(latency_ms=-1.0)
