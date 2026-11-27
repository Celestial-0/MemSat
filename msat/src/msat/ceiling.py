"""The analytical recall ceiling and the corpus-fraction covariate."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from msat.datasets.schema import Question

CEILING_GRID: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

SATURATION_TOL: Final = 0.005

FLOAT_DP: Final = 6


def _r(value: float) -> float:
    # `+ 0.0` turns -0.0 into 0.0 so canonical bytes stay stable.
    return round(float(value) + 0.0, FLOAT_DP)


def _key(k: int) -> str:
    return str(k)


def ceiling_curve(n_gold: Sequence[int], grid: Sequence[int] = CEILING_GRID) -> dict[str, float]:
    if not n_gold:
        return {}
    n = len(n_gold)
    return {_key(k): _r(sum(1 for g in n_gold if g <= k) / n) for k in grid}


def partial_credit_curve(
    n_gold: Sequence[int], grid: Sequence[int] = CEILING_GRID
) -> dict[str, float]:
    if not n_gold:
        return {}
    n = len(n_gold)
    out: dict[str, float] = {}
    for k in grid:
        total = sum(min(g, k) / g for g in n_gold if g > 0)
        out[_key(k)] = _r(total / n)
    return out


def saturation_k(ceiling: dict[str, float], tol: float = SATURATION_TOL) -> int | None:
    threshold = 1.0 - tol
    for k in sorted(int(key) for key in ceiling):
        if ceiling[_key(k)] >= threshold:
            return k
    return None


def survivors_at(n_sessions: Sequence[int], grid: Sequence[int] = CEILING_GRID) -> dict[str, float]:
    if not n_sessions:
        return {}
    n = len(n_sessions)
    return {_key(k): _r(sum(1 for s in n_sessions if s >= k) / n) for k in grid}


def degeneracy_fraction(n_sessions: Sequence[int], k: int) -> float:
    if not n_sessions:
        return 0.0
    return _r(sum(1 for s in n_sessions if k >= s) / len(n_sessions))


def by_type_table(
    rows: Iterable[tuple[str, int]], grid: Sequence[int] = CEILING_GRID
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for ability, n_gold in rows:
        buckets[ability].append(n_gold)

    probe = (1, 2, 4)
    out: dict[str, dict[str, Any]] = {}
    for ability in sorted(buckets):
        values = buckets[ability]
        n = len(values)
        curve = ceiling_curve(values, grid)
        at1 = curve.get(_key(1), 0.0)
        if at1 >= 1.0:
            kind = "retriever-independent"
        elif at1 <= 0.0:
            kind = "retriever-limited"
        else:
            kind = "mixed"
        out[ability] = {
            "n": n,
            "mean_gold_sessions": _r(statistics.mean(values)),
            "max_gold_sessions": max(values),
            "ceiling": curve,
            "probe": {_key(k): ceiling_curve(values, (k,)).get(_key(k), 0.0) for k in probe},
            "classification": kind,
        }
    return out


def corpus_fraction(
    n_sessions: Sequence[int], grid: Sequence[int] = CEILING_GRID
) -> dict[str, dict[str, Any]]:
    if not n_sessions:
        return {}
    out: dict[str, dict[str, Any]] = {}
    n = len(n_sessions)
    for k in grid:
        ratios = sorted(k / s for s in n_sessions if s > 0)
        out[_key(k)] = {
            "frac_full_context": _r(sum(1 for r in ratios if r == 1.0) / n),
            "frac_k_gt_corpus": _r(sum(1 for r in ratios if r > 1.0) / n),
            "quantiles": {
                "min": _r(ratios[0]),
                "p10": _r(ratios[max(0, int(0.10 * n) - 1)]),
                "p25": _r(ratios[max(0, int(0.25 * n) - 1)]),
                "p50": _r(ratios[max(0, int(0.50 * n) - 1)]),
                "p75": _r(ratios[max(0, int(0.75 * n) - 1)]),
                "p90": _r(ratios[max(0, int(0.90 * n) - 1)]),
                "max": _r(ratios[-1]),
            },
        }
    return out


_KEY_SEMANTICS: Final[dict[str, str]] = {
    "ceiling": "Pr(|answer_session_ids| <= K) — max recall@K for a PERFECT retriever.",
    "partial": "Mean fraction of evidence retrievable within K.",
    "survivors_at": "Fraction of questions whose corpus has >= K sessions. "
    "NOT 'fraction degenerate' — the complement is, and is emitted as degeneracy_fraction.",
    "degeneracy_fraction[K]": "Fraction of questions with K >= their own corpus size. "
    "Degenerate = the condition has silently become full-context.",
    "analytic_saturation_k": "K* = min{K : ceiling(K) >= 0.995}. Beyond it, no retriever "
    "can add evidence; all change is reader-side.",
    "corpus_fraction_ratio": "K / |corpus| per question. 1.0 means full-context.",
}


def build_ceiling_artifact(
    questions: Sequence[Question], grid: Sequence[int] = CEILING_GRID
) -> dict[str, Any]:
    answerable = [q for q in questions if q.answerable]
    abstention = [q for q in questions if not q.answerable]
    n_gold = [q.corpus.n_gold_sessions for q in answerable]
    sizes = [q.corpus.n_sessions for q in questions]

    ceiling = ceiling_curve(n_gold, grid)
    artifact: dict[str, Any] = {
        "_key_semantics": dict(_KEY_SEMANTICS),
        "_provenance": {
            "grid": list(grid),
            "saturation_tol": SATURATION_TOL,
            "float_dp": FLOAT_DP,
            "source": "answer_session_ids + has_answer, from the cleaned official release",
            "api_calls": 0,
        },
        "n_questions": len(questions),
        "n_answerable": len(answerable),
        "n_abstention": len(abstention),
        "ceiling": ceiling,
        "partial": partial_credit_curve(n_gold, grid),
        "analytic_saturation_k": saturation_k(ceiling),
        "corpus_size": {
            "n_sessions_min": min(sizes) if sizes else 0,
            "n_sessions_max": max(sizes) if sizes else 0,
            "n_sessions_mean": _r(statistics.mean(sizes)) if sizes else 0.0,
            "n_sessions_median": _r(statistics.median(sizes)) if sizes else 0.0,
            "unit": "session",
        },
        "survivors_at": survivors_at(sizes, grid),
        "degeneracy_fraction": {_key(k): degeneracy_fraction(sizes, k) for k in grid},
        "by_type": by_type_table(((q.ability, q.corpus.n_gold_sessions) for q in answerable), grid),
        "question_types_seen": sorted({q.ability for q in answerable}),
    }
    return artifact


def build_corpus_fraction_artifact(
    by_scale: Mapping[str, Sequence[Question]], grid: Sequence[int] = CEILING_GRID
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "_key_semantics": dict(_KEY_SEMANTICS),
        "_provenance": {
            "grid": list(grid),
            "ratio": "K / |corpus| where |corpus| = n_sessions for that question",
            "float_dp": FLOAT_DP,
            "api_calls": 0,
        },
        "scales": {},
    }
    for scale in sorted(by_scale):
        qs = by_scale[scale]
        sizes = [q.corpus.n_sessions for q in qs]
        out["scales"][scale] = {
            "n_questions": len(qs),
            "corpus_size": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": _r(statistics.mean(sizes)) if sizes else 0.0,
            },
            "per_k": corpus_fraction(sizes, grid),
        }

    if len(out["scales"]) > 1:
        out["_cross_scale_full_context"] = {
            _key(k): {
                scale: block["per_k"].get(_key(k), {}).get("frac_full_context")
                for scale, block in out["scales"].items()
            }
            for k in grid
        }
    return out


def build_ceiling_multiscale(
    by_scale: Mapping[str, Sequence[Question]], grid: Sequence[int] = CEILING_GRID
) -> dict[str, Any]:
    per_scale: dict[str, Any] = {}
    for scale in sorted(by_scale):
        per_scale[scale] = build_ceiling_artifact(by_scale[scale], grid)

    artifact: dict[str, Any] = {
        "_key_semantics": dict(_KEY_SEMANTICS),
        "_provenance": {
            "grid": list(grid),
            "saturation_tol": SATURATION_TOL,
            "float_dp": FLOAT_DP,
            "source": "answer_session_ids + has_answer, from the cleaned official release",
            "api_calls": 0,
            "generator": "msat ceiling",
        },
        "scales": per_scale,
    }

    if len(per_scale) > 1:
        keys = sorted(next(iter(per_scale.values()))["ceiling"])
        artifact["scale_invariance"] = {
            "ceiling_by_scale": {
                scale: {k: per_scale[scale]["ceiling"][k] for k in keys}
                for scale in sorted(per_scale)
            },
            "identical_across_scales": all(
                len({per_scale[s]["ceiling"][k] for s in per_scale}) == 1 for k in keys
            ),
            "analytic_saturation_k_by_scale": {
                s: per_scale[s]["analytic_saturation_k"] for s in sorted(per_scale)
            },
        }
    return artifact


def build_dataset_report(by_scale: Mapping[str, Sequence[Question]]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "_semantics": {
            "n_records": "top-level records streamed",
            "sessions_per_question": "per-question |haystack_session_ids| — the K-guard's input",
            "n_gold_sessions": "|answer_session_ids|",
            "tokens": "NOT REPORTED. Requires the tokenizer pinned in "
            "configs/retrieval.yaml (Phase 2). Only exact character counts are "
            "emitted — never an estimated token count presented as a measurement.",
        },
        "scales": {},
    }
    for scale in sorted(by_scale):
        qs = by_scale[scale]
        sizes = [q.corpus.n_sessions for q in qs]
        gold = [q.corpus.n_gold_sessions for q in qs]
        chars = [q.corpus.n_chars for q in qs]
        report["scales"][scale] = {
            "n_records": len(qs),
            "n_answerable": sum(q.answerable for q in qs),
            "n_abstention": sum(not q.answerable for q in qs),
            "sessions_per_question": {
                "min": min(sizes) if sizes else 0,
                "max": max(sizes) if sizes else 0,
                "mean": _r(statistics.mean(sizes)) if sizes else 0.0,
                "median": _r(statistics.median(sizes)) if sizes else 0.0,
            },
            "gold_sessions": {
                "max": max(gold) if gold else 0,
                "mean": _r(statistics.mean(gold)) if gold else 0.0,
                "distribution": {str(k): gold.count(k) for k in sorted(set(gold))},
            },
            "corpus_chars": {
                "total": sum(chars),
                "mean_per_question": int(statistics.mean(chars)) if chars else 0,
            },
            "question_types": _count_types(qs),
        }
    return report


def _count_types(questions: Sequence[Question]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for q in questions:
        counts[q.ability] += 1
    return dict(sorted(counts.items()))


def dump_canonical(artifact: dict[str, Any], path: Path) -> bytes:
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    data = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def verify_artifact(artifact: dict[str, Any], committed: Path) -> tuple[bool, str]:
    if not committed.is_file():
        return False, f"committed artifact missing: {committed}"
    regenerated = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    on_disk = committed.read_bytes()
    if regenerated == on_disk:
        return True, f"identical ({len(on_disk)} bytes)"
    detail = _first_difference(on_disk, regenerated)
    return False, (
        f"DRIFT in {committed}: regenerated {len(regenerated)} bytes vs committed "
        f"{len(on_disk)} bytes; {detail}"
    )


def _first_difference(a: bytes, b: bytes) -> str:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            lo = max(0, i - 60)
            return (
                f"first difference at byte {i}: "
                f"committed {a[lo : i + 60]!r} vs regenerated {b[lo : i + 60]!r}"
            )
    return f"one is a prefix of the other (lengths {len(a)} vs {len(b)})"


def is_finite_number(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)
