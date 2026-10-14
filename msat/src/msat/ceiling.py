"""The analytical recall ceiling, and the corpus-fraction covariate.

THIS IS THE PAPER'S ONLY ZERO-COST RESULT.

    ceiling(K) = Pr( |answer_session_ids| <= K )

The maximum `recall@K` that *any* retriever could achieve, computed from
benchmark metadata alone with **zero API calls**. It bounds the whole
accuracy-vs-K curve from above and locates the K at which retrieval must stop
mattering.

What it is not
--------------
It bounds a **perfect** retriever, not ours. EvalMem measures 22.1% retrieval
failure; the honest use is as a **bound drawn on a figure**, never as a shading
over the measured curve (PRD §R9.6). Shading would assume away the very
crossover this study measures.

Scale invariance
----------------
`ceiling(K)` is a property of *evidence structure*, not corpus size: on
LongMemEval it is identical at ~47.7 and ~475.3 sessions per conversation
(measured — see `results/ceiling.json`). That is what makes it a benchmark
diagnostic rather than an artefact of one corpus.

Type-conditionality
-------------------
The ceiling is strongly type-conditional, which is why stratified analysis is
primary and a pooled saturation point is a meaningless weighted average of two
regimes (PRD §R9.5).

Key naming (PRD §R2.5)
----------------------
`survivors_at[K]` is the fraction of questions whose corpus has **at least K**
sessions — the fraction for which K is a legal retrieval depth. The legacy
script called this `degenerate_at`, which names the *complement* and invites
exactly the misreading this project exists to prevent. The meaning is also
written into every artifact under `_key_semantics` so a reader cannot guess.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from msat.datasets.schema import Question

#: Reporting grid for the ceiling curve.
#:
#: PRD §R0.1 makes `configs/design.yaml` the single source of the *design* grid.
#: This is a different thing: the set of K at which we publish the analytic
#: bound, which must be a **superset** of the design grid so every experimental
#: cell has a ceiling to be plotted against. `tests/test_ceiling.py` asserts
#: `CEILING_GRID ⊇ design.yaml K_grid` — that test is what stops this from
#: becoming a second, competing grid.
#:
#: The 3-point spacing near 6 is deliberate: K* lands at 6, and a coarse grid
#: would round the single most-cited number in the paper.
CEILING_GRID: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

#: K* = min{ K : ceiling(K) >= 1 - SATURATION_TOL }. PRD §R2.1 specifies 0.995.
SATURATION_TOL: Final = 0.005

#: Decimal places every emitted float is rounded to. Byte-compare in
#: `ceiling --verify` is only meaningful if the writer is deterministic.
FLOAT_DP: Final = 6


def _r(value: float) -> float:
    """Round for a stable artifact. `+ 0.0` normalises `-0.0` to `0.0`."""
    return round(float(value) + 0.0, FLOAT_DP)


def _key(k: int) -> str:
    return str(k)


# ── the core curves ────────────────────────────────────────────────────────────
def ceiling_curve(n_gold: Sequence[int], grid: Sequence[int] = CEILING_GRID) -> dict[str, float]:
    """`Pr(n_gold <= K)` — the full-credit ceiling: all evidence retrievable within K."""
    if not n_gold:
        return {}
    n = len(n_gold)
    return {_key(k): _r(sum(1 for g in n_gold if g <= k) / n) for k in grid}


def partial_credit_curve(
    n_gold: Sequence[int], grid: Sequence[int] = CEILING_GRID
) -> dict[str, float]:
    """Mean fraction of evidence retrievable within K.

    PRD §R2.6 requires this to honour its `grid` argument. The legacy
    `context/scripts/ceiling_analysis.py` loops over `grid` and is therefore
    already correct on this point — the requirement was written against an
    earlier revision that returned only the last K. Rather than "fixing" a
    non-bug, `tests/test_ceiling.py::test_partial_credit_honours_its_grid` pins
    the behaviour so a future refactor cannot regress it silently.
    """
    if not n_gold:
        return {}
    n = len(n_gold)
    out: dict[str, float] = {}
    for k in grid:
        total = sum(min(g, k) / g for g in n_gold if g > 0)
        out[_key(k)] = _r(total / n)
    return out


def saturation_k(ceiling: dict[str, float], tol: float = SATURATION_TOL) -> int | None:
    """`K* = min{ K : ceiling(K) >= 1 - tol }`, or `None` if the grid never reaches it.

    Beyond K*, no retriever can add evidence, so every further change in answer
    accuracy is a reader-side (utilization) effect. This is the analytic
    saturation point, and it is what the whole "depth-as-a-setting" argument
    turns on: the field pins K at 5, 12, or 15 without knowing it has already
    passed 6.
    """
    threshold = 1.0 - tol
    for k in sorted(int(key) for key in ceiling):
        if ceiling[_key(k)] >= threshold:
            return k
    return None


def survivors_at(n_sessions: Sequence[int], grid: Sequence[int] = CEILING_GRID) -> dict[str, float]:
    """Fraction of questions whose corpus has **at least K** sessions.

    This is the fraction for which `K` is a legal retrieval depth. `survivors_at[K]`
    is NOT "fraction degenerate" — the complement is, and it is emitted
    separately as `degeneracy_fraction` (PRD §R2.5, §R3.3).
    """
    if not n_sessions:
        return {}
    n = len(n_sessions)
    return {_key(k): _r(sum(1 for s in n_sessions if s >= k) / n) for k in grid}


def degeneracy_fraction(n_sessions: Sequence[int], k: int) -> float:
    """Fraction of questions for which `K` is degenerate, per question.

    Degenerate means `K >= n_sessions` **for that question** — the condition has
    silently become full-context. PRD §R3.1 forbids computing this from
    corpus-wide percentiles: K=32 is 80% of a 40-session conversation, yet a
    mean-based guard passes. This is the per-question logic, and it is the
    project's most expensive lesson (the K=64 defect) encoded as arithmetic.
    """
    if not n_sessions:
        return 0.0
    return _r(sum(1 for s in n_sessions if k >= s) / len(n_sessions))


# ── stratification (PRD §R2.2) ────────────────────────────────────────────────
def by_type_table(
    rows: Iterable[tuple[str, int]], grid: Sequence[int] = CEILING_GRID
) -> dict[str, dict[str, Any]]:
    """Per-`question_type` ceiling, with the retriever-limited classification.

    Classification is a direct consequence of `ceiling(1)`:

    - **retriever-independent** — `ceiling(1) == 1`. The answer is in one
      session, so any K>=1 condition is pure reader-side; a K-effect here
      measures the prompt, not the store.
    - **retriever-limited** — `ceiling(1) == 0`. The answer needs two or more
      sessions, so K=1 is *structurally impossible*; it is not retriever failure.
    - **mixed** — in between.

    Reading a K=1 dip as "our retriever is bad" on a retriever-limited type is
    the specific mistake this table exists to prevent.
    """
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


# ── corpus fraction (PRD §R2.7, gap G12) ──────────────────────────────────────
def corpus_fraction(
    n_sessions: Sequence[int], grid: Sequence[int] = CEILING_GRID
) -> dict[str, dict[str, Any]]:
    """Distribution of `K / |corpus|` per question, and how often it equals 1.0.

    This is the formal version of a bug the project found by accident: on S,
    K=64 is full-context for **0%** of conversations (the median corpus is 48
    sessions), while on M it is full-context for **100%** (the minimum corpus is
    460). *The same grid point is a different system at different scales.*

    Hence PRD §R2.8: `corpus_fraction` is a covariate in every model and an
    annotation on every figure axis. Without it, a K-curve that mixes regimes is
    uninterpretable.

    Three numbers per (K):

    - `frac_full_context` — `K / |corpus| == 1.0` exactly, i.e. the condition has
      silently become full-context.
    - `frac_k_gt_corpus` — `K > |corpus|`, i.e. the K-guard would **reject** this
      cell. Reported rather than clamped so degeneracy is visible, never hidden.
    - quantiles of the raw ratio, which may exceed 1 in that case.
    """
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


# ── artifact assembly ──────────────────────────────────────────────────────────
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
    """Assemble `ceiling.json` from already-loaded questions.

    Abstention items (`_abs`) are excluded from `ceiling` and `partial` — the
    benchmark's own rule, since there is no evidence to retrieve — but they are
    counted and reported, because they stay in end-to-end accuracy (PRD §R8.4).
    """
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
        # Observed types only — not the full enum. A type that never appears in
        # the data must not be reported as if it had been measured.
        "question_types_seen": sorted({q.ability for q in answerable}),
    }
    return artifact


def build_corpus_fraction_artifact(
    by_scale: Mapping[str, Sequence[Question]], grid: Sequence[int] = CEILING_GRID
) -> dict[str, Any]:
    """Assemble `corpus_fraction.json` per (scale, K) — PRD §R2.7, gap G12.

    Accepts several scales so the comparison the artifact exists to make
    ("K=64 is CAG on S, RAG on M") is in one file rather than reconstructed by
    hand from two.
    """
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

    # The cross-scale contrast is the point of the artifact — emit it directly
    # so no reader has to join two blocks by hand.
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
    """Assemble the multi-scale `ceiling.json`.

    Multi-scale rather than one-file-per-split for three reasons:

    1. **Order-independence.** `msat ceiling <split>` writes the *same* file
       whichever split you name, so the Makefile's
       `ceiling longmemeval_m && ceiling longmemeval_s && ceiling --verify`
       sequence cannot leave the artifact holding whichever split ran last.
    2. **Idempotence** (PRD §N5). Re-running with a different argument produces
       identical bytes.
    3. **The scale-invariance claim is the finding.** Putting S and M side by
       side is what lets a reader check that `ceiling(K)` does not move with
       corpus size, instead of taking it on trust.

    A `scale_invariant` block is emitted explicitly for the same reason — it is
    a claim the artifact should be able to falsify, not merely suggest.
    """
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
    """Loader integrity report — what the loader actually saw.

    Separate from `ceiling.json` on purpose: `ceiling.json` is a scientific
    result the paper cites, this is an engineering receipt. Conflating them would
    mean a schema change to the loader silently rewrites a committed number.

    Deliberately records only **measured** quantities. Token counts are *not*
    emitted here: tokenising needs the tokenizer pinned in
    `configs/retrieval.yaml`, which lands in Phase 2, and an estimated token
    count presented as a measurement is exactly the kind of quiet fudge this
    project exists to avoid. Exact character counts are emitted instead.
    """
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


# ── deterministic serialisation (PRD §R2.4) ───────────────────────────────────
def dump_canonical(artifact: dict[str, Any], path: Path) -> bytes:
    """Write `artifact` deterministically and return the exact bytes.

    `ceiling --verify` byte-compares a regeneration against the committed file,
    so the writer must be stable: sorted keys, fixed indentation, floats already
    rounded by `_r`, and a trailing newline. No timestamps anywhere in the
    artifact — a regeneration must not differ merely because it ran later.
    """
    payload = (
        json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    )
    data = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def verify_artifact(artifact: dict[str, Any], committed: Path) -> tuple[bool, str]:
    """Regenerate and byte-compare. Returns `(ok, detail)`.

    Compares the canonical encoding rather than re-parsing both files, so a
    change in *key order* or float precision is caught too — not just a change
    in value. PRD §R2.4 requires a non-zero exit on drift.
    """
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
    """Locate the first differing byte and show a little context."""
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
    """Guard for `allow_nan=False` — a NaN in an artifact should fail loudly."""
    return not isinstance(value, float) or math.isfinite(value)
