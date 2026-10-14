"""Re-derive feasibility and cost with B honoured and VRAM as the real ceiling.

Run:  uv run --no-sync python scripts/analyze_feasibility.py

This is ANALYSIS, not configuration. It does not write to `configs/`, because the
reader panel is still an open decision (blocker B10). It answers: given the three
checkpoints that actually exist, on the card we actually have, which cells can
run, and what does the grid really cost?

Every number below is DERIVED, not typed:

  KV bytes/token = 2 (K and V) x layers x kv_heads x head_dim x bytes/elem
  prompt_tokens  = (vram_gb - weight_gb - overhead_gb) / kv_mb_per_1k * 1000

The architecture numbers come from each repo's `config.json`, fetched from the
HuggingFace API. The SHAs are pinned so this script describes specific weights,
not a moving tag.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from msat.design import (
    ReaderLimit,
    build_cells,
    context_feasibility,
    duplicate_conditions,
    estimate_compute,
    validate_token_anchors,
)

# ── verified against the HuggingFace API on 2026-10-13 ────────────────────────
# (config.json values, sha, lastModified)
MODELS = {
    "gemma-4-E4B-it": {
        "repo": "google/gemma-4-E4B-it",
        "sha": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        "released": "2026-07-20",
        "params_b": 4.0,
        "context_window": 131072,
        "layers": 42,
        "kv_heads": 2,
        "head_dim": 256,
    },
    "Qwen3.5-9B": {
        "repo": "Qwen/Qwen3.5-9B",
        "sha": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        "released": "2026-03-02",
        "params_b": 9.0,
        "context_window": 262144,
        "layers": 32,
        "kv_heads": 4,
        "head_dim": 256,
    },
    "Phi-4-mini-instruct": {
        "repo": "microsoft/Phi-4-mini-instruct",
        "sha": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "released": "2025-12-10",
        "params_b": 3.8,
        "context_window": 131072,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
    },
    # Kept for comparison only: the config's original "primary reader".
    "gemma-4-12B-it": {
        "repo": "google/gemma-4-12B-it",
        "sha": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
        "released": "2026-07-20",
        "params_b": 12.0,
        "context_window": 262144,
        "layers": 48,
        "kv_heads": 8,
        "head_dim": 256,
    },
}

VRAM_GB = 8.0
OVERHEAD_GB = 0.5
BITS_PER_PARAM = 0.55  # int4 + quantisation metadata, empirically ~0.55 B/param

K_GRID = [1, 2, 4, 8, 16, 32, 64]
B_GRID = [(4096, "absolute_tokens"), (16384, "absolute_tokens"), (65536, "absolute_tokens")]
SCALES = ["S", "M"]
RUNS_PER_CELL = 3
ITEMS = 500

PRICING = {
    "prompt_overhead_tokens": 500,
    "token_estimates": {
        "S": {"prompt_tokens_at_K1": 400, "prompt_tokens_at_K64": 12000, "output_tokens": 80},
        "M": {"prompt_tokens_at_K1": 2000, "prompt_tokens_at_K64": 120000, "output_tokens": 80},
    },
    "compute_budget": {
        "throughput_assumed": {
            "prefill_tokens_per_sec": 8000,
            "decode_tokens_per_sec": 300,
        },
        "output_tokens": {"reader": 80, "judge": 100},
        "dtype": "4bit",
        "batch_size": 8,
        "max_wall_clock_hours": 336,
    },
    "judge": {
        "input_tokens_approx": 4000,
        "judges_per_output": 2,
        "reliability_audit_multiplier": 1.0,
    },
}


def kv_mb_per_1k(m: dict[str, object]) -> float:
    return 2 * m["layers"] * m["kv_heads"] * m["head_dim"] * 2 * 1000 / 1048576  # type: ignore[index,operator]


def main() -> int:
    panel = ["gemma-4-E4B-it", "Qwen3.5-9B", "Phi-4-mini-instruct"]

    print("=" * 78)
    print("1. WHAT EACH CHECKPOINT CAN SERVE ON AN 8 GB RTX 4060")
    print("=" * 78)
    print(
        f"{'model':<24}{'params':>8}{'4-bit':>9}{'KV MB/1k':>11}{'VRAM cap':>11}{'window':>10}  binds"
    )
    print("-" * 78)

    limits: dict[str, ReaderLimit] = {}
    for name in [*panel, "gemma-4-12B-it"]:
        m = MODELS[name]
        weight_gb = m["params_b"] * BITS_PER_PARAM  # type: ignore[operator]
        kv = kv_mb_per_1k(m)
        lim = ReaderLimit.from_vram(
            name,
            context_window=m["context_window"],  # type: ignore[arg-type]
            weight_gb=weight_gb,
            kv_mb_per_1k=kv,
            vram_gb=VRAM_GB,
            overhead_gb=OVERHEAD_GB,
        )
        limits[name] = lim
        print(
            f"{name:<24}{m['params_b']:>7.1f}B{weight_gb:>8.1f}G{kv:>11.0f}"
            f"{lim.max_prompt_tokens:>11,}{m['context_window']:>10,}  {lim.binding}"
        )

    print()
    print("=" * 78)
    print("2. PROMPT SIZE PER (K, B)  —  min(K_content, B) + 500")
    print("=" * 78)
    cells_all = build_cells(K_GRID, B_GRID, SCALES, panel)
    # Key on (scale, B, K). Keying on (scale, B) alone silently collapses the K
    # axis — the last K written wins and every row shows K=64, which makes the
    # whole table look like B does all the work. It does not.
    rows: dict[tuple[str, float, int], int] = {}
    for c in cells_all:
        anchors = PRICING["token_estimates"][c.scale]
        rows[(c.scale, c.b, c.k)] = int(
            min(_interp(c.k, anchors), c.b) + PRICING["prompt_overhead_tokens"]
        )

    hdr = "".join(f"{b:>9,}" for b, _ in B_GRID)
    print(f"{'scale':<8}{'K':>4}" + hdr)
    print("-" * 46)
    for scale in SCALES:
        for k in K_GRID:
            print(f"{scale:<8}{k:>4}" + "".join(f"{rows[(scale, b, k)]:>9,}" for b, _ in B_GRID))
    print()
    print("Note the S row: B=65536 never binds because S content tops out at ~12k.")
    print("Note the M row: B caps the prompt, so K stops mattering once content exceeds B.")

    print()
    print("=" * 78)
    print("3. FEASIBILITY — both ceilings checked")
    print("=" * 78)
    report = context_feasibility(cells_all, {p: limits[p] for p in panel}, PRICING)
    for r, info in report["readers"].items():
        print(
            f"  {r:<24} ceiling {info['ceiling']:>7,}  (blocked_by={info['blocked_by']})  "
            f"largest prompt {info['largest_prompt']:>7,}"
        )
    print(f"\n  infeasible cells: {report['n_infeasible']} of {len(cells_all)}")
    by_b: dict[float, int] = {}
    for rows_ in report["infeasible_cells"].values():
        for row in rows_:
            by_b[row["B"]] = by_b.get(row["B"], 0) + 1
    if by_b:
        print(f"  by B level: {{{', '.join(f'B={int(b)}: {n}' for b, n in sorted(by_b.items()))}}}")
        scales_hit = {row["scale"] for r in report["infeasible_cells"].values() for row in r}
        print(f"  scales hit: {sorted(scales_hit)}")
        print("  -> infeasibility is a B-level effect, not a K effect.")

    print()
    print("=" * 78)
    print("4. COST — with B honoured (the 314.5 h figure did NOT honour B)")
    print("=" * 78)
    comp = estimate_compute(
        cells_all,
        PRICING,
        runs_per_cell=RUNS_PER_CELL,
        items_per_cell=ITEMS,
        judges_per_output=2,
    )
    if comp.get("available"):
        tot = comp["total_hours"]
        print(f"  reader hours    : {comp['reader_hours']:>10,.1f} h")
        print(
            f"  judge hours     : {comp['judge_hours']:>10,.1f} h  ({comp['judge_share_of_total']:.0%} of total)"
        )
        print(f"  audit hours     : {comp['judge_audit_hours']:>10,.1f} h")
        print(f"  TOTAL           : {tot:>10,.1f} h  =  {comp['total_days']:,.1f} days")
        print(f"  budget cap      : {comp['max_wall_clock_hours']:>10} h")
        print(f"  within budget   : {comp['within_budget']}")
        print(f"  calls           : {comp['n_calls']:>10,}")
        print()
        print("  per reader:")
        for r, v in comp["per_reader"].items():
            print(f"    {r:<24}{v['hours']:>8,.1f} h   {v['calls']:>9,} calls")
        print()
        print("  previously reported (B ignored): 314.5 h")
        print(f"  corrected                     : {tot:,.1f} h")
        print(f"  difference                    : {tot - 314.5:+,.1f} h ({tot / 314.5 - 1:+.1%})")
    else:
        print("  compute budget unavailable")

    print()
    print("=" * 78)
    print("5. DEGENERATE CONDITIONS — two cells can produce a BYTE-IDENTICAL prompt")
    print("=" * 78)
    print("R4.4: a crossed cell retrieves top-K in RANK ORDER, then truncates to B")
    print("tokens. So prompt(K,B) =")
    print("    top-K in full          if content(K) <= B")
    print("    first B tokens of the  if content(K) >  B   <- a PREFIX, so all K")
    print("    same rank-ordered stream                         beyond the cutoff")
    print("                                                          are IDENTICAL")
    print()

    dup_report = duplicate_conditions(cells_all, PRICING)
    seen_sets: set[tuple] = set()
    for d in dup_report["duplicate_sets"]:
        m = d["members"]
        shape = tuple(sorted((x["scale"], x["K"], int(x["B"])) for x in m))
        if shape in seen_sets:
            continue  # same shape for the other readers; one line is enough
        seen_sets.add(shape)
        label = f"{m[0]['scale']} K={m[0]['K']} B={int(m[0]['B'])}  ==  " + ", ".join(
            f"K={x['K']} B={int(x['B'])}" for x in m[1:]
        )
        print(f"  {d['n_members']} cells collapse:  {label}")

    print()
    print(
        f"  duplicate cells: {dup_report['n_redundant_cells']} of {dup_report['n_cells']}"
        f"  ({dup_report['redundant_fraction']:.0%})"
    )
    print(f"  unique conditions: {dup_report['n_unique_conditions']}")
    print()
    print("  cell_key_collisions() CANNOT see this: the keys differ (B differs),")
    print("  so the grid looks collision-free while rendering identical prompts.")
    print("  Every duplicate is charged full compute AND counted as an independent")
    print("  condition, which is exactly how the power analysis overstates the design.")

    # ── 6. are the token anchors themselves right? ────────────────────────────
    print()
    print("=" * 78)
    print("6. TOKEN ANCHORS vs P1's MEASURED CORPUS")
    print("=" * 78)
    ds = json.loads(Path("results/dataset_report.json").read_text(encoding="utf-8"))["scales"]
    chars = {s: float(ds[s]["corpus_chars"]["mean_per_question"]) for s in ("S", "M")}
    sess = {s: float(ds[s]["sessions_per_question"]["mean"]) for s in ("S", "M")}
    for s in ("S", "M"):
        print(
            f"  {s}: {chars[s]:>10,.0f} chars/question / {sess[s]:.2f} sessions "
            f"= {chars[s] / sess[s]:>8,.0f} chars per session"
        )
    print()
    anchor_report = validate_token_anchors(PRICING, chars, sess)
    print(f"  {'scale':<6}{'anchor':<26}{'declared':>10}{'expected':>11}{'ratio':>8}  ok")
    print("  " + "-" * 66)
    for f in anchor_report["findings"]:
        print(
            f"  {f['scale']:<6}{f['anchor']:<26}{f['declared']:>10,.0f}"
            f"{f['expected_from_measured_corpus']:>11,.0f}{f['ratio']:>8.2f}"
            f"  {'yes' if f['ok'] else 'NO'}"
        )
    print()
    print(f"  {anchor_report['verdict']}")
    print(f"  chars_per_token assumed: {anchor_report['chars_per_token_assumed']} (ASSUMPTION)")
    print("  The gap is far larger than any plausible ratio, so it does not depend on it.")

    # ── 7. measured content model vs the placeholder anchors ─────────────────
    print()
    print("=" * 78)
    print("7. MEASURED CONTENT MODEL vs THE PLACEHOLDER ANCHORS")
    print("=" * 78)
    print("  content(K) = min(K, sessions_mean) * tokens_per_session")
    print("  the min() is load-bearing: S has 47.7 sessions, so it SATURATES at")
    print("  K=48 and every K beyond renders the same corpus.")
    print()

    import copy

    cpt = float(anchor_report["chars_per_token_assumed"])
    measured = copy.deepcopy(PRICING)
    for s in ("S", "M"):
        te = measured["token_estimates"][s]
        te["sessions_mean"] = sess[s]
        te["tokens_per_session"] = chars[s] / sess[s] / cpt
        te["corpus_tokens"] = chars[s] / cpt

    comparison = []
    for label, pr in (("PLACEHOLDER  ", PRICING), ("MEASURED (P1)  ", measured)):
        f2 = context_feasibility(cells_all, {p: limits[p] for p in panel}, pr)
        c2 = estimate_compute(
            cells_all, pr, runs_per_cell=3, items_per_cell=500, judges_per_output=2
        )
        d2 = duplicate_conditions(cells_all, pr)
        comparison.append(
            {
                "label": label.strip(),
                "n_infeasible": f2["n_infeasible"],
                "n_redundant": d2["n_redundant_cells"],
                "n_unique": d2["n_unique_conditions"],
                "total_hours": c2["total_hours"],
            }
        )
        print(
            f"  {label}  infeasible {f2['n_infeasible']:>4}/126"
            f"   redundant {d2['n_redundant_cells']:>4}/126"
            f"   unique {d2['n_unique_conditions']:>4}"
            f"   cost {c2['total_hours']:>8,.1f} h"
        )
    print()

    # ── 8. can ANY B grid be non-degenerate? ─────────────────────────────────
    print("=" * 78)
    print("8. CAN ANY B GRID BE NON-DEGENERATE?  (the staircase)")
    print("=" * 78)
    print("  At depth K the number of ACTIVE budget levels is the number of B")
    print("  values below content(K). That count is monotone in K — it can only")
    print("  grow. So with a truncating budget:")
    print("    B < content(1)  -> active at every K, collapses all of them")
    print("    B > content(64) -> inert at every K,  collapses all of them")
    print("    in between      -> a STAIRCASE: active high, inert low")
    print()
    print("  distinct B outcomes per K (max 3):")
    for bvals in ([4096, 16384, 65536], [2048, 8192, 32768], [1536, 6144, 24576]):
        line = f"    B={bvals!s:<26}"
        for scale in ("S", "M"):
            row = [len({min(_content(scale, k), b) for b in bvals}) for k in K_GRID]
            line += f"{scale}:{''.join(str(v) for v in row)}   "
        print(line)
    print()
    print("  Saturated at 3 except at K=1,2 where only sub-content(K) budgets can")
    print("  discriminate. A stair of 2 is the best a truncating B can do at the")
    print("  low-K end. The interaction is therefore a SATURATION CURVE — 'where")
    print("  does the budget start to bite' — not a free 2-factor crossing.")
    out = Path("results/feasibility_analysis.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "vram_gb": VRAM_GB,
                "overhead_gb": OVERHEAD_GB,
                "panel": {p: MODELS[p] for p in panel},
                "feasibility": report,
                "compute": comp,
                "duplicates": dup_report,
                "anchors": anchor_report,
                "comparison": comparison,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


def _content(scale: str, k: int) -> float:
    """The measured content model: min(K, sessions) * tokens per session."""
    ds = json.loads(Path("results/dataset_report.json").read_text(encoding="utf-8"))["scales"]
    sessions = float(ds[scale]["sessions_per_question"]["mean"])
    chars = float(ds[scale]["corpus_chars"]["mean_per_question"])
    return min(float(k), sessions) * chars / sessions / 4.0


def _interp(k: int, anchors: dict[str, float]) -> float:
    lo, hi = 1, 64
    if k <= lo:
        return anchors["prompt_tokens_at_K1"]
    if k >= hi:
        return anchors["prompt_tokens_at_K64"]
    import math

    t = (math.log2(k) - math.log2(lo)) / (math.log2(hi) - math.log2(lo))
    return math.exp(
        math.log(anchors["prompt_tokens_at_K1"])
        + t * (math.log(anchors["prompt_tokens_at_K64"]) - math.log(anchors["prompt_tokens_at_K1"]))
    )


if __name__ == "__main__":
    raise SystemExit(main())
