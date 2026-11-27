"""Benchmark local backends on real corpus prompts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from msat.providers import GenerationRequest, backend_availability, select_backend

PROMPT_SIZES: list[tuple[str, str, int]] = [
    ("S", "K1", 400),
    ("S", "K16", 4_000),
    ("S", "K64", 12_000),
    ("M", "K16", 20_000),
    ("M", "K64", 120_000),
]

FILLER = (
    "The user asked about a scheduling conflict. The assistant replied that the "
    "meeting would need to move to Thursday. "
)


def build_prompt(target_tokens: int) -> str:
    approx = FILLER * max(1, target_tokens // 20)
    return (
        "Conversation excerpts follow.\n\n"
        + approx
        + "\n\nQuestion: when was the meeting moved to?"
    )


def run_backend(
    name: str, model: str, sizes: list[tuple[str, str, int]], n_prompts: int
) -> dict[str, Any]:
    backend = select_backend(name)
    health = backend.health()
    if not health.get("ready"):
        return {"backend": name, "available": False, **health}

    rows: list[dict[str, Any]] = []
    try:
        for scale, k_label, target in sizes:
            prompt = build_prompt(target)
            requests = [
                GenerationRequest(
                    model=model,
                    prompt=prompt,
                    system="You are answering a question about a conversation history.",
                    max_tokens=64,
                    temperature=0.7,
                    seed=i,
                )
                for i in range(n_prompts)
            ]
            t0 = time.perf_counter()
            results = backend.generate_batch(requests)
            wall = time.perf_counter() - t0

            prompt_tokens = sum(r.prompt_tokens for r in results)
            completion_tokens = sum(r.completion_tokens for r in results)
            rows.append(
                {
                    "scale": scale,
                    "k": k_label,
                    "target_tokens": target,
                    "n_prompts": len(results),
                    "actual_prompt_tokens": prompt_tokens,
                    "wall_seconds": round(wall, 2),
                    "prefill_tok_per_s": round(prompt_tokens / wall, 1) if wall else 0,
                    "decode_tok_per_s": round(completion_tokens / wall, 1) if wall else 0,
                    "mean_latency_ms": round(statistics.mean(r.latency_ms for r in results), 1)
                    if results
                    else 0,
                    "finish_reasons": sorted({r.finish_reason for r in results}),
                }
            )
            print(
                f"    [{scale}/{k_label}] {wall:7.2f}s  "
                f"prefill {rows[-1]['prefill_tok_per_s']:>9,.0f} tok/s  "
                f"decode {rows[-1]['decode_tok_per_s']:>7,.1f} tok/s",
                flush=True,
            )
    finally:
        backend.close()

    prefill_rates = [r["prefill_tok_per_s"] for r in rows if r["prefill_tok_per_s"] > 0]
    decode_rates = [r["decode_tok_per_s"] for r in rows if r["decode_tok_per_s"] > 0]
    return {
        "backend": name,
        "available": True,
        "model": model,
        "rows": rows,
        "median_prefill_tok_per_s": round(statistics.median(prefill_rates), 1)
        if prefill_rates
        else 0,
        "median_decode_tok_per_s": round(statistics.median(decode_rates), 1) if decode_rates else 0,
    }


def project(results: dict[str, Any], assumed_hours: float) -> dict[str, Any] | None:
    assumed_prefill = 8_000.0
    if not results.get("available"):
        return None
    p = results.get("median_prefill_tok_per_s") or 0
    d = results.get("median_decode_tok_per_s") or 0
    if p <= 0 or d <= 0:
        return None
    speedup = assumed_prefill / p
    projected = assumed_hours / speedup
    return {
        "assumed_total_hours": assumed_hours,
        "measured_prefill_tok_per_s": p,
        "measured_decode_tok_per_s": d,
        "speedup_vs_assumption": round(speedup, 2),
        "projected_total_hours": round(projected, 1),
        "projected_total_days": round(projected / 24.0, 1),
        "caveat": "prefill-weighted projection; verify against the full grid before "
        "committing the schedule",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-4-9b")
    parser.add_argument("--backends", default="transformers")
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--assumed-hours", type=float, default=314.5)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    print("=== availability ===")
    availability = backend_availability()
    for name, info in availability.items():
        mark = "OK " if info["available"] else "-- "
        print(f"  {mark}{name:<20} {info.get('remedy', '')}")

    report: dict[str, Any] = {
        "model": args.model,
        "prompts_per_size": args.prompts,
        "availability": availability,
        "results": [],
    }

    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        print(f"\n=== {name} ===")
        try:
            result = run_backend(name, args.model, PROMPT_SIZES, args.prompts)
        except Exception as exc:  # noqa: BLE001 - a benchmark must not crash
            result = {
                "backend": name,
                "available": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        if result.get("available"):
            result["projection"] = project(result, args.assumed_hours)
            if result["projection"]:
                p = result["projection"]
                print(
                    f"    -> {p['projected_total_hours']:,.1f} h "
                    f"({p['projected_total_days']} days), "
                    f"{p['speedup_vs_assumption']}x the assumed throughput"
                )
        report["results"].append(result)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")

    print(
        "\nNext: paste the measured numbers into configs/pricing/latest.yaml "
        "(`throughput_assumed`) and re-run `msat estimate-power`."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
