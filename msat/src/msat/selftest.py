"""`make selftest` (Architecture §P3 EXIT) — the whole P3 path on a fixture.

Runs write → guard → manifest on the synthetic corpus with **no network and no
GPU**, and must finish in seconds. If this cannot run anywhere, it will only be run
once, and the first time it runs will be the first time it is broken.

WHAT IT PROVES
--------------
1. The write path is deterministic — the same corpus produces the same
   `write_hash` twice.
2. The write path is audited — no source session is missing from the log, and the
   encoding ceiling is reported.
3. The guards reject at build time, before anything is spent.
4. The manifest round-trips: it writes, reloads, and hashes the same.

WHY SYNTHETIC AND NOT A SUBSET OF THE REAL DATA
-----------------------------------------------
The real corpus is 2.6 GB and its questions are 475 sessions each; a selftest over
it would be slow, would need the data present, and would test the LOADER more than
the pipeline. Three questions x three corpus sizes exercises the guards properly —
see `scripts/make_synthetic_fixture.py` — and runs anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def synthetic_corpus(chars_per_token: float = 4.0) -> tuple[dict[str, list], dict[str, list]]:
    """Write every fixture question to memory and return guard inputs.

    Goes through the REAL write path rather than synthesising counts, so the
    guards are evaluated against what memory actually holds. Counts invented for a
    test would test nothing about the thing that produces them.
    """
    from msat.manifest import build_from_memory
    from msat.scripts_fixture import load_synthetic

    corpus: dict[str, list] = {"S": [], "M": []}
    raw: dict[str, list[tuple[int, int]]] = {"S": [], "M": []}

    for question in load_synthetic():
        sessions = [(s["session_id"], s["text"]) for s in question["sessions"]]
        _, write_record, stats = build_from_memory(question["question_id"], sessions)
        if write_record.encoding_ceiling < 1.0:
            raise AssertionError(
                f"{question['question_id']}: encoding ceiling "
                f"{write_record.encoding_ceiling} — the fixture should write cleanly, "
                f"and a drop here would mean the write path is losing sources"
            )
        # The fixture's small and mid questions stand in for the S scale, the large
        # one for M. Both scales are populated so the manifest exercises the
        # per-scale path rather than the empty-corpus branch.
        scale = "S" if len(sessions) < 20 else "M"
        corpus[scale].append(stats)
        raw[scale].append((len(sessions), sum(len(t) for _, t in sessions)))

    return corpus, raw


def run_selftest(configs: Path | None = None) -> dict[str, Any]:
    """Execute every P3 invariant. Returns a report; raises on any failure."""
    from msat.config import load_bundle
    from msat.manifest import build_manifest

    root = Path(configs) if configs else Path("configs")
    bundle = load_bundle(root)
    cpt = float(bundle.pricing.get("token_estimates", {}).get("chars_per_token", 4.0))

    from msat.scripts_fixture import load_synthetic

    checks: list[dict[str, Any]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "ok": ok, "detail": detail})
        if not ok:
            raise AssertionError(f"selftest failed: {name} — {detail}")

    # ── 1. determinism ───────────────────────────────────────────────────────
    from msat.memory import write as mem_write

    question = load_synthetic()[0]
    sessions = [(s["session_id"], s["text"]) for s in question["sessions"]]
    r1 = mem_write(question["question_id"], sessions)[1]
    r2_rec = mem_write(question["question_id"], sessions)[1]
    record("write is deterministic", r1.write_hash == r2_rec.write_hash, r1.write_hash)

    # ── 2. audited ───────────────────────────────────────────────────────────
    record(
        "every source session is logged",
        r1.n_units == len(sessions),
        f"{r1.n_units} records for {len(sessions)} sources",
    )
    record("encoding ceiling reported", r1.encoding_ceiling == 1.0, str(r1.encoding_ceiling))

    # ── 3. guards reject, before anything is spent ──────────────────────────
    corpus, raw = synthetic_corpus(cpt)
    manifest = build_manifest(bundle, corpus, chars_per_token=cpt, raw_counts=raw)
    record("manifest enumerated every cell", len(manifest.cells) > 0, str(len(manifest.cells)))
    record(
        "no condition-key collisions",
        manifest.collisions == [],
        str(manifest.collisions[:2]),
    )
    deepest = max(bundle.design.K_grid)
    record(
        "K-guard rejects the over-deep cell",
        any(c.condition.k == deepest and c.status == "rejected" for c in manifest.cells),
        f"the deepest declared K={deepest} is rejected on the small fixture questions",
    )

    # ── 4. round-trips ───────────────────────────────────────────────────────
    payload = manifest.as_dict()
    reloaded = json.loads(json.dumps(payload))
    record(
        "manifest round-trips through JSON",
        reloaded["config_hash"] == manifest.config_hash
        and len(reloaded["cells"]) == len(manifest.cells),
    )
    record(
        "config hash recorded",
        bool(manifest.config_hash) and len(manifest.config_hash) == 64,
        manifest.config_hash,
    )

    return {
        "n_checks": len(checks),
        "all_passed": all(c["ok"] for c in checks),
        "checks": checks,
        "cells": len(manifest.cells),
        "accepted": len(manifest.accepted),
        "rejected": len(manifest.rejected),
    }


if __name__ == "__main__":  # pragma: no cover - the `make selftest` entry point
    import json
    import sys

    try:
        result = run_selftest()
    except AssertionError as exc:
        print(f"SELFTEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["all_passed"] else 1)
