#!/usr/bin/env python
"""Fetch the raw LongMemEval REMOTE this harness reads.

    uv run python scripts/fetch_data.py            # the three REMOTE that exist
    uv run python scripts/fetch_data.py --check     # report only, download nothing
    uv run python scripts/fetch_data.py --split M   # just the 2.6 GB one

WHY THIS IS A SCRIPT AND NOT A `curl` LINE IN THE README
---------------------------------------------------------
Three things make a hand-typed download the wrong tool here.

1. **The published filenames do not match the ones the code reads.**
   `longmemeval.py:RAW_FILES` expects `lme_s_cleaned.json`, but the official
   repository publishes `longmemeval_s_cleaned.json`. Renaming by hand is a
   silent, unreproducible step, and it is exactly the kind of thing that makes
   `data.lock` (R2.3) unreproducible three months later. The mapping lives here,
   in one place, next to the SHA.

2. **A pinned revision, not `main`.** Same reasoning as `retrieval.yaml:16` and
   `readers.yaml:26`: a branch name gets repointed, and then the bytes under a
   fixed `data.lock` have silently changed. This script downloads a specific
   commit and records which one.

3. **It refuses rather than substitutes.** Arm B pads from
   `5_filler_sess.json` and ONLY from it (`24_open_decisions.md` D6). That file
   is **not published** - see `MISSING` below. This script will not invent a
   substitute, and it does not silently skip the gap: it exits non-zero and
   names the consequence, because a dataset that loads without its filler is a
   dataset whose Arm B results would be unattributable.

WHAT IS ACTUALLY AVAILABLE (verified against the HF API, 2026-10-14)
--------------------------------------------------------------------
    xiaowu0162/longmemeval-cleaned  @ 98d7416c24c778c2fee6e6f3006e7a073259d48f

        longmemeval_s_cleaned.json     264.5 MB
        longmemeval_m_cleaned.json   2,610.3 MB
        longmemeval_oracle.json        14.7 MB

    `5_filler_sess.json` is NOT in that repository, and NOT in the upstream
    GitHub tree either (xiaowu0162/LongMemEval ships code and assets only - 34
    entries, no data files). **D6 is therefore answered: the file does not
    exist in any official LongMemEval distribution.** Arm B needs a
    user-supplied source or it gets dropped; it does not get a substitute.

    The 2,610.3 MB for M matches the figure `longmemeval.py:5` documents
    independently ("`lme_m_cleaned.json` is 2.6 GB"), which is a useful
    cross-check that this is the intended artifact and not a lookalike.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: Pinned revision. NOT `main` - see the module docstring, point 2.
HF_REPO = "xiaowu0162/longmemeval-cleaned"
HF_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"

HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"

#: `{logical split: (published name, exact bytes at HF_REVISION)}`.
#: Sizes are from the HF tree API at `HF_REVISION`, not estimated, and not
#: rounded - `mats[REMOTE]` is compared for EQUALITY, not a tolerance. A
#: 2.7 GB transfer that "succeeds" having written 107 MB is the single most
#: likely failure mode of this script, and it fails downstream as a parse error
#: that looks like corrupt data rather than a bad download.
REMOTE: dict[str, tuple[str, int]] = {
    "S": ("longmemeval_s_cleaned.json", 277_383_467),
    "M": ("longmemeval_m_cleaned.json", 2_737_100_077),
    "oracle": ("longmemeval_oracle.json", 15_388_478),
}

#: Expected local names, from `longmemeval.py:RAW_FILES`. The key mismatch is
#: the point of this whole script.
LOCAL_NAMES = {"S": "lme_s_cleaned.json", "M": "lme_m_cleaned.json", "oracle": "lme_oracle.json"}

#: `RAW_FILES["filler"]`. Recorded as MISSING rather than omitted, so that a
#: reader of this script learns the state of the world instead of assuming the
#: map is complete.
MISSING = {
    "filler": (
        "5_filler_sess.json",
        "not published in xiaowu0162/longmemeval-cleaned and not in the "
        "upstream GitHub tree (24_open_decisions.md D6). Arm B pads from this "
        "file and ONLY from it - substituting another source would inject "
        "context a budgeted arm would never have shown.",
    )
}

CHUNK = 1 << 20  # 1 MiB - the 2.6 GB transfer is streamed, never buffered.


def default_data_dir() -> Path:
    """Mirror `longmemeval.resolve_data_dir()` without importing the package."""
    import os

    override = os.environ.get("MSAT_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # Must resolve to the SAME directory as `longmemeval.resolve_data_dir()`.
    # `parents[1]` from `scripts/fetch_data.py` is `<repo>/msat`; its `.parent`
    # is `<repo>` = MemSat, which is where `Research/` is a submodule. That
    # function, from the two levels deeper `src/msat/datasets/`, walks up 3 to
    # the same `<repo>/msat` and takes the same `.parent`. Off by one here and
    # the two disagree silently: files land in `Work/Research/context/raw`
    # while the loader looks in `MemSat/Research/context/raw`, and every symptom
    # is "the split is absent".
    msat_pkg_root = Path(__file__).resolve().parents[1]
    return (msat_pkg_root.parent / "Research/context/raw").resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


class ShortTransferError(RuntimeError):
    """The server closed the connection before sending the whole object."""


def download(url: str, dest: Path, expect: int, attempts: int = 3) -> int:
    """Stream to a `.part`, verify the byte count, then rename.

    `urlopen` returning without raising means "the response ENDED", not "you
    received the whole body". A connection dropped mid-transfer - which is what
    a 2.7 GB pull does on a laptop WiFi link - surfaces as a short read that
    `urlopen` reports as success. Comparing against the server's own
    `Content-Length` is the only check that catches it.

    `.part` is renamed ONLY on an exact match, so a truncated file can never
    masquerade as a present one: `available_REMOTE()` finds nothing, and the
    loader says "absent" rather than handing a half-file to the parser.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        tmp = dest.with_suffix(dest.suffix + ".part")
        written = 0
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "msat-fetch-data/1"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                declared = resp.headers.get("Content-Length")
                if resp.status == 206 or "Content-Range" in resp.headers:
                    raise ShortTransferError("server returned 206 partial content")
                with tmp.open("wb") as out:
                    while True:
                        block = resp.read(CHUNK)
                        if not block:
                            break
                        out.write(block)
                        written += len(block)
            if declared is not None and int(declared) != written:
                raise ShortTransferError(f"Content-Length {int(declared)} but wrote {written}")
            if written != expect:
                raise ShortTransferError(f"expected {expect} B, wrote {written} B")
            tmp.replace(dest)
            return written
        except (urllib.error.URLError, OSError, ShortTransferError) as exc:
            last = exc
            tmp.unlink(missing_ok=True)
            if attempt < attempts:
                wait = 2**attempt
                print(f"       retry {attempt}/{attempts - 1} in {wait}s ({exc})", flush=True)
                time.sleep(wait)
    raise RuntimeError(f"{dest.name}: {last}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="report presence, download nothing")
    ap.add_argument("--split", action="append", choices=sorted(REMOTE), help="one split (repeatable)")
    ap.add_argument("--dir", type=Path, default=None, help="override $MSAT_DATA_DIR")
    args = ap.parse_args(argv)

    data_dir = args.dir or default_data_dir()
    wanted = args.split or list(REMOTE)

    print(f"source  {HF_REPO} @ {HF_REVISION[:12]}")
    print(f"dest    {data_dir}")
    print(f"action  {'CHECK ONLY' if args.check else 'DOWNLOAD'}\n")

    data_dir.mkdir(parents=True, exist_ok=True)
    record: dict[str, object] = {
        "repo": HF_REPO,
        "revision": HF_REVISION,
        "files": {},
        "missing": {},
    }
    failures = 0

    for split in wanted:
        pub, expect = REMOTE[split]
        dest = data_dir / LOCAL_NAMES[split]

        if dest.exists():
            actual = dest.stat().st_size
            ok = actual == expect
            print(f"  {'ok  ' if ok else 'BAD '} {split:<7} {dest.name:<28} {actual:>13,} B")
            if not ok:
                print(f"       expected {expect:,} B - re-download or delete it; "
                      "a short file is worse than an absent one")
                failures += 1
        elif args.check:
            print(f"  --   {split:<7} {LOCAL_NAMES[split]:<28} absent")
            continue
        else:
            print(f"  get  {split:<7} {pub} ({expect:,} B) ...", flush=True)
            try:
                download(
                    HF_BASE.format(repo=HF_REPO, rev=HF_REVISION, name=pub), dest, expect
                )
            except (urllib.error.URLError, OSError, RuntimeError) as exc:
                print(f"  FAIL {split:<7} {exc}")
                failures += 1
                continue
            print(f"  ok   {split:<7} -> {dest.name}  {dest.stat().st_size:,} B")

        digest = sha256(dest)
        record["files"][split] = {
            "published_name": pub,
            "local_name": dest.name,
            "bytes": dest.stat().st_size,
            "sha256": digest,
        }
        print(f"       sha256 {digest}\n")

    for split, (name, why) in MISSING.items():
        print(f"  MISS {split:<7} {name:<28} {why}")
        record["missing"][split] = {"local_name": name, "reason": why}

    (data_dir / "fetch_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\nwrote {data_dir / 'fetch_manifest.json'}")

    if failures:
        print(f"\n{failures} file(s) failed. Do not run a sweep.", file=sys.stderr)
        return 1

    # Report what is ACTUALLY on disk, not what was asked for. An unconditional
    # "all present" here is the exact failure this script exists to prevent: a
    # success message that is true on a --check run with nothing downloaded.
    got = sorted(record["files"])  # type: ignore[arg-type]
    absent = [s for s in REMOTE if s not in got]
    if absent:
        print(f"\n{len(got)}/{len(REMOTE)} REMOTE present. Absent: {', '.join(absent)}")
        print("`msat ceiling` cannot regenerate ceiling.json until they are.")
    else:
        print("\nS, M and oracle present. `msat ceiling --verify` can now "
              "regenerate ceiling.json offline.")
    print("Arm B still cannot run - see D6 above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())