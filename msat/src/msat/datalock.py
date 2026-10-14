"""`data.lock` — SHA-256 + byte size + row count for every raw benchmark file.

PRD §R2.3. The raw benchmarks are large and separately licensed, so they are
NOT redistributed with this repository. What ships instead is this lock: it is
enough to prove that a regenerated `ceiling.json` was computed from the same
bytes the paper describes, without shipping gigabytes.

Streaming is not optional here. `lme_m_cleaned.json` is 2.74 GB; hashing it in
1 MiB chunks keeps peak memory flat. Row counts come from a separate ijson pass
that counts top-level objects without parsing them — a second read, but the
alternative (loading the file) is exactly what PRD §N2 forbids.

Like every committed artifact, the lock carries **no timestamp**. A lock that
changed on every run would prove nothing.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Final

from msat.datasets.longmemeval import RAW_FILES, count_records, resolve_data_dir

#: Read size for hashing. 1 MiB keeps RSS flat without materially slowing the
#: digest on a spinning disk or over SMB.
CHUNK_BYTES: Final = 1024 * 1024

LOCK_VERSION: Final = 1


def sha256_file(path: Path, chunk_bytes: int = CHUNK_BYTES) -> str:
    """SHA-256 of a file, read in fixed-size chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path, *, count_rows: bool = True) -> dict[str, Any]:
    """One file's lock entry. Streams both the digest and the row count."""
    entry: dict[str, Any] = {
        "filename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if count_rows:
        try:
            entry["rows"] = count_records(path)
        except Exception as exc:  # noqa: BLE001 - a lock entry is best-effort
            # A non-JSON-array file still gets a digest; it just has no row count.
            entry["rows"] = None
            entry["rows_error"] = f"{type(exc).__name__}: {exc}"
    return entry


def build_lock(data_dir: Path | None = None) -> dict[str, Any]:
    """Build the lock over every raw file that is actually present.

    Files in `RAW_FILES` that are absent are recorded as `present: false` rather
    than silently omitted. `5_filler_sess` is currently missing, and the lock is
    the right place for that to be visible — it is Arm B's only permitted padding
    source, so its absence is a P7 blocker, not a footnote.
    """
    root = data_dir or resolve_data_dir()
    files: dict[str, Any] = {}
    for split in sorted(RAW_FILES):
        path = root / RAW_FILES[split]
        if path.is_file():
            entry = describe(path)
            entry["present"] = True
            files[split] = entry
        else:
            files[split] = {
                "filename": RAW_FILES[split],
                "present": False,
                "note": f"not found under {root}",
            }
    return {
        "_semantics": {
            "sha256": "hex digest of the raw file bytes",
            "bytes": "file size on disk",
            "rows": "number of top-level JSON objects (0 if the file is not a JSON array)",
            "absent_files": "recorded with present=false rather than omitted, so a "
            "missing input is visible in the lock itself",
        },
        "lock_version": LOCK_VERSION,
        "data_dir": str(root),
        "files": files,
    }


def dump_lock(lock: dict[str, Any], path: Path) -> bytes:
    """Write the lock deterministically (sorted keys, no timestamps)."""
    payload = json.dumps(lock, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    data = payload.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def verify_lock(lock_path: Path, data_dir: Path | None = None) -> list[str]:
    """Re-hash every present file and return a list of mismatch descriptions.

    Empty list means the inputs are byte-identical to what produced the
    committed ceiling artifacts. A missing file that the lock claims is present,
    or a changed digest, both appear here.
    """
    if not lock_path.is_file():
        return [f"data.lock missing: {lock_path}"]
    committed = json.loads(lock_path.read_text(encoding="utf-8"))
    root = data_dir or resolve_data_dir()
    problems: list[str] = []

    for split, entry in sorted(committed.get("files", {}).items()):
        if not entry.get("present"):
            path = root / entry["filename"]
            if path.is_file():
                problems.append(
                    f"{split}: {entry['filename']} is present on disk but recorded absent"
                )
            continue
        path = root / entry["filename"]
        if not path.is_file():
            problems.append(f"{split}: {entry['filename']} recorded present but missing at {path}")
            continue
        actual_bytes = path.stat().st_size
        if actual_bytes != entry["bytes"]:
            problems.append(
                f"{split}: size {actual_bytes} != locked {entry['bytes']} ({entry['filename']})"
            )
        actual_hash = sha256_file(path)
        if actual_hash != entry["sha256"]:
            problems.append(
                f"{split}: sha256 {actual_hash[:16]}… != locked {entry['sha256'][:16]}… "
                f"({entry['filename']})"
            )
    return problems
