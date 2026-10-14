"""`data.lock` — provenance of the raw inputs (PRD §R2.3).

The benchmarks are large and separately licensed, so they are not redistributed.
`data.lock` is what stands in for them: SHA-256 + byte size + row count for every
raw file, enough to prove a regenerated `ceiling.json` came from the same bytes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from msat.datalock import build_lock, dump_lock, sha256_file, verify_lock
from msat.datasets.longmemeval import RAW_FILES
from tests.conftest import requires_s


def test_sha256_matches_hashlib_on_a_known_file(tmp_path: Path) -> None:
    import hashlib

    payload = b"msat" * 10_000
    path = tmp_path / "blob.bin"
    path.write_bytes(payload)
    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_streams_rather_than_reads(tmp_path: Path) -> None:
    """A ceiling is never acceptable — verify chunked reads still see it all."""
    import hashlib

    payload = bytes(range(256)) * 40_000  # ~10 MB
    path = tmp_path / "big.bin"
    path.write_bytes(payload)
    assert sha256_file(path, chunk_bytes=1024) == hashlib.sha256(payload).hexdigest()


def test_absent_files_are_recorded_not_omitted(tmp_path: Path) -> None:
    """`5_filler_sess` is missing today; the lock must say so.

    Omitting it would make the lock look complete while Arm B's only permitted
    padding source is absent — a P7 blocker hidden inside a provenance file.
    """
    lock = build_lock(tmp_path)
    assert set(lock["files"]) == set(RAW_FILES)
    for split in RAW_FILES:
        assert lock["files"][split]["present"] is False
        assert "filename" in lock["files"][split]


def test_lock_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "x.json").write_text("[]", encoding="utf-8")
    first = json.dumps(build_lock(tmp_path), indent=2, sort_keys=True) + "\n"
    second = json.dumps(build_lock(tmp_path), indent=2, sort_keys=True) + "\n"
    assert first == second


def test_lock_contains_no_timestamp(tmp_path: Path) -> None:
    """A lock that changes on every run proves nothing."""
    (tmp_path / "x.json").write_text("[]", encoding="utf-8")
    blob = json.dumps(build_lock(tmp_path)).lower()
    for word in ("timestamp", "generated_at", "created", "now"):
        assert word not in blob


def test_verify_lock_detects_a_mutated_file(tmp_path: Path) -> None:
    raw = tmp_path / RAW_FILES["S"]
    raw.write_text(json.dumps([{"question_id": "a"}]), encoding="utf-8")
    lock_path = tmp_path / "data.lock"
    dump_lock(build_lock(tmp_path), lock_path)
    assert verify_lock(lock_path, tmp_path) == []

    raw.write_text(json.dumps([{"question_id": "a"}, {"question_id": "b"}]), encoding="utf-8")
    problems = verify_lock(lock_path, tmp_path)
    assert problems
    assert any("size" in p or "sha256" in p for p in problems)


def test_verify_lock_reports_a_vanished_file(tmp_path: Path) -> None:
    raw = tmp_path / RAW_FILES["S"]
    raw.write_text("[]", encoding="utf-8")
    lock_path = tmp_path / "data.lock"
    dump_lock(build_lock(tmp_path), lock_path)

    raw.unlink()
    problems = verify_lock(lock_path, tmp_path)
    assert any("missing" in p for p in problems)


def test_verify_lock_reports_a_missing_lock(tmp_path: Path) -> None:
    problems = verify_lock(tmp_path / "nope.lock", tmp_path)
    assert any("missing" in p for p in problems)


@requires_s
def test_committed_lock_verifies_against_the_real_data(project_root: Path, tmp_path: Path) -> None:
    """The P1 exit gate's second half: `data.lock` matches the raw inputs."""
    committed = project_root / "data.lock"
    if not committed.is_file():
        pytest.fail("data.lock is not committed — run `make data-lock`")
    problems = verify_lock(committed)
    assert problems == [], "raw inputs drifted from the committed lock:\n" + "\n".join(problems)


@requires_s
def test_committed_lock_covers_the_real_splits(project_root: Path) -> None:
    lock = json.loads((project_root / "data.lock").read_text(encoding="utf-8"))
    for split in ("S", "M", "oracle"):
        entry = lock["files"][split]
        assert entry["present"] is True, f"{split} should be locked"
        assert len(entry["sha256"]) == 64
        assert entry["bytes"] > 0
        assert entry["rows"] == 500
