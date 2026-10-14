"""Content-addressed response store (PRD §R6.1, §R6.2, §R6.4, §R6.5).

THE SPINE OF THE PROJECT
------------------------
*"Evaluation depends on Storage, never on the live model."* Every metric in this
project must be reproducible offline from cached responses. That property is not
maintained by discipline — it falls out of this module being the only way a
response can enter the run.

WHAT IS ADDRESSED
-----------------
The key is SHA-256 over a canonical body that contains TWO things:

    1. the CONDITION  - arm, K, B, B_mode, scale, reader
    2. the REQUEST    - model, messages, sampling params, seed

Both, always. The second is the obvious half and the first is hazard **H4**: a
prefix cached at B=4096 must never be served at B=65536, and the two produce
genuinely different prompts. Hashing only the request text makes those collide
whenever the prompt happens to be identical, which is precisely the case that
matters. `test_the_same_prompt_at_two_budgets_does_not_collide` pins it.

WHY `seed` IS INSIDE THE HASH
-----------------------------
It looks like it should be outside, so repeated runs can reuse cached responses.
But then two replicates of the same cell at the same temperature produce the SAME
key, dedupe to one file, and seed variance — which PRD §R7.1 requires to vary
something — becomes impossible to estimate without any error being raised.
Deduplication is for *identical* work; two seeds are not identical work.

ATOMIC WRITES, BECAUSE `kill -9` IS A TEST
------------------------------------------
Every write goes to a temp file and is then `os.replace`d into place. A reader
either sees a complete record or no record — never a half-written one. This is
what makes R5.2's "kill -9 mid-run, resume, identical results" achievable at all:
the resume path's correctness rests on there being no state where a file exists and
is truncated.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STORE_SCHEMA = 1

#: Keys the canonical body always carries from the condition. Named explicitly so
#: that adding a field to `ConditionKey` without adding it here is a type error
#: rather than a silent cache collision.
CONDITION_FIELDS: tuple[str, ...] = ("arm", "k", "b", "b_mode", "scale", "reader")


class DiskBudgetExceededError(RuntimeError):
    """The store would exceed its declared budget.

    ABORT rather than trim. Silently dropping the oldest responses makes a run
    non-reproducible in a way that only shows up as a missing cell weeks later,
    and the budget exists precisely so the disk filling up is a loud failure at the
    moment it happens (PRD §R6.4).
    """


@dataclass(frozen=True, slots=True)
class ConditionKey:
    """The whole condition. Namespace for the cache key (PRD §R5.6, §R6.5)."""

    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str

    @property
    def key(self) -> tuple[str, int, float, str, str, str]:
        return (self.arm, self.k, round(self.b, 6), self.b_mode, self.scale, self.reader)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "scale": self.scale,
            "reader": self.reader,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConditionKey:
        return cls(
            arm=str(raw["arm"]),
            k=int(raw["K"]),
            b=float(raw["B"]),
            b_mode=str(raw["B_mode"]),
            scale=str(raw["scale"]),
            reader=str(raw["reader"]),
        )


def canonical_body(
    condition: ConditionKey,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """The exact bytes that get hashed. Explicit, never derived from `__dict__`.

    Building this by hand means adding a field to the condition or the request is a
    DELIBERATE act that changes the cache key. A reflection-based version would let
    a new field change or not change the key depending on how it was added, and a
    cache that silently reuses responses for a changed request is the worst failure
    this project has.
    """
    body: dict[str, Any] = {
        "schema_version": STORE_SCHEMA,
        "condition": condition.as_dict(),
        "model": request["model"],
        "messages": request["messages"],
        "max_tokens": request.get("max_tokens", 512),
        "temperature": request.get("temperature", 0.0),
        "top_p": request.get("top_p", 1.0),
        "stop": list(request.get("stop", []) or []),
        "seed": request.get("seed", 0),
    }
    return body


def canonical_json(body: Mapping[str, Any]) -> str:
    """Stable serialisation: sorted keys, no incidental whitespace, UTF-8 intact.

    `ensure_ascii=False` matters — the corpus is not ASCII, and escaping it would
    make the hash depend on an encoding choice that has nothing to do with the
    request.
    """
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(condition: ConditionKey, request: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json(canonical_body(condition, request)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredResponse:
    """One response, as it will be read back during analysis.

    `raw` is the provider's JSON VERBATIM (PRD §R6.2). Not a normalised summary:
    a summary cannot be re-derived into a different one, and the whole reason to
    keep responses is that the metric code will change and want something the
    earlier code did not think to save.
    """

    key: str
    condition: dict[str, Any]
    request: dict[str, Any]
    text: str
    finish_reason: str
    model_version: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    queue_wait_ms: float | None = None
    """Recorded SEPARATELY from `latency_ms`. Otherwise concurrency (PRD §R5.3)
    shows up as server slowness, and the latency we report is a property of our
    own scheduling rather than of the thing being measured."""
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    qid: str = ""
    """Which question this response answers.

    METADATA, deliberately NOT part of `canonical_body`. The cache key is
    question-agnostic on purpose - two replicates of one condition produce an
    identical request and must hit the cache once - but a record that cannot say
    which question it belongs to cannot be scored. `recall@K` joins on `(condition,
    qid)` and `msat score` joins on `(condition_key, qid)`, so the field has to
    travel with the body.

    Kept off the key because the two concerns differ: the key answers "have I
    already generated this?", the record answers "for what?". Putting a qid in the
    key would turn every replicate into a fresh call and make the 3x replicate
    design measure the provider three times over.
    """

    replicates: tuple[int, ...] = ()
    """Which replicates resolved to this one response.

    Non-empty and longer than 1 means the replicates were CACHED rather than
    generated. That is correct behaviour and also a fact about the run worth
    keeping: a run reporting 3 replicates of a cell may have paid for one, and
    reporting the apparent replicate count as if it were the independent sample
    count would overstate the evidence by 3x.
    """

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA,
            "key": self.key,
            "condition": self.condition,
            "request": self.request,
            "response": {
                "text": self.text,
                "finish_reason": self.finish_reason,
                "model_version": self.model_version,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "latency_ms": self.latency_ms,
                "ttft_ms": self.ttft_ms,
                "queue_wait_ms": self.queue_wait_ms,
                "backend": self.backend,
                "raw": self.raw,
            },
            # Provenance, OUTSIDE `response`. It is not part of the generated text
            # and it is not part of the key; keeping it in its own top-level block
            # means a reader can tell at a glance which fields describe the model
            # and which describe the experiment.
            "qid": self.qid,
            "replicates": list(self.replicates),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StoredResponse:
        r = d["response"]
        return cls(
            key=str(d["key"]),
            condition=dict(d["condition"]),
            request=dict(d["request"]),
            text=str(r["text"]),
            finish_reason=str(r["finish_reason"]),
            model_version=str(r["model_version"]),
            prompt_tokens=int(r["prompt_tokens"]),
            completion_tokens=int(r["completion_tokens"]),
            latency_ms=float(r["latency_ms"]),
            ttft_ms=r.get("ttft_ms"),
            queue_wait_ms=r.get("queue_wait_ms"),
            backend=str(r.get("backend", "")),
            raw=dict(r.get("raw", {})),
            qid=str(d.get("qid", r.get("qid", ""))),
            replicates=tuple(int(x) for x in d.get("replicates", r.get("replicates", ()) or ())),
        )


def field_default() -> Any:
    return None


class ResponseStore:
    """Sharded content-addressed store with a hard disk budget.

    SHARDING IS NOT COSMETIC. The full grid is ~189,000 reader responses plus the
    judge panel. A flat directory with that many entries is slow on Windows and
    slow on Linux, and it fails in a way that looks like the filesystem's fault.
    Two hex characters gives 256 directories and a bounded directory size.
    """

    def __init__(self, root: Path, *, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.responses = self.root / "responses"
        self.responses.mkdir(parents=True, exist_ok=True)

    # ── addressing ────────────────────────────────────────────────────────────
    def path_for(self, key: str) -> Path:
        return self.responses / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    # ── reading ───────────────────────────────────────────────────────────────
    def get(self, key: str) -> StoredResponse | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return StoredResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError):
            # A truncated record must read as ABSENT, not as a crash. The atomic
            # write makes truncation near-impossible, and a partially written file
            # left by a killed process on a filesystem without rename guarantees
            # is exactly the case this path exists for.
            return None

    def iter_all(self) -> Iterator[StoredResponse]:
        for path in sorted(self.responses.rglob("*.json")):
            try:
                yield StoredResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                continue

    def __len__(self) -> int:
        return sum(1 for _ in self.responses.rglob("*.json"))

    # ── writing ───────────────────────────────────────────────────────────────
    def put(self, record: StoredResponse) -> Path:
        self._check_budget(record)
        return self._atomic_write(self.path_for(record.key), record.as_dict())

    def _atomic_write(self, dest: Path, payload: Mapping[str, Any]) -> Path:
        """Write, then rename into place. The rename is the commit point."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(payload, fh, sort_keys=True, ensure_ascii=False, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return dest

    # ── the disk budget (R6.4) ───────────────────────────────────────────────
    def used_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.responses.rglob("*.json"))

    def _check_budget(self, incoming: StoredResponse) -> None:
        if self.max_bytes is None:
            return
        projected = self.used_bytes() + len(canonical_json(incoming.as_dict()))
        if projected > self.max_bytes:
            raise DiskBudgetExceededError(
                f"writing {incoming.key[:12]} would take the store to "
                f"{projected:,} bytes, over the {self.max_bytes:,} budget. "
                f"Aborting rather than trimming: a store that silently drops old "
                f"responses is not reproducible, and the budget exists so this "
                f"fails loudly now."
            )
