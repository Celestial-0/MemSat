"""Content-addressed response store."""

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

CONDITION_FIELDS: tuple[str, ...] = ("arm", "k", "b", "b_mode", "scale", "reader")


class DiskBudgetExceededError(RuntimeError):
    """The store would exceed its declared disk budget."""


@dataclass(frozen=True, slots=True)
class Condition:
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
    def from_dict(cls, raw: Mapping[str, Any]) -> Condition:
        k_val = raw.get("k") if "k" in raw else raw.get("K", 0)
        b_val = raw.get("b") if "b" in raw else raw.get("B", 0.0)
        b_mode_val = raw.get("b_mode") if "b_mode" in raw else raw.get("B_mode", "")
        return cls(
            arm=str(raw["arm"]),
            k=int(k_val or 0),
            b=float(b_val or 0.0),
            b_mode=str(b_mode_val or ""),
            scale=str(raw["scale"]),
            reader=str(raw["reader"]),
        )


def canonical_body(
    condition: Condition,
    request: Mapping[str, Any],
) -> dict[str, Any]:
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
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def cache_key(condition: Condition, request: Mapping[str, Any]) -> str:
    # The key covers the model but not the hardware, so a rerun on another GPU reuses it.
    return hashlib.sha256(
        canonical_json(canonical_body(condition, request)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class StoredResponse:
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
    backend: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    qid: str = ""
    repetitions: tuple[int, ...] = ()

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
            "qid": self.qid,
            "replicates": list(self.repetitions),
            "repetitions": list(self.repetitions),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> StoredResponse:
        r = d["response"]
        reps = d.get("repetitions")
        if reps is None:
            reps = d.get("replicates", r.get("repetitions", r.get("replicates", ())) or ())
        reps_tuple = tuple(int(x) for x in reps)
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
            repetitions=reps_tuple,
        )


def field_default() -> Any:
    return None


@dataclass
class Observation:
    cache_key: str
    condition_key: str
    qid: str
    metric: str
    value: float | None
    arm: str
    k: int
    b: float
    b_mode: str
    scale: str
    reader: str
    repetition: int | None = None
    cached: bool = False
    is_abstention: bool = False
    answerable: bool = True
    finish_reason: str = ""
    latency_ms: float | None = None
    ttft_ms: float | None = None
    queue_wait_ms: float | None = None
    config_hash: str | None = None
    scoring_config_hash: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STORE_SCHEMA,
            "cache_key": self.cache_key,
            "condition_key": self.condition_key,
            "qid": self.qid,
            "arm": self.arm,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "scale": self.scale,
            "reader": self.reader,
            "repetition": self.repetition,
            "cached": self.cached,
            "metric": self.metric,
            "value": self.value,
            "is_abstention": self.is_abstention,
            "answerable": self.answerable,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "ttft_ms": self.ttft_ms,
            "queue_wait_ms": self.queue_wait_ms,
            "config_hash": self.config_hash,
            "scoring_config_hash": self.scoring_config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        repetition = data.get("repetition")
        if repetition is None:
            repetition = data.get("replicate")
        return cls(
            cache_key=str(data["cache_key"]),
            condition_key=str(data["condition_key"]),
            qid=str(data["qid"]),
            metric=str(data["metric"]),
            value=data.get("value"),
            arm=str(data["arm"]),
            k=int(data.get("k", data.get("K", 0))),
            b=float(data.get("b", data.get("B", 0.0))),
            b_mode=str(data.get("b_mode", data.get("B_mode", ""))),
            scale=str(data["scale"]),
            reader=str(data["reader"]),
            repetition=repetition,
            cached=bool(data.get("cached", False)),
            is_abstention=bool(data.get("is_abstention", False)),
            answerable=bool(data.get("answerable", True)),
            finish_reason=str(data.get("finish_reason", "")),
            latency_ms=data.get("latency_ms"),
            ttft_ms=data.get("ttft_ms"),
            queue_wait_ms=data.get("queue_wait_ms"),
            config_hash=data.get("config_hash"),
            scoring_config_hash=data.get("scoring_config_hash"),
        )


class ResponseStore:
    def __init__(self, root: Path, *, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes
        self.responses = self.root / "responses"
        self.responses.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.responses / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        return self.path_for(key).exists()

    def get(self, key: str) -> StoredResponse | None:
        path = self.path_for(key)
        if not path.exists():
            return None
        try:
            return StoredResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def iter_all(self) -> Iterator[StoredResponse]:
        for path in sorted(self.responses.rglob("*.json")):
            try:
                yield StoredResponse.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, KeyError):
                continue

    def observations(self) -> Iterator[Observation]:
        for stored in self.iter_all():
            cond = Condition.from_dict(stored.condition)
            cond_key = "|".join(str(x) for x in cond.key)
            reps = stored.repetitions if stored.repetitions else (1,)
            for rep in reps:
                yield Observation(
                    cache_key=stored.key,
                    condition_key=cond_key,
                    qid=stored.qid or str(stored.request.get("qid", "")),
                    metric="response",
                    value=None,
                    arm=cond.arm,
                    k=cond.k,
                    b=cond.b,
                    b_mode=cond.b_mode,
                    scale=cond.scale,
                    reader=cond.reader,
                    repetition=rep,
                    cached=False,
                    finish_reason=stored.finish_reason,
                    latency_ms=stored.latency_ms,
                    ttft_ms=stored.ttft_ms,
                    queue_wait_ms=stored.queue_wait_ms,
                )

    def __len__(self) -> int:
        return sum(1 for _ in self.responses.rglob("*.json"))

    def put(self, record: StoredResponse) -> Path:
        self._check_budget(record)
        return self._atomic_write(self.path_for(record.key), record.as_dict())

    def _atomic_write(self, dest: Path, payload: Mapping[str, Any]) -> Path:
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


__all__ = [
    "CONDITION_FIELDS",
    "Condition",
    "DiskBudgetExceededError",
    "Observation",
    "ResponseStore",
    "StoredResponse",
    "cache_key",
    "canonical_body",
    "canonical_json",
]
