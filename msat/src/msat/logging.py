"""Structured JSON logging for `msat`."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, Final

LOGGER_NAME: Final = "msat"

REQUIRED_FIELDS: Final[tuple[str, ...]] = (
    "run_id",
    "config_hash",
    "arm",
    "K",
    "B",
    "scale",
    "reader",
    "question_id",
)

_RESERVED: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("name", 0, "pathname", 0, "msg", (), None))
) | {"message", "asctime", "taskName"}

_configured = False


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, os.PathLike):
        return str(value)
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _jsonable(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


class ContextLogger(logging.LoggerAdapter):
    def process(  # type: ignore[override]  # signature fixed by stdlib
        self, msg: Any, kwargs: Any
    ) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


@contextmanager
def log_for(**fields: Any) -> Iterator[ContextLogger]:
    ctx = ContextLogger(get_logger(), {})
    previous = ctx.extra
    ctx.extra = fields
    try:
        yield ctx
    finally:
        ctx.extra = previous


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def _utf8_stream(stream: Any) -> Any:
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream
    with contextlib.suppress(ValueError, OSError, AttributeError):
        reconfigure(encoding="utf-8", errors="backslashreplace")
    return stream


def configure_logging(
    *,
    level: str | int | None = None,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    global _configured
    if _configured and not force:
        return get_logger()

    resolved = level if level is not None else os.environ.get("MSAT_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    logger = get_logger()
    logger.setLevel(resolved)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    target = _utf8_stream(stream if stream is not None else sys.stderr)
    handler = logging.StreamHandler(target)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    _configured = True
    return logger


def missing_attribution(fields: dict[str, Any]) -> list[str]:
    return [f for f in REQUIRED_FIELDS if f not in fields]
