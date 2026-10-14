"""Structured JSON logging for `msat` (PRD §N6).

Every line is one JSON object carrying the attribution triple — `run_id`,
`config_hash`, and the condition key `(arm, K, B, scale, reader, question_id)`.

> An unattributable log line produces an unattributable result.

Plain-text log lines are not optional: `grep`, `awk`, and a 12-tab review of a
126,000-call run all need structure, not prose. The logger is configured once at
process start by the CLI, not by the library modules that emit into it.
"""

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

#: Fields every log line should carry for attribution (PRD §N6).
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

#: `LogRecord` attributes that are not user fields. Filtering these is what keeps
#: the JSON payload from carrying `pathname`, `lineno`, `exc_text`, and friends
#: on every single line.
_RESERVED: Final[frozenset[str]] = frozenset(
    vars(logging.LogRecord("name", 0, "pathname", 0, "msg", (), None))
) | {"message", "asctime", "taskName"}

_configured = False


def _jsonable(value: Any) -> Any:
    """Coerce a log field to something `json.dumps` will accept.

    Paths and enums are common in this codebase; letting one reach `json.dumps`
    should raise at the call site, not silently truncate a log line mid-run.
    """
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
    """Render one `LogRecord` as a single-line JSON object."""

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
    """A logger that carries attribution fields for the duration of a block.

    Used as a context manager so `run_id` / `config_hash` are stamped once at the
    top of a run rather than threaded through every call site:

    ```python
    with log_for(run_id=run_id, config_hash=config_hash):
        ...
    ```
    """

    def process(  # type: ignore[override]  # signature fixed by stdlib
        self, msg: Any, kwargs: Any
    ) -> tuple[Any, Any]:
        extra = dict(self.extra or {})
        extra.update(kwargs.get("extra") or {})
        kwargs["extra"] = extra
        return msg, kwargs


@contextmanager
def log_for(**fields: Any) -> Iterator[ContextLogger]:
    """Bind attribution fields for the duration of the block."""
    ctx = ContextLogger(get_logger(), {})
    previous = ctx.extra
    ctx.extra = fields
    try:
        yield ctx
    finally:
        ctx.extra = previous


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger (or a child of it)."""
    return logging.getLogger(LOGGER_NAME if name is None else f"{LOGGER_NAME}.{name}")


def _utf8_stream(stream: Any) -> Any:
    """Force UTF-8 on a text stream, degrading rather than raising.

    On Windows, a redirected stderr (which is how `make` and CI capture logs)
    uses the ANSI code page — cp1252 here. One non-ASCII character in a logged
    answer string would then raise `UnicodeEncodeError` *inside the logging
    handler*, which is a spectacular way to lose a 126,000-call run. Errors are
    `backslashreplace`d so the line still lands and the run continues.
    """
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return stream
    with contextlib.suppress(ValueError, OSError, AttributeError):
        # Non-TextIOWrapper (a StringIO in tests) or a closed stream. Leave it be.
        reconfigure(encoding="utf-8", errors="backslashreplace")
    return stream


def configure_logging(
    *,
    level: str | int | None = None,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    """Install the JSON handler. Idempotent unless `force`.

    `level` defaults to `$MSAT_LOG_LEVEL`, then `INFO`. A debug level also turns
    on uv's own resolver chatter, which is useful when diagnosing `uv sync`.
    """
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
    """Return which `REQUIRED_FIELDS` are absent from `fields`.

    Exposed so tests — and a future lint — can assert that run-time log lines
    actually carry attribution, rather than trusting call sites to remember.
    """
    return [f for f in REQUIRED_FIELDS if f not in fields]
