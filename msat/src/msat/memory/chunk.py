"""Retrieval units (PRD §R1.6).

`unit` is EXPLICIT and travels with every number: `session | chunk | fact`.

Why this is not a detail. K counts units. "Retrieve 16" means sixteen SESSIONS or
sixteen CHUNKS or sixteen FACTS, and those are different experiments with
different ceilings. The same K at two units is not the same condition, and a
result reported without its unit is uninterpretable - which is why `Retrieved`
below carries the label and cannot be constructed without it.

The baseline uses `session` as the default unit. That is a choice, not a
neutrality: sessions are the unit LongMemEval's gold sessions are expressed in, so
recall@K against `gold_rank` is comparable to the ceiling analysis in P1 without a
conversion that would quietly change what the numbers mean.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

#: Whitespace-token approximation. Deliberately crude, and the SAME approximation
#: the config's prompt model uses, so the K-guard and the renderer cannot disagree
#: about how big a unit is. A guard and a renderer that count differently reject
#: the wrong cells.
_WORDISH = re.compile(r"\S+")

SESSION_STARTS = re.compile(r"^\s*(?:user|assistant|human|system)\s*:", re.IGNORECASE)


def approx_tokens(text: str) -> int:
    """~4 chars per token, the same ratio `configs/pricing/latest.yaml` declares."""
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class Unit:
    """One retrievable thing, with its size."""

    unit_id: str
    unit_type: str
    source_id: str
    text: str
    n_tokens: int

    @classmethod
    def make(cls, unit_id: str, unit_type: str, source_id: str, text: str) -> Unit:
        return cls(
            unit_id=unit_id,
            unit_type=unit_type,
            source_id=source_id,
            text=text,
            n_tokens=approx_tokens(text),
        )


def session_unit(session_id: str, text: str) -> Unit:
    return Unit.make(f"session:{session_id}", "session", session_id, text)


def chunk_session(
    session_id: str,
    text: str,
    *,
    target_tokens: int = 256,
    overlap_tokens: int = 0,
) -> list[Unit]:
    """Split one session into overlapping chunks on turn boundaries.

    Splits on `user:`/`assistant:` prefixes where it can, and on whitespace where
    it cannot, so a session with no turn markers still chunks rather than
    returning one giant unit. A chunker that assumes well-formed turns silently
    produces one unit for malformed ones, and the K-guard then sees a corpus of
    size 1 where there are 400 sessions' worth of text.

    No cleverness: fixed size, deterministic, no sentence splitting. A weak
    chunker is acceptable; a changing one is not.
    """
    stripped = text.strip()
    if not stripped:
        return []

    target_chars = max(1, target_tokens * 4)
    overlap_chars = max(0, overlap_tokens * 4)

    chunks: list[str] = []
    if SESSION_STARTS.search(stripped) and len(stripped) > target_chars:
        parts = SESSION_STARTS.split(stripped)
        current = ""
        for part in parts:
            if len(current) + len(part) > target_chars and current:
                chunks.append(current.strip())
                tail = current[-overlap_chars:] if overlap_chars else ""
                current = tail + part
            else:
                current += part
        if current.strip():
            chunks.append(current.strip())
    else:
        step = max(1, target_chars - overlap_chars)
        for start in range(0, len(stripped), step):
            piece = stripped[start : start + target_chars]
            if piece.strip():
                chunks.append(piece.strip())
            if start + target_chars >= len(stripped):
                break

    return [
        Unit.make(f"chunk:{session_id}:{i}", "chunk", session_id, c) for i, c in enumerate(chunks)
    ]


def to_units(
    sessions: Iterable[tuple[str, str]],
    unit: str = "session",
    *,
    target_tokens: int = 256,
) -> list[Unit]:
    """Build the unit list a retriever will search over."""
    out: list[Unit] = []
    for session_id, text in sessions:
        if unit == "session":
            if text.strip():
                out.append(session_unit(session_id, text))
        elif unit == "chunk":
            out.extend(chunk_session(session_id, text, target_tokens=target_tokens))
        elif unit == "fact":
            # A fact is a single non-empty turn. Kept separate from chunking
            # because "one turn" and "one sentence" are different units and
            # conflating them changes what recall@K means.
            for i, turn in enumerate(_turns(text)):
                if turn.strip():
                    out.append(Unit.make(f"fact:{session_id}:{i}", "fact", session_id, turn))
        else:
            raise ValueError(f"unknown unit {unit!r}; expected session, chunk or fact")
    return out


def _turns(text: str) -> Sequence[str]:
    parts = SESSION_STARTS.split(text)
    return [p for p in parts if p.strip()]
