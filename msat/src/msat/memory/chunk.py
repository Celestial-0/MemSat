"""Retrieval units."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_WORDISH = re.compile(r"\S+")

SESSION_STARTS = re.compile(r"^\s*(?:user|assistant|human|system)\s*:", re.IGNORECASE)


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass(frozen=True, slots=True)
class Unit:
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
    out: list[Unit] = []
    for session_id, text in sessions:
        if unit == "session":
            if text.strip():
                out.append(session_unit(session_id, text))
        elif unit == "chunk":
            out.extend(chunk_session(session_id, text, target_tokens=target_tokens))
        elif unit == "fact":
            for i, turn in enumerate(_turns(text)):
                if turn.strip():
                    out.append(Unit.make(f"fact:{session_id}:{i}", "fact", session_id, turn))
        else:
            raise ValueError(f"unknown unit {unit!r}; expected session, chunk or fact")
    return out


def _turns(text: str) -> Sequence[str]:
    parts = SESSION_STARTS.split(text)
    return [p for p in parts if p.strip()]
