"""Synthetic corpus fixture implementation for plan and offline testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("q_small", 3, 40),
    ("q_medium", 12, 30),
    ("q_large", 40, 25),
)

GOLD: dict[str, str] = {"q_small": "s0", "q_medium": "s0", "q_large": "s0"}


def _session_text(question_id: str, index: int, n_tokens: int) -> str:
    marker = f"marker{question_id}_{index}"
    filler = " ".join(f"{question_id}filler{index}_{j}" for j in range(max(1, n_tokens // 8)))
    return f"{question_id} session {index}. discussed {marker}. {filler}"


def build() -> dict[str, Any]:
    questions = []
    for qid, n_sessions, n_tokens in LAYOUT:
        sessions = [
            {"session_id": f"{qid}_s{i}", "text": _session_text(qid, i, n_tokens)}
            for i in range(n_sessions)
        ]
        questions.append(
            {
                "question_id": qid,
                "question": f"Which session discussed marker{qid}_0?",
                "answer": f"{qid} discussed marker{qid}_0",
                "sessions": sessions,
                "gold_session_ids": [GOLD[qid]],
            }
        )
    return {"split": "synthetic", "questions": questions}


def write_fixture(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "synthetic.json"
    out.write_text(json.dumps(build(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_synthetic() -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = build()["questions"]
    return questions
