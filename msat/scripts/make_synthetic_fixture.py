"""Write a tiny synthetic corpus so P3 is testable with no network and no GPU.

Three questions x three corpus sizes, because the guards are about SIZE and a
fixture with one size cannot exercise a K-guard at all. Generated rather than
checked in as blobs: a binary fixture nobody can read is a fixture nobody can
review, and the shape of the data is exactly what these tests are about.

Deliberately uneven. The three sizes are 3, 12 and 40 sessions so that:
  - K == n_sessions is reachable exactly at the smallest
  - K > n_sessions is reachable at every size
  - a percentile-based guard would PASS K=32 on the 40-session question (80% of it)
    while failing on the 3-session one - which is the entire K=64 lesson

Content is lexically varied per session so the embedder has something to rank:
each session repeats its own index as a token, which makes retrieval
deterministic and checkable without depending on an embedding model's behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

#: (question_id, n_sessions, n_tokens_per_session)
LAYOUT: tuple[tuple[str, int, int], ...] = (
    ("q_small", 3, 40),
    ("q_medium", 12, 30),
    ("q_large", 40, 25),
)

GOLD = {"q_small": "s0", "q_medium": "s0", "q_large": "s0"}


def _session_text(question_id: str, index: int, n_tokens: int) -> str:
    """Each session carries a token unique to it: `marker{question}_{index}`.

    The uniqueness is the point. A fixture where every session says the same words
    cannot test that retrieval RANKS anything - it only tests that the retriever
    returned something. With one distinguishing token per session, the question
    naming one of them has exactly one correct answer, so `gold_rank` is a real
    assertion rather than a coincidence.
    """
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
                # Names the gold session's marker, so exactly one session matches.
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


if __name__ == "__main__":  # pragma: no cover - manual regeneration
    import sys

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/fixtures/synthetic")
    print(write_fixture(target))
