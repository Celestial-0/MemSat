"""Validated question inputs and benchmark session loaders."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from msat.arms.negative import PooledMemory
from msat.arms.render import Memory
from msat.datasets.longmemeval import (
    available_splits,
    iter_questions,
)
from msat.datasets.schema import Question


class MissingInputError(RuntimeError):
    """Raised when required dataset or control inputs are missing."""


def load_control_pool(path: Path | str) -> tuple[PooledMemory, dict[str, str]]:
    """Load the cross-question memory pool for Arm E."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"control pool file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object mapping qid -> {{question, memories}}.")
    by_question: dict[str, list[Memory]] = {}
    question_text: dict[str, str] = {}
    for qid, entry in raw.items():
        if not isinstance(entry, dict) or "memories" not in entry:
            raise ValueError(f"{path}: entry {qid!r} must be an object with a `memories` list.")
        question_text[str(qid)] = str(entry.get("question", ""))
        memories = [
            Memory(
                memory_id=str(raw_mem["memory_id"]),
                session_id=str(raw_mem["session_id"]),
                source_question_id=str(qid),
                text=str(raw_mem.get("text", "")),
                n_tokens=int(raw_mem.get("n_tokens", 0)),
            )
            for raw_mem in entry["memories"]
        ]
        by_question[str(qid)] = memories
    return PooledMemory(by_question=by_question), question_text


def load_gold_answers(path: Path | str) -> dict[str, str]:
    """Load gold answer text per question."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"gold answers file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    answers: dict[str, str] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if isinstance(v, str):
                answers[str(k)] = v
            elif isinstance(v, list) and v:
                answers[str(k)] = str(v[0])
            elif isinstance(v, dict) and "answer" in v:
                ans = v["answer"]
                answers[str(k)] = str(ans[0] if isinstance(ans, list) else ans)
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                qid = str(item.get("question_id") or item.get("qid") or "")
                ans = item.get("answer") or item.get("gold_answer") or ""
                if qid and ans:
                    answers[qid] = str(ans[0] if isinstance(ans, list) else ans)
    return answers


def load_gold_sessions(path: Path | str) -> dict[str, list[str]]:
    """Load mapping of qid -> [answer_session_ids]."""
    path = Path(path)
    if not path.is_file():
        raise MissingInputError(f"gold sessions file not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be a JSON object mapping qid -> [answer_session_ids].")
    return {str(k): [str(s) for s in v] for k, v in raw.items()}


@dataclass
class RunInputs:
    """Validated inputs for a run lifecycle execution."""

    questions: dict[str, Question] = field(default_factory=dict)
    questions_by_scale: dict[tuple[str, str], Question] = field(default_factory=dict)
    pool: PooledMemory | None = None
    question_text: dict[str, str] = field(default_factory=dict)
    gold_answers: dict[str, str] = field(default_factory=dict)
    gold_sessions: dict[str, list[str]] = field(default_factory=dict)
    filler_sessions: tuple[str, ...] = ()
    is_synthetic: bool = False
    allow_test_adapters: bool = False

    def __post_init__(self) -> None:
        if self.questions and not self.questions_by_scale:
            for q in self.questions.values():
                scale = q.scale or "S"
                self.questions_by_scale[(scale, q.qid)] = q
        elif self.questions_by_scale:
            for (_scale, qid), q in self.questions_by_scale.items():
                if qid not in self.questions:
                    self.questions[qid] = q

    def qids_by_scale(self) -> dict[str, list[str]]:
        """Map scale (e.g. 'S', 'M') to list of question IDs preserving both scales."""
        mapping: dict[str, list[str]] = {}
        if self.questions_by_scale:
            for scale, qid in self.questions_by_scale:
                lst = mapping.setdefault(scale, [])
                if qid not in lst:
                    lst.append(qid)
        for qid, q in self.questions.items():
            scale = q.scale or "S"
            lst = mapping.setdefault(scale, [])
            if qid not in lst:
                lst.append(qid)
        if not mapping and self.gold_sessions:
            # Default to Scale S when questions only define gold sessions
            mapping["S"] = list(self.gold_sessions.keys())
        return mapping

    def get_question(self, qid: str, scale: str | None = None) -> Question | None:
        if scale is not None and (scale, qid) in self.questions_by_scale:
            return self.questions_by_scale[(scale, qid)]
        if qid in self.questions:
            q = self.questions[qid]
            if scale is None or (q.scale or "S") == scale:
                return q
        for (s, q_id), q in self.questions_by_scale.items():
            if q_id == qid and (scale is None or s == scale):
                return q
        return None

    def get_question_text(self, qid: str, scale: str | None = None) -> str:
        q = self.get_question(qid, scale=scale)
        if q and q.question:
            return q.question
        if self.question_text.get(qid):
            return self.question_text[qid]
        return ""

    def get_gold_sessions(self, qid: str, scale: str | None = None) -> frozenset[str]:
        q = self.get_question(qid, scale=scale)
        if q and q.gold_session_ids:
            return frozenset(q.gold_session_ids)
        if qid in self.gold_sessions:
            return frozenset(self.gold_sessions[qid])
        return frozenset()

    def get_gold_answers(self, qid: str, scale: str | None = None) -> tuple[str, ...]:
        q = self.get_question(qid, scale=scale)
        if q and q.gold_answers:
            return tuple(str(a) for a in q.gold_answers if str(a))
        if qid in self.gold_answers:
            ans = self.gold_answers[qid]
            if isinstance(ans, str) and ans:
                return (ans,)
            if isinstance(ans, (list, tuple)):
                return tuple(str(a) for a in ans if str(a))
        return ()

    def get_gold_answer(self, qid: str, scale: str | None = None) -> str | None:
        answers = self.get_gold_answers(qid, scale=scale)
        return answers[0] if answers else None

    def build_control_pool(self, *, scale: str = "S") -> PooledMemory:
        """Derive cross-question memory pool from loaded questions."""
        if self.pool is not None:
            return self.pool
        by_question: dict[str, list[Memory]] = {}
        all_questions = (
            list(self.questions_by_scale.values())
            if self.questions_by_scale
            else list(self.questions.values())
        )
        for q in all_questions:
            if not q.sessions or (q.scale or "S") != scale:
                continue
            memories: list[Memory] = []
            for s in q.sessions:
                text = " ".join(t.content for t in s.turns if t.content).strip()
                n_tokens = max(1, len(text) // 4)
                memories.append(
                    Memory(
                        memory_id=s.session_id,
                        session_id=s.session_id,
                        source_question_id=q.qid,
                        text=text,
                        n_tokens=n_tokens,
                    )
                )
            by_question[q.qid] = memories
        self.pool = PooledMemory(by_question=by_question)
        return self.pool

    @classmethod
    def from_legacy_paths(
        cls,
        *,
        questions_path: Path | str | None = None,
        gold_path: Path | str | None = None,
        pool_path: Path | str | None = None,
        filler_path: Path | str | None = None,
        data_dir: Path | str | None = None,
        scales: Sequence[str] | None = None,
        is_synthetic: bool = False,
        allow_test_adapters: bool = False,
    ) -> RunInputs:
        """Build inputs from explicit paths passed by CLI or test harnesses."""
        questions: dict[str, Question] = {}
        questions_by_scale: dict[tuple[str, str], Question] = {}
        question_text: dict[str, str] = {}
        gold_sessions: dict[str, list[str]] = {}
        gold_answers: dict[str, str] = {}
        pool: PooledMemory | None = None
        filler_sessions: tuple[str, ...] = ()

        if questions_path is not None:
            q_path = Path(questions_path)
            if not q_path.is_file():
                raise MissingInputError(f"questions file not found: {q_path}")
            with q_path.open("r", encoding="utf-8-sig") as stream:
                mapping_input = stream.read(4096).lstrip().startswith("{")
            if mapping_input:
                gold_sessions.update(load_gold_sessions(q_path))
            else:
                for q in iter_questions(q_path):
                    scale = q.scale or "S"
                    if (scale, q.qid) in questions_by_scale:
                        raise ValueError("Duplicate question identity in benchmark input.")
                    questions_by_scale[(scale, q.qid)] = q
                    questions[q.qid] = q
                    question_text[q.qid] = q.question
                    if q.gold_session_ids:
                        gold_sessions[q.qid] = list(q.gold_session_ids)
                    if q.gold_answers:
                        gold_answers[q.qid] = q.gold_answers[0]

        if gold_path is not None:
            g_path = Path(gold_path)
            if not g_path.is_file():
                raise MissingInputError(f"gold answers file not found: {g_path}")
            gold_answers.update(load_gold_answers(g_path))

        if pool_path is not None:
            p_path = Path(pool_path)
            if not p_path.is_file():
                raise MissingInputError(f"control pool file not found: {p_path}")
            pool, p_text = load_control_pool(p_path)
            question_text.update(p_text)

        if filler_path is not None:
            f_path = Path(filler_path)
            if not f_path.is_file():
                raise MissingInputError(f"filler file not found: {f_path}")
            from msat.arms.padding import FillerSource

            filler = FillerSource.load(f_path)
            filler_sessions = filler.sessions

        # Only auto-load data_dir when explicitly passed
        if data_dir is not None:
            resolved_dir = Path(data_dir)
            if not resolved_dir.is_dir():
                raise MissingInputError(f"data directory not found: {resolved_dir}")
            for split, p in available_splits(resolved_dir).items():
                if split in ("S", "M") and (scales is None or split in scales):
                    for q in iter_questions(p):
                        scale = q.scale or split
                        questions_by_scale[(scale, q.qid)] = q
                        if q.qid not in questions:
                            questions[q.qid] = q
                            question_text.setdefault(q.qid, q.question)
                            if q.gold_session_ids:
                                gold_sessions.setdefault(q.qid, list(q.gold_session_ids))
                            if q.gold_answers:
                                gold_answers.setdefault(q.qid, q.gold_answers[0])

        return cls(
            questions=questions,
            questions_by_scale=questions_by_scale,
            pool=pool,
            question_text=question_text,
            gold_answers=gold_answers,
            gold_sessions=gold_sessions,
            filler_sessions=filler_sessions,
            is_synthetic=is_synthetic,
            allow_test_adapters=allow_test_adapters,
        )

    @classmethod
    def from_synthetic(
        cls,
        questions: Sequence[Question] = (),
        *,
        pool: PooledMemory | None = None,
        gold_answers: Mapping[str, str] | None = None,
        gold_sessions: Mapping[str, Sequence[str]] | None = None,
        filler_sessions: Sequence[str] = (),
    ) -> RunInputs:
        """Create explicit synthetic test inputs."""
        q_dict: dict[str, Question] = {}
        q_by_scale: dict[tuple[str, str], Question] = {}
        q_text: dict[str, str] = {}
        for q in questions:
            scale = q.scale or "S"
            q_by_scale[(scale, q.qid)] = q
            q_dict[q.qid] = q
            q_text[q.qid] = q.question

        g_answers = dict(gold_answers or {})
        g_sessions = {k: list(v) for k, v in (gold_sessions or {}).items()}
        for q in questions:
            if q.gold_answers and q.qid not in g_answers:
                g_answers[q.qid] = q.gold_answers[0]
            if q.gold_session_ids and q.qid not in g_sessions:
                g_sessions[q.qid] = list(q.gold_session_ids)

        inputs = cls(
            questions=q_dict,
            questions_by_scale=q_by_scale,
            pool=pool,
            question_text=q_text,
            gold_answers=g_answers,
            gold_sessions=g_sessions,
            filler_sessions=tuple(filler_sessions),
            is_synthetic=True,
            allow_test_adapters=True,
        )
        if pool is None and (q_dict or q_by_scale):
            inputs.build_control_pool()
        return inputs
