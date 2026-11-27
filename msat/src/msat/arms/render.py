"""Arm rendering."""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from msat.store import Condition

DEFAULT_PER_MEMORY_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Memory:
    memory_id: str
    session_id: str
    source_question_id: str
    text: str
    n_tokens: int

    def is_gold_for(self, gold_sessions: Container[str]) -> bool:
        return self.session_id in gold_sessions

    @property
    def gold_source(self) -> str:
        return self.source_question_id


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    arm: str
    condition: Condition
    system: str
    user: str
    memory_ids: tuple[str, ...]
    ranks: tuple[int, ...]
    k: int
    b: float
    b_mode: str
    corpus_fraction: float
    position_variant: str = "natural"
    evidence_survival: float = 1.0
    qid: str = ""
    session_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.session_ids) != len(self.memory_ids):
            raise ValueError(
                f"arm {self.arm!r}: {len(self.memory_ids)} memory ids but "
                f"{len(self.session_ids)} session ids. These are parallel tuples - a "
                f"length mismatch silently misaligns every rank, which would move "
                f"gold into the wrong rank without changing any count."
            )
        if len(self.ranks) != len(self.memory_ids):
            raise ValueError(
                f"arm {self.arm!r}: {len(self.memory_ids)} memories but "
                f"{len(self.ranks)} ranks. `ranks` is the ground truth for gold-rank "
                f"analysis and must line up exactly."
            )
        if sorted(self.ranks) != list(range(1, len(self.ranks) + 1)):
            raise ValueError(
                f"arm {self.arm!r}: ranks are {self.ranks[:5]}... Ranks must be a "
                f"PERMUTATION of 1..{len(self.ranks)} - 1-based, no gaps, no repeats. "
                f"A gap or a repeat means the ordering was computed from something "
                f"other than the rendered sequence."
            )

    def _ranked_pairs(self) -> list[tuple[int, str]]:
        return sorted(zip(self.ranks, self.session_ids, strict=True))

    def ordered_session_ids(self) -> tuple[str, ...]:
        # Rank decides order, not list position, so a reordered prompt still scores by rank.
        if not self.ranks:
            return self.session_ids
        return tuple(sid for _, sid in self._ranked_pairs())

    def gold_ranks(self, gold_sessions: frozenset[str]) -> tuple[int, ...]:
        seen: set[str] = set()
        out: list[int] = []
        for rank, sid in self._ranked_pairs():
            if sid in gold_sessions and sid not in seen:
                seen.add(sid)
                out.append(rank)
        return tuple(out)

    @property
    def prompt(self) -> str:
        return f"{self.system}\n\n{self.user}"

    @property
    def n_memories(self) -> int:
        return len(self.memory_ids)

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "condition": self.condition.as_dict(),
            "system": self.system,
            "user": self.user,
            "memory_ids": list(self.memory_ids),
            "session_ids": list(self.session_ids),
            "ranks": list(self.ranks),
            "qid": self.qid,
            "K": self.k,
            "B": self.b,
            "B_mode": self.b_mode,
            "corpus_fraction": self.corpus_fraction,
            "position_variant": self.position_variant,
            "evidence_survival": self.evidence_survival,
            "n_memories": self.n_memories,
        }


@runtime_checkable
class Arm(Protocol):
    id: str
    isolates: str
    template: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
        **kwargs: Any,
    ) -> RenderedPrompt: ...


class TemplateError(RuntimeError):
    """Raised when a template references a variable the renderer was not given."""


def strip_leading_jinja_comment(source: str) -> str:
    text = source.lstrip()
    if not text.startswith("{#"):
        return source
    end = text.find("#}")
    return source if end == -1 else text[end + 2 :]


def render_template(source: str, context: Mapping[str, Any]) -> tuple[str, str]:
    import yaml
    from jinja2 import StrictUndefined, Template
    from jinja2.exceptions import TemplateError as _JinjaError

    try:
        parsed: dict[str, Any] = yaml.safe_load(strip_leading_jinja_comment(source)) or {}
    except yaml.YAMLError as exc:
        raise TemplateError(f"template is not valid YAML: {exc}") from exc

    if "system" not in parsed or "user" not in parsed:
        raise TemplateError(
            f"template must define `system:` and `user:`; got {sorted(parsed)}. A "
            f"missing block renders as an empty prompt and looks like a retrieval "
            f"failure rather than a broken template."
        )

    def render_block(block: str) -> str:
        try:
            rendered: str = (
                Template(block, undefined=StrictUndefined, autoescape=False)
                .render(**context)
                .rstrip()
            )
        except _JinjaError as exc:
            raise TemplateError(f"template render failed: {exc}") from exc
        return rendered

    return render_block(str(parsed["system"])) + "\n", render_block(str(parsed["user"])) + "\n"


def format_memories(memories: Sequence[Memory], budget_tokens: int | None) -> str:
    blocks: list[str] = []
    for rank, memory in enumerate(memories, start=1):
        text = memory.text
        if budget_tokens is not None and memory.n_tokens > budget_tokens:
            text = _truncate_to_tokens(text, budget_tokens)
        blocks.append(f"[{rank}] {text}")
    return "\n\n".join(blocks)


def _truncate_to_tokens(text: str, budget_tokens: int) -> str:
    return text[: max(1, budget_tokens * 4)].rstrip()


@dataclass(frozen=True, slots=True)
class SimpleArm:
    id: str
    isolates: str
    template: str
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
        *,
        budget_tokens: int | None = DEFAULT_PER_MEMORY_TOKENS,
        corpus_fraction: float = 0.0,
        position_variant: str = "natural",
        qid: str = "",
        **_: Any,
    ) -> RenderedPrompt:
        system, user = render_template(
            self.template,
            {
                "memories": format_memories(memories, budget_tokens),
                "question": question,
                "K": condition.k,
                "B": condition.b,
                "arm": self.id,
            },
        )
        return RenderedPrompt(
            arm=self.id,
            condition=condition,
            system=system,
            user=user,
            memory_ids=tuple(m.memory_id for m in memories),
            session_ids=tuple(m.session_id for m in memories),
            ranks=tuple(range(1, len(memories) + 1)),
            k=condition.k,
            b=condition.b,
            b_mode=condition.b_mode,
            corpus_fraction=corpus_fraction,
            position_variant=position_variant,
            qid=qid,
        )


@dataclass(frozen=True, slots=True)
class DelegatingArm(SimpleArm):
    id: str
    isolates: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: Condition,
        **kwargs: Any,
    ) -> RenderedPrompt:
        out = SimpleArm.render(self, question, memories, condition, **kwargs)
        return replace(out, arm=self.id, condition=replace(condition, arm=self.id))


@dataclass
class ArmRegistry:
    arms: dict[str, SimpleArm] = field(default_factory=dict)

    def add(self, arm: SimpleArm) -> None:
        self.arms[arm.id] = arm

    def get(self, arm_id: str) -> SimpleArm:
        if arm_id not in self.arms:
            raise KeyError(f"unknown arm {arm_id!r}; known: {sorted(self.arms)}")
        return self.arms[arm_id]

    def enabled(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))


def build_registry(
    specs: Sequence[Mapping[str, Any]],
    templates: Mapping[str, str],
) -> ArmRegistry:
    registry = ArmRegistry()
    for spec in specs:
        if not spec.get("enabled", True):
            continue
        name = str(spec["prompt_template"])
        if name not in templates:
            raise KeyError(
                f"arm {spec['id']!r} names template {name!r}, which is not present. "
                f"A missing template renders as an empty prompt and reads as a "
                f"retrieval failure."
            )
        registry.add(
            SimpleArm(
                id=str(spec["id"]),
                isolates=str(spec["isolates"]),
                template=templates[name],
                prompt_template_name=name,
            )
        )
    return registry
