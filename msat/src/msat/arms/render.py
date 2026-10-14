"""Arm rendering (PRD §R4.1, §R4.2).

ARMS ARE DATA
-------------
`configs/arms.yaml` names a template and `configs/prompts/*.j2` holds the text.
Adding an arm needs no pipeline change, which is the property that lets the design
grow arms during the pilot without every addition touching code.

PURITY IS A REQUIREMENT, NOT A STYLE
------------------------------------
`render(question, condition) -> RenderedPrompt` takes its inputs as arguments and
reads nothing else: no globals, no network, no file I/O, no clock, no environment.
That is checkable, and `tests/test_arms_p5.py` checks it — by rendering twice with
the filesystem made unreadable and by asserting the result is byte-identical.

It matters because a purity leak is silent. A template that reached for `os.environ`
would render differently on the execution box than in the test, and the difference
would appear as a result nobody could reproduce.

THE MANIFEST IS THE GROUND TRUTH
--------------------------------
Every render emits ordered memory IDs with their ranks, plus K, B, B_mode and
corpus_fraction. That manifest — not a reimplementation of the retrieval logic —
is what `recall@K` and the gold-rank analysis read. Reimplementing the selector
"just for the metrics" is how the metric and the thing it measures drift apart,
and the drift is invisible because both look plausible.
"""

from __future__ import annotations

from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

from msat.store.responses import ConditionKey

#: Filled from `configs/pricing/latest.yaml`. A module-level default so a renderer
#: can be constructed in a test without a config, and so the budget is ONE number
#: rather than re-derived per call site.
DEFAULT_PER_MEMORY_TOKENS = 256


@dataclass(frozen=True, slots=True)
class Memory:
    """One retrievable memory, with the provenance the guards need.

    `session_id` is REQUIRED and `is_gold` is GONE, deliberately.

    Gold is a property of a SESSION (`answer_session_ids`), but a memory is a
    CHUNK of one. Recomputing "is this gold?" by parsing a chunk id back into its
    session, or by trusting a boolean someone set at construction, are the two ways
    the headline metric silently becomes wrong. This project has already paid for
    the second failure mode three times — a value that drifted away from the thing
    it describes, with nothing watching.

    So the single source of truth is `session_id`, and gold is DERIVED:
    `memory.is_gold_for(gold_sessions)`. Every call site has to state which session
    a chunk came from, and nothing has to remember to set a flag.
    """

    memory_id: str
    session_id: str
    source_question_id: str
    text: str
    n_tokens: int

    def is_gold_for(self, gold_sessions: Container[str]) -> bool:
        """Derived, never stored. `gold_sessions` is `answer_session_ids`."""
        return self.session_id in gold_sessions

    @property
    def gold_source(self) -> str:
        return self.source_question_id


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    """What an arm produces. Typed, hashable, and complete."""

    arm: str
    condition: ConditionKey
    system: str
    user: str
    memory_ids: tuple[str, ...]
    """Ordered, rank-ordered. The manifest for `recall@K` and gold-rank work."""
    ranks: tuple[int, ...]
    """1-based rank of each memory in the rendered prompt."""
    k: int
    b: float
    b_mode: str
    corpus_fraction: float
    position_variant: str = "natural"
    evidence_survival: float = 1.0
    """Fraction of gold sessions present in the rendered prompt (arms that cut).

    Defaults to 1.0 because every arm that does not DISCARD context keeps all of
    it. It exists so that `full_context_max` can be read alongside arms that do,
    and so an arm that silently drops evidence cannot look identical to one that
    does not.
    """
    qid: str = ""
    """Which question this render is for.

    Carried here so `recall@K` never has to recover it from a memory id. A
    rendered prompt that does not say which question it answers cannot be scored,
    and recovering it by parsing ids is exactly the kind of guess that goes wrong
    quietly.
    """
    session_ids: tuple[str, ...] = ()
    """The SESSION each memory came from, parallel to `memory_ids`.

    Gold is a property of a session and a memory is a chunk of one, so this tuple
    - not `memory_ids` - is what `recall@K` is computed over. Emitting it makes the
    manifest self-describing per PRD §R4.2, and means recall needs no re-retrieval.
    """

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
        """`(rank, session_id)` pairs, in RANK order.

        The single place the pairing is formed. `gold_ranks` and
        `ordered_session_ids` must agree, and they agree because both call this
        rather than each sorting `zip(ranks, sessions)` on their own - the two
        zips disagree the moment list order differs from rank order, and they
        disagree SILENTLY.
        """
        return sorted(zip(self.ranks, self.session_ids, strict=True))

    def ordered_session_ids(self) -> tuple[str, ...]:
        """Session ids in RANK order.

        Sorted by `ranks` rather than trusted positionally, so a prompt whose
        memories are listed in a different order than they are ranked still scores
        the right rank. Ranks are validated as a permutation, so the sort is total.
        """
        if not self.ranks:
            return self.session_ids
        return tuple(sid for _, sid in self._ranked_pairs())

    def gold_ranks(self, gold_sessions: frozenset[str]) -> tuple[int, ...]:
        """Ranks at which GOLD SESSIONS first appear. Empty means not retrieved.

        Matches on `session_id`, not `memory_id`, and takes the FIRST appearance of
        each gold session. The earlier version matched chunk ids against gold, so a
        gold session split into three chunks was reported as three separate gold
        ranks and the gold-rank histogram became a function of the chunk size.
        """
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
    """An arm is a pure function of its arguments (PRD §R4.1)."""

    id: str
    isolates: str
    template: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        **kwargs: Any,
    ) -> RenderedPrompt: ...


class TemplateError(RuntimeError):
    """A template referenced a variable the renderer was not given.

    `StrictUndefined` makes this an error instead of an empty string. An
    undefined `{{ memories }}` rendering as empty would produce a prompt with no
    evidence at all — which looks exactly like a retrieval failure and is not one.
    """


def strip_leading_jinja_comment(source: str) -> str:
    """Remove a leading `{# ... #}` block so the rest parses as YAML.

    The templates document themselves in a Jinja comment, and `{#` is not valid
    YAML. Only a LEADING comment is removed — one appearing inside a block scalar
    is content and must survive. Documentation lives above the structure, never
    inside it.
    """
    text = source.lstrip()
    if not text.startswith("{#"):
        return source
    end = text.find("#}")
    return source if end == -1 else text[end + 2 :]


def render_template(source: str, context: Mapping[str, Any]) -> tuple[str, str]:
    """Render a `.j2` arm template to `(system, user)`.

    PARSE FIRST, THEN RENDER. The template's structure is parsed as YAML, and only
    the resulting `system` / `user` STRINGS are handed to Jinja.

    The obvious order - render, then parse - is wrong, and wrong in a way that only
    shows up with real content. A memory block is multi-line, so substituting
    `{{ memories }}` into an indented block scalar puts every line after the first
    at column 0. YAML then reads `[3] memory 2 ...` as a top-level key and the
    render fails outright. It works with single-line fixtures and breaks on the
    first real memory, which is the worst possible moment to find out.

    Parsing first also means a Jinja expression inside a memory's TEXT can never
    escape into the template.
    """
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
    """Render the numbered memory block, applying the per-memory budget.

    The budget truncates EACH memory, which is what makes it independent of K:
    changing B changes density without changing how many memories were retrieved.
    An earlier design truncated the concatenated list, which made B a cutoff on K
    expressed in tokens and left 78 of 126 grid cells rendering identical prompts.
    """
    blocks: list[str] = []
    for rank, memory in enumerate(memories, start=1):
        text = memory.text
        if budget_tokens is not None and memory.n_tokens > budget_tokens:
            text = _truncate_to_tokens(text, budget_tokens)
        blocks.append(f"[{rank}] {text}")
    return "\n\n".join(blocks)


def _truncate_to_tokens(text: str, budget_tokens: int) -> str:
    """Character truncation at the same ~4 chars/token ratio the config declares.

    Using the SAME ratio as `chunk.approx_tokens` matters: a renderer and a guard
    that count differently reject the wrong cells and size the wrong prompts, and
    the disagreement is invisible.
    """
    return text[: max(1, budget_tokens * 4)].rstrip()


@dataclass(frozen=True, slots=True)
class SimpleArm:
    """The plain arm: natural top-K in rank order.

    Deliberately the plainest thing in the set. Anything clever here would become
    a confound, because a curve that moves could then be attributed to retrieval
    depth or to the template, and there is no way to tell them apart.
    """

    id: str
    isolates: str
    template: str
    prompt_template_name: str = ""

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
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
    """Renders EXACTLY like another arm, differing only in its `id`.

    Arms B, E and F are manipulated in the MEMORY SET, not in the prompt: they pad,
    they randomise, or they supply the oracle's evidence sessions, and the template
    is held identical to the treatment on purpose. For E that is the whole point -
    a collapse has to be attributable to the absence of evidence rather than to a
    template change.

    `arms.yaml` names `arm_e.j2` for arm E, which reads like a prompt but is
    actually E's CONFIG block (`source`, `seed`, `exclude_gold`, the pass
    thresholds) parsed by `ControlConfig.from_template`. So these arms inherit a
    template and declare only their identity.

    Making the byte-identity a TYPE-level fact rather than a convention is the
    point: a test can assert two rendered prompts are equal, but a type cannot drift
    away from it.
    """

    id: str
    isolates: str

    def render(
        self,
        question: str,
        memories: Sequence[Memory],
        condition: ConditionKey,
        **kwargs: Any,
    ) -> RenderedPrompt:
        out = SimpleArm.render(self, question, memories, condition, **kwargs)
        # Explicit base call, NOT `super().render(...)`: with `slots=True` the
        # dataclass decorator REBUILDS the class, which breaks the implicit
        # `__class__` cell a zero-argument `super()` depends on. The failure is a
        # runtime `TypeError: super(type, obj)`, which is a confusing way to learn
        # about a slotted dataclass.
        return replace(out, arm=self.id, condition=replace(condition, arm=self.id))


@dataclass
class ArmRegistry:
    """Arms as data. Lookup by id, no dispatch table in code."""

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
    """Assemble arms from config. A missing template is an error at build time."""
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
