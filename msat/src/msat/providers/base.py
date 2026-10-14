"""Local inference backends.

One interface, N engines. The choice is a CONFIG value, not a hard binding,
because the best engine for throughput is not available on every platform:

| backend              | Windows | Linux | continuous batching | notes |
|----------------------|---------|-------|---------------------|-------|
| `transformers`       | yes     | yes   | no (manual batching)| always available; the baseline |
| `vllm`               | **no**  | yes   | **yes**             | 3-5x throughput; Linux wheels only |
| `onnxruntime-genai`  | yes     | yes   | partial             | Windows-capable alternative |

`transformers` is the default so the project installs and runs everywhere.
`vllm` is an optional extra (`uv sync --extra vllm`) because PyPI publishes no
Windows wheel for it — attempting `uv add vllm` on Windows fails and leaves the
lock in a dirty state.

Why a backend interface at all, rather than just calling `transformers`? Because
the compute budget is the binding constraint (314 h estimated for the full grid)
and the fix for that is a serving engine. Baking an engine into the executor
would mean rewriting the executor when the benchmark says to switch. The backend
boundary means the benchmark decides, and switching is a config change.

What the interface deliberately does NOT do
--------------------------------------------
It does not own the prompt, the sampling parameters, or the request identity.
Those belong to the experiment (`arms/` renders, `config` pins, `store` hashes).
A backend that could rewrite a prompt would break the `config_hash` guarantee
that makes every artifact attributable — which is a worse bug than being slow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# ── the wire format ────────────────────────────────────────────────────────────
# An OpenAI-compatible subset, deliberately. vLLM and onnxruntime-genai both
# speak it, `transformers` can be wrapped to it in a few lines, and it means the
# content-addressed store (PRD §R6.1) hashes something stable across a backend
# switch instead of something engine-specific.
#
# `seed` is included in the hashed body on purpose. PRD §R7.1 requires seeds to
# vary something, and R6.1 dedupes identical requests to one file — so if the
# seed were outside the hash, repeated runs at the same temperature would
# silently collapse into a single cached response and seed-variance estimation
# would be impossible without any error being raised.


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """One prompt to one model. Immutable, hashable, and engine-neutral."""

    model: str
    prompt: str
    system: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: tuple[str, ...] = ()
    seed: int = 0
    request_id: str = ""
    """Stable caller-assigned id. Used for tracing, NOT for cache identity —
    the cache keys on `canonical_body()`."""

    def messages(self) -> list[dict[str, str]]:
        if self.system:
            return [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.prompt},
            ]
        return [{"role": "user", "content": self.prompt}]

    def canonical_body(self) -> dict[str, Any]:
        """The exact bytes that get hashed and sent.

        Deliberately explicit rather than derived from `__dict__`: adding a field
        to this dataclass must be a deliberate act that changes the cache key,
        not an accident that reuses a stale cached response.
        """
        return {
            "model": self.model,
            "messages": self.messages(),
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "stop": list(self.stop),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One completion, plus everything needed to attribute it."""

    text: str
    finish_reason: str
    """`stop` | `length` | `refusal` | `error`.

    PRD §R5.5 requires these to be handled explicitly and counted separately. A
    silent bias here is unacceptable, so this is a required field with no
    default — a caller cannot forget to record it.
    """
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    """Time to first token, measured separately from total (PRD §R8.5). A
    bootstrap CI on a heavy-tailed mean is not a defensible latency summary."""
    queue_wait_ms: float | None = None
    """Time spent waiting for a batch slot. Recorded SEPARATELY from server
    time, or concurrency (PRD §R5.3) pollutes the latency signal we report."""
    backend: str = ""
    model_version: str = ""
    """Mandatory (PRD §R6.2): prices and behaviour both drift, so a re-run a
    month later is not comparable without it."""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LocalBackend(Protocol):
    """What every backend must provide."""

    name: str

    def health(self) -> dict[str, Any]:
        """Report readiness WITHOUT loading weights.

        `design-power` and `make env` both call this, and neither can afford to
        pay 30 seconds and 18 GB to find out the answer is "no".
        """
        ...

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """One completion. Blocking.

        Batching is the backend's business: `transformers` batches via
        `generate_batch`, vLLM via continuous batching behind this same call.
        The executor stays identical either way.
        """
        ...

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...

    def context_window(self, model: str) -> int:
        """Declared maximum. The context-feasibility check (PRD §R3.2) reads this
        BEFORE a run, because a cell that cannot fit is invalid, not slow."""
        ...

    def close(self) -> None: ...


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot run here.

    Carries a remediation string, because the answer is always the same shape —
    "use WSL2", "install the extra", "pick another backend" — and a bare
    ImportError makes the reader work that out from scratch.
    """


def select_backend(name: str) -> LocalBackend:
    """Instantiate a backend by config name.

    Fails with an actionable message rather than an ImportError, because
    "No module named vllm" is not a diagnosis.
    """
    if name == "transformers":
        from msat.providers.transformers_backend import TransformersBackend

        return TransformersBackend()
    if name == "vllm":
        from msat.providers.vllm_backend import VLLMBackend

        return VLLMBackend()
    if name == "onnxruntime-genai":
        from msat.providers.ort_backend import OrtGenAiBackend

        return OrtGenAiBackend()
    raise ValueError(
        f"unknown backend {name!r}; expected one of 'transformers', 'vllm', 'onnxruntime-genai'"
    )


def backend_availability() -> dict[str, dict[str, Any]]:
    """Which backends could run here, and why not if they could not.

    Surfaces in `msat validate-config`, so an unavailable backend is a WARNING at
    design time rather than a crash at hour 14 of a run.

    Probes each backend's own `health()` rather than guessing from imports:
    "the module imported" is not "the engine can start", and the difference is
    exactly the failure you want to hear about at design time.
    """
    out: dict[str, dict[str, Any]] = {}
    for name in ("transformers", "vllm", "onnxruntime-genai"):
        try:
            health = select_backend(name).health()
            out[name] = {"available": bool(health.get("ready")), **health}
        except Exception as exc:  # noqa: BLE001 - an availability probe must never raise
            out[name] = {
                "available": False,
                "ready": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "remedy": "unknown backend name; expected one of 'transformers', "
                "'vllm', 'onnxruntime-genai'",
            }
    return out
