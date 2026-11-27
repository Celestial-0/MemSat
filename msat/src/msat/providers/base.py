"""One backend interface, N engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    model: str
    prompt: str
    system: str = ""
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    stop: tuple[str, ...] = ()
    seed: int = 0
    request_id: str = ""

    def messages(self) -> list[dict[str, str]]:
        if self.system:
            return [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.prompt},
            ]
        return [{"role": "user", "content": self.prompt}]

    def canonical_body(self) -> dict[str, Any]:
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
    text: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    ttft_ms: float | None = None
    queue_wait_ms: float | None = None
    backend: str = ""
    model_version: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LocalBackend(Protocol):
    name: str

    def health(self) -> dict[str, Any]: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]: ...

    def context_window(self, model: str) -> int: ...

    def close(self) -> None: ...


class BackendUnavailableError(RuntimeError):
    """The requested backend cannot run on this platform."""


def select_backend(name: str) -> LocalBackend:
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
