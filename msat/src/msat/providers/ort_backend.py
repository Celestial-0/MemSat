"""The `onnxruntime-genai` backend — the Windows-capable serving option.

Exists because vLLM publishes no Windows wheel. This is the throughput escape
hatch for a native-Windows execution box, and it is meaningfully weaker than
vLLM: batching is limited, and the supported model set is narrower (it targets
ONNX-exported decoder models).

It is here as a MEASURED OPTION, not a recommendation. The order of preference on
any machine is:

1. `vllm` if the platform has it (Linux, or WSL2 with GPU passthrough)
2. `transformers` — always available, the reproducible baseline
3. this one, only if a benchmark on the real box shows it beats `transformers`

Reaching for it because "it is Windows-native" rather than because it was
measured faster would be exactly the kind of unvalidated choice this project's
config layer exists to prevent.
"""

from __future__ import annotations

from typing import Any

from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)
from msat.providers.transformers_backend import classify_finish


class OrtGenAiBackend:
    """ONNX Runtime GenAI. Windows and Linux."""

    name = "onnxruntime-genai"

    def __init__(self, *, model_dir: str | None = None, provider: str = "cuda") -> None:
        self.model_dir = model_dir
        self.provider = provider
        self._model: Any = None

    def health(self) -> dict[str, Any]:
        try:
            import onnxruntime as ort
            import onnxruntime_genai  # noqa: F401
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return {
                "ready": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "remedy": "optional extra: `uv sync --extra ort`",
            }
        return {
            "ready": True,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
            "device": self.provider,
        }

    def _ensure_model(self, model: str) -> Any:
        if self._model is not None:
            return self._model
        import onnxruntime_genai as genai

        path = self.model_dir or model
        self._model = genai.Model(path)
        return self._model

    def generate(self, request: GenerationRequest) -> GenerationResult:
        import time

        import onnxruntime_genai as genai

        model = self._ensure_model(request.model)
        tokenizer = genai.Tokenizer(model)

        # ORT-GenAI takes a prompt string, not messages, so the chat template has
        # to be applied on our side. Doing it here rather than inside the engine
        # is what keeps `canonical_body()` the single description of what was
        # sent — and therefore keeps the content-addressed store honest.
        tokens = tokenizer.create_encoded_prompt(
            tokenizer.apply_chat_template(request.messages(), add_generation_prompt=True)
        )

        params = genai.GeneratorParams(model)
        params.set_search_options(
            max_length=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        if request.seed:
            params.set_seed(request.seed)

        generator = genai.Generator(model, params)

        t0 = time.perf_counter()
        first_token_ms: float | None = None
        text_parts: list[str] = []
        prompt_len = len(tokens)
        completion = 0
        truncated = False
        while not generator.is_done():
            generator.compute_next_token()
            token_id = generator.get_next_tokens()[0]
            if token_id == tokenizer.eos_token_id:
                break
            text_parts.append(tokenizer.decode([token_id]))
            completion += 1
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - t0) * 1000.0
            if completion >= request.max_tokens:
                truncated = True
                break

        text = "".join(text_parts)
        return GenerationResult(
            text=text,
            finish_reason=classify_finish(text, completion, request) if not truncated else "length",
            prompt_tokens=prompt_len,
            completion_tokens=completion,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            ttft_ms=first_token_ms,
            backend=self.name,
            model_version=request.model,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """Serial by default; ORT-GenAI's batching is not worth faking badly."""
        return [self.generate(r) for r in requests]

    def context_window(self, model: str) -> int:
        raise BackendUnavailableError(
            "onnxruntime-genai does not expose a context window through a stable API. "
            "Declare it in configs/readers.yaml instead — which is where "
            "context_feasibility reads it from anyway."
        )

    def close(self) -> None:
        self._model = None
