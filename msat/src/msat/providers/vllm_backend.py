"""The `vllm` backend — Linux only, and the reason the backend interface exists.

vLLM is the answer to the compute budget. 189,000 short independent generations
on one RTX 4060 is a throughput problem, and continuous batching is the specific
thing that fixes it: naive batching leaves most of the GPU idle waiting for the
longest sequence in the batch, while vLLM admits new sequences the moment a slot
frees.

PLATFORM
--------
**PyPI publishes no Windows wheel for vLLM.** That is not a uv problem or a
resolution problem — the wheels do not exist. Running `uv add vllm` on Windows
fails. So:

- the default backend is `transformers`, which runs everywhere;
- `vllm` is an optional extra (`uv sync --extra vllm`);
- switching to it on Windows requires WSL2 with GPU passthrough, or a Linux box.

BEFORE SWITCHING
----------------
Benchmark it. `scripts/bench_backend.py` runs the same prompts through both
backends and reports prefill/decode throughput. The reason this backend is behind
an interface rather than wired in is that the decision should be made from a
measurement on YOUR hardware, not from a benchmark quoted in a README.
"""

from __future__ import annotations

import os
import time
from typing import Any

from msat.providers.base import (
    BackendUnavailableError,
    GenerationRequest,
    GenerationResult,
)
from msat.providers.transformers_backend import classify_finish


class VLLMBackend:
    """Continuous-batching inference over an OpenAI-compatible HTTP surface.

    Two ways to attach, both supported because they suit different deployments:

    - **in-process** (`engine="local"`): constructs `vllm.LLM` in this process.
      Lowest latency, but the GPU is exclusively ours and a crash takes the run
      with it.
    - **client** (`engine="server"`): talks to a vLLM server over HTTP. The
      executor can then resume independently of the engine, which matters for
      PRD §R5.2 (`kill -9` mid-run -> resume -> identical results) — a separate
      server process is far easier to reason about than an in-process engine.
    """

    name = "vllm"

    def __init__(
        self,
        *,
        engine: str = "local",
        base_url: str | None = None,
        model_path: str | None = None,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int | None = None,
    ) -> None:
        self.engine = engine
        self.base_url = base_url or os.environ.get("MSAT_VLLM_URL", "http://127.0.0.1:8000/v1")
        self.model_path = model_path
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self._llm: Any = None
        self._client: Any = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        import sys

        if sys.platform == "win32":
            # Two independent reasons, both fatal. Stated together because
            # fixing one does not fix the other, and a half-fixed vLLM setup
            # wastes an afternoon.
            return {
                "ready": False,
                "reason": "vLLM has no Windows wheel, AND vLLM 0.31 pins "
                "torch==2.13.0 exactly, which conflicts with this project's "
                "torch>=2.14.1 on the cu132 index.",
                "remedy": "run `make vllm` — vLLM lives in its OWN environment and "
                "this backend reaches it over HTTP. Requires WSL2 with GPU "
                "passthrough, or a Linux box.",
            }
        try:
            import vllm
        except Exception as exc:  # noqa: BLE001 - a health check must never raise
            return {
                "ready": False,
                "reason": f"{type(exc).__name__}: {exc}",
                "remedy": "start a vLLM server (`make vllm`) and point MSAT_VLLM_URL at it.",
            }
        if self.engine == "server":
            return {
                "ready": self._ping_server(),
                "vllm": getattr(vllm, "__version__", "unknown"),
            }
        return {"ready": True, "vllm": getattr(vllm, "__version__", "unknown")}

    def _ping_server(self) -> bool:
        try:
            import httpx

            response = httpx.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except Exception:  # noqa: BLE001 - a ping failing IS the answer
            return False

    def _ensure_engine(self, model: str) -> Any:
        if self._llm is not None:
            return self._llm
        from vllm import LLM

        kwargs: dict[str, Any] = {
            "model": self.model_path or model,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "dtype": "auto",
        }
        if self.max_model_len:
            # Bounding max_model_len is how the K=64 M-scale cell is made to FIT.
            # It is a real constraint (PRD §R3.2), not a performance knob, and
            # `context_feasibility` must be checked against the same number.
            kwargs["max_model_len"] = self.max_model_len
        self._llm = LLM(**kwargs)
        return self._llm

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if self.engine == "server":
            return self._generate_http(request)
        return self._generate_local(request)

    def _generate_local(self, request: GenerationRequest) -> GenerationResult:
        llm = self._ensure_engine(request.model)
        from vllm import SamplingParams

        t0 = time.perf_counter()
        outputs = llm.generate(
            [request.messages()],
            SamplingParams(
                n=1,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=list(request.stop) or None,
                seed=request.seed,
            ),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        out = outputs[0].outputs[0]
        text = out.text
        return GenerationResult(
            text=text,
            finish_reason=classify_finish(text, int(out.token_count), request),
            prompt_tokens=int(out.prompt_token_count or 0),
            completion_tokens=int(out.token_count or 0),
            latency_ms=latency_ms,
            backend=self.name,
            model_version=request.model,
        )

    def _generate_http(self, request: GenerationRequest) -> GenerationResult:
        import httpx

        t0 = time.perf_counter()
        with httpx.Client(timeout=600.0) as client:
            response = client.post(
                f"{self.base_url}/chat/completions",
                json=request.canonical_body(),
            )
            response.raise_for_status()
            payload = response.json()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = payload["choices"][0]
        usage = payload.get("usage", {})
        text = choice["message"]["content"]
        return GenerationResult(
            text=text,
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=latency_ms,
            backend=self.name,
            model_version=payload.get("model", request.model),
            raw=payload,
        )

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        """vLLM's own batching is automatic; the HTTP path benefits from a gather."""
        if not requests:
            return []
        if self.engine == "server":
            import concurrent.futures

            # Concurrent dispatch is what the server's continuous batching is FOR.
            # A serial loop here would throw away the entire advantage.
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
                return list(pool.map(self._generate_http, requests))
        return [self._generate_local(r) for r in requests]

    def context_window(self, model: str) -> int:
        if self.max_model_len:
            return self.max_model_len
        try:
            from transformers import AutoConfig

            cfg = AutoConfig.from_pretrained(model)
            return int(getattr(cfg, "max_position_embeddings", 0) or getattr(cfg, "n_positions", 0))
        except Exception as exc:  # pragma: no cover - depends on the model
            raise BackendUnavailableError(
                f"cannot determine the context window for {model!r}: {exc}"
            ) from exc

    def close(self) -> None:
        self._llm = None
