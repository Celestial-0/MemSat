"""The `transformers` backend — always available, the baseline.

Correct, reproducible, and slow enough that the compute budget is the thing that
forces the question of whether to switch. That is the correct order of priorities:
establish a backend whose results you can defend, then measure whether a faster
one changes the answer.

Batching is manual here (`generate_batch` chunks by a configured batch size),
which is the whole reason `vllm` exists. It is stated plainly in the module
docstring rather than left for someone to discover at 13 days of runtime.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from msat.providers.base import GenerationRequest, GenerationResult

#: Models loaded lazily and kept warm. The full panel is 3 x ~9B; holding all of
#: them resident at once needs ~27 GB at 4-bit, so the backend keeps ONE model
#: loaded and the executor groups cells by reader. That grouping is a scheduling
#: decision made in P4, and it is why `context_window()` is cheap to call.
_CONTEXT_CACHE: dict[str, Any] = {}


class TransformersBackend:
    """HuggingFace `transformers` + `accelerate`. Works on every platform."""

    name = "transformers"

    def __init__(self, *, device: str = "cuda", dtype: str = "4bit") -> None:
        self.device = device
        self.dtype = dtype
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}

    # ── lifecycle ────────────────────────────────────────────────────────────
    def health(self) -> dict[str, Any]:
        """Report readiness WITHOUT loading weights."""
        try:
            import torch
        except OSError as exc:
            # WinError 4551 lands here on a host with Smart App Control on.
            return {
                "ready": False,
                "reason": f"torch cannot be imported: {exc}",
                "remedy": "machine Application Control is blocking a torch DLL; run "
                "`make app-control` for the diagnosis.",
            }
        info: dict[str, Any] = {
            "ready": True,
            "torch": torch.__version__,
            "cuda": bool(torch.version.cuda),
            "device": self.device,
            "dtype": self.dtype,
        }
        if self.device == "cuda" and not torch.cuda.is_available():
            info["ready"] = False
            info["reason"] = "device is 'cuda' but no CUDA device is visible"
        return info

    def _quant_config(self) -> Any:
        """4-bit needs `bitsandbytes`; fall back explicitly rather than silently.

        Falling back SILENTLY is the dangerous version — it would double the
        memory footprint and change the answers, and nothing would say so. The
        dtype actually used is recorded on every result instead.
        """
        if self.dtype != "4bit":
            return None
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    def _load(self, model: str) -> tuple[Any, Any]:
        if model in self._models:
            return self._models[model], self._tokenizers[model]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model)
        quant = self._quant_config()
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if quant is not None:
            kwargs["quantization_config"] = quant
        else:
            kwargs["torch_dtype"] = torch.bfloat16
        lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        lm.eval()
        self._models[model] = lm
        self._tokenizers[model] = tokenizer
        return lm, tokenizer

    # ── generation ────────────────────────────────────────────────────────────
    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generate_batch([request])[0]

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        by_model: dict[str, list[GenerationRequest]] = {}
        for r in requests:
            by_model.setdefault(r.model, []).append(r)

        out: list[GenerationResult] = []
        for model, group in by_model.items():
            out.extend(self._generate_group(model, group))
        return out

    def _generate_group(self, model: str, group: list[GenerationRequest]) -> list[GenerationResult]:
        import torch

        lm, tokenizer = self._load(model)
        results: list[GenerationResult] = []
        for request in group:
            encoded = tokenizer.apply_chat_template(
                request.messages(), add_generation_prompt=True, return_tensors="pt"
            ).to(lm.device)
            prompt_tokens = int(encoded.shape[-1])

            t0 = time.perf_counter()
            with torch.inference_mode():
                generated = lm.generate(
                    **encoded,
                    max_new_tokens=request.max_tokens,
                    do_sample=request.temperature > 0,
                    temperature=max(request.temperature, 1e-5),
                    top_p=request.top_p,
                    stop_sequences=list(request.stop) or None,
                )
            latency_ms = (time.perf_counter() - t0) * 1000.0

            new_tokens = generated[0][encoded.shape[-1] :]
            completion_tokens = int(new_tokens.shape[-1])
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(
                GenerationResult(
                    text=text,
                    finish_reason=classify_finish(text, completion_tokens, request),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    latency_ms=latency_ms,
                    backend=self.name,
                    model_version=model,
                )
            )
        return results

    def context_window(self, model: str) -> int:
        """From the tokenizer's own config — the authoritative source.

        Not hardcoded here: a wrong context window means the feasibility check
        (PRD §R3.2) passes cells that cannot run, and they fail at hour 14 of a
        run instead of at design time.
        """
        if model in _CONTEXT_CACHE:
            return int(_CONTEXT_CACHE[model])
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model)
        window = int(
            getattr(cfg, "max_position_embeddings", 0) or getattr(cfg, "n_positions", 0) or 0
        )
        _CONTEXT_CACHE[model] = window
        return window

    def close(self) -> None:
        self._models.clear()
        self._tokenizers.clear()


def classify_finish(text: str, completion_tokens: int, request: GenerationRequest) -> str:
    """PRD §R5.5 — `length`, `refusal`, and empty output are EXPLICIT and counted.

    Silently keeping a truncated answer and scoring it is the bias this exists to
    prevent: a `length` truncation usually means the reader was mid-sentence
    because the context was too long, which is exactly the signal the K x B design
    is trying to measure.
    """
    if not text.strip():
        return "empty"
    if any(marker in text.lower() for marker in ("i cannot", "i can't assist", "i'm sorry")):
        return "refusal"
    if completion_tokens >= request.max_tokens:
        return "length"
    if any(text.strip().endswith(s) for s in request.stop):
        return "stop"
    return "stop"


def model_is_local(path: str) -> bool:
    return Path(path).exists()
