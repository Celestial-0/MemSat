"""The `transformers` backend: always available, the baseline."""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from msat.providers.base import BackendUnavailableError, GenerationRequest, GenerationResult

CONTEXT_CACHE: dict[tuple[str, str | None], Any] = {}
_CONTEXT_CACHE = CONTEXT_CACHE


class TransformersBackend:
    name = "transformers"

    def __init__(
        self,
        *,
        device: str = "cuda",
        dtype: str = "4bit",
        revisions: Mapping[str, str] | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self._models: dict[str, Any] = {}
        self._tokenizers: dict[str, Any] = {}
        self._revisions = dict(revisions or {})
        self.local_files_only = local_files_only
        self._cpu_embeddings: list[Any] = []

    def bind_revision(self, model: str, revision: str) -> None:
        """Bind the declared checkpoint before loading or probing a model."""
        if model in self._models and self._revisions.get(model) != revision:
            raise ValueError("Cannot change the checkpoint of an already loaded reader.")
        self._revisions[model] = revision

    def health(self) -> dict[str, Any]:
        try:
            import torch
        except OSError as exc:
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
        if self.dtype != "4bit":
            return None
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )

    def _load(self, model: str) -> tuple[Any, Any]:
        if model in self._models:
            return self._models[model], self._tokenizers[model]
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        try:
            revision_args: dict[str, Any] = (
                {"revision": self._revisions[model]} if model in self._revisions else {}
            )
            if self.local_files_only:
                revision_args["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(model, **revision_args)
            config = AutoConfig.from_pretrained(model, **revision_args)
            quant = self._quant_config()
            kwargs: dict[str, Any] = {
                "device_map": "auto" if self.device == "cuda" else self.device,
                "dtype": torch.bfloat16,
                **revision_args,
            }
            if quant is not None:
                kwargs["quantization_config"] = quant
            if getattr(config, "model_type", "") == "gemma4":
                from msat.providers.cpu_embedding import checkpoint_file, gemma_cpu_embedding_class

                text_config = config.get_text_config()
                text_config._commit_hash = getattr(config, "_commit_hash", None)
                kwargs.update(
                    config=text_config,
                    key_mapping={r"^model\.language_model\.": "model."},
                    output_loading_info=True,
                )
                text_model = gemma_cpu_embedding_class(
                    checkpoint_file(
                        model, self._revisions.get(model), local_files_only=self.local_files_only
                    ),
                    self._cpu_embeddings,
                    dtype=torch.bfloat16,
                )
                lm, loading = text_model.from_pretrained(model, **kwargs)
                if loading.get("missing_keys") or loading.get("mismatched_keys"):
                    raise ValueError(f"The text reader weights did not load completely: {loading}")
            else:
                lm = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            raise BackendUnavailableError(
                f"Unable to load reader {model!r}: {type(exc).__name__}: {exc}"
            ) from exc
        lm.eval()
        self._models[model] = lm
        self._tokenizers[model] = tokenizer
        return lm, tokenizer

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
                request.messages(),
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                enable_thinking=False,
            ).to(lm.device)
            input_ids = encoded["input_ids"]
            prompt_tokens = int(input_ids.shape[-1])

            generation_args: dict[str, Any] = {
                "max_new_tokens": request.max_tokens,
                "do_sample": request.temperature > 0,
            }
            if request.temperature > 0:
                generation_args.update(temperature=request.temperature, top_p=request.top_p)
            if request.stop:
                generation_args.update(stop_strings=list(request.stop), tokenizer=tokenizer)
            if self._cpu_embeddings and prompt_tokens > 512:
                generation_args["prefill_chunk_size"] = 512

            t0 = time.perf_counter()
            devices = list(range(torch.cuda.device_count())) if self.device == "cuda" else []
            try:
                with torch.random.fork_rng(devices=devices), torch.inference_mode():
                    if request.seed is not None:
                        torch.manual_seed(request.seed)
                    generated = lm.generate(
                        **encoded,
                        **generation_args,
                    )
            except (OSError, ValueError, RuntimeError) as exc:
                raise BackendUnavailableError(
                    f"Reader {model!r} failed generation: {type(exc).__name__}: {exc}"
                ) from exc
            latency_ms = (time.perf_counter() - t0) * 1000.0

            new_tokens = generated[0][prompt_tokens:]
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
                    model_version=getattr(getattr(lm, "config", None), "_commit_hash", None)
                    or self._revisions.get(model, model),
                    raw={
                        "dtype": self.dtype,
                        "model_class": type(lm).__name__,
                        "requested_revision": self._revisions.get(model),
                        "prefill_chunk_size": generation_args.get("prefill_chunk_size"),
                        "cpu_embeddings": [
                            embedding.source_info for embedding in self._cpu_embeddings
                        ],
                        "device_map": {
                            k: str(v) for k, v in getattr(lm, "hf_device_map", {}).items()
                        },
                    },
                )
            )
        return results

    def context_window(self, model: str) -> int:
        key = (model, self._revisions.get(model))
        if key in _CONTEXT_CACHE:
            return int(_CONTEXT_CACHE[key])
        from transformers import AutoConfig

        revision_args: dict[str, Any] = {"revision": key[1]} if key[1] else {}
        if self.local_files_only:
            revision_args["local_files_only"] = True
        cfg = AutoConfig.from_pretrained(model, **revision_args)
        cfg = getattr(cfg, "text_config", cfg)
        window = int(
            getattr(cfg, "max_position_embeddings", 0) or getattr(cfg, "n_positions", 0) or 0
        )
        _CONTEXT_CACHE[key] = window
        return window

    def close(self) -> None:
        self._models.clear()
        self._tokenizers.clear()
        for embedding in self._cpu_embeddings:
            embedding.close()
        self._cpu_embeddings.clear()


def classify_finish(text: str, completion_tokens: int, request: GenerationRequest) -> str:
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
