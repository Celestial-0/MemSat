"""Regression loops for local model loading and text generation."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from msat.providers.base import BackendUnavailableError, GenerationRequest
from msat.providers.transformers_backend import TransformersBackend

OFFLOAD_ERROR = (
    "Some modules are dispatched on the CPU or the disk. "
    "You need to set llm_int8_enable_fp32_cpu_offload=True and pass a custom device_map."
)


def test_quantized_model_can_use_declared_cpu_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = SimpleNamespace(eval=lambda: None)

    def load(model, **kwargs):
        quant = kwargs["quantization_config"]
        if not getattr(quant, "llm_int8_enable_fp32_cpu_offload", False):
            raise ValueError(OFFLOAD_ERROR)
        return loaded

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type="plain")
            ),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        ),
    )
    backend = TransformersBackend()
    assert backend._load("cached-model")[0] is loaded


def test_model_load_errors_retain_cause_as_backend_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def load(*args, **kwargs):
        raise ValueError(OFFLOAD_ERROR)

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            BitsAndBytesConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            AutoConfig=SimpleNamespace(
                from_pretrained=lambda *args, **kwargs: SimpleNamespace(model_type="plain")
            ),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: object()),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        ),
    )
    with pytest.raises(BackendUnavailableError, match="llm_int8_enable_fp32_cpu_offload"):
        TransformersBackend().generate(GenerationRequest(model="cached-model", prompt="Question"))


def test_generation_uses_tokenizer_mapping_and_supported_stopping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class Encoding(dict):
        def to(self, device):
            return Encoding({name: value.to(device) for name, value in self.items()})

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs.get("return_dict") is True
            return Encoding(input_ids=torch.tensor([[1, 2]]), attention_mask=torch.tensor([[1, 1]]))

        def decode(self, tokens, **kwargs):
            return "answer"

    tokenizer = Tokenizer()

    def generate(**kwargs):
        assert "stop_sequences" not in kwargs
        assert kwargs["stop_strings"] == ["END"]
        assert kwargs["tokenizer"] is tokenizer
        return torch.tensor([[1, 2, 3]])

    model = SimpleNamespace(device="cpu", generate=generate)
    backend = TransformersBackend(device="cpu", dtype="bfloat16")
    monkeypatch.setattr(backend, "_load", lambda name: (model, tokenizer))
    result = backend.generate(
        GenerationRequest(model="cached-model", prompt="Question", stop=("END",))
    )
    assert result.text == "answer"
    assert result.prompt_tokens == 2
    assert result.completion_tokens == 1


def test_multimodal_gemma_checkpoint_loads_its_exact_text_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4TextConfig

    config = Gemma4TextConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        global_head_dim=4,
        hidden_size_per_layer_input=4,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=0,
    )
    source = Gemma4ForCausalLM(config)
    state = {
        key.replace("model.", "model.language_model.", 1)
        if key.startswith("model.")
        else key: value.detach().clone().contiguous()
        for key, value in source.state_dict().items()
    }
    save_file(state, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "text_config": config.to_dict(),
                "vision_config": None,
                "audio_config": None,
                "tie_word_embeddings": True,
                "architectures": ["Gemma4ForConditionalGeneration"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *args, **kwargs: object())
    backend = TransformersBackend(device="cpu", dtype="bfloat16")
    model, _ = backend._load(str(tmp_path))
    assert isinstance(model, Gemma4ForCausalLM)
    expected = source.get_input_embeddings().weight.to(torch.bfloat16)
    assert torch.equal(model.get_input_embeddings().weight, expected)
    generated = model.generate(
        input_ids=torch.tensor([[2, 5, 6]]), max_new_tokens=2, do_sample=False
    )
    assert generated.shape[-1] > 3
    source.to(torch.bfloat16).eval()
    ids = torch.tensor([[2, 5, 6]])
    assert torch.equal(
        model.model.embed_tokens_per_layer(ids), source.model.embed_tokens_per_layer(ids)
    )
    expected_tokens = source.generate(input_ids=ids, max_new_tokens=2, do_sample=False)
    assert torch.equal(generated, expected_tokens)
    chunked_tokens = model.generate(
        input_ids=ids, max_new_tokens=2, do_sample=False, prefill_chunk_size=1
    )
    assert torch.equal(chunked_tokens, expected_tokens)
    backend.close()
