"""Regressions verifying HuggingFace revision pinning in TransformersBackend and LocalReader."""

from __future__ import annotations

import re
from collections.abc import Generator
from types import SimpleNamespace
from typing import Any

import pytest

from msat.providers.base import GenerationRequest, GenerationResult
from msat.providers.local import LocalReader, ReaderVramSpec, reader_from_family
from msat.providers.transformers_backend import CONTEXT_CACHE, TransformersBackend

SAMPLE_FAMILY: dict[str, Any] = {
    "id": "pinned-test",
    "model": "org/pinned-reader",
    "version": "c0ffee" * 6 + "abcd",
    "released": "2026-01-01",
    "role": "primary",
    "reader_strength": 0.8,
    "context_window": 32768,
    "exact_checkpoint_confirmed": True,
    "vram": {
        "params_b": 7.0,
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "bytes_per_param": 0.55,
    },
}


class DummyModel:
    def __init__(self, commit_hash: str | None = None, device: str = "cpu") -> None:
        self.device = device
        self.config = SimpleNamespace(_commit_hash=commit_hash)

    def eval(self) -> DummyModel:
        return self

    def generate(self, **kwargs: Any) -> Any:
        import torch

        return torch.tensor([[10, 20, 30, 40]])


class DummyTokenizer:
    def apply_chat_template(self, messages: Any, **kwargs: Any) -> Any:
        import torch

        class Encoding(dict):
            def to(self, device: Any) -> Encoding:
                return self

        return Encoding(input_ids=torch.tensor([[10, 20]]))

    def decode(self, tokens: Any, **kwargs: Any) -> str:
        return "pinned-generation-output"


class FakeBackend:
    name = "fake"

    def __init__(self, result: GenerationResult | None = None) -> None:
        self._result = result or GenerationResult(
            text="injected-fake-output",
            finish_reason="stop",
            prompt_tokens=4,
            completion_tokens=2,
            latency_ms=1.2,
            model_version="fake-custom-version",
        )

    def health(self) -> dict[str, Any]:
        return {"ready": True, "fake": True}

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self._result

    def generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [self._result for _ in requests]

    def context_window(self, model: str) -> int:
        return 8192

    def close(self) -> None:
        pass


@pytest.fixture(autouse=True)
def _reset_context_cache() -> Generator[None, None, None]:
    CONTEXT_CACHE.clear()
    yield
    CONTEXT_CACHE.clear()


def test_transformers_backend_constructor_revisions_mapping() -> None:
    initial_revisions = {"org/model-a": "rev-a1", "org/model-b": "rev-b1"}
    backend = TransformersBackend(device="cpu", dtype="bfloat16", revisions=initial_revisions)
    assert backend._revisions == {"org/model-a": "rev-a1", "org/model-b": "rev-b1"}

    # Mutating original dictionary does not mutate backend internal copy
    initial_revisions["org/model-a"] = "rev-mutated"
    assert backend._revisions["org/model-a"] == "rev-a1"

    # Default constructor without revisions leaves mapping empty
    empty_backend = TransformersBackend(device="cpu", dtype="bfloat16")
    assert empty_backend._revisions == {}

    none_backend = TransformersBackend(device="cpu", dtype="bfloat16", revisions=None)
    assert none_backend._revisions == {}


def test_bind_revision_for_unloaded_model() -> None:
    backend = TransformersBackend(device="cpu", dtype="bfloat16")
    backend.bind_revision("org/unloaded-model", "rev-first")
    assert backend._revisions["org/unloaded-model"] == "rev-first"

    # Rebinding before model is loaded updates the pin cleanly
    backend.bind_revision("org/unloaded-model", "rev-second")
    assert backend._revisions["org/unloaded-model"] == "rev-second"


def test_bind_revision_cannot_change_loaded_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    dummy_model = DummyModel(commit_hash="rev-pinned")
    dummy_tokenizer = DummyTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: dummy_tokenizer
    )
    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="plain"),
    )
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: dummy_model
    )

    backend = TransformersBackend(
        device="cpu", dtype="bfloat16", revisions={"org/locked-model": "rev-pinned"}
    )
    backend._load("org/locked-model")
    assert "org/locked-model" in backend._models

    # Rebinding with the exact same revision is allowed (idempotent)
    backend.bind_revision("org/locked-model", "rev-pinned")

    # Attempting to change revision of already loaded model raises ValueError
    with pytest.raises(
        ValueError, match=re.escape("Cannot change the checkpoint of an already loaded reader.")
    ):
        backend.bind_revision("org/locked-model", "rev-new-checkpoint")

    # After close() clears the loaded models, rebinding to a different revision succeeds
    backend.close()
    assert "org/locked-model" not in backend._models
    backend.bind_revision("org/locked-model", "rev-new-checkpoint")
    assert backend._revisions["org/locked-model"] == "rev-new-checkpoint"


def test_load_forwards_declared_pin_to_tokenizer_config_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    tokenizer_calls: list[dict[str, Any]] = []
    config_calls: list[dict[str, Any]] = []
    model_calls: list[dict[str, Any]] = []

    def mock_tokenizer_load(model_id: str, **kwargs: Any) -> DummyTokenizer:
        tokenizer_calls.append({"model": model_id, **kwargs})
        return DummyTokenizer()

    def mock_config_load(model_id: str, **kwargs: Any) -> SimpleNamespace:
        config_calls.append({"model": model_id, **kwargs})
        return SimpleNamespace(model_type="plain")

    def mock_model_load(model_id: str, **kwargs: Any) -> DummyModel:
        model_calls.append({"model": model_id, **kwargs})
        return DummyModel(commit_hash=kwargs.get("revision"))

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_tokenizer_load)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_config_load)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", mock_model_load)

    pinned_sha = "abcdef0123456789abcdef0123456789abcdef01"
    backend = TransformersBackend(
        device="cpu", dtype="bfloat16", revisions={"org/pinned-arch": pinned_sha}
    )

    model, tokenizer = backend._load("org/pinned-arch")
    assert isinstance(model, DummyModel)
    assert isinstance(tokenizer, DummyTokenizer)

    # Tokenizer, AutoConfig, and Model all received the declared revision
    assert len(tokenizer_calls) == 1
    assert tokenizer_calls[0]["model"] == "org/pinned-arch"
    assert tokenizer_calls[0]["revision"] == pinned_sha

    assert len(config_calls) == 1
    assert config_calls[0]["model"] == "org/pinned-arch"
    assert config_calls[0]["revision"] == pinned_sha

    assert len(model_calls) == 1
    assert model_calls[0]["model"] == "org/pinned-arch"
    assert model_calls[0]["revision"] == pinned_sha

    # Calling _load again returns cached instances without invoking HF loader stubs again
    cached_model, cached_tokenizer = backend._load("org/pinned-arch")
    assert cached_model is model
    assert cached_tokenizer is tokenizer
    assert len(tokenizer_calls) == 1
    assert len(config_calls) == 1
    assert len(model_calls) == 1


def test_load_without_revision_does_not_forward_revision_arg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    tokenizer_kwargs_captured: dict[str, Any] = {}
    config_kwargs_captured: dict[str, Any] = {}
    model_kwargs_captured: dict[str, Any] = {}

    def mock_tokenizer_load(model_id: str, **kwargs: Any) -> DummyTokenizer:
        tokenizer_kwargs_captured.update(kwargs)
        return DummyTokenizer()

    def mock_config_load(model_id: str, **kwargs: Any) -> SimpleNamespace:
        config_kwargs_captured.update(kwargs)
        return SimpleNamespace(model_type="plain")

    def mock_model_load(model_id: str, **kwargs: Any) -> DummyModel:
        model_kwargs_captured.update(kwargs)
        return DummyModel()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_tokenizer_load)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_config_load)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", mock_model_load)

    backend = TransformersBackend(device="cpu", dtype="bfloat16")
    backend._load("org/unpinned-arch")

    assert "revision" not in tokenizer_kwargs_captured
    assert "revision" not in config_kwargs_captured
    assert "revision" not in model_kwargs_captured


def test_context_window_unrelated_versions_never_share_cached_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    queries: list[tuple[str, str | None]] = []

    def mock_config(model_id: str, **kwargs: Any) -> SimpleNamespace:
        rev = kwargs.get("revision")
        queries.append((model_id, rev))
        if rev == "rev-alpha":
            return SimpleNamespace(model_type="plain", max_position_embeddings=2048)
        if rev == "rev-beta":
            return SimpleNamespace(model_type="plain", max_position_embeddings=8192)
        return SimpleNamespace(model_type="plain", max_position_embeddings=1024)

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_config)

    backend_alpha = TransformersBackend(
        device="cpu", dtype="bfloat16", revisions={"org/dual-model": "rev-alpha"}
    )
    backend_beta = TransformersBackend(
        device="cpu", dtype="bfloat16", revisions={"org/dual-model": "rev-beta"}
    )
    backend_unpinned = TransformersBackend(device="cpu", dtype="bfloat16")

    # Distinct versions of the same model return their independent context bounds
    win_alpha = backend_alpha.context_window("org/dual-model")
    win_beta = backend_beta.context_window("org/dual-model")
    win_unpinned = backend_unpinned.context_window("org/dual-model")

    assert win_alpha == 2048
    assert win_beta == 8192
    assert win_unpinned == 1024
    assert queries == [
        ("org/dual-model", "rev-alpha"),
        ("org/dual-model", "rev-beta"),
        ("org/dual-model", None),
    ]

    # Repeated calls hit the cache per (model, revision) key and do not query AutoConfig again
    queries.clear()
    assert backend_alpha.context_window("org/dual-model") == 2048
    assert backend_beta.context_window("org/dual-model") == 8192
    assert backend_unpinned.context_window("org/dual-model") == 1024
    assert len(queries) == 0, "Cached results must not re-query AutoConfig"


def test_context_window_uses_nested_text_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    def mock_nested_config(model_id: str, **kwargs: Any) -> SimpleNamespace:
        if "positions-fallback" in model_id:
            text_cfg = SimpleNamespace(n_positions=12288)
        else:
            text_cfg = SimpleNamespace(max_position_embeddings=65536)
        # Root config has 0 / None, while nested text_config carries the true position limits
        return SimpleNamespace(
            model_type="plain",
            max_position_embeddings=0,
            text_config=text_cfg,
        )

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_nested_config)

    backend = TransformersBackend(
        device="cpu", dtype="bfloat16", revisions={"org/multimodal": "rev-mm"}
    )
    window = backend.context_window("org/multimodal")
    assert window == 65536

    backend.bind_revision("org/multimodal-positions-fallback", "rev-fb")
    fallback_window = backend.context_window("org/multimodal-positions-fallback")
    assert fallback_window == 12288


def test_local_reader_backend_binds_declared_version_to_transformers_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import transformers

    loaded_revisions: dict[str, str | None] = {}

    def mock_tokenizer(model_id: str, **kwargs: Any) -> DummyTokenizer:
        loaded_revisions["tokenizer"] = kwargs.get("revision")
        return DummyTokenizer()

    def mock_config(model_id: str, **kwargs: Any) -> SimpleNamespace:
        loaded_revisions["config"] = kwargs.get("revision")
        return SimpleNamespace(model_type="plain", _commit_hash=kwargs.get("revision"))

    def mock_model(model_id: str, **kwargs: Any) -> DummyModel:
        loaded_revisions["model"] = kwargs.get("revision")
        return DummyModel(commit_hash=kwargs.get("revision"))

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", mock_tokenizer)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", mock_config)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", mock_model)

    reader = reader_from_family(SAMPLE_FAMILY, card_gb=24.0, backend="transformers")
    # Calling backend() automatically binds the declared reader version
    backend = reader.backend()
    assert isinstance(backend, TransformersBackend)
    assert backend._revisions[reader.model] == reader.version

    # Running generate via reader public interface forwards the declared pin
    req = GenerationRequest(model=reader.model, prompt="Test prompt")
    res = reader.generate(req)
    assert res.text == "pinned-generation-output"
    assert res.model_version == reader.version
    assert loaded_revisions["tokenizer"] == reader.version
    assert loaded_revisions["config"] == reader.version
    assert loaded_revisions["model"] == reader.version


def test_injected_fake_backend_constructor_support() -> None:
    fake = FakeBackend()
    spec = ReaderVramSpec(params_b=7.0, layers=32, kv_heads=8, head_dim=128)
    reader = LocalReader(
        id="fake-injected",
        model="org/custom-model",
        version="v-custom-12345",
        released="2026-01-01",
        role="primary",
        reader_strength=0.75,
        declared_context_window=8192,
        vram=spec,
        backend_name="transformers",
        _backend=fake,
    )

    # Injected backend is returned directly without requiring bind_revision
    assert reader.backend() is fake
    health = reader.health()
    assert health["ready"] is True
    assert health["engine"]["fake"] is True

    req = GenerationRequest(model=reader.model, prompt="Hello fake backend")
    res = reader.generate(req)
    assert res.text == "injected-fake-output"
    assert res.model_version == "fake-custom-version"


def test_injected_fake_backend_stays_supported_via_setattr() -> None:
    reader = reader_from_family(SAMPLE_FAMILY, card_gb=24.0, backend="transformers")
    fake = FakeBackend(
        GenerationResult(
            text="setattr-fake-result",
            finish_reason="stop",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=3.0,
            model_version="stamped-version",
        )
    )
    object.__setattr__(reader, "_backend", fake)

    assert reader.backend() is fake
    req = GenerationRequest(model=reader.model, prompt="Generate with setattr backend")
    res = reader.generate(req)
    assert res.text == "setattr-fake-result"
    assert res.model_version == "stamped-version"
