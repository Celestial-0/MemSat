"""The VRAM arithmetic behind the open hardware decisions (D1, D4).

`msat vram <checkpoint> --tokens N --card GB` computes, from the checkpoint's own
`config.json`:

    KV bytes/token = 2 x n_layers x n_kv_heads x head_dim x dtype_bytes

These tests pin the ARITHMETIC, which is checkable by hand, and pin the refusals,
which are the part that keeps it honest. No GPU and no downloaded checkpoint are
needed for any of it.

The Mistral-7B-class fixture below is a real architecture class, so its numbers can
be verified against a textbook KV calculation rather than against another
implementation of this code.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from msat.cli import app
from msat.hardware.vram import (
    DEFAULT_HEADROOM_GB,
    GB,
    ConfigMissingError,
    ModelShape,
    kv_ceiling_tokens,
    load_config,
    plan_vram,
    shape_from_config,
)

RUNNER = CliRunner()

MISTRAL7B = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 14336,
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "torch_dtype": "bfloat16",
}


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    d = tmp_path / "mistral-7b-like"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(MISTRAL7B), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def shape() -> ModelShape:
    return shape_from_config(MISTRAL7B, name="mistral-7b-like")


# ── the arithmetic, checkable by hand ──────────────────────────────────────────


def test_kv_bytes_per_token_is_2_k_and_v_by_layers_heads_and_dtype(shape) -> None:
    """2 x 32 layers x 8 KV heads x 128 head_dim x 2 bytes = 131,072.

    The `2` is K and V. Every one of the other four factors is a field the
    checkpoint declares, which is why this is arithmetic rather than a guess.
    """
    assert shape.kv_bytes_per_token() == 2 * 32 * 8 * 128 * 2
    assert shape.kv_bytes_per_token() == 131_072


def test_the_gqa_factor_is_reported_and_matters(shape) -> None:
    """8 KV heads against 32 attention heads means the KV cache is 4x smaller than a
    full-attention model of the same depth.

    This single field is why a 7B model can serve long context at all, and why
    ignoring it (assuming no GQA) demands a card the checkpoint does not need.
    """
    assert shape.gqa_factor == 4


def test_weights_at_bf16_are_two_bytes_per_parameter(shape) -> None:
    """bytes per parameter x count, over GB. The factor of 8 belongs to BITS, not
    bytes - a `/8` here would under-report by 8x, which is the kind of arithmetic
    slip that turns a 24 GB answer into a 3 GB one."""
    assert shape.dtype_bytes == 2.0
    assert shape.weights_gb() == pytest.approx(shape.n_params * 2.0 / GB)


def test_int4_is_one_quarter_of_bf16_weights(shape) -> None:
    assert shape.weights_gb(quant_bits=4.0) == pytest.approx(shape.weights_gb() / 4, rel=1e-9)


def test_kv_gb_is_linear_in_tokens(shape) -> None:
    one = shape.kv_gb(1000)
    assert shape.kv_gb(2000) == pytest.approx(2 * one)


def test_the_plan_separates_the_floor_from_the_headroom(shape) -> None:
    p = plan_vram(shape, n_tokens=4096)
    assert p.floor_gb == pytest.approx(p.weights_gb + p.kv_gb)
    assert p.total_gb == pytest.approx(p.floor_gb + DEFAULT_HEADROOM_GB)


def test_the_plan_lists_what_it_omits(shape) -> None:
    """weights + KV is the FLOOR. Presenting it as the requirement is the same class
    of error as every hand-typed ceiling this module exists to replace."""
    p = plan_vram(shape, n_tokens=4096)
    joined = "; ".join(p.omitted)
    assert "activation" in joined
    assert "CUDA context" in joined
    assert "fragmentation" in joined
    assert "FLOOR" in p.as_dict()["note"]


# ── the two open decisions, in arithmetic ──────────────────────────────────────


def test_arm_d_at_122k_tokens_needs_more_than_an_eight_gb_card(shape) -> None:
    """D4. S's full corpus is 122,324 tokens.

    At bf16: weights 12.2 GB + KV 14.9 GB + 3 GB headroom = 30.2 GB.
    At int4: weights 3.1 GB, so 21.0 GB.

    Both are far above the 8 GB card this project has, which is why `full_context`
    was refused at render time.
    """
    bf16 = plan_vram(shape, n_tokens=122_324)
    assert bf16.total_gb == pytest.approx(30.2, abs=0.5)
    int4 = plan_vram(shape, n_tokens=122_324, quant_bits=4.0)
    assert int4.total_gb == pytest.approx(21.0, abs=0.5)


def test_the_ceiling_on_an_eight_gb_card_is_zero_not_a_small_number(shape) -> None:
    """The weights alone do not fit, so the honest answer is zero tokens.

    Returning a small positive number here would be how someone schedules a 292-hour
    run on hardware that cannot hold the model.
    """
    assert kv_ceiling_tokens(shape, 8.0) == 0


@pytest.mark.parametrize(
    ("gb", "expected_band"),
    [(16, (4000, 8000)), (24, (60000, 80000)), (40, (180000, 220000))],
)
def test_the_ceiling_matches_the_plan_for_several_cards(shape, gb, expected_band) -> None:
    """`kv_ceiling_tokens` and `plan_vram` must agree; they are the same arithmetic
    pointed in opposite directions."""
    ceiling = kv_ceiling_tokens(shape, gb)
    plan = plan_vram(shape, n_tokens=ceiling)
    assert plan.total_gb <= gb
    assert expected_band[0] < ceiling < expected_band[1]


# ── the refusals, which are the part that keeps it honest ──────────────────────


def test_a_missing_kv_head_count_is_refused_not_defaulted(checkpoint) -> None:
    """The single most dangerous guess in this project.

    Assuming `num_attention_heads` models a model WITHOUT grouped-query attention
    and overstates the KV cache by exactly the GQA factor - commonly 8x - which
    would demand a card the checkpoint does not need. Some checkpoints genuinely
    omit the field, and a default is the wrong answer.
    """
    cfg = {**MISTRAL7B}
    cfg.pop("num_key_value_heads")
    d = checkpoint.parent / "no-kv-heads"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ConfigMissingError, match="grouped-query attention"):
        shape_from_config(cfg, name="no-kv-heads")


def test_a_missing_layer_count_is_refused(checkpoint) -> None:
    cfg = {**MISTRAL7B, "num_hidden_layers": None}
    with pytest.raises(ConfigMissingError, match="num_hidden_layers"):
        shape_from_config(cfg)


def test_a_missing_checkpoint_names_what_to_pass(tmp_path) -> None:
    with pytest.raises(ConfigMissingError, match="no network access"):
        load_config(tmp_path / "absent")


def test_an_estimated_parameter_count_is_flagged_as_estimated(shape) -> None:
    """An estimate presented as a count is how a 24 GB card gets mistaken for a
    48 GB one, and no downstream number may quietly present one as the other."""
    assert shape.n_params_is_exact is False


def test_an_explicit_parameter_count_is_flagged_as_exact() -> None:
    s = shape_from_config({**MISTRAL7B, "num_parameters": 7_000_000_000})
    assert s.n_params_is_exact is True
    assert s.n_params == 7_000_000_000


def test_moe_experts_are_counted_not_ignored() -> None:
    """Experts are the overwhelming majority of a MoE model's parameters.

    Ignoring them understates the weights requirement by roughly the expert count,
    which is wrong in exactly the direction that gets someone to under-buy a card.
    """
    dense = shape_from_config(MISTRAL7B, name="dense")
    moe = shape_from_config(
        {**MISTRAL7B, "num_local_experts": 8, "moe_intermediate_size": 1024},
        name="moe",
    )
    assert moe.n_params > dense.n_params


def test_a_smaller_expert_width_is_used_for_experts() -> None:
    """MoE models routinely give experts a SMALLER feed-forward width. Using the
    dense width for all 60 experts produced a 508B estimate for a "30B" model -
    an absurd number that is the tell that the arithmetic was wrong."""
    s = shape_from_config(
        {
            **MISTRAL7B,
            "num_local_experts": 60,
            "moe_intermediate_size": 512,
            "num_hidden_layers": 48,
        },
        name="moe60",
    )
    # 48 layers x 3 x 4096 x 512 x 60 experts, plus a dense path and embeddings.
    expected_experts = 48 * 3 * 4096 * 512 * 60
    assert s.n_params > expected_experts
    assert s.n_params < 100e9, "a 30B-class MoE must not estimate to hundreds of billions"


def test_a_moe_plan_says_kv_depends_on_attention_not_parameters() -> None:
    s = shape_from_config(
        {**MISTRAL7B, "num_local_experts": 8, "moe_intermediate_size": 1024},
        name="moe",
    )
    assert s.is_moe is True
    assert "expert-routing workspace" in "; ".join(plan_vram(s, n_tokens=2048).omitted)


# ── the command ────────────────────────────────────────────────────────────────


def test_the_cli_reports_the_shape_and_the_plan(checkpoint) -> None:
    r = RUNNER.invoke(app, ["vram", str(checkpoint), "--tokens", "122324"])
    assert r.exit_code == 0, r.output
    assert "KV per token 131,072 bytes" in r.output
    assert "GQA factor 4x" in r.output
    assert "24 GB" in r.output or "40 GB" in r.output
    assert "omitted" in r.output


def test_the_cli_reports_a_ceiling_for_a_named_card(checkpoint) -> None:
    r = RUNNER.invoke(app, ["vram", str(checkpoint), "--tokens", "4096", "--card", "24"])
    assert r.exit_code == 0, r.output
    assert "ceiling on a 24 GB card" in r.output


def test_the_cli_fails_cleanly_on_a_missing_kv_head_count(tmp_path) -> None:
    d = tmp_path / "broken"
    d.mkdir()
    cfg = {**MISTRAL7B}
    cfg.pop("num_key_value_heads")
    (d / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    r = RUNNER.invoke(app, ["vram", str(d)])
    assert r.exit_code == 1
    assert "grouped-query attention" in r.output


def test_the_cli_fails_cleanly_on_a_missing_checkpoint(tmp_path) -> None:
    r = RUNNER.invoke(app, ["vram", str(tmp_path / "nope")])
    assert r.exit_code == 1
    assert "no network access" in r.output


def test_the_command_is_on_the_surface() -> None:
    assert "vram" in RUNNER.invoke(app, ["--help"]).output
