# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.v1.attention.backend import AttentionCGSupport

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS, SLOT_COUNT
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    text_suppression_ids,
)
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    ATTENTION,
    GDN,
    STEP_COLUMNS,
    WRITE,
    DuplexIOFlashAttentionMetadataBuilder,
    DuplexIOGDNAttentionBackend,
    DuplexIOGDNAttentionMetadataBuilder,
    DuplexIOQwenGatedDeltaNetAttention,
    DuplexIOQwenModel,
    FlatStep,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config() -> DuplexIOConfig:
    return DuplexIOConfig(
        user_asr_config={"model_type": "nemotron_asr_streaming"},
        text_config={
            "model_type": "qwen3_5_text",
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
            "layer_types": ["full_attention", "linear_attention"],
        },
        flowmap_config={"mlp_dim": 16, "mlp_depth": 2, "inference_steps": 8, "sampling_temperature": 1.0},
        audio_adapter_config={"hidden_size": 16},
        pad_token_id=1,
        silence_token_id=2,
        sequence_layout="flat",
    )


def test_duplexio_config_round_trips_nested_text_config() -> None:
    config = _config()
    restored = DuplexIOConfig.from_dict(config.to_dict())

    assert restored.get_text_config().model_type == "qwen3_5_text"
    assert restored.get_text_config().hidden_size == 32
    assert restored.flowmap_config["inference_steps"] == 8
    assert restored.audio_adapter_config == {"hidden_size": 16}


def test_duplexio_text_config_uses_one_dimensional_rope() -> None:
    config = _config()
    config.text_config.rope_parameters = {
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
        "partial_rotary_factor": 0.25,
        "rope_theta": 10_000_000,
        "rope_type": "default",
    }

    restored = DuplexIOConfig.from_dict(config.to_dict())
    rope_parameters = restored.get_text_config().rope_parameters

    assert "mrope_section" not in rope_parameters
    assert "mrope_interleaved" not in rope_parameters
    assert rope_parameters["rope_theta"] == 10_000_000
    assert rope_parameters["partial_rotary_factor"] == 0.25


def test_flat_step_installs_passes_at_stable_buffers_and_resets_inert() -> None:
    step = FlatStep(4 * FRAME_POSITIONS, torch.device("cpu"))
    pointers = (step.positions.data_ptr(), step.rope.data_ptr(), step.requests.data_ptr())
    positions = torch.arange(2 * FRAME_POSITIONS * STEP_COLUMNS, dtype=torch.int32).view(-1, STEP_COLUMNS)
    requests = torch.tensor([[0, 5, 1, 1], [5, 10, 0, 0]], dtype=torch.int32)

    step.install(positions, requests)

    torch.testing.assert_close(step.positions[: 2 * FRAME_POSITIONS], positions)
    torch.testing.assert_close(step.rope[: 2 * FRAME_POSITIONS], positions[:, 0].long())
    assert step.request_count(2 * FRAME_POSITIONS) == 2
    # An empty step leaves nothing addressable: every position is a no-op of its own frame.
    step.reset()
    assert torch.equal(step.positions[:, WRITE], torch.full((4 * FRAME_POSITIONS,), -1, dtype=torch.int32))
    attention = step.positions[:, ATTENTION]
    assert not (attention[:, 1:7] != 0).any() and bool((attention[:, 7] == 1).all())
    assert not step.positions[:, GDN][:, 1].any()
    assert step.request_count(3 * FRAME_POSITIONS) == 3
    assert pointers == (step.positions.data_ptr(), step.rope.data_ptr(), step.requests.data_ptr())


def test_duplexio_config_requires_the_flat_layout() -> None:
    config = _config().to_dict()
    config["sequence_layout"] = "grid"
    with pytest.raises(ValueError, match="flat"):
        DuplexIOConfig.from_dict(config)


def test_duplexio_config_requires_hybrid_qwen_backbone() -> None:
    config = _config().to_dict()
    config["text_config"]["layer_types"] = [
        "full_attention",
        "full_attention",
    ]

    with pytest.raises(ValueError, match="full- or linear-attention"):
        DuplexIOConfig.from_dict(config)


def test_duplexio_pipeline_uses_the_duplex_plugin() -> None:
    assert DUPLEXIO_PIPELINE.duplex_plugin.endswith(".DuplexIODuplexPlugin")
    assert len(DUPLEXIO_PIPELINE.stages) == 1
    assert DUPLEXIO_PIPELINE.stages[0].retains_state_across_chunks


def test_duplexio_qwen_backbone_uses_vllm_compile_boundary() -> None:
    assert TorchCompileWithNoGuardsWrapper in DuplexIOQwenModel.__bases__


def test_duplexio_cudagraph_supports_uniform_frame_batches() -> None:
    single = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1))
    multiple = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))

    # Paged attention masks by slot address, so any batch shape replays; the GDN
    # recurrence still needs one graph per batch size.
    for config in (single, multiple):
        assert DuplexIOFlashAttentionMetadataBuilder.get_cudagraph_support(config, None) is AttentionCGSupport.ALWAYS
    assert DuplexIOGDNAttentionMetadataBuilder.get_cudagraph_support(single, None) is AttentionCGSupport.ALWAYS
    assert DuplexIOGDNAttentionMetadataBuilder.get_cudagraph_support(multiple, None) is AttentionCGSupport.UNIFORM_BATCH


def test_duplexio_gdn_cache_allocation_preserves_fp32_recurrence() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(DuplexIOQwenGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.model_config = config.model_config
    planned = DuplexIOForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    assert (
        planned
        == attention.get_state_dtype()
        == (
            torch.bfloat16,
            torch.float32,
        )
    )


def test_duplexio_gdn_uses_its_own_backend() -> None:
    """vLLM's stock GDN metadata plans chunks by tokens; flat positions step their own streams."""
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(DuplexIOQwenGatedDeltaNetAttention)
    nn.Module.__init__(attention)

    assert attention.get_attn_backend() is DuplexIOGDNAttentionBackend


def test_mlp_adapters_return_backbone_inputs_without_a_skip() -> None:
    torch.manual_seed(0)
    # One adapter serves both audio cells; the agent's voice arrives as the
    # pinned prompt in its own cell, not as a conditioning vector here.
    user = AudioInputAdapter(5, 7, 11)
    agent_in = AudioInputAdapter(5, 7, 11)
    audio = torch.randn(4, 5)

    user_hidden = user(audio)
    agent_hidden = agent_in(audio)

    assert user_hidden.shape == (4, 11)
    assert agent_hidden.shape == (4, 11)


def test_unfinished_tool_context_does_not_run_prediction_heads() -> None:
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)

    predictions = model.sample_frames(
        torch.zeros(FRAME_POSITIONS, 4),
        [(0, FRAME_POSITIONS)],
        [{"duplex": {}}],
        [False],
    )

    assert predictions is None


def test_checkpoint_names_map_onto_the_flat_heads() -> None:
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(4, 8, bias=False)
    model.channel_emb = nn.Parameter(torch.zeros(4, 4))
    model.flat_cell_emb = nn.Parameter(torch.zeros(SLOT_COUNT, 4))
    model.user_emit_head = nn.Linear(4, 1)
    weights = {name: torch.randn_like(value) for name, value in model.state_dict().items()}
    checkpoint = {
        (
            "llm.base_model." + name
            if name.startswith("lm_head")
            else "llm." + name
            if name.startswith("channel_emb")
            else name
        ): value
        for name, value in weights.items()
    }
    checkpoint["user_asr.model.decoder.weight"] = torch.zeros(1)  # The ASR decoder is never loaded.

    assert model.load_weights(checkpoint.items()) == set(weights)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name])


def test_only_the_tool_stream_may_emit_tool_call_markers() -> None:
    encoded = {
        "<|im_start|>": [11],
        "<|im_end|>": [12],
        "<think>": [14],
        "</think>": [15],
        "<tool_call>": [18],
        "</tool_call>": [19],
    }
    tokenizer = SimpleNamespace(all_special_ids=[11, 12], encode=lambda text, **_: encoded[text])
    agent_ids, tool_ids = text_suppression_ids(tokenizer, 13)
    assert agent_ids == [11, 12, 13, 14, 15, 18, 19]
    assert tool_ids == [11, 12, 13, 14, 15]
    encoded["<think>"] = [14, 16]
    with pytest.raises(ValueError):
        text_suppression_ids(tokenizer, 13)
