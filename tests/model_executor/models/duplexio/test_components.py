# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend, GDNAttentionMetadata

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.kv_reclamation import (
    DuplexIOFrameMetadata,
    DuplexIOKVLayout,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    EmitSamplingTemperatures,
    TokenSamplingOptions,
    _sample_factorized_text_ids,
    text_suppression_ids,
)
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOFlashAttentionMetadataBuilder,
    DuplexIOGDNAttentionBackend,
    DuplexIOGDNAttentionMetadataBuilder,
    DuplexIOQwenGatedDeltaNetAttention,
    DuplexIOQwenModel,
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




def test_duplexio_installs_cell_addressing_at_stable_buffers() -> None:
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    layout = DuplexIOKVLayout(
        block_size=16,
        audio_window_frames=2,
        max_model_len=120,
    )
    model.frame = DuplexIOFrameMetadata(12, torch.device("cpu"))
    frame = model.frame
    buffers = (
        frame.key_active,
        frame.persistent_ordinal,
        frame.persistent_last,
        frame.audio_first,
        frame.audio_last,
        frame.positions,
    )
    pointers = tuple(buffer.data_ptr() for buffer in buffers)
    info = {
        "duplexio": {
            "key_active": torch.tensor([True, False, False, False, True, True]),
            "persistent_ordinal": torch.tensor([7, 0, 0, 0, 0, 0], dtype=torch.int32),
            "persistent_last": torch.full((6,), 6, dtype=torch.int32),
            "audio_first": torch.full((6,), 1, dtype=torch.int32),
            "audio_last": torch.full((6,), 4, dtype=torch.int32),
            "positions": torch.arange(30, 36),
        }
    }

    model.update_graph_inputs([info, info])

    for name, buffer in zip(info["duplexio"], buffers, strict=True):
        expected = info["duplexio"][name]
        torch.testing.assert_close(buffer[:12], torch.cat((expected, expected)))

    # An empty step must leave nothing addressable behind.
    model.update_graph_inputs([])

    assert torch.equal(frame.write_slots(12, layout), torch.full((12,), -1))
    start, end, persistent = frame.row_reads(12)
    assert torch.equal(start, end) and not persistent.any()
    assert pointers == tuple(buffer.data_ptr() for buffer in buffers)


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


def test_duplexio_cudagraph_supports_uniform_six_cell_batches() -> None:
    single = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=1))
    multiple = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2))

    # Paged attention masks by slot address, so any batch shape replays; the GDN
    # recurrence still needs one graph per batch size.
    for config in (single, multiple):
        assert (
            DuplexIOFlashAttentionMetadataBuilder.get_cudagraph_support(config, None)
            is AttentionCGSupport.ALWAYS
        )
    assert (
        DuplexIOGDNAttentionMetadataBuilder.get_cudagraph_support(single, None)
        is AttentionCGSupport.ALWAYS
    )
    assert (
        DuplexIOGDNAttentionMetadataBuilder.get_cudagraph_support(multiple, None)
        is AttentionCGSupport.UNIFORM_BATCH
    )


def test_duplexio_gdn_uses_frame_graph_backend() -> None:
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(
        DuplexIOQwenGatedDeltaNetAttention
    )
    nn.Module.__init__(attention)
    attention.full_cudagraph_enabled = True

    assert attention.get_attn_backend() is DuplexIOGDNAttentionBackend


def test_duplexio_gdn_cache_allocation_preserves_fp32_recurrence() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.bfloat16))
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(DuplexIOQwenGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.model_config = config.model_config
    planned = DuplexIOForConditionalGeneration.get_mamba_state_dtype_from_config(config)
    assert planned == attention.get_state_dtype() == (
        torch.bfloat16, torch.float32,
    )


def test_duplexio_gdn_uses_standard_backend_in_eager_mode() -> None:
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(
        DuplexIOQwenGatedDeltaNetAttention
    )
    nn.Module.__init__(attention)
    attention.full_cudagraph_enabled = False

    assert attention.get_attn_backend() is GDNAttentionBackend


def test_duplexio_gdn_graph_refreshes_every_recurrent_state_input(
    monkeypatch,
) -> None:
    builder = DuplexIOGDNAttentionMetadataBuilder.__new__(
        DuplexIOGDNAttentionMetadataBuilder
    )
    builder.kv_cache_spec = object()
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align")
    )
    metadata = SimpleNamespace(
        non_spec_state_indices_tensor=torch.zeros(1, dtype=torch.long),
        has_initial_state=torch.zeros(1, dtype=torch.bool),
        prefill_state_indices=torch.zeros(1, dtype=torch.long),
        prefill_has_initial_state=torch.zeros(1, dtype=torch.bool),
    )
    builder.full_graph_metadata = {1: metadata}
    common = SimpleNamespace(
        num_reqs=1,
        block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
        seq_lens=torch.tensor([12], dtype=torch.int32),
        compute_num_computed_tokens=lambda: torch.tensor([6]),
    )
    monkeypatch.setattr(
        "vllm_omni.model_executor.models.duplexio.qwen_backbone."
        "mamba_get_block_table_tensor",
        lambda *args: torch.tensor([[7]], dtype=torch.long),
    )

    refreshed = builder.refresh_full_graph_metadata(common)

    assert refreshed is metadata
    torch.testing.assert_close(
        metadata.non_spec_state_indices_tensor,
        torch.tensor([7]),
    )
    torch.testing.assert_close(
        metadata.prefill_state_indices,
        torch.tensor([7]),
    )
    torch.testing.assert_close(
        metadata.has_initial_state,
        torch.tensor([True]),
    )
    torch.testing.assert_close(
        metadata.prefill_has_initial_state,
        torch.tensor([True]),
    )


def test_gdn_graph_buffers_are_independent_per_batch_size() -> None:
    builder = DuplexIOGDNAttentionMetadataBuilder.__new__(DuplexIOGDNAttentionMetadataBuilder)
    builder.full_graph_metadata = {}
    for requests in (1, 3, 8):
        boundaries = torch.arange(requests + 1, dtype=torch.int32) * 6
        metadata = GDNAttentionMetadata(
            num_prefills=requests, num_prefill_tokens=requests * 6,
            num_decodes=0, num_decode_tokens=0, num_spec_decodes=0,
            num_spec_decode_tokens=0, num_actual_tokens=requests * 6,
            non_spec_query_start_loc=boundaries,
            non_spec_state_indices_tensor=torch.arange(requests, dtype=torch.int32),
            has_initial_state=torch.ones(requests, dtype=torch.bool),
            chunk_indices=torch.stack((torch.arange(requests), torch.zeros(requests, dtype=torch.long)), 1),
            chunk_offsets=torch.arange(requests + 1),
        )
        retained = builder.retain_full_graph_metadata(metadata)
        boundaries.fill_(-1)
        torch.testing.assert_close(retained.prefill_query_start_loc, torch.arange(requests + 1, dtype=torch.int32) * 6)
    for requests, metadata in builder.full_graph_metadata.items():
        assert metadata.num_actual_tokens == requests * 6
        torch.testing.assert_close(metadata.prefill_query_start_loc, torch.arange(requests + 1, dtype=torch.int32) * 6)


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


def test_factorized_text_argmax_excludes_silence_from_content() -> None:
    logits = torch.tensor(
        [
            [0.0, 1.0, 100.0, 3.0],
            [4.0, 2.0, 100.0, 1.0],
            [1.0, 5.0, 100.0, 2.0],
        ]
    )
    emit_logits = torch.tensor([-1.0, 0.0, 1.0])

    sampled = _sample_factorized_text_ids(
        logits,
        emit_logits,
        silence_token_id=2,
        sampling=TokenSamplingOptions(
            temperature=0.0,
            top_k=4,
            top_p=0.95,
            suppressed_token_ids=torch.tensor([2], dtype=torch.long),
        ),
        emit_temperature=0.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(sampled, torch.tensor([2, 0, 1]))


def test_emit_temperatures_are_read_per_stream() -> None:
    temperatures = EmitSamplingTemperatures.model_validate({"user": 0.6, "agent": 0.4, "tool_call": 0.8})

    assert temperatures.user == 0.6
    assert temperatures.agent == 0.4
    assert temperatures.tool_call == 0.8


def test_unfinished_tool_context_does_not_run_prediction_heads() -> None:
    model = DuplexIOForConditionalGeneration.__new__(
        DuplexIOForConditionalGeneration
    )

    predictions = model.sample_frames(
        torch.zeros(6, 4),
        [(0, 6)],
        [{"duplex": {}}],
        [False],
    )

    assert predictions is None


def test_token_heads_load_without_weight_name_mapping() -> None:
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.llm = nn.Module()
    model.llm.output_head_proj = nn.ModuleDict(
        {name: nn.Linear(4, 4) for name in ("agent", "tool_call")}
    )
    model.user_token_projection = nn.Linear(24, 4)
    model.user_emit_head = nn.Linear(24, 1)
    weights = {name: torch.randn_like(value) for name, value in model.state_dict().items()}

    assert model.load_weights(weights.items()) == set(weights)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, weights[name])


def test_vocabulary_suppression_is_model_owned_and_sampling_temperature_stays_dynamic() -> None:
    encoded = {
        "<|im_start|>": [11], "<|im_end|>": [12],
        "<think>": [14, 15], "</think>": [16, 17],
        "<tool_call>": [18], "</tool_call>": [19],
    }
    tokenizer = SimpleNamespace(all_special_ids=[11, 12], encode=lambda text, **_: encoded[text])
    agent_ids, tool_ids = text_suppression_ids(tokenizer, 13)
    assert agent_ids == [11, 12, 13, 18, 19]
    assert tool_ids == [11, 12, 13]
    suppressed = torch.tensor(agent_ids, dtype=torch.long)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.agent_suppressed_token_ids = suppressed
    model.tool_suppressed_token_ids = torch.tensor(tool_ids)
    model.user_suppressed_token_ids = suppressed
    model.text_config = SimpleNamespace(vocab_size=20)
    model.config = SimpleNamespace(flowmap_config={"sampling_temperature": 1.0})
    options = {"temperature": 0.3, "top_k": 5, "top_p": 1.0}
    runtime = {"duplexio_text_sampling": options, "duplexio_user_sampling": {"content": options},
               "duplexio_emit_temperatures": {"user": 1.0, "agent": 1.0, "tool_call": 1.0}}
    first = model.resolve_sampling(runtime).agent
    options["temperature"] = 1.2
    second = model.resolve_sampling(runtime).agent
    assert (first.temperature, second.temperature) == (0.3, 1.2)
    assert first.suppressed_token_ids is second.suppressed_token_ids is suppressed
