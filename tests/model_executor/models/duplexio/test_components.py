# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.compilation.wrapper import TorchCompileWithNoGuardsWrapper
from vllm.v1.attention.backend import AttentionCGSupport
from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AgentAudioInputAdapter,
    AgentAudioOutputAdapter,
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    DelayedMimiRepresentation,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import (
    DuplexIOConfig,
)
from vllm_omni.model_executor.models.duplexio.depth_sampler import (
    DepthAutoregressiveSampler,
    DepthSamplerConfig,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerConfig,
    FastConformerUserEncoder,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    TokenSamplingOptions,
    _emit_temperatures,
    _sample_factorized_text_ids,
)
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE
from vllm_omni.model_executor.models.duplexio.qwen_backbone import (
    DuplexIOFlexAttentionMetadataBuilder,
    DuplexIOGDNAttentionBackend,
    DuplexIOGDNAttentionMetadataBuilder,
    DuplexIOQwenGatedDeltaNetAttention,
    DuplexIOQwenModel,
    _decode_uint32,
    _encode_uint32,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _config() -> DuplexIOConfig:
    return DuplexIOConfig(
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
        audio_codec_config={
            "model_type": "mimi",
            "num_quantizers": 32,
            "codebook_size": 2_048,
            "sampling_rate": 24_000,
            "frame_rate": 12.5,
        },
        audio_adapter_config={
            "architecture": "mlp",
            "hidden_size": 16,
            "agent_audio_skip_dropout": 0.2,
        },
        user_asr_encoder_config={
            "implementation": "nvidia_fastconformer_streaming_multi",
            "source_sample_rate": 24_000,
            "sample_rate": 16_000,
            "frame_size": 1_920,
            "features": 80,
            "n_fft": 512,
            "window_size": 400,
            "window_stride": 160,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 256,
            "num_layers": 17,
            "dim": 512,
            "feedforward_dim": 2_048,
            "num_heads": 8,
            "attention_left_context": 70,
            "attention_right_context": 0,
            "convolution_kernel_size": 9,
        },
        quantized_audio_config={
            "num_codebooks": 8,
            "codebook_size": 2_048,
            "embedding_dim": 512,
            "acoustic_delay_frames": 1,
        },
        depth_transformer_config={
            "implementation": "duplexio_speaker_adaptive_depth_v1",
            "low_rank_embeddings": 8,
            "dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_dim": 64,
            "sampling_temperature": 0.8,
            "sampling_top_k": 32,
            "semantic_sampling_top_k": 1,
            "codebook_loss_weights": [3, 3, 3, 2, 2, 2, 1, 1],
        },
        pad_token_id=1,
        silence_token_id=2,
    )


def test_duplexio_uint32_metadata_round_trip() -> None:
    values = torch.tensor([0, 1, 255, 256, 65_535, 2**32 - 1])

    encoded = _encode_uint32(values, torch.bfloat16)

    torch.testing.assert_close(_decode_uint32(encoded), values)


def test_duplexio_config_round_trips_nested_text_config() -> None:
    config = _config()
    restored = DuplexIOConfig.from_dict(config.to_dict())

    assert restored.get_text_config().model_type == "qwen3_5_text"
    assert restored.get_text_config().hidden_size == 32
    assert restored.quantized_audio_config["num_codebooks"] == 8
    assert restored.depth_transformer_config["implementation"] == (
        "duplexio_speaker_adaptive_depth_v1"
    )
    assert restored.user_asr_encoder_config["attention_left_context"] == 70
    assert restored.audio_adapter_config["agent_audio_skip_dropout"] == 0.2
    assert restored.initial_agent_prefix == "<|im_start|>assistant\n"
    assert restored.initial_user_prefix == "<|im_start|>user\n"


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


def test_duplexio_gdn_convolution_uses_only_valid_initial_state() -> None:
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(
        DuplexIOQwenGatedDeltaNetAttention
    )
    nn.Module.__init__(attention)
    attention.full_cudagraph_enabled = False
    attention.conv1d = nn.Conv1d(1, 1, 3, groups=1, bias=False)
    attention.conv1d.weight.data.copy_(torch.tensor([[[1.0, 0.0, 0.0]]]))
    attention.activation = "silu"
    conv_state = torch.tensor([[[2.0, 3.0]], [[2.0, 3.0]]])

    output = attention.apply_stream_causal_conv(
        mixed_qkv=torch.tensor([[5.0], [5.0]]),
        conv_state=conv_state,
        state_indices=torch.tensor([0, 1]),
        query_start_loc=torch.tensor([0, 1, 2]),
        has_initial_state=torch.tensor([True, False]),
    )

    torch.testing.assert_close(
        output[:, 0],
        torch.tensor([torch.nn.functional.silu(torch.tensor(2.0)), 0.0]),
    )
    torch.testing.assert_close(
        conv_state,
        torch.tensor([[[3.0, 5.0]], [[0.0, 5.0]]]),
    )


def test_duplexio_single_request_convolution_updates_state_without_scalar_reads() -> None:
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(
        DuplexIOQwenGatedDeltaNetAttention
    )
    nn.Module.__init__(attention)
    attention.full_cudagraph_enabled = True
    attention.conv1d = nn.Conv1d(1, 1, 3, groups=1, bias=False)
    attention.conv1d.weight.data.copy_(torch.tensor([[[1.0, 0.0, 0.0]]]))
    attention.activation = "silu"
    conv_state = torch.tensor([[[2.0, 3.0]], [[7.0, 8.0]]])

    output = attention.apply_stream_causal_conv(
        mixed_qkv=torch.tensor([[5.0], [6.0]]),
        conv_state=conv_state,
        state_indices=torch.tensor([0], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2]),
        has_initial_state=torch.tensor([True]),
    )

    torch.testing.assert_close(
        output[:, 0],
        torch.nn.functional.silu(torch.tensor([2.0, 3.0])),
    )
    torch.testing.assert_close(
        conv_state,
        torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
    )


def test_duplexio_copies_current_metadata_into_stable_graph_buffers() -> None:
    model = DuplexIOForConditionalGeneration.__new__(
        DuplexIOForConditionalGeneration
    )
    nn.Module.__init__(model)
    model.register_buffer(
        "graph_key_active",
        torch.ones(6, dtype=torch.bool),
        persistent=False,
    )
    model.register_buffer(
        "graph_request_epochs",
        torch.zeros(6, dtype=torch.long),
        persistent=False,
    )
    model.register_buffer(
        "graph_text_ordinals",
        torch.zeros(6, dtype=torch.long),
        persistent=False,
    )
    pointers = (
        model.graph_key_active.data_ptr(),
        model.graph_request_epochs.data_ptr(),
        model.graph_text_ordinals.data_ptr(),
    )
    info = {
        "duplexio": {
            "key_active": torch.tensor([True, False, True, True, False, True]),
            "request_epochs": torch.arange(6),
            "text_ordinals": torch.arange(6) + 10,
        }
    }

    model.update_graph_inputs([info])

    torch.testing.assert_close(
        model.graph_key_active,
        info["duplexio"]["key_active"],
    )
    torch.testing.assert_close(
        model.graph_request_epochs,
        info["duplexio"]["request_epochs"],
    )
    torch.testing.assert_close(
        model.graph_text_ordinals,
        info["duplexio"]["text_ordinals"],
    )
    assert pointers == (
        model.graph_key_active.data_ptr(),
        model.graph_request_epochs.data_ptr(),
        model.graph_text_ordinals.data_ptr(),
    )


def test_duplexio_graph_replays_only_live_frames() -> None:
    model = DuplexIOForConditionalGeneration.__new__(
        DuplexIOForConditionalGeneration
    )

    assert model.supports_cudagraph_replay([{"duplex": {}}])
    assert not model.supports_cudagraph_replay(
        [{"duplex": {"duplexio_prefill": True}}]
    )
    assert not model.supports_cudagraph_replay(
        [{"duplex": {"duplexio_system_input": True}}]
    )


def test_duplexio_config_rejects_non_row_quantum() -> None:
    with pytest.raises(ValueError, match="six cells"):
        DuplexIOConfig(
            text_config={"model_type": "qwen3_5_text"},
            num_cells=5,
            audio_codec_config={
                "model_type": "mimi",
                "num_quantizers": 32,
                "codebook_size": 2_048,
                "sampling_rate": 24_000,
                "frame_rate": 12.5,
            },
            audio_adapter_config={"architecture": "mlp"},
            quantized_audio_config={
                "num_codebooks": 8,
                "codebook_size": 2_048,
            },
        )


def test_duplexio_config_rejects_old_depth_checkpoint() -> None:
    config = _config().to_dict()
    config["depth_transformer_config"] = {
        "implementation": "moshi_original_depformer",
    }

    with pytest.raises(ValueError, match="speaker-conditioned"):
        DuplexIOConfig.from_dict(config)


def test_duplexio_config_accepts_full_width_depth_embeddings() -> None:
    config = _config().to_dict()
    config["depth_transformer_config"]["low_rank_embeddings"] = None

    assert DuplexIOConfig.from_dict(config).depth_transformer_config[
        "low_rank_embeddings"
    ] is None


def test_duplexio_config_rejects_non_native_acoustic_delay() -> None:
    config = _config().to_dict()
    config["quantized_audio_config"]["acoustic_delay_frames"] = 2

    with pytest.raises(ValueError, match="one acoustic delay frame"):
        DuplexIOConfig.from_dict(config)


def test_duplexio_config_requires_hybrid_qwen_backbone() -> None:
    config = _config().to_dict()
    config["text_config"]["layer_types"] = [
        "full_attention",
        "full_attention",
    ]

    with pytest.raises(ValueError, match="full-attention and linear-attention"):
        DuplexIOConfig.from_dict(config)


def test_duplexio_pipeline_uses_native_full_duplex_control_plane() -> None:
    assert DUPLEXIO_PIPELINE.duplex_control_enabled
    assert DUPLEXIO_PIPELINE.duplex_runtime_extension.endswith(
        ".DuplexIORuntimeExtension"
    )
    assert DUPLEXIO_PIPELINE.duplex_serving_adapter.endswith(
        ".DuplexIOServingRuntimeAdapter"
    )
    assert len(DUPLEXIO_PIPELINE.stages) == 1
    assert DUPLEXIO_PIPELINE.stages[0].retains_state_across_chunks


def test_duplexio_qwen_backbone_uses_vllm_compile_boundary() -> None:
    assert TorchCompileWithNoGuardsWrapper in DuplexIOQwenModel.__bases__


@pytest.mark.parametrize(
    ("builder", "multiple_request_support"),
    [
        (DuplexIOFlexAttentionMetadataBuilder, AttentionCGSupport.NEVER),
        (
            DuplexIOGDNAttentionMetadataBuilder,
            AttentionCGSupport.UNIFORM_BATCH,
        ),
    ],
)
def test_duplexio_full_cudagraph_support_requires_one_request(
    builder: type,
    multiple_request_support: AttentionCGSupport,
) -> None:
    single = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=1)
    )
    multiple = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=2)
    )

    assert builder.get_cudagraph_support(single, None) is AttentionCGSupport.ALWAYS
    assert (
        builder.get_cudagraph_support(multiple, None)
        is multiple_request_support
    )


def test_duplexio_gdn_uses_single_request_graph_backend() -> None:
    attention = DuplexIOQwenGatedDeltaNetAttention.__new__(
        DuplexIOQwenGatedDeltaNetAttention
    )
    nn.Module.__init__(attention)
    attention.full_cudagraph_enabled = True

    assert attention.get_attn_backend() is DuplexIOGDNAttentionBackend


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
    builder.full_graph_metadata = metadata
    common = SimpleNamespace(
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


def test_mlp_adapters_keep_one_local_skip_per_audio_cell() -> None:
    torch.manual_seed(0)
    user = AudioInputAdapter(5, 7, 11)
    agent_in = AgentAudioInputAdapter(5, 3, 7, 11)
    agent_out = AgentAudioOutputAdapter(11, 7, 3, 13)
    audio = torch.randn(4, 5)
    speakers = torch.randn(2, 3)
    request_indices = torch.tensor([0, 0, 1, 1])

    user_hidden, user_skip = user(audio)
    agent_hidden, agent_skip = agent_in(audio, speakers, request_indices)
    output = agent_out(
        agent_hidden,
        agent_skip,
        speakers,
        request_indices,
    )

    assert user_hidden.shape == (4, 11)
    assert user_skip.shape == (4, 7)
    assert agent_hidden.shape == (4, 11)
    assert agent_skip.shape == (4, 7)
    assert output.shape == (4, 11)


def test_delayed_mimi_streaming_reassembles_raw_columns() -> None:
    representation = DelayedMimiRepresentation(
        num_codebooks=3,
        codebook_size=10,
        acoustic_delay_frames=1,
    )
    encode_state = representation.new_state(device=torch.device("cpu"))
    decode_state = representation.new_state(device=torch.device("cpu"))
    raw = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]

    delayed = [
        representation.encode_column(column, encode_state) for column in raw
    ]
    decoded = [
        representation.decode_column(column, decode_state) for column in delayed
    ]

    assert torch.equal(delayed[0], torch.tensor([1, 10, 10]))
    assert torch.equal(delayed[1], torch.tensor([4, 2, 3]))
    assert decoded[0] is None
    assert torch.equal(decoded[1], raw[0])


def test_delayed_mimi_sequence_matches_column_streaming() -> None:
    representation = DelayedMimiRepresentation(
        num_codebooks=3,
        codebook_size=10,
        acoustic_delay_frames=1,
    )
    raw = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    streaming_state = representation.new_state(device=torch.device("cpu"))
    sequence_state = representation.new_state(device=torch.device("cpu"))

    streaming = torch.stack(
        [
            representation.encode_column(column, streaming_state)
            for column in raw
        ]
    )
    sequence = representation.encode_sequence(raw, sequence_state)

    torch.testing.assert_close(sequence, streaming)
    torch.testing.assert_close(
        sequence_state.previous_acoustic_codes,
        streaming_state.previous_acoustic_codes,
    )


def test_mimi_embedding_keeps_checkpoint_module_layout() -> None:
    embedding = MimiEmbedding(3, 7, 5)

    assert set(embedding.state_dict()) == {
        "embeddings.0.weight",
        "embeddings.1.weight",
        "embeddings.2.weight",
    }
    assert embedding(torch.tensor([[1, 2, 3]])).shape == (1, 5)


def test_speaker_depth_sampling_is_deterministic_at_top_k_one() -> None:
    torch.manual_seed(1)
    model = DepthAutoregressiveSampler(
        DepthSamplerConfig(
            conditioning_dim=5,
            speaker_embedding_dim=6,
            text_vocab_size=11,
            codebook_size=7,
            num_codebooks=3,
            low_rank_embeddings=2,
            dim=8,
            num_layers=2,
            num_heads=2,
            feedforward_dim=12,
            sampling_top_k=1,
        )
    ).eval()
    conditioning = torch.randn(2, 5)
    text_tokens = torch.tensor([2, 4])
    speakers = torch.randn(2, 6)

    first = model.sample(conditioning, text_tokens, speakers)
    second = model.sample(conditioning, text_tokens, speakers)

    torch.testing.assert_close(first, second)


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
            mode="argmax",
            temperature=1.0,
            top_k=4,
            top_p=0.95,
            suppressed_token_ids=(),
        ),
        emit_temperature=0.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(sampled, torch.tensor([2, 0, 1]))


def test_emit_temperatures_are_read_per_stream() -> None:
    temperatures = _emit_temperatures(
        {
            "duplex": {
                "runtime_config": {
                    "duplexio_emit_temperatures": {
                        "user": 0.2,
                        "agent": 0.4,
                        "tool_call": 0.8,
                    }
                }
            }
        }
    )

    assert temperatures.user == 0.2
    assert temperatures.agent == 0.4
    assert temperatures.tool_call == 0.8


def test_speaker_depth_conditioning_changes_adaptive_normalization() -> None:
    torch.manual_seed(2)
    model = DepthAutoregressiveSampler(
        DepthSamplerConfig(
            conditioning_dim=5,
            speaker_embedding_dim=6,
            text_vocab_size=11,
            codebook_size=7,
            num_codebooks=3,
            low_rank_embeddings=2,
            dim=8,
            num_layers=2,
            num_heads=2,
            feedforward_dim=12,
            sampling_top_k=1,
        )
    ).eval()
    hidden = torch.randn(2, 1, 8)
    speakers = torch.randn(2, 6)
    speaker_states = model.speaker_projection(speakers)
    norm = model.transformer.layers[0].attention_norm

    first = norm(hidden, speaker_states)
    second = norm(hidden, speaker_states.flip(0))

    assert not torch.allclose(first, second)


def test_depth_sampler_uses_current_checkpoint_module_names() -> None:
    model = DepthAutoregressiveSampler(
        DepthSamplerConfig(
            conditioning_dim=5,
            speaker_embedding_dim=6,
            text_vocab_size=11,
            codebook_size=7,
            num_codebooks=3,
            low_rank_embeddings=2,
            dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=12,
        )
    )
    keys = set(model.state_dict())

    assert "conditioning_projections.0.weight" in keys
    assert "speaker_projection.weight" in keys
    assert "previous_codebook_embeddings.0.output_projection.weight" in keys
    assert "text_embedding.output_projection.weight" in keys
    assert "transformer.layers.0.attention.input_projections.0.weight" in keys
    assert "transformer.layers.0.feedforward.layers.0.input.weight" in keys
    assert "transformer.layers.0.attention_norm.modulation.1.weight" in keys
    assert "heads.0.weight" in keys


def test_fastconformer_streaming_matches_full_prefix_and_stays_bounded() -> None:
    torch.manual_seed(4)
    config = FastConformerConfig(
        source_sample_rate=96,
        sample_rate=64,
        frame_size=96,
        features=8,
        n_fft=32,
        window_size=24,
        window_stride=8,
        subsampling_conv_channels=4,
        num_layers=2,
        dim=8,
        feedforward_dim=16,
        num_heads=2,
        attention_left_context=3,
        convolution_kernel_size=3,
    )
    model = FastConformerUserEncoder(config).eval()
    model.preprocessor.featurizer.window.copy_(torch.hann_window(24))
    model.preprocessor.featurizer.fb.fill_(1 / 17)
    for layer in model.encoder.layers:
        layer.self_attn.pos_bias_u.normal_()
        layer.self_attn.pos_bias_v.normal_()
    state = model.new_state(device=torch.device("cpu"))
    frames = torch.randn(20, 1, 1, 96)
    streaming = []
    with torch.inference_mode():
        for frame in frames:
            feature, state = model.step(frame, state)
            streaming.append(feature)
        offline = model.encode_prefix(frames[:, 0, 0])

    torch.testing.assert_close(
        torch.stack(streaming),
        offline,
        rtol=1e-5,
        atol=1e-6,
    )
    assert state.frames_seen == 20
    assert state.sample_buffer.shape == (24,)
    assert state.feature_buffer.shape == (8, 16)
    assert all(cache.shape == (3, 8) for cache in state.attention_caches)
    assert all(cache.shape == (8, 2) for cache in state.convolution_caches)

    with torch.inference_mode():
        prefix_features, silent_state = model.encode_silent_prefix(
            5,
            device=torch.device("cpu"),
        )
        silent_waveform = torch.zeros(5, 96)
        silent_offline = model.encode_prefix(silent_waveform)
        next_frame = torch.randn(1, 1, 96)
        next_feature, silent_state = model.step(next_frame, silent_state)
        extended_offline = model.encode_prefix(
            torch.cat((silent_waveform, next_frame[0]))
        )

    torch.testing.assert_close(
        prefix_features,
        silent_offline,
        rtol=1e-5,
        atol=1e-6,
    )
    torch.testing.assert_close(
        next_feature,
        extended_offline[-1],
        rtol=1e-5,
        atol=1e-6,
    )
    assert silent_state.frames_seen == 6


@pytest.mark.parametrize("prefix_frames", [0, 3])
def test_fastconformer_step_sequence_matches_serial_steps(
    prefix_frames: int,
) -> None:
    torch.manual_seed(8)
    config = FastConformerConfig(
        source_sample_rate=96,
        sample_rate=64,
        frame_size=96,
        features=8,
        n_fft=32,
        window_size=24,
        window_stride=8,
        subsampling_conv_channels=4,
        num_layers=2,
        dim=8,
        feedforward_dim=16,
        num_heads=2,
        attention_left_context=3,
        convolution_kernel_size=3,
    )
    model = FastConformerUserEncoder(config).eval()
    model.preprocessor.featurizer.window.copy_(torch.hann_window(24))
    model.preprocessor.featurizer.fb.fill_(1 / 17)
    frames = torch.randn(prefix_frames + 5, 1, 1, 96)

    prefix_state = model.new_state(device=torch.device("cpu"))
    with torch.inference_mode():
        for frame in frames[:prefix_frames]:
            _, prefix_state = model.step(frame, prefix_state)

    serial_state = prefix_state
    serial_features = []
    with torch.inference_mode():
        for frame in frames[prefix_frames:]:
            feature, serial_state = model.step(frame, serial_state)
            serial_features.append(feature)
        sequence_features, sequence_state = model.step_sequence(
            frames[prefix_frames:, 0].reshape(1, 1, -1),
            prefix_state,
        )

    torch.testing.assert_close(sequence_features, torch.stack(serial_features))
    assert sequence_state.frames_seen == serial_state.frames_seen == prefix_frames + 5
    torch.testing.assert_close(
        sequence_state.sample_buffer,
        serial_state.sample_buffer,
    )
    torch.testing.assert_close(
        sequence_state.feature_buffer,
        serial_state.feature_buffer,
    )
    for sequence_cache, serial_cache in zip(
        sequence_state.attention_caches,
        serial_state.attention_caches,
        strict=True,
    ):
        torch.testing.assert_close(sequence_cache, serial_cache)
    for sequence_cache, serial_cache in zip(
        sequence_state.convolution_caches,
        serial_state.convolution_caches,
        strict=True,
    ):
        torch.testing.assert_close(sequence_cache, serial_cache)
