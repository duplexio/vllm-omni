# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

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
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    _sample_factorized_text_ids,
)
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE

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


def test_duplexio_config_round_trips_nested_text_config() -> None:
    config = _config()
    restored = DuplexIOConfig.from_dict(config.to_dict())

    assert restored.get_text_config().model_type == "qwen3_5_text"
    assert restored.get_text_config().hidden_size == 32
    assert restored.quantized_audio_config["num_codebooks"] == 8
    assert restored.depth_transformer_config["implementation"] == (
        "duplexio_speaker_adaptive_depth_v1"
    )
    assert restored.audio_adapter_config["agent_audio_skip_dropout"] == 0.2
    assert restored.initial_agent_prefix == "<|im_start|>assistant\n"
    assert restored.initial_user_prefix == "<|im_start|>user\n"


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
        temperature=0.0,
        generator=torch.Generator().manual_seed(1),
    )

    assert torch.equal(sampled, torch.tensor([2, 0, 1]))


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


def test_duplexio_config_rejects_pre_v3_export() -> None:
    config = _config().to_dict()
    config["duplexio_export_version"] = 2

    with pytest.raises(ValueError, match="export version"):
        DuplexIOConfig.from_dict(config)
