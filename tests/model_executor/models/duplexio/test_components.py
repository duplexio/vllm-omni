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
from vllm_omni.model_executor.models.duplexio.moshi_depth import (
    MoshiDepthConfig,
    MoshiDepthTransformer,
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
        audio_adapter_config={"architecture": "mlp", "hidden_size": 16},
        quantized_audio_config={
            "num_codebooks": 8,
            "codebook_size": 2_048,
            "embedding_dim": 512,
            "acoustic_delay_frames": 1,
        },
        depth_transformer_config={
            "implementation": "moshi_original_depformer",
            "original_moshi_compatible": True,
            "low_rank_embeddings": 8,
            "dim": 32,
            "num_layers": 2,
            "num_heads": 4,
            "mlp_dim": 64,
            "sampling_temperature": 0.8,
            "sampling_top_k": 32,
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
        "moshi_original_depformer"
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


def test_duplexio_config_rejects_non_original_depth_checkpoint() -> None:
    config = _config().to_dict()
    config["depth_transformer_config"] = {
        "implementation": "duplexio_speaker_adaptive_depth_v1",
        "original_moshi_compatible": False,
    }

    with pytest.raises(ValueError, match="original Moshi"):
        DuplexIOConfig.from_dict(config)


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


def test_moshi_depth_teacher_logits_match_sequential_argmax() -> None:
    torch.manual_seed(1)
    model = MoshiDepthTransformer(
        MoshiDepthConfig(
            conditioning_dim=5,
            text_vocab_size=11,
            codebook_size=7,
            num_codebooks=3,
            low_rank_embeddings=2,
            dim=8,
            num_layers=2,
            num_heads=2,
            feedforward_dim=12,
            sampling_top_k=3,
        )
    ).eval()
    conditioning = torch.randn(2, 5)
    text_tokens = torch.tensor([2, 4])

    sampled = model.sample(conditioning, text_tokens, temperature=0, top_k=3)
    teacher_logits = model(conditioning, text_tokens, sampled)

    torch.testing.assert_close(teacher_logits.argmax(dim=-1), sampled)


def test_moshi_depth_is_causal_across_codebooks() -> None:
    torch.manual_seed(2)
    model = MoshiDepthTransformer(
        MoshiDepthConfig(
            conditioning_dim=5,
            text_vocab_size=11,
            codebook_size=7,
            num_codebooks=3,
            low_rank_embeddings=2,
            dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=12,
        )
    ).eval()
    conditioning = torch.randn(2, 5)
    text_tokens = torch.tensor([2, 4])
    target = torch.tensor([[1, 2, 3], [3, 2, 1]])
    changed = target.clone()
    changed[:, 1] = (changed[:, 1] + 1) % 7

    logits = model(conditioning, text_tokens, target)
    changed_logits = model(conditioning, text_tokens, changed)

    torch.testing.assert_close(logits[:, :2], changed_logits[:, :2])
    assert not torch.equal(logits[:, 2], changed_logits[:, 2])


def test_moshi_depth_uses_original_checkpoint_module_names() -> None:
    model = MoshiDepthTransformer(
        MoshiDepthConfig(
            conditioning_dim=5,
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

    assert "depformer_in.0.weight" in keys
    assert "depformer_emb.0.low_rank.weight" in keys
    assert "depformer_text_emb.low_rank.weight" in keys
    assert "depformer.layers.0.self_attn.in_projs.0.weight" in keys
    assert "depformer.layers.0.gating.0.linear_in.weight" in keys
    assert "depformer.layers.0.norm1.alpha" in keys
    assert "linears.0.weight" in keys
    assert not any("speaker" in key for key in keys)
