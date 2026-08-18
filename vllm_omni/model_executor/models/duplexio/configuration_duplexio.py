# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration for a self-contained native DuplexIO checkpoint."""

from __future__ import annotations

import math
from typing import Any

from transformers import AutoConfig, PretrainedConfig

INITIAL_AGENT_PREFIX = "<|im_start|>assistant\n"
INITIAL_USER_PREFIX = "<|im_start|>user\n"


class DuplexIOConfig(PretrainedConfig):
    """Serving-only configuration exported by the DuplexIO training package."""

    model_type = "duplexio"
    is_composition = True
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | PretrainedConfig | None = None,
        audio_codec_config: dict[str, Any] | None = None,
        duplexio_export_version: int = 3,
        stream_names: list[str] | None = None,
        audio_cell_names: list[str] | None = None,
        num_cells: int = 6,
        sample_rate: int = 24_000,
        frame_rate: float = 12.5,
        frame_size: int = 1_920,
        audio_attention_window_frames: int = 4_096,
        silence_token_id: int | None = None,
        default_system_prompt: str = "",
        initial_agent_prefix: str = INITIAL_AGENT_PREFIX,
        initial_user_prefix: str = INITIAL_USER_PREFIX,
        speaker_embed_dim: int = 2_048,
        default_voice: str | None = None,
        audio_adapter_config: dict[str, Any] | None = None,
        quantized_audio_config: dict[str, Any] | None = None,
        depth_transformer_config: dict[str, Any] | None = None,
        tied_weight_aliases: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.text_config = _text_config(text_config)
        super().__init__(**kwargs)
        self.audio_codec_config = dict(audio_codec_config or {})
        self.duplexio_export_version = duplexio_export_version
        self.stream_names = stream_names or ["system", "user", "agent", "tool_call"]
        self.audio_cell_names = audio_cell_names or ["user_audio", "agent_audio"]
        self.num_cells = num_cells
        self.sample_rate = sample_rate
        self.frame_rate = frame_rate
        self.frame_size = frame_size
        self.audio_attention_window_frames = audio_attention_window_frames
        self.silence_token_id = silence_token_id
        self.default_system_prompt = default_system_prompt
        self.initial_agent_prefix = initial_agent_prefix
        self.initial_user_prefix = initial_user_prefix
        self.speaker_embed_dim = speaker_embed_dim
        self.default_voice = default_voice
        self.audio_adapter_config = dict(
            audio_adapter_config or {"architecture": "mlp"}
        )
        self.quantized_audio_config = dict(
            quantized_audio_config or {"num_codebooks": 8}
        )
        self.depth_transformer_config = dict(depth_transformer_config or {})
        self.tied_weight_aliases = dict(tied_weight_aliases or {})
        self._validate_duplexio_contract()

    def get_text_config(
        self,
        decoder: bool | None = None,
        encoder: bool | None = None,
    ) -> PretrainedConfig:
        """Return Qwen's config for vLLM cache and parallelism planning."""
        return self.text_config

    def _validate_duplexio_contract(self) -> None:
        # Version 3 is version 2 minus the user-ASR (FastConformer) surface.
        if self.duplexio_export_version != 3:
            raise ValueError(
                "Unsupported DuplexIO export version: "
                f"{self.duplexio_export_version}"
            )
        if self.stream_names != ["system", "user", "agent", "tool_call"]:
            raise ValueError(f"Unsupported DuplexIO stream layout: {self.stream_names}")
        if self.audio_cell_names != ["user_audio", "agent_audio"]:
            raise ValueError(
                f"Unsupported DuplexIO audio-cell layout: {self.audio_cell_names}"
            )
        if self.num_cells != 6:
            raise ValueError(f"DuplexIO requires six cells per frame, got {self.num_cells}")
        if self.initial_agent_prefix != INITIAL_AGENT_PREFIX:
            raise ValueError(
                "Native DuplexIO requires its fixed initial agent-stream prefix"
            )
        if self.initial_user_prefix != INITIAL_USER_PREFIX:
            raise ValueError(
                "Native DuplexIO requires its fixed initial user-stream prefix"
            )
        if not math.isclose(
            self.frame_size * self.frame_rate,
            self.sample_rate,
        ):
            raise ValueError(
                "DuplexIO frame geometry is inconsistent: "
                f"{self.frame_size} * {self.frame_rate} != {self.sample_rate}"
            )
        if self.audio_adapter_config.get("architecture") != "mlp":
            raise ValueError("Native DuplexIO requires MLP audio adapters")
        adapter_hidden_size = self.audio_adapter_config.get("hidden_size")
        if adapter_hidden_size is not None and (
            not isinstance(adapter_hidden_size, int) or adapter_hidden_size < 1
        ):
            raise ValueError("DuplexIO adapter hidden_size must be positive")
        skip_dropout = self.audio_adapter_config.get(
            "agent_audio_skip_dropout",
            0.0,
        )
        if not isinstance(skip_dropout, (int, float)) or not 0 <= skip_dropout < 1:
            raise ValueError(
                "DuplexIO agent_audio_skip_dropout must be in [0, 1)"
            )
        if self.audio_codec_config.get("model_type") != "mimi":
            raise ValueError("Native DuplexIO requires the original Mimi codec")
        if (
            self.audio_codec_config.get("sampling_rate") != self.sample_rate
            or self.audio_codec_config.get("frame_rate") != self.frame_rate
        ):
            raise ValueError("DuplexIO and Mimi frame geometry differ")
        num_codebooks = self.quantized_audio_config.get("num_codebooks")
        if num_codebooks != 8:
            raise ValueError("Native DuplexIO currently requires eight Mimi codebooks")
        codec_quantizers = self.audio_codec_config.get("num_quantizers")
        if not isinstance(codec_quantizers, int) or codec_quantizers < num_codebooks:
            raise ValueError(
                "Native DuplexIO Mimi checkpoint does not contain eight codebooks"
            )
        codec_size = self.audio_codec_config.get("codebook_size")
        representation_size = self.quantized_audio_config.get("codebook_size")
        if codec_size != representation_size:
            raise ValueError(
                "DuplexIO Mimi and audio-representation codebook sizes differ"
            )
        representation_dim = self.quantized_audio_config.get("embedding_dim")
        if not isinstance(representation_dim, int) or representation_dim < 1:
            raise ValueError("DuplexIO audio embedding_dim must be positive")
        if self.quantized_audio_config.get("acoustic_delay_frames") != 1:
            raise ValueError("Native DuplexIO requires one acoustic delay frame")
        if (
            self.depth_transformer_config.get("implementation")
            != "duplexio_speaker_adaptive_depth_v1"
        ):
            raise ValueError(
                "Native DuplexIO requires the speaker-conditioned depth checkpoint"
            )
        depth_dim = self.depth_transformer_config.get("dim")
        depth_layers = self.depth_transformer_config.get("num_layers")
        depth_heads = self.depth_transformer_config.get("num_heads")
        depth_mlp_dim = self.depth_transformer_config.get("mlp_dim")
        low_rank = self.depth_transformer_config.get("low_rank_embeddings")
        if not all(
            isinstance(value, int) and value > 0
            for value in (
                depth_dim,
                depth_layers,
                depth_heads,
                depth_mlp_dim,
            )
        ):
            raise ValueError("DuplexIO depformer dimensions must be positive integers")
        if low_rank is not None and (
            not isinstance(low_rank, int) or low_rank < 1
        ):
            raise ValueError(
                "DuplexIO low_rank_embeddings must be positive or null"
            )
        if depth_dim % depth_heads != 0:
            raise ValueError("DuplexIO depformer dim must be divisible by num_heads")
        temperature = self.depth_transformer_config.get("sampling_temperature")
        top_k = self.depth_transformer_config.get("sampling_top_k")
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError("DuplexIO depth sampling_temperature must be positive")
        if not isinstance(top_k, int) or not 1 <= top_k <= representation_size:
            raise ValueError("DuplexIO depth sampling_top_k is outside the codebook")
        semantic_top_k = self.depth_transformer_config.get(
            "semantic_sampling_top_k"
        )
        if semantic_top_k is not None and (
            not isinstance(semantic_top_k, int)
            or not 1 <= semantic_top_k <= representation_size
        ):
            raise ValueError(
                "DuplexIO semantic_sampling_top_k is outside the codebook"
            )
        codebook_loss_weights = self.depth_transformer_config.get(
            "codebook_loss_weights"
        )
        if (
            not isinstance(codebook_loss_weights, list)
            or len(codebook_loss_weights) != num_codebooks
            or any(
                not isinstance(weight, (int, float)) or weight <= 0
                for weight in codebook_loss_weights
            )
        ):
            raise ValueError(
                "DuplexIO depth checkpoint requires one positive loss weight "
                "per codebook"
            )
        if self.audio_attention_window_frames < 1:
            raise ValueError("DuplexIO audio attention window must be positive")
        if self.speaker_embed_dim < 1:
            raise ValueError("DuplexIO speaker_embed_dim must be positive")

        text_config = self.text_config
        if text_config.model_type != "qwen3_5_text":
            raise ValueError("Native DuplexIO requires the dense qwen3_5_text backbone")
        layer_types = getattr(text_config, "layer_types", None)
        if (
            not isinstance(layer_types, list)
            or len(layer_types) != text_config.num_hidden_layers
            or any(
                layer_type not in {"full_attention", "linear_attention"}
                for layer_type in layer_types
            )
        ):
            raise ValueError(
                "DuplexIO text_config must define one supported layer type per layer"
            )
        if not {"full_attention", "linear_attention"}.issubset(layer_types):
            raise ValueError(
                "Native DuplexIO requires Qwen3.5 full-attention and "
                "linear-attention layers"
            )


def _text_config(
    value: dict[str, Any] | PretrainedConfig | None,
) -> PretrainedConfig:
    if isinstance(value, PretrainedConfig):
        return value
    config = dict(value or {"model_type": "qwen3_5_text"})
    model_type = config.pop("model_type", None)
    if not isinstance(model_type, str):
        raise ValueError("DuplexIO text_config must contain a model_type")
    return AutoConfig.for_model(model_type, **config)
