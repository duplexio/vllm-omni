# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Configuration of a DuplexIO checkpoint."""

from __future__ import annotations

from typing import Any

from transformers import AutoConfig, PretrainedConfig


class DuplexIOConfig(PretrainedConfig):
    """A Qwen3.5 text backbone with a streaming user ASR encoder and a FlowMap agent-audio head."""

    model_type = "duplexio"
    is_composition = True
    has_no_defaults_at_init = True

    def __init__(
        self,
        *,
        text_config: dict[str, Any] | PretrainedConfig,
        user_asr_config: dict[str, Any],
        flowmap_config: dict[str, Any],
        silence_token_id: int,
        audio_adapter_config: dict[str, Any] | None = None,
        audio_attention_window_frames: int = 4_096,
        voice_prompt_max_frames: int = 125,
        default_system_prompt: str = "",
        head_dtype: str = "float32",
        **kwargs: Any,
    ) -> None:
        self.text_config = _text_config(text_config)
        super().__init__(head_dtype=head_dtype, **kwargs)
        self.user_asr_config = dict(user_asr_config)
        self.flowmap_config = dict(flowmap_config)
        self.audio_adapter_config = dict(audio_adapter_config or {})
        self.audio_attention_window_frames = audio_attention_window_frames
        self.voice_prompt_max_frames = voice_prompt_max_frames
        self.silence_token_id = silence_token_id
        self.default_system_prompt = default_system_prompt
        self.validate()

    def get_text_config(self, decoder: bool | None = None, encoder: bool | None = None) -> PretrainedConfig:
        """Return Qwen's config for vLLM cache and parallelism planning."""
        return self.text_config

    def validate(self) -> None:
        if self.pad_token_id is None:
            raise ValueError("DuplexIO requires a pad_token_id")
        if self.user_asr_config.get("model_type") != "nemotron_asr_streaming":
            raise ValueError("DuplexIO requires a streaming Nemotron RNN-T user ASR encoder")
        for name in ("mlp_dim", "mlp_depth", "inference_steps"):
            value = self.flowmap_config.get(name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"FlowMap {name} must be a positive integer")
        temperature = self.flowmap_config.get("sampling_temperature")
        if not isinstance(temperature, int | float) or temperature < 0:
            raise ValueError("FlowMap sampling_temperature must be nonnegative")
        adapter_hidden_size = self.audio_adapter_config.get("hidden_size")
        if adapter_hidden_size is not None and (not isinstance(adapter_hidden_size, int) or adapter_hidden_size < 1):
            raise ValueError("DuplexIO adapter hidden_size must be positive")
        if self.audio_attention_window_frames < 1:
            raise ValueError("DuplexIO audio attention window must be positive")
        if self.voice_prompt_max_frames < 1:
            raise ValueError("DuplexIO needs room for at least one voice-prompt frame")
        text_config = self.text_config
        if text_config.model_type != "qwen3_5_text":
            raise ValueError("DuplexIO requires a dense qwen3_5_text backbone")
        layer_types = text_config.layer_types
        if len(layer_types) != text_config.num_hidden_layers or set(layer_types) != {
            "full_attention",
            "linear_attention",
        }:
            raise ValueError("DuplexIO requires one full- or linear-attention type per layer, using both")


def _text_config(value: dict[str, Any] | PretrainedConfig) -> PretrainedConfig:
    if not isinstance(value, PretrainedConfig):
        config = dict(value)
        value = AutoConfig.for_model(config.pop("model_type"), **config)
    # DuplexIO positions are 1-D frame indices. Qwen3.5's M-RoPE keys would make
    # vLLM feed (3, num_tokens) positions; M-RoPE over identical position streams
    # is standard RoPE, so dropping them changes nothing.
    rope_parameters = value.rope_parameters
    if rope_parameters:
        rope_parameters.pop("mrope_section", None)
        rope_parameters.pop("mrope_interleaved", None)
    return value
