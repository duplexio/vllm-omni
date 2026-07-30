# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Kyutai and the Hugging Face Inc. team.
# Copyright contributors to the vLLM project.
"""Native, inference-only Mimi codec for DuplexIO streaming sessions.

The module layout intentionally matches ``transformers.MimiModel`` so an
exported DuplexIO checkpoint loads without a conversion step.  The runtime
itself only depends on PyTorch.  Streaming state belongs to a session and is
passed explicitly; model modules never retain request state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class MimiConfig:
    """Validated subset of the original Mimi configuration used by DuplexIO."""

    sampling_rate: int
    audio_channels: int
    hidden_size: int
    num_filters: int
    num_residual_layers: int
    upsampling_ratios: tuple[int, ...]
    kernel_size: int
    last_kernel_size: int
    residual_kernel_size: int
    dilation_growth_rate: int
    use_causal_conv: bool
    pad_mode: str
    compress: int
    trim_right_ratio: float
    codebook_size: int
    codebook_dim: int
    num_quantizers: int
    use_conv_shortcut: bool
    vector_quantization_hidden_dimension: int
    num_semantic_quantizers: int
    upsample_groups: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_act: str
    max_position_embeddings: int
    norm_eps: float
    sliding_window: int
    attention_dropout: float
    layer_scale_initial_scale: float
    attention_bias: bool
    rope_theta: float
    frame_rate: float

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MimiConfig:
        """Parse the external checkpoint configuration at the serving boundary."""
        ratios = tuple(value.get("upsampling_ratios") or (8, 6, 5, 4))
        sampling_rate = value.get("sampling_rate", 24_000)
        num_residual_layers = value.get("num_residual_layers", 1)
        frame_size = math.prod(ratios) * 2
        frame_rate = value.get("frame_rate", value.get("_frame_rate"))
        if frame_rate is None:
            frame_rate = sampling_rate / frame_size
        hidden_size = value.get("hidden_size", 512)
        num_attention_heads = value.get("num_attention_heads", 8)
        rope = value.get("rope_parameters") or {
            "rope_type": "default",
            "rope_theta": 10_000.0,
        }
        if not isinstance(rope, Mapping):
            raise ValueError("Mimi rope_parameters must be a mapping")
        rope_type = rope.get("rope_type", "default")
        rope_theta = rope.get("rope_theta", 10_000.0)
        if not isinstance(rope_type, str):
            raise ValueError("Mimi rope_type must be a string")
        if not isinstance(rope_theta, int | float):
            raise ValueError("Mimi rope_theta must be numeric")
        config = cls(
            sampling_rate=sampling_rate,
            audio_channels=value.get("audio_channels", 1),
            hidden_size=hidden_size,
            num_filters=value.get("num_filters", 64),
            num_residual_layers=num_residual_layers,
            upsampling_ratios=ratios,
            kernel_size=value.get("kernel_size", 7),
            last_kernel_size=value.get("last_kernel_size", 3),
            residual_kernel_size=value.get("residual_kernel_size", 3),
            dilation_growth_rate=value.get("dilation_growth_rate", 2),
            use_causal_conv=value.get("use_causal_conv", True),
            pad_mode=value.get("pad_mode", "constant"),
            compress=value.get("compress", 2),
            trim_right_ratio=value.get("trim_right_ratio", 1.0),
            codebook_size=value.get("codebook_size", 2_048),
            codebook_dim=value.get("codebook_dim", 256),
            num_quantizers=value.get("num_quantizers", 32),
            use_conv_shortcut=value.get("use_conv_shortcut", False),
            vector_quantization_hidden_dimension=value.get(
                "vector_quantization_hidden_dimension",
                256,
            ),
            num_semantic_quantizers=value.get("num_semantic_quantizers", 1),
            upsample_groups=value.get("upsample_groups", 512),
            num_hidden_layers=value.get("num_hidden_layers", 8),
            intermediate_size=value.get("intermediate_size", 2_048),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=value.get("num_key_value_heads", 8),
            head_dim=value.get("head_dim")
            or hidden_size // num_attention_heads,
            hidden_act=value.get("hidden_act", "gelu"),
            max_position_embeddings=value.get("max_position_embeddings", 8_000),
            norm_eps=value.get("norm_eps", 1e-5),
            sliding_window=value.get("sliding_window", 250),
            attention_dropout=value.get("attention_dropout", 0.0),
            layer_scale_initial_scale=value.get(
                "layer_scale_initial_scale",
                0.01,
            ),
            attention_bias=value.get("attention_bias", False),
            rope_theta=rope_theta,
            frame_rate=frame_rate,
        )
        config._validate(rope_type)
        return config

    @property
    def encodec_frame_rate(self) -> int:
        return math.ceil(self.sampling_rate / math.prod(self.upsampling_ratios))

    @property
    def frame_size(self) -> int:
        return math.prod(self.upsampling_ratios) * 2

    def _validate(self, rope_type: str) -> None:
        if not self.use_causal_conv:
            raise ValueError("Native DuplexIO Mimi requires causal convolutions")
        if self.pad_mode not in ("constant", "replicate"):
            raise ValueError(f"Unsupported Mimi padding mode: {self.pad_mode}")
        if self.trim_right_ratio != 1.0:
            raise ValueError("Native DuplexIO Mimi requires trim_right_ratio=1")
        if self.hidden_act != "gelu":
            raise ValueError(f"Unsupported Mimi activation: {self.hidden_act}")
        if rope_type != "default":
            raise ValueError(f"Unsupported Mimi RoPE type: {rope_type}")
        if self.residual_kernel_size < 1:
            raise ValueError("Mimi residual_kernel_size must be positive")
        if self.num_semantic_quantizers >= self.num_quantizers:
            raise ValueError("Mimi semantic quantizers must be fewer than all quantizers")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("Mimi hidden_size must equal num_heads * head_dim")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("Mimi attention heads must divide evenly into KV heads")
        if self.hidden_size % self.upsample_groups:
            raise ValueError("Mimi hidden_size must divide evenly into upsample_groups")
        if not math.isclose(
            self.frame_rate,
            self.sampling_rate / self.frame_size,
        ):
            raise ValueError("Mimi frame rate does not match its convolution strides")
        if self.encodec_frame_rate / self.frame_rate != 2:
            raise ValueError("Native DuplexIO Mimi requires the original 2x resampler")


@dataclass
class MimiConv1dState:
    """Causal prefix retained by one convolution for one session."""

    previous: Tensor

    def fork(self) -> MimiConv1dState:
        return MimiConv1dState(self.previous)


@dataclass
class MimiConvTranspose1dState:
    """Overlap retained by one transposed convolution for one session."""

    partial: Tensor

    def fork(self) -> MimiConvTranspose1dState:
        return MimiConvTranspose1dState(self.partial)


MimiConvolutionState = MimiConv1dState | MimiConvTranspose1dState


@dataclass
class MimiTransformerState:
    """Per-layer causal KV state and absolute position for one transformer."""

    keys: list[Tensor | None]
    values: list[Tensor | None]
    position: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> MimiTransformerState:
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    def fork(self) -> MimiTransformerState:
        return MimiTransformerState(
            keys=list(self.keys),
            values=list(self.values),
            position=self.position,
        )


@dataclass
class MimiStreamingState:
    """All state required to encode and decode one live Mimi session."""

    encoder_convs: dict[int, MimiConvolutionState] = field(default_factory=dict)
    decoder_convs: dict[int, MimiConvolutionState] = field(default_factory=dict)
    encoder_transformer: MimiTransformerState | None = None
    decoder_transformer: MimiTransformerState | None = None

    def fork(self) -> MimiStreamingState:
        """Copy containers while sharing immutable tensor snapshots."""
        return MimiStreamingState(
            encoder_convs={
                index: state.fork()
                for index, state in self.encoder_convs.items()
            },
            decoder_convs={
                index: state.fork()
                for index, state in self.decoder_convs.items()
            },
            encoder_transformer=(
                None
                if self.encoder_transformer is None
                else self.encoder_transformer.fork()
            ),
            decoder_transformer=(
                None
                if self.decoder_transformer is None
                else self.decoder_transformer.fork()
            ),
        )


class MimiConv1d(nn.Module):
    """Mimi causal convolution with optional explicit streaming state."""

    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        pad_mode: str | None = None,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.pad_mode = config.pad_mode if pad_mode is None else pad_mode
        self.layer_idx = layer_idx
        self.in_channels = in_channels
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = stride
        self.kernel_size = (kernel_size - 1) * dilation + 1
        self.padding_total = self.kernel_size - stride

    def forward(
        self,
        hidden_states: Tensor,
        streaming_state: dict[int, MimiConvolutionState] | None = None,
    ) -> Tensor:
        if streaming_state is None:
            extra_padding = (-hidden_states.shape[-1]) % self.stride
            hidden_states = F.pad(
                hidden_states,
                (self.padding_total, extra_padding),
                mode=self.pad_mode,
            )
            return self.conv(hidden_states)

        assert self.layer_idx is not None
        state = streaming_state.get(self.layer_idx)
        if state is None:
            if self.pad_mode == "replicate":
                previous = hidden_states[..., :1].expand(
                    -1,
                    -1,
                    self.padding_total,
                ).clone()
            else:
                previous = hidden_states.new_zeros(
                    hidden_states.shape[0],
                    self.in_channels,
                    self.padding_total,
                )
            state = MimiConv1dState(previous)
            streaming_state[self.layer_idx] = state
        assert isinstance(state, MimiConv1dState)
        assert hidden_states.shape[-1] > 0
        assert hidden_states.shape[-1] % self.stride == 0
        if self.padding_total:
            hidden_states = torch.cat((state.previous, hidden_states), dim=-1)
            state.previous = hidden_states[..., -self.padding_total :].detach().clone()
        return self.conv(hidden_states)


class MimiConvTranspose1d(nn.Module):
    """Mimi causal transposed convolution with streaming overlap-add."""

    def __init__(
        self,
        config: MimiConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            groups=groups,
            bias=bias,
        )
        self.padding_right = kernel_size - stride

    def forward(
        self,
        hidden_states: Tensor,
        streaming_state: dict[int, MimiConvolutionState] | None = None,
    ) -> Tensor:
        hidden_states = self.conv(hidden_states)
        if streaming_state is None:
            if self.padding_right:
                hidden_states = hidden_states[..., : -self.padding_right]
            return hidden_states

        assert self.layer_idx is not None
        state = streaming_state.get(self.layer_idx)
        if state is None:
            state = MimiConvTranspose1dState(
                hidden_states.new_zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    self.padding_right,
                )
            )
            streaming_state[self.layer_idx] = state
        assert isinstance(state, MimiConvTranspose1dState)
        if self.padding_right:
            hidden_states[..., : self.padding_right] += state.partial
            partial = hidden_states[..., -self.padding_right :]
            if self.conv.bias is not None:
                partial = partial - self.conv.bias[:, None]
            state.partial = partial.detach().clone()
            hidden_states = hidden_states[..., : -self.padding_right]
        return hidden_states


class MimiResnetBlock(nn.Module):
    """SEANet residual block used by Mimi."""

    def __init__(self, config: MimiConfig, dim: int, dilations: tuple[int, int]):
        super().__init__()
        hidden = dim // config.compress
        self.block = nn.ModuleList(
            [
                nn.ELU(),
                MimiConv1d(
                    config,
                    dim,
                    hidden,
                    config.residual_kernel_size,
                    dilation=dilations[0],
                ),
                nn.ELU(),
                MimiConv1d(config, hidden, dim, 1, dilation=dilations[1]),
            ]
        )
        self.shortcut: nn.Module
        if config.use_conv_shortcut:
            self.shortcut = MimiConv1d(config, dim, dim, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        hidden_states: Tensor,
        streaming_state: dict[int, MimiConvolutionState] | None = None,
    ) -> Tensor:
        residual = hidden_states
        for layer in self.block:
            if isinstance(layer, MimiConv1d):
                hidden_states = layer(hidden_states, streaming_state)
            else:
                hidden_states = layer(hidden_states)
        if isinstance(self.shortcut, MimiConv1d):
            residual = self.shortcut(residual, streaming_state)
        return residual + hidden_states


def _assign_convolution_indices(module: nn.Module, start: int = 0) -> int:
    index = start
    for child in module.modules():
        if isinstance(child, (MimiConv1d, MimiConvTranspose1d)):
            child.layer_idx = index
            index += 1
    return index


class MimiEncoder(nn.Module):
    """Native SEANet encoder with checkpoint-compatible module names."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.audio_channels,
                config.num_filters,
                config.kernel_size,
            )
        ]
        scaling = 1
        for ratio in reversed(config.upsampling_ratios):
            current_scale = scaling * config.num_filters
            for layer_index in range(config.num_residual_layers):
                model.append(
                    MimiResnetBlock(
                        config,
                        current_scale,
                        (config.dilation_growth_rate**layer_index, 1),
                    )
                )
            model.extend(
                [
                    nn.ELU(),
                    MimiConv1d(
                        config,
                        current_scale,
                        current_scale * 2,
                        ratio * 2,
                        stride=ratio,
                    ),
                ]
            )
            scaling *= 2
        model.extend(
            [
                nn.ELU(),
                MimiConv1d(
                    config,
                    scaling * config.num_filters,
                    config.hidden_size,
                    config.last_kernel_size,
                ),
            ]
        )
        self.layers = nn.ModuleList(model)
        self.num_streaming_convolutions = _assign_convolution_indices(self)

    def forward(
        self,
        hidden_states: Tensor,
        streaming_state: dict[int, MimiConvolutionState] | None = None,
    ) -> Tensor:
        for layer in self.layers:
            if isinstance(layer, (MimiConv1d, MimiResnetBlock)):
                hidden_states = layer(hidden_states, streaming_state)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class MimiLayerScale(nn.Module):
    """Learned diagonal residual scale."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        self.scale = nn.Parameter(
            torch.full((config.hidden_size,), config.layer_scale_initial_scale)
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.scale * hidden_states


def _rotate_half(hidden_states: Tensor) -> Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class MimiRotaryEmbedding(nn.Module):
    """Original split-half Mimi rotary embedding."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        frequencies = torch.einsum(
            "i,bt->bti",
            self.inv_freq.float(),
            position_ids.float(),
        )
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return (
            embedding.cos().to(hidden_states.dtype),
            embedding.sin().to(hidden_states.dtype),
        )


class MimiMLP(nn.Module):
    """Mimi transformer feed-forward block."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(hidden_states)))


class MimiAttention(nn.Module):
    """Causal sliding-window attention used by Mimi's two transformers."""

    def __init__(self, config: MimiConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = config.sliding_window
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        state: MimiTransformerState | None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)

        cos, sin = position_embeddings
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        query = query * cos + _rotate_half(query) * sin
        key = key * cos + _rotate_half(key) * sin

        key_start = 0
        if state is not None:
            past_key = state.keys[self.layer_idx]
            past_value = state.values[self.layer_idx]
            if past_key is not None:
                assert past_value is not None
                key = torch.cat((past_key, key), dim=2)
                value = torch.cat((past_value, value), dim=2)
            key_start = state.position - (key.shape[2] - sequence_length)

        repeated_key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        repeated_value = value.repeat_interleave(
            self.num_key_value_groups,
            dim=1,
        )
        key_positions = torch.arange(
            key_start,
            key_start + key.shape[2],
            device=hidden_states.device,
        )
        visible = (
            (key_positions[None, :] <= position_ids[0, :, None])
            & (
                key_positions[None, :]
                > position_ids[0, :, None] - self.sliding_window
            )
        )
        attention = torch.matmul(query, repeated_key.transpose(2, 3))
        attention = attention * self.scaling
        attention = attention.masked_fill(
            ~visible[None, None, :, :],
            torch.finfo(attention.dtype).min,
        )
        attention = F.softmax(attention, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        output = torch.matmul(attention, repeated_value)
        output = output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            -1,
        )

        if state is not None:
            state.keys[self.layer_idx] = (
                key[:, :, -self.sliding_window :].detach().clone()
            )
            state.values[self.layer_idx] = value[
                :, :, -self.sliding_window :
            ].detach().clone()
        return self.o_proj(output)


class MimiTransformerLayer(nn.Module):
    """One original Mimi pre-normalized transformer layer."""

    def __init__(self, config: MimiConfig, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = MimiAttention(config, layer_idx)
        self.mlp = MimiMLP(config)
        self.input_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.post_attention_layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.norm_eps,
        )
        self.self_attn_layer_scale = MimiLayerScale(config)
        self.mlp_layer_scale = MimiLayerScale(config)

    def forward(
        self,
        hidden_states: Tensor,
        position_embeddings: tuple[Tensor, Tensor],
        position_ids: Tensor,
        state: MimiTransformerState | None,
    ) -> Tensor:
        attention = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            position_ids,
            state,
        )
        hidden_states = hidden_states + self.self_attn_layer_scale(attention)
        feed_forward = self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states + self.mlp_layer_scale(feed_forward)


class MimiTransformerModel(nn.Module):
    """Mimi transformer with explicit per-session KV state."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MimiTransformerLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.rotary_emb = MimiRotaryEmbedding(config)

    def forward(
        self,
        hidden_states: Tensor,
        state: MimiTransformerState | None = None,
    ) -> Tensor:
        position = 0 if state is None else state.position
        position_ids = torch.arange(
            position,
            position + hidden_states.shape[1],
            device=hidden_states.device,
        ).unsqueeze(0)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                position_ids,
                state,
            )
        if state is not None:
            state.position += hidden_states.shape[1]
        return hidden_states


class MimiDecoder(nn.Module):
    """Native SEANet decoder with streaming overlap-add."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        scaling = 2 ** len(config.upsampling_ratios)
        model: list[nn.Module] = [
            MimiConv1d(
                config,
                config.hidden_size,
                scaling * config.num_filters,
                config.kernel_size,
            )
        ]
        for ratio in config.upsampling_ratios:
            current_scale = scaling * config.num_filters
            model.extend(
                [
                    nn.ELU(),
                    MimiConvTranspose1d(
                        config,
                        current_scale,
                        current_scale // 2,
                        ratio * 2,
                        stride=ratio,
                    ),
                ]
            )
            for layer_index in range(config.num_residual_layers):
                model.append(
                    MimiResnetBlock(
                        config,
                        current_scale // 2,
                        (config.dilation_growth_rate**layer_index, 1),
                    )
                )
            scaling //= 2
        model.extend(
            [
                nn.ELU(),
                MimiConv1d(
                    config,
                    config.num_filters,
                    config.audio_channels,
                    config.last_kernel_size,
                ),
            ]
        )
        self.layers = nn.ModuleList(model)
        self.num_streaming_convolutions = _assign_convolution_indices(self)

    def forward(
        self,
        hidden_states: Tensor,
        streaming_state: dict[int, MimiConvolutionState] | None = None,
    ) -> Tensor:
        for layer in self.layers:
            if isinstance(
                layer,
                (MimiConv1d, MimiConvTranspose1d, MimiResnetBlock),
            ):
                hidden_states = layer(hidden_states, streaming_state)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


class MimiEuclideanCodebook(nn.Module):
    """Original normalized Euclidean Mimi codebook."""

    initialized: Tensor
    cluster_usage: Tensor
    embed_sum: Tensor

    def __init__(self, config: MimiConfig, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.register_buffer("initialized", torch.tensor([True], dtype=torch.float32))
        self.register_buffer("cluster_usage", torch.ones(config.codebook_size))
        self.register_buffer(
            "embed_sum",
            torch.zeros(config.codebook_size, config.codebook_dim),
        )
        self.epsilon = epsilon

    @property
    def embed(self) -> Tensor:
        return self.embed_sum / self.cluster_usage.clamp(min=self.epsilon)[:, None]

    def encode(self, hidden_states: Tensor) -> Tensor:
        shape = hidden_states.shape
        flattened = hidden_states.reshape(-1, shape[-1])
        distances = torch.cdist(
            flattened[None].float(),
            self.embed[None].float(),
            p=2,
        )[0]
        return distances.argmin(dim=-1).view(*shape[:-1])

    def decode(self, indices: Tensor) -> Tensor:
        return F.embedding(indices, self.embed)


class MimiVectorQuantization(nn.Module):
    """One Mimi vector-quantization level."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        self.codebook = MimiEuclideanCodebook(config)

    def encode(self, hidden_states: Tensor) -> Tensor:
        return self.codebook.encode(hidden_states.permute(0, 2, 1))

    def decode(self, indices: Tensor) -> Tensor:
        return self.codebook.decode(indices).permute(0, 2, 1)


class MimiResidualVectorQuantizer(nn.Module):
    """Residual stack of Mimi codebooks."""

    def __init__(
        self,
        config: MimiConfig,
        num_quantizers: int,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.layers = nn.ModuleList(
            [MimiVectorQuantization(config) for _ in range(num_quantizers)]
        )
        self.input_proj: nn.Conv1d | None = None
        self.output_proj: nn.Conv1d | None = None
        if config.vector_quantization_hidden_dimension != config.hidden_size:
            self.input_proj = nn.Conv1d(
                config.hidden_size,
                config.vector_quantization_hidden_dimension,
                1,
                bias=False,
            )
            self.output_proj = nn.Conv1d(
                config.vector_quantization_hidden_dimension,
                config.hidden_size,
                1,
                bias=False,
            )

    def encode(self, embeddings: Tensor, num_quantizers: int) -> Tensor:
        if self.input_proj is not None:
            embeddings = self.input_proj(embeddings)
        residual = embeddings
        indices = []
        for layer in self.layers[:num_quantizers]:
            assert isinstance(layer, MimiVectorQuantization)
            level_indices = layer.encode(residual)
            residual = residual - layer.decode(level_indices)
            indices.append(level_indices)
        return torch.stack(indices)

    def decode(self, codes: Tensor) -> Tensor:
        codes_by_level = codes.transpose(0, 1)
        first_layer = self.layers[0]
        assert isinstance(first_layer, MimiVectorQuantization)
        quantized = first_layer.decode(codes_by_level[0])
        for level, indices in enumerate(codes_by_level[1:], start=1):
            layer = self.layers[level]
            assert isinstance(layer, MimiVectorQuantization)
            quantized = quantized + layer.decode(indices)
        if self.output_proj is not None:
            quantized = self.output_proj(quantized)
        return quantized


class MimiSplitResidualVectorQuantizer(nn.Module):
    """Separate semantic and acoustic Mimi residual quantizers."""

    def __init__(self, config: MimiConfig) -> None:
        super().__init__()
        self.num_semantic_quantizers = config.num_semantic_quantizers
        self.max_num_quantizers = config.num_quantizers
        self.semantic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config,
            config.num_semantic_quantizers,
        )
        self.acoustic_residual_vector_quantizer = MimiResidualVectorQuantizer(
            config,
            config.num_quantizers - config.num_semantic_quantizers,
        )

    def encode(self, embeddings: Tensor, num_quantizers: int) -> Tensor:
        semantic = self.semantic_residual_vector_quantizer.encode(
            embeddings,
            self.num_semantic_quantizers,
        )
        if num_quantizers == self.num_semantic_quantizers:
            return semantic
        acoustic = self.acoustic_residual_vector_quantizer.encode(
            embeddings,
            num_quantizers - self.num_semantic_quantizers,
        )
        return torch.cat((semantic, acoustic), dim=0)

    def decode(self, codes: Tensor) -> Tensor:
        quantized = self.semantic_residual_vector_quantizer.decode(
            codes[:, : self.num_semantic_quantizers]
        )
        if codes.shape[1] > self.num_semantic_quantizers:
            quantized = quantized + self.acoustic_residual_vector_quantizer.decode(
                codes[:, self.num_semantic_quantizers :]
            )
        return quantized


class MimiModel(nn.Module):
    """Original quantized Mimi architecture with native streaming execution."""

    def __init__(self, config: MimiConfig | Mapping[str, Any]) -> None:
        super().__init__()
        if not isinstance(config, MimiConfig):
            config = MimiConfig.from_dict(config)
        self.config = config
        self.encoder = MimiEncoder(config)
        self.encoder_transformer = MimiTransformerModel(config)
        self.downsample = MimiConv1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=4,
            stride=2,
            bias=False,
            pad_mode="replicate",
            layer_idx=self.encoder.num_streaming_convolutions,
        )
        self.upsample = MimiConvTranspose1d(
            config,
            config.hidden_size,
            config.hidden_size,
            kernel_size=4,
            stride=2,
            bias=False,
            groups=config.upsample_groups,
            layer_idx=0,
        )
        self.decoder_transformer = MimiTransformerModel(config)
        self.decoder = MimiDecoder(config)
        _assign_convolution_indices(self.decoder, start=1)
        self.quantizer = MimiSplitResidualVectorQuantizer(config)

    def new_streaming_state(self) -> MimiStreamingState:
        """Create independent state for one full-duplex request."""
        return MimiStreamingState(
            encoder_transformer=MimiTransformerState.empty(
                len(self.encoder_transformer.layers)
            ),
            decoder_transformer=MimiTransformerState.empty(
                len(self.decoder_transformer.layers)
            ),
        )

    def encode_latent(
        self,
        input_values: Tensor,
        state: MimiStreamingState | None = None,
    ) -> Tensor:
        """Encode waveform ``(B, C, samples)`` to pre-quantization latents."""
        encoder_convs = None if state is None else state.encoder_convs
        transformer_state = None if state is None else state.encoder_transformer
        hidden_states = self.encoder(input_values, encoder_convs)
        hidden_states = self.encoder_transformer(
            hidden_states.transpose(1, 2),
            transformer_state,
        ).transpose(1, 2)
        return self.downsample(hidden_states, encoder_convs)

    def encode(
        self,
        input_values: Tensor,
        num_quantizers: int,
        state: MimiStreamingState | None = None,
    ) -> Tensor:
        """Encode waveform to codes shaped ``(B, codebooks, frames)``."""
        if not self.config.num_semantic_quantizers <= num_quantizers <= (
            self.config.num_quantizers
        ):
            raise ValueError(
                f"Mimi num_quantizers must be in "
                f"[{self.config.num_semantic_quantizers}, "
                f"{self.config.num_quantizers}], got {num_quantizers}"
            )
        latent = self.encode_latent(input_values, state)
        return self.quantizer.encode(latent, num_quantizers).transpose(0, 1)

    def decode(
        self,
        audio_codes: Tensor,
        state: MimiStreamingState | None = None,
    ) -> Tensor:
        """Decode codes ``(B, codebooks, frames)`` to a waveform chunk."""
        decoder_convs = None if state is None else state.decoder_convs
        transformer_state = None if state is None else state.decoder_transformer
        hidden_states = self.quantizer.decode(audio_codes)
        hidden_states = self.upsample(hidden_states, decoder_convs)
        hidden_states = self.decoder_transformer(
            hidden_states.transpose(1, 2),
            transformer_state,
        ).transpose(1, 2)
        return self.decoder(hidden_states, decoder_convs)


__all__ = [
    "MimiConfig",
    "MimiModel",
    "MimiStreamingState",
]
