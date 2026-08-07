# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Native bounded streaming path for DuplexIO's frozen FastConformer encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class FastConformerConfig:
    source_sample_rate: int = 24_000
    sample_rate: int = 16_000
    frame_size: int = 1_920
    features: int = 80
    n_fft: int = 512
    window_size: int = 400
    window_stride: int = 160
    subsampling_factor: int = 8
    subsampling_conv_channels: int = 256
    num_layers: int = 17
    dim: int = 512
    feedforward_dim: int = 2_048
    num_heads: int = 8
    attention_left_context: int = 70
    convolution_kernel_size: int = 9

    @property
    def native_frame_size(self) -> int:
        return self.frame_size * self.sample_rate // self.source_sample_rate

    @property
    def feature_chunk_size(self) -> int:
        return self.native_frame_size // self.window_stride

    @property
    def feature_buffer_size(self) -> int:
        return 2 * self.subsampling_factor

    @property
    def sample_history_size(self) -> int:
        required = self.window_stride + self.window_size // 2 + 1
        return math.ceil(required / self.window_stride) * self.window_stride


@dataclass(frozen=True)
class FastConformerStreamingState:
    """All request state has architecture-defined, session-invariant bounds."""

    sample_buffer: Tensor
    feature_buffer: Tensor
    attention_caches: tuple[Tensor, ...]
    convolution_caches: tuple[Tensor, ...]
    frames_seen: int = 0


class FilterbankFeatures(nn.Module):
    """The two persistent buffers in NeMo's frozen mel preprocessor."""

    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.register_buffer("window", torch.empty(config.window_size))
        self.register_buffer("fb", torch.empty(1, config.features, config.n_fft // 2 + 1))


class AudioToMelSpectrogramPreprocessor(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.config = config
        self.featurizer = FilterbankFeatures(config)

    def forward(self, samples: Tensor) -> Tensor:
        """Return NeMo-compatible log-mel features shaped ``(features, time)``."""
        config = self.config
        emphasized = torch.cat(
            (
                samples[:1],
                samples[1:] - 0.97 * samples[:-1],
            )
        )
        spectrum = torch.stft(
            emphasized.unsqueeze(0),
            n_fft=config.n_fft,
            hop_length=config.window_stride,
            win_length=config.window_size,
            center=True,
            window=self.featurizer.window.float(),
            return_complex=True,
            pad_mode="reflect",
        )
        magnitude = torch.sqrt(torch.view_as_real(spectrum).square().sum(-1))
        spectrum = magnitude.square()
        mel = torch.matmul(self.featurizer.fb.float(), spectrum.float())
        return torch.log(mel + 2**-24)[0]

    def streaming_chunk(
        self,
        samples: Tensor,
        frame_offset: int,
        frame_count: int,
    ) -> Tensor:
        """Return a contiguous slice of 10-ms log-mel frames."""
        return self(samples)[:, frame_offset : frame_offset + frame_count]


class CausalConv2d(nn.Conv2d):
    def forward(self, hidden: Tensor) -> Tensor:
        hidden = F.pad(hidden, (2, 1, 2, 1))
        return super().forward(hidden)


class ConvSubsampling(nn.Module):
    """The exact causal ``dw_striding`` pre-encoder used by the checkpoint."""

    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        channels = config.subsampling_conv_channels
        self.conv = nn.Sequential(
            CausalConv2d(1, channels, 3, stride=2),
            nn.ReLU(),
            CausalConv2d(channels, channels, 3, stride=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
            CausalConv2d(channels, channels, 3, stride=2, groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(),
        )
        frequency = config.features
        for _ in range(3):
            frequency = (frequency + 3 - 3) // 2 + 1
        self.out = nn.Linear(channels * frequency, config.dim)

    def forward(self, features: Tensor) -> Tensor:
        hidden = self.conv(features.T.unsqueeze(0).unsqueeze(0))
        hidden = hidden.transpose(1, 2).flatten(2)
        return self.out(hidden)[:, -1]


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear2(F.silu(self.linear1(hidden)))


class RelPositionAttention(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads
        self.left_context = config.attention_left_context
        self.linear_q = nn.Linear(config.dim, config.dim)
        self.linear_k = nn.Linear(config.dim, config.dim)
        self.linear_v = nn.Linear(config.dim, config.dim)
        self.linear_out = nn.Linear(config.dim, config.dim)
        self.linear_pos = nn.Linear(config.dim, config.dim, bias=False)
        self.pos_bias_u = nn.Parameter(torch.empty(config.num_heads, self.head_dim))
        self.pos_bias_v = nn.Parameter(torch.empty(config.num_heads, self.head_dim))

    def forward(self, hidden: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        sequence = torch.cat((cache, hidden.unsqueeze(0)))
        query = self.linear_q(hidden).view(self.num_heads, self.head_dim)
        key = self.linear_k(sequence).view(-1, self.num_heads, self.head_dim)
        value = self.linear_v(sequence).view(-1, self.num_heads, self.head_dim)
        offsets = torch.arange(
            sequence.shape[0] - 1,
            -1,
            -1,
            dtype=torch.float32,
            device=hidden.device,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                self.num_heads * self.head_dim,
                2,
                dtype=torch.float32,
                device=hidden.device,
            )
            * -(math.log(10_000.0) / (self.num_heads * self.head_dim))
        )
        position = hidden.new_zeros(sequence.shape[0], self.num_heads * self.head_dim)
        position[:, 0::2] = torch.sin(offsets * frequencies).to(position.dtype)
        position[:, 1::2] = torch.cos(offsets * frequencies).to(position.dtype)
        position = self.linear_pos(position).view(
            -1,
            self.num_heads,
            self.head_dim,
        )
        content_scores = torch.einsum(
            "hd,thd->ht",
            query + self.pos_bias_u,
            key,
        )
        position_scores = torch.einsum(
            "hd,thd->ht",
            query + self.pos_bias_v,
            position,
        )
        probabilities = F.softmax(
            (content_scores + position_scores) / math.sqrt(self.head_dim),
            dim=-1,
        )
        attended = torch.einsum("ht,thd->hd", probabilities, value)
        output = self.linear_out(attended.flatten())
        return output, sequence[-self.left_context :]


class ConformerConvolution(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        dim = config.dim
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            config.convolution_kernel_size,
            groups=dim,
        )
        self.batch_norm = nn.LayerNorm(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, hidden: Tensor, cache: Tensor) -> tuple[Tensor, Tensor]:
        pointwise = F.glu(
            self.pointwise_conv1(hidden.view(1, -1, 1)),
            dim=1,
        )
        sequence = torch.cat((cache, pointwise[0]), dim=-1)
        convolved = self.depthwise_conv(sequence.unsqueeze(0))
        convolved = self.batch_norm(convolved.transpose(1, 2))
        convolved = F.silu(convolved).transpose(1, 2)
        output = self.pointwise_conv2(convolved)[0, :, 0]
        return output, sequence[:, 1:]


class ConformerLayer(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        dim = config.dim
        self.norm_feed_forward1 = nn.LayerNorm(dim)
        self.feed_forward1 = ConformerFeedForward(dim, config.feedforward_dim)
        self.norm_conv = nn.LayerNorm(dim)
        self.conv = ConformerConvolution(config)
        self.norm_self_att = nn.LayerNorm(dim)
        self.self_attn = RelPositionAttention(config)
        self.norm_feed_forward2 = nn.LayerNorm(dim)
        self.feed_forward2 = ConformerFeedForward(dim, config.feedforward_dim)
        self.norm_out = nn.LayerNorm(dim)

    def forward(
        self,
        hidden: Tensor,
        attention_cache: Tensor,
        convolution_cache: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden = hidden + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(hidden)
        )
        update, attention_cache = self.self_attn(
            self.norm_self_att(hidden),
            attention_cache,
        )
        hidden = hidden + update
        update, convolution_cache = self.conv(
            self.norm_conv(hidden),
            convolution_cache,
        )
        hidden = hidden + update
        hidden = hidden + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden)
        )
        return self.norm_out(hidden), attention_cache, convolution_cache


class ConformerEncoder(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.input_scale = math.sqrt(config.dim)
        self.pre_encode = ConvSubsampling(config)
        self.layers = nn.ModuleList(
            [ConformerLayer(config) for _ in range(config.num_layers)]
        )

    def step(
        self,
        features: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        hidden = self.pre_encode(features)[0] * self.input_scale
        attention_caches = []
        convolution_caches = []
        for layer, attention_cache, convolution_cache in zip(
            self.layers,
            state.attention_caches,
            state.convolution_caches,
            strict=True,
        ):
            hidden, attention_cache, convolution_cache = layer(
                hidden,
                attention_cache,
                convolution_cache,
            )
            attention_caches.append(attention_cache)
            convolution_caches.append(convolution_cache)
        return hidden, FastConformerStreamingState(
            sample_buffer=state.sample_buffer,
            feature_buffer=state.feature_buffer,
            attention_caches=tuple(attention_caches),
            convolution_caches=tuple(convolution_caches),
            frames_seen=state.frames_seen,
        )


class FastConformerUserEncoder(nn.Module):
    """One frozen 512-dim streaming feature per DuplexIO audio frame."""

    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.config = config
        self.preprocessor = AudioToMelSpectrogramPreprocessor(config)
        self.encoder = ConformerEncoder(config)
        self.output_dim = config.dim

    def new_state(self, *, device: torch.device) -> FastConformerStreamingState:
        config = self.config
        dtype = self.preprocessor.featurizer.window.dtype
        return FastConformerStreamingState(
            sample_buffer=torch.zeros(
                config.sample_history_size,
                device=device,
                dtype=dtype,
            ),
            feature_buffer=torch.zeros(
                (config.features, config.feature_buffer_size),
                device=device,
                dtype=dtype,
            ),
            attention_caches=tuple(
                torch.empty(0, config.dim, device=device, dtype=dtype)
                for _ in range(config.num_layers)
            ),
            convolution_caches=tuple(
                torch.zeros(
                    config.dim,
                    config.convolution_kernel_size - 1,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ),
        )

    def step(
        self,
        waveform: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        config = self.config
        resampled = F.interpolate(
            waveform.float(),
            size=config.native_frame_size,
            mode="linear",
            align_corners=False,
        )[0, 0]
        if state.frames_seen:
            # NeMo's centered STFT reflects past the current waveform end. The
            # next append replaces that reflected tail, so recompute the prior
            # endpoint feature together with the eight new valid features.
            samples = torch.cat((state.sample_buffer, resampled))
            frame_offset = (
                config.sample_history_size // config.window_stride - 1
            )
            feature_update = self.preprocessor.streaming_chunk(
                samples,
                frame_offset,
                config.feature_chunk_size + 1,
            )
            corrected_previous = feature_update[:, :1]
            feature_chunk = feature_update[:, 1:]
        else:
            samples = resampled
            feature_chunk = self.preprocessor.streaming_chunk(
                samples,
                0,
                config.feature_chunk_size,
            )
        history_size = min(
            state.frames_seen * config.feature_chunk_size,
            config.feature_buffer_size,
        )
        history = (
            state.feature_buffer[:, -history_size:]
            if history_size
            else state.feature_buffer[:, :0]
        )
        if state.frames_seen:
            history = torch.cat((history[:, :-1], corrected_previous), dim=-1)
        # Three causal stride-2 convolutions emit at mel indices 0, 8, 16, ...;
        # the retained history keeps that global phase after the startup rows.
        features = torch.cat((history, feature_chunk[:, :1]), dim=-1)
        feature_buffer = torch.cat(
            (history, feature_chunk),
            dim=-1,
        )[:, -config.feature_buffer_size :]
        sample_buffer = resampled[-config.sample_history_size :]
        state = FastConformerStreamingState(
            sample_buffer=sample_buffer,
            feature_buffer=feature_buffer,
            attention_caches=state.attention_caches,
            convolution_caches=state.convolution_caches,
            frames_seen=state.frames_seen + 1,
        )
        return self.encoder.step(features, state)


__all__ = [
    "FastConformerConfig",
    "FastConformerStreamingState",
    "FastConformerUserEncoder",
]
