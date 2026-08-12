# Copyright (c) 2020-2025, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Bounded streaming path for DuplexIO's frozen FastConformer encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

COMPILED_FLEX_ATTENTION: Any | None = None


def fused_flex_attention() -> Any:
    """Return the same shape-dynamic FlexAttention callable used in training."""
    global COMPILED_FLEX_ATTENTION
    if COMPILED_FLEX_ATTENTION is None:
        COMPILED_FLEX_ATTENTION = torch.compile(flex_attention, dynamic=True)
    return COMPILED_FLEX_ATTENTION


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
    expected_cudnn_version: int | None = None

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
    """Architecture-bounded state retained across user-audio frames."""

    sample_buffer: Tensor
    feature_buffer: Tensor
    attention_caches: tuple[Tensor | None, ...]
    convolution_caches: tuple[Tensor | None, ...]
    frames_seen: int = 0


class FilterbankFeatures(nn.Module):
    """The two persistent buffers in NeMo's frozen mel preprocessor."""

    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        self.register_buffer(
            "window",
            torch.empty(config.window_size, dtype=torch.float32),
        )
        self.register_buffer(
            "fb",
            torch.empty(
                1,
                config.features,
                config.n_fft // 2 + 1,
                dtype=torch.float32,
            ),
        )


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
        return self.out(hidden)[0]

    @staticmethod
    def output_frame_count(feature_count: int) -> int:
        """Return rows emitted by the fixed three-stage stride-2 stack."""
        for _ in range(3):
            feature_count = feature_count // 2 + 1
        return feature_count


class ConformerFeedForward(nn.Module):
    def __init__(self, dim: int, feedforward_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, feedforward_dim)
        self.linear2 = nn.Linear(feedforward_dim, dim)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.linear2(F.silu(self.linear1(hidden)))


class RelPositionAttention(nn.Module):
    """The compiled relative-position attention used by the training run."""

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
        self.pos_bias_u = nn.Parameter(
            torch.empty(config.num_heads, self.head_dim)
        )
        self.pos_bias_v = nn.Parameter(
            torch.empty(config.num_heads, self.head_dim)
        )

    def forward(self, hidden: Tensor, pos_emb: Tensor, mask: Tensor) -> Tensor:
        frames, _ = hidden.shape
        heads = self.num_heads
        head_dim = self.head_dim
        q = self.linear_q(hidden).view(1, frames, heads, head_dim)
        k = self.linear_k(hidden).view(1, frames, heads, head_dim).transpose(1, 2)
        v = self.linear_v(hidden).view(1, frames, heads, head_dim).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)

        window = min(self.left_context, frames - 1)
        position = self.linear_pos(pos_emb).view(
            1,
            -1,
            heads,
            head_dim,
        ).transpose(1, 2)
        position = position[:, :, frames - 1 - window : frames]
        scale = 1.0 / math.sqrt(head_dim)
        position_scores = (
            torch.einsum("bhtd,ahmd->bhtm", q_v, position) * scale
        )
        keep = ~mask

        def mask_mod(b, h, q_idx, kv_idx):
            return keep[b, q_idx, kv_idx]

        def score_mod(score, b, h, q_idx, kv_idx):
            position_index = (kv_idx - q_idx + window).clamp(0, window)
            return score + position_scores[b, h, q_idx, position_index]

        block_mask = create_block_mask(
            mask_mod,
            1,
            1,
            frames,
            frames,
            device=hidden.device,
        )
        attend = fused_flex_attention() if hidden.is_cuda else flex_attention
        output = attend(
            q_u,
            k,
            v,
            score_mod=score_mod,
            block_mask=block_mask,
            scale=scale,
        )
        output = output.transpose(1, 2).reshape(1, frames, heads * head_dim)
        return self.linear_out(output)[0]

    def step(
        self,
        hidden: Tensor,
        cache: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Attend one frame to its bounded causal prefix."""
        sequence = (
            hidden.unsqueeze(0)
            if cache is None
            else torch.cat((cache, hidden.unsqueeze(0)))
        )
        query = self.linear_q(hidden).view(self.num_heads, self.head_dim)
        key = self.linear_k(sequence).view(
            -1,
            self.num_heads,
            self.head_dim,
        )
        value = self.linear_v(sequence).view(
            -1,
            self.num_heads,
            self.head_dim,
        )
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
        position = hidden.new_zeros(
            sequence.shape[0],
            self.num_heads * self.head_dim,
        )
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

    def step_sequence(
        self,
        hidden: Tensor,
        cache: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Attend a causal continuation to its cached bounded prefix."""
        sequence = hidden if cache is None else torch.cat((cache, hidden))
        frames = hidden.shape[0]
        prefix_frames = sequence.shape[0] - frames
        query = self.linear_q(hidden).view(
            frames,
            self.num_heads,
            self.head_dim,
        )
        key = self.linear_k(sequence).view(
            -1,
            self.num_heads,
            self.head_dim,
        )
        value = self.linear_v(sequence).view(
            -1,
            self.num_heads,
            self.head_dim,
        )
        query_positions = prefix_frames + torch.arange(
            frames,
            device=hidden.device,
        )
        key_positions = torch.arange(sequence.shape[0], device=hidden.device)
        distances = query_positions[:, None] - key_positions[None, :]
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
        position = hidden.new_zeros(
            frames,
            sequence.shape[0],
            self.num_heads * self.head_dim,
        )
        position[:, :, 0::2] = torch.sin(
            distances.unsqueeze(-1) * frequencies
        ).to(position.dtype)
        position[:, :, 1::2] = torch.cos(
            distances.unsqueeze(-1) * frequencies
        ).to(position.dtype)
        position = self.linear_pos(position).view(
            frames,
            sequence.shape[0],
            self.num_heads,
            self.head_dim,
        )
        content_scores = torch.einsum(
            "fhd,thd->fht",
            query + self.pos_bias_u,
            key,
        )
        position_scores = torch.einsum(
            "fhd,fthd->fht",
            query + self.pos_bias_v,
            position,
        )
        invalid = (distances < 0) | (distances > self.left_context)
        scores = (content_scores + position_scores) / math.sqrt(self.head_dim)
        probabilities = F.softmax(
            scores.masked_fill(invalid.unsqueeze(1), -torch.inf),
            dim=-1,
        )
        attended = torch.einsum("fht,thd->fhd", probabilities, value)
        output = self.linear_out(attended.flatten(1))
        return output, sequence[-self.left_context :]


class ConformerConvolution(nn.Module):
    def __init__(self, config: FastConformerConfig) -> None:
        super().__init__()
        dim = config.dim
        self.kernel_size = config.convolution_kernel_size
        self.pointwise_conv1 = nn.Conv1d(dim, 2 * dim, 1)
        self.depthwise_conv = nn.Conv1d(
            dim,
            dim,
            self.kernel_size,
            groups=dim,
        )
        self.batch_norm = nn.LayerNorm(dim)
        self.pointwise_conv2 = nn.Conv1d(dim, dim, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        output, _ = self.forward_sequence(hidden)
        return output

    def forward_sequence(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        """Apply a causal sequence and return its bounded streaming cache."""
        pointwise = F.glu(
            self.pointwise_conv1(hidden.T.unsqueeze(0)),
            dim=1,
        )
        convolved = self.depthwise_conv(
            F.pad(pointwise, (self.kernel_size - 1, 0))
        )
        convolved = self.batch_norm(convolved.transpose(1, 2))
        convolved = F.silu(convolved).transpose(1, 2)
        cache_size = self.kernel_size - 1
        cache = (
            F.pad(
                pointwise[0],
                (max(0, cache_size - pointwise.shape[-1]), 0),
            )[:, -cache_size:]
            if cache_size
            else pointwise[0, :, :0]
        )
        return self.pointwise_conv2(convolved)[0].T, cache

    def step(
        self,
        hidden: Tensor,
        cache: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply one causal convolution frame and advance its fixed cache."""
        pointwise = F.glu(
            self.pointwise_conv1(hidden.view(1, -1, 1)),
            dim=1,
        )
        if cache is None:
            cache = pointwise.new_zeros(
                pointwise.shape[1],
                self.kernel_size - 1,
            )
        sequence = torch.cat((cache, pointwise[0]), dim=-1)
        convolved = self.depthwise_conv(sequence.unsqueeze(0))
        convolved = self.batch_norm(convolved.transpose(1, 2))
        convolved = F.silu(convolved).transpose(1, 2)
        output = self.pointwise_conv2(convolved)[0, :, 0]
        return output, sequence[:, 1:]

    def step_sequence(
        self,
        hidden: Tensor,
        cache: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Apply a causal continuation and advance its fixed cache."""
        pointwise = F.glu(
            self.pointwise_conv1(hidden.T.unsqueeze(0)),
            dim=1,
        )
        if cache is None:
            cache = pointwise.new_zeros(
                pointwise.shape[1],
                self.kernel_size - 1,
            )
        sequence = torch.cat((cache, pointwise[0]), dim=-1)
        convolved = self.depthwise_conv(sequence.unsqueeze(0))
        convolved = self.batch_norm(convolved.transpose(1, 2))
        convolved = F.silu(convolved).transpose(1, 2)
        output = self.pointwise_conv2(convolved)[0].T
        return output, sequence[:, -(self.kernel_size - 1) :]


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

    def forward(self, hidden: Tensor, pos_emb: Tensor, mask: Tensor) -> Tensor:
        hidden, _, _ = self.forward_sequence(hidden, pos_emb, mask)
        return hidden

    def forward_sequence(
        self,
        hidden: Tensor,
        pos_emb: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply a sequence and return the exact state needed by ``step``."""
        hidden = hidden + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(hidden)
        )
        attention_input = self.norm_self_att(hidden)
        hidden = hidden + self.self_attn(
            attention_input,
            pos_emb,
            mask,
        )
        convolution, convolution_cache = self.conv.forward_sequence(
            self.norm_conv(hidden)
        )
        hidden = hidden + convolution
        hidden = hidden + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden)
        )
        attention_cache = (
            attention_input[-self.self_attn.left_context :]
            if self.self_attn.left_context
            else attention_input[:0]
        )
        return self.norm_out(hidden), attention_cache, convolution_cache

    def step(
        self,
        hidden: Tensor,
        attention_cache: Tensor | None,
        convolution_cache: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance one frame with the same operation order as ``forward``."""
        hidden = hidden + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(hidden)
        )
        update, attention_cache = self.self_attn.step(
            self.norm_self_att(hidden),
            attention_cache,
        )
        hidden = hidden + update
        update, convolution_cache = self.conv.step(
            self.norm_conv(hidden),
            convolution_cache,
        )
        hidden = hidden + update
        hidden = hidden + 0.5 * self.feed_forward2(
            self.norm_feed_forward2(hidden)
        )
        return self.norm_out(hidden), attention_cache, convolution_cache

    def step_sequence(
        self,
        hidden: Tensor,
        attention_cache: Tensor | None,
        convolution_cache: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance a causal continuation with the bounded streaming caches."""
        hidden = hidden + 0.5 * self.feed_forward1(
            self.norm_feed_forward1(hidden)
        )
        update, attention_cache = self.self_attn.step_sequence(
            self.norm_self_att(hidden),
            attention_cache,
        )
        hidden = hidden + update
        update, convolution_cache = self.conv.step_sequence(
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
        self.config = config
        self.input_scale = math.sqrt(config.dim)
        self.pre_encode = ConvSubsampling(config)
        self.layers = nn.ModuleList(
            [ConformerLayer(config) for _ in range(config.num_layers)]
        )

    def forward(self, features: Tensor) -> Tensor:
        hidden, _, _ = self.forward_sequence(features)
        return hidden

    def forward_sequence(
        self,
        features: Tensor,
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Encode a causal sequence and return its terminal streaming state."""
        hidden = self.pre_encode(features) * self.input_scale
        frames = hidden.shape[0]
        offsets = torch.arange(
            frames - 1,
            -frames,
            -1,
            dtype=torch.float32,
            device=hidden.device,
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(
                0,
                self.config.dim,
                2,
                dtype=torch.float32,
                device=hidden.device,
            )
            * -(math.log(10_000.0) / self.config.dim)
        )
        pos_emb = hidden.new_zeros(1, 2 * frames - 1, self.config.dim)
        pos_emb[0, :, 0::2] = torch.sin(offsets * frequencies).to(hidden.dtype)
        pos_emb[0, :, 1::2] = torch.cos(offsets * frequencies).to(hidden.dtype)

        positions = torch.arange(frames, device=hidden.device)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)
        mask = (distance < 0) | (distance > self.config.attention_left_context)
        mask = mask.unsqueeze(0)
        attention_caches = []
        convolution_caches = []
        for layer in self.layers:
            hidden, attention_cache, convolution_cache = layer.forward_sequence(
                hidden,
                pos_emb,
                mask,
            )
            attention_caches.append(attention_cache)
            convolution_caches.append(convolution_cache)
        return hidden, tuple(attention_caches), tuple(convolution_caches)

    def step(
        self,
        features: Tensor,
        attention_caches: tuple[Tensor | None, ...],
        convolution_caches: tuple[Tensor | None, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Encode one newly available subsampled frame."""
        hidden = self.pre_encode(features)[-1] * self.input_scale
        next_attention_caches = []
        next_convolution_caches = []
        for layer, attention_cache, convolution_cache in zip(
            self.layers,
            attention_caches,
            convolution_caches,
            strict=True,
        ):
            hidden, attention_cache, convolution_cache = layer.step(
                hidden,
                attention_cache,
                convolution_cache,
            )
            next_attention_caches.append(attention_cache)
            next_convolution_caches.append(convolution_cache)
        return (
            hidden,
            tuple(next_attention_caches),
            tuple(next_convolution_caches),
        )

    def step_sequence(
        self,
        features: Tensor,
        history_feature_count: int,
        frame_count: int,
        attention_caches: tuple[Tensor | None, ...],
        convolution_caches: tuple[Tensor | None, ...],
    ) -> tuple[Tensor, tuple[Tensor, ...], tuple[Tensor, ...]]:
        """Encode a causal continuation from mel history and bounded caches."""
        first_frame = self.pre_encode.output_frame_count(
            history_feature_count + 1
        ) - 1
        hidden = self.pre_encode(features)[
            first_frame : first_frame + frame_count
        ] * self.input_scale
        assert hidden.shape[0] == frame_count
        next_attention_caches = []
        next_convolution_caches = []
        for layer, attention_cache, convolution_cache in zip(
            self.layers,
            attention_caches,
            convolution_caches,
            strict=True,
        ):
            hidden, attention_cache, convolution_cache = layer.step_sequence(
                hidden,
                attention_cache,
                convolution_cache,
            )
            next_attention_caches.append(attention_cache)
            next_convolution_caches.append(convolution_cache)
        return (
            hidden,
            tuple(next_attention_caches),
            tuple(next_convolution_caches),
        )


class FastConformerUserEncoder(nn.Module):
    """One frozen 512-dim feature per DuplexIO audio frame."""

    def __init__(
        self,
        config: FastConformerConfig,
        *,
        compute_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.config = config
        self.compute_dtype = compute_dtype
        self.preprocessor = AudioToMelSpectrogramPreprocessor(config)
        self.encoder = ConformerEncoder(config)
        self.output_dim = config.dim
        self.requires_grad_(False)

    def validate_cudnn_version(self, device: torch.device) -> None:
        """Require the CUDA convolution implementation used by training."""
        if device.type != "cuda" or self.config.expected_cudnn_version is None:
            return
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version != self.config.expected_cudnn_version:
            raise RuntimeError(
                "FastConformer requires the checkpoint's training cuDNN "
                f"version {self.config.expected_cudnn_version}, got "
                f"{cudnn_version}"
            )

    def new_state(self, *, device: torch.device) -> FastConformerStreamingState:
        self.validate_cudnn_version(device)
        config = self.config
        feature_dtype = self.preprocessor.featurizer.window.dtype
        return FastConformerStreamingState(
            sample_buffer=torch.zeros(
                config.sample_history_size,
                device=device,
                dtype=feature_dtype,
            ),
            feature_buffer=torch.zeros(
                (config.features, config.feature_buffer_size),
                device=device,
                dtype=feature_dtype,
            ),
            attention_caches=(None,) * config.num_layers,
            convolution_caches=(None,) * config.num_layers,
        )

    def encode_silent_prefix(
        self,
        frame_count: int,
        *,
        device: torch.device,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        """Encode the complete silent prompt and retain its streaming state."""
        state = self.new_state(device=device)
        if frame_count == 0:
            return state.feature_buffer.new_empty(0, self.output_dim), state
        config = self.config
        resampled = torch.zeros(
            frame_count * config.native_frame_size,
            device=device,
            dtype=self.preprocessor.featurizer.window.dtype,
        )
        features = self.preprocessor(resampled)
        state_features = features[
            :, : frame_count * config.feature_chunk_size
        ]
        encoder_features = features[
            :, : (frame_count - 1) * config.feature_chunk_size + 1
        ]
        with torch.autocast(
            device_type=features.device.type,
            dtype=self.compute_dtype,
            enabled=features.is_cuda and self.compute_dtype != torch.float32,
        ):
            (
                prefix_features,
                attention_caches,
                convolution_caches,
            ) = self.encoder.forward_sequence(encoder_features)
        return prefix_features, FastConformerStreamingState(
            sample_buffer=resampled[-config.sample_history_size :],
            feature_buffer=state_features[:, -config.feature_buffer_size :],
            attention_caches=attention_caches,
            convolution_caches=convolution_caches,
            frames_seen=frame_count,
        )

    def encode_prefix(self, waveform: Tensor) -> Tensor:
        """Encode ``(frames, frame_size)`` audio like training's batch path."""
        frames = waveform.shape[0]
        assert waveform.shape == (frames, self.config.frame_size)
        if frames == 0:
            return waveform.new_empty(0, self.output_dim)
        self.validate_cudnn_version(waveform.device)
        resampled = F.interpolate(
            waveform.flatten().view(1, 1, -1).float(),
            size=frames * self.config.native_frame_size,
            mode="linear",
            align_corners=False,
        )[0, 0]
        features = self.preprocessor(resampled)
        with torch.autocast(
            device_type=features.device.type,
            dtype=self.compute_dtype,
            enabled=features.is_cuda and self.compute_dtype != torch.float32,
        ):
            return self.encoder(features)[:frames]

    def step(
        self,
        waveform: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        """Advance the bounded streaming state by one 80-ms frame."""
        assert waveform.shape == (1, 1, self.config.frame_size)
        features, state = self.step_sequence(waveform, state)
        return features[0], state

    def steady_step(
        self,
        waveform: Tensor,
        sample_buffer: Tensor,
        feature_buffer: Tensor,
        *caches: Tensor,
    ) -> tuple[Tensor, ...]:
        """Advance one frame once every bounded encoder cache is populated."""
        layer_count = self.config.num_layers
        assert len(caches) == 2 * layer_count
        state = FastConformerStreamingState(
            sample_buffer=sample_buffer,
            feature_buffer=feature_buffer,
            attention_caches=tuple(caches[:layer_count]),
            convolution_caches=tuple(caches[layer_count:]),
            frames_seen=self.config.attention_left_context,
        )
        features, next_state = self.step_sequence(waveform, state)
        return (
            features,
            next_state.sample_buffer,
            next_state.feature_buffer,
            *next_state.attention_caches,
            *next_state.convolution_caches,
        )

    def step_sequence(
        self,
        waveform: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        """Advance a contiguous sequence with one causal encoder pass."""
        assert waveform.ndim == 3 and waveform.shape[:2] == (1, 1)
        assert waveform.shape[-1] % self.config.frame_size == 0
        config = self.config
        frame_count = waveform.shape[-1] // config.frame_size
        assert frame_count > 0
        resampled = F.interpolate(
            waveform.float(),
            size=frame_count * config.native_frame_size,
            mode="linear",
            align_corners=False,
        )[0, 0]
        if state.frames_seen:
            # NeMo's centered STFT reflects past the current waveform end. The
            # next append replaces that reflected tail, so recompute the prior
            # endpoint feature together with the eight new valid features.
            samples = torch.cat((state.sample_buffer, resampled))
            frame_offset = config.sample_history_size // config.window_stride - 1
            feature_update = self.preprocessor.streaming_chunk(
                samples,
                frame_offset,
                frame_count * config.feature_chunk_size + 1,
            )
            corrected_previous = feature_update[:, :1]
            feature_chunk = feature_update[:, 1:]
        else:
            feature_chunk = self.preprocessor.streaming_chunk(
                resampled,
                0,
                frame_count * config.feature_chunk_size,
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
        features = torch.cat((history, feature_chunk), dim=-1)
        feature_buffer = torch.cat(
            (history, feature_chunk),
            dim=-1,
        )[:, -config.feature_buffer_size :]
        with torch.autocast(
            device_type=features.device.type,
            dtype=self.compute_dtype,
            enabled=features.is_cuda and self.compute_dtype != torch.float32,
        ):
            hidden, attention_caches, convolution_caches = (
                self.encoder.step_sequence(
                    features,
                    history.shape[-1],
                    frame_count,
                    state.attention_caches,
                    state.convolution_caches,
                )
            )
        return hidden, FastConformerStreamingState(
            sample_buffer=resampled[-config.sample_history_size :],
            feature_buffer=feature_buffer,
            attention_caches=attention_caches,
            convolution_caches=convolution_caches,
            frames_seen=state.frames_seen + frame_count,
        )


__all__ = [
    "FastConformerConfig",
    "FastConformerStreamingState",
    "FastConformerUserEncoder",
]
