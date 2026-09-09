# Copyright (c) Kyutai, all rights reserved.
# See PocketTTSLicense.txt in this directory.
"""Inference-only Pocket Mimi, with explicit functional stream state."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class Conv1dState:
    previous: Tensor | None
    first: bool


class StreamingConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        pad_mode: str = "constant",
    ):
        super().__init__()
        self.pad_mode = pad_mode
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def get_initial_state(self) -> Conv1dState:
        return Conv1dState(previous=None, first=True)

    def step(self, x: Tensor, state: Conv1dState) -> tuple[Tensor, Conv1dState]:
        kernel = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        stride = self.conv.stride[0]
        previous_len = kernel - stride
        if previous_len:
            if state.previous is None:
                previous = torch.zeros(
                    x.shape[0],
                    self.conv.in_channels,
                    previous_len,
                    device=x.device,
                    dtype=x.dtype,
                )
            else:
                previous = state.previous
            if self.pad_mode == "replicate" and state.first:
                previous = x[..., :1].expand(-1, -1, previous_len).clone()
            x_padded = torch.cat([previous, x], dim=-1)
        else:
            x_padded = x
        y = self.conv(x_padded)
        new_previous = x_padded[..., -previous_len:].clone() if previous_len else None
        return (y, Conv1dState(previous=new_previous, first=False))


@dataclass
class ConvTranspose1dState:
    partial: Tensor | None


class StreamingConvTranspose1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.convtr = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias
        )

    def get_initial_state(self) -> ConvTranspose1dState:
        return ConvTranspose1dState(partial=None)

    def step(
        self, x: Tensor, state: ConvTranspose1dState
    ) -> tuple[Tensor, ConvTranspose1dState]:
        y = self.convtr(x)
        crop = self.convtr.kernel_size[0] - self.convtr.stride[0]
        if not crop:
            return (y, ConvTranspose1dState(partial=None))
        if state.partial is not None:
            y[..., :crop] += state.partial
        partial = y[..., -crop:].clone()
        if self.convtr.bias is not None:
            partial = partial - self.convtr.bias[:, None]
        return (y[..., :-crop], ConvTranspose1dState(partial=partial))


@dataclass
class SEANetResnetBlockState:
    conv_states: list[Conv1dState]


class SEANetResnetBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1, compress: int = 2):
        super().__init__()
        hidden = dim // compress
        self.block = nn.ModuleList(
            [
                nn.ELU(alpha=1.0),
                StreamingConv1d(dim, hidden, kernel_size=3, dilation=dilation),
                nn.ELU(alpha=1.0),
                StreamingConv1d(hidden, dim, kernel_size=1),
            ]
        )

    def get_initial_state(self) -> SEANetResnetBlockState:
        return SEANetResnetBlockState(
            conv_states=[
                layer.get_initial_state()
                for layer in self.block
                if isinstance(layer, StreamingConv1d)
            ]
        )

    def step(
        self, x: Tensor, state: SEANetResnetBlockState
    ) -> tuple[Tensor, SEANetResnetBlockState]:
        y = x
        conv_states = []
        idx = 0
        for layer in self.block:
            if isinstance(layer, StreamingConv1d):
                y, conv_state = layer.step(y, state.conv_states[idx])
                conv_states.append(conv_state)
                idx += 1
            else:
                y = layer(y)
        return (x + y, SEANetResnetBlockState(conv_states=conv_states))


SEANetLayerState = Conv1dState | ConvTranspose1dState | SEANetResnetBlockState


@dataclass
class SEANetState:
    layer_states: list[SEANetLayerState]


class SEANetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 64
        ratios = [6, 5, 4]
        mult = 1
        model: list[nn.Module] = [StreamingConv1d(1, n_filters, kernel_size=7)]
        for ratio in reversed(ratios):
            model.append(SEANetResnetBlock(mult * n_filters))
            model.append(nn.ELU(alpha=1.0))
            model.append(
                StreamingConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )
            mult *= 2
        model.append(nn.ELU(alpha=1.0))
        model.append(StreamingConv1d(mult * n_filters, 512, kernel_size=3))
        self.model = nn.ModuleList(model)

    def get_initial_state(self) -> SEANetState:
        return SEANetState(
            layer_states=[
                layer.get_initial_state()
                for layer in self.model
                if isinstance(layer, (StreamingConv1d, SEANetResnetBlock))
            ]
        )

    def step(self, x: Tensor, state: SEANetState) -> tuple[Tensor, SEANetState]:
        layer_states = []
        idx = 0
        for layer in self.model:
            if isinstance(layer, StreamingConv1d):
                layer_state = state.layer_states[idx]
                assert isinstance(layer_state, Conv1dState)
                x, layer_state = layer.step(x, layer_state)
                layer_states.append(layer_state)
                idx += 1
            elif isinstance(layer, SEANetResnetBlock):
                layer_state = state.layer_states[idx]
                assert isinstance(layer_state, SEANetResnetBlockState)
                x, layer_state = layer.step(x, layer_state)
                layer_states.append(layer_state)
                idx += 1
            else:
                x = layer(x)
        return (x, SEANetState(layer_states=layer_states))


class SEANetDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        n_filters = 64
        ratios = [6, 5, 4]
        mult = 2 ** len(ratios)
        model: list[nn.Module] = [StreamingConv1d(512, mult * n_filters, kernel_size=7)]
        for ratio in ratios:
            model.append(nn.ELU(alpha=1.0))
            model.append(
                StreamingConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                )
            )
            model.append(SEANetResnetBlock(mult * n_filters // 2))
            mult //= 2
        model.append(nn.ELU(alpha=1.0))
        model.append(StreamingConv1d(n_filters, 1, kernel_size=3))
        self.model = nn.ModuleList(model)

    def get_initial_state(self) -> SEANetState:
        return SEANetState(
            layer_states=[
                layer.get_initial_state()
                for layer in self.model
                if isinstance(
                    layer,
                    (StreamingConv1d, StreamingConvTranspose1d, SEANetResnetBlock),
                )
            ]
        )

    def step(self, x: Tensor, state: SEANetState) -> tuple[Tensor, SEANetState]:
        layer_states = []
        idx = 0
        for layer in self.model:
            if isinstance(layer, StreamingConv1d):
                layer_state = state.layer_states[idx]
                assert isinstance(layer_state, Conv1dState)
                x, layer_state = layer.step(x, layer_state)
                layer_states.append(layer_state)
                idx += 1
            elif isinstance(layer, StreamingConvTranspose1d):
                layer_state = state.layer_states[idx]
                assert isinstance(layer_state, ConvTranspose1dState)
                x, layer_state = layer.step(x, layer_state)
                layer_states.append(layer_state)
                idx += 1
            elif isinstance(layer, SEANetResnetBlock):
                layer_state = state.layer_states[idx]
                assert isinstance(layer_state, SEANetResnetBlockState)
                x, layer_state = layer.step(x, layer_state)
                layer_states.append(layer_state)
                idx += 1
            else:
                x = layer(x)
        return (x, SEANetState(layer_states=layer_states))


def apply_rope(
    q: Tensor,
    k: Tensor,
    offset: int = 0,
    max_period: float = 10000.0,
    positions: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    b, t, h, d = q.shape
    assert (b, t, d) == (k.shape[0], k.shape[1], k.shape[3])
    freqs = torch.exp(
        torch.arange(d // 2, device=q.device, dtype=torch.float32)
        * (-math.log(max_period) * 2 / d)
    )
    if positions is None:
        positions = torch.arange(t, device=q.device, dtype=torch.float32) + offset
    ts = positions.view(-1, 1, 1)
    q = q.view(b, t, h, d // 2, 2)
    k = k.view(b, t, k.shape[2], d // 2, 2)
    cos = torch.cos(freqs * ts)
    sin = torch.sin(freqs * ts)

    def rotate(x: Tensor) -> Tensor:
        real = x[..., 0].float()
        imag = x[..., 1].float()
        return torch.stack(
            [
                (real * cos - imag * sin).to(x.dtype),
                (real * sin + imag * cos).to(x.dtype),
            ],
            dim=-1,
        ).view(*x.shape[:-2], d)

    return (rotate(q), rotate(k))


class LayerScale(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.scale * x


@dataclass
class AttentionState:
    k: Tensor
    v: Tensor
    seq_len: int


class StreamingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, context: int = 250):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.context = context
        self.dim_per_head = embed_dim // num_heads
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def get_initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> AttentionState:
        shape = (batch_size, 0, self.num_heads, self.dim_per_head)
        return AttentionState(
            k=torch.empty(shape, device=device, dtype=dtype),
            v=torch.empty(shape, device=device, dtype=dtype),
            seq_len=0,
        )

    def step(self, x: Tensor, state: AttentionState) -> tuple[Tensor, AttentionState]:
        b, t, _ = x.shape
        q, k, v = (
            self.in_proj(x)
            .view(b, t, 3, self.num_heads, self.dim_per_head)
            .unbind(dim=2)
        )
        q, k = apply_rope(q, k, offset=state.seq_len)
        k_cache = torch.cat([state.k, k], dim=1)
        v_cache = torch.cat([state.v, v], dim=1)
        pos_q = state.seq_len + torch.arange(t, device=x.device)
        pos_k = (
            state.seq_len
            - state.k.shape[1]
            + torch.arange(k_cache.shape[1], device=x.device)
        )
        delta = pos_q[:, None] - pos_k[None, :]
        mask = (delta >= 0) & (delta < self.context)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_cache.transpose(1, 2),
            v_cache.transpose(1, 2),
            mask.view(1, 1, t, k_cache.shape[1]),
            dropout_p=0.0,
        )
        # The next query can see at most context - 1 previous keys. Clone the
        # retained window so a large prefill allocation is actually released.
        keep = min(self.context - 1, k_cache.shape[1])
        start = k_cache.shape[1] - keep
        new_state = AttentionState(
            k=k_cache[:, start:].clone(),
            v=v_cache[:, start:].clone(),
            seq_len=state.seq_len + t,
        )
        return (
            self.out_proj(y.transpose(1, 2).reshape(b, t, self.embed_dim)),
            new_state,
        )


@dataclass
class TransformerLayerState:
    self_attn: AttentionState


class StreamingTransformerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = StreamingMultiheadAttention()
        self.norm1 = nn.LayerNorm(512, eps=1e-05)
        self.norm2 = nn.LayerNorm(512, eps=1e-05)
        self.linear1 = nn.Linear(512, 2048, bias=False)
        self.linear2 = nn.Linear(2048, 512, bias=False)
        self.layer_scale_1 = LayerScale(512)
        self.layer_scale_2 = LayerScale(512)

    def get_initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> TransformerLayerState:
        return TransformerLayerState(
            self_attn=self.self_attn.get_initial_state(batch_size, device, dtype)
        )

    def step(
        self, x: Tensor, state: TransformerLayerState
    ) -> tuple[Tensor, TransformerLayerState]:
        attn, attn_state = self.self_attn.step(self.norm1(x), state.self_attn)
        x = x + self.layer_scale_1(attn)
        x = x + self.layer_scale_2(self.linear2(F.gelu(self.linear1(self.norm2(x)))))
        return (x, TransformerLayerState(self_attn=attn_state))


@dataclass
class TransformerState:
    layers: list[TransformerLayerState]


class StreamingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([StreamingTransformerLayer() for _ in range(2)])

    def get_initial_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> TransformerState:
        return TransformerState(
            layers=[
                layer.get_initial_state(batch_size, device, dtype)
                for layer in self.layers
                if isinstance(layer, StreamingTransformerLayer)
            ]
        )

    def step(
        self, x: Tensor, state: TransformerState
    ) -> tuple[Tensor, TransformerState]:
        states = []
        for layer, layer_state in zip(self.layers, state.layers):
            assert isinstance(layer, StreamingTransformerLayer)
            x, new_state = layer.step(x, layer_state)
            states.append(new_state)
        return (x, TransformerState(layers=states))


@dataclass
class ProjectedTransformerState:
    transformer: TransformerState


class ProjectedTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = StreamingTransformer()

    def get_initial_state(self, x: Tensor) -> ProjectedTransformerState:
        return ProjectedTransformerState(
            transformer=self.transformer.get_initial_state(
                x.shape[0], x.device, x.dtype
            )
        )

    def step(
        self, x: Tensor, state: ProjectedTransformerState
    ) -> tuple[list[Tensor], ProjectedTransformerState]:
        y, new_state = self.transformer.step(x.transpose(1, 2), state.transformer)
        return ([y.transpose(1, 2)], ProjectedTransformerState(transformer=new_state))


class ConvDownsample1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = StreamingConv1d(
            512, 32, kernel_size=32, stride=16, bias=False, pad_mode="replicate"
        )

    def get_initial_state(self) -> Conv1dState:
        return self.conv.get_initial_state()

    def step(self, x: Tensor, state: Conv1dState) -> tuple[Tensor, Conv1dState]:
        return self.conv.step(x, state)


class ConvTrUpsample1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.convtr = StreamingConvTranspose1d(
            512, 512, kernel_size=32, stride=16, groups=512, bias=False
        )

    def get_initial_state(self) -> ConvTranspose1dState:
        return self.convtr.get_initial_state()

    def step(
        self, x: Tensor, state: ConvTranspose1dState
    ) -> tuple[Tensor, ConvTranspose1dState]:
        return self.convtr.step(x, state)


class DummyQuantizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_proj = nn.Conv1d(32, 512, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.output_proj(x)


@dataclass
class ContinuousMimiState:
    encoder: SEANetState
    decoder: SEANetState
    downsample: Conv1dState
    upsample: ConvTranspose1dState
    encoder_transformer: ProjectedTransformerState
    decoder_transformer: ProjectedTransformerState


class PocketMimi(nn.Module):
    """The checkpoint's 32-dimensional, 24 kHz continuous codec.

    State updates return new containers and tensors: sharing a state between
    speculative appends cannot mutate the accepted conversation history.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = SEANetEncoder()
        self.encoder_transformer = ProjectedTransformer()
        self.downsample = ConvDownsample1d()
        self.quantizer = DummyQuantizer()
        self.upsample = ConvTrUpsample1d()
        self.decoder_transformer = ProjectedTransformer()
        self.decoder = SEANetDecoder()

    def new_state(self, batch_size: int) -> ContinuousMimiState:
        parameter = self.quantizer.output_proj.weight
        empty = parameter.new_empty(batch_size, 512, 0)
        return ContinuousMimiState(
            encoder=self.encoder.get_initial_state(),
            decoder=self.decoder.get_initial_state(),
            downsample=self.downsample.get_initial_state(),
            upsample=self.upsample.get_initial_state(),
            encoder_transformer=self.encoder_transformer.get_initial_state(empty),
            decoder_transformer=self.decoder_transformer.get_initial_state(empty),
        )

    def encode(
        self,
        waveform: Tensor,
        state: ContinuousMimiState,
    ) -> tuple[Tensor, ContinuousMimiState]:
        """Consume (batch, 1, frames * 1920); return (batch, 32, frames)."""
        hidden, encoder = self.encoder.step(waveform, state.encoder)
        (hidden,), transformer = self.encoder_transformer.step(
            hidden,
            state.encoder_transformer,
        )
        latent, downsample = self.downsample.step(hidden, state.downsample)
        return latent, ContinuousMimiState(
            encoder=encoder,
            decoder=state.decoder,
            downsample=downsample,
            upsample=state.upsample,
            encoder_transformer=transformer,
            decoder_transformer=state.decoder_transformer,
        )

    def decode(
        self,
        latent: Tensor,
        state: ContinuousMimiState,
    ) -> tuple[Tensor, ContinuousMimiState]:
        """Consume (batch, 32, frames); return (batch, 1, frames * 1920)."""
        hidden = self.quantizer(latent)
        hidden, upsample = self.upsample.step(hidden, state.upsample)
        (hidden,), transformer = self.decoder_transformer.step(
            hidden,
            state.decoder_transformer,
        )
        waveform, decoder = self.decoder.step(hidden, state.decoder)
        return waveform, ContinuousMimiState(
            encoder=state.encoder,
            decoder=decoder,
            downsample=state.downsample,
            upsample=upsample,
            encoder_transformer=state.encoder_transformer,
            decoder_transformer=transformer,
        )
