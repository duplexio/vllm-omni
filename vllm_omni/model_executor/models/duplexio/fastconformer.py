# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Cache-aware streaming FastConformer encoder for the user's audio."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from functools import cached_property, partial
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
from torchaudio.transforms import Resample

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
NUM_LOOKAHEAD_TOKENS = 0
# Encoder graphs are captured for these batch sizes; a batch pads to the next.
GRAPH_BATCH_SIZES = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)


def graph_batch_size(rows: int) -> int:
    return next((size for size in GRAPH_BATCH_SIZES if size >= rows), rows)


class FastConformerStreamState:
    """Explicit upstream encoder caches for one streaming utterance.

    A state split from a batched forward stays a lazy row of that batch until
    read individually. Batches are never mutated after splitting, so a later
    batch copies its rows straight out of them, whatever groups they came from.
    """

    def __init__(self, past_key_values: Any = None, padding_cache: Any = None) -> None:
        self._past_key_values = past_key_values
        self._padding_cache = padding_cache
        self.batch: FastConformerStreamState | None = None
        self.row = 0
        self.length = 0
        # A packed batch keeps every cache as a view of one ``[batch, elements]``
        # buffer, so moving rows costs one copy; containers are built on demand.
        self.flat: Tensor | None = None
        self.layout: FlatCacheLayout | None = None

    @classmethod
    def packed(cls, layout: FlatCacheLayout, flat: Tensor) -> FastConformerStreamState:
        state = cls()
        state.flat, state.layout = flat, layout
        return state

    @classmethod
    def row_of(cls, batch: FastConformerStreamState, row: int, length: int) -> FastConformerStreamState:
        state = cls()
        state.batch, state.row, state.length = batch, row, length
        return state

    @property
    def past_key_values(self) -> Any:
        self.materialize()
        return self._past_key_values

    @property
    def padding_cache(self) -> Any:
        self.materialize()
        return self._padding_cache

    def materialize(self) -> None:
        """Build packed containers, or slice this row's out of its batch."""
        if self.flat is not None and self._past_key_values is None:
            self._past_key_values, self._padding_cache = self.layout.containers(self.tensors())
        batch = self.batch
        if batch is None:
            return
        batch.materialize()
        index = self.row
        kv = copy.copy(batch._past_key_values)
        kv.layers = []
        for layer in batch._past_key_values.layers:
            single = copy.copy(layer)
            single.keys = layer.keys[index:index + 1]
            single.values = layer.values[index:index + 1]
            single.cumulative_length = self.length
            kv.layers.append(single)
        padding = copy.copy(batch._padding_cache)
        padding.layers = {}
        for name, layer in batch._padding_cache.layers.items():
            single = copy.copy(layer)
            single.cache = layer.cache[index:index + 1]
            padding.layers[name] = single
        self._past_key_values, self._padding_cache, self.batch = kv, padding, None

    @property
    def cached_frames(self) -> int:
        if self.batch is not None:
            return self.batch.cached_frames
        if self.flat is not None:
            return self.layout.cached_frames
        return 0 if self._past_key_values is None else self._past_key_values.layers[0].keys.shape[-2]

    def tensors(self) -> list[Tensor]:
        """Initialized cache storage, in native layer order."""
        if self.flat is not None:
            return self.layout.views(self.flat)
        return [
            tensor for layer in self.past_key_values.layers for tensor in (layer.keys, layer.values)
        ] + [layer.cache for layer in self.padding_cache.layers.values()]

    @staticmethod
    def copy_rows(states: list[FastConformerStreamState], destination: FastConformerStreamState) -> None:
        """Write each state's caches into its row of packed batch ``destination``.

        Rows that sit consecutively in one source batch move as one slice, so a
        group that re-forms in order costs a single copy, or one per cache
        tensor when the source is not packed alike.
        """
        runs: list[tuple[int, FastConformerStreamState, int, int]] = []
        for row, state in enumerate(states):
            source, index = (state, 0) if state.batch is None else (state.batch, state.row)
            if runs and runs[-1][1] is source and runs[-1][2] + runs[-1][3] == index:
                start, _, first, length = runs[-1]
                runs[-1] = (start, source, first, length + 1)
            else:
                runs.append((row, source, index, 1))
        targets: list[Tensor] = []
        values: list[Tensor] = []
        unpacked: dict[int, list[Tensor]] = {}
        for start, source, first, length in runs:
            if source.flat is not None and source.layout.key == destination.layout.key:
                targets.append(destination.flat[start:start + length])
                values.append(source.flat[first:first + length])
                continue
            if not unpacked:
                unpacked[id(destination)] = destination.tensors()
            if id(source) not in unpacked:
                unpacked[id(source)] = source.tensors()
            targets += [tensor[start:start + length] for tensor in unpacked[id(destination)]]
            values += [tensor[first:first + length] for tensor in unpacked[id(source)]]
        torch._foreach_copy_(targets, values)

    @property
    def seq_length(self) -> int:
        """Absolute frames consumed, beyond the retained window."""
        if self.batch is not None:
            return self.length
        if self.flat is not None:
            return self.layout.cached_frames
        return 0 if self._past_key_values is None else self._past_key_values.get_seq_length()

    @classmethod
    def stack(cls, states: list[FastConformerStreamState]) -> FastConformerStreamState:
        """Batch equal cache windows without mutating accepted request state.

        Nemotron uses relative positions and one-frame attention chunks. Its
        temporary batch can therefore start at the retained window length;
        absolute per-request lengths are restored when splitting the result.
        """
        first = states[0]
        if first.past_key_values is None:
            assert all(state.past_key_values is None for state in states)
            return cls()
        kv = copy.copy(first.past_key_values)
        kv.layers = []
        for index, layer in enumerate(first.past_key_values.layers):
            batched = copy.copy(layer)
            batched.keys = torch.cat([state.past_key_values.layers[index].keys for state in states])
            batched.values = torch.cat([state.past_key_values.layers[index].values for state in states])
            batched.cumulative_length = first.cached_frames
            kv.layers.append(batched)
        padding = copy.copy(first.padding_cache)
        padding.layers = {}
        for name, layer in first.padding_cache.layers.items():
            batched = copy.copy(layer)
            batched.cache = torch.cat([state.padding_cache.layers[name].cache for state in states])
            padding.layers[name] = batched
        return cls(kv, padding)

    def unbind(self, previous: list[FastConformerStreamState], frames: int) -> list[FastConformerStreamState]:
        """Restore request-owned cache containers after one batched forward."""
        return [
            FastConformerStreamState.row_of(self, index, old.seq_length + frames)
            for index, old in enumerate(previous)
        ]


class FlatCacheLayout:
    """Where each cache of one batch row sits within a flat row of one dtype."""

    def __init__(self, template: FastConformerStreamState) -> None:
        tensors = template.tensors()
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("Packed FastConformer caches need one dtype")
        self.shapes = [tensor.shape[1:] for tensor in tensors]
        self.sizes = [tensor[0].numel() for tensor in tensors]
        self.key = (tensors[0].dtype, tuple(self.shapes))
        self.cached_frames = template.cached_frames
        # Key and value windows lead; the convolution padding caches follow.
        self.kv_tensors = 2 * len(template.past_key_values.layers)
        # Keep only the container structure, not the template's storage.
        self.template = FastConformerStreamState(*self.fill(template, [None] * len(tensors)))

    @staticmethod
    def pack(tensors: list[Tensor]) -> Tensor:
        """Flatten batched tensors into one row-major ``[batch, elements]`` buffer."""
        return torch.cat([tensor.reshape(tensor.shape[0], -1) for tensor in tensors], dim=1)

    def views(self, flat: Tensor) -> list[Tensor]:
        pieces = flat.split(self.sizes, dim=1)
        return [piece.view(-1, *shape) for piece, shape in zip(pieces, self.shapes, strict=True)]

    def containers(self, tensors: list[Tensor]) -> tuple[Any, Any]:
        """Native cache containers shaped like the template's, holding ``tensors``."""
        return self.fill(self.template, tensors)

    def fill(self, state: FastConformerStreamState, tensors: list[Tensor | None]) -> tuple[Any, Any]:
        template = state.past_key_values
        kv = copy.copy(template)
        kv.layers = []
        position = 0
        for layer in template.layers:
            batched = copy.copy(layer)
            batched.keys, batched.values = tensors[position], tensors[position + 1]
            batched.cumulative_length = self.cached_frames
            position += 2
            kv.layers.append(batched)
        template = state.padding_cache
        padding = copy.copy(template)
        padding.layers = {}
        for name, layer in template.layers.items():
            batched = copy.copy(layer)
            batched.cache = tensors[position]
            position += 1
            padding.layers[name] = batched
        return kv, padding


@dataclass
class FastConformerAudioStreamState:
    """Raw-audio frontend and encoder state for one live utterance."""

    encoder: FastConformerStreamState = field(
        default_factory=FastConformerStreamState,
    )
    audio_buffer: Tensor | None = None
    buffer_start_sample: int = 0
    next_mel_frame: int = 0
    # Source-rate samples preceding the next chunk, so chunked resampling
    # matches the whole-signal resample (see streaming_resample_chunk).
    resample_tail: Tensor | None = None


def streaming_resample_chunk(
    chunk: Tensor,
    tail: Tensor | None,
    resampler: Resample,
) -> tuple[Tensor, Tensor]:
    """Resample one source-rate chunk with left context from earlier audio.

    Emits exactly ``len(chunk) * new / orig`` samples, phase-locked to the
    whole-signal resample, up to floating-point rounding. The final sinc
    half-width of each push sees zeros in place of the
    not-yet-received future samples (the same zero padding the whole-signal
    resample applies at the true end of the audio).
    """
    orig_stride = resampler.orig_freq // resampler.gcd
    new_stride = resampler.new_freq // resampler.gcd
    if chunk.ndim not in (1, 2) or chunk.shape[-1] % orig_stride:
        raise ValueError(
            f"Streaming resample chunks must be 1-D or batched multiples of {orig_stride} "
            f"source samples, got shape {tuple(chunk.shape)}"
        )
    # Comfortably beyond torchaudio's default sinc half-width, and a stride
    # multiple so the polyphase output stays on the whole-signal grid.
    context = 16 * orig_stride
    buffer = chunk if tail is None else torch.cat((tail, chunk), dim=-1)
    resampled = resampler(buffer)
    skip = (buffer.shape[-1] - chunk.shape[-1]) * new_stride // orig_stride
    emit = chunk.shape[-1] * new_stride // orig_stride
    return resampled[..., skip : skip + emit], buffer[..., -context:]


def streaming_resample_batch(
    chunks: list[Tensor], tails: list[Tensor | None], resampler: Resample,
) -> tuple[list[Tensor], list[Tensor]]:
    """Share resampling launches across equal-sized chunks, not audio histories."""
    groups = defaultdict(list)
    for index, (chunk, tail) in enumerate(zip(chunks, tails, strict=True)):
        groups[(chunk.shape[-1], 0 if tail is None else tail.shape[-1])].append(index)
    outputs = {}
    updated = {}
    for (_, tail_size), indices in groups.items():
        tail = torch.stack([tails[index] for index in indices]) if tail_size else None
        batch, new_tail = streaming_resample_chunk(
            torch.stack([chunks[index] for index in indices]), tail, resampler,
        )
        for row, index in enumerate(indices):
            outputs[index] = batch[row]
            updated[index] = new_tail[row]
    return [outputs[index] for index in range(len(chunks))], [updated[index] for index in range(len(chunks))]


class StreamingChunkSizes(NamedTuple):
    """Fixed streaming window sizes, in raw samples and mel frames."""

    first_samples: int
    samples: int
    first_mel_frames: int
    mel_frames: int


class FastConformerGraph:
    """Replay one full-window encoder batch without retaining request-owned state.

    Inputs and outputs are packed: caches come in with one copy per run of rows
    and leave, together with the hidden states, as one flat clone. Graphs of
    every batch size may share one memory ``pool``: replays run one at a time on
    one stream, and only each graph's packed output outlives its replay. They
    also share a capture ``stream`` of their own: cuBLAS keys its workspace by
    stream, and graphs captured on torch's default capture stream all bake in
    one workspace, which a replay beside other graphs would race on.
    """

    def __init__(
        self,
        encode: Callable[[Tensor, FastConformerStreamState], tuple[Tensor, FastConformerStreamState]],
        features: Tensor,
        state: FastConformerStreamState,
        pool: tuple[int, int] | None = None,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        self.features = features.clone()
        self.layout = FlatCacheLayout(state)
        self.state = FastConformerStreamState.packed(self.layout, self.layout.pack(state.tensors()))

        def run() -> tuple[Tensor, FastConformerStreamState]:
            # The encoder rewrites its padding caches in place, so those get private
            # copies; the key and value windows, the bulk of the state, are only read.
            tensors = self.layout.views(self.state.flat)
            count = self.layout.kv_tensors
            private = [*tensors[:count], *(tensor.clone() for tensor in tensors[count:])]
            return encode(self.features, FastConformerStreamState(*self.layout.containers(private)))

        self.stream = stream or torch.cuda.Stream(device=features.device)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(self.stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool, stream=self.stream):
            hidden, output_state = run()
            self.output = FlatCacheLayout.pack([hidden, *output_state.tensors()])
        output_layout = FlatCacheLayout(output_state)
        # Steady windows keep their shape, so outputs feed the next replay row-wise.
        self.output_layout = self.layout if output_layout.key == self.layout.key else output_layout
        self.hidden_shape = hidden.shape
        self.hidden_size = hidden[0].numel()

    def __call__(
        self, features: Tensor, states: list[FastConformerStreamState],
    ) -> tuple[Tensor, FastConformerStreamState]:
        # Rows past the batch replay whatever an earlier batch left there.
        rows = features.shape[0]
        self.features[:rows].copy_(features)
        FastConformerStreamState.copy_rows(states, self.state)
        self.graph.replay()
        # Both features and caches must survive subsequent replays for other requests.
        flat = self.output[:rows].clone()
        return (
            flat[:, :self.hidden_size].view(rows, *self.hidden_shape[1:]),
            FastConformerStreamState.packed(self.output_layout, flat[:, self.hidden_size:]),
        )


class FastConformerEncoder(nn.Module):
    """The frozen, cache-aware FastConformer encoder of a streaming RNN-T; the model reads its hidden states."""

    encoder: Any
    processor: Any
    mel_filters: Tensor
    stft_window: Tensor

    @classmethod
    def from_export(
        cls, config: dict[str, Any], root: Path, *, use_cuda_graph: bool = False,
    ) -> FastConformerEncoder:
        """Construct locally; the native model loader supplies all weights."""
        from transformers import AutoConfig, AutoProcessor
        from transformers.models.nemotron_asr_streaming.modeling_nemotron_asr_streaming import (
            NemotronAsrStreamingEncoder,
        )

        values = dict(config)
        encoder = NemotronAsrStreamingEncoder(AutoConfig.for_model(values.pop("model_type"), **values).encoder_config)
        processor = AutoProcessor.from_pretrained(root / "user_asr", local_files_only=True)
        return cls(encoder, processor, use_cuda_graph=use_cuda_graph)

    def __init__(self, encoder: Any, processor: Any, *, use_cuda_graph: bool = False) -> None:
        super().__init__()
        self.encoder = encoder
        self.processor = processor
        self.use_cuda_graph = use_cuda_graph
        self.graphs: dict[int, FastConformerGraph] = {}
        self.processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
        self.output_dim = encoder.config.hidden_size
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        feature_extractor = processor.feature_extractor
        self.feature_hop_length = feature_extractor.hop_length
        self.feature_n_fft = feature_extractor.n_fft
        self.feature_win_length = feature_extractor.win_length
        self.feature_preemphasis = feature_extractor.preemphasis
        # Signal-processing constants stay FP32, independently of model precision.
        self.register_buffer("mel_filters", feature_extractor.mel_filters.float())
        self.register_buffer("stft_window", torch.hann_window(feature_extractor.win_length, periodic=False, dtype=torch.float32))

    @cached_property
    def chunk_sizes(self) -> StreamingChunkSizes:
        # The processor re-merges its kwargs on every access; the lookahead is fixed.
        processor = self.processor
        return StreamingChunkSizes(
            processor.num_samples_first_audio_chunk, processor.num_samples_per_audio_chunk,
            processor.num_mel_frames_first_audio_chunk, processor.num_mel_frames_per_audio_chunk,
        )

    def encode_feature_chunk(
        self, input_features: Tensor, state: FastConformerStreamState
    ) -> tuple[Tensor, FastConformerStreamState]:
        """Advance the native cache-aware encoder by one exact feature chunk."""
        output = self.encoder(
            input_features=input_features,
            past_key_values=state.past_key_values,
            padding_cache=state.padding_cache,
            num_lookahead_tokens=NUM_LOOKAHEAD_TOKENS,
            use_cache=True,
            output_attention_mask=False,
        )
        return (
            output.last_hidden_state,
            FastConformerStreamState(
                past_key_values=output.past_key_values,
                padding_cache=output.padding_cache,
            ),
        )

    def prepare_streaming_audio_chunk(self, waveform: Tensor, *, first: bool) -> Tensor:
        """Extract exact streaming mels from one window or a batch of equal windows."""
        mel_filters, window = self.mel_filters, self.stft_window
        waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform
        with torch.autocast(waveform.device.type, enabled=False):
            if self.feature_preemphasis is not None:
                waveform = torch.cat((
                    waveform[:, :1], waveform[:, 1:] - self.feature_preemphasis * waveform[:, :-1],
                ), dim=-1)
            spectrum = torch.stft(
                waveform, self.feature_n_fft, hop_length=self.feature_hop_length,
                win_length=self.feature_win_length, window=window,
                return_complex=True, pad_mode="constant", center=first,
            )
            # Preserve the upstream sqrt-then-square rounding, not abs().square().
            magnitudes = torch.view_as_real(spectrum).pow(2).sum(-1).sqrt().pow(2)
            features = (mel_filters @ magnitudes + 2**-24).log().transpose(1, 2)
        required_frames = (
            self.chunk_sizes.first_mel_frames
            if first
            else self.chunk_sizes.mel_frames
        )
        actual_frames = features.shape[1]
        if actual_frames < required_frames or (
            not first and actual_frames != required_frames
        ):
            raise ValueError(
                f"Streaming audio chunk produced {actual_frames} mel frames; expected {required_frames}"
            )
        model_param = next(self.encoder.parameters())
        return features[:, :required_frames].to(
            device=model_param.device, dtype=model_param.dtype
        )

    def take_audio_windows(
        self, waveform: Tensor, state: FastConformerAudioStreamState
    ) -> tuple[list[Tensor], FastConformerAudioStreamState]:
        """Collect complete STFT windows, retaining only the unfinished tail."""
        if waveform.ndim != 1:
            raise ValueError(
                f"Live FastConformer input must be one waveform chunk, got shape {tuple(waveform.shape)}"
            )
        if state.audio_buffer is None:
            audio_buffer = waveform
        else:
            assert state.audio_buffer.device == waveform.device
            assert state.audio_buffer.dtype == waveform.dtype
            audio_buffer = torch.cat((state.audio_buffer, waveform))
        buffer_start = state.buffer_start_sample
        next_mel_frame = state.next_mel_frame
        raw_chunks = []
        while True:
            first = next_mel_frame == 0
            if first:
                raw_start = 0
                raw_size = self.chunk_sizes.first_samples
            else:
                raw_start = (
                    next_mel_frame * self.feature_hop_length
                    - self.feature_n_fft // 2
                )
                raw_size = self.chunk_sizes.samples
            raw_end = raw_start + raw_size
            available_end = buffer_start + audio_buffer.shape[0]
            if available_end < raw_end:
                break
            valid_start = max(raw_start, 0)
            local_start = valid_start - buffer_start
            local_end = raw_end - buffer_start
            raw_chunk = audio_buffer[local_start:local_end]
            if raw_start < 0:
                raw_chunk = torch.cat((raw_chunk.new_zeros(-raw_start), raw_chunk))
            raw_chunks.append(raw_chunk)
            next_mel_frame += (
                self.chunk_sizes.first_mel_frames
                if first else self.chunk_sizes.mel_frames
            )
            next_raw_start = max(
                next_mel_frame * self.feature_hop_length
                - self.feature_n_fft // 2,
                0,
            )
            drop = next_raw_start - buffer_start
            audio_buffer = audio_buffer[drop:]
            buffer_start = next_raw_start
        return raw_chunks, replace(
            state, audio_buffer=audio_buffer, buffer_start_sample=buffer_start,
            next_mel_frame=next_mel_frame,
        )

    def encode_audio_batch(
        self, waveforms: list[Tensor], states: list[FastConformerAudioStreamState],
    ) -> tuple[list[Tensor], list[FastConformerAudioStreamState]]:
        """Batch frontends and equal-sized encoder windows across live streams."""
        windows = []
        updated = []
        for waveform, state in zip(waveforms, states, strict=True):
            raw, new_state = self.take_audio_windows(waveform, state)
            windows.append(raw)
            updated.append(new_state)
        features: list[list[Tensor]] = [[] for _ in states]
        for first in (True, False):
            selected = [
                (index, raw)
                for index, chunks in enumerate(windows)
                for chunk_index, raw in enumerate(chunks)
                if (states[index].next_mel_frame == 0 and chunk_index == 0) == first
            ]
            if selected:
                with torch.profiler.record_function("duplexio.asr_frontend"):
                    batch = self.prepare_streaming_audio_chunk(torch.stack([raw for _, raw in selected]), first=first)
                for (index, _), feature in zip(selected, batch, strict=True):
                    features[index].append(feature)
        groups = defaultdict(list)
        inputs = {}
        for index, chunks in enumerate(features):
            if chunks:
                inputs[index] = torch.cat(chunks)
                groups[(inputs[index].shape[0], states[index].encoder.cached_frames)].append(index)
        parameter = next(self.encoder.parameters())
        outputs = [parameter.new_empty((1, 0, self.output_dim)) for _ in states]
        for indices in groups.values():
            previous = [states[index].encoder for index in indices]
            with torch.profiler.record_function("duplexio.asr_encoder"):
                features = torch.stack([inputs[index] for index in indices])
                steady = (
                    previous[0].cached_frames == self.encoder.config.sliding_window - 1
                    and features.shape[1] == self.chunk_sizes.mel_frames
                )
                if self.use_cuda_graph and steady:
                    batch_size = graph_batch_size(len(indices))
                    if batch_size not in self.graphs:
                        # A pool lives only as long as a graph using it.
                        first = next(iter(self.graphs.values()), None)
                        padding = batch_size - len(indices)
                        self.graphs[batch_size] = FastConformerGraph(
                            partial(steady_encode, self),
                            torch.cat([features, features[-1:].expand(padding, -1, -1)]),
                            FastConformerStreamState.stack(previous + previous[-1:] * padding),
                            first and first.graph.pool(), first and first.stream,
                        )
                    encoded, cache = self.graphs[batch_size](features, previous)
                else:
                    with torch.profiler.record_function("duplexio.asr_cache_pack"):
                        cache = FastConformerStreamState.stack(previous)
                    encoded, cache = self.encode_feature_chunk(features, cache)
            with torch.profiler.record_function("duplexio.asr_cache_split"):
                caches = cache.unbind(previous, encoded.shape[1])
            for row, (index, encoder_state) in enumerate(zip(indices, caches, strict=True)):
                outputs[index] = encoded[row:row + 1]
                updated[index].encoder = encoder_state
        return outputs, updated

    @torch.inference_mode()
    def capture_graphs(self) -> None:
        """Compile and capture every steady batch size before serving starts.

        Otherwise the first steady window of each size would stall all live
        streams while it compiles, and allocate graph memory that vLLM's memory
        profiling never saw.
        """
        if not self.use_cuda_graph or self.graphs:
            return
        streams = GRAPH_BATCH_SIZES[-1]
        states = [FastConformerAudioStreamState() for _ in range(streams)]
        silence = [torch.zeros(FRAME_SAMPLES, device=self.mel_filters.device) for _ in range(streams)]
        self.use_cuda_graph = False
        try:
            while states[0].encoder.cached_frames < self.encoder.config.sliding_window - 1:
                _, states = self.encode_audio_batch(silence, states)
        finally:
            self.use_cuda_graph = True
        # Size one specializes; the second size compiles once for all the rest.
        for size in GRAPH_BATCH_SIZES:
            _, states[:size] = self.encode_audio_batch(silence[:size], states[:size])
        assert set(self.graphs) == set(GRAPH_BATCH_SIZES)

# Only captured steady windows compile: their shapes differ in batch alone, and
# fusing the encoder's elementwise work shortens every replay.
steady_encode = torch.compile(FastConformerEncoder.encode_feature_chunk, fullgraph=True)
