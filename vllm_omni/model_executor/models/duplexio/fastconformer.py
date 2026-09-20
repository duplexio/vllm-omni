"""Cache-aware FastConformer and RNN-T from the exported checkpoint."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torchaudio import functional as AF

SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1_280
NUM_LOOKAHEAD_TOKENS = 0
# The RNN-T's sub-word vocabulary marks the first piece of every word.
WORD_START = "\u2581"
# A word is normally released when the next one starts, but the last word before
# a pause has no successor — and is the one the agent needs to take its turn, so
# silence ends it instead. Measured over 6881 aligned words (300 Emilia clips):
# waiting longer splits fewer words across two releases, which matters because a
# split re-encodes as Qwen ids the backbone never saw for that word (1.5% of
# words at 4 frames, 3.9% at 3, 18.5% at 1), but it also delays the whole stream.
# Training placed a word at word_end + first-piece emission latency (p50 3
# frames, support ≤13); a streaming decoder cannot beat last-piece + this wait,
# which is p50 8 / p90 12 frames here — inside that support, where 8 frames
# (p50 12 / p90 16) left half the words past anything training showed.
WORD_END_SILENCE_FRAMES = 4


@dataclass
class FastConformerStreamState:
    """Explicit upstream encoder caches for one streaming utterance."""

    past_key_values: Any = None
    padding_cache: Any = None

    @property
    def cached_frames(self) -> int:
        return 0 if self.past_key_values is None else self.past_key_values.layers[0].keys.shape[-2]

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
        results = []
        for index, old in enumerate(previous):
            kv = copy.copy(self.past_key_values)
            kv.layers = []
            length = 0 if old.past_key_values is None else old.past_key_values.get_seq_length()
            for layer in self.past_key_values.layers:
                single = copy.copy(layer)
                single.keys = layer.keys[index:index + 1]
                single.values = layer.values[index:index + 1]
                single.cumulative_length = length + frames
                kv.layers.append(single)
            padding = copy.copy(self.padding_cache)
            padding.layers = {}
            for name, layer in self.padding_cache.layers.items():
                single = copy.copy(layer)
                single.cache = layer.cache[index:index + 1]
                padding.layers[name] = single
            results.append(FastConformerStreamState(kv, padding))
        return results


@dataclass
class RNNTGreedyState:
    """Resumable greedy RNN-T decode state for one live utterance."""

    # Prediction-network LSTM state, and its output for the emitted prefix.
    hidden: Tensor | None = None
    cell: Tensor | None = None
    prediction: Tensor | None = None
    # Sub-word ids of the word still being spoken. A word is released only once
    # the next one starts, so text the model has already seen never changes.
    word_token_ids: tuple[int, ...] = ()
    # Encoder frames since the last emitted sub-word.
    silent_frames: int = 0


@dataclass
class FastConformerAudioStreamState:
    """Raw-audio frontend and encoder state for one live utterance."""

    encoder: FastConformerStreamState = field(
        default_factory=FastConformerStreamState,
    )
    rnnt: RNNTGreedyState = field(default_factory=RNNTGreedyState)
    audio_buffer: Tensor | None = None
    buffer_start_sample: int = 0
    next_mel_frame: int = 0
    # Source-rate samples preceding the next chunk, so chunked resampling
    # matches the whole-signal resample (see streaming_resample_chunk).
    resample_tail: Tensor | None = None


def streaming_resample_chunk(
    chunk: Tensor,
    tail: Tensor | None,
    orig_freq: int,
    new_freq: int,
) -> tuple[Tensor, Tensor]:
    """Resample one source-rate chunk with left context from earlier audio.

    Emits exactly ``len(chunk) * new / orig`` samples, phase-locked to the
    whole-signal resample: every emitted sample matches it bit-for-bit except
    the final sinc half-width of each push, which sees zeros in place of the
    not-yet-received future samples (the same zero padding the whole-signal
    resample applies at the true end of the audio).
    """
    orig_stride = orig_freq // math.gcd(orig_freq, new_freq)
    new_stride = new_freq // math.gcd(orig_freq, new_freq)
    if chunk.ndim not in (1, 2) or chunk.shape[-1] % orig_stride:
        raise ValueError(
            f"Streaming resample chunks must be 1-D or batched multiples of {orig_stride} "
            f"source samples, got shape {tuple(chunk.shape)}"
        )
    # Comfortably beyond torchaudio's default sinc half-width, and a stride
    # multiple so the polyphase output stays on the whole-signal grid.
    context = 16 * orig_stride
    buffer = chunk if tail is None else torch.cat((tail, chunk), dim=-1)
    resampled = AF.resample(buffer, orig_freq, new_freq)
    skip = (buffer.shape[-1] - chunk.shape[-1]) * new_stride // orig_stride
    emit = chunk.shape[-1] * new_stride // orig_stride
    return resampled[..., skip : skip + emit], buffer[..., -context:]


def streaming_resample_batch(
    chunks: list[Tensor], tails: list[Tensor | None], orig_freq: int, new_freq: int,
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
            torch.stack([chunks[index] for index in indices]), tail, orig_freq, new_freq,
        )
        for row, index in enumerate(indices):
            outputs[index] = batch[row]
            updated[index] = new_tail[row]
    return [outputs[index] for index in range(len(chunks))], [updated[index] for index in range(len(chunks))]


class FastConformerRNNT(nn.Module):
    """Own the pretrained FastConformer encoder and its native RNN-T.

    The conversational model consumes ``last_hidden_state``. Transcripts are
    decoded separately by the checkpoint's prediction and joint networks.
    """

    model: Any
    processor: Any
    mel_filters: Tensor
    stft_window: Tensor

    @classmethod
    def from_export(cls, config: dict[str, Any], root: Path) -> FastConformerRNNT:
        """Construct locally; the native model loader supplies all weights."""
        from transformers import AutoConfig, AutoModelForRNNT, AutoProcessor

        values = dict(config)
        model_type = values.pop("model_type")
        model = AutoModelForRNNT.from_config(AutoConfig.for_model(model_type, **values))
        processor = AutoProcessor.from_pretrained(
            root / "user_asr", local_files_only=True
        )
        return cls(model, processor)

    def __init__(self, model: Any, processor: Any) -> None:
        super().__init__()
        self.model = model
        self.processor = processor
        self.processor.set_num_lookahead_tokens(NUM_LOOKAHEAD_TOKENS)
        config = model.config
        self.output_dim = config.encoder_config.hidden_size
        self.blank_token_id = config.blank_token_id
        self.model.requires_grad_(False)
        self.model.eval()
        feature_extractor = processor.feature_extractor
        self.feature_hop_length = feature_extractor.hop_length
        self.feature_n_fft = feature_extractor.n_fft
        self.feature_win_length = feature_extractor.win_length
        self.feature_preemphasis = feature_extractor.preemphasis
        self.register_buffer("mel_filters", feature_extractor.mel_filters)
        self.register_buffer(
            "stft_window",
            torch.hann_window(feature_extractor.win_length, periodic=False),
        )

    def train(self, mode: bool = True) -> FastConformerRNNT:
        """Keep a frozen ASR subsystem deterministic under parent ``train()``."""
        super().train(False)
        return self

    def encode_feature_chunk(
        self, input_features: Tensor, state: FastConformerStreamState
    ) -> tuple[Tensor, FastConformerStreamState]:
        """Advance the native cache-aware encoder by one exact feature chunk."""
        output = self.model.get_audio_features(
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

    def streaming_raw_audio_slice(
        self, waveform: Tensor, start: int, size: int
    ) -> Tensor:
        """Slice one STFT-aligned raw chunk, padding only utterance boundaries."""
        end = start + size
        valid_start = max(start, 0)
        valid_end = min(end, waveform.shape[0])
        parts = []
        if start < 0:
            parts.append(waveform.new_zeros(-start))
        parts.append(waveform[valid_start:valid_end])
        if end > waveform.shape[0]:
            parts.append(waveform.new_zeros(end - waveform.shape[0]))
        return torch.cat(parts)

    def prepare_streaming_audio_chunk(self, waveform: Tensor, *, first: bool) -> Tensor:
        """Extract exact streaming mels from one window or a batch of equal windows."""
        inputs = self.processor(
            waveform.unbind(0) if waveform.ndim == 2 else waveform,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            return_tensors="pt",
            device=str(waveform.device),
        )
        required_frames = (
            self.processor.num_mel_frames_first_audio_chunk
            if first
            else self.processor.num_mel_frames_per_audio_chunk
        )
        actual_frames = inputs.input_features.shape[1]
        if actual_frames < required_frames or (
            not first and actual_frames != required_frames
        ):
            raise ValueError(
                f"Streaming audio chunk produced {actual_frames} mel frames; expected {required_frames}"
            )
        model_param = next(self.model.parameters())
        return inputs.input_features[:, :required_frames].to(
            device=model_param.device, dtype=model_param.dtype
        )

    def streaming_feature_chunks(self, waveform: Tensor) -> Iterable[Tensor]:
        """Yield exact upstream mel chunks for a complete 16 kHz waveform."""
        if waveform.ndim != 1:
            raise ValueError(
                f"Streaming RNN-T expects one waveform, got shape {tuple(waveform.shape)}"
            )
        if waveform.shape[0] % FRAME_SAMPLES:
            raise ValueError(
                f"Streaming RNN-T expects complete 80 ms frames, got {waveform.shape[0]} samples"
            )
        frame_count = waveform.shape[0] // FRAME_SAMPLES
        if frame_count == 0:
            raise ValueError("Streaming RNN-T requires at least one 80 ms frame")
        mel_frame = 0
        for frame in range(frame_count):
            first = frame == 0
            if first:
                start = 0
                size = self.processor.num_samples_first_audio_chunk
            else:
                start = (
                    mel_frame * self.processor.feature_extractor.hop_length
                    - self.processor.feature_extractor.n_fft // 2
                )
                size = self.processor.num_samples_per_audio_chunk
            raw_chunk = self.streaming_raw_audio_slice(waveform, start, size)
            features = self.prepare_streaming_audio_chunk(raw_chunk, first=first)
            yield features
            mel_frame += features.shape[1]

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
                raw_size = self.processor.num_samples_first_audio_chunk
            else:
                raw_start = (
                    next_mel_frame * self.processor.feature_extractor.hop_length
                    - self.processor.feature_extractor.n_fft // 2
                )
                raw_size = self.processor.num_samples_per_audio_chunk
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
                self.processor.num_mel_frames_first_audio_chunk
                if first else self.processor.num_mel_frames_per_audio_chunk
            )
            next_raw_start = max(
                next_mel_frame * self.processor.feature_extractor.hop_length
                - self.processor.feature_extractor.n_fft // 2,
                0,
            )
            drop = next_raw_start - buffer_start
            audio_buffer = audio_buffer[drop:]
            buffer_start = next_raw_start
        return raw_chunks, replace(
            state, audio_buffer=audio_buffer, buffer_start_sample=buffer_start,
            next_mel_frame=next_mel_frame,
        )

    def encode_audio_chunk(
        self, waveform: Tensor, state: FastConformerAudioStreamState,
    ) -> tuple[Tensor, FastConformerAudioStreamState]:
        """Consume available live 16 kHz audio and emit complete encoder states."""
        outputs, states = self.encode_audio_batch([waveform], [state])
        return outputs[0], states[0]

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
        parameter = next(self.model.parameters())
        outputs = [parameter.new_empty((1, 0, self.output_dim)) for _ in states]
        for indices in groups.values():
            previous = [states[index].encoder for index in indices]
            with torch.profiler.record_function("duplexio.asr_cache_pack"):
                cache = FastConformerStreamState.stack(previous)
            with torch.profiler.record_function("duplexio.asr_encoder"):
                encoded, cache = self.encode_feature_chunk(torch.stack([inputs[index] for index in indices]), cache)
            with torch.profiler.record_function("duplexio.asr_cache_split"):
                caches = cache.unbind(previous, encoded.shape[1])
            for row, (index, encoder_state) in enumerate(zip(indices, caches, strict=True)):
                outputs[index] = encoded[row:row + 1]
                updated[index].encoder = encoder_state
        return outputs, updated

    def prediction_step(
        self,
        token_id: int,
        hidden: Tensor | None,
        cell: Tensor | None,
        device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance the prediction network by one emitted token."""
        decoder = self.model.decoder
        embeddings = decoder.embedding(
            torch.tensor([[token_id]], dtype=torch.long, device=device)
        )
        output, (hidden, cell) = decoder.lstm(
            embeddings, None if hidden is None else (hidden, cell)
        )
        return decoder.decoder_projector(output), hidden, cell

    @torch.inference_mode()
    def decode_words(
        self, encoded: Tensor, state: RNNTGreedyState
    ) -> tuple[tuple[str, ...], RNNTGreedyState]:
        """Greedily transcribe encoder states, returning the words that finished.

        Each returned word carries the leading space its ``▁`` piece stands for,
        and a group that continues an already released word (punctuation after a
        pause) carries none, so the caller concatenates them as-is.

        The native greedy schedule: a blank advances one encoder frame, a symbol
        stays on the frame and advances the prediction network, and
        ``max_symbols_per_step`` symbols force an advance. Unlike ``generate``,
        which consumes a whole utterance, the LSTM state lives in ``state`` so a
        live session resumes on its next 80 ms append.
        """
        # The joint head consumes projected encoder states, while the
        # conversational model embeds the unprojected ones that
        # encode_audio_chunk returns; projecting again here keeps the transcript
        # decode independent of the feature path for one small matmul a frame.
        projected = self.model.encoder_projector(encoded)
        tokenizer = self.processor.tokenizer
        hidden, cell, prediction = state.hidden, state.cell, state.prediction
        word_token_ids = state.word_token_ids
        if prediction is None:
            # generate() seeds the prediction network with a single blank step.
            prediction, hidden, cell = self.prediction_step(
                self.blank_token_id, hidden, cell, encoded.device
            )
        def release(token_ids: tuple[int, ...]) -> str:
            lead = (
                " "
                if tokenizer.convert_ids_to_tokens(token_ids[0]).startswith(WORD_START)
                else ""
            )
            return lead + tokenizer.decode(list(token_ids))

        words: list[str] = []
        silent_frames = state.silent_frames
        for index in range(projected.shape[1]):
            frame = projected[:, index : index + 1, None, :]
            emitted = False
            for _ in range(self.model.max_symbols_per_step):
                logits = self.model.joint(
                    encoder_hidden_states=frame,
                    decoder_hidden_states=prediction[:, None],
                )
                token_id = int(logits.flatten().argmax())
                if token_id == self.blank_token_id:
                    break
                starts_word = tokenizer.convert_ids_to_tokens(token_id).startswith(
                    WORD_START
                )
                if starts_word and word_token_ids:
                    words.append(release(word_token_ids))
                    word_token_ids = ()
                prediction, hidden, cell = self.prediction_step(
                    token_id, hidden, cell, encoded.device
                )
                word_token_ids += (token_id,)
                emitted = True
            silent_frames = 0 if emitted else silent_frames + 1
            if word_token_ids and silent_frames >= WORD_END_SILENCE_FRAMES:
                words.append(release(word_token_ids))
                word_token_ids = ()
        return tuple(words), RNNTGreedyState(
            hidden, cell, prediction, word_token_ids, silent_frames
        )

    @torch.inference_mode()
    def transcribe_streaming_features(
        self,
        chunks: Iterable[Tensor],
        *,
        max_new_tokens: int,
        streamer: Any | None = None,
    ) -> list[str]:
        """Decode exact mel chunks through the native streaming RNN-T state."""
        feature_generator = (chunk for chunk in chunks)
        generated = self.model.generate(
            input_features=feature_generator,
            num_lookahead_tokens=NUM_LOOKAHEAD_TOKENS,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            return_dict_in_generate=True,
            # The export carries no generation config; the prediction network
            # starts from blank, exactly as decode_words seeds it.
            decoder_start_token_id=self.blank_token_id,
        )
        return self.processor.batch_decode(
            generated.sequences, skip_special_tokens=True
        )

    @torch.inference_mode()
    def transcribe_streaming_waveform(
        self, waveform: Tensor, *, streamer: Any | None = None
    ) -> str:
        """Decode one complete waveform through native streaming RNN-T state."""
        frame_count = waveform.shape[0] // FRAME_SAMPLES
        return self.transcribe_streaming_features(
            self.streaming_feature_chunks(waveform),
            max_new_tokens=frame_count * self.model.max_symbols_per_step,
            streamer=streamer,
        )[0]
