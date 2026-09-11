"""Cache-aware FastConformer and RNN-T from the exported checkpoint."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
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
    if chunk.ndim != 1 or chunk.shape[0] % orig_stride:
        raise ValueError(
            f"Streaming resample chunks must be 1-D multiples of {orig_stride} "
            f"source samples, got shape {tuple(chunk.shape)}"
        )
    # Comfortably beyond torchaudio's default sinc half-width, and a stride
    # multiple so the polyphase output stays on the whole-signal grid.
    context = 16 * orig_stride
    buffer = chunk if tail is None else torch.cat((tail, chunk))
    resampled = AF.resample(buffer, orig_freq, new_freq)
    skip = (buffer.shape[0] - chunk.shape[0]) * new_stride // orig_stride
    emit = chunk.shape[0] * new_stride // orig_stride
    return resampled[skip : skip + emit], buffer[-context:]


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
        """Extract one exact upstream streaming mel chunk from 16 kHz audio."""
        inputs = self.processor(
            waveform,
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

    def encode_audio_chunk(
        self, waveform: Tensor, state: FastConformerAudioStreamState
    ) -> tuple[Tensor, FastConformerAudioStreamState]:
        """Consume available live 16 kHz audio and emit complete encoder states."""
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
        encoder_state = state.encoder
        outputs = []
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
            features = self.prepare_streaming_audio_chunk(raw_chunk, first=first)
            encoded, encoder_state = self.encode_feature_chunk(features, encoder_state)
            outputs.append(encoded)
            next_mel_frame += features.shape[1]
            next_raw_start = max(
                next_mel_frame * self.processor.feature_extractor.hop_length
                - self.processor.feature_extractor.n_fft // 2,
                0,
            )
            drop = next_raw_start - buffer_start
            audio_buffer = audio_buffer[drop:]
            buffer_start = next_raw_start
        if outputs:
            states = torch.cat(outputs, dim=1)
        else:
            model_param = next(self.model.parameters())
            states = model_param.new_empty((1, 0, self.output_dim))
        return (
            states,
            FastConformerAudioStreamState(
                encoder=encoder_state,
                rnnt=state.rnnt,
                audio_buffer=audio_buffer,
                buffer_start_sample=buffer_start,
                next_mel_frame=next_mel_frame,
            ),
        )

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
