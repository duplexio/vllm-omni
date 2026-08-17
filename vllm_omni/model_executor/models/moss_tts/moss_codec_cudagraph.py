"""CUDA Graph acceleration for the MOSS Audio Tokenizer codec decoder.

Captures MossAudioTokenizerModel._decode for a set of fixed frame-count
bucket sizes, then replays the captured graph at inference time to eliminate
kernel-launch overhead.  Inputs that exceed all captured sizes fall back to
eager execution transparently.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerDecoderOutput,
    MossAudioTokenizerModel,
)

logger = init_logger(__name__)


class MossTTSCUDAGraphCodecWrapper:
    """CUDA Graph wrapper for MossAudioTokenizerModel._decode.

    Graphs are keyed by ``(batch_size, padded_T)``.  On each call the actual
    request count and frame count are bucket-matched to the smallest
    pre-captured sizes.  The static code buffer ``[NQ, B, padded_T]`` is filled
    left-aligned (right-zero-padded) and the graph is replayed.  The output
    audio is sliced to each request's correct length by scaling from the
    captured audio shape (``actual_T / padded_T * captured_len``), avoiding
    any assumption about downsample_rate vs effective decoder upsample.  Each
    slice is cloned before returning so the static buffer can be reused.

    Usage::

        wrapper = MossTTSCUDAGraphCodecWrapper(codec_model, capture_sizes, nq)
        wrapper.warmup(device)

        # batched decode:
        outputs = wrapper.decode_batch(codes_list)  # each code tensor: [NQ, T]

        # single-request compatibility:
        out = wrapper.decode(codes_nq_t)
    """

    def __init__(
        self,
        model: MossAudioTokenizerModel,
        capture_sizes: list[int],
        num_quantizers: int,
        capture_batch_sizes: list[int] | None = None,
        enabled: bool = True,
    ) -> None:
        self.model = model
        self.capture_sizes: list[int] = sorted(capture_sizes)
        self.num_quantizers = num_quantizers
        self.capture_batch_sizes: list[int] = sorted(set(capture_batch_sizes or [1]))
        self.enabled = enabled

        # All dictionaries are keyed by (batch_size, padded_T).
        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_codes: dict[tuple[int, int], torch.Tensor] = {}
        # static_lengths is kept alive here — the captured graph holds a
        # reference to the underlying storage and must not be GC'd.
        self.static_lengths: dict[tuple[int, int], torch.Tensor] = {}
        self.static_audio: dict[tuple[int, int], torch.Tensor] = {}

        # v1 calls this method _decode while v2 calls it _decode_frame.  The
        # tensor-only signatures are identical, so resolve the variant once at
        # construction instead of branching in every replay.
        decode = getattr(model, "_decode", None)
        if decode is None:
            decode = model._decode_frame
        self._decode_tensor: Callable[[torch.Tensor, torch.Tensor], Any] = decode

        self._warmed_up = False

    # ------------------------------------------------------------------
    # Size helpers
    # ------------------------------------------------------------------

    def _get_padded_size(self, actual_t: int) -> int | None:
        """Return the smallest capture size >= actual_t, or None if too large."""
        for s in self.capture_sizes:
            if actual_t <= s:
                return s
        return None

    def _get_padded_batch_size(self, actual_batch: int) -> int | None:
        """Return the smallest captured batch size that fits ``actual_batch``."""
        for size in self.capture_batch_sizes:
            if actual_batch <= size:
                return size
        return None

    # ------------------------------------------------------------------
    # Warmup / capture
    # ------------------------------------------------------------------

    def warmup(self, device: torch.device) -> None:
        """Allocate static buffers and capture CUDA Graphs for all sizes."""
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        nq = self.num_quantizers
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup: nq=%d capture_batch_sizes=%s capture_sizes=%s",
            nq,
            self.capture_batch_sizes,
            self.capture_sizes,
        )
        t0 = time.perf_counter()

        # One eager run per size to let cuDNN / CUDA allocate memory before
        # the capture window (graph capture forbids new CUDA allocs during it).
        for batch_size in self.capture_batch_sizes:
            for size in self.capture_sizes:
                dummy_codes = torch.zeros(nq, batch_size, size, dtype=torch.long, device=device)
                dummy_lengths = torch.full((batch_size,), size, dtype=torch.long, device=device)
                with torch.no_grad():
                    _ = self._decode_tensor(dummy_codes, dummy_lengths)

        torch.accelerator.synchronize(device)

        for batch_size in self.capture_batch_sizes:
            for size in self.capture_sizes:
                try:
                    self._capture(batch_size, size, device)
                    logger.info("  Captured CUDA Graph for batch_size=%d size=%d", batch_size, size)
                except Exception:
                    logger.warning(
                        "  Failed to capture CUDA Graph for batch_size=%d size=%d; "
                        "this pair will fall back to eager decode",
                        batch_size,
                        size,
                        exc_info=True,
                    )

        self._warmed_up = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(
            "MOSS-TTS codec CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(self.capture_sizes) * len(self.capture_batch_sizes),
            elapsed_ms,
        )

    def _capture(self, batch_size: int, size: int, device: torch.device) -> None:
        nq = self.num_quantizers
        static_codes = torch.zeros(nq, batch_size, size, dtype=torch.long, device=device)
        # lengths holds the number of valid code frames; set to full size at
        # capture time so the decoder emits a full-size audio buffer for every
        # graph lane.  Runtime requests are padded to this frame count and
        # trimmed after replay; inactive lanes are ignored by the caller.
        static_lengths = torch.full((batch_size,), size, dtype=torch.long, device=device)

        # Extra eager warmup inside capture to ensure all kernels are compiled.
        with torch.no_grad():
            _ = self._decode_tensor(static_codes, static_lengths)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_out = self._decode_tensor(static_codes, static_lengths)

        key = (batch_size, size)
        self.graphs[key] = graph
        self.static_codes[key] = static_codes
        self.static_lengths[key] = static_lengths
        # static_out.audio is a static buffer reused every replay; hold a
        # reference so it is not garbage-collected.
        self.static_audio[key] = static_out.audio  # [B, C, size * effective_upsample]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(self, codes_nq_t: torch.Tensor) -> MossAudioTokenizerDecoderOutput:
        """Decode [NQ, T] codes to waveform using a CUDA Graph when possible.

        Falls back to eager batch_decode when:
          - CUDA Graph is disabled or not yet warmed up
          - an outer CUDA stream capture is active (e.g. vLLM FULL graph mode)
          - actual T exceeds all pre-captured sizes
        """
        return self.decode_batch([codes_nq_t])[0]

    def _eager_decode_batch(
        self,
        codes_list: list[torch.Tensor],
    ) -> list[MossAudioTokenizerDecoderOutput]:
        """Decode a variable-length batch eagerly and split it per request."""
        result = self.model.batch_decode(codes_list=codes_list, num_quantizers=self.num_quantizers)
        if result.audio is None:
            return [MossAudioTokenizerDecoderOutput(audio=None, audio_lengths=None) for _ in codes_list]

        outputs: list[MossAudioTokenizerDecoderOutput] = []
        for index, _ in enumerate(codes_list):
            audio = result.audio[index : index + 1]
            if result.audio_lengths is None:
                length = audio.shape[-1]
                lengths = None
            else:
                length = int(result.audio_lengths[index].item())
                audio = audio[..., :length]
                lengths = torch.tensor([length], dtype=torch.long, device=audio.device)
            outputs.append(MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=lengths))
        return outputs

    @torch.no_grad()
    def decode_batch(
        self,
        codes_list: list[torch.Tensor],
    ) -> list[MossAudioTokenizerDecoderOutput]:
        """Decode variable-length requests using bucketed batched graph replay.

        The scheduler remains free to submit and complete requests
        independently.  Only requests present in this call are grouped into
        a padded frame bucket and a captured batch-size bucket.  The graph
        computes all lanes together; host-side request splitting happens after
        replay.
        """
        if not codes_list:
            return []
        if not self.enabled or not self._warmed_up or torch.cuda.is_current_stream_capturing():
            return self._eager_decode_batch(codes_list)

        outputs: list[MossAudioTokenizerDecoderOutput | None] = [None] * len(codes_list)
        bucketed: dict[int, list[tuple[int, torch.Tensor]]] = {}
        eager_indices: list[int] = []

        for index, codes in enumerate(codes_list):
            padded_size = self._get_padded_size(int(codes.shape[-1]))
            if padded_size is None:
                eager_indices.append(index)
            else:
                bucketed.setdefault(padded_size, []).append((index, codes))

        for padded_size, items in bucketed.items():
            # If the ready set is larger than the largest captured graph, use
            # one eager tensor batch instead of serialising it into smaller
            # graph replays.  This is the normal high-throughput path when the
            # safe graph configuration captures only batch size 1.
            if len(items) > self.capture_batch_sizes[-1]:
                eager_indices.extend(index for index, _ in items)
                continue

            start = 0
            while start < len(items):
                batch_size = self._get_padded_batch_size(len(items) - start)
                if batch_size is None:
                    eager_indices.extend(index for index, _ in items[start:])
                    break

                chunk = items[start : start + batch_size]
                key = (batch_size, padded_size)
                if key not in self.graphs:
                    eager_indices.extend(index for index, _ in chunk)
                    start += len(chunk)
                    continue

                static_codes = self.static_codes[key]
                static_codes.zero_()
                for lane, (_, codes) in enumerate(chunk):
                    static_codes[:, lane, : codes.shape[-1]].copy_(codes)
                self.graphs[key].replay()

                static_audio = self.static_audio[key]
                captured_len = static_audio.shape[-1]
                for lane, (index, codes) in enumerate(chunk):
                    actual_t = int(codes.shape[-1])
                    actual_wav_len = captured_len * actual_t // padded_size
                    audio = static_audio[lane : lane + 1, ..., :actual_wav_len].clone()
                    lengths = torch.tensor([actual_wav_len], dtype=torch.long, device=audio.device)
                    outputs[index] = MossAudioTokenizerDecoderOutput(audio=audio, audio_lengths=lengths)
                start += len(chunk)

        if eager_indices:
            eager_codes = [codes_list[index] for index in eager_indices]
            eager_outputs = self._eager_decode_batch(eager_codes)
            for index, output in zip(eager_indices, eager_outputs, strict=True):
                outputs[index] = output

        if any(output is None for output in outputs):
            raise RuntimeError("MOSS codec batch decode did not produce an output for every request.")
        return [output for output in outputs if output is not None]


__all__ = ["MossTTSCUDAGraphCodecWrapper"]
