"""Causal multi-frame encoding must preserve streaming outputs and state."""

import pytest
import torch
from tokenizers import Tokenizer, models
from transformers import (
    AutoModelForRNNT,
    NemotronAsrStreamingConfig,
    NemotronAsrStreamingEncoderConfig,
    NemotronAsrStreamingFeatureExtractor,
    NemotronAsrStreamingProcessor,
    PreTrainedTokenizerFast,
)

from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FRAME_SAMPLES,
    FastConformerAudioStreamState,
    FastConformerRNNT,
)


@pytest.fixture
def encoder() -> FastConformerRNNT:
    torch.manual_seed(9)
    config = NemotronAsrStreamingConfig(
        encoder_config=NemotronAsrStreamingEncoderConfig(
            hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
            intermediate_size=32, subsampling_conv_channels=4, num_mel_bins=8,
            sliding_window=7,
        ),
        vocab_size=2, blank_token_id=0, decoder_hidden_size=8, num_decoder_layers=1,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(models.WordLevel({"<blank>": 0, "<unk>": 1}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    processor = NemotronAsrStreamingProcessor(NemotronAsrStreamingFeatureExtractor(feature_size=8), tokenizer)
    return FastConformerRNNT(AutoModelForRNNT.from_config(config), processor).eval()


@pytest.mark.parametrize("frames", [1, 4, 16])
@torch.inference_mode()
def test_parallel_prefill_matches_streaming_and_continuation(
    encoder: FastConformerRNNT, frames: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    waveform = torch.randn(frames * FRAME_SAMPLES) * 0.1
    serial_state = FastConformerAudioStreamState()
    serial = []
    for frame in waveform.split(FRAME_SAMPLES):
        output, serial_state = encoder.encode_audio_chunk(frame, serial_state)
        serial.append(output)
    calls = []
    frontend_batches = []
    prepare = encoder.prepare_streaming_audio_chunk

    def record_frontend(waveform, *, first):
        frontend_batches.append(1 if waveform.ndim == 1 else waveform.shape[0])
        return prepare(waveform, first=first)

    monkeypatch.setattr(encoder, "prepare_streaming_audio_chunk", record_frontend)
    hook = encoder.model.encoder.register_forward_hook(lambda *args: calls.append(1))
    bulk, bulk_state = encoder.encode_audio_chunk(waveform, FastConformerAudioStreamState())
    hook.remove()
    monkeypatch.setattr(encoder, "prepare_streaming_audio_chunk", prepare)
    assert frontend_batches == ([1] if frames == 1 else [1, frames - 1])
    torch.testing.assert_close(bulk, torch.cat(serial, dim=1), atol=2e-5, rtol=2e-4)
    for continuation in (torch.zeros(3 * FRAME_SAMPLES), torch.randn(2 * FRAME_SAMPLES) * 0.1):
        actual, bulk_state = encoder.encode_audio_chunk(continuation, bulk_state)
        expected = []
        for frame in continuation.split(FRAME_SAMPLES):
            output, serial_state = encoder.encode_audio_chunk(frame, serial_state)
            expected.append(output)
        torch.testing.assert_close(actual, torch.cat(expected, dim=1), atol=2e-5, rtol=2e-4)
        assert bulk_state.next_mel_frame == serial_state.next_mel_frame
        assert bulk_state.buffer_start_sample == serial_state.buffer_start_sample
        torch.testing.assert_close(bulk_state.audio_buffer, serial_state.audio_buffer)
    assert len(calls) == 1
