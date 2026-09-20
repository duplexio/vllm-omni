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
    streaming_resample_batch,
    streaming_resample_chunk,
)


def test_batched_resampling_keeps_each_stream_history():
    torch.manual_seed(29)
    chunks = [torch.randn(length) for length in (1920, 3840, 1920, 1920)]
    tails = [torch.randn(48), None, torch.randn(48), None]
    expected = [streaming_resample_chunk(chunk, tail, 24000, 16000)
                for chunk, tail in zip(chunks, tails, strict=True)]
    actual, updated = streaming_resample_batch(chunks, tails, 24000, 16000)
    for (reference, tail), output, new_tail in zip(expected, actual, updated, strict=True):
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(new_tail, tail)


@torch.inference_mode()
def test_model_preprocess_batches_audio_without_persisting_prepared_results(encoder):
    from tests.model_executor.models.duplexio.test_bulk_prefill import input_info, model_fixture, request_state

    model = model_fixture()
    model.user_asr = encoder
    model.user_audio_input_adapter = torch.nn.Linear(encoder.output_dim, 11)
    infos = {}
    expected = {}
    for index in range(3):
        state = request_state(model)
        state.voice_prompt = torch.randn_like(state.voice_prompt) * 0.1
        info = input_info(state, [3, 4, 5], system=False)
        info["_omni_num_scheduled_tokens"] = 30
        info["duplex"]["runtime_config"]["duplexio_record_inputs"] = True
        infos[str(index)] = info
        expected[str(index)] = model.preprocess(torch.zeros(30, dtype=torch.long), None, **info)
    batch_sizes = []
    hook = encoder.model.encoder.register_forward_hook(
        lambda module, args, result: batch_sizes.append(result.last_hidden_state.shape[0]),
    )
    prepared = model.preprocess_batch(req_ids=list(infos), model_intermediate_buffer=infos, device=torch.device("cpu"))
    hook.remove()
    assert batch_sizes == [3]
    for request_id, info in infos.items():
        assert "prepared_audio" not in info
        _, embeddings, updates = model.preprocess(torch.zeros(30, dtype=torch.long), None, **info, **prepared[request_id])
        torch.testing.assert_close(embeddings, expected[request_id][1], atol=2e-5, rtol=2e-4)
        assert updates["duplexio_working_state"].frames_seen == 5
        assert info["duplexio_model_state"].frames_seen == 0


@torch.inference_mode()
def test_batched_streams_preserve_age_order_and_accepted_state(encoder):
    states = []
    for frames in (0, 2, 9, 15):
        state = FastConformerAudioStreamState()
        if frames:
            _, state = encoder.encode_audio_chunk(torch.randn(frames * FRAME_SAMPLES) * 0.1, state)
        states.append(state)
    for order in ([3, 0, 2, 1], [2, 1, 3], [3, 2]):
        waveforms = [torch.randn(FRAME_SAMPLES) * 0.1 for _ in order]
        previous = [states[index] for index in order]
        expected = [encoder.encode_audio_chunk(waveform, state)[0]
                    for waveform, state in zip(waveforms, previous, strict=True)]
        batch_sizes = []
        hook = encoder.model.encoder.register_forward_hook(lambda module, args, result: batch_sizes.append(result.last_hidden_state.shape[0]))
        actual, updated = encoder.encode_audio_batch(waveforms, previous)
        hook.remove()
        assert 2 in batch_sizes  # Unequal ages beyond the cache window batch together.
        for index, reference, output, state in zip(order, expected, actual, updated, strict=True):
            torch.testing.assert_close(output, reference, atol=2e-5, rtol=2e-4)
            old_length = states[index].encoder.past_key_values.get_seq_length() if states[index].encoder.past_key_values is not None else 0
            assert state.encoder.past_key_values.get_seq_length() == old_length + 1
            states[index] = state
        replay, _ = encoder.encode_audio_batch(waveforms, previous)
        for first, second in zip(actual, replay, strict=True):
            torch.testing.assert_close(first, second, atol=0, rtol=0)


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


@torch.inference_mode()
def test_tool_burst_leaves_real_asr_continuation_unchanged(encoder: FastConformerRNNT) -> None:
    from tests.model_executor.models.duplexio.test_bulk_prefill import (
        input_info,
        model_fixture,
        request_state,
    )

    model = model_fixture()
    model.user_asr = encoder
    model.user_audio_input_adapter = torch.nn.Linear(encoder.output_dim, 11)
    prefix = input_info(request_state(model), [3, 4, 5], system=False)
    _, _, update = model.preprocess(torch.zeros(30, dtype=torch.long), None, **prefix)
    before = update["duplexio_working_state"]
    _, _, update = model.preprocess(
        torch.zeros(18, dtype=torch.long), None, **input_info(before, [6, 7, 8], system=True),
    )
    after = update["duplexio_working_state"]
    waveform = torch.randn(1920) * 0.1
    continuations = []
    for state in (before, after):
        _, _, update = model.preprocess(
            torch.zeros(6, dtype=torch.long), None,
            duplexio_model_state=state, duplex_token_offset=0, duplex_prompt_len=6,
            duplex={"frame_count": 1, "pcm": waveform.numpy().tobytes(),
                    "runtime_config": {"duplexio_record_inputs": True}},
        )
        continuations.append(update["duplexio_replay"]["user_features"])
    torch.testing.assert_close(*continuations, atol=0, rtol=0)
    assert before.user_asr is after.user_asr
