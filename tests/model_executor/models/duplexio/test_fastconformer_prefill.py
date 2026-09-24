"""Causal multi-frame encoding must preserve streaming outputs and state."""

import pytest
import torch
from tokenizers import Tokenizer, models
from torchaudio import functional as AF
from torchaudio.transforms import Resample
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
    FastConformerStreamState,
    graph_batch_size,
    streaming_resample_batch,
    streaming_resample_chunk,
)


@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
def test_batched_resampling_keeps_each_stream_history(device):
    torch.manual_seed(29)
    chunks = [torch.randn(length, device=device) for length in (1920, 3840, 1920, 1920)]
    tails = [torch.randn(48, device=device), None, torch.randn(48, device=device), None]
    resampler = Resample(24_000, 16_000, dtype=torch.float32).to(device)
    expected = [streaming_resample_chunk(chunk, tail, resampler)
                for chunk, tail in zip(chunks, tails, strict=True)]
    actual, updated = streaming_resample_batch(chunks, tails, resampler)
    for (reference, tail), output, new_tail in zip(expected, actual, updated, strict=True):
        torch.testing.assert_close(output, reference)
        torch.testing.assert_close(new_tail, tail)
    for chunk, tail, value in zip(chunks, tails, actual, strict=True):
        buffer = chunk if tail is None else torch.cat((tail, chunk))
        start = 0 if tail is None else tail.numel() * 2 // 3
        reference = AF.resample(buffer, 24_000, 16_000)[start:start + chunk.numel() * 2 // 3]
        torch.testing.assert_close(value, reference, atol=1e-6, rtol=1e-5)


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


@pytest.mark.parametrize("first", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param(
    "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
)])
@torch.inference_mode()
def test_cached_frontend_matches_upstream_processor(encoder, first, device):
    encoder = encoder.to(device)
    mel_frames = (encoder.processor.num_mel_frames_first_audio_chunk if first
                  else encoder.processor.num_mel_frames_per_audio_chunk)
    samples = (mel_frames - 1) * encoder.feature_hop_length + encoder.feature_n_fft
    if first:
        samples -= encoder.feature_n_fft // 2
    waveform = torch.randn(4, samples, device=device) * 0.1
    for value in (waveform, waveform * 0):
        expected = encoder.processor(
            value.unbind(0), sampling_rate=16_000, is_streaming=True,
            is_first_audio_chunk=first, return_tensors="pt", device=device,
        ).input_features[:, :mel_frames]
        actual = encoder.prepare_streaming_audio_chunk(value, first=first)
        # The upstream processor returns CPU features even for CUDA waveforms.
        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=1e-5, rtol=1e-6)
    assert len(encoder.frontend_constants) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_encoder_graph_preserves_requests_and_output_ownership(encoder, dtype):
    import copy

    encoder = encoder.to(device="cuda", dtype=dtype)
    reference = copy.deepcopy(encoder)
    encoder.use_cuda_graph = True
    actual_states = [FastConformerAudioStreamState() for _ in range(3)]
    expected_states = copy.deepcopy(actual_states)
    retained = []
    for order in ([0, 1, 2],) * 12 + ([2, 0], [1, 2, 0], [0, 2]):
        waveforms = [torch.randn(FRAME_SAMPLES, device="cuda") * 0.1 for _ in order]
        expected, old = reference.encode_audio_batch(waveforms, [expected_states[index] for index in order])
        actual, new = encoder.encode_audio_batch(waveforms, [actual_states[index] for index in order])
        for index, value, target, state, target_state in zip(order, actual, expected, new, old, strict=True):
            torch.testing.assert_close(value, target, atol=1e-5, rtol=1e-4)
            actual_states[index], expected_states[index] = state, target_state
            retained.append((value, value.clone()))
            for cache, reference_cache in zip(state.encoder.tensors(), target_state.encoder.tensors(), strict=True):
                torch.testing.assert_close(cache, reference_cache, atol=1e-5, rtol=1e-4)
    assert set(encoder.graphs) == {2, 4}
    for value, original in retained:
        torch.testing.assert_close(value, original, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA graphs")
@torch.inference_mode()
def test_encoder_graph_rebatches_groups_without_restacking(encoder, monkeypatch):
    import copy

    encoder = encoder.to(device="cuda")
    reference = copy.deepcopy(encoder)
    encoder.use_cuda_graph = True
    actual_states = [FastConformerAudioStreamState() for _ in range(3)]
    expected_states = copy.deepcopy(actual_states)
    stack = FastConformerStreamState.stack.__func__
    packed = []
    monkeypatch.setattr(FastConformerStreamState, "stack", classmethod(
        lambda cls, states: packed.append(len(states)) or stack(cls, states)
    ))
    foreach_copy = torch._foreach_copy_
    copies = []
    monkeypatch.setattr(torch, "_foreach_copy_", lambda targets, values: (
        copies.append(len(targets)) or foreach_copy(targets, values)
    ))
    snapshot = None
    # Steady from step 13: a regrouped subset, a merge of rows from two earlier
    # batches, and a reordering must all feed a captured graph without restacking.
    for step, order in enumerate(([0, 1, 2],) * 14 + ([1, 2],) * 3 + ([0, 1, 2],) * 2 + ([2, 0, 1],) * 2):
        waveforms = [torch.randn(FRAME_SAMPLES, device="cuda") * 0.1 for _ in order]
        expected, old = reference.encode_audio_batch(waveforms, [expected_states[index] for index in order])
        packed.clear()
        copies.clear()
        captured = graph_batch_size(len(order)) in encoder.graphs
        actual, new = encoder.encode_audio_batch(waveforms, [actual_states[index] for index in order])
        for index, value, target, state, target_state in zip(order, actual, expected, new, old, strict=True):
            torch.testing.assert_close(value, target, atol=1e-5, rtol=1e-4)
            actual_states[index], expected_states[index] = state, target_state
        if step >= 13 and captured:
            assert not packed
            if step > 14:
                # One copy per run of rows from the same packed batch.
                assert copies and sum(copies) <= len(order)
            # Rows stay views of one packed replay output, so the next replay copies each run once.
            assert all(actual_states[index].encoder.batch.flat is not None for index in order)
        if step == 12:
            # A rejected step rolls back to earlier state, which later replays must not touch.
            snapshot = [copy.copy(state.encoder) for state in actual_states]
            saved = [[tensor.clone() for tensor in state.tensors()] for state in snapshot]
    assert set(encoder.graphs) == {2, 4}
    for state, tensors in zip(snapshot, saved, strict=True):
        for cache, original in zip(state.tensors(), tensors, strict=True):
            torch.testing.assert_close(cache, original, atol=0, rtol=0)
    for state, target_state in zip(actual_states, expected_states, strict=True):
        assert state.encoder.seq_length == target_state.encoder.seq_length
        for cache, reference_cache in zip(state.encoder.tensors(), target_state.encoder.tensors(), strict=True):
            torch.testing.assert_close(cache, reference_cache, atol=1e-5, rtol=1e-4)


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
