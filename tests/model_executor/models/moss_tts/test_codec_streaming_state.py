"""Cached decoding must preserve waveform values and request isolation."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer import (
    MossAudioTokenizerConfig,
    MossAudioTokenizerModel,
)
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_codec import (
    MossTTSCodecDecoder,
    _MossCodecStreamSession,
)
from vllm_omni.utils.mm_outputs import partition_payload_list


def make_codec(device: str) -> MossAudioTokenizerModel:
    def transformer(input_dim: int, output_dim: int) -> dict:
        return {
            "module_type": "Transformer",
            "input_dimension": input_dim,
            "output_dimension": output_dim,
            "d_model": 8,
            "num_heads": 2,
            "num_layers": 2,
            "dim_feedforward": 16,
            "layer_scale": 0.3,
        }

    torch.manual_seed(31)
    config = MossAudioTokenizerConfig(
        sampling_rate=12,
        downsample_rate=6,
        causal_transformer_context_duration=1.5,
        encoder_kwargs=[{"module_type": "PatchedPretransform", "patch_size": 6}, transformer(6, 8)],
        decoder_kwargs=[
            transformer(8, 8),
            {"module_type": "PatchedPretransform", "patch_size": 2},
            transformer(4, 6),
            {"module_type": "PatchedPretransform", "patch_size": 3},
            transformer(2, 1),
        ],
        quantizer_kwargs={
            "input_dim": 8,
            "rvq_dim": 8,
            "output_dim": 8,
            "num_quantizers": 2,
            "codebook_size": 16,
            "codebook_dim": 4,
        },
    )
    return MossAudioTokenizerModel(config).to(device).eval()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="requires CUDA",
            ),
        ),
    ],
)
@torch.inference_mode()
def test_cached_chunks_match_one_shot_across_context_rollover(device: str) -> None:
    codec = make_codec(device)
    codes = torch.randint(0, 16, (2, 2, 17), device=device)
    expected = codec._decode(codes, torch.full((2,), 17, device=device, dtype=torch.long)).audio
    state = codec.new_decode_state(2)
    slots = torch.tensor([0, 1], device=device)
    outputs = []
    cursor = 0
    for size in (2, 1, 5, 3, 6):
        outputs.append(codec.decode_chunk(codes[:, :, cursor : cursor + size], slots, state).audio)
        cursor += size
    torch.testing.assert_close(torch.cat(outputs, dim=-1), expected, atol=1e-7, rtol=1e-4)
    torch.testing.assert_close(state.offsets, torch.tensor([17, 17], device=device))
    assert [layers[0].keys.shape[2] for layers in state.layers if layers is not None] == [3, 6, 18]


@torch.inference_mode()
def test_paused_streams_reordered_batches_and_slot_reuse() -> None:
    codec = make_codec("cpu")
    state = codec.new_decode_state(2)
    codes = torch.randint(0, 16, (2, 2, 11))
    expected = codec._decode(codes, torch.tensor([11, 11])).audio
    output_a = [codec.decode_chunk(codes[:, :1, :5], torch.tensor([0]), state).audio]
    output_b = [codec.decode_chunk(codes[:, 1:, :2], torch.tensor([1]), state).audio]
    batch = torch.stack([codes[:, 1, 2:4], codes[:, 0, 5:7]], dim=1)
    mixed = codec.decode_chunk(batch, torch.tensor([1, 0]), state).audio
    output_a.append(mixed[1:])
    output_b.append(mixed[:1])
    output_a.append(codec.decode_chunk(codes[:, :1, 7:], torch.tensor([0]), state).audio)
    output_b.append(codec.decode_chunk(codes[:, 1:, 4:], torch.tensor([1]), state).audio)
    torch.testing.assert_close(torch.cat(output_a, dim=-1), expected[:1], atol=1e-7, rtol=1e-4)
    torch.testing.assert_close(torch.cat(output_b, dim=-1), expected[1:], atol=1e-7, rtol=1e-4)
    state.reset(torch.tensor([0]))
    restarted = codec.decode_chunk(codes[:, :1, :5], torch.tensor([0]), state).audio
    torch.testing.assert_close(restarted, expected[:1, :, :30], atol=1e-7, rtol=1e-4)
    torch.testing.assert_close(state.offsets, torch.tensor([5, 11]))


@torch.inference_mode()
def test_stream_session_retains_history_and_resets_released_slot() -> None:
    codec = make_codec("cpu")
    session = _MossCodecStreamSession(codec, stream_slots=2, n_vq=2)
    first_slot, second_slot = session.acquire(), session.acquire()
    assert first_slot is not None and second_slot is not None
    codes = torch.randint(0, 16, (2, 8))
    expected = codec.batch_decode([codes]).audio[0]
    first = session.step({first_slot: codes[:, :3]})[first_slot]
    second = session.step({second_slot: codes[:, :5], first_slot: codes[:, 3:]})
    torch.testing.assert_close(torch.cat([first, second[first_slot]], dim=-1), expected, atol=1e-7, rtol=1e-4)
    session.release(first_slot)
    assert session.acquire() == first_slot
    restarted = session.step({first_slot: codes[:, :3]})[first_slot]
    torch.testing.assert_close(restarted, first, atol=1e-7, rtol=1e-4)
    session.close()


@torch.inference_mode()
def test_context_codes_encode_completed_waveform_and_clear_request_state() -> None:
    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(rvq=2)))
    decoder = MossTTSCodecDecoder(vllm_config=config)
    decoder._codec = make_codec("cpu")
    first = torch.randn(18)
    second = torch.randn(24)
    info = [{"meta": {"req_id": ["user"], "codec_streaming": True, "return_context_codes": True}}]
    pending = decoder.encode_context_outputs([first], info)
    assert pending[0].shape == (2, 0)
    info[0]["meta"]["stream_finished"] = True
    completed = decoder.encode_context_outputs([second], info)
    _, client_outputs = partition_payload_list([{"context_codes": completed[0]}])
    torch.testing.assert_close(client_outputs[0]["context_codes"], completed[0])
    expected = decoder._codec.batch_encode([torch.cat([first, second])], num_quantizers=2).audio_codes
    torch.testing.assert_close(torch.cat([pending[0], completed[0]], dim=-1), expected[:, 0])
    assert not decoder.context_audio
    offline = [{"meta": {"return_context_codes": True}}]
    assert decoder.encode_context_outputs([torch.empty(0)], offline)[0].shape == (2, 0)
    info[0]["meta"]["stream_finished"] = False
    decoder.encode_context_outputs([first], info)
    decoder.on_requests_finished(["user"])
    assert not decoder.context_audio


def test_bridge_forwards_context_encoding_request_including_terminal_packet() -> None:
    from vllm_omni.model_executor.stage_input_processors.moss_tts import talker2codec, talker2codec_raw_async_chunk

    info = {"meta": {"return_context_codes": True}}
    codes = torch.zeros(2, 2, dtype=torch.long)
    sync = talker2codec([{"codes": {"audio": codes}}], {"additional_information": info})
    assert sync[0]["additional_information"]["meta"]["return_context_codes"] is True
    transfer = SimpleNamespace(connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 2}}))
    request = SimpleNamespace(external_req_id="user", additional_information=info)
    chunk = talker2codec_raw_async_chunk(transfer, {"codes": {"audio": codes}}, request)
    assert chunk.meta.return_context_codes is True
    terminal = talker2codec_raw_async_chunk(transfer, None, request, is_finished=True)
    assert terminal.meta.return_context_codes is True


@torch.inference_mode()
@pytest.mark.parametrize("terminal_packet", [False, True])
def test_streamed_codes_accumulate_once_along_time(terminal_packet: bool) -> None:
    from vllm_omni.engine.mm_outputs import MultimodalPayload
    from vllm_omni.engine.output_processor import OmniRequestState

    config = SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(rvq=2)))
    decoder = MossTTSCodecDecoder(vllm_config=config)
    decoder._codec = make_codec("cpu")
    codes = torch.randint(0, 16, (2, 8))
    state = OmniRequestState.__new__(OmniRequestState)
    state.mm_type = "audio"
    state.mm_accumulated = MultimodalPayload()
    chunks = [codes[:, :4], codes[:, 4:]]
    if terminal_packet:
        chunks.append(torch.empty((2, 0), dtype=torch.long))
    for index, chunk in enumerate(chunks):
        meta = {
            "req_id": ["user"],
            "codec_streaming": True,
            "return_context_codes": True,
            "stream_finished": index == len(chunks) - 1,
            "code_flat_numel": chunk.numel(),
        }
        inputs = chunk.flatten() if chunk.numel() else torch.zeros(1, dtype=torch.long)
        output = decoder(
            input_ids=inputs, runtime_additional_information=[{"meta": meta}], seq_token_counts=[inputs.numel()]
        )
        state.add_multimodal_tensor({key: value[0] for key, value in output.multimodal_outputs.items()}, "audio")
        state._consolidate_multimodal_tensors()
    torch.testing.assert_close(state.mm_accumulated.tensors["audio_codes"], codes)
    waveform = decoder._codec.batch_decode([codes]).audio[0, 0]
    torch.testing.assert_close(state.mm_accumulated.tensors["audio"], waveform, atol=1e-7, rtol=1e-4)
    encoded = decoder._codec.batch_encode([waveform], num_quantizers=2).audio_codes
    torch.testing.assert_close(state.mm_accumulated.tensors["context_codes"], encoded[:, 0])
