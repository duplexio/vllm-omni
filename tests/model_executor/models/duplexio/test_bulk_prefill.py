# SPDX-License-Identifier: Apache-2.0
"""Context rows encode silence in frame order without advancing live audio time."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    DelayedMimiRepresentation,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
)
from vllm_omni.model_executor.models.duplexio.mimi import MimiStreamingState, MimiTransformerState
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIORequestState,
    frame_inputs,
)


class TextEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class CountingASR:
    output_dim = 8

    def encode_audio_chunk(self, waveform, state):
        frames = waveform.numel() // 1280
        positions = torch.arange(state.next_mel_frame + 1, state.next_mel_frame + frames + 1)
        state.next_mel_frame += frames
        return positions.float()[None, :, None].expand(1, -1, self.output_dim), state


class CountingCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.waveforms = []

    def encode(self, waveform, codebooks, state):
        self.waveforms.append(waveform.flatten())
        frames = waveform.numel() // 1920
        start = state.encoder_transformer.position
        state.encoder_transformer.position += frames
        return torch.arange(start + 1, start + frames + 1)[None, None].expand(1, codebooks, -1)


def model_fixture() -> DuplexIOForConditionalGeneration:
    torch.manual_seed(17)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.pad_token_id = 1
    model.silence_token_id = 2
    model.full_cudagraph_enabled = False
    model.frame_inputs = frame_inputs
    # Audio cells see a two-frame window here, so eviction shows up in the test.
    model.config = SimpleNamespace(audio_attention_window_frames=2, frame_size=1920, sample_rate=24000)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32)
    )
    model.audio_representation = DelayedMimiRepresentation(
        num_codebooks=3,
        codebook_size=64,
        acoustic_delay_frames=1,
    )
    model.agent_audio_embedding = MimiEmbedding(3, 64, 5)
    model.user_audio_input_adapter = AudioInputAdapter(8, 7, 11)
    model.agent_audio_input_adapter = AudioInputAdapter(5, 7, 11)
    model.user_asr = CountingASR()
    model.audio_codec = CountingCodec()
    model.llm = SimpleNamespace(
        base_model=SimpleNamespace(model=TextEmbedding(11)),
        channel_emb=nn.Parameter(torch.randn(4, 11)),
    )
    return model


def request_state(model: DuplexIOForConditionalGeneration) -> DuplexIORequestState:
    return DuplexIORequestState(
        text_input_ids=torch.full((4,), 2, dtype=torch.long),
        agent_audio_codes=model.initial_agent_audio(1)[0],
        user_asr=FastConformerAudioStreamState(),
        agent_delay=model.audio_representation.new_state(device=torch.device("cpu")),
        output_mimi=MimiStreamingState(),
        input_mimi=MimiStreamingState(encoder_transformer=MimiTransformerState.empty(0)),
        voice_prompt=torch.ones(2 * 1920),
        system_token_ids=(3, 4, 5),
        sampling_generator=torch.Generator().manual_seed(9),
    )


def input_info(
    state: DuplexIORequestState, tokens: list[int], *, system: bool
):
    return {
        "duplexio_model_state": state,
        "duplex_token_offset": state.frames_seen * 6,
        "duplex_prompt_len": (state.frames_seen + len(tokens) + (0 if system else 2)) * 6,
        "duplex": {
            "frame_count": len(tokens) + (0 if system else 2),
            "duplexio_prefill": not system,
            "duplexio_system_input": system,
            "duplexio_system_token_ids": tokens,
            "runtime_config": {},
        },
    }


@torch.inference_mode()
def test_tool_bulk_and_serial_frames_are_identical() -> None:
    model = model_fixture()
    initial = request_state(model)
    initial.frames_seen = 5  # The complete speaker/system prefix precedes tools.
    tokens = [6, 7, 8]
    _, bulk, bulk_update = model.preprocess(
        torch.zeros(18, dtype=torch.long),
        None,
        **input_info(initial, tokens, system=True),
    )
    serial_state = initial
    embeddings = []
    metadata = {
        key: []
        for key in (
            "key_active",
            "text_ordinals",
            "text_last",
            "audio_first",
            "audio_last",
        )
    }
    for token in tokens:
        _, hidden, update = model.preprocess(
            torch.zeros(6, dtype=torch.long),
            None,
            **input_info(serial_state, [token], system=True),
        )
        serial_state = update["duplexio_working_state"]
        embeddings.append(hidden)
        for key in metadata:
            metadata[key].append(update["duplexio"][key])
    torch.testing.assert_close(bulk, torch.cat(embeddings))
    for key, values in metadata.items():
        torch.testing.assert_close(bulk_update["duplexio"][key], torch.cat(values))

    state = bulk_update["duplexio_working_state"]
    assert state.frames_seen == serial_state.frames_seen == 8
    assert state.active_text_tokens == serial_state.active_text_tokens == 3
    assert state.audio_position == serial_state.audio_position == 0
    assert state.user_asr.next_mel_frame == 3
    assert initial.user_asr.next_mel_frame == 0
    assert state.input_mimi.encoder_transformer.position == 3
    assert initial.input_mimi.encoder_transformer.position == 0
    assert not bulk_update["duplexio"]["key_active"].view(3, 6)[:, 4:].any()
    torch.testing.assert_close(state.agent_audio_codes, model.initial_agent_audio(1)[0])


@torch.inference_mode()
def test_live_audio_advances_encoder_and_inserts_generated_feedback() -> None:
    model = model_fixture()
    state = request_state(model)
    state.text_input_ids[1] = 9
    state.text_input_ids[2] = 12
    state.agent_audio_codes = torch.tensor([3, 4, 5])
    user_features = torch.ones(1, 8)  # First frame of the counting encoder.
    info = dict(
        duplexio_model_state=state,
        duplex_token_offset=0, duplex_prompt_len=6,
        duplex={
            "frame_count": 1,
            "runtime_config": {"duplexio_record_inputs": True},
            "pcm": torch.randn(1920).numpy().tobytes(),
        },
    )
    _, embeddings, update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **info)
    expected_user = model.llm.base_model.model.embed_input_ids(torch.tensor([9]))[0]
    torch.testing.assert_close(embeddings[1], expected_user + model.llm.channel_emb[1])
    torch.testing.assert_close(
        embeddings[4:5], model.user_audio_input_adapter(user_features)
    )
    torch.testing.assert_close(
        embeddings[5:6],
        model.agent_audio_input_adapter(
            model.agent_audio_embedding(state.agent_audio_codes[None])
        ),
    )
    assert update["duplexio_working_state"].audio_position == 1
    assert state.audio_position == 0
    assert update["duplexio_working_state"].user_asr.next_mel_frame == 1
    assert state.user_asr.next_mel_frame == 0
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 9, 12, 2]]
    torch.testing.assert_close(replay["user_features"], user_features)
    torch.testing.assert_close(replay["agent_audio"], torch.tensor([[3, 4, 5]]))
    assert replay["audio_mask"].tolist() == [True]
    assert update["duplexio"]["key_active"].tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]


def test_recorded_prefix_contains_encoded_silence_and_frozen_audio_clock() -> None:
    model = model_fixture()
    info = input_info(request_state(model), [3, 4, 5], system=False)
    info["duplex"]["runtime_config"]["duplexio_record_inputs"] = True
    _, _, update = model.preprocess(torch.zeros(30, dtype=torch.long), None, **info)
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 2, 2, 2], [2, 2, 2, 2], [3, 2, 2, 2], [4, 2, 2, 2], [5, 2, 2, 2]]
    assert not replay["audio_mask"].any()
    assert replay["prompt_frames"].tolist() == [True, True, False, False, False]
    torch.testing.assert_close(replay["user_features"], torch.arange(1, 6).float()[:, None].expand(-1, 8))
    torch.testing.assert_close(replay["agent_audio"], torch.tensor([[1, 64, 64], [2, 1, 1], [3, 2, 2], [4, 3, 3], [5, 4, 4]]))
    state = update["duplexio_working_state"]
    assert state.frames_seen == state.input_audio_frames == 5
    assert state.audio_position == 0
    masks = update["duplexio"]
    assert masks["key_active"].view(5, 6)[:, 4:].tolist() == [[True, True]] * 2 + [[False, False]] * 3
    assert masks["prompt_ordinal"].view(5, 6)[:, 4].tolist() == [1, 2, 0, 0, 0]
    assert masks["prompt_last"].view(5, 6)[:, 0].tolist() == [0, 1, 2, 2, 2]
    assert len(model.audio_codec.waveforms) == 1
    torch.testing.assert_close(model.audio_codec.waveforms[0], torch.cat((torch.ones(2 * 1920), torch.zeros(3 * 1920))))


@pytest.mark.parametrize("frames,frames_seen", [(1, 0), (4, 0), (6, 0), (5, 5)])
def test_prefix_must_be_complete_and_consumed_once(frames: int, frames_seen: int) -> None:
    model = model_fixture()
    state = request_state(model)
    state.frames_seen = frames_seen
    with pytest.raises(ValueError, match="complete speaker and system prefix exactly once"):
        model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None,
            duplexio_model_state=state,
            duplex_token_offset=frames_seen * 6, duplex_prompt_len=(frames_seen + frames) * 6,
            duplex={"frame_count": frames, "duplexio_prefill": True,
                    "runtime_config": {"duplexio_record_inputs": True}},
        )


@pytest.mark.parametrize("system", [False, True])
@pytest.mark.parametrize("chunks", [(1, 1, 3), (3, 2), (2, 2, 1)])
@torch.inference_mode()
def test_scheduler_chunks_preserve_embeddings_masks_and_replay(system: bool, chunks: tuple[int, ...]) -> None:
    model = model_fixture()
    tokens = [3, 4, 5, 6, 7] if system else [3, 4, 5]
    initial = request_state(model)
    # Tool appends begin after existing context, unlike the initial prefix.
    initial.frames_seen = 7 if system else 0
    info = input_info(initial, tokens, system=system)
    info["duplex"]["runtime_config"]["duplexio_record_inputs"] = True
    _, bulk, bulk_update = model.preprocess(torch.zeros(30, dtype=torch.long), None, **info)
    state = initial
    embeddings = []
    updates = []
    for frames in chunks:
        info["duplexio_model_state"] = state
        info["duplex_token_offset"] = state.frames_seen * 6
        _, hidden, update = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)
        state = update["duplexio_working_state"]
        embeddings.append(hidden)
        updates.append(update)
    torch.testing.assert_close(torch.cat(embeddings), bulk)
    for group in ("duplexio", "duplexio_replay"):
        for key, expected in bulk_update[group].items():
            torch.testing.assert_close(torch.cat([update[group][key] for update in updates]), expected)
    for name in ("frames_seen", "input_audio_frames", "audio_position", "active_text_tokens"):
        assert getattr(state, name) == getattr(bulk_update["duplexio_working_state"], name)


def test_live_feedback_advances_encoder_before_later_context_silence() -> None:
    model = model_fixture()
    state = request_state(model)
    state.agent_waveform = torch.ones(1920)
    state.agent_audio_codes = torch.tensor([7, 8, 9])
    info = dict(
        duplexio_model_state=state,
        duplex_token_offset=0, duplex_prompt_len=6,
        duplex={"frame_count": 1, "runtime_config": {"duplexio_record_inputs": True},
                "pcm": torch.ones(1920).numpy().tobytes()},
    )
    _, _, update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **info)
    torch.testing.assert_close(update["duplexio_replay"]["agent_audio"], state.agent_audio_codes[None])
    consumed = update["duplexio_working_state"]
    assert consumed.input_mimi.encoder_transformer.position == 1
    assert state.input_mimi.encoder_transformer.position == 0
    _, _, update = model.preprocess(torch.zeros(6, dtype=torch.long), None,
        **input_info(consumed, [5], system=True),
    )
    assert update["duplexio_working_state"].input_mimi.encoder_transformer.position == 2


@pytest.mark.parametrize("encoded_frames,waveform_frame", [(1, 1), (2, 1)])
def test_tool_context_consumes_late_waveforms_only_once(encoded_frames: int, waveform_frame: int) -> None:
    model = model_fixture()
    state = request_state(model)
    state.frames_seen = 2
    state.input_audio_frames = encoded_frames
    state.input_mimi.encoder_transformer.position = encoded_frames
    state.agent_waveform = torch.ones(1920)
    state.agent_waveform_frame = waveform_frame
    _, _, update = model.preprocess(torch.zeros(6, dtype=torch.long), None,
        **input_info(state, [5], system=True),
    )
    consumed = update["duplexio_working_state"]
    assert consumed.input_audio_frames == consumed.input_mimi.encoder_transformer.position == 3
    assert consumed.agent_waveform is None
    assert state.input_mimi.encoder_transformer.position == encoded_frames
