# SPDX-License-Identifier: Apache-2.0
"""Text-only rows leave acoustic state untouched across chunked prefill."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn
from torchaudio.transforms import Resample

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import ContinuousAudioRepresentation
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIORequestState,
    frame_inputs,
)
from vllm_omni.model_executor.models.duplexio.pocket_mimi import LATENT_DIM
from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig


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
        return positions.float()[None, :, None].expand(1, -1, self.output_dim), replace(state, next_mel_frame=state.next_mel_frame + frames)

    def encode_audio_batch(self, waveforms, states):
        outputs = [self.encode_audio_chunk(waveform, state) for waveform, state in zip(waveforms, states, strict=True)]
        return [output for output, _ in outputs], [state for _, state in outputs]


class CountingCodec(nn.Module):
    """Stands in for Pocket Mimi: a stream's frame k encodes to a latent full of k; the state is its position."""

    def __init__(self):
        super().__init__()
        self.waveforms = []

    def encode_batch(self, waveforms, positions):
        latents = []
        for waveform, position in zip(waveforms, positions, strict=True):
            self.waveforms.append(waveform.flatten())
            frames = waveform.numel() // 1920
            latents.append(torch.arange(position + 1, position + frames + 1).float()[None, None].expand(1, LATENT_DIM, -1))
        return latents, [position + waveform.numel() // 1920 for waveform, position in zip(waveforms, positions)]


def latent(*values: float) -> Tensor:
    """Agent-audio rows, each a latent filled with one value."""
    return torch.tensor(values, dtype=torch.float32)[:, None].expand(-1, LATENT_DIM)


def model_fixture() -> DuplexIOForConditionalGeneration:
    torch.manual_seed(17)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.pad_token_id = 1
    model.silence_token_id = 2
    model.full_cudagraph_enabled = False
    model.frame_inputs = frame_inputs
    # Audio cells see a two-frame window here, so eviction shows up in the test.
    model.config = SimpleNamespace(audio_attention_window_frames=2, flowmap_config={"sampling_temperature": 1.0})
    model.text_config = SimpleNamespace(max_position_embeddings=262144, vocab_size=32)
    model.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.float32)
    )
    model.audio_representation = ContinuousAudioRepresentation(LATENT_DIM)
    model.user_audio_input_adapter = AudioInputAdapter(8, 7, 11)
    model.agent_audio_input_adapter = AudioInputAdapter(LATENT_DIM, 7, 11)
    model.user_asr = CountingASR()
    model.user_audio_resampler = Resample(24_000, 16_000, dtype=torch.float32)
    model.audio_codec = CountingCodec()
    model.llm = SimpleNamespace(
        base_model=SimpleNamespace(model=TextEmbedding(11)),
        channel_emb=nn.Parameter(torch.randn(4, 11)),
    )
    return model


def request_state(model: DuplexIOForConditionalGeneration) -> DuplexIORequestState:
    return DuplexIORequestState(
        text_input_ids=(2, 2, 2, 2),
        agent_latent=model.initial_agent_latents(1)[0],
        user_asr=FastConformerAudioStreamState(),
        output_mimi=0,
        input_mimi=0,
        voice_prompt=torch.ones(2 * 1920),
        system_token_ids=(3, 4, 5),
        sampling=model.resolve_sampling(SamplingConfig().model_dump()),
    )


def append_info(
    state: DuplexIORequestState, *, prefix: bool = False, tool: tuple[int, ...] = (), pcm: bytes | None = None,
):
    """One append: ``[voice prompt, system tokens if prefix] + [live frame] + [tool tokens]``."""
    frames = (2 + len(state.system_token_ids) if prefix else 0) + 1 + len(tool)
    return {
        "duplexio_model_state": state,
        "duplex_token_offset": state.frames_seen * 6,
        "duplex_prompt_len": (state.frames_seen + frames) * 6,
        "duplex": {
            "frame_count": frames,
            "duplexio_prefix": prefix,
            "duplexio_tool_token_ids": list(tool),
            "duplexio_tool_generation": 0,
            "pcm": torch.zeros(1920).numpy().tobytes() if pcm is None else pcm,
            "runtime_config": {"duplexio_record_inputs": True},
        },
    }


@torch.inference_mode()
def test_packed_preprocess_preserves_mixed_request_boundaries() -> None:
    model = model_fixture()
    prefix = append_info(request_state(model), prefix=True)
    tool_state = request_state(model)
    tool_state.frames_seen = 19
    tool_state.audio_position = 11
    tool_state.persistent_keys = 23
    tool = append_info(tool_state, tool=(6, 7, 8))
    live_state = request_state(model)
    live_state.frames_seen = 40
    live_state.audio_position = 30
    live_state.persistent_keys = 53
    live_state.text_input_ids = (2, 9, 12, 2)
    live = append_info(live_state, pcm=torch.randn(1920).numpy().tobytes())
    # A prefix continuation begins inside the speaker prompt, then crosses into text.
    partial = append_info(request_state(model), prefix=True)
    partial["duplexio_model_state"].frames_seen = 1
    partial["duplexio_model_state"].persistent_keys = 1
    partial["duplex_token_offset"] = 6
    partial["duplex_prompt_len"] = 36
    infos = dict(zip(("live", "prefix", "tool", "partial"), (live, prefix, tool, partial), strict=True))
    counts = (1, 6, 4, 4)
    expected = []
    for info, frames in zip(infos.values(), counts, strict=True):
        info["_omni_num_scheduled_tokens"] = frames * 6
        expected.append(model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info))
    calls = []
    hook = model.user_audio_input_adapter.register_forward_hook(lambda module, args, result: calls.append(args[0].shape[0]))
    prepared = model.preprocess_batch(req_ids=list(infos), model_intermediate_buffer=infos, device=torch.device("cpu"))
    hook.remove()
    assert calls == [sum(counts)]
    for (key, info), frames, (_, reference, reference_updates) in zip(infos.items(), counts, expected, strict=True):
        _, actual, updates = model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info, **prepared[key])
        torch.testing.assert_close(actual, reference)
        for field in ("duplexio", "duplexio_replay"):
            for name, value in reference_updates[field].items():
                torch.testing.assert_close(updates[field][name], value, rtol=0, atol=0)
        for name in ("frames_seen", "audio_position", "persistent_keys"):
            assert getattr(updates["duplexio_working_state"], name) == getattr(reference_updates["duplexio_working_state"], name)
    assert live_state.frames_seen == 40
    assert tool_state.persistent_keys == 23


@torch.inference_mode()
def test_tool_rows_follow_the_live_frame_as_text() -> None:
    model = model_fixture()
    initial = request_state(model)
    initial.frames_seen = 6  # The prefix and the first live frame precede tools.
    initial.text_input_ids = (2, 9, 12, 2)
    initial.agent_latent = latent(3)[0]
    _, embeddings, update = model.preprocess(
        torch.zeros(24, dtype=torch.long), None, **append_info(initial, tool=(6, 7, 8)),
    )
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 9, 12, 2], [6, 2, 2, 2], [7, 2, 2, 2], [8, 2, 2, 2]]
    assert replay["audio_mask"].tolist() == [True, False, False, False]
    torch.testing.assert_close(replay["user_features"], torch.tensor([1, 0, 0, 0]).float()[:, None].expand(-1, 8))
    torch.testing.assert_close(replay["agent_audio"], latent(3, 0, 0, 0))
    state = update["duplexio_working_state"]
    assert state.frames_seen == 10
    assert state.persistent_keys == 5  # The user and agent feedback, then three tool tokens.
    assert state.audio_position == 1
    assert state.user_asr.next_mel_frame == 1
    assert initial.user_asr.next_mel_frame == 0
    assert state.input_mimi == 0
    assert not update["duplexio"]["key_active"].view(4, 6)[1:, 4:].any()
    assert not embeddings.view(4, 6, -1)[1:, 4:].any()
    assert model.audio_codec.waveforms == []


@torch.inference_mode()
def test_live_audio_advances_encoder_and_inserts_generated_feedback() -> None:
    model = model_fixture()
    state = request_state(model)
    state.text_input_ids = (2, 9, 12, 2)
    state.agent_latent = latent(3)[0]
    user_features = torch.ones(1, 8)  # First frame of the counting encoder.
    info = append_info(state, pcm=torch.randn(1920).numpy().tobytes())
    _, embeddings, update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **info)
    expected_user = model.llm.base_model.model.embed_input_ids(torch.tensor([9]))[0]
    torch.testing.assert_close(embeddings[1], expected_user + model.llm.channel_emb[1])
    torch.testing.assert_close(
        embeddings[4:5], model.user_audio_input_adapter(user_features)
    )
    torch.testing.assert_close(
        embeddings[5:6],
        model.agent_audio_input_adapter(state.agent_latent[None]),
    )
    assert update["duplexio_working_state"].audio_position == 1
    assert state.audio_position == 0
    assert update["duplexio_working_state"].user_asr.next_mel_frame == 1
    assert state.user_asr.next_mel_frame == 0
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 9, 12, 2]]
    torch.testing.assert_close(replay["user_features"], user_features)
    torch.testing.assert_close(replay["agent_audio"], latent(3))
    assert replay["audio_mask"].tolist() == [True]
    assert update["duplexio"]["key_active"].tolist() == [
        False,
        True,
        True,
        False,
        True,
        True,
    ]


def test_prefix_encodes_the_speaker_prompt_then_the_first_live_frame() -> None:
    model = model_fixture()
    _, _, update = model.preprocess(torch.zeros(36, dtype=torch.long), None, **append_info(request_state(model), prefix=True))
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 2, 2, 2]] * 2 + [[3, 2, 2, 2], [4, 2, 2, 2], [5, 2, 2, 2], [2, 2, 2, 2]]
    assert replay["audio_mask"].tolist() == [False] * 5 + [True]
    assert replay["prompt_frames"].tolist() == [True, True, False, False, False, False]
    # The encoder hears silence under the voice prompt, then the live frame.
    torch.testing.assert_close(replay["user_features"], torch.tensor([1, 2, 0, 0, 0, 3]).float()[:, None].expand(-1, 8))
    torch.testing.assert_close(
        replay["agent_audio"], latent(1, 2, 0, 0, 0, 0),
    )
    state = update["duplexio_working_state"]
    assert state.frames_seen == 6
    assert state.user_asr.next_mel_frame == 3
    assert state.input_mimi == 2
    assert state.audio_position == 1
    masks = update["duplexio"]
    assert masks["key_active"].view(6, 6)[:, 4:].tolist() == [[False, True]] * 2 + [[False, False]] * 3 + [[True, True]]
    # Prompt keys, then the system text, share one dense persistent ordinal.
    ordinals = masks["persistent_ordinal"].view(6, 6)
    assert ordinals[:5, 5].tolist() == [1, 2, 0, 0, 0]
    assert ordinals[:5, 0].tolist() == [0, 0, 3, 4, 5]
    assert masks["persistent_last"].view(6, 6)[:5, 0].tolist() == [0, 1, 2, 3, 4]
    assert len(model.audio_codec.waveforms) == 1
    torch.testing.assert_close(model.audio_codec.waveforms[0], torch.ones(2 * 1920))


def given_info(state: DuplexIORequestState, agent_pcm: Tensor, *, prefix: bool = False, **frame: int):
    """An append whose live frame replays history: ``frame`` names its given token ids."""
    info = append_info(state, prefix=prefix)
    info["duplex"]["duplexio_given_frame"] = {
        "user_token_id": 2, "agent_token_id": 2, "tool_call_token_id": 2,
        **frame, "agent_pcm": agent_pcm.numpy().tobytes(),
    }
    return info


@torch.inference_mode()
def test_given_frames_replay_history_through_the_voice_prompt_codec_stream() -> None:
    model = model_fixture()
    first, second = torch.randn(1920), torch.randn(1920)
    _, _, update = model.preprocess(
        torch.zeros(36, dtype=torch.long), None,
        **given_info(request_state(model), first, prefix=True, user_token_id=9, agent_token_id=12),
    )
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist()[-1] == [2, 9, 12, 2]
    # The codec hears the voice prompt, then the given agent frame, as one stream.
    torch.testing.assert_close(replay["agent_audio"][:2], latent(1, 2))
    torch.testing.assert_close(replay["agent_audio"][-1], latent(3)[0])
    torch.testing.assert_close(model.audio_codec.waveforms[0], torch.cat((torch.ones(2 * 1920), first)))
    state = update["duplexio_working_state"]
    assert state.input_mimi == 3
    assert state.persistent_keys == 7  # Two prompt keys, three system tokens, then the given user and agent text.
    # A sampled prediction is waiting; the next given frame replaces it.
    state.text_input_ids = (2, 5, 6, 2)
    state.agent_latent = latent(7)[0]
    _, _, update = model.preprocess(
        torch.zeros(6, dtype=torch.long), None, **given_info(state, second, agent_token_id=13),
    )
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[2, 2, 13, 2]]
    torch.testing.assert_close(replay["agent_audio"], latent(4))
    torch.testing.assert_close(model.audio_codec.waveforms[1], second)
    assert update["duplexio_working_state"].audio_position == 2


@torch.inference_mode()
def test_given_frames_must_lead_and_hold_no_tool_calls() -> None:
    model = model_fixture()
    _, _, update = model.preprocess(torch.zeros(36, dtype=torch.long), None, **append_info(request_state(model), prefix=True))
    sampled = update["duplexio_working_state"]
    assert not sampled.history_open
    with pytest.raises(ValueError, match="lead the conversation"):
        model.preprocess(torch.zeros(6, dtype=torch.long), None, **given_info(sampled, torch.zeros(1920)))
    fresh = request_state(model)
    with pytest.raises(ValueError, match="tool calls"):
        model.preprocess(
            torch.zeros(36, dtype=torch.long), None,
            **given_info(fresh, torch.zeros(1920), prefix=True, tool_call_token_id=6),
        )
    with pytest.raises(ValueError, match="one 1920-sample"):
        model.preprocess(torch.zeros(36, dtype=torch.long), None, **given_info(fresh, torch.zeros(960), prefix=True))


@pytest.mark.parametrize(
    "frames,frames_seen,match",
    [(1, 0, "do not match"), (5, 0, "do not match"), (7, 0, "do not match"), (6, 6, "exactly once")],
)
def test_prefix_must_be_complete_and_consumed_once(frames: int, frames_seen: int, match: str) -> None:
    model = model_fixture()
    state = request_state(model)
    state.frames_seen = frames_seen
    info = append_info(state, prefix=True)
    info["duplex"]["frame_count"] = frames
    info["duplex_prompt_len"] = (frames_seen + frames) * 6
    with pytest.raises(ValueError, match=match):
        model.preprocess(torch.zeros(frames * 6, dtype=torch.long), None, **info)


@pytest.mark.parametrize("prefix", [True, False])
@pytest.mark.parametrize("chunks", [(1, 1, 4), (3, 3), (2, 2, 2), (5, 1)])
@torch.inference_mode()
def test_scheduler_chunks_preserve_embeddings_masks_and_replay(prefix: bool, chunks: tuple[int, ...]) -> None:
    model = model_fixture()
    initial = request_state(model)
    # Tool appends begin after existing context, unlike the initial prefix.
    initial.frames_seen = 0 if prefix else 7
    info = append_info(initial, prefix=prefix, tool=() if prefix else (3, 4, 5, 6, 7),
                       pcm=torch.randn(1920).numpy().tobytes())
    _, bulk, bulk_update = model.preprocess(torch.zeros(36, dtype=torch.long), None, **info)
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
    for name in ("frames_seen", "audio_position", "persistent_keys"):
        assert getattr(state, name) == getattr(bulk_update["duplexio_working_state"], name)
    assert state.user_asr.next_mel_frame == bulk_update["duplexio_working_state"].user_asr.next_mel_frame
    assert state.input_mimi == bulk_update["duplexio_working_state"].input_mimi


def test_trailing_tool_rows_leave_the_live_frame_unchanged() -> None:
    model = model_fixture()
    state = request_state(model)
    state.agent_latent = latent(7)[0]
    state.text_input_ids = (2, 9, 12, 2)
    pcm = torch.ones(1920).numpy().tobytes()
    _, live, live_update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **append_info(state, pcm=pcm))
    _, tool, tool_update = model.preprocess(
        torch.zeros(24, dtype=torch.long), None, **append_info(state, tool=(5, 6, 7), pcm=pcm),
    )
    torch.testing.assert_close(tool[:6], live)
    for key in ("text_ids", "user_features", "agent_audio"):
        torch.testing.assert_close(tool_update["duplexio_replay"][key][:1], live_update["duplexio_replay"][key])
    assert state.input_mimi == 0
    assert model.audio_codec.waveforms == []


@torch.inference_mode()
def test_compacted_scheduler_offsets_do_not_change_model_positions() -> None:
    model = model_fixture()
    state = request_state(model)
    state.frames_seen = 50_000
    info = append_info(state, tool=(6, 7, 8))
    # The scheduler resumes after the live frame, at the tool rows.
    info["duplex_token_offset"] = 6
    info["duplex_prompt_len"] = 24
    _, _, update = model.preprocess(torch.zeros(18, dtype=torch.long), None, **info)
    torch.testing.assert_close(update["duplexio"]["positions"], torch.arange(300_000, 300_018))
    assert update["duplexio_working_state"].frames_seen == 50_003
    assert update["duplexio_working_state"].persistent_keys == 3
