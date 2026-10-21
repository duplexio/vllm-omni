# SPDX-License-Identifier: Apache-2.0
"""Appends become flat frames: text rows leave acoustic state untouched, chunks change nothing."""

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
from vllm_omni.model_executor.models.duplexio.flat_plan import Cursor, Embed, append_frame_count
from vllm_omni.model_executor.models.duplexio.frame_layout import (
    AGENT,
    BOS_AGENT_AUDIO_SLOT,
    BOS_USER_AUDIO_SLOT,
    FRAME_POSITIONS,
    QUERY_SLOT_OFFSET,
    SLOT_COUNT,
    TOOL_CALL,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIORequestState,
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
        # Nemotron's schedule: state k needs 16 kHz audio through 200 samples
        # (k = 0) or 1280 * k + 304, so the resampler's lag delays none.
        heard = state.buffer_start_sample + waveform.numel()
        ready = 0 if heard < 200 else 1 + max(0, heard - 304) // 1280
        positions = torch.arange(state.next_mel_frame + 1, ready + 1)
        return positions.float()[None, :, None].expand(1, -1, self.output_dim), replace(
            state, next_mel_frame=ready, buffer_start_sample=heard
        )

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
            latents.append(
                torch.arange(position + 1, position + frames + 1).float()[None, None].expand(1, LATENT_DIM, -1)
            )
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
    model.step_inputs = None
    # Audio keys see a two-frame window here, so eviction shows up in the test.
    model.config = SimpleNamespace(audio_attention_window_frames=2, flowmap_config={"sampling_temperature": 1.0})
    model.text_config = SimpleNamespace(max_position_embeddings=262144, vocab_size=32)
    model.vllm_config = SimpleNamespace(model_config=SimpleNamespace(dtype=torch.float32, max_model_len=4096))
    model.audio_representation = ContinuousAudioRepresentation(LATENT_DIM)
    model.user_audio_input_adapter = AudioInputAdapter(8, 7, 11)
    model.agent_audio_input_adapter = AudioInputAdapter(LATENT_DIM, 7, 11)
    model.user_asr = CountingASR()
    model.user_asr.output_dim = 8
    model.user_audio_resampler = Resample(24_000, 16_000, dtype=torch.float32)
    model.audio_codec = CountingCodec()
    model.model = TextEmbedding(11)
    model.channel_emb = nn.Parameter(torch.randn(4, 11))
    model.flat_cell_emb = nn.Parameter(torch.randn(SLOT_COUNT, 11))
    return model


def request_state(model: DuplexIORequestState) -> DuplexIORequestState:
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
    state: DuplexIORequestState,
    *,
    prefix: bool = False,
    tool: tuple[int, ...] = (),
    pcm: bytes | None = None,
):
    """One whole append: the opening (voice prompt and system tokens) if prefix, else a live frame; then tools."""
    frames = append_frame_count(2, len(state.system_token_ids), len(tool), opening=prefix)
    return {
        "duplexio_model_state": state,
        "duplex_token_offset": 0,
        "duplex_prompt_len": frames * FRAME_POSITIONS,
        "duplex": {
            "frame_count": frames,
            "duplexio_prefix": prefix,
            "duplexio_tool_token_ids": list(tool),
            "duplexio_tool_generation": 0,
            "duplexio_given_frame": None,
            "decode_audio": True,
            "epoch": 0,
            "turn_id": 0,
            "final": False,
            "pcm": b"" if prefix else torch.zeros(1920).numpy().tobytes() if pcm is None else pcm,
            "runtime_config": {"duplexio_record_inputs": True, "duplexio_record_hiddens": False},
        },
    }


def given_info(state: DuplexIORequestState, agent_pcm: Tensor, *, tool: tuple[int, ...] = (), **frame: int):
    """An append whose live frame replays history: ``frame`` names its given token ids."""
    info = append_info(state, tool=tool)
    info["duplex"]["duplexio_given_frame"] = {
        "user_token_id": 2,
        "agent_token_id": 2,
        "tool_call_token_id": 2,
        **frame,
        "agent_pcm": agent_pcm.numpy().tobytes(),
    }
    return info


def preprocess(model, info, tokens: int | None = None):
    tokens = info["duplex_prompt_len"] - info["duplex_token_offset"] if tokens is None else tokens
    _, embeddings, update = model.preprocess(torch.zeros(tokens, dtype=torch.long), None, **info)
    return embeddings, update, update["duplexio_replay"]


def first_pass(model):
    """The installed pass's positions, by request: (embed kind, value) per position."""
    inputs = model.step_inputs
    return [[(kind, value) for kind, value, *_ in inputs.embed[start:end]] for start, end, _, _ in inputs.requests]


def opened(model) -> DuplexIORequestState:
    """A session after its opening append."""
    _, update, _ = preprocess(model, append_info(request_state(model), prefix=True))
    return update["duplexio_working_state"]


@torch.inference_mode()
def test_the_opening_lays_out_prompt_and_system_rows_then_the_bos_frame_deciding_row_zero() -> None:
    model = model_fixture()
    embeddings, update, replay = preprocess(model, append_info(request_state(model), prefix=True))
    # Rows, not positions: two prompt rows and three system rows; no live row yet.
    assert replay["text_ids"].tolist() == [[2, 2, 2, 2]] * 2 + [[3, 2, 2, 2], [4, 2, 2, 2], [5, 2, 2, 2]]
    assert replay["audio_mask"].tolist() == [False] * 5
    assert replay["prompt_frames"].tolist() == [True, True, False, False, False]
    torch.testing.assert_close(replay["agent_audio"][:2], latent(1, 2))
    (positions,) = first_pass(model)
    assert positions == [
        (Embed.AGENT_AUDIO, 0),
        (Embed.AGENT_AUDIO, 1),
        (Embed.TOKEN, 3),
        (Embed.TOKEN, 4),
        (Embed.TOKEN, 5),
        (Embed.NOOP, 0),
        (Embed.CELL, QUERY_SLOT_OFFSET + AGENT),
        (Embed.CELL, QUERY_SLOT_OFFSET + TOOL_CALL),
        (Embed.CELL, BOS_USER_AUDIO_SLOT),
        (Embed.CELL, BOS_AGENT_AUDIO_SLOT),
    ]
    # The append ends at the BOS frame: its last frame decides live row 0.
    assert embeddings.shape[0] == 10
    torch.testing.assert_close(embeddings[0], model.agent_audio_input_adapter(latent(1))[0])
    torch.testing.assert_close(embeddings[2], model.model.embed_input_ids(torch.tensor([3]))[0] + model.channel_emb[0])
    torch.testing.assert_close(embeddings[6], model.flat_cell_emb[QUERY_SLOT_OFFSET + AGENT])
    state = update["duplexio_working_state"]
    assert state.cursor == Cursor(
        row=5, live_rows=0, ring_keys=0, bos_clocks=[1], bos_keys=2, persistent_keys=5, gdn_fresh=False
    )
    # The user encoder heard silence under the voice prompt; no user audio was consumed.
    assert state.user_asr.next_mel_frame == 2
    assert state.history_open
    assert state.input_mimi == 2
    torch.testing.assert_close(model.audio_codec.waveforms[0], torch.ones(2 * 1920))


@torch.inference_mode()
def test_tool_rows_follow_the_live_frame_as_text_then_a_bos_frame_decides() -> None:
    model = model_fixture()
    initial = request_state(model)
    initial.cursor = Cursor(
        row=6, live_rows=1, ring_keys=2, bos_clocks=[1], bos_keys=2, persistent_keys=5, gdn_fresh=False
    )
    initial.text_input_ids = (2, 9, 12, 2)
    initial.agent_latent = latent(3)[0]
    embeddings, update, replay = preprocess(model, append_info(initial, tool=(6, 7, 8)))
    assert replay["text_ids"].tolist() == [[2, 9, 12, 2], [6, 2, 2, 2], [7, 2, 2, 2], [8, 2, 2, 2]]
    assert replay["audio_mask"].tolist() == [True, False, False, False]
    torch.testing.assert_close(replay["agent_audio"][0], latent(3)[0])
    (positions,) = first_pass(model)
    assert [kind for kind, _ in positions] == [
        Embed.TOKEN,
        Embed.TOKEN,
        Embed.CELL,
        Embed.USER_AUDIO,
        Embed.AGENT_AUDIO,
        Embed.TOKEN,
        Embed.TOKEN,
        Embed.TOKEN,
        Embed.NOOP,
        Embed.NOOP,
        Embed.NOOP,
        Embed.CELL,
        Embed.CELL,
        Embed.CELL,
        Embed.CELL,
    ]
    state = update["duplexio_working_state"]
    # The live row's user and agent tokens, then three tool tokens; a BOS frame for the next row.
    assert (state.cursor.persistent_keys, state.cursor.row, state.cursor.live_rows) == (10, 10, 2)
    assert state.cursor.bos_clocks == [1, 3]
    assert state.user_asr.next_mel_frame == 1 and initial.user_asr.next_mel_frame == 0
    assert initial.cursor.persistent_keys == 5
    assert model.audio_codec.waveforms == []
    torch.testing.assert_close(embeddings[5], model.model.embed_input_ids(torch.tensor([6]))[0] + model.channel_emb[0])


@torch.inference_mode()
def test_live_frame_embeds_its_tokens_queries_and_audio() -> None:
    model = model_fixture()
    state = request_state(model)
    state.cursor = Cursor(
        row=6, live_rows=1, ring_keys=2, bos_clocks=[1], bos_keys=2, persistent_keys=5, gdn_fresh=False
    )
    state.text_input_ids = (2, 9, 2, 2)
    state.agent_latent = latent(3)[0]
    embeddings, update, replay = preprocess(model, append_info(state, pcm=torch.randn(1920).numpy().tobytes()))
    torch.testing.assert_close(embeddings[0], model.model.embed_input_ids(torch.tensor([9]))[0] + model.channel_emb[1])
    torch.testing.assert_close(embeddings[1], model.flat_cell_emb[QUERY_SLOT_OFFSET + AGENT])
    torch.testing.assert_close(embeddings[2], model.flat_cell_emb[QUERY_SLOT_OFFSET + TOOL_CALL])
    torch.testing.assert_close(embeddings[3], model.user_audio_input_adapter(torch.ones(1, 8))[0])
    torch.testing.assert_close(embeddings[4], model.agent_audio_input_adapter(latent(3))[0])
    assert replay["text_ids"].tolist() == [[2, 9, 2, 2]]
    assert update["duplexio_working_state"].cursor.live_rows == 2 and state.cursor.live_rows == 1


@torch.inference_mode()
def test_given_frames_replay_history_through_the_voice_prompt_codec_stream() -> None:
    model = model_fixture()
    first, second = torch.randn(1920), torch.randn(1920)
    _, update, replay = preprocess(model, given_info(opened(model), first, user_token_id=9, agent_token_id=12))
    assert replay["text_ids"].tolist() == [[2, 9, 12, 2]]
    # The codec hears the voice prompt, then the given agent frame, as one stream.
    torch.testing.assert_close(replay["agent_audio"], latent(3))
    torch.testing.assert_close(model.audio_codec.waveforms[1], first)
    state = update["duplexio_working_state"]
    assert state.input_mimi == 3
    assert state.cursor.persistent_keys == 7  # Two prompt keys, three system tokens, the given user and agent text.
    # A sampled prediction is waiting; the next given frame replaces it.
    state.text_input_ids = (2, 5, 6, 2)
    state.agent_latent = latent(7)[0]
    _, update, replay = preprocess(model, given_info(state, second, agent_token_id=13))
    assert replay["text_ids"].tolist() == [[2, 2, 13, 2]]
    torch.testing.assert_close(replay["agent_audio"], latent(4))
    torch.testing.assert_close(model.audio_codec.waveforms[2], second)
    assert update["duplexio_working_state"].cursor.live_rows == 2


@torch.inference_mode()
def test_given_frames_must_lead_and_hold_no_tool_calls() -> None:
    model = model_fixture()
    _, update, _ = preprocess(model, append_info(opened(model)))
    sampled = update["duplexio_working_state"]
    assert not sampled.history_open
    with pytest.raises(ValueError, match="lead the conversation"):
        preprocess(model, given_info(sampled, torch.zeros(1920)))
    with pytest.raises(ValueError, match="tool calls"):
        preprocess(model, given_info(opened(model), torch.zeros(1920), tool_call_token_id=6))
    with pytest.raises(ValueError, match="one 1920-sample"):
        preprocess(model, given_info(opened(model), torch.zeros(960)))


@pytest.mark.parametrize(
    "frames,frames_seen,match", [(1, 0, "do not match"), (3, 0, "do not match"), (2, 2, "exactly once")]
)
def test_prefix_must_be_complete_and_consumed_once(frames: int, frames_seen: int, match: str) -> None:
    model = model_fixture()
    state = request_state(model)
    state.frames_seen = frames_seen
    info = append_info(state, prefix=True)
    info["duplex"]["frame_count"] = frames
    info["duplex_prompt_len"] = frames * FRAME_POSITIONS
    with pytest.raises(ValueError, match=match):
        preprocess(model, info)


@pytest.mark.parametrize("prefix,chunks", [(True, (1, 1)), (False, (1, 1, 2)), (False, (2, 2)), (False, (3, 1))])
@torch.inference_mode()
def test_scheduler_chunks_preserve_embeddings_and_replay(prefix: bool, chunks: tuple[int, ...]) -> None:
    model = model_fixture()
    initial = request_state(model)
    if prefix:
        info = append_info(initial, prefix=True)
    else:
        initial.cursor = Cursor(
            row=7, live_rows=1, ring_keys=2, bos_clocks=[1], bos_keys=2, persistent_keys=6, gdn_fresh=False
        )
        info = given_info(initial, torch.randn(1920), tool=(3, 4, 5, 6, 7, 8), user_token_id=9)
    assert sum(chunks) == info["duplex"]["frame_count"]
    model.audio_codec = CountingCodec()
    bulk, bulk_update, bulk_replay = preprocess(model, info)
    bulk_positions = first_pass(model)[0]
    model.audio_codec = CountingCodec()
    state, embeddings, replays, positions = initial, [], [], []
    offset = 0
    for count in chunks:
        info["duplexio_model_state"] = state
        info["duplex_token_offset"] = offset
        hidden, update, replay = preprocess(model, info, count * FRAME_POSITIONS)
        offset += count * FRAME_POSITIONS
        state = update["duplexio_working_state"]
        embeddings.append(hidden)
        replays.append(replay)
        positions += first_pass(model)[0]
    torch.testing.assert_close(torch.cat(embeddings), bulk)
    assert positions == bulk_positions
    # A chunk of only the BOS frame records no row.
    for key, expected in bulk_replay.items():
        torch.testing.assert_close(torch.cat([replay[key] for replay in replays]), expected)
    assert state.cursor == bulk_update["duplexio_working_state"].cursor
    assert state.user_asr.next_mel_frame == bulk_update["duplexio_working_state"].user_asr.next_mel_frame
    assert state.input_mimi == bulk_update["duplexio_working_state"].input_mimi
