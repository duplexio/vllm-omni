# SPDX-License-Identifier: Apache-2.0
"""Text-only prefixes use masked initial audio, never advance codec state."""

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor, nn

from vllm_omni.model_executor.models.duplexio.audio_adapters import (
    AgentAudioInputAdapter,
    AudioInputAdapter,
)
from vllm_omni.model_executor.models.duplexio.audio_representation import (
    DelayedMimiRepresentation,
    MimiEmbedding,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
)
from vllm_omni.model_executor.models.duplexio.mimi import MimiStreamingState
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


def model_fixture() -> DuplexIOForConditionalGeneration:
    torch.manual_seed(17)
    model = DuplexIOForConditionalGeneration.__new__(DuplexIOForConditionalGeneration)
    nn.Module.__init__(model)
    model.pad_token_id = 1
    model.silence_token_id = 2
    model.full_cudagraph_enabled = False
    model.frame_inputs = frame_inputs
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
    model.agent_audio_input_adapter = AgentAudioInputAdapter(5, 3, 7, 11)
    model.user_asr = SimpleNamespace(output_dim=8)
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
        speaker_embedding=torch.randn(3),
        depth_speaker_conditioning=None,
        system_token_ids=(3, 4, 5),
        sampling_generator=torch.Generator().manual_seed(9),
        cache_epoch=7,
    )


def input_info(
    state: DuplexIORequestState, tokens: list[int], *, system: bool, final: bool
):
    return {
        "duplexio_model_state": state,
        "duplex": {
            "frame_count": len(tokens),
            "duplexio_prefill": not system,
            "duplexio_prefill_final": final and not system,
            "duplexio_system_input": system,
            "duplexio_system_input_final": final and system,
            "duplexio_system_token_ids": tokens,
            "runtime_config": {},
        },
    }


@torch.inference_mode()
def test_live_user_projection_batches_independent_requests_and_excludes_prefixes() -> None:
    model = model_fixture()
    features = torch.randn(3, 8)
    requests = {
        str(index): {
            "embed": {"speech_feat": row[None]},
            "duplex": {"payload": {"format": "duplexio_features"}},
        }
        for index, row in enumerate(features)
    }
    requests["prefix"] = input_info(request_state(model), [3, 4, 5], system=False, final=True)
    model.preprocess_batch(req_ids=list(requests), model_intermediate_buffer=requests, device=torch.device("cpu"))
    for index, row in enumerate(features):
        torch.testing.assert_close(requests[str(index)]["embed"]["user_hidden"], model.user_audio_input_adapter(row[None]))
    assert "embed" not in requests["prefix"]


@pytest.mark.parametrize("system", [False, True])
@torch.inference_mode()
def test_text_only_bulk_and_serial_frames_are_identical(system: bool) -> None:
    model = model_fixture()
    initial = request_state(model)
    tokens = [6, 7, 8] if system else [3, 4, 5]
    _, bulk, bulk_update = model.preprocess(
        torch.zeros(18, dtype=torch.long),
        None,
        **input_info(initial, tokens, system=system, final=True),
    )
    serial_state = initial
    embeddings = []
    metadata = {
        key: []
        for key in (
            "key_active",
            "request_epochs",
            "text_ordinals",
            "audio_positions",
        )
    }
    for index, token in enumerate(tokens):
        _, hidden, update = model.preprocess(
            torch.zeros(6, dtype=torch.long),
            None,
            **input_info(serial_state, [token], system=system, final=index == 2),
        )
        serial_state = update["duplexio_working_state"]
        embeddings.append(hidden)
        for key in metadata:
            metadata[key].append(update["duplexio"][key])
    torch.testing.assert_close(bulk, torch.cat(embeddings))
    for key, values in metadata.items():
        torch.testing.assert_close(bulk_update["duplexio"][key], torch.cat(values))

    state = bulk_update["duplexio_working_state"]
    assert state.frames_seen == serial_state.frames_seen == 3
    assert state.active_text_tokens == serial_state.active_text_tokens == 3
    assert state.audio_position == serial_state.audio_position == 0
    assert state.user_asr is initial.user_asr
    assert state.user_asr.encoder.past_key_values is None
    assert not bulk_update["duplexio"]["key_active"].view(3, 6)[:, 4:].any()
    torch.testing.assert_close(state.agent_audio_codes, model.initial_agent_audio(1)[0])


@torch.inference_mode()
def test_cached_live_frame_inserts_user_token_and_generated_agent_feedback() -> None:
    model = model_fixture()
    state = request_state(model)
    state.system_token_offset = 3
    state.text_input_ids[2] = 12
    state.agent_audio_codes = torch.tensor([3, 4, 5])
    user_features = torch.randn(1, 8)
    info = dict(
        duplexio_model_state=state,
        embed={"speech_feat": user_features},
        duplex={
            "frame_count": 1,
            "runtime_config": {"duplexio_record_inputs": True},
            "payload": {"format": "duplexio_features"},
            "user_token_id": 9,
        },
    )
    model.preprocess_batch(req_ids=["live"], model_intermediate_buffer={"live": info}, device=torch.device("cpu"))
    _, embeddings, update = model.preprocess(torch.zeros(6, dtype=torch.long), None, **info)
    expected_user = model.llm.base_model.model.embed_input_ids(torch.tensor([9]))[0]
    torch.testing.assert_close(embeddings[1], expected_user + model.llm.channel_emb[1])
    torch.testing.assert_close(
        embeddings[4:5], model.user_audio_input_adapter(user_features)
    )
    torch.testing.assert_close(
        embeddings[5:6],
        model.agent_audio_input_adapter(
            model.agent_audio_embedding(state.agent_audio_codes[None]),
            state.speaker_embedding[None],
        ),
    )
    assert update["duplexio_working_state"].audio_position == 1
    assert state.audio_position == 0
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


def test_recorded_prefix_contains_only_system_text_and_masked_audio() -> None:
    model = model_fixture()
    info = input_info(request_state(model), [3, 4, 5], system=False, final=True)
    info["duplex"]["runtime_config"]["duplexio_record_inputs"] = True
    _, _, update = model.preprocess(torch.zeros(18, dtype=torch.long), None, **info)
    replay = update["duplexio_replay"]
    assert replay["text_ids"].tolist() == [[3, 2, 2, 2], [4, 2, 2, 2], [5, 2, 2, 2]]
    assert not replay["audio_mask"].any()
    assert not replay["user_features"].any()
    torch.testing.assert_close(replay["agent_audio"], model.initial_agent_audio(3))
