# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace

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
    FastConformerStreamingState,
)
from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiStreamingState,
    MimiTransformerState,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIORequestState,
)


class FakeCodec(nn.Module):
    def __init__(self, frame_size: int, num_codebooks: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.frame_size = frame_size
        self.num_codebooks = num_codebooks

    def new_streaming_state(self) -> MimiStreamingState:
        return MimiStreamingState(
            encoder_transformer=MimiTransformerState.empty(0),
            decoder_transformer=MimiTransformerState.empty(0),
        )

    def encode(
        self,
        waveform: Tensor,
        num_codebooks: int,
        state: MimiStreamingState,
    ) -> Tensor:
        assert num_codebooks == self.num_codebooks
        assert state.encoder_transformer is not None
        frame_count = waveform.shape[-1] // self.frame_size
        start = state.encoder_transformer.position
        frames = torch.arange(
            start,
            start + frame_count,
            device=waveform.device,
        )
        codebooks = torch.arange(num_codebooks, device=waveform.device)
        state.encoder_transformer.position += frame_count
        return (frames[:, None] + codebooks[None, :] + 1).T.unsqueeze(0)


class FakeTextModel(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_size)

    def embed_input_ids(self, input_ids: Tensor) -> Tensor:
        return self.embedding(input_ids)


class FakeUserASREncoder:
    def step_sequence(
        self,
        waveform: Tensor,
        state: FastConformerStreamingState,
    ) -> tuple[Tensor, FastConformerStreamingState]:
        frame_count = waveform.shape[-1] // 4
        frame_ids = torch.arange(
            state.frames_seen,
            state.frames_seen + frame_count,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        features = frame_ids[:, None] + torch.arange(
            4,
            dtype=waveform.dtype,
            device=waveform.device,
        )
        return features, FastConformerStreamingState(
            sample_buffer=state.sample_buffer,
            feature_buffer=state.feature_buffer,
            attention_caches=state.attention_caches,
            convolution_caches=state.convolution_caches,
            frames_seen=state.frames_seen + frame_count,
        )


def model_fixture() -> SimpleNamespace:
    torch.manual_seed(17)
    frame_size = 4
    num_codebooks = 3
    representation_dim = 5
    hidden_size = 11
    codec = FakeCodec(frame_size, num_codebooks)
    representation = DelayedMimiRepresentation(
        num_codebooks=num_codebooks,
        codebook_size=64,
        acoustic_delay_frames=1,
    )
    text_model = FakeTextModel(hidden_size)
    model = SimpleNamespace(
        config=SimpleNamespace(frame_size=frame_size),
        pad_token_id=1,
        silence_token_id=2,
        audio_codec=codec,
        audio_representation=representation,
        user_audio_embedding=MimiEmbedding(
            num_codebooks,
            64,
            representation_dim,
        ),
        agent_audio_embedding=MimiEmbedding(
            num_codebooks,
            64,
            representation_dim,
        ),
        user_asr_proj=nn.Linear(4, representation_dim, bias=False),
        user_asr_encoder=FakeUserASREncoder(),
        user_audio_input_adapter=AudioInputAdapter(
            representation_dim,
            7,
            hidden_size,
        ),
        agent_audio_input_adapter=AgentAudioInputAdapter(
            representation_dim,
            3,
            7,
            hidden_size,
        ),
        llm=SimpleNamespace(
            base_model=SimpleNamespace(model=text_model),
            channel_emb=nn.Parameter(torch.randn(4, hidden_size)),
        ),
    )
    model.encode_silent_agent_frame = MethodType(
        DuplexIOForConditionalGeneration.encode_silent_agent_frame,
        model,
    )
    model.encode_silent_agent_frames = MethodType(
        DuplexIOForConditionalGeneration.encode_silent_agent_frames,
        model,
    )
    return model


def request_state(model: SimpleNamespace) -> DuplexIORequestState:
    agent_mimi = model.audio_codec.new_streaming_state()
    agent_input_delay = model.audio_representation.new_state(
        device=torch.device("cpu")
    )
    agent_audio_codes = DuplexIOForConditionalGeneration.encode_silent_agent_frame(
        model,
        agent_mimi,
        agent_input_delay,
        torch.device("cpu"),
    )
    return DuplexIORequestState(
        text_input_ids=torch.full((4,), 2, dtype=torch.long),
        agent_audio_codes=agent_audio_codes,
        agent_input_delay=agent_input_delay,
        agent_mimi=agent_mimi,
        user_delay=model.audio_representation.new_state(device=torch.device("cpu")),
        agent_delay=model.audio_representation.new_state(device=torch.device("cpu")),
        user_mimi=model.audio_codec.new_streaming_state(),
        output_mimi=model.audio_codec.new_streaming_state(),
        user_asr=FastConformerStreamingState(
            sample_buffer=torch.empty(0),
            feature_buffer=torch.empty(0),
            attention_caches=(),
            convolution_caches=(),
        ),
        user_asr_prefill_features=torch.randn(3, 4),
        speaker_embedding=torch.randn(3),
        system_token_ids=(3, 4, 5),
        sampling_generator=torch.Generator().manual_seed(9),
        cache_epoch=7,
    )


def prefill_info(
    state: DuplexIORequestState,
    frame_count: int,
    *,
    final: bool,
) -> dict[str, object]:
    return {
        "duplexio_model_state": state,
        "duplex": {
            "frame_count": frame_count,
            "duplexio_prefill": True,
            "duplexio_prefill_final": final,
            "duplexio_system_input": False,
            "duplexio_system_input_final": False,
            "runtime_config": {},
        },
    }


def system_input_info(
    state: DuplexIORequestState,
    token_ids: list[int],
) -> dict[str, object]:
    return {
        "duplexio_model_state": state,
        "duplex": {
            "frame_count": len(token_ids),
            "duplexio_prefill": False,
            "duplexio_prefill_final": False,
            "duplexio_system_input": True,
            "duplexio_system_input_final": False,
            "duplexio_system_token_ids": token_ids,
            "runtime_config": {},
        },
    }


def test_bulk_prefill_preprocess_matches_serial_frames() -> None:
    model = model_fixture()
    initial = request_state(model)

    _, bulk_embeddings, bulk_update = DuplexIOForConditionalGeneration.preprocess(
        model,
        torch.zeros(18, dtype=torch.long),
        None,
        **prefill_info(initial.fork(), 3, final=True),
    )
    bulk_state = bulk_update["duplexio_working_state"]
    bulk_metadata = bulk_update["duplexio"]

    serial_state = initial.fork()
    serial_embeddings = []
    serial_metadata: dict[str, list[Tensor]] = {
        "key_active": [],
        "request_epochs": [],
        "text_ordinals": [],
    }
    final_serial_metadata = None
    for index in range(3):
        _, embeddings, update = DuplexIOForConditionalGeneration.preprocess(
            model,
            torch.zeros(6, dtype=torch.long),
            None,
            **prefill_info(serial_state, 1, final=index == 2),
        )
        serial_state = update["duplexio_working_state"]
        serial_embeddings.append(embeddings)
        final_serial_metadata = update["duplexio"]
        for name in serial_metadata:
            serial_metadata[name].append(update["duplexio"][name])

    torch.testing.assert_close(bulk_embeddings, torch.cat(serial_embeddings))
    for name, values in serial_metadata.items():
        torch.testing.assert_close(bulk_metadata[name], torch.cat(values))
    assert final_serial_metadata is not None
    torch.testing.assert_close(
        bulk_metadata["agent_audio_skip"],
        final_serial_metadata["agent_audio_skip"],
    )
    assert isinstance(bulk_state, DuplexIORequestState)
    assert bulk_state.system_token_offset == serial_state.system_token_offset == 3
    assert bulk_state.frames_seen == serial_state.frames_seen == 3
    assert bulk_state.active_text_tokens == serial_state.active_text_tokens == 3
    torch.testing.assert_close(
        bulk_state.agent_audio_codes,
        serial_state.agent_audio_codes,
    )
    torch.testing.assert_close(
        bulk_state.agent_input_delay.previous_acoustic_codes,
        serial_state.agent_input_delay.previous_acoustic_codes,
    )
    assert bulk_state.agent_mimi.encoder_transformer is not None
    assert serial_state.agent_mimi.encoder_transformer is not None
    assert (
        bulk_state.agent_mimi.encoder_transformer.position
        == serial_state.agent_mimi.encoder_transformer.position
        == 3
    )


def test_bulk_system_input_preprocess_matches_serial_frames() -> None:
    model = model_fixture()
    initial = request_state(model)
    token_ids = [6, 7, 8]

    _, bulk_embeddings, bulk_update = DuplexIOForConditionalGeneration.preprocess(
        model,
        torch.zeros(18, dtype=torch.long),
        None,
        **system_input_info(initial.fork(), token_ids),
    )
    bulk_state = bulk_update["duplexio_working_state"]
    bulk_metadata = bulk_update["duplexio"]

    serial_state = initial.fork()
    serial_embeddings = []
    serial_metadata: dict[str, list[Tensor]] = {
        "key_active": [],
        "request_epochs": [],
        "text_ordinals": [],
    }
    final_serial_metadata = None
    for token_id in token_ids:
        _, embeddings, update = DuplexIOForConditionalGeneration.preprocess(
            model,
            torch.zeros(6, dtype=torch.long),
            None,
            **system_input_info(serial_state, [token_id]),
        )
        serial_state = update["duplexio_working_state"]
        serial_embeddings.append(embeddings)
        final_serial_metadata = update["duplexio"]
        for name in serial_metadata:
            serial_metadata[name].append(update["duplexio"][name])

    torch.testing.assert_close(bulk_embeddings, torch.cat(serial_embeddings))
    for name, values in serial_metadata.items():
        torch.testing.assert_close(bulk_metadata[name], torch.cat(values))
    assert final_serial_metadata is not None
    torch.testing.assert_close(
        bulk_metadata["agent_audio_skip"],
        final_serial_metadata["agent_audio_skip"],
    )
    assert isinstance(bulk_state, DuplexIORequestState)
    assert bulk_state.frames_seen == serial_state.frames_seen == 3
    assert bulk_state.active_text_tokens == serial_state.active_text_tokens == 3
    assert bulk_state.user_asr.frames_seen == serial_state.user_asr.frames_seen == 3
    torch.testing.assert_close(
        bulk_state.agent_audio_codes,
        serial_state.agent_audio_codes,
    )
    torch.testing.assert_close(
        bulk_state.agent_input_delay.previous_acoustic_codes,
        serial_state.agent_input_delay.previous_acoustic_codes,
    )
    assert bulk_state.agent_mimi.encoder_transformer is not None
    assert serial_state.agent_mimi.encoder_transformer is not None
    assert (
        bulk_state.agent_mimi.encoder_transformer.position
        == serial_state.agent_mimi.encoder_transformer.position
        == 4
    )
    assert bulk_state.user_mimi.encoder_transformer is not None
    assert serial_state.user_mimi.encoder_transformer is not None
    assert (
        bulk_state.user_mimi.encoder_transformer.position
        == serial_state.user_mimi.encoder_transformer.position
        == 3
    )
