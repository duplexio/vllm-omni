# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm_omni.model_executor.models.duplexio.audio_representation import (
    DelayedMimiState,
)
from vllm_omni.model_executor.models.duplexio.depth_sampler import (
    DepthSpeakerConditioning,
)
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
)
from vllm_omni.model_executor.models.duplexio.mimi import (
    MimiStreamingState,
    MimiTransformerState,
)
from vllm_omni.model_executor.models.duplexio.modeling_duplexio import (
    DuplexIOForConditionalGeneration,
    DuplexIORequestState,
)


def test_gdn_request_state_shape_is_fixed_for_the_session_lifetime() -> None:
    text_config = SimpleNamespace(
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=text_config),
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
    )

    convolution, recurrent = (
        DuplexIOForConditionalGeneration.get_mamba_state_shape_from_config(
            cast(Any, vllm_config)
        )
    )

    assert set(convolution) == {18, 48}
    assert recurrent == (2, 8, 8)


def test_request_state_fork_shares_immutable_prefix_tensors() -> None:
    generator = torch.Generator().manual_seed(7)
    state = DuplexIORequestState(
        text_input_ids=torch.zeros(4, dtype=torch.long),
        agent_audio_codes=torch.zeros(8, dtype=torch.long),
        user_asr=FastConformerAudioStreamState(),
        agent_delay=DelayedMimiState(torch.zeros(7, dtype=torch.long)),
        output_mimi=MimiStreamingState(
            encoder_transformer=MimiTransformerState.empty(2),
            decoder_transformer=MimiTransformerState.empty(2),
        ),
        speaker_embedding=torch.zeros(512),
        depth_speaker_conditioning=DepthSpeakerConditioning(
            attention=(),
            feedforward=(),
        ),
        system_token_ids=(1, 2, 3),
        sampling_generator=generator,
        frames_seen=100_000,
        audio_position=99_000,
        active_text_tokens=250_000,
        cache_epoch=91,
    )

    fork = state.fork()

    assert fork.frames_seen == 100_000
    assert fork.audio_position == 99_000
    assert fork.active_text_tokens == 250_000
    assert fork.text_input_ids.shape == (4,)
    assert fork.agent_audio_codes.shape == (8,)
    assert fork.agent_delay.previous_acoustic_codes.shape == (7,)
    assert fork.speaker_embedding.shape == (512,)
    assert fork.output_mimi is not state.output_mimi
    assert (
        fork.output_mimi.decoder_transformer
        is not state.output_mimi.decoder_transformer
    )
    assert fork.depth_speaker_conditioning is state.depth_speaker_conditioning
