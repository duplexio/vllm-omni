# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import torch

from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
)
from vllm_omni.model_executor.models.duplexio.flat_plan import Cursor
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

    convolution, recurrent = DuplexIOForConditionalGeneration.get_mamba_state_shape_from_config(cast(Any, vllm_config))

    assert set(convolution) == {18, 48}
    # Two value heads per TP rank, one recurrent group per stream.
    assert recurrent == (2 * 6, 8, 8)


def test_request_state_fork_shares_immutable_prefix_and_codec_state() -> None:
    codec_state = object()  # Pocket Mimi states are replaced, never mutated.
    sampling = object()  # Fixed when the session starts.
    state = DuplexIORequestState(
        text_input_ids=(0, 0, 0, 0),
        agent_latent=torch.zeros(32),
        user_asr=FastConformerAudioStreamState(),
        input_mimi=codec_state,
        output_mimi=codec_state,
        voice_prompt=torch.zeros(1_920),
        system_token_ids=(1, 2, 3),
        sampling=sampling,
        frames_seen=100_000,
        cursor=Cursor(row=101_000, live_rows=99_000, bos_clocks=[1, 50], persistent_keys=250_000),
    )

    fork = state.fork()
    fork.frames_seen += 1
    fork.cursor.live_rows += 1
    fork.cursor.bos_clocks.append(99_001)

    assert state.frames_seen == 100_000
    # The cursor is the request's addressing: a failed step must not advance it.
    assert (state.cursor.live_rows, state.cursor.bos_clocks) == (99_000, [1, 50])
    assert (fork.cursor.persistent_keys, fork.text_input_ids) == (250_000, (0, 0, 0, 0))
    # The pinned prompt is immutable for the session, so a fork shares it.
    assert fork.voice_prompt is state.voice_prompt
    assert fork.output_mimi is codec_state
    assert fork.sampling is sampling
