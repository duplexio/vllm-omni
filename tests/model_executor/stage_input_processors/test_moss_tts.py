# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.moss_tts import (
    talker2codec_delay_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

AUDIO_PAD_CODE = 1024


def delayed_codes(raw: torch.Tensor) -> torch.Tensor:
    frames, codebooks = raw.shape
    delayed = torch.full(
        (frames + codebooks - 1, codebooks),
        AUDIO_PAD_CODE,
        dtype=torch.long,
    )
    for codebook in range(codebooks):
        delayed[codebook : codebook + frames, codebook] = raw[:, codebook]
    return delayed


def request_with_codec_context(context: torch.Tensor | None) -> SimpleNamespace:
    info = None
    if context is not None:
        info = serialize_additional_information({"codes": {"audio": context}})
    return SimpleNamespace(request_id="req-1", additional_information=info)


def test_delay_processor_decodes_continuation_with_causal_codec_context() -> None:
    context = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    generated = torch.tensor([[10, 20], [11, 21], [12, 22]], dtype=torch.long)

    payload = talker2codec_delay_async_chunk(
        SimpleNamespace(),
        {"codes": {"audio": delayed_codes(generated)}},
        request_with_codec_context(context),
        is_finished=True,
    )

    assert payload is not None
    assert payload.meta is not None
    assert payload.meta.left_context_size == context.shape[0]
    assert payload.codes is not None
    assert payload.codes.audio == [1, 3, 10, 11, 12, 2, 4, 20, 21, 22]


def test_delay_processor_without_context_decodes_only_generated_codes() -> None:
    generated = torch.tensor([[10, 20], [11, 21]], dtype=torch.long)

    payload = talker2codec_delay_async_chunk(
        SimpleNamespace(),
        {"codes": {"audio": delayed_codes(generated)}},
        request_with_codec_context(None),
        is_finished=True,
    )

    assert payload is not None
    assert payload.meta is not None
    assert payload.meta.left_context_size == 0
    assert payload.codes is not None
    assert payload.codes.audio == [10, 11, 20, 21]


def test_delay_processor_rejects_incomplete_audio_frame() -> None:
    generated = torch.tensor([[10, 20], [11, 21]], dtype=torch.long)
    delayed = delayed_codes(generated)
    delayed[1, 1] = AUDIO_PAD_CODE

    with pytest.raises(RuntimeError, match="delay tail did not finish cleanly"):
        talker2codec_delay_async_chunk(
            SimpleNamespace(),
            {"codes": {"audio": delayed}},
            request_with_codec_context(None),
            is_finished=True,
        )
