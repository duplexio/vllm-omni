# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Offline streams open sessions as realtime serving does and return each append's raw outputs."""

from types import SimpleNamespace

import pytest
from vllm.sampling_params import RequestOutputKind, SamplingParams

from tests.model_executor.models.duplexio.test_duplexio_plugin import (  # noqa: F401 (fixtures)
    REFERENCE_FRAMES,
    byte_tokenizer,
    model_config,
    open_session,
    pack_frame,
    pcm,
    session_config,
)
from vllm_omni.model_executor.models.duplexio.duplex import DuplexIOStream
from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class EchoEngine:
    """Answers every append with one non-predicting chunk, then its prediction."""

    def __init__(self, config) -> None:
        self.model_config = config
        self.default_sampling_params_list = [SamplingParams(max_tokens=100)]
        self.appends: list[dict] = []

    async def generate(self, inputs, *, sampling_params_list, request_id):
        (params,) = sampling_params_list
        assert params.max_tokens == 1 and params.output_kind == RequestOutputKind.DELTA
        async for item in inputs:
            self.appends.append(item.prompt)
            yield SimpleNamespace(multimodal_output={"frame": pack_frame()})
            yield SimpleNamespace(multimodal_output={"frame": pack_frame(predicted=True)})


@pytest.fixture
def stream_tokenizer(monkeypatch, model_config):  # noqa: F811
    # The stream tokenizes tool results with the same tokenizer the plugin renders prompts with.
    from vllm_omni.model_executor.models.duplexio.duplex import plugin

    monkeypatch.setattr(
        "vllm_omni.model_executor.models.duplexio.duplex.stream.cached_tokenizer_from_config",
        plugin.cached_tokenizer_from_config,
    )
    return model_config


@pytest.mark.asyncio
async def test_stream_opens_the_session_realtime_serving_would(stream_tokenizer) -> None:
    session = session_config(start_role="agent", duplexio_sampling={"agent": {"content": {"temperature": 0.2}}})
    stream = DuplexIOStream.open(EchoEngine(stream_tokenizer), "offline", session, record_inputs=True)
    _, served = await open_session(
        stream_tokenizer, start_role="agent", duplexio_sampling={"agent": {"content": {"temperature": 0.2}}}
    )
    assert stream.runtime_config == {**served, "duplexio_record_inputs": True}
    await stream.close()


@pytest.mark.asyncio
async def test_appends_follow_the_realtime_layout_and_collect_through_the_prediction(stream_tokenizer) -> None:
    engine = EchoEngine(stream_tokenizer)
    stream = DuplexIOStream.open(engine, "offline", session_config())
    prefix_frames = REFERENCE_FRAMES + len(stream.runtime_config["duplexio_system_token_ids"])
    assert stream.prefix_rows == prefix_frames and stream.max_rows == 6 * 4096 // FRAME_POSITIONS

    stream.append(pcm(1))
    assert [frame_fields(output)["predicted"] for output in await stream.collect()] == [False, True]
    # Tool results arrive as text and are rendered as realtime serving renders them.
    assert stream.tool_result_rows("sunny") == 3
    stream.append(pcm(1), tool_results=["sunny"], final=True)
    await stream.collect()
    tool = [20, 21, 22]

    first, second = (append["model_intermediate_buffer"]["duplex"] for append in engine.appends)
    frames = -(-prefix_frames // FRAME_POSITIONS) + 2
    assert first["duplexio_prefix"] and first["frame_count"] == frames
    assert len(engine.appends[0]["prompt_token_ids"]) == frames * FRAME_POSITIONS
    # The live frame, the tool result's text frame, then a BOS frame.
    assert not second["duplexio_prefix"] and second["frame_count"] == 3
    assert (second["duplexio_tool_token_ids"], second["duplexio_tool_generation"], second["final"]) == (tool, 1, True)
    assert first["runtime_config"]["duplexio_record_inputs"] is False
    await stream.close()
