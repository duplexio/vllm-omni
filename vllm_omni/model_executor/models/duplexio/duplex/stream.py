# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One DuplexIO session driven frame by frame on an in-process engine, returning raw frame outputs.

Realtime sessions project outputs into Realtime events; offline drivers such as
evaluations and rollouts need each frame's sampled ids, log probabilities and,
when recording, its inputs. The session opens exactly as a realtime one does.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from torch import Tensor
from vllm.engine.protocol import StreamingInput
from vllm.tokenizers import cached_tokenizer_from_config

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.model_executor.models.duplexio.duplex.plugin import (
    GivenFrame,
    append_fields,
    session_runtime_config,
    stage_sampling_params,
    tool_result_token_ids,
)
from vllm_omni.model_executor.models.duplexio.frame_output import frame_fields

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni


class DuplexIOStream:
    """Append frames in order and collect each append's outputs.

    An append is ``[voice prompt, system tokens]`` (the first one only), one
    live frame, then any tool-result tokens, as realtime serving plans it. The
    scheduler may split an append into chunks: each chunk yields one output with
    its rows, and the append's last chunk is the one that predicts.
    """

    def __init__(
        self, engine: AsyncOmni, request_id: str, runtime_config: dict[str, Any], *, timeout: float = 120.0
    ) -> None:
        self.runtime_config = runtime_config
        self.request_id = request_id
        self.timeout = timeout
        self.tokenizer = cached_tokenizer_from_config(engine.model_config)
        self.inputs: asyncio.Queue[StreamingInput] = asyncio.Queue()
        self.appends = 0
        self.tool_generation = 0
        params = stage_sampling_params(tuple(engine.default_sampling_params_list))
        self.outputs = engine.generate(self.stream(), sampling_params_list=list(params), request_id=request_id)

    @classmethod
    def open(
        cls,
        engine: AsyncOmni,
        request_id: str,
        session: DuplexSessionConfig,
        *,
        record_inputs: bool = False,
        record_hiddens: bool = False,
        timeout: float = 120.0,
    ) -> DuplexIOStream:
        """Open a session from the same configuration a realtime client sends.

        ``record_inputs`` returns each frame's model inputs with its outputs, so
        a trajectory can be replayed; ``record_hiddens`` also returns the
        predictor's hidden states.
        """
        runtime_config = session_runtime_config(session, engine.model_config)
        runtime_config["duplexio_record_inputs"] = record_inputs
        runtime_config["duplexio_record_hiddens"] = record_hiddens
        return cls(engine, request_id, runtime_config, timeout=timeout)

    async def stream(self) -> AsyncGenerator[StreamingInput, None]:
        while True:
            yield await self.inputs.get()

    def tool_result_token_ids(self, output: str) -> list[int]:
        """The rows a tool result is fed as, exactly as realtime serving feeds it."""
        return tool_result_token_ids(self.tokenizer, output)

    def append(
        self,
        pcm: bytes,
        *,
        tool_token_ids: Sequence[int] = (),
        final: bool = False,
        decode_audio: bool = False,
        given: GivenFrame | None = None,
    ) -> None:
        """Queue one live frame of float32 user audio, after the prefix if it is the first.

        ``tool_token_ids`` (from ``tool_result_token_ids``) follow the live frame.
        A ``given`` frame replays history: the model hears its tokens and agent
        audio instead of its own last prediction. Given frames lead the stream.
        """
        if tool_token_ids:
            self.tool_generation += 1
        prompt_token_ids, fields = append_fields(
            self.runtime_config,
            pcm,
            prefix=self.appends == 0,
            tool_token_ids=list(tool_token_ids),
            tool_generation=self.tool_generation if tool_token_ids else 0,
            decode_audio=decode_audio,
            given_frame=given,
        )
        self.appends += 1
        duplex = {**fields, "runtime_config": self.runtime_config, "epoch": 0, "turn_id": 0, "final": final}
        self.inputs.put_nowait(
            StreamingInput(
                prompt={
                    "prompt_token_ids": prompt_token_ids,
                    "model_intermediate_buffer": {
                        "request_id": self.request_id,
                        "global_request_id": [self.request_id],
                        "duplex": duplex,
                    },
                }
            )
        )

    async def collect(self) -> list[Mapping[str, Tensor]]:
        """The oldest uncollected append's outputs, through the one that predicts."""
        outputs = []
        async with asyncio.timeout(self.timeout):
            while True:
                output = await anext(self.outputs)
                outputs.append(output.multimodal_output)
                if frame_fields(outputs[-1])["predicted"]:
                    return outputs

    async def close(self) -> None:
        """Abort the request; its caches are freed."""
        await self.outputs.aclose()
