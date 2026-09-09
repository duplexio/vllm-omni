# SPDX-License-Identifier: Apache-2.0
"""Unpaced, concurrent conversations through the native duplex engine."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from vllm_omni.experimental.fullduplex.duplexio.tool_simulator import ToolSimulator, decode_tool_calls
from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.experimental.fullduplex.engine.contracts import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase

    from vllm_omni.entrypoints.async_omni import AsyncOmni

SYSTEM_INPUT_CHUNK_FRAMES = 128


@dataclass
class ToolParticipant:
    """Answers the student's tool calls mid-session with simulated results.

    `max_session_rows` is the engine's request budget in frame rows
    (max_model_len / 6 cells); results that would not fit, or arrive after
    `max_calls` answered calls, are dropped and the call stays unanswered.
    """

    simulator: ToolSimulator
    tokenizer: PreTrainedTokenizerBase
    silence_token_id: int
    pad_token_id: int
    max_session_rows: int = 4096
    max_calls: int = 8
    max_result_tokens: int = 400


class PreparedConversation(BaseModel):
    """CPU streaming encoder outputs and transcript tokens at their arrival rows."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    conversation_id: str
    system_token_ids: list[int] = Field(min_length=1)
    user_features: Tensor
    user_token_ids: Tensor
    voice: str
    voice_embedding_index: int = Field(default=0, ge=0)
    tools: list[dict[str, Any]]
    metadata: dict[str, Any]

    @model_validator(mode="after")
    def validate_frames(self) -> PreparedConversation:
        features, tokens = self.user_features, self.user_token_ids
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("User features must have shape (positive frames, ASR dimension)")
        if tokens.shape != (features.shape[0],):
            raise ValueError("One user token is required per input feature frame")
        if features.dtype != torch.float32 or tokens.dtype != torch.long:
            raise ValueError("Prepared features must retain float32 outputs and user token IDs int64")
        if features.device.type != "cpu" or tokens.device.type != "cpu":
            raise ValueError("Prepared conversation files contain CPU tensors")
        if (tokens < 0).any() or any(token < 0 for token in self.system_token_ids):
            raise ValueError("Input token IDs must be nonnegative")
        return self


@dataclass(frozen=True)
class PoolConversation:
    """Index entry of a rollout pool built by duplexio's prepare_opd_pool."""

    conversation_id: str
    file_stem: str
    frames: int


def load_pool_shard(pool_dir: Path, shard: int, shards: int) -> list[PoolConversation]:
    """Return this actor's slice of the pool index, sorted by conversation ID."""
    index = json.loads((pool_dir / "index.json").read_text())
    entries = sorted(
        (PoolConversation(c["conversation_id"], c["file_stem"], c["frames"]) for c in index["conversations"]),
        key=lambda entry: entry.conversation_id,
    )
    return entries[shard::shards]


def prepared_from_pool(pool_dir: Path, entry: PoolConversation) -> PreparedConversation:
    """Load one conversation's inputs when its session starts."""
    from safetensors.torch import load_file

    info = json.loads((pool_dir / f"{entry.file_stem}.json").read_text())
    tensors = load_file(pool_dir / f"{entry.file_stem}.safetensors")
    return PreparedConversation(
        conversation_id=info["conversation_id"],
        system_token_ids=tensors["system_token_ids"].tolist(),
        user_features=tensors["user_features"].float(),
        user_token_ids=tensors["user_token_ids"],
        voice=info["voice"],
        voice_embedding_index=info["voice_embedding_index"],
        tools=info["tools"],
        metadata={**info["metadata"], "teacher_system": info["teacher_system"]},
    )


def conversation_seed(base_seed: int, conversation_id: str) -> int:
    """Keep RNG identity stable across batching, reordering, and GPU sharding."""
    digest = hashlib.blake2b(conversation_id.encode(), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest)) % (2**63 - 1)


class RolloutGate:
    """Pause point shared by every session of one actor.

    Weights change only while no engine input is outstanding, so each prediction
    row belongs to exactly one policy version. Sessions keep their caches and
    continue under the new weights.
    """

    def __init__(self, version: int = 0) -> None:
        self.version = version
        self.open = asyncio.Event()
        self.open.set()
        self.idle = asyncio.Event()
        self.idle.set()
        self.outstanding = 0

    def submitted(self) -> None:
        self.outstanding += 1
        self.idle.clear()

    def collected(self) -> None:
        self.outstanding -= 1
        if self.outstanding == 0:
            self.idle.set()

    async def pause(self) -> None:
        self.open.clear()
        await self.idle.wait()

    def resume(self, version: int) -> None:
        self.version = version
        self.open.set()


async def rollout_conversation(
    engine: AsyncOmni,
    conversation: PreparedConversation,
    *,
    sampling_config: dict[str, Any],
    seed: int,
    gate: RolloutGate,
    tools: ToolParticipant | None = None,
) -> dict[str, Any]:
    """Run one full prefix followed by causal inputs; never inject agent targets.

    The final prefix predicts output zero. Each live input consumes that output
    as feedback and predicts the next output. Thus N live inputs yield N+1
    predictions. Replay stores this contract explicitly, including the last one.
    With a tool participant, each completed tool call is answered by a
    simulated result injected as a system-input burst while user audio keeps
    streaming, exactly as the realtime adapter would.
    """
    session_id = f"opd-{conversation.conversation_id}-{seed}"
    fence = DuplexFence(session_id)
    runtime = {
        **sampling_config,
        "duplexio_system_token_ids": conversation.system_token_ids,
        "duplexio_voice": conversation.voice,
        "duplexio_voice_embedding_index": conversation.voice_embedding_index,
        "duplexio_tools": conversation.tools,
        "duplexio_tool_choice": {"mode": "auto" if conversation.tools else "none"},
        "duplexio_sampling_seed": seed,
        "duplexio_record_inputs": True,
        "duplex_stage_max_tokens": {"0": 1},
    }
    await engine.open_duplex_session_async(
        session_id,
        capabilities={
            "input_modes": ["append_audio_chunk"],
            "implementation_level": "model_native",
        },
        runtime_config=runtime,
        fence=fence,
        timeout=120.0,
    )
    recorder = TrajectoryRecorder()
    started = time.perf_counter()
    pending = asyncio.Semaphore(8)
    first_submitted = asyncio.Event()
    prefix_frames = len(conversation.system_token_ids)
    prefix_segments = (prefix_frames + 255) // 256
    frame_count = conversation.user_features.shape[0]
    expected_segments = prefix_segments + frame_count
    live_final_submitted = False
    tool_history: list[dict[str, Any]] = []
    tool_tasks: set[asyncio.Task[None]] = set()
    tool_rows_budget = (tools.max_session_rows - prefix_frames - frame_count) if tools else 0
    answered_calls = 0
    last_tool_sequence = 0  # the engine re-reports a call every frame until the next one

    async def append(payload: dict[str, Any], *, final: bool = False) -> None:
        await pending.acquire()
        await gate.open.wait()
        gate.submitted()
        await engine.append_duplex_input_async(
            session_id,
            mode="append_audio_chunk",
            payload=payload,
            final=final,
            fence=fence,
            timeout=120.0,
            collect_outputs=False,
        )
        first_submitted.set()

    async def submit() -> None:
        for offset in range(0, prefix_frames, 256):
            frames = min(256, prefix_frames - offset)
            await append(
                {
                    "type": "audio",
                    "audio": "",
                    "format": "pcm_f32le",
                    "sample_rate_hz": 24000,
                    "frame_size": 1920,
                    "frame_count": frames,
                    "valid_samples": frames * 1920,
                    "duplexio_prefill": True,
                    "duplexio_prefill_final": offset + frames == prefix_frames,
                    "decode_audio": False,
                }
            )
        nonlocal live_final_submitted
        for frame, user_token in enumerate(conversation.user_token_ids.tolist()):
            final = frame + 1 == frame_count
            if final:
                # Late tool results cannot follow the final live frame.
                await asyncio.gather(*tool_tasks)
                live_final_submitted = True
            await append(
                {
                    "format": "duplexio_features",
                    "features": conversation.user_features[frame : frame + 1],
                    "user_token_id": user_token,
                    "decode_audio": False,
                },
                final=final,
            )

    def transcript() -> str:
        assert tools is not None
        text_ids = torch.cat([segment["text_ids"] for segment in recorder.segments])
        lines = []
        for column, speaker in ((1, "user"), (2, "assistant")):
            ids = text_ids[:, column]
            ids = ids[(ids != tools.silence_token_id) & (ids != tools.pad_token_id)]
            if ids.numel():
                lines.append(f"{speaker}: {tools.tokenizer.decode(ids.tolist())}")
        return "\n".join(lines)

    async def answer(call: dict[str, Any]) -> None:
        assert tools is not None
        nonlocal expected_segments, tool_rows_budget, answered_calls
        if answered_calls >= tools.max_calls:
            return
        answered_calls += 1
        result = await tools.simulator.execute(
            call,
            tools=conversation.tools,
            assistant_system_prompt=str(conversation.metadata.get("teacher_system", "")),
            transcript=transcript(),
            history=tool_history,
        )
        token_ids = tools.tokenizer.encode(f"<tool_response>\n{result}\n</tool_response>", add_special_tokens=False)
        if len(token_ids) > tools.max_result_tokens:
            result = json.dumps({"error": "tool result too long for the session"})
            token_ids = tools.tokenizer.encode(f"<tool_response>\n{result}\n</tool_response>", add_special_tokens=False)
        fits = len(token_ids) <= tool_rows_budget
        print(json.dumps({"session": session_id, "tool": call["name"], "result_tokens": len(token_ids),
                          "injected": fits and not live_final_submitted}), flush=True)
        if live_final_submitted or not fits:
            return
        tool_rows_budget -= len(token_ids)
        for offset in range(0, len(token_ids), SYSTEM_INPUT_CHUNK_FRAMES):
            chunk = token_ids[offset : offset + SYSTEM_INPUT_CHUNK_FRAMES]
            expected_segments += 1
            await append(
                {
                    "type": "audio",
                    "audio": "",
                    "format": "pcm_f32le",
                    "sample_rate_hz": 24000,
                    "frame_size": 1920,
                    "frame_count": len(chunk),
                    "valid_samples": len(chunk) * 1920,
                    "duplexio_system_input": True,
                    "duplexio_system_input_final": offset + len(chunk) == len(token_ids),
                    "duplexio_system_token_ids": chunk,
                    "decode_audio": False,
                }
            )

    async def collect() -> tuple[float, float]:
        nonlocal last_tool_sequence
        await first_submitted.wait()
        completed = 0
        idle_since: float | None = None
        # Tool tasks may still be waiting on their LLM after every queued segment
        # has returned, so poll briefly and only fail when segments are outstanding.
        while completed < expected_segments or any(not task.done() for task in tool_tasks):
            outputs = await engine.collect_duplex_data_plane_outputs_async(
                duplex_resource_request_id(fence, "stage0"), timeout=5.0,
            )
            if not outputs:
                if completed < expected_segments:
                    idle_since = idle_since or time.monotonic()
                    if time.monotonic() - idle_since > 120.0:
                        raise TimeoutError("Timed out waiting for a queued input segment")
                continue
            idle_since = None
            for output in outputs:
                if output.error is not None:
                    raise RuntimeError(output.error)
                recorder.append(output.multimodal_output, gate.version)
                completed += 1
                if completed == prefix_segments:
                    prefill_finished = time.perf_counter()
                gate.collected()
                pending.release()
                if tools is not None:
                    for call in decode_tool_calls(output.multimodal_output.get("tool_call_json")):
                        if call["sequence"] <= last_tool_sequence:
                            continue
                        last_tool_sequence = call["sequence"]
                        task = asyncio.create_task(answer(call))
                        tool_tasks.add(task)
                        task.add_done_callback(tool_tasks.discard)
        return prefill_finished, time.perf_counter()

    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(submit())
            collection = tasks.create_task(collect())
        prefill_finished, decode_finished = collection.result()
    finally:
        await engine.close_duplex_session_async(session_id, fence=fence, timeout=120.0)
    return {
        "conversation_id": conversation.conversation_id,
        "runtime_config": runtime,
        "metadata": conversation.metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "prefill_seconds": prefill_finished - started,
        "decode_seconds": decode_finished - prefill_finished,
        **recorder.tensors(),
    }


TrajectorySink = Callable[[int, dict[str, Any]], Awaitable[None]]


async def rollout_conversations(
    engine: AsyncOmni,
    conversations: Sequence[Any],
    *,
    concurrency: int,
    sampling_config: dict[str, Any],
    seed: int,
    sink: TrajectorySink,
    gate: RolloutGate,
    passes: int | None = 1,
    tools: ToolParticipant | None = None,
    load: Callable[[Any], PreparedConversation] | None = None,
) -> None:
    """Continuously refill independent sessions; no cross-conversation barrier.

    `passes=None` cycles the pool forever with a fresh seed per pass. The sink
    receives each completed trajectory with its running index. With `load`,
    `conversations` are lightweight entries materialized as a session starts.
    """
    if concurrency < 1:
        raise ValueError("Rollout concurrency must be positive")

    def schedule():
        index = 0
        pass_index = 0
        while passes is None or pass_index < passes:
            for conversation in conversations:
                yield index, pass_index, conversation
                index += 1
            pass_index += 1

    pending = schedule()

    async def worker() -> None:
        for index, pass_index, entry in pending:
            conversation = load(entry) if load is not None else entry
            trace = await rollout_conversation(
                engine,
                conversation,
                sampling_config=sampling_config,
                seed=conversation_seed(seed + 1_000_003 * pass_index, conversation.conversation_id),
                gate=gate,
                tools=tools,
            )
            await sink(index, trace)

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(conversations))):
            group.create_task(worker())
