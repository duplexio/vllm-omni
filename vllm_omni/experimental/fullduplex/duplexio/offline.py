# SPDX-License-Identifier: Apache-2.0
"""Unpaced, concurrent conversations through the native duplex engine."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.experimental.fullduplex.engine.contracts import duplex_resource_request_id
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni


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


def conversation_seed(base_seed: int, conversation_id: str) -> int:
    """Keep RNG identity stable across batching, reordering, and GPU sharding."""
    digest = hashlib.blake2b(conversation_id.encode(), digest_size=8).digest()
    return (base_seed + int.from_bytes(digest)) % (2**63 - 1)


async def rollout_conversation(
    engine: AsyncOmni,
    conversation: PreparedConversation,
    *,
    policy_version: str,
    sampling_config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Run one full prefix followed by causal inputs; never inject agent targets.

    The final prefix predicts output zero. Each live input consumes that output
    as feedback and predicts the next output. Thus N live inputs yield N+1
    predictions. Replay stores this contract explicitly, including the last one.
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

    async def append(payload: dict[str, Any], *, final: bool = False) -> None:
        await pending.acquire()
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
        for frame, user_token in enumerate(conversation.user_token_ids.tolist()):
            await append(
                {
                    "format": "duplexio_features",
                    "features": conversation.user_features[frame : frame + 1],
                    "user_token_id": user_token,
                    "decode_audio": False,
                },
                final=frame + 1 == frame_count,
            )

    async def collect() -> tuple[float, float]:
        await first_submitted.wait()
        completed = 0
        while completed < prefix_segments + frame_count:
            outputs = await engine.collect_duplex_data_plane_outputs_async(
                duplex_resource_request_id(fence, "stage0"), timeout=120.0,
            )
            if not outputs:
                raise TimeoutError("Timed out waiting for a queued input segment")
            for output in outputs:
                if output.error is not None:
                    raise RuntimeError(output.error)
                recorder.append(output.multimodal_output)
                completed += 1
                if completed == prefix_segments:
                    prefill_finished = time.perf_counter()
                pending.release()
        return prefill_finished, time.perf_counter()

    try:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(submit())
            collection = tasks.create_task(collect())
        prefill_finished, decode_finished = collection.result()
    finally:
        await engine.close_duplex_session_async(session_id, fence=fence, timeout=120.0)
    return {
        "policy_version": policy_version,
        "conversation_id": conversation.conversation_id,
        "runtime_config": runtime,
        "metadata": conversation.metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "prefill_seconds": prefill_finished - started,
        "decode_seconds": decode_finished - prefill_finished,
        **recorder.tensors(),
    }


async def rollout_conversations(
    engine: AsyncOmni,
    conversations: list[PreparedConversation],
    *,
    concurrency: int,
    policy_version: str,
    sampling_config: dict[str, Any],
    seed: int,
    output_dir: Path,
) -> None:
    """Continuously refill independent sessions; no cross-conversation barrier."""
    if concurrency < 1:
        raise ValueError("Rollout concurrency must be positive")
    pending = iter(enumerate(conversations))
    output_dir.mkdir(parents=True, exist_ok=True)

    async def worker() -> None:
        for index, conversation in pending:
            trace = await rollout_conversation(
                engine,
                conversation,
                policy_version=policy_version,
                sampling_config=sampling_config,
                seed=conversation_seed(seed, conversation.conversation_id),
            )
            # File names do not depend on potentially path-like dataset IDs.
            torch.save(trace, output_dir / f"trajectory_{index:06d}.pt")

    async with asyncio.TaskGroup() as group:
        for _ in range(min(concurrency, len(conversations))):
            group.create_task(worker())
