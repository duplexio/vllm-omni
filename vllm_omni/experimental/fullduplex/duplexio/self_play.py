# SPDX-License-Identifier: Apache-2.0
"""Synchronous two-party DuplexIO self-play.

The runner deliberately keeps both policies on a single frame clock.  Each
model receives the other model's previous decoded PCM frame, then both models
advance one frame concurrently.  The agent trace contains the exact consumed
rows required by the training-side replay path; the user trace is retained for
diagnostics and remains a frozen simulator policy for a rollout round.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from vllm_omni.experimental.fullduplex.duplexio.trajectory import TrajectoryRecorder
from vllm_omni.experimental.fullduplex.engine.messages import DuplexFence

if TYPE_CHECKING:
    from vllm_omni.entrypoints.async_omni import AsyncOmni


class DuplexPairEngine(Protocol):
    """Engine methods used by the lock-step runner."""

    async def open_duplex_session_async(self, session_id: str, **kwargs: Any) -> dict[str, object]: ...

    async def append_duplex_input_async(self, session_id: str, **kwargs: Any) -> dict[str, object]: ...

    async def close_duplex_session_async(self, session_id: str, **kwargs: Any) -> dict[str, object]: ...


class PreparedRole(BaseModel):
    """One Convogen role after prompt rendering and voice selection."""

    model_config = ConfigDict(extra="forbid")

    system_token_ids: list[int] = Field(min_length=1)
    voice: str = Field(min_length=1)
    voice_embedding_index: int = Field(default=0, ge=0)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("system_token_ids")
    @classmethod
    def validate_token_ids(cls, token_ids: list[int]) -> list[int]:
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("Role system token IDs must be nonnegative")
        return token_ids


class PreparedScenarioPair(BaseModel):
    """Prepared user and agent prompts for one synchronous self-play case."""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str = Field(min_length=1)
    agent: PreparedRole
    user: PreparedRole
    metadata: dict[str, Any] = Field(default_factory=dict)


def _role_runtime(
    role: PreparedRole,
    sampling_config: Mapping[str, Any],
    *,
    seed: int,
    policy_version: str,
    record_hiddens: bool,
) -> dict[str, Any]:
    return {
        **sampling_config,
        "duplexio_system_token_ids": role.system_token_ids,
        "duplexio_voice": role.voice,
        "duplexio_voice_embedding_index": role.voice_embedding_index,
        "duplexio_tools": role.tools,
        "duplexio_tool_choice": {"mode": "auto" if role.tools else "none"},
        "duplexio_sampling_seed": seed,
        "duplexio_policy_version": policy_version,
        "duplexio_record_inputs": True,
        "duplexio_record_hiddens": record_hiddens,
        "duplex_stage_max_tokens": {"0": 1},
    }


async def _open_role(
    engine: DuplexPairEngine,
    session_id: str,
    role: PreparedRole,
    sampling_config: Mapping[str, Any],
    *,
    seed: int,
    policy_version: str,
    record_hiddens: bool,
    timeout: float,
) -> tuple[str, DuplexFence, dict[str, Any]]:
    fence = DuplexFence(session_id)
    runtime = _role_runtime(
        role,
        sampling_config,
        seed=seed,
        policy_version=policy_version,
        record_hiddens=record_hiddens,
    )
    await engine.open_duplex_session_async(
        session_id,
        capabilities={
            "input_modes": ["append_audio_chunk"],
            "implementation_level": "model_native",
        },
        runtime_config=runtime,
        fence=fence,
        timeout=timeout,
    )
    return session_id, fence, runtime


def _payload_mapping(output: object) -> Mapping[str, Any]:
    payload = getattr(output, "multimodal_output", None)
    if isinstance(payload, Mapping):
        return payload
    raise RuntimeError(
        "DuplexIO append returned no multimodal payload; "
        f"got {type(payload).__name__}"
    )


async def _append_output(
    engine: DuplexPairEngine,
    session_id: str,
    fence: DuplexFence,
    payload: dict[str, Any],
    *,
    final: bool,
    timeout: float,
) -> Mapping[str, Any]:
    result = await engine.append_duplex_input_async(
        session_id,
        mode="append_audio_chunk",
        payload=payload,
        final=final,
        fence=fence,
        timeout=timeout,
        collect_outputs=True,
    )
    outputs = result.get("data_plane_outputs")
    if not isinstance(outputs, list) or len(outputs) != 1:
        raise RuntimeError(
            "DuplexIO synchronous pair expected exactly one output per append; "
            f"got {len(outputs) if isinstance(outputs, list) else type(outputs).__name__}"
        )
    output = outputs[0]
    error = getattr(output, "error", None)
    if error is not None:
        raise RuntimeError(str(error))
    return _payload_mapping(output)


def _prefill_payload(frame_count: int, *, final: bool, decode_audio: bool) -> dict[str, Any]:
    return {
        "type": "audio",
        "audio": "",
        "format": "pcm_f32le",
        "sample_rate_hz": 24_000,
        "frame_size": 1_920,
        "frame_count": frame_count,
        "valid_samples": frame_count * 1_920,
        "duplexio_prefill": True,
        "duplexio_prefill_final": final,
        "decode_audio": decode_audio,
    }


def _pcm_payload(audio: Tensor) -> dict[str, Any]:
    if audio.shape != (1_920,):
        raise ValueError(f"DuplexIO pair audio frames must have shape (1920,), got {tuple(audio.shape)}")
    frame = audio.detach().to(device="cpu", dtype=torch.float32).contiguous()
    encoded = base64.b64encode(frame.numpy().tobytes()).decode("ascii")
    return {
        "format": "pcm_f32le",
        "audio": encoded,
        "sample_rate_hz": 24_000,
        "frame_size": 1_920,
        "frame_count": 1,
        "valid_samples": 1_920,
        "decode_audio": True,
    }


def _audio_frame(payload: Mapping[str, Any]) -> Tensor:
    audio = payload.get("audio")
    if audio is None:
        return torch.zeros(1_920, dtype=torch.float32)
    if not isinstance(audio, Tensor):
        raise RuntimeError(f"DuplexIO output audio must be a tensor, got {type(audio).__name__}")
    if audio.numel() == 0:
        return torch.zeros(1_920, dtype=torch.float32)
    if audio.shape != (1_920,):
        raise RuntimeError(f"DuplexIO output audio frame has shape {tuple(audio.shape)}, expected (1920,)")
    return audio.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _scalar_int(value: object) -> int | None:
    if isinstance(value, Tensor):
        if value.numel() == 0:
            return None
        return int(value.detach().cpu().reshape(-1)[-1].item())
    if isinstance(value, list | tuple):
        return _scalar_int(value[-1]) if value else None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _scalar_bool(value: object) -> bool | None:
    if isinstance(value, Tensor):
        if value.numel() == 0:
            return None
        return bool(value.detach().cpu().reshape(-1)[-1].item())
    if isinstance(value, list | tuple):
        return _scalar_bool(value[-1]) if value else None
    return value if isinstance(value, bool) else None


def _event(frame: int, agent: Mapping[str, Any], user: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame": frame,
        "agent_token_id": _scalar_int(agent.get("agent_token_id")),
        "agent_tool_call_token_id": _scalar_int(agent.get("tool_call_token_id")),
        "agent_model_listen": _scalar_bool(agent.get("model_listen")),
        "user_token_id": _scalar_int(user.get("agent_token_id")),
        "user_tool_call_token_id": _scalar_int(user.get("tool_call_token_id")),
        "user_model_listen": _scalar_bool(user.get("model_listen")),
    }


async def _prefill(
    engine: DuplexPairEngine,
    session_id: str,
    fence: DuplexFence,
    role: PreparedRole,
    *,
    timeout: float,
) -> tuple[TrajectoryRecorder, Tensor, float]:
    recorder = TrajectoryRecorder()
    started = time.perf_counter()
    prefix_frames = len(role.system_token_ids)
    initial_audio = torch.zeros(1_920, dtype=torch.float32)
    for offset in range(0, prefix_frames, 256):
        frames = min(256, prefix_frames - offset)
        final = offset + frames == prefix_frames
        output = await _append_output(
            engine,
            session_id,
            fence,
            _prefill_payload(frames, final=final, decode_audio=final),
            final=False,
            timeout=timeout,
        )
        recorder.append(output)
        if final:
            initial_audio = _audio_frame(output)
    return recorder, initial_audio, time.perf_counter() - started


async def rollout_scenario_pair(
    agent_engine: AsyncOmni,
    user_engine: AsyncOmni,
    pair: PreparedScenarioPair,
    *,
    policy_version: str,
    user_policy_version: str = "frozen-user",
    agent_sampling_config: Mapping[str, Any],
    user_sampling_config: Mapping[str, Any] | None = None,
    seed: int,
    max_frames: int,
    timeout: float = 120.0,
    record_hiddens: bool = False,
) -> dict[str, Any]:
    """Run a synchronous user/agent conversation and return replay artifacts.

    ``max_frames`` is the number of live frame exchanges after each role's
    prompt prefill.  Both engines are inference-only and keep their weights
    fixed for this call.  The caller must refresh the agent engine between
    rounds; changing weights while either session is open would invalidate the
    recorded policy version.

    Tool calls are surfaced as a hard error because a tool-result simulator is
    a third participant and cannot be fabricated by either speech policy.
    """
    if max_frames < 1:
        raise ValueError("Synchronous pair rollout requires at least one live frame")
    if user_sampling_config is None:
        user_sampling_config = agent_sampling_config
    session_ids = {
        "agent": f"opd-self-play-{pair.conversation_id}-agent-{seed}",
        "user": f"opd-self-play-{pair.conversation_id}-user-{seed + 1}",
    }
    started = time.perf_counter()
    opened: list[tuple[DuplexPairEngine, str, DuplexFence]] = []
    try:
        open_tasks = [
            asyncio.create_task(
                _open_role(
                    agent_engine,
                    session_ids["agent"],
                    pair.agent,
                    agent_sampling_config,
                    seed=seed,
                    policy_version=policy_version,
                    record_hiddens=record_hiddens,
                    timeout=timeout,
                )
            ),
            asyncio.create_task(
                _open_role(
                    user_engine,
                    session_ids["user"],
                    pair.user,
                    user_sampling_config,
                    seed=seed + 1,
                    policy_version=user_policy_version,
                    record_hiddens=False,
                    timeout=timeout,
                )
            ),
        ]
        open_results = await asyncio.gather(*open_tasks, return_exceptions=True)
        open_failure: BaseException | None = None
        for result, engine in zip(open_results, (agent_engine, user_engine), strict=True):
            if isinstance(result, BaseException):
                open_failure = open_failure or result
                continue
            opened.append((engine, result[0], result[1]))
        if open_failure is not None:
            raise open_failure
        agent_open, user_open = open_results
        assert not isinstance(agent_open, BaseException)
        assert not isinstance(user_open, BaseException)
        results = (agent_open, user_open)
        (agent_session, agent_fence, agent_runtime), (user_session, user_fence, user_runtime) = results
        (agent_recorder, agent_audio, agent_prefill_seconds), (
            user_recorder,
            user_audio,
            user_prefill_seconds,
        ) = await asyncio.gather(
            _prefill(agent_engine, agent_session, agent_fence, pair.agent, timeout=timeout),
            _prefill(user_engine, user_session, user_fence, pair.user, timeout=timeout),
        )
        events: list[dict[str, Any]] = []
        live_started = time.perf_counter()
        for frame in range(max_frames):
            agent_output, user_output = await asyncio.gather(
                _append_output(
                    agent_engine,
                    agent_session,
                    agent_fence,
                    _pcm_payload(user_audio),
                    final=frame + 1 == max_frames,
                    timeout=timeout,
                ),
                _append_output(
                    user_engine,
                    user_session,
                    user_fence,
                    _pcm_payload(agent_audio),
                    final=frame + 1 == max_frames,
                    timeout=timeout,
                ),
            )
            agent_recorder.append(agent_output)
            user_recorder.append(user_output)
            events.append(_event(frame, agent_output, user_output))
            if _scalar_bool(agent_output.get("tool_call_complete")):
                raise RuntimeError(
                    "Synchronous pair rollout encountered an agent tool call; "
                    "provide a third-party tool-result simulator before using this scenario"
                )
            agent_audio = _audio_frame(agent_output)
            user_audio = _audio_frame(user_output)
        live_seconds = time.perf_counter() - live_started
        return {
            "policy_version": policy_version,
            "user_policy_version": user_policy_version,
            "conversation_id": pair.conversation_id,
            "metadata": pair.metadata,
            "agent_runtime_config": agent_runtime,
            "user_runtime_config": user_runtime,
            "elapsed_seconds": time.perf_counter() - started,
            "agent_trace": {
                "policy_version": policy_version,
                "conversation_id": pair.conversation_id,
                "runtime_config": agent_runtime,
                "metadata": pair.metadata,
                "elapsed_seconds": time.perf_counter() - started,
                "prefill_seconds": agent_prefill_seconds,
                "decode_seconds": live_seconds,
                **agent_recorder.tensors(),
            },
            "user_trace": {
                "policy_version": user_policy_version,
                "conversation_id": pair.conversation_id,
                "runtime_config": user_runtime,
                "metadata": pair.user.metadata,
                "elapsed_seconds": time.perf_counter() - started,
                "prefill_seconds": user_prefill_seconds,
                "decode_seconds": live_seconds,
                **user_recorder.tensors(),
            },
            "events": events,
        }
    finally:
        await asyncio.gather(
            *(
                engine.close_duplex_session_async(
                    session_id,
                    reason="self_play_complete",
                    fence=fence,
                    timeout=timeout,
                )
                for engine, session_id, fence in opened
            ),
            return_exceptions=True,
        )


async def rollout_scenario_pairs(
    agent_engine: AsyncOmni,
    user_engine: AsyncOmni,
    pairs: list[PreparedScenarioPair],
    *,
    concurrency: int,
    policy_version: str,
    user_policy_version: str = "frozen-user",
    agent_sampling_config: Mapping[str, Any],
    user_sampling_config: Mapping[str, Any] | None = None,
    seed: int,
    max_frames: int,
    output_dir: Path,
    timeout: float = 120.0,
    record_hiddens: bool = False,
) -> None:
    """Run independent pairs concurrently and persist one result per case."""
    if concurrency < 1:
        raise ValueError("Self-play concurrency must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = iter(enumerate(pairs))

    async def worker() -> None:
        for index, pair in pending:
            result = await rollout_scenario_pair(
                agent_engine,
                user_engine,
                pair,
                policy_version=policy_version,
                user_policy_version=user_policy_version,
                agent_sampling_config=agent_sampling_config,
                user_sampling_config=user_sampling_config,
                seed=seed + index * 2,
                max_frames=max_frames,
                timeout=timeout,
                record_hiddens=record_hiddens,
            )
            torch.save(result, output_dir / f"trajectory_{index:06d}.pt")

    async with asyncio.TaskGroup() as tasks:
        for _ in range(min(concurrency, len(pairs))):
            tasks.create_task(worker())


__all__ = [
    "PreparedRole",
    "PreparedScenarioPair",
    "rollout_scenario_pair",
    "rollout_scenario_pairs",
]
