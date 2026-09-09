"""Rollout actors for on-policy distillation, driven by a running trainer.

One engine per listed GPU. Each actor registers with the trainer, joins its
NCCL policy group, cycles the prepared conversation pool forever, streams every
finished trajectory back, and loads new weights in place whenever the trainer
pushes a version. Sessions pause only while an update is in flight.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import socket
from pathlib import Path
from typing import Any

import torch
from duplexio.opd_link import JOIN_GROUP, JOINED, PREPARE_UPDATE, READY, SHUTDOWN, UPDATED, ActorLink
from pydantic import TypeAdapter

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.experimental.fullduplex.duplexio.offline import (
    PreparedConversation,
    RolloutGate,
    ToolParticipant,
    rollout_conversations,
)
from vllm_omni.experimental.fullduplex.duplexio.tool_simulator import ToolSimulator
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import DuplexIOConfig

RECEIVER = "vllm_omni.experimental.fullduplex.duplexio.policy_receiver.PolicyWeightReceiver"
TRAJECTORY_KEYS = (
    "text_ids", "agent_audio", "audio_mask", "prediction_rows",
    "sampled_agent_ids", "sampled_tool_ids", "sampled_audio", "row_versions",
)


def sampling_runtime(config: DuplexIOConfig) -> dict[str, Any]:
    sampling = config.rollout_sampling_config
    return {
        "duplexio_scheduler_token_id": config.pad_token_id,
        "duplexio_text_sampling": sampling,
        "duplexio_emit_temperatures": {
            "user": 0.0,
            "agent": sampling["emit_temperature"],
            "tool_call": sampling["emit_temperature"],
        },
        "duplexio_depth_sampling": {
            "temperature": 0.7,
            "top_k": config.depth_transformer_config.get("sampling_top_k", 250),
        },
    }


async def rpc(engine: AsyncOmni, method: str, args: tuple[Any, ...], timeout: float) -> None:
    """Worker RPC failures come back as result dicts; treat them as fatal."""
    for result in await engine.collective_rpc(method, args=args, timeout=timeout):
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(f"{method} failed on the engine worker: {result['error']}")


async def run(args: argparse.Namespace, actor_index: int = 0) -> None:
    device = args.devices[actor_index]
    config = DuplexIOConfig.from_pretrained(args.checkpoint, local_files_only=True)
    conversations = TypeAdapter(list[PreparedConversation]).validate_python(
        torch.load(args.inputs, map_location="cpu", weights_only=True)
    )[actor_index::len(args.devices)]
    if not conversations:
        raise ValueError("Each rollout actor needs at least one conversation")
    engine = AsyncOmni(
        model=str(args.checkpoint),
        deploy_config=args.deploy_config,
        init_timeout=args.init_timeout,
        stage_overrides={"0": {"devices": str(device), "num_replicas": 1}},
        worker_extension_cls=RECEIVER,
    )
    tools = None
    if args.tool_model:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
        tools = ToolParticipant(
            ToolSimulator(args.tool_model, base_url=args.tool_base_url),
            tokenizer,
            silence_token_id=config.silence_token_id,
            pad_token_id=config.pad_token_id,
            max_session_rows=args.max_session_rows,
            max_calls=args.max_tool_calls,
        )
    link = ActorLink(args.trainer, f"{socket.gethostname()}-gpu{device}")
    gate = RolloutGate(version=args.initial_version)
    sent = 0
    keys = TRAJECTORY_KEYS + (("predictor_hiddens",) if args.record_hiddens else ())

    async def sink(index: int, trace: dict[str, Any]) -> None:
        nonlocal sent
        link.send_trajectory(
            trace["conversation_id"],
            {key: trace[key] for key in keys}
            | {"elapsed_seconds": trace["elapsed_seconds"], "seed": trace["runtime_config"]["duplexio_sampling_seed"]},
        )
        sent += 1

    rollouts = asyncio.create_task(
        rollout_conversations(
            engine,
            conversations,
            concurrency=args.concurrency,
            sampling_config=sampling_runtime(config) | {"duplexio_record_hiddens": args.record_hiddens},
            seed=args.seed + 7919 * actor_index,
            sink=sink,
            gate=gate,
            passes=None,
            tools=tools,
        )
    )
    try:
        while not rollouts.done():
            # One socket, one thread: poll without blocking the event loop.
            message = link.poll(0)
            if message is None:
                await asyncio.sleep(0.05)
                continue
            header, _ = message
            kind = header["type"]
            if kind == JOIN_GROUP:
                await rpc(
                    engine,
                    "join_policy_group",
                    (header["host"], header["port"], header["rank"], header["world_size"]),
                    args.init_timeout,
                )
                link.send({"type": JOINED})
            elif kind == PREPARE_UPDATE:
                version = header["version"]
                await gate.pause()
                link.send({"type": READY, "version": version})
                await rpc(engine, "receive_policy_weights", (header["weights"],), args.init_timeout)
                gate.resume(version)
                link.send({"type": UPDATED, "version": version})
                print(json.dumps({"actor": actor_index, "version": version, "trajectories_sent": sent}), flush=True)
            elif kind == SHUTDOWN:
                rollouts.cancel()
                break
            else:
                raise ValueError(f"Unexpected trainer message {kind!r}")
        with contextlib.suppress(asyncio.CancelledError):
            await rollouts  # surfaces rollout errors
    finally:
        engine.shutdown()


def run_actor(actor_index: int, args: argparse.Namespace) -> None:
    asyncio.run(run(args, actor_index))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Native export the engine starts from")
    parser.add_argument("inputs", type=Path, help="Prepared conversation pool (prepare_convogen.py)")
    parser.add_argument("--trainer", required=True, help="Trainer rank-0 endpoint, e.g. tcp://dgx065:29600")
    parser.add_argument("--devices", type=int, nargs="+", required=True, help="One actor per GPU")
    parser.add_argument("--concurrency", type=int, default=32, help="Concurrent conversations per actor")
    parser.add_argument("--initial-version", type=int, default=0, help="Policy version of the starting export")
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--init-timeout", type=int, default=1200)
    parser.add_argument("--record-hiddens", action="store_true", help="Also stream predictor states (parity probes only)")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/duplexio_opd_h100.yaml")
    parser.add_argument("--tool-model", help="Answer tool calls with this OpenAI-compatible model; omit to leave calls unanswered")
    parser.add_argument("--tool-base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--max-session-rows", type=int, default=4096, help="Engine max_model_len / 6 cells")
    parser.add_argument("--max-tool-calls", type=int, default=8, help="Answered calls per conversation")
    args = parser.parse_args()
    if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
        parser.error("Devices must be distinct nonnegative GPU indices")
    if len(args.devices) > 1:
        torch.multiprocessing.spawn(run_actor, args=(args,), nprocs=len(args.devices))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
