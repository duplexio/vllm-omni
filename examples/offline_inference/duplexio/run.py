"""Generate replayable agent rollouts from prepared, causally encoded users."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from multiprocessing.synchronize import Barrier
from pathlib import Path

import torch
from pydantic import TypeAdapter

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.experimental.fullduplex.duplexio.offline import (
    PreparedConversation,
    rollout_conversations,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import DuplexIOConfig


async def run(args: argparse.Namespace, actor_index: int = 0, round_barrier: Barrier | None = None) -> None:
    config = DuplexIOConfig.from_pretrained(args.checkpoint, local_files_only=True)
    conversations = TypeAdapter(list[PreparedConversation]).validate_python(
        torch.load(args.inputs, map_location="cpu", weights_only=True)
    )
    actor_count = len(args.devices) if args.devices else 1
    if len(conversations) < actor_count:
        raise ValueError("Each rollout actor needs at least one conversation")
    conversations = conversations[actor_index::actor_count]
    actor_output_dir = args.output_dir / f"actor_{actor_index}" if actor_count > 1 else args.output_dir
    sampling = config.rollout_sampling_config
    runtime = {
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
    engine_options = {}
    if args.devices:
        engine_options["stage_overrides"] = {
            "0": {"devices": str(args.devices[actor_index]), "num_replicas": 1},
        }
    if args.profile_dir is not None:
        engine_options["profiler_config"] = {
            "profiler": "torch",
            "torch_profiler_dir": str(args.profile_dir.resolve()),
            "torch_profiler_with_stack": False,
            "torch_profiler_dump_cuda_time_total": False,
            "delay_iterations": args.profile_delay_steps,
            "max_iterations": args.profile_steps,
            "ignore_frontend": bool(args.profile_delay_steps or args.profile_steps),
        }
    if args.layer_capture is not None:
        engine_options["worker_extension_cls"] = (
            "examples.offline_inference.duplexio.capture_layers.LayerCaptureWorker"
        )
    engine = AsyncOmni(
        model=str(args.checkpoint), deploy_config=args.deploy_config, init_timeout=args.init_timeout, **engine_options
    )
    started = time.perf_counter()
    try:
        if args.layer_capture is not None:
            await engine.collective_rpc("start_layer_capture", args=(str(args.layer_capture.resolve()), args.attention_layer, args.capture_max_forwards))
        for round_index in range(-args.warmup_rounds, args.rounds):
            if round_barrier is not None:
                await asyncio.to_thread(round_barrier.wait, timeout=args.init_timeout)
            runtime["duplexio_record_hiddens"] = args.record_hiddens or round_index == args.record_hiddens_round
            output_dir = actor_output_dir
            if args.warmup_rounds or args.rounds > 1:
                output_dir /= f"round_{round_index}"
            profiled = args.profile_dir is not None and round_index == args.rounds - 1
            if profiled:
                await engine.start_profile()
            round_started = time.perf_counter()
            try:
                await rollout_conversations(
                    engine,
                    conversations,
                    concurrency=args.concurrency,
                    policy_version=args.policy_version,
                    sampling_config=runtime,
                    seed=args.seed,
                    output_dir=output_dir,
                )
            finally:
                round_finished = time.perf_counter()
                round_seconds = round_finished - round_started
                if profiled:
                    await engine.stop_profile()
            frames = sum(conversation.user_features.shape[0] for conversation in conversations)
            print(json.dumps({
                "round": round_index,
                "actor": actor_index,
                "actors": actor_count,
                "started": round_started,
                "finished": round_finished,
                "warmup": round_index < 0,
                "profiled": profiled,
                "seconds": round_seconds,
                "audio_frames_per_second": frames / round_seconds,
                "concurrency": args.concurrency,
                "output_dir": str(output_dir),
            }), flush=True)
        if args.layer_capture is not None:
            await engine.collective_rpc("finish_layer_capture")
    finally:
        engine.shutdown()
    print(
        json.dumps(
            {
                "conversations": len(conversations),
                "input_audio_frames": sum(conversation.user_features.shape[0] for conversation in conversations),
                "seconds_including_shutdown": time.perf_counter() - started,
                "output_dir": str(actor_output_dir),
            }
        ),
        flush=True,
    )


def run_actor(actor_index: int, args: argparse.Namespace, round_barrier: Barrier) -> None:
    """Own one engine/frontend pair; only benchmark round boundaries synchronize."""
    asyncio.run(run(args, actor_index, round_barrier))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--concurrency", type=int, default=1, help="Concurrent conversations per actor")
    parser.add_argument("--devices", type=int, nargs="+", help="One independent rollout actor per listed GPU")
    parser.add_argument("--init-timeout", type=int, default=600, help="Total worker startup allowance in seconds")
    parser.add_argument("--seed", type=int, default=17000)
    parser.add_argument("--warmup-rounds", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--profile-dir", type=Path, help="Profile the final round, separately from throughput rounds")
    parser.add_argument("--profile-delay-steps", type=int, default=0, help="Skip this many engine steps before recording")
    parser.add_argument("--profile-steps", type=int, default=0, help="Limit recorded engine steps; zero records the full round")
    parser.add_argument("--record-hiddens", action="store_true", help="Save predictor states for backbone parity checks")
    parser.add_argument("--record-hiddens-round", type=int, help="Record states in only this round, e.g. -1 for warmup")
    parser.add_argument("--layer-capture", type=Path, help="Capture first sixteen native forwards for layerwise diagnosis")
    parser.add_argument("--attention-layer", type=int, default=3, help="Full-attention layer to capture in detail")
    parser.add_argument("--capture-max-forwards", type=int, default=16)
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/duplexio_opd.yaml")
    args = parser.parse_args()
    if args.init_timeout <= 0:
        parser.error("Startup timeout must be positive")
    if args.warmup_rounds < 0 or args.rounds < 1:
        parser.error("Warmup rounds must be nonnegative and measured rounds positive")
    if args.capture_max_forwards < 1:
        parser.error("Capture limit must be positive")
    if args.profile_delay_steps < 0 or args.profile_steps < 0:
        parser.error("Profile delay and limit must be nonnegative")
    if args.output_dir.exists():
        parser.error("Output directory already exists; use a new directory for each rollout round")
    if args.devices and (min(args.devices) < 0 or len(set(args.devices)) != len(args.devices)):
        parser.error("Devices must be distinct nonnegative GPU indices")
    if args.devices and len(args.devices) > 1:
        context = torch.multiprocessing.get_context("spawn")
        barrier = context.Barrier(len(args.devices))
        torch.multiprocessing.spawn(run_actor, args=(args, barrier), nprocs=len(args.devices))
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
