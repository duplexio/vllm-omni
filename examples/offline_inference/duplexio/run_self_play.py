# SPDX-License-Identifier: Apache-2.0
"""Run Convogen user/agent pairs on two native DuplexIO engines."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import torch
from pydantic import TypeAdapter

from vllm_omni.entrypoints.async_omni import AsyncOmni
from vllm_omni.experimental.fullduplex.duplexio.self_play import (
    PreparedScenarioPair,
    rollout_scenario_pairs,
)
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import DuplexIOConfig


def build_engine(
    checkpoint: Path,
    deploy_config: str,
    devices: list[int],
    init_timeout: int,
) -> AsyncOmni:
    return AsyncOmni(
        model=str(checkpoint),
        deploy_config=deploy_config,
        init_timeout=init_timeout,
        stage_overrides={
            "0": {
                "devices": ",".join(str(device) for device in devices),
                "num_replicas": 1,
            }
        },
    )


async def run(args: argparse.Namespace) -> None:
    config = DuplexIOConfig.from_pretrained(args.checkpoint, local_files_only=True)
    pairs = TypeAdapter(list[PreparedScenarioPair]).validate_python(
        torch.load(args.inputs, map_location="cpu", weights_only=True)
    )
    if not pairs:
        raise ValueError("Self-play input contains no scenario pairs")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory is not empty: {args.output_dir}")
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
    agent_engine: AsyncOmni | None = None
    user_engine: AsyncOmni | None = None
    try:
        agent_engine = build_engine(
            args.checkpoint,
            args.deploy_config,
            args.agent_devices,
            args.init_timeout,
        )
        user_engine = build_engine(
            args.checkpoint,
            args.deploy_config,
            args.user_devices,
            args.init_timeout,
        )
        await rollout_scenario_pairs(
            agent_engine,
            user_engine,
            pairs,
            concurrency=args.concurrency,
            policy_version=args.policy_version,
            user_policy_version=args.user_policy_version,
            agent_sampling_config=runtime,
            seed=args.seed,
            max_frames=args.max_frames,
            output_dir=args.output_dir,
            timeout=args.timeout,
            record_hiddens=args.record_hiddens,
        )
    finally:
        if agent_engine is not None:
            agent_engine.shutdown()
        if user_engine is not None:
            user_engine.shutdown()
    print(
        json.dumps(
            {
                "pairs": len(pairs),
                "policy_version": args.policy_version,
                "max_frames": args.max_frames,
                "output_dir": str(args.output_dir),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("inputs", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--user-policy-version", default="frozen-user")
    parser.add_argument("--agent-devices", type=int, nargs="+", required=True)
    parser.add_argument("--user-devices", type=int, nargs="+", required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17_000)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--init-timeout", type=int, default=600)
    parser.add_argument("--record-hiddens", action="store_true")
    parser.add_argument("--deploy-config", default="vllm_omni/deploy/duplexio_opd.yaml")
    args = parser.parse_args()
    devices = args.agent_devices + args.user_devices
    if min(devices) < 0 or len(set(devices)) != len(devices):
        parser.error("Agent and user devices must be distinct nonnegative GPU indices")
    if args.concurrency < 1 or args.max_frames < 1 or args.timeout <= 0 or args.init_timeout <= 0:
        parser.error("Concurrency, max-frames, and timeouts must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
