# SPDX-License-Identifier: Apache-2.0
"""Reference dump: run the TRAINING repo's proven streaming rollout.

Runs read-only against /dcai/users/thuand/duplexio (training venv, cwd there)
on the exact smoke input, with argmax text/emit/depth sampling, and dumps
per-frame agent tokens + the 8 depth codebook codes + the decoded waveform
for mechanistic comparison against the vllm-omni serving dump.

Invocation (GPU node):
  cd /dcai/users/thuand/duplexio && PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python /dcai/users/thuand/vllm-omni-work/probe_dumps/training_probe.py \
    --out /dcai/users/thuand/vllm-omni-work/probe_dumps/training_ref.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path("/dcai/users/thuand/duplexio")
EXPORT = REPO / "checkpoints/duplexio-489780-checkpoint-1-vllm-v3"
CHECKPOINT = REPO / "runs/duplexio_train/489780/checkpoints/checkpoint_1"
INPUT = Path("/dcai/users/thuand/vllm-omni-work/probe_dumps/input_24k.npy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voice", default="id10014")
    parser.add_argument(
        "--start-role",
        choices=("user", "agent"),
        default="user",
        help="agent: assistant-prefix prompt with silent user input "
        "(deterministic long agent narration)",
    )
    parser.add_argument("--silent-frames", type=int, default=0)
    args = parser.parse_args()

    import duplexio.full_duplex_bench as full_duplex_bench
    from duplexio.config import load_config
    from duplexio.evaluate import load_replicated_fsdp_model
    from duplexio.full_duplex_bench import (
        load_voice_embedding,
        streaming_rollout,
    )
    from duplexio.models.duplexio import TokenSamplingOptions
    from duplexio.tokenizer import (
        load_duplexio_tokenizer,
        suppressed_special_token_ids,
    )

    config = load_config(REPO / "configs/train.yaml")
    model_config = config.model
    model_config.gradient_checkpointing = False
    # Deterministic depth sampling: top_k == 1 is argmax in sample_logits.
    model_config.depth_autoregressive.sampling_top_k = 1

    tokenizer = load_duplexio_tokenizer(model_config)
    silence_token_id = int(tokenizer.silence_token_id)
    system_prompt = json.loads((EXPORT / "config.json").read_text())[
        "default_system_prompt"
    ]
    print(f"[probe] system prompt: {system_prompt!r}")

    model = load_replicated_fsdp_model(
        model_config,
        CHECKPOINT,
        torch.device("cuda"),
        flex_attention_dynamic=False,
    )
    speaker_embedding = load_voice_embedding(EXPORT, args.voice)
    if args.start_role == "agent":
        from duplexio.multistream.tokenization import (
            student_assistant_prefix_token_ids,
        )

        # streaming_rollout hardcodes the user-start prefix; agent-first
        # sessions use the assistant prefix with silent user input.
        full_duplex_bench.student_user_prefix_token_ids = (
            student_assistant_prefix_token_ids
        )
        frame_size = int(model.frame_size)
        waveform = torch.zeros(args.silent_frames * frame_size)
    else:
        waveform = torch.from_numpy(np.load(INPUT))
    print(f"[probe] input: {waveform.shape[0]} samples "
          f"({waveform.shape[0] // int(model.frame_size)} frames)")

    token_sampling = TokenSamplingOptions(
        mode="argmax",
        temperature=1.0,
        emit_temperature=1.0,
        emit_threshold=0.5,
        top_k=1,
        top_p=1.0,
        suppressed_token_ids=suppressed_special_token_ids(
            tokenizer,
            silence_token_id,
        ),
    )

    # Record the depth code columns as they are sampled.
    code_columns: list[torch.Tensor] = []
    original_sample_audio = model.sample_audio

    def recording_sample_audio(*sample_args, **sample_kwargs):
        column = original_sample_audio(*sample_args, **sample_kwargs)
        code_columns.append(column.detach().to("cpu", torch.long))
        return column

    model.sample_audio = recording_sample_audio

    with model.dynamic_flex_attention():
        rollout = streaming_rollout(
            model,
            tokenizer,
            waveform,
            speaker_embedding,
            system_prompt,
            token_sampling,
        )

    events = rollout.events
    agent_ids = [
        event["agent_token_id"]
        for event in events
        if event["agent_token_id"] != silence_token_id
    ]
    print(f"[probe] {len(events)} events; agent tokens: {agent_ids}")
    print(f"[probe] agent text: "
          f"{tokenizer.decode(agent_ids, skip_special_tokens=True)!r}")
    user_ids = [
        event["user_token_id"]
        for event in events
        if event["user_token_id"] != silence_token_id
    ]
    print(f"[probe] user text: "
          f"{tokenizer.decode(user_ids, skip_special_tokens=True)!r}")

    torch.save(
        {
            "events": events,
            "codes": torch.cat(code_columns, dim=0),
            "output_audio": rollout.output_audio,
            "silence_token_id": silence_token_id,
            "system_prompt": system_prompt,
        },
        args.out,
    )
    print(f"[probe] dump written to {args.out}")


if __name__ == "__main__":
    main()
