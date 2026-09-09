"""Check native sampling against real training weights, without loading Qwen.

Put the DuplexIO training checkout on PYTHONPATH. This is head-level numerical
parity, not an end-to-end serving or throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from duplexio.modules.flowmap import FlowMap as TrainingFlowMap

from vllm_omni.model_executor.models.duplexio.flowmap import FlowMap


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--autocast", action="store_true", help="Use the rollouts' BF16 autocast context.")
    args = parser.parse_args()
    checkpoint = args.checkpoint / "pytorch_model_fsdp_0"
    metadata = dcp.FileSystemReader(checkpoint).read_metadata().state_dict_metadata
    prefix = "model.audio_sampler.flow."
    latent_dim = metadata[prefix + "input_projection.weight"].size[1]
    model_dim, condition_dim = metadata[prefix + "conditioning_embedding.weight"].size
    blocks = sum(name.startswith(prefix + "blocks.") and name.endswith(".linear1.weight") for name in metadata)
    reference = TrainingFlowMap(latent_dim, model_dim, condition_dim, blocks)
    state = {"model": {"audio_sampler.flow." + name: tensor for name, tensor in reference.state_dict().items()}}
    dcp.load(state, checkpoint_id=checkpoint, no_dist=True)
    reference.load_state_dict(
        {name.removeprefix("audio_sampler.flow."): tensor for name, tensor in state["model"].items()},
        strict=True,
    )
    native = FlowMap(latent_dim, model_dim, condition_dim, blocks)
    inference_weights = reference.state_dict()
    del inference_weights["log_precision"]
    native.load_state_dict(inference_weights, strict=True)
    reference.cuda().eval()
    native.cuda().eval()

    for batch_size in (1, 8, 32):
        torch.manual_seed(71)
        conditioning = torch.randn(batch_size, condition_dim, device="cuda")
        for steps in (1, 2, 4):
            for temperature in (0.0, 0.3, 1.0):
                native.inference_steps = steps
                native.sampling_temperature = temperature
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.autocast):
                    torch.manual_seed(19)
                    expected = reference.sample(conditioning, num_steps=steps, temperature=temperature)
                    torch.manual_seed(19)
                    noise = torch.randn(batch_size, latent_dim, device="cuda")
                    actual = native.sample(conditioning, noise)
                    individual = torch.cat(
                        [native.sample(conditioning[i : i + 1], noise[i : i + 1]) for i in range(batch_size)]
                    )
                torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)
                torch.testing.assert_close(actual, individual, atol=2e-5, rtol=2e-4)
                print(
                    json.dumps(
                        {
                            "batch_size": batch_size,
                            "autocast": args.autocast,
                            "steps": steps,
                            "temperature": temperature,
                            "training_max_abs_error": (actual - expected).abs().max().item(),
                            "batch_max_abs_error": (actual - individual).abs().max().item(),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
