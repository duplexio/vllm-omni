"""Stand-in trainer: push perturbed weights into live actors, then replay-check.

Run with the training environment and both repositories on PYTHONPATH. The
actors start from the native export; this process rebuilds the same training
model from that export, perturbs every trainable tensor, broadcasts the result
through the policy link, and verifies that the actors' subsequent predictor
states match a packed training replay of the perturbed model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from duplexio.modules.fixed_linear import fixed_linear
from duplexio.opd import RolloutPool, pack_policy_replay
from duplexio.opd_link import Coordinator, iterate_serving_weights, serving_weight_plan

from examples.offline_inference.duplexio.check_backbone_parity import load_reference


@torch.inference_mode()
def replay_metrics(model, pool: RolloutPool, trajectory, version: int) -> dict[str, float]:
    batch = pack_policy_replay(
        [trajectory],
        pool.speaker_embedding(trajectory.conversation_id).unsqueeze(0),
        silence_token_id=model.silence_token_id,
        device=torch.device("cuda"),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch.model_inputs)
    rows = trajectory.prediction_rows.cuda()
    current = (trajectory.row_versions == version).cuda()
    reference = output.cell_hidden[rows].float()
    native = trajectory.predictor_hiddens.cuda().float()
    projection = model.llm.stream_output_projection("agent")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        native_logits = fixed_linear(projection(native[:, 2]), model.token_head.weight).float()
        replay_logits = fixed_linear(projection(reference[:, 2]), model.token_head.weight).float()
    for logits in (native_logits, replay_logits):
        logits[:, model.silence_token_id] = -torch.inf
    native_logp = native_logits.log_softmax(-1)
    replay_logp = replay_logits.log_softmax(-1)
    kl = (native_logp.exp() * (native_logp - replay_logp)).nan_to_num(0.0).sum(-1)
    hidden_delta = (native - reference).abs().amax(dim=(1, 2))

    def stats(mask: torch.Tensor) -> dict[str, float]:
        if not mask.any():
            return {}
        return {
            "rows": int(mask.sum()),
            "kl_mean": kl[mask].mean().item(),
            "kl_max": kl[mask].max().item(),
            "hidden_max_abs": hidden_delta[mask].max().item(),
        }

    return {"current_version": stats(current), "older_versions": stats(~current)}


def load_training_checkpoint(config_path: Path, checkpoint: Path):
    """Build the training model and load one FSDP checkpoint, as export does."""
    import torch.distributed.checkpoint as dcp
    from duplexio.config import load_config
    from duplexio.models.duplexio import DuplexIOModel

    model_config = load_config(str(config_path)).model.model_copy(deep=True)
    model_config.gradient_checkpointing = False
    model = DuplexIOModel.load(model_config).eval()
    state = {"model": model.state_dict()}
    dcp.load(state, checkpoint_id=str(checkpoint / "pytorch_model_fsdp_0"), no_dist=True)
    model.load_state_dict(state["model"], strict=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path, help="Native export the actors started from")
    parser.add_argument("prepared", type=Path, help="Prepared conversation pool the actors use")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--port", type=int, default=29600)
    parser.add_argument("--actors", type=int, default=1)
    parser.add_argument("--trajectories", type=int, default=3, help="Trajectories fully under the pushed version")
    parser.add_argument("--noise", type=float, default=0.02, help="Relative perturbation of every trainable tensor")
    parser.add_argument("--checkpoint", type=Path, help="Push this FSDP training checkpoint instead of the export's weights")
    parser.add_argument("--config", type=Path, help="Training config matching --checkpoint")
    parser.add_argument("--timeout", type=float, default=1200.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True)

    if args.checkpoint is not None:
        model = load_training_checkpoint(args.config, args.checkpoint).cuda().eval()
    else:
        model = load_reference(args.export).cuda().eval()
    model.user_asr.requires_grad_(False)
    model.audio_codec.requires_grad_(False)
    generator = torch.Generator(device="cuda").manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad and args.noise > 0:
                spread = param.float().std() if param.numel() > 1 else param.float().abs().mean()
                scale = args.noise * spread.clamp_min(1e-6)
                param.add_((torch.randn(param.shape, generator=generator, device="cuda") * scale).to(param.dtype))
    plan = serving_weight_plan(model)
    payload_gib = sum(
        entry.dtype.itemsize * torch.Size(entry.shape).numel() for entry in plan
    ) / 1024**3
    pool = RolloutPool(args.prepared, args.export)
    coordinator = Coordinator(port=args.port, expected_actors=args.actors,
                              device=torch.device("cuda"), timeout_seconds=args.timeout)
    print(json.dumps({"endpoint": f"tcp://{coordinator.host}:{args.port}", "tensors": len(plan),
                      "payload_gib": payload_gib}), flush=True)
    deadline = time.monotonic() + args.timeout
    while not coordinator.form_group():
        if time.monotonic() > deadline:
            raise TimeoutError("Actors did not register")
        time.sleep(0.5)
    started = time.perf_counter()
    paused = coordinator.push_weights(1, plan, iterate_serving_weights(model, plan))
    print(json.dumps({"pushed_version": 1, "seconds": time.perf_counter() - started,
                      "actor_pause_seconds": paused}), flush=True)

    complete = 0
    index = 0
    try:
        while complete < args.trajectories:
            if time.monotonic() > deadline:
                raise TimeoutError("Actors did not deliver enough trajectories")
            messages = coordinator.take_trajectories()
            if not messages:
                time.sleep(0.5)
                continue
            for message in messages:
                trajectory = pool.trajectory(message)
                torch.save(trajectory.model_dump(), args.output_dir / f"trajectory_{index:06d}.pt")
                metrics = replay_metrics(model, pool, trajectory, version=1)
                versions = sorted(set(trajectory.row_versions.tolist()))
                print(json.dumps({"trajectory": index, "conversation": trajectory.conversation_id,
                                  "frames": trajectory.text_ids.shape[0], "row_versions": versions,
                                  **metrics}), flush=True)
                if versions == [1]:
                    complete += 1
                index += 1
    finally:
        coordinator.shutdown()


if __name__ == "__main__":
    main()
