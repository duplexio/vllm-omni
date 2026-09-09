"""Compare concurrent native trajectories with one packed training batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from duplexio.opd import PolicyTrajectory, pack_policy_replay
from safetensors.torch import load_file

from examples.offline_inference.duplexio.check_backbone_parity import VoiceManifest, load_reference


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trajectories", type=Path, nargs="+")
    args = parser.parse_args()
    traces = [
        PolicyTrajectory.model_validate(torch.load(path, map_location="cpu", weights_only=True))
        for path in args.trajectories
    ]
    model = load_reference(args.checkpoint).cuda()
    voices = VoiceManifest.model_validate_json((args.checkpoint / "voices.json").read_text())
    pools = load_file(args.checkpoint / "voices.safetensors")
    speakers = torch.stack([
        pools[voices.voices[trace.runtime_config["duplexio_voice"]].tensor][
            trace.runtime_config["duplexio_voice_embedding_index"]
        ]
        for trace in traces
    ])
    batch = pack_policy_replay(
        traces,
        speakers,
        policy_version=traces[0].policy_version,
        silence_token_id=model.silence_token_id,
        device=torch.device("cuda"),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch.model_inputs)
    hidden = output.cell_hidden.float().cpu()
    offset = 0
    for trace in traces:
        assert trace.predictor_hiddens is not None
        reference = hidden[trace.prediction_rows + offset]
        native = trace.predictor_hiddens.float()
        assert torch.isfinite(native - reference).all()
        print(json.dumps({
            "conversation": trace.conversation_id,
            "max_absolute": (native - reference).abs().max().item(),
            "relative_l2": ((native - reference).norm() / reference.norm()).item(),
            "predictions": reference.shape[0],
        }), flush=True)
        offset += trace.text_ids.shape[0]
    print("Packed replay differences reported above; use check_backbone_parity for token-policy KL.", flush=True)


if __name__ == "__main__":
    main()
