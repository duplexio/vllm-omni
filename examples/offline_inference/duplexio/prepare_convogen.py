"""Cache streaming-equivalent ASR frames for the saved held-out Convogen cases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from duplexio.datasets.conversation import Conversation
from pydantic import BaseModel, ConfigDict, TypeAdapter
from safetensors import safe_open
from torch import Tensor

from vllm_omni.experimental.fullduplex.duplexio.offline import PreparedConversation
from vllm_omni.model_executor.models.duplexio.configuration_duplexio import DuplexIOConfig
from vllm_omni.model_executor.models.duplexio.fastconformer import (
    FastConformerAudioStreamState,
    FastConformerRNNT,
    streaming_resample_chunk,
)


class SavedCase(BaseModel):
    """Boundary for logs/text_policy_convogen_inputs.pt from the training repo."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    metadata: dict[str, Any]
    prompt_ids: list[int]
    user_audio: Tensor
    user_token_ids: Tensor
    speaker_embedding: Tensor


def load_encoder(checkpoint: Path, device: torch.device) -> FastConformerRNNT:
    """Load only the frozen user subsystem, including its exported frontend."""
    config = DuplexIOConfig.from_pretrained(checkpoint, local_files_only=True)
    encoder = FastConformerRNNT.from_export(config.user_asr_config, checkpoint)
    encoder.model.to(dtype=torch.bfloat16)
    encoder.to(device)
    weights = {}
    for path in sorted(checkpoint.glob("model-*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name.startswith("user_asr."):
                    weights[name.removeprefix("user_asr.")] = shard.get_tensor(name)
    encoder.load_state_dict(weights, strict=True)
    return encoder


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("conversations", type=Path, help="Source Convogen conversations.jsonl")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Prepared output already exists")
    cases = TypeAdapter(list[SavedCase]).validate_python(torch.load(args.cases, weights_only=True))
    case_ids = {case.metadata["id"] for case in cases}
    sources = {}
    with args.conversations.open() as lines:
        for line in lines:
            conversation = Conversation.model_validate_json(line)
            if conversation.id in case_ids:
                sources[conversation.id] = conversation
    assert set(sources) == case_ids
    config = DuplexIOConfig.from_pretrained(args.checkpoint, local_files_only=True)
    device = torch.device("cuda")
    encoder = load_encoder(args.checkpoint, device)
    prepared = []
    started = time.perf_counter()
    for case in cases:
        source = sources[case.metadata["id"]]
        state = FastConformerAudioStreamState()
        waveform = case.user_audio.to(device).view(-1, config.frame_size)
        features = []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for frame in waveform:
                resampled, tail = streaming_resample_chunk(frame, state.resample_tail, config.sample_rate, 16000)
                encoded, state = encoder.encode_audio_chunk(resampled, state)
                state.resample_tail = tail
                assert encoded.shape == (1, 1, encoder.output_dim)
                features.append(encoded[0])
        prepared.append(
            PreparedConversation(
                conversation_id=source.id,
                system_token_ids=case.prompt_ids,
                user_features=torch.cat(features).cpu(),
                user_token_ids=case.user_token_ids,
                voice=config.default_voice,
                tools=[tool.as_openai_tool() for tool in source.tools],
                metadata={
                    **case.metadata,
                    "teacher_system": source.assistant.system_prompt,
                    "voice_source": "exported voice pool, not the original dataset speaker",
                },
            ).model_dump()
        )
    torch.save(prepared, args.output)
    print(
        json.dumps({"prepared": len(prepared), "seconds": time.perf_counter() - started, "output": str(args.output)}),
        flush=True,
    )


if __name__ == "__main__":
    main()
