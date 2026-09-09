"""Replay a native trajectory through the actual training forward implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import torch
from duplexio.config import DuplexioConfig
from duplexio.models.duplexio import DuplexIOModel
from duplexio.modules.audio_codec import MimiCodec, PocketMimiCodec
from duplexio.modules.audio_representation import ContinuousAudioRepresentation, QuantizedAudioRepresentation
from duplexio.modules.continuous_mimi import ContinuousMimiModel
from duplexio.modules.fastconformer_rnnt import FastConformerRNNT
from duplexio.modules.fixed_linear import fixed_linear
from duplexio.multistream.checkpoint import prepare_qwen_for_multistream
from duplexio.multistream.modeling import MultiStreamQwen
from duplexio.opd import PolicyTrajectory, pack_policy_replay
from pydantic import BaseModel
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForRNNT, AutoProcessor, AutoTokenizer, MimiModel


class ReferenceExport(BaseModel):
    """Portable metadata needed by the training-side checker, without vLLM."""

    duplexio_export_version: Literal[5]
    emit_head_input: Literal["full_frame"]
    audio_conditioning: Literal["agent_audio_cell"]
    text_config: dict[str, Any]
    user_asr_config: dict[str, Any]
    audio_codec_config: dict[str, Any]
    audio_adapter_config: dict[str, Any]
    audio_representation: Literal["continuous", "quantized"]
    continuous_audio_config: dict[str, Any] = {}
    flowmap_config: dict[str, Any] = {}
    quantized_audio_config: dict[str, Any] = {}
    depth_transformer_config: dict[str, Any] = {}
    tied_weight_aliases: dict[str, str]
    silence_token_id: int
    speaker_embed_dim: int
    speaker_lda_dim: int | None
    audio_attention_window_frames: int
    sample_rate: int
    frame_rate: float
    frame_size: int


class WeightIndex(BaseModel):
    weight_map: dict[str, str]


class VoiceEntry(BaseModel):
    tensor: str


class VoiceManifest(BaseModel):
    voices: dict[str, VoiceEntry]


def load_reference(checkpoint: Path) -> DuplexIOModel:
    """Rebuild training modules without fetching separate pretrained assets.

    This is an inference/replay check, not optimizer or audio-loss resumption.
    FlowMap's training-only loss precision is deliberately absent from exports.
    Current validation checkpoints do not use speaker LDA.
    """
    export = ReferenceExport.model_validate_json((checkpoint / "config.json").read_text())
    if export.speaker_lda_dim is not None:
        raise ValueError("This reference checker does not yet load speaker-LDA assets")
    index = WeightIndex.model_validate_json((checkpoint / "model.safetensors.index.json").read_text())
    weights = {}
    for shard in sorted(set(index.weight_map.values())):
        weights.update(load_file(checkpoint / shard))
    for alias, original in export.tied_weight_aliases.items():
        weights[alias] = weights[original]

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    print("Constructing reference modules from exported metadata", flush=True)
    text_config = dict(export.text_config)
    base = AutoModelForCausalLM.from_config(
        AutoConfig.for_model(text_config.pop("model_type"), **text_config),
        dtype=torch.bfloat16,
    )
    prepare_qwen_for_multistream(base, num_channels=6, use_flex_attention=True)
    llm = MultiStreamQwen(
        base_model=base,
        tokenizer=tokenizer,
        silence_token_id=export.silence_token_id,
        model_id=str(checkpoint),
        initialize_channel_embeddings=False,
    ).to(dtype=torch.bfloat16)
    rotary = llm.base_model.model.rotary_emb
    rotary.inv_freq = weights.pop("llm.base_model.model.rotary_emb.inverse_frequencies")
    rotary.original_inv_freq = rotary.inv_freq
    rotary.attention_scaling = weights.pop("llm.base_model.model.rotary_emb.attention_scaling").item()
    asr_config = dict(export.user_asr_config)
    asr = FastConformerRNNT(
        AutoModelForRNNT.from_config(
            AutoConfig.for_model(asr_config.pop("model_type"), **asr_config),
            dtype=torch.bfloat16,
        ),
        AutoProcessor.from_pretrained(checkpoint / "user_asr", local_files_only=True),
        frozen=True,
    )
    for name in ("mel_filters", "stft_window"):
        torch.testing.assert_close(getattr(asr, name), weights.pop(f"user_asr.{name}"), rtol=0, atol=0)

    model_options = {
        "llm_multistream_checkpoint": str(checkpoint),
        "audio_representation": export.audio_representation,
        "audio_adapter": export.audio_adapter_config,
        "speaker_embed_dim": export.speaker_embed_dim,
        "audio_attention_window_frames": export.audio_attention_window_frames,
        "sample_rate": export.sample_rate,
        "frame_rate": export.frame_rate,
        "gradient_checkpointing": False,
        "vap": {"enabled": False},
    }
    if export.audio_representation == "continuous":
        codec = PocketMimiCodec(ContinuousMimiModel(), frame_size=export.frame_size)
        representation = ContinuousAudioRepresentation(
            export.continuous_audio_config["embedding_dim"],
            None,
            default_mean=weights.pop("audio_representation.embedding_mean"),
            default_scale=weights.pop("audio_representation.embedding_scale"),
        )
        model_options["audio_codec"] = "pocket-tts:en"
        model_options["latent_flow_map"] = export.flowmap_config
    else:
        codec_config = dict(export.audio_codec_config)
        codec = MimiCodec(
            MimiModel(AutoConfig.for_model(codec_config.pop("model_type"), **codec_config)).to(torch.bfloat16),
            frame_size=export.frame_size,
            sample_rate=export.sample_rate,
            frame_rate=export.frame_rate,
        )
        quantized = export.quantized_audio_config
        representation = QuantizedAudioRepresentation(**quantized)
        model_options["audio_codec"] = "kyutai/mimi"
        model_options["quantized_representation"] = {
            key: value for key, value in quantized.items() if key != "codebook_size"
        }
        model_options["depth_autoregressive"] = {
            key: value for key, value in export.depth_transformer_config.items() if key != "implementation"
        }
    model = DuplexIOModel(DuplexioConfig.model_validate(model_options), codec, representation, asr, llm=llm)
    training_weights = {
        ("audio_codec.model." + name.removeprefix("audio_codec.") if name.startswith("audio_codec.") else name): tensor
        for name, tensor in weights.items()
    }
    missing, unexpected = model.load_state_dict(training_weights, strict=False)
    expected_missing = ["audio_sampler.flow.log_precision"] if export.audio_representation == "continuous" else []
    assert missing == expected_missing and not unexpected, (missing, unexpected)
    print("Loaded all exported reference weights; only training-only loss precision omitted", flush=True)
    return model.eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("--layer-capture", type=Path)
    parser.add_argument("--output", type=Path, help="Save reference tensors for first-divergence analysis")
    parser.add_argument("--cached", action="store_true", help="Also replay through training's cached prefill/decode")
    args = parser.parse_args()
    trace = PolicyTrajectory.model_validate(torch.load(args.trajectory, map_location="cpu", weights_only=True))
    assert trace.predictor_hiddens is not None, "Generate the trajectory with --record-hiddens"
    model = load_reference(args.checkpoint).cuda()
    voices = VoiceManifest.model_validate_json((args.checkpoint / "voices.json").read_text())
    voice = voices.voices[trace.runtime_config["duplexio_voice"]]
    pool = load_file(args.checkpoint / "voices.safetensors")[voice.tensor]
    speaker = pool[trace.runtime_config["duplexio_voice_embedding_index"]]
    batch = pack_policy_replay(
        [trace],
        speaker.unsqueeze(0),
        silence_token_id=model.silence_token_id,
        device=torch.device("cuda"),
    )
    handles = []
    reference_layers = {}
    if args.layer_capture is not None:
        records = torch.load(args.layer_capture, map_location="cpu", weights_only=True)
        positions = torch.cat([record["positions"] for record in records]).cuda()

        def compare(name: str, hidden: torch.Tensor) -> None:
            expected = hidden.flatten(0, 1).index_select(0, positions).float().cpu()
            reference_layers[name] = expected
            actual = torch.cat([record[name] for record in records]).float()
            full = [record[f"full_{name}"] for record in records if f"full_{name}" in record]
            if full:
                full_actual = torch.cat(full).float()
                full_expected = hidden.flatten(0, 1)[:full_actual.shape[0]].float().cpu()
                print(json.dumps({
                    "full_stage": name,
                    "max_absolute": (full_actual - full_expected).abs().max().item(),
                    "different_values": (full_actual != full_expected).count_nonzero().item(),
                }), flush=True)
            delta = actual - expected
            print(
                json.dumps(
                    {
                        "stage": name,
                        "max_absolute": delta.abs().max().item(),
                        "relative_rms": (delta.square().mean() / expected.square().mean()).sqrt().item(),
                        "max_absolute_by_cell": delta.reshape(-1, 6, delta.shape[-1]).abs().amax((0, 2)).tolist(),
                        "by_append": [
                            (part.float() - ref.float()).abs().max().item()
                            for part, ref in zip(
                                actual.split([len(r["positions"]) for r in records]),
                                expected.split([len(r["positions"]) for r in records]),
                                strict=True,
                            )
                        ],
                    }
                ),
                flush=True,
            )

        def capture_input(module, args, kwargs):
            compare("input", kwargs["inputs_embeds"])

        def capture_layer(index: int):
            def capture(module, args, output):
                compare(f"layer_{index}", output)

            return capture

        backbone = model.llm.base_model.model
        def capture_rope(module, inputs, output):
            reference_layers["rope_cos"] = output[0].detach().cpu()
            reference_layers["rope_sin"] = output[1].detach().cpu()

        handles.append(backbone.rotary_emb.register_forward_hook(capture_rope))
        handles.append(backbone.register_forward_pre_hook(capture_input, with_kwargs=True))
        for index, layer in enumerate(backbone.layers):
            handles.append(layer.register_forward_hook(capture_layer(index)))
            if f"op{index}_input" in records[0]:

                def operator_hook(name: str, prenorm: bool = False):
                    def capture(module, args, output):
                        values = {name: output[0] if prenorm else output}
                        if prenorm:
                            values[name + "_attention"] = args[0]
                        for key, value in values.items():
                            actual = torch.cat([record[key] for record in records if key in record]).float()
                            expected = value.flatten(0, 1)[: actual.shape[0]].float().cpu()
                            reference_layers[key] = expected
                            delta = (actual - expected).abs()
                            print(
                                json.dumps(
                                    {
                                        "operator": key,
                                        "max_by_cell": delta.view(-1, 6, delta.shape[-1]).amax((0, 2)).tolist(),
                                        "different_values": delta.count_nonzero().item(),
                                    }
                                ),
                                flush=True,
                            )

                    return capture

                handles.append(layer.input_layernorm.register_forward_hook(operator_hook(f"op{index}_input")))
                handles.append(
                    layer.post_attention_layernorm.register_forward_hook(operator_hook(f"op{index}_post", prenorm=True))
                )
                handles.append(layer.mlp.register_forward_hook(operator_hook(f"op{index}_mlp")))
                if layer.block_type == "full_attention":
                    def save_projection(name: str, use_input: bool = False):
                        def capture(module, inputs, output):
                            value = inputs[0] if use_input else output
                            reference_layers[name] = value.detach().cpu()
                        return capture

                    attention = layer.self_attn
                    for name, projection in (
                        ("q", attention.q_proj), ("k", attention.k_proj),
                        ("v", attention.v_proj), ("q_norm", attention.q_norm),
                        ("k_norm", attention.k_norm),
                    ):
                        handles.append(projection.register_forward_hook(save_projection(f"op{index}_{name}")))
                    handles.append(attention.o_proj.register_forward_hook(save_projection(f"op{index}_gated", use_input=True)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(**batch.model_inputs)
    for handle in handles:
        handle.remove()
    reference = output.cell_hidden[trace.prediction_rows.cuda()].float().cpu()
    native = trace.predictor_hiddens.float()
    difference = native - reference
    assert torch.isfinite(difference).all()
    print(
        json.dumps(
            {
                "conversation": trace.conversation_id,
                "max_absolute_by_cell": difference.abs().amax((0, 2)).tolist(),
                "relative_rms_by_cell": (difference.square().mean((0, 2)) / reference.square().mean((0, 2)))
                .sqrt()
                .tolist(),
                "first_prediction_max_by_cell": difference[0].abs().amax(-1).tolist(),
            }
        ),
        flush=True,
    )
    kl_values, selected_logp_errors, top1_matches = [], [], []
    agent_projection = model.llm.stream_output_projection("agent")
    for start in range(0, native.shape[0], 16):
        stop = start + 16
        with torch.autocast("cuda", dtype=torch.bfloat16):
            native_logits = fixed_linear(agent_projection(native[start:stop, 2].cuda()), model.token_head.weight)
            reference_logits = fixed_linear(agent_projection(reference[start:stop, 2].cuda()), model.token_head.weight)
        native_logits[:, model.silence_token_id] = -torch.inf
        reference_logits[:, model.silence_token_id] = -torch.inf
        native_logp = native_logits.float().log_softmax(-1)
        reference_logp = reference_logits.float().log_softmax(-1)
        logp_delta = native_logp - reference_logp
        logp_delta[:, model.silence_token_id] = 0
        kl_values.append((-reference_logp.exp() * logp_delta).sum(-1).cpu())
        top1_matches.append((native_logits.argmax(-1) == reference_logits.argmax(-1)).cpu())
        ids = trace.sampled_agent_ids[start:stop].cuda()
        emitted = ids != model.silence_token_id
        selected_logp_errors.append(logp_delta.gather(1, ids[:, None])[emitted, 0].abs().cpu())
    kl = torch.cat(kl_values)
    selected_errors = torch.cat(selected_logp_errors)
    print(
        json.dumps(
            {
                "agent_distribution_same_projection": {
                    "mean_kl_reference_to_native": kl.mean().item(),
                    "max_kl_reference_to_native": kl.max().item(),
                    "first_prediction_kl": kl[0].item(),
                    "top1_agreement": torch.cat(top1_matches).float().mean().item(),
                    "emitted_logp_max_error": selected_errors.max().item() if selected_errors.numel() else None,
                },
            }
        ),
        flush=True,
    )
    native_heads, reference_heads = [], []
    if args.layer_capture is not None:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cell_hidden = output.cell_hidden
            agent_projected = model.llm.stream_output_projection("agent")(cell_hidden[:, 2])
            tool_projected = model.llm.stream_output_projection("tool_call")(cell_hidden[:, 3])
            emit_logits = model._emit_logits_from_frame_hidden(cell_hidden.flatten(1))
            for record in records:
                if "token_logits" not in record:
                    continue
                frame = record["positions"][-1].item() // 6
                logits = fixed_linear(
                    torch.stack((agent_projected[frame], tool_projected[frame])), model.token_head.weight
                )
                expected = {
                    "token_logits": logits.float().cpu(),
                    "agent_emit": emit_logits[frame, :1].float().cpu(),
                    "tool_emit": emit_logits[frame, 1:].float().cpu(),
                }
                actual = {name: record[name].float() for name in expected}
                native_heads.append(actual)
                reference_heads.append(expected)
                print(
                    json.dumps(
                        {
                            "frame": frame,
                            "heads_max_absolute": {
                                name: (actual[name] - value).abs().max().item() for name, value in expected.items()
                            },
                        }
                    ),
                    flush=True,
                )
    if args.output is not None:
        torch.save(
            {
                "reference_layers": reference_layers,
                "prediction_hiddens": reference,
                "cell_hidden": output.cell_hidden.cpu(),
                "native_heads": native_heads,
                "reference_heads": reference_heads,
            },
            args.output,
        )
    if args.cached:
        frame_inputs = {
            name: batch.model_inputs[name].unsqueeze(0)
            for name in (
                "system_ids", "user_ids", "agent_ids", "tool_call_ids",
                "mask", "user_audio_enc", "agent_audio_enc",
            )
        }
        prefill = trace.prediction_rows[0].item() + 1
        cache = None
        cached_hiddens = []
        partitions = [(0, prefill), *((i, i + 1) for i in range(prefill, trace.text_ids.shape[0]))]
        for start, stop in partitions:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cached = model(
                    **{name: value[:, start:stop] for name, value in frame_inputs.items()},
                    agent_speaker_embeddings=batch.model_inputs["agent_speaker_embeddings"],
                    past_key_values=cache,
                    use_cache=True,
                    return_audio_conditioning=False,
                )
            cache = cached.past_key_values
            cached_hiddens.append(cached.cell_hidden[0].float().cpu())
        cached_hiddens = torch.cat(cached_hiddens)
        torch.testing.assert_close(cached_hiddens, output.cell_hidden.float().cpu(), rtol=0, atol=0)
        print("Cached training prefill/decode matches packed training exactly", flush=True)
    print("Native/training numerical differences reported above; bitwise equality is not required.", flush=True)


if __name__ == "__main__":
    main()
