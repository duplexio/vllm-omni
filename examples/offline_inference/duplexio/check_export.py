"""Load a real serving export through the native vLLM model loader."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch
from pydantic import BaseModel
from quack import rmsnorm
from quack.mlp import mlp_func
from safetensors import safe_open
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated
from vllm.config import set_current_vllm_config
from vllm.distributed import init_distributed_environment, initialize_model_parallel
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader import get_model

from vllm_omni.engine.arg_utils import register_omni_models_to_vllm


class WeightIndex(BaseModel):
    weight_map: dict[str, str]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    register_omni_models_to_vllm()
    engine_args = EngineArgs(
        model=str(args.checkpoint),
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        async_scheduling=False,
        max_num_seqs=1,
        max_num_batched_tokens=24576,
        max_model_len=24576,
        compilation_config={"mode": 0, "cudagraph_mode": "NONE"},
    )
    config = engine_args.create_engine_config()
    with (
        tempfile.TemporaryDirectory(prefix="duplexio-native-load-") as directory,
        set_current_vllm_config(config),
    ):
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"file://{directory}/distributed",
        )
        try:
            initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
            model = get_model(vllm_config=config)
            assert model.agent_emit_head.in_features == 6 * model.text_config.hidden_size
            assert not hasattr(model, "agent_audio_output_adapter")
            assert not hasattr(model, "user_audio_embedding")
            assert not hasattr(model, "emit_heads")
            index = WeightIndex.model_validate_json((args.checkpoint / "model.safetensors.index.json").read_text())
            norm_count = 0
            for filename in sorted(set(index.weight_map.values())):
                with safe_open(args.checkpoint / filename, framework="pt") as shard:
                    for name in shard.keys():
                        if name.endswith(".scale") and name.startswith("llm.base_model.model."):
                            source = shard.get_tensor(name)
                            parameter = model.get_parameter(name.removesuffix("scale") + "weight")
                            torch.testing.assert_close(parameter.cpu(), source, rtol=0, atol=0)
                            norm_count += 1
            native_norm = model.llm.base_model.model.layers[0].linear_attn.norm
            reference_norm = Qwen3_5RMSNormGated(native_norm.weight.numel(), eps=model.text_config.rms_norm_eps)
            reference_norm = reference_norm.to(device=native_norm.weight.device, dtype=native_norm.weight.dtype)
            reference_norm.load_state_dict(native_norm.state_dict())
            torch.manual_seed(321)
            probe = native_norm.weight.new_empty(64, native_norm.weight.numel()).normal_()
            gate = torch.randn_like(probe)
            torch.testing.assert_close(native_norm(probe, gate), reference_norm(probe, gate), rtol=0, atol=0)
            layer = model.llm.base_model.model.layers[0]
            hidden = probe.new_empty(64, model.text_config.hidden_size).normal_()
            residual = torch.randn_like(hidden)
            norm = layer.input_layernorm
            expected_norm = rmsnorm(hidden, norm.weight, eps=model.text_config.rms_norm_eps)
            actual_norm = norm(hidden)
            expected_fused = rmsnorm(hidden, norm.weight, residual=residual,
                                     eps=model.text_config.rms_norm_eps, prenorm=True)
            actual_fused = norm(hidden, residual)
            expected_mlp = mlp_func(hidden, layer.mlp.gate_up_proj.weight, layer.mlp.down_proj.weight,
                                    activation="swiglu", recompute=False, concat_layout=True)
            actual_mlp = layer.mlp(hidden)
            comparisons = {
                "rmsnorm": (actual_norm, expected_norm),
                "fused_rmsnorm": (actual_fused[0], expected_fused[0]),
                "residual": (actual_fused[1], expected_fused[1]),
                "mlp": (actual_mlp, expected_mlp),
            }
            print(json.dumps({
                name: {"max_absolute": (actual - expected).abs().max().item(),
                       "different_fraction": (actual != expected).float().mean().item()}
                for name, (actual, expected) in comparisons.items()
            }), flush=True)
            for actual, expected in comparisons.values():
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            print(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint),
                        "audio_representation": model.config.audio_representation,
                        "parameters": sum(parameter.numel() for parameter in model.parameters()),
                        "peak_memory_bytes": torch.accelerator.max_memory_allocated(),
                        "result": "native_model_loaded",
                        "exact_norm_weights": norm_count,
                    }
                ),
                flush=True,
            )
        finally:
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
