"""Compare GDN intermediates across the actual serving and training environments."""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5RMSNormGated


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("capture", type=Path)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    config = json.loads((args.checkpoint / "config.json").read_text())
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = "llm.base_model.model.layers.0.linear_attn."
    weights = {}
    for name, shard in index.items():
        if name.startswith(prefix):
            with safe_open(args.checkpoint / shard, framework="pt", device="cuda") as source:
                weights[name.removeprefix(prefix)] = source.get_tensor(name)
    records = torch.load(args.capture, weights_only=True)
    hidden = records[0]["op0_input"].cuda()
    length = hidden.shape[0]
    trace = torch.load(args.trajectory, weights_only=True)
    mask = (
        torch.cat((trace["text_ids"] != config["silence_token_id"], trace["audio_mask"][:, None].expand(-1, 2)), -1)
        .flatten()[:length]
        .cuda()
    )
    text_config = config["text_config"]
    kh, vh = text_config["linear_num_key_heads"], text_config["linear_num_value_heads"]
    kd, vd = text_config["linear_key_head_dim"], text_config["linear_value_head_dim"]
    widths = (kh * kd * 2 + vh * vd, vh * vd, vh, vh)
    projected_weights = [weights[f"in_proj_{name}.weight"] for name in ("qkv", "z", "b", "a")]
    result = {"input": hidden}
    if args.native:
        from vllm_omni.model_executor.models.duplexio.numerics import fixed_linear

        qkvz = fixed_linear(hidden, torch.cat(projected_weights[:2]))
        ba = fixed_linear(hidden, torch.cat(projected_weights[2:]))
        qkv, z = qkvz.split(widths[:2], -1)
        b, a = ba.chunk(2, -1)
    else:
        from duplexio.modules.fixed_linear import fixed_linear

        qkv, z, b, a = fixed_linear(hidden, torch.cat(projected_weights)).split(widths, -1)
    result.update(qkv=qkv, z=z, b=b, a=a)
    conv_weight = weights["conv1d.weight"]
    if args.native:
        from vllm_omni.model_executor.models.duplexio.row_semantics import expand_stream_conv_weight
        from vllm_omni.model_executor.models.duplexio.stream_conv import stream_causal_conv
        from vllm_omni.model_executor.models.duplexio.stream_gdn import (
            append_gdn,
            gdn_cache_dtypes,
            gdn_cache_shapes,
            prepare_gdn_inputs,
        )

        cache = tuple(
            torch.empty(1, *shape, device="cuda", dtype=dtype)
            for shape, dtype in zip(
                gdn_cache_shapes(1, kh, vh, kd, vd, conv_weight.shape[-1]), gdn_cache_dtypes(hidden.dtype), strict=True
            )
        )
        boundaries = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        slots = torch.zeros(1, device="cuda", dtype=torch.int32)
        has_state = torch.zeros(1, device="cuda", dtype=torch.bool)
        chunks = torch.tensor([(0, i) for i in range((length + 63) // 64)], device="cuda", dtype=torch.int32)
        conv = stream_causal_conv(
            qkv,
            expand_stream_conv_weight(conv_weight, num_cells=6),
            None,
            cache[0].transpose(-1, -2),
            slots,
            boundaries,
            has_state,
            chunks,
        )
        q, k, v, g, beta = prepare_gdn_inputs(
            conv,
            a.masked_fill(~mask[:, None], -torch.inf),
            b.masked_fill(~mask[:, None], -torch.inf),
            weights["A_log"].float(),
            weights["dt_bias"],
            kh,
            kd,
            vd,
        )
        core = append_gdn(q, k, v, g, beta, cache[1], slots, boundaries, has_state, chunks)
    else:
        from duplexio.modules.fla_block_gated_delta_rule.chunk import chunk_gated_delta_rule, l2_normalize
        from duplexio.modules.qwen3_5_stream_delta import BlockCausalConv1d, _qwen_beta_gate

        layer = nn.Conv1d(widths[0], widths[0], conv_weight.shape[-1], groups=widths[0], bias=False)
        layer.weight = nn.Parameter(conv_weight)
        conv = BlockCausalConv1d(layer, 6)(qkv.T[None], activation="silu")[0].T
        q, k, v = conv.split((kh * kd, kh * kd, vh * vd), -1)
        q, k = (l2_normalize(t.view(1, length, kh, kd))[0] for t in (q, k))
        v = v.view(length, vh, vd)
        beta, g = _qwen_beta_gate(b, a, weights["A_log"], weights["dt_bias"])
        beta, g = (torch.where(mask[:, None], t, 0) for t in (beta, g))
        core, _ = chunk_gated_delta_rule(*(t[None] for t in (q, k, v, g, beta)))
        core = core[0]
    norm = Qwen3_5RMSNormGated(vd, eps=text_config["rms_norm_eps"])
    norm.weight = nn.Parameter(weights["norm.weight"])
    norm.compile(dynamic=True, fullgraph=True)
    normalized = norm(core.reshape(-1, vd), z.reshape(-1, vd)).view(length, -1)
    output = fixed_linear(normalized, weights["out_proj.weight"])
    result.update(conv=conv, q=q, k=k, v=v, g=g, beta=beta, core=core, normalized=normalized, output=output)
    from duplexio.modules.fla_block_gated_delta_rule.chunk import chunk_gated_delta_rule_fwd_h
    from duplexio.modules.fla_block_gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from duplexio.modules.fla_block_gated_delta_rule.chunk_o import chunk_fwd_kernel_o
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2

    summed = chunk_local_cumsum(g[None], chunk_size=64, scale=RCP_LN2)
    w, u, triangle = chunk_gated_delta_rule_fwd_intra(
        k[None],
        v[None].contiguous(),
        summed,
        beta[None],
        use_exp2=True,
    )
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k[None], w, u, summed)
    result.update(summed=summed, w=w, u=u, triangle=triangle, h=h, v_new=v_new)
    for fusion in (False, True):
        out = torch.empty_like(v[None])
        compiled = chunk_fwd_kernel_o.fn.fn[(1, length // 64, vh)](
            q=q[None],
            k=k[None],
            v=v_new,
            h=h,
            g=summed,
            g_gamma=None,
            o=out,
            scale=kd**-0.5,
            cu_seqlens=None,
            chunk_indices=None,
            T=length,
            H=kh,
            HV=vh,
            K=kd,
            V=vd,
            BT=64,
            BK=128,
            BV=128,
            USE_G=True,
            USE_G_GAMMA=False,
            USE_EXP2=True,
            TRANSPOSE_STATE=False,
            IS_VARLEN=False,
            num_warps=8,
            num_stages=3,
            enable_fp_fusion=fusion,
        )
        result[f"core_fusion_{fusion}"] = out[0]
        args.output.with_suffix(f".fusion_{fusion}.ptx").write_text(compiled.asm["ptx"])
    torch.save({name: tensor.cpu() for name, tensor in result.items()}, args.output)
    print("native capture max error", (output.cpu() - records[0]["op0_post_attention"]).abs().max().item())


if __name__ == "__main__":
    main()
