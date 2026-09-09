"""Replay a captured paged-attention call without loading the model."""

import argparse
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.duplexio.kv_reclamation import DuplexIOKVLayout
from vllm_omni.model_executor.models.duplexio.qwen_backbone import DuplexIOFlexAttentionImpl


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture")
    args = parser.parse_args()
    record = torch.load(args.capture, weights_only=True, map_location="cuda")
    query, key, value = (record[name] for name in ("query", "key", "value"))
    cache = record["cache"]
    layout = DuplexIOKVLayout(**record["layout"])
    metadata = SimpleNamespace(
        num_actual_tokens=record["num_actual_tokens"],
        doc_ids=record["doc_ids"],
        query_start_loc=record["query_start_loc"],
        duplexio_layout=layout,
        block_size=layout.block_size,
        duplexio_packed_capacity=record["capacity"],
        block_table=torch.arange(cache.shape[0], device="cuda", dtype=torch.int32)[None],
    )
    for name in ("query", "key", "value", "cache", "output"):
        tensor = record[name]
        print(name, tuple(tensor.shape), "finite", torch.isfinite(tensor).float().mean().item(), flush=True)
    heads, dim, kv_heads = query.shape[1], query.shape[2] - 24, key.shape[1]
    backend = DuplexIOFlexAttentionImpl(heads, dim + 24, dim**-0.5, kv_heads, None, None, "auto")
    scales = SimpleNamespace(_k_scale=torch.ones((), device="cuda"), _v_scale=torch.ones((), device="cuda"))
    output = backend.forward(scales, query, key, value, cache, metadata, torch.empty_like(query))[..., :dim]
    print("replayed finite", torch.isfinite(output).float().mean().item(), flush=True)
    torch.testing.assert_close(output, record["output"].view_as(query)[..., :dim], rtol=0, atol=0)


if __name__ == "__main__":
    main()
