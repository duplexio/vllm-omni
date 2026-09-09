"""Opt-in worker diagnostics for training/native backbone parity, not profiling."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle
from vllm.forward_context import get_forward_context


class LayerCapture:
    """Observe bounded engine forwards, including every row of small decode batches."""

    def __init__(self, model: nn.Module, path: Path, attention_layer: int, max_forwards: int) -> None:
        self.path = path
        self.max_forwards = max_forwards
        self.request_seeds: dict[int, int] = {}
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.handles: list[RemovableHandle] = []
        self.handles.append(model.register_forward_pre_hook(self.identify_requests, with_kwargs=True))
        backbone = model.llm.base_model.model
        self.rotary_emb = backbone.rotary_emb
        self.handles.append(backbone.layers[0].register_forward_pre_hook(self.begin, with_kwargs=True))
        for index, layer in enumerate(backbone.layers):
            self.handles.append(layer.register_forward_hook(self.layer_hook(index)))
            if index in (0, 1, attention_layer):
                self.handles.append(layer.input_layernorm.register_forward_hook(self.operator_hook(f"op{index}_input")))
                self.handles.append(
                    layer.post_attention_layernorm.register_forward_hook(
                        self.operator_hook(f"op{index}_post", prenorm=True)
                    )
                )
                self.handles.append(layer.mlp.register_forward_hook(self.operator_hook(f"op{index}_mlp")))
        self.full_attention = backbone.layers[attention_layer].self_attn
        self.handles.append(self.full_attention.attn.register_forward_hook(self.attention_hook))
        self.handles.append(model.logits_processor.register_forward_hook(self.head_hook("token_logits")))
        self.handles.append(model.agent_emit_head.register_forward_hook(self.head_hook("agent_emit")))
        self.handles.append(model.tool_call_emit_head.register_forward_hook(self.head_hook("tool_emit")))

    def select(self, tensor: Tensor) -> Tensor:
        selected = tensor if tensor.shape[0] <= 48 else torch.cat((tensor[:6], tensor[-6:]))
        return selected.detach().clone()

    def identify_requests(self, module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        for info in kwargs["model_intermediate_buffer"]:
            state = info["duplexio_working_state"]
            self.request_seeds[state.cache_epoch] = state.sampling_generator.initial_seed()

    def begin(self, module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        self.current = None
        if len(self.records) == self.max_forwards:
            return
        self.current = {
            "positions": self.select(kwargs["logical_positions"]),
            "epochs": self.select(kwargs["request_epochs"]),
            "input": self.select(kwargs["hidden_states"]),
        }
        self.records.append(self.current)
        if len(self.records) == 1:
            self.current["rope_positions"] = kwargs["positions"].detach().cpu().clone()

    def layer_hook(self, index: int):
        def capture(module: nn.Module, args: tuple, output: tuple[Tensor, Tensor]) -> None:
            if self.current is not None:
                hidden, residual = output
                self.current[f"layer_{index}"] = self.select(hidden + residual)
                if len(self.records) <= 3 and hidden.shape[0] <= 1536:
                    self.current[f"full_layer_{index}"] = (hidden + residual).detach().cpu().clone()

        return capture

    def finish(self) -> None:
        for handle in self.handles:
            handle.remove()
        assert self.records, "Layer hooks did not observe native forwards"
        for record in self.records:
            for name, value in record.items():
                if isinstance(value, Tensor):
                    record[name] = value.cpu()
        self.records[0]["request_seeds"] = self.request_seeds
        torch.save(self.records, self.path)

    def head_hook(self, name: str):
        def capture(module: nn.Module, args: tuple, output: Tensor) -> None:
            if self.current is not None:
                self.current[name] = output.detach().cpu().clone()

        return capture

    def operator_hook(self, name: str, prenorm: bool = False):
        def capture(module: nn.Module, args: tuple, output) -> None:
            if self.current is None or len(self.records) > 3 or args[0].shape[0] > 1536:
                return
            if prenorm:
                self.current[name + "_attention"] = args[0].detach().cpu().clone()
                output = output[0]
            self.current[name] = output.detach().cpu().clone()

        return capture

    def attention_hook(self, module: nn.Module, args: tuple, output: Tensor) -> None:
        if self.current is None:
            return
        metadata = get_forward_context().attn_metadata[module.layer_name]
        if metadata.block_table.shape[0] != 1:
            return
        if len(self.records) == 1:
            for name, value in zip(("attn_query", "attn_key", "attn_value"), args, strict=True):
                self.current[name] = value.detach().cpu().clone()
            self.current["attn_output"] = output.detach().cpu().clone()
            self.current["rope_cache"] = self.rotary_emb.cos_sin_cache[:256].cpu().clone()
            torch.save(
                {name: value.cpu() for name, value in self.full_attention.state_dict().items()},
                self.path.with_suffix(".weights.pt"),
            )
        if len(self.records) != 3:
            return
        layout = metadata.duplexio_layout
        pages = metadata.block_table[0, : layout.max_blocks]
        torch.save(
            {
                "query": args[0].cpu(),
                "key": args[1].cpu(),
                "value": args[2].cpu(),
                "output": output.cpu(),
                "cache": module.kv_cache[pages].cpu(),
                "layout": asdict(layout),
                "capacity": metadata.duplexio_packed_capacity,
                "query_start_loc": metadata.query_start_loc.cpu(),
                "doc_ids": metadata.doc_ids.cpu(),
                "num_actual_tokens": metadata.num_actual_tokens,
            },
            self.path.with_suffix(".attention.pt"),
        )


class LayerCaptureWorker:
    """Use vLLM's explicit worker-extension/RPC interface to install hooks."""

    def start_layer_capture(self, path: str, attention_layer: int, max_forwards: int) -> None:
        self.layer_capture = LayerCapture(self.model_runner.model, Path(path), attention_layer, max_forwards)

    def finish_layer_capture(self) -> None:
        self.layer_capture.finish()
        del self.layer_capture
