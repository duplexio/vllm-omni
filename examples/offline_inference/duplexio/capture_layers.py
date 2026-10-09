"""Opt-in worker diagnostics for training/native backbone parity, not profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle


class LayerCapture:
    """Observe bounded engine forwards, including every row of small decode batches."""

    def __init__(self, model: nn.Module, path: Path, attention_layer: int, max_forwards: int) -> None:
        self.path = path
        self.max_forwards = max_forwards
        self.positions: Tensor | None = None
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.handles: list[RemovableHandle] = []
        self.handles.append(model.register_forward_pre_hook(self.take_positions, with_kwargs=True))
        backbone = model.llm.base_model.model
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
        self.handles.append(model.logits_processor.register_forward_hook(self.head_hook("token_logits")))
        self.handles.append(model.agent_emit_head.register_forward_hook(self.head_hook("agent_emit")))
        self.handles.append(model.tool_call_emit_head.register_forward_hook(self.head_hook("tool_emit")))

    def select(self, tensor: Tensor) -> Tensor:
        selected = tensor if tensor.shape[0] <= 48 else torch.cat((tensor[:6], tensor[-6:]))
        return selected.detach().clone()

    def take_positions(self, module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        """vLLM's token positions index training's flat row-major sequence."""
        positions = kwargs["positions"]
        self.positions = positions if positions.ndim == 1 else positions[0]

    def begin(self, module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        self.current = None
        if len(self.records) == self.max_forwards:
            return
        assert self.positions is not None
        self.current = {
            "positions": self.select(self.positions),
            "input": self.select(kwargs["hidden_states"]),
        }
        self.records.append(self.current)

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


class LayerCaptureWorker:
    """Use vLLM's explicit worker-extension/RPC interface to install hooks."""

    def start_layer_capture(self, path: str, attention_layer: int, max_forwards: int) -> None:
        self.layer_capture = LayerCapture(self.model_runner.model, Path(path), attention_layer, max_forwards)

    def finish_layer_capture(self) -> None:
        self.layer_capture.finish()
        del self.layer_capture
