"""Explicit compiler boundaries and fixed projection arithmetic."""

from collections.abc import Callable
from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from quack.linear import linear_func
from torch import Tensor, nn

T = TypeVar("T")


@torch.compiler.disable
def call_compiled_function(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    with torch.compiler.set_stance("default"):
        return function(*args, **kwargs)


@torch.compiler.disable
def fixed_linear(inputs: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Quack owns this GEMM's compilation and autograd, including fixed tiles."""
    if inputs.device.type != "cuda":
        return F.linear(inputs, weight, bias)
    return linear_func(inputs, weight, bias, tuned=False)


class FixedLinear(nn.Linear):
    """Ordinary unsharded head weights with training's fixed GEMM configuration."""

    def forward(self, inputs: Tensor) -> Tensor:
        return fixed_linear(inputs, self.weight, self.bias)
