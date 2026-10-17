# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit compiler boundaries."""

from collections.abc import Callable
from typing import Any, TypeVar

import torch

T = TypeVar("T")


@torch.compiler.disable
def call_compiled_function(function: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    with torch.compiler.set_stance("default"):
        return function(*args, **kwargs)
