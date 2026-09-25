# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def fresh_compile_cache():
    """Start each module with no compiled graphs.

    Modules drive the compiled helpers with their own toy shapes and grad
    modes, so across a whole run one helper would collect more graphs than
    Dynamo's recompile limit allows.
    """
    torch._dynamo.reset()
    yield
