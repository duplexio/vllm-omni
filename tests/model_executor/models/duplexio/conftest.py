# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    """Start each test with no compiled graphs.

    Tests drive the compiled helpers with their own toy shapes and grad modes,
    so across many tests one helper would collect more graphs than Dynamo's
    recompile limit allows.
    """
    torch._dynamo.reset()
    yield
