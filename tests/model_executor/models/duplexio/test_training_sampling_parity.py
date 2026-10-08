# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The serving sampling config parses exactly as the training config does."""

import pytest

from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig


def test_training_and_serving_sampling_contracts_agree() -> None:
    training = pytest.importorskip("duplexio.config")
    settings = {
        "agent": {"emission": {"temperature": 1.0}, "content": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}},
        "user": {"emission": {"temperature": 1.0}, "content": {"temperature": 1.0}},
    }
    parsed = SamplingConfig.model_validate(settings)
    assert training.TokenSamplingConfig.model_validate(settings["agent"]).model_dump() == parsed.agent.model_dump()
    assert training.TokenSamplingConfig.model_validate(settings["user"]).model_dump() == parsed.user.model_dump()
