# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Sessions own their sampling; user exploration must not alter the agent policy."""

import pytest
from pydantic import ValidationError

from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig


def test_user_sampling_is_independent_of_the_agent_default() -> None:
    serving = SamplingConfig()
    exploring = SamplingConfig.model_validate(
        {
            "user": {"emission": {"temperature": 1.0}, "content": {"temperature": 1.0}},
        }
    )
    assert exploring.agent == serving.agent
    assert serving.agent.content.model_dump() == {"temperature": 0.6, "top_k": 20, "top_p": 0.95}
    assert exploring.user.content.model_dump() == {"temperature": 1.0, "top_k": None, "top_p": None}
    assert serving.user.emission.temperature == serving.user.content.temperature == 0.0
    assert serving.audio.temperature is None


@pytest.mark.parametrize("content", [{"mode": "argmax"}, {"temperature": -1}, {"top_k": 0}, {"top_p": 0}])
def test_invalid_or_obsolete_settings_fail_at_request_boundary(content) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"user": {"content": content}})
