"""Requests own sampling; user exploration must not alter the agent policy."""

import pytest
from pydantic import ValidationError

from vllm_omni.model_executor.models.duplexio.sampling_config import SamplingConfig, sampling_runtime


def test_user_rl_is_independent_of_agent_and_serving() -> None:
    serving = SamplingConfig()
    rl = SamplingConfig.model_validate({
        "user": {"emission": {"temperature": 1.0}, "content": {"temperature": 1.0}},
    })
    runtime = sampling_runtime(rl)
    assert rl.agent == serving.agent
    assert runtime["duplexio_text_sampling"] == {
        "temperature": 0.6, "top_k": 20, "top_p": 0.95,
    }
    assert runtime["duplexio_user_sampling"]["content"] == {
        "temperature": 1.0, "top_k": None, "top_p": None,
    }
    assert runtime["duplexio_emit_temperatures"]["user"] == 1.0
    assert serving.user.emission.temperature == serving.user.content.temperature == 0.0


def test_training_and_serving_sampling_contracts_agree() -> None:
    training = pytest.importorskip("duplexio.config")
    settings = {
        "agent": {"emission": {"temperature": 1.0}, "content": {"temperature": 0.6, "top_k": 20, "top_p": 0.95}},
        "user": {"emission": {"temperature": 1.0}, "content": {"temperature": 1.0}},
    }
    parsed = SamplingConfig.model_validate(settings)
    assert training.TokenSamplingConfig.model_validate(settings["agent"]).model_dump() == parsed.agent.model_dump()
    assert training.TokenSamplingConfig.model_validate(settings["user"]).model_dump() == parsed.user.model_dump()


@pytest.mark.parametrize("content", [{"mode": "argmax"}, {"temperature": -1}, {"top_k": 0}, {"top_p": 0}])
def test_invalid_or_obsolete_settings_fail_at_request_boundary(content) -> None:
    with pytest.raises(ValidationError):
        SamplingConfig.model_validate({"user": {"content": content}})
