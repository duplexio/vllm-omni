"""EngineCore wire round trips must retain model-input tensors."""

import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.async_engine_utils import upgrade_to_omni_request
from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens
from vllm_omni.request import OmniRequest, OmniStreamingUpdate


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("direct", [True, False])
def test_model_buffer_survives_engine_wire_and_streaming_update(dtype, direct) -> None:
    features = torch.randn(2, 8, dtype=dtype).t()
    prompt = {
        "prompt_token_ids": [0] * 6,
        "model_intermediate_buffer": {
            "embed": {"speech_feat": features},
            "duplex": {"seq": 3, "payload": {"format": "duplexio_features"}},
        },
    }
    params = SamplingParams(max_tokens=1)
    if direct:
        request = build_engine_core_request_from_tokens("wire-test", prompt, params, resumable=True)
    else:
        base = EngineCoreRequest(
            request_id="wire-test", prompt_token_ids=prompt["prompt_token_ids"],
            mm_features=None, sampling_params=params, pooling_params=None,
            arrival_time=0.0, lora_request=None, cache_salt=None,
            data_parallel_rank=None, resumable=True,
        )
        request = upgrade_to_omni_request(base, prompt)
    encoded = MsgpackEncoder().encode(request)
    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(encoded)
    scheduled = OmniRequest.from_engine_core_request(decoded, None)
    update = OmniStreamingUpdate.from_request(scheduled)
    restored = update.model_intermediate_buffer
    torch.testing.assert_close(restored["embed"]["speech_feat"], features)
    assert restored["duplex"] == prompt["model_intermediate_buffer"]["duplex"]
