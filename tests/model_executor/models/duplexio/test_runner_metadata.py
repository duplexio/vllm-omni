# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import MethodType, SimpleNamespace
from typing import Any, cast

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner


def test_normal_attention_metadata_build_runs_model_extension(monkeypatch) -> None:
    metadata = object()
    upstream_call: dict[str, object] = {}

    def build_upstream(_runner: object, **kwargs: object):
        upstream_call.update(kwargs)
        return metadata, None

    monkeypatch.setattr(
        GPUModelRunner,
        "_build_attention_metadata",
        build_upstream,
    )
    runner = cast(Any, object.__new__(OmniGPUModelRunner))
    runner.input_batch = SimpleNamespace(req_ids=["first", "second"])
    extension_call: dict[str, object] = {}

    def attach_extension(
        _runner: object,
        attn_metadata: object,
        num_reqs: int,
    ) -> None:
        extension_call.update(
            attn_metadata=attn_metadata,
            num_reqs=num_reqs,
        )

    runner._maybe_update_model_attention_metadata = MethodType(
        attach_extension,
        runner,
    )

    returned = runner._build_attention_metadata(
        num_tokens=12,
        num_reqs=2,
        max_query_len=6,
        num_scheduled_tokens={"first": 6, "second": 6},
        slot_mappings={},
    )

    assert returned == (metadata, None)
    assert upstream_call["num_tokens"] == 12
    assert extension_call["attn_metadata"] is metadata
    assert extension_call["num_reqs"] == 2
