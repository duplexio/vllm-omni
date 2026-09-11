# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DuplexIO's native single-stage deployment."""

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"


def test_default_deploy_uses_flexattention_compatible_tiles() -> None:
    assert DUPLEXIO_PIPELINE.default_deploy_config_name == "duplexio.yaml"
    deploy = load_deploy_config(DEPLOY_DIR / "duplexio.yaml")
    stage = deploy.stages[0]

    # 12288 frames of six cells: a 16-minute session, past the model's
    # 4096-frame audio window so the audio ring actually recycles slots.
    assert stage.max_model_len == 12288 * 6
    assert stage.max_num_batched_tokens == 4096
    assert deploy.enable_chunked_prefill
    assert not stage.enforce_eager
    # mode 0: graph capture without an inductor pass, because a fullgraph
    # dynamo trace rejects the model's torch.compiler.disable()d quack GEMMs.
    assert stage.compilation_config == {
        "mode": 0,
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [6],
        "cudagraph_copy_inputs": True,
    }
    # FlexAttention tiles pages, so the page size must be a power of two; the
    # hybrid cache would otherwise derive it from the mamba state size.
    assert stage.engine_extras["block_size"] == 1024


def test_multistream_deploy_uses_batched_flexattention_path() -> None:
    deploy = load_deploy_config(DEPLOY_DIR / "duplexio-multistream.yaml")
    stage = deploy.stages[0]

    assert deploy.active_stream_window == 2
    assert deploy.duplex_session.max_sessions == 2
    assert stage.max_num_seqs == 2
    assert not stage.enforce_eager
    assert stage.compilation_config == {
        "mode": 0,
        "cudagraph_mode": "NONE",
    }
