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

    assert stage.max_model_len == 4096 * 6
    assert stage.max_num_batched_tokens == 4096
    assert deploy.enable_chunked_prefill
    assert not stage.enforce_eager
    assert stage.compilation_config == {
        "mode": 3,
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [6],
        "compile_sizes": ["cudagraph_capture_sizes"],
        "cudagraph_copy_inputs": True,
    }
    # flex_attn_kv_block_size stays unset: the DuplexIO metadata builder
    # defaults it to the largest power of two dividing the physical page,
    # which tracks the cache-metadata width automatically.
    assert stage.engine_extras["attention_config"] == {
        "flex_attn_block_m": 16,
        "flex_attn_block_n": 16,
    }


def test_multistream_deploy_uses_batched_flexattention_path() -> None:
    deploy = load_deploy_config(DEPLOY_DIR / "duplexio-multistream.yaml")
    stage = deploy.stages[0]

    assert deploy.active_stream_window == 2
    assert deploy.duplex_session.max_sessions == 2
    assert stage.max_num_seqs == 2
    assert not stage.enforce_eager
    assert stage.compilation_config == {
        "mode": 3,
        "cudagraph_mode": "NONE",
    }
