# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for DuplexIO's single-stage deployments."""

from pathlib import Path

import pytest

from vllm_omni.config.stage_config import load_deploy_config
from vllm_omni.model_executor.models.duplexio.frame_layout import FRAME_POSITIONS
from vllm_omni.model_executor.models.duplexio.pipeline import DUPLEXIO_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

DEPLOY_DIR = Path(__file__).resolve().parents[4] / "vllm_omni" / "deploy"


def test_default_deploy_pins_the_kv_page_size() -> None:
    assert DUPLEXIO_PIPELINE.default_deploy_config_name == "duplexio.yaml"
    deploy = load_deploy_config(DEPLOY_DIR / "duplexio.yaml")
    stage = deploy.stages[0]

    # Persistent KV for 73728 tokens and voice-prompt rows; audio lives in fixed rings.
    assert stage.max_model_len == 73728
    # Budgets are whole five-position frames, so no frame is split across steps.
    assert stage.max_num_batched_tokens == 1228 * FRAME_POSITIONS
    assert deploy.enable_chunked_prefill
    assert not stage.enforce_eager
    assert stage.compilation_config == {
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [FRAME_POSITIONS],
        "cudagraph_copy_inputs": True,
    }
    # The hybrid cache would otherwise derive the page from the mamba state size.
    assert stage.engine_extras["block_size"] == 1024


def test_multistream_deploy_batches_two_sessions() -> None:
    deploy = load_deploy_config(DEPLOY_DIR / "duplexio-multistream.yaml")
    stage = deploy.stages[0]

    assert deploy.active_stream_window == 2
    assert deploy.duplex_session.max_sessions == 2
    assert stage.max_num_seqs == 2
    assert stage.max_num_batched_tokens == 1228 * FRAME_POSITIONS
    assert deploy.enable_chunked_prefill
    assert not stage.enforce_eager
    assert stage.compilation_config == {"cudagraph_mode": "NONE"}
