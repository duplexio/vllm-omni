# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-stage native DuplexIO pipeline topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

DUPLEXIO_PIPELINE = PipelineConfig(
    model_type="duplexio",
    model_arch="DuplexIOForConditionalGeneration",
    hf_architectures=("DuplexIOForConditionalGeneration",),
    duplex_runtime_extension=(
        "vllm_omni.experimental.fullduplex.duplexio.runtime."
        "DuplexIORuntimeExtension"
    ),
    duplex_serving_adapter=(
        "vllm_omni.experimental.fullduplex.duplexio.serving_adapter."
        "DuplexIOServingRuntimeAdapter"
    ),
    duplex_control_enabled=True,
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="duplexio",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="audio",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="audio",
            retains_state_across_chunks=True,
            sampling_constraints={"detokenize": True},
        ),
    ),
)
