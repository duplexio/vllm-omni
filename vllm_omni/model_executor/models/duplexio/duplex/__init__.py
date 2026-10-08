# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexIO full-duplex integration through ``PipelineConfig.duplex_plugin``."""

from .plugin import DuplexIODuplexPlugin, GivenFrame, append_fields, stage_sampling_params

__all__ = ["DuplexIODuplexPlugin", "GivenFrame", "append_fields", "stage_sampling_params"]
