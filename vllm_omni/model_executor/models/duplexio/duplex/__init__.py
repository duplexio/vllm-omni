# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DuplexIO full-duplex integration through ``PipelineConfig.duplex_plugin``."""

from .plugin import DuplexIODuplexPlugin, append_fields, stage_sampling_params

__all__ = ["DuplexIODuplexPlugin", "append_fields", "stage_sampling_params"]
