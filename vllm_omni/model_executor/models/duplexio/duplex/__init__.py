# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexIO full-duplex integration through ``PipelineConfig.duplex_plugin``, and its offline stream."""

from .plugin import DuplexIODuplexPlugin, GivenFrame
from .stream import DuplexIOStream

__all__ = ["DuplexIODuplexPlugin", "DuplexIOStream", "GivenFrame"]
