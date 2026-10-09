# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A DuplexIO frame: 80 ms of 24 kHz audio, packed as six consecutive cells.

The four text cells carry the frame's system, user, agent and tool-call tokens;
the two audio cells carry what the user and the agent said during the frame.
"""

SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920  # Samples per frame.

TEXT_STREAM_NAMES = ("system", "user", "agent", "tool_call")
NUM_TEXT_CELLS = len(TEXT_STREAM_NAMES)
NUM_CELLS = NUM_TEXT_CELLS + 2
SYSTEM_CELL, USER_CELL, AGENT_CELL, TOOL_CALL_CELL, USER_AUDIO_CELL, AGENT_AUDIO_CELL = range(NUM_CELLS)
