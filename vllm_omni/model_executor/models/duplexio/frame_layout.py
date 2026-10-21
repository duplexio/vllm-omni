# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""DuplexIO's flat sequence: rows of 80 ms frames laid out as single positions.

A row (time frame) holds up to four text tokens (system, user, agent, tool
call) and two audio cells. The flat layout keeps only what a row carries, in
row order:

- a voice-prompt row is ``[A]``, pinned so the audio window never evicts it;
- a text row (system prefix, tool result) is its token;
- a live row is ``[its tokens, U, A]``: the tokens sampled at the frame
  before, then the user's and the agent's audio of the frame.

Frame ``r`` decides row ``r + 1``: the user emit and word at ``U``; the agent
and tool emit, the next agent audio frame and VAP at ``A``; the agent and tool
token at the stream's cell, which is its token on row ``r`` or, where the row
has none, a query: a learned embedding visible only to itself that never
enters the KV cache and never commits its GatedDeltaNet step. A live row after
a text run (or at the start) is decided by a BOS frame ``[queries, U_bos,
A_bos]`` of learned cells on the row it decides.

Serving appends whole frames of ``FRAME_POSITIONS`` positions, so every frame,
live or BOS, decides from the same places: ``[user token, agent cell, tool
cell, U, A]``. A row without a user token, and the BOS frame's first position,
hold a no-op: an invisible position that changes nothing. Text runs are padded
with no-ops to whole frames, which keeps every frame inside one scheduler chunk.

Slots and learned-cell indices follow ``duplexio.flat_layout``.
"""

SAMPLE_RATE = 24_000
FRAME_SIZE = 1_920  # Samples per frame.

TEXT_STREAM_NAMES = ("system", "user", "agent", "tool_call")
NUM_TEXT_STREAMS = len(TEXT_STREAM_NAMES)
SYSTEM, USER, AGENT, TOOL_CALL = range(NUM_TEXT_STREAMS)

# Slots: a token's slot is its stream; then the audio cells, the BOS frame's
# audio cells and one query per stream. ``flat_cell_emb`` is indexed by slot.
USER_AUDIO_SLOT = NUM_TEXT_STREAMS
AGENT_AUDIO_SLOT = NUM_TEXT_STREAMS + 1
BOS_USER_AUDIO_SLOT = NUM_TEXT_STREAMS + 2
BOS_AGENT_AUDIO_SLOT = NUM_TEXT_STREAMS + 3
QUERY_SLOT_OFFSET = NUM_TEXT_STREAMS + 4
SLOT_COUNT = QUERY_SLOT_OFFSET + NUM_TEXT_STREAMS

# GatedDeltaNet streams: the four text streams, then user and agent audio. BOS
# audio cells step the audio streams; a query steps (without committing) its own.
USER_AUDIO_STREAM, AGENT_AUDIO_STREAM = NUM_TEXT_STREAMS, NUM_TEXT_STREAMS + 1
NUM_GDN_STREAMS = NUM_TEXT_STREAMS + 2

# One frame: [user token or no-op, agent token or query, tool token or query, U, A].
FRAME_POSITIONS = 5
USER_TOKEN_INDEX, AGENT_CELL_INDEX, TOOL_CELL_INDEX, USER_AUDIO_INDEX, AGENT_AUDIO_INDEX = range(FRAME_POSITIONS)
