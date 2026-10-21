# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host bookkeeping of DuplexIO's flat sequence: appends as frames, frames as positions.

An append is a list of frames (``frame_layout.FRAME_POSITIONS`` positions
each), so the scheduler, whose budgets are whole frames, never splits one.
Every append ends with the frame deciding the next row, and returns that
decision:

- the session's opening append, before any user audio: text frames of the
  voice prompt, the system tokens and any tool result, padded with no-ops,
  then the BOS frame deciding live row 0;
- each user-audio append: its live frame, then, with a tool result, the
  result's text frames and a BOS frame.

``Cursor`` holds what a request's positions so far leave behind: its row (the
RoPE position), the audio clock, and how much of each KV region
(``kv_reclamation.DuplexIOKVLayout``) it has written. ``FramePlanner`` lays a
frame out from it, as ``duplexio.decode.Decoder`` appends it, and advances it.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field, replace
from enum import IntEnum

from vllm_omni.model_executor.models.duplexio.frame_layout import (
    AGENT,
    AGENT_AUDIO_STREAM,
    BOS_AGENT_AUDIO_SLOT,
    BOS_USER_AUDIO_SLOT,
    FRAME_POSITIONS,
    QUERY_SLOT_OFFSET,
    SYSTEM,
    TOOL_CALL,
    USER,
    USER_AUDIO_STREAM,
)


class Region(IntEnum):
    """The KV region a key is written to."""

    RING = 0  # Live audio, modulo the ring.
    BOS = 1  # BOS frames' audio, modulo the ring.
    PERSISTENT = 2  # Tokens and the pinned voice prompt.


class Embed(IntEnum):
    """What a position's input embedding is."""

    NOOP = 0  # Zero.
    TOKEN = 1  # The token's embedding plus its stream's channel embedding.
    CELL = 2  # A learned cell, ``flat_cell_emb[slot]``.
    USER_AUDIO = 3  # The user adapter over the frame's encoder features.
    AGENT_AUDIO = 4  # The agent adapter over an agent latent.


class FrameKind(IntEnum):
    TEXT = 0
    BOS = 1
    LIVE = 2


@dataclass(frozen=True)
class Frame:
    """One frame of an append; a text frame lists ``(stream, token id)`` entries, stream -1 for a prompt row."""

    kind: FrameKind
    entries: tuple[tuple[int, int], ...] = ()


PROMPT = -1  # A text-frame entry that is a pinned voice-prompt row; its id is the prompt row.


def append_frames(
    prompt_rows: int, system_token_ids: tuple[int, ...], tool_token_ids: tuple[int, ...], *, opening: bool
) -> list[Frame]:
    """The frames of one append: the session's opening one, or a user-audio one."""
    text = [(SYSTEM, token) for token in tool_token_ids]
    frames = [] if opening else [Frame(FrameKind.LIVE)]
    if opening:
        text = [(PROMPT, row) for row in range(prompt_rows)] + [(SYSTEM, token) for token in system_token_ids] + text
    if text or opening:
        frames += [*text_frames(text), Frame(FrameKind.BOS)]
    return frames


def text_frames(entries: list[tuple[int, int]]) -> list[Frame]:
    return [
        Frame(FrameKind.TEXT, tuple(entries[start : start + FRAME_POSITIONS]))
        for start in range(0, len(entries), FRAME_POSITIONS)
    ]


def append_frame_count(prompt_rows: int, system_tokens: int, tool_tokens: int, *, opening: bool) -> int:
    """``len(append_frames(...))`` from the counts alone, for the scheduler side."""
    text = tool_tokens + (prompt_rows + system_tokens if opening else 0)
    return (0 if opening else 1) + (-(-text // FRAME_POSITIONS) + 1 if text or opening else 0)


@dataclass
class Cursor:
    """What a request's positions so far leave behind."""

    row: int = 0  # The next row: its RoPE position.
    live_rows: int = 0  # The audio clock: live rows so far.
    ring_keys: int = 0  # Keys written to the audio ring.
    bos_clocks: list[int] = field(default_factory=list)  # The audio clock of each BOS frame.
    bos_keys: int = 0  # Keys written to the BOS ring, two per BOS frame.
    persistent_keys: int = 0
    gdn_fresh: bool = True  # No position has written the request's GDN slot yet.

    def fork(self) -> Cursor:
        return replace(self, bos_clocks=list(self.bos_clocks))


@dataclass
class Position:
    """One planned position: its input and its addressing (``qwen_backbone`` step columns)."""

    embed: Embed
    value: int = 0  # Token id, cell slot, or the row of its audio input.
    stream: int = 0  # A token's stream, for its channel embedding.
    rope: int = 0
    write: int = -1  # Slot of its key within ``region``, -1 for none.
    region: Region = Region.PERSISTENT
    ranges: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0)
    own_key: bool = False  # Attends to its own key, which the cache does not hold.
    gdn_stream: int = USER
    gdn_real: bool = False


class FramePlanner:
    """Lay frames out against one request's cursor, which it advances.

    Audio keys stay visible for ``audio_window_frames`` frames of audio clock;
    each ring holds ``2 * (audio_window_frames + 1)`` keys.
    """

    def __init__(self, cursor: Cursor, *, audio_window_frames: int, max_persistent_keys: int) -> None:
        self.cursor = cursor
        self.audio_window_frames = audio_window_frames
        self.ring_slots = 2 * (audio_window_frames + 1)
        self.max_persistent_keys = max_persistent_keys

    def ranges(self, clock: int) -> tuple[int, int, int, int, int, int]:
        """What a position with audio clock ``clock`` sees of what is written: audio of clocks ``clock - window`` on.

        Text and voice-prompt keys are persistent and always seen.
        """
        cursor = self.cursor
        oldest = clock - self.audio_window_frames
        audio_start = min(2 * (max(1, oldest) - 1), cursor.ring_keys)
        bos_start = min(2 * bisect.bisect_left(cursor.bos_clocks, oldest), cursor.bos_keys)
        return (audio_start, cursor.ring_keys, bos_start, cursor.bos_keys, 0, cursor.persistent_keys)

    def real(self, position: Position, clock: int, region: Region) -> Position:
        """Write ``position``'s key to ``region``, then let it see everything up to and including it."""
        cursor = self.cursor
        position.region = region
        if region == Region.RING:
            position.write = cursor.ring_keys % self.ring_slots
            cursor.ring_keys += 1
        elif region == Region.BOS:
            position.write = cursor.bos_keys % self.ring_slots
            cursor.bos_keys += 1
        else:
            if cursor.persistent_keys >= self.max_persistent_keys:
                raise RuntimeError("DuplexIO request exceeded its persistent KV capacity")
            position.write = cursor.persistent_keys
            cursor.persistent_keys += 1
        position.ranges = self.ranges(clock)
        position.gdn_real = True
        return position

    def invisible(self, position: Position, clock: int) -> Position:
        """A query or no-op: it sees what is written and its own key, and writes nothing."""
        position.ranges = self.ranges(clock)
        position.own_key = True
        return position

    def plan(self, frame: Frame, live_tokens: dict[int, int] | None = None) -> list[Position]:
        """Lay ``frame`` out; a live frame carries ``live_tokens``, its present tokens by stream."""
        cursor = self.cursor
        if frame.kind == FrameKind.TEXT:
            positions = []
            for stream, value in frame.entries:
                clock = cursor.live_rows
                if stream == PROMPT:
                    position = Position(Embed.AGENT_AUDIO, value, rope=cursor.row, gdn_stream=AGENT_AUDIO_STREAM)
                else:
                    position = Position(Embed.TOKEN, value, stream, rope=cursor.row, gdn_stream=stream)
                positions.append(self.real(position, clock, Region.PERSISTENT))
                cursor.row += 1
            while len(positions) < FRAME_POSITIONS:
                positions.append(self.invisible(Position(Embed.NOOP, rope=cursor.row), cursor.live_rows))
            return positions
        clock = cursor.live_rows + 1
        row = cursor.row
        if frame.kind == FrameKind.BOS:
            cursor.bos_clocks.append(clock)
            positions = [
                self.invisible(Position(Embed.NOOP, rope=row), clock),
                self.invisible(Position(Embed.CELL, QUERY_SLOT_OFFSET + AGENT, rope=row, gdn_stream=AGENT), clock),
                self.invisible(
                    Position(Embed.CELL, QUERY_SLOT_OFFSET + TOOL_CALL, rope=row, gdn_stream=TOOL_CALL), clock
                ),
            ]
            positions.append(
                self.real(
                    Position(Embed.CELL, BOS_USER_AUDIO_SLOT, rope=row, gdn_stream=USER_AUDIO_STREAM), clock, Region.BOS
                )
            )
            positions.append(
                self.real(
                    Position(Embed.CELL, BOS_AGENT_AUDIO_SLOT, rope=row, gdn_stream=AGENT_AUDIO_STREAM),
                    clock,
                    Region.BOS,
                )
            )
            return positions
        assert live_tokens is not None
        positions = []
        for stream in (USER, AGENT, TOOL_CALL):
            token = live_tokens.get(stream)
            if token is not None:
                positions.append(
                    self.real(
                        Position(Embed.TOKEN, token, stream, rope=row, gdn_stream=stream), clock, Region.PERSISTENT
                    )
                )
            elif stream == USER:
                positions.append(self.invisible(Position(Embed.NOOP, rope=row), clock))
            else:
                positions.append(
                    self.invisible(Position(Embed.CELL, QUERY_SLOT_OFFSET + stream, rope=row, gdn_stream=stream), clock)
                )
        positions.append(
            self.real(
                Position(Embed.USER_AUDIO, rope=row, gdn_stream=USER_AUDIO_STREAM),
                clock,
                Region.RING,
            )
        )
        positions.append(
            self.real(
                Position(Embed.AGENT_AUDIO, rope=row, gdn_stream=AGENT_AUDIO_STREAM),
                clock,
                Region.RING,
            )
        )
        cursor.row += 1
        cursor.live_rows += 1
        return positions


__all__ = [
    "PROMPT",
    "Cursor",
    "Region",
    "Embed",
    "Frame",
    "FrameKind",
    "FramePlanner",
    "Position",
    "append_frame_count",
    "append_frames",
]
