# SPDX-License-Identifier: Apache-2.0
"""Flat positions see exactly what duplexio's decoder lets them see, and step the streams it steps."""

import pytest

from vllm_omni.model_executor.models.duplexio.flat_plan import (
    PROMPT,
    Cursor,
    Embed,
    Frame,
    FrameKind,
    FramePlanner,
    Region,
    append_frame_count,
    append_frames,
)
from vllm_omni.model_executor.models.duplexio.frame_layout import (
    AGENT,
    AGENT_AUDIO_STREAM,
    FRAME_POSITIONS,
    QUERY_SLOT_OFFSET,
    SYSTEM,
    TOOL_CALL,
    USER,
    USER_AUDIO_STREAM,
)


def test_append_frames_are_whole_frames_ending_in_their_decision() -> None:
    first = append_frames(3, (11, 12, 13), (), prefix=True)
    assert [frame.kind for frame in first] == [FrameKind.TEXT, FrameKind.TEXT, FrameKind.BOS, FrameKind.LIVE]
    assert first[0].entries == ((PROMPT, 0), (PROMPT, 1), (PROMPT, 2), (SYSTEM, 11), (SYSTEM, 12))
    assert first[1].entries == ((SYSTEM, 13),)
    tool = append_frames(0, (), tuple(range(7)), prefix=False)
    assert [frame.kind for frame in tool] == [FrameKind.LIVE, FrameKind.TEXT, FrameKind.TEXT, FrameKind.BOS]
    assert [frame.kind for frame in append_frames(0, (), (), prefix=False)] == [FrameKind.LIVE]
    for prompt, system, tools, prefix in ((3, 3, 0, True), (125, 40, 0, True), (0, 0, 7, False), (0, 0, 0, False)):
        frames = append_frames(prompt, tuple(range(system)), tuple(range(tools)), prefix=prefix)
        assert append_frame_count(prompt, system, tools, prefix=prefix) == len(frames)


def reference_visibility(entries, window):
    """``duplexio.decode.attention_decode``: causal over real keys, ``key_in_window``, a query sees itself."""
    visible = []
    for query, (q_clock, q_real, _, _) in enumerate(entries):
        keys = set()
        for key, (k_clock, k_real, k_text, k_pinned) in enumerate(entries[: query + 1]):
            if not k_real and key != query:
                continue
            if k_text or k_pinned or q_clock - k_clock <= window:
                keys.add(key)
        visible.append(keys)
    return visible


@pytest.mark.parametrize("window", [2, 3, 50])
def test_planned_ranges_match_the_reference_window(window: int) -> None:
    cursor = Cursor()
    planner = FramePlanner(cursor, audio_window_frames=window, max_persistent_keys=1000)
    script: list[tuple[Frame, dict[int, int] | None]] = [
        (Frame(FrameKind.TEXT, ((PROMPT, 0), (PROMPT, 1), (SYSTEM, 5))), None),
        (Frame(FrameKind.BOS), None),
        (Frame(FrameKind.LIVE), {}),
        (Frame(FrameKind.LIVE), {USER: 7, AGENT: 8}),
        (Frame(FrameKind.LIVE), {AGENT: 9}),
        (Frame(FrameKind.LIVE), {TOOL_CALL: 3}),
        (Frame(FrameKind.TEXT, ((SYSTEM, 21), (SYSTEM, 22))), None),
        (Frame(FrameKind.BOS), None),
        *[(Frame(FrameKind.LIVE), {USER: 30 + index} if index % 2 else {}) for index in range(6)],
        (Frame(FrameKind.TEXT, ((SYSTEM, 40),)), None),
        (Frame(FrameKind.BOS), None),
        (Frame(FrameKind.LIVE), {AGENT: 41}),
    ]
    positions, entries, keys = [], [], {}
    rows_seen = []
    for frame, tokens in script:
        live_rows = cursor.live_rows
        planned = planner.plan(frame, tokens)
        assert len(planned) == FRAME_POSITIONS
        for position in planned:
            clock = live_rows + (frame.kind != FrameKind.TEXT)
            text = position.embed == Embed.TOKEN
            pinned = frame.kind == FrameKind.TEXT and position.embed == Embed.AGENT_AUDIO
            entries.append((clock, position.gdn_real, text, pinned))
            if position.gdn_real:
                keys[(position.region, position.write, len(keys))] = len(positions)
            positions.append(position)
            rows_seen.append(position.rope)
    expected = reference_visibility(entries, window)
    # Each ring slot holds the newest key written to it.
    for index, position in enumerate(positions):
        audio_start, audio_end, bos_start, bos_end, persistent_start, persistent_end = position.ranges
        seen = set()
        for region, start, end in (
            (Region.RING, audio_start, audio_end),
            (Region.BOS, bos_start, bos_end),
            (Region.PERSISTENT, persistent_start, persistent_end),
        ):
            for logical in range(start, end):
                slot = logical % planner.ring_slots if region != Region.PERSISTENT else logical
                writers = [
                    writer
                    for (key_region, key_slot, _), writer in keys.items()
                    if key_region == region and key_slot == slot and writer <= index
                ]
                seen.add(max(writers))
        if position.own_key:
            seen.add(index)
        assert seen == expected[index], (index, sorted(seen ^ expected[index]))
    # RoPE positions count rows; a BOS frame sits on the row it decides.
    assert rows_seen[:15] == [0, 1, 2, 3, 3] + [3] * 5 + [3] * 5


def test_frames_step_their_streams() -> None:
    planner = FramePlanner(Cursor(), audio_window_frames=10, max_persistent_keys=100)
    bos = planner.plan(Frame(FrameKind.BOS))
    assert [(p.embed, p.value, p.gdn_stream, p.gdn_real) for p in bos[1:]] == [
        (Embed.CELL, QUERY_SLOT_OFFSET + AGENT, AGENT, False),
        (Embed.CELL, QUERY_SLOT_OFFSET + TOOL_CALL, TOOL_CALL, False),
        (Embed.CELL, 6, USER_AUDIO_STREAM, True),
        (Embed.CELL, 7, AGENT_AUDIO_STREAM, True),
    ]
    live = planner.plan(Frame(FrameKind.LIVE), {USER: 5, TOOL_CALL: 6})
    assert [(p.embed, p.gdn_stream, p.gdn_real) for p in live] == [
        (Embed.TOKEN, USER, True),
        (Embed.CELL, AGENT, False),
        (Embed.TOKEN, TOOL_CALL, True),
        (Embed.USER_AUDIO, USER_AUDIO_STREAM, True),
        (Embed.AGENT_AUDIO, AGENT_AUDIO_STREAM, True),
    ]
    assert live[0].stream == USER and live[2].stream == TOOL_CALL


def test_persistent_capacity_is_enforced() -> None:
    planner = FramePlanner(Cursor(), audio_window_frames=10, max_persistent_keys=2)
    planner.plan(Frame(FrameKind.TEXT, ((SYSTEM, 1), (SYSTEM, 2))))
    with pytest.raises(RuntimeError, match="persistent"):
        planner.plan(Frame(FrameKind.TEXT, ((SYSTEM, 3),)))
