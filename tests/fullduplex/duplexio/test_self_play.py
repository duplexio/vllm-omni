# SPDX-License-Identifier: Apache-2.0
"""Lock-step self-play keeps the two DuplexIO sessions causally aligned."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.experimental.fullduplex.duplexio.self_play import (
    PreparedRole,
    PreparedScenarioPair,
    rollout_scenario_pair,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def prepared_pair() -> PreparedScenarioPair:
    role = PreparedRole(system_token_ids=[11], voice="voice")
    return PreparedScenarioPair(
        conversation_id="case-1",
        agent=role,
        user=PreparedRole(system_token_ids=[22], voice="voice"),
    )


class FakeEngine:
    def __init__(
        self,
        *,
        initial_audio: float,
        live_audio: tuple[float, ...],
        tool_call: bool = False,
        fail_open: bool = False,
    ) -> None:
        self.initial_audio = initial_audio
        self.live_audio = live_audio
        self.tool_call = tool_call
        self.fail_open = fail_open
        self.opened: list[str] = []
        self.closed: list[str] = []
        self.payloads: list[dict[str, object]] = []
        self.finals: list[bool] = []

    async def open_duplex_session_async(self, session_id: str, **kwargs: object) -> dict[str, object]:
        if self.fail_open:
            raise RuntimeError("open failed")
        self.opened.append(session_id)
        return {"ok": True}

    async def append_duplex_input_async(self, session_id: str, **kwargs: object) -> dict[str, object]:
        payload = kwargs["payload"]
        assert isinstance(payload, dict)
        self.payloads.append(payload)
        final = kwargs["final"]
        assert isinstance(final, bool)
        self.finals.append(final)
        is_prefill = payload.get("duplexio_prefill", False)
        if is_prefill:
            frame_count = payload["frame_count"]
            assert isinstance(frame_count, int)
            prediction = bool(payload["duplexio_prefill_final"])
            audio_value = self.initial_audio
            audio = torch.full((1_920,), audio_value) if prediction else torch.empty(0)
        else:
            frame = len([item for item in self.payloads if not item.get("duplexio_prefill", False)]) - 1
            prediction = True
            audio = torch.full((1_920,), self.live_audio[frame])
            frame_count = 1
        output = {
            "replay_text_ids": torch.zeros(frame_count, 4, dtype=torch.long),
            "replay_user_features": torch.zeros(frame_count, 8),
            "replay_agent_audio": torch.zeros(frame_count, 3),
            "replay_audio_mask": torch.zeros(frame_count, dtype=torch.bool),
            "agent_audio_token_ids": torch.ones(3) if prediction else torch.empty(0),
            "agent_token_id": torch.tensor([17]),
            "tool_call_token_id": torch.tensor([2]),
            "predictor_hiddens": torch.empty(0),
            "audio": audio,
            "model_listen": torch.tensor([False]),
            "tool_call_complete": torch.tensor([self.tool_call and not is_prefill]),
        }
        return {"data_plane_outputs": [SimpleNamespace(error=None, multimodal_output=output)]}

    async def close_duplex_session_async(self, session_id: str, **kwargs: object) -> dict[str, object]:
        self.closed.append(session_id)
        return {"ok": True}


def decode_audio(payload: dict[str, object]) -> float:
    encoded = payload["audio"]
    assert isinstance(encoded, str)
    values = torch.frombuffer(base64.b64decode(encoded), dtype=torch.float32)
    return float(values[0])


@pytest.mark.asyncio
async def test_pair_exchanges_previous_decoded_frame_and_records_both_roles() -> None:
    agent = FakeEngine(initial_audio=1.0, live_audio=(3.0, 5.0))
    user = FakeEngine(initial_audio=2.0, live_audio=(4.0, 6.0))

    result = await rollout_scenario_pair(
        agent,
        user,
        prepared_pair(),
        policy_version="policy-7",
        agent_sampling_config={"duplexio_text_sampling": {"mode": "argmax"}},
        seed=9,
        max_frames=2,
    )

    # The first live append receives the other model's final-prefill frame;
    # subsequent appends receive the previous live output.
    assert [decode_audio(payload) for payload in agent.payloads[1:]] == [2.0, 4.0]
    assert [decode_audio(payload) for payload in user.payloads[1:]] == [1.0, 3.0]
    assert agent.finals[-1] is True
    assert user.finals[-1] is True
    assert result["agent_trace"]["policy_version"] == "policy-7"
    assert result["user_trace"]["policy_version"] == "frozen-user"
    assert result["agent_trace"]["text_ids"].shape == (3, 4)
    assert result["agent_trace"]["prediction_rows"].tolist() == [0, 1, 2]
    assert [event["frame"] for event in result["events"]] == [0, 1]
    assert agent.closed and user.closed


@pytest.mark.asyncio
async def test_pair_rejects_tool_call_without_a_tool_backend() -> None:
    agent = FakeEngine(initial_audio=1.0, live_audio=(3.0,), tool_call=True)
    user = FakeEngine(initial_audio=2.0, live_audio=(4.0,))

    with pytest.raises(RuntimeError, match="third-party tool-result simulator"):
        await rollout_scenario_pair(
            agent,
            user,
            prepared_pair(),
            policy_version="policy-7",
            agent_sampling_config={},
            seed=9,
            max_frames=1,
        )
    assert agent.closed and user.closed


@pytest.mark.asyncio
async def test_pair_closes_the_role_that_opened_when_the_other_open_fails() -> None:
    agent = FakeEngine(initial_audio=1.0, live_audio=(3.0,))
    user = FakeEngine(initial_audio=2.0, live_audio=(4.0,), fail_open=True)

    with pytest.raises(RuntimeError, match="open failed"):
        await rollout_scenario_pair(
            agent,
            user,
            prepared_pair(),
            policy_version="policy-7",
            agent_sampling_config={},
            seed=9,
            max_frames=1,
        )
    assert agent.closed


def test_pair_rejects_empty_role_prompt() -> None:
    with pytest.raises(ValueError, match="at least 1 item"):
        PreparedRole(system_token_ids=[], voice="voice")
