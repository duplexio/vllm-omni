import asyncio
import base64
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from vllm_omni.experimental.fullduplex.duplexio.offline import (
    FirstTurnPolicy,
    PreparedConversation,
    RolloutGate,
    ToolParticipant,
    VoicePrompt,
    rollout_conversation,
    rollout_conversations,
)
from vllm_omni.experimental.fullduplex.duplexio.runtime import build_duplexio_data_plane_prompt
from vllm_omni.experimental.fullduplex.engine.contracts import DuplexInputMode


def test_voice_reference_is_frame_aligned_and_bounded(tmp_path):
    path = tmp_path / "reference.wav"
    samples = np.arange(3 * 1920 + 100, dtype=np.float32) / 10000
    sf.write(path, samples, 24000, subtype="FLOAT")
    prompt = VoicePrompt.from_file(path, max_frames=2)
    assert prompt.frames == 2
    np.testing.assert_array_equal(
        np.frombuffer(base64.b64decode(prompt.audio), dtype=np.float32), samples[:3840],
    )


class RecordingEngine:
    def __init__(self):
        self.outputs = asyncio.Queue()
        self.inputs = []
        self.runtime = None
        self.collected = 0

    async def open_duplex_session_async(self, _session_id, **kwargs):
        self.runtime = kwargs["runtime_config"]

    async def append_duplex_input_async(self, _session_id, **kwargs):
        assert len(self.inputs) == self.collected, "A conversation must collect its previous output before submitting"
        assert kwargs["collect_outputs"]
        payload = kwargs["payload"]
        build_duplexio_data_plane_prompt(
            request_id="test", fence=kwargs["fence"], session_config={},
            runtime_config=self.runtime, seq=len(self.inputs), turn_seq=0,
            mode=DuplexInputMode.APPEND_AUDIO_CHUNK,
            payload=payload, final=kwargs["final"],
        )
        self.inputs.append(payload)
        voice = payload.get("duplexio_voice_prompt", False)
        system = payload.get("duplexio_prefill", False) or payload.get("duplexio_system_input", False)
        frames = payload.get("frame_count", 1)
        predict = (
            payload.get("duplexio_prefill_final", False)
            or payload.get("duplexio_system_input_final", False)
            or not (voice or system)
        )
        output = {
            "replay_text_ids": torch.zeros(frames, 4, dtype=torch.long),
            "replay_user_features": torch.zeros(frames, 8),
            "replay_agent_audio": torch.zeros(frames, 3),
            "replay_audio_mask": torch.full((frames,), not (voice or system)),
            "replay_prompt_frames": torch.full((frames,), voice),
            "agent_token_id": torch.tensor([17]),
            "user_token_id": torch.tensor([0]),
            "user_emit": torch.tensor([True]),
            "user_emit_logprob": torch.tensor([-0.2]),
            "user_token_logprob": torch.tensor([-0.3]),
            "user_action_logprob": torch.tensor([-0.5]),
            "user_action_eligible": torch.tensor([True]),
            "policy_version": torch.tensor([0]),
            "tool_call_token_id": torch.tensor([2]),
            "agent_audio_token_ids": torch.ones(3) if predict else torch.empty(0),
            "agent_token_logprob": torch.zeros(1) if predict else torch.empty(0),
            "agent_emit_logprob": torch.zeros(1) if predict else torch.empty(0),
            "tool_token_logprob": torch.zeros(1) if predict else torch.empty(0),
            "tool_emit_logprob": torch.zeros(1) if predict else torch.empty(0),
            "predictor_hiddens": torch.empty(0),
        }
        await self.outputs.put(SimpleNamespace(error=None, multimodal_output=output))
        return {"data_plane_outputs": await self.collect_duplex_data_plane_outputs_async(
            _session_id, timeout=kwargs["timeout"],
        )}

    async def collect_duplex_data_plane_outputs_async(self, _request_id, **_kwargs):
        try:
            output = await asyncio.wait_for(self.outputs.get(), timeout=_kwargs["timeout"])
            self.collected += 1
            return [output]
        except TimeoutError:
            return []

    async def close_duplex_session_async(self, *_args, **_kwargs):
        pass


def test_rollout_pins_voice_before_system_and_records_exact_replay_rows():
    async def run():
        engine = RecordingEngine()
        conversation = PreparedConversation(
            conversation_id="one", system_token_ids=[7, 8],
            user_features=torch.ones(2, 1024), user_token_ids=torch.tensor([10, 11]),
            tools=[], metadata={},
        )
        trace = await rollout_conversation(
            engine, conversation, voice_prompt=VoicePrompt("reference", 3),
            sampling_config={}, seed=1, gate=RolloutGate(),
        )
        assert engine.runtime["duplexio_voice_prompt_audio"] == "reference"
        assert "duplexio_voice" not in engine.runtime
        assert engine.inputs[0]["duplexio_voice_prompt"]
        assert engine.inputs[1]["duplexio_prefill_final"]
        assert all("user_token_id" not in part for part in engine.inputs[2:])
        assert trace["prompt_frames"].tolist() == [True] * 3 + [False] * 4
        assert trace["audio_mask"].tolist() == [False] * 5 + [True] * 2
        assert trace["prediction_rows"].tolist() == [4, 5, 6]

    asyncio.run(run())


def test_tool_result_bursts_cannot_interleave_with_live_frames():
    class CallingEngine(RecordingEngine):
        def __init__(self):
            super().__init__()
            self.called = False

        async def append_duplex_input_async(self, session_id, **kwargs):
            result = await super().append_duplex_input_async(session_id, **kwargs)
            # The real engine yields while an append crosses process boundaries.
            await asyncio.sleep(0)
            return result

        async def collect_duplex_data_plane_outputs_async(self, request_id, **kwargs):
            outputs = await super().collect_duplex_data_plane_outputs_async(request_id, **kwargs)
            if outputs and not self.called and outputs[0].multimodal_output["replay_audio_mask"].any():
                self.called = True
                outputs[0].multimodal_output["tool_call_json"] = (
                    b'{"sequence": 1, "name": "lookup", "arguments": {}}'
                )
            return outputs

    class Simulator:
        async def execute(self, *args, **kwargs):
            return "result"

    async def run():
        engine = CallingEngine()
        conversation = PreparedConversation(
            conversation_id="tool", system_token_ids=[7, 8],
            user_features=torch.ones(40, 1024), user_token_ids=torch.full((40,), 2),
            tools=[], metadata={},
        )
        tokenizer = SimpleNamespace(
            encode=lambda *args, **kwargs: [3] * 257,
            decode=lambda *args: "",
        )
        await rollout_conversation(
            engine, conversation, voice_prompt=VoicePrompt("reference", 3),
            sampling_config={}, seed=1, gate=RolloutGate(),
            tools=ToolParticipant(Simulator(), tokenizer, silence_token_id=2, pad_token_id=0),
        )
        indices = [i for i, part in enumerate(engine.inputs) if part.get("duplexio_system_input")]
        assert len(indices) == 3
        assert indices == list(range(indices[0], indices[0] + 3))
        assert engine.inputs[indices[-1]]["duplexio_system_input_final"]
        assert not engine.inputs[-1].get("duplexio_system_input", False)

    asyncio.run(asyncio.wait_for(run(), timeout=10))


def test_waiting_for_one_conversation_does_not_block_another():
    async def run():
        blocked = asyncio.Event()
        release = asyncio.Event()
        ready_finished = asyncio.Event()
        sessions = {}
        gate = RolloutGate()

        class Engine:
            async def open_duplex_session_async(self, session_id, **kwargs):
                sessions[session_id] = RecordingEngine()
                await sessions[session_id].open_duplex_session_async(session_id, **kwargs)

            async def append_duplex_input_async(self, session_id, **kwargs):
                if "blocked" in session_id and not sessions[session_id].inputs:
                    blocked.set()
                    await release.wait()
                return await sessions[session_id].append_duplex_input_async(session_id, **kwargs)

            async def close_duplex_session_async(self, session_id, **kwargs):
                await sessions[session_id].close_duplex_session_async(session_id, **kwargs)

        async def sink(index, trace):
            if trace["conversation_id"] == "ready":
                ready_finished.set()

        conversations = [
            PreparedConversation(
                conversation_id=name, system_token_ids=[7],
                user_features=torch.ones(2, 1024), user_token_ids=torch.tensor([10, 11]),
                tools=[], metadata={},
            )
            for name in ("blocked", "ready")
        ]
        task = asyncio.create_task(rollout_conversations(
            Engine(), conversations, voice_prompt=VoicePrompt("reference", 3),
            concurrency=2, sampling_config={}, seed=1, gate=gate, sink=sink,
        ))
        try:
            await blocked.wait()
            await ready_finished.wait()
            assert gate.outstanding == 1
        finally:
            release.set()
            await task
        assert gate.outstanding == 0
        assert all(len(engine.inputs) == engine.collected == 4 for engine in sessions.values())

    asyncio.run(asyncio.wait_for(run(), timeout=10))


@pytest.mark.parametrize("speech_frame, expected_frames", [(None, 4), (2, 5)])
def test_first_turn_stopping_uses_the_latest_collected_prediction(speech_frame, expected_frames):
    class SilentEngine(RecordingEngine):
        live_frames = 0

        async def collect_duplex_data_plane_outputs_async(self, request_id, **kwargs):
            outputs = await super().collect_duplex_data_plane_outputs_async(request_id, **kwargs)
            for output in outputs:
                if output.multimodal_output["replay_audio_mask"].any():
                    self.live_frames += 1
                output.multimodal_output["agent_token_id"] = torch.tensor([
                    17 if self.live_frames == speech_frame else 2,
                ])
            return outputs

    async def run():
        engine = SilentEngine()
        conversation = PreparedConversation(
            conversation_id="stop", system_token_ids=[7],
            user_features=torch.ones(20, 1024), user_token_ids=torch.tensor([10] + [2] * 19),
            tools=[], metadata={},
        )
        await rollout_conversation(
            engine, conversation, voice_prompt=VoicePrompt("reference", 3),
            sampling_config={}, seed=1, gate=RolloutGate(),
            first_turn=FirstTurnPolicy(
                silence_features=torch.zeros(1, 1024), silence_token_id=2, pad_token_id=0,
                margin_frames=0, min_response_frames=0, stop_after_silence_frames=2,
            ),
        )
        assert sum(part["format"] == "duplexio_features" for part in engine.inputs) == expected_frames

    asyncio.run(asyncio.wait_for(run(), timeout=10))
