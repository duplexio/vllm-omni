# SPDX-License-Identifier: Apache-2.0
"""The actor publishes versions only after receipt and a drained commit."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from vllm_omni.experimental.fullduplex.duplexio.offline import RolloutGate


@pytest.fixture
def actor(monkeypatch):
    # The wire protocol lives in the optional training repository. The actor
    # only needs its constants/types here; no trainer or NCCL group is started.
    protocol = ModuleType("duplexio.opd_link")
    for name in ("JOIN_GROUP", "JOINED", "PREPARE_UPDATE", "READY", "SHUTDOWN", "UPDATED"):
        setattr(protocol, name, name.lower())
    protocol.ActorLink = object
    monkeypatch.setitem(sys.modules, "duplexio.opd_link", protocol)
    path = Path(__file__).resolve().parents[3] / "examples/offline_inference/duplexio/run_opd_actor.py"
    spec = importlib.util.spec_from_file_location("policy_update_actor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actor_keeps_decoding_until_staged_then_drains_and_commits(actor):
    async def run():
        gate = RolloutGate(version=4)
        messages, calls = [], []
        polls = 0
        collected = asyncio.Event()

        async def decode():
            await gate.wait_and_submit()
            # Simulate a row which finishes only after pause starts draining.
            while gate.open.is_set():
                await asyncio.sleep(0)
            assert gate.version == 4
            gate.collected()
            collected.set()

        decode_task = None

        async def collective_rpc(method, **kwargs):
            nonlocal polls, decode_task
            calls.append(method)
            assert gate.version == 4
            if method == "start_policy_weight_update":
                assert gate.open.is_set()
                return [[2]]
            if method == "policy_weight_update_ready":
                assert gate.open.is_set()
                polls += 1
                if polls == 1:
                    decode_task = asyncio.create_task(decode())
                return [[polls > 1]]
            assert method == "commit_policy_weight_update"
            assert not gate.open.is_set()
            assert gate.outstanding == 0
            assert collected.is_set()
            assert messages == [{"type": actor.READY, "version": 5}]
            return [[2]]

        await actor.update_policy(
            SimpleNamespace(collective_rpc=collective_rpc),
            gate,
            SimpleNamespace(send=messages.append),
            {"version": 5, "weights": []},
            5,
        )
        await decode_task
        assert gate.open.is_set() and gate.version == 5
        assert calls == [
            "start_policy_weight_update",
            "policy_weight_update_ready",
            "policy_weight_update_ready",
            "commit_policy_weight_update",
        ]
        assert messages[-1] == {"type": actor.UPDATED, "version": 5}

    asyncio.run(run())


@pytest.mark.parametrize("fail_method", ["policy_weight_update_ready", "commit_policy_weight_update"])
def test_failed_update_does_not_publish_or_acknowledge_version(actor, fail_method):
    async def run():
        gate = RolloutGate(version=4)
        messages = []

        async def collective_rpc(method, **kwargs):
            if method == fail_method:
                return [[{"error": "update failed"}]]
            return [[True]]

        with pytest.raises(RuntimeError, match="update failed"):
            await actor.update_policy(
                SimpleNamespace(collective_rpc=collective_rpc),
                gate,
                SimpleNamespace(send=messages.append),
                {"version": 5, "weights": []},
                5,
            )
        assert gate.version == 4
        assert messages == [{"type": actor.READY, "version": 5}]
        if fail_method == "commit_policy_weight_update":
            assert not gate.open.is_set()

    asyncio.run(run())


def test_receive_timeout_does_not_pause_or_acknowledge(actor):
    async def run():
        gate = RolloutGate(version=4)
        messages = []

        async def collective_rpc(method, **kwargs):
            return [[False]]

        with pytest.raises(TimeoutError):
            await actor.update_policy(
                SimpleNamespace(collective_rpc=collective_rpc),
                gate,
                SimpleNamespace(send=messages.append),
                {"version": 5, "weights": []},
                0.01,
            )
        assert gate.version == 4 and gate.open.is_set()
        assert messages == [{"type": actor.READY, "version": 5}]

    asyncio.run(run())


@pytest.mark.parametrize("result", [[], [[]], [[{"todo": True}]], [[{"error": "broken"}]]])
def test_rpc_rejects_empty_unsupported_and_nested_errors(actor, result):
    async def collective_rpc(*args, **kwargs):
        return result

    with pytest.raises(RuntimeError):
        asyncio.run(actor.rpc(SimpleNamespace(collective_rpc=collective_rpc), "test", (), 5))


def test_gate_rechecks_woken_waiters_after_it_closes_again():
    async def run():
        gate = RolloutGate(version=1)
        await gate.pause()
        waiter = asyncio.create_task(gate.wait_and_submit())
        await asyncio.sleep(0)
        gate.resume(2)
        # Close again before the woken waiter can run.
        await gate.pause()
        await asyncio.sleep(0)
        assert not waiter.done()
        assert gate.outstanding == 0
        gate.resume(3)
        await waiter
        assert gate.outstanding == 1
        gate.collected()
        await gate.pause()

    asyncio.run(run())
