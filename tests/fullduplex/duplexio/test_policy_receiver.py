# SPDX-License-Identifier: Apache-2.0
"""Policy receipt must never mutate weights used by concurrent decode."""

from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.experimental.fullduplex.duplexio.policy_receiver import PolicyWeightReceiver

PLAN = [["weight", "float32", [2, 2]], ["bias", "float32", [2]]]


class Model(torch.nn.Linear):
    def __init__(self, device="cpu"):
        super().__init__(2, 2, device=device)
        with torch.no_grad():
            self.weight.fill_(1)
            self.bias.fill_(1)

    def load_weights(self, weights):
        loaded = set()
        for name, tensor in weights:
            getattr(self, name).copy_(tensor)
            loaded.add(name)
        return loaded


def make_receiver(device, stream, broadcast):
    receiver = PolicyWeightReceiver()
    receiver.device = torch.device(device)
    receiver.model_runner = SimpleNamespace(model=Model(device))
    receiver.policy_group = SimpleNamespace(broadcast=broadcast)
    receiver._policy_stream = stream
    receiver._policy_pending = None
    receiver._policy_buffer = []
    receiver._policy_plan = None
    receiver._policy_version = None
    receiver._policy_commit_failed = False
    return receiver


@pytest.fixture
def receiver(monkeypatch):
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda _: nullcontext())
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    return make_receiver("cpu", Mock(), lambda tensor: tensor.fill_(3))


def finish_receiving(receiver):
    receiver._policy_pending[1].result(timeout=5)


def test_receipt_overlaps_decode_and_commit_preserves_parameter_addresses(receiver):
    started, release = Event(), Event()

    def broadcast(tensor):
        started.set()
        assert release.wait(5)
        tensor.fill_(3)

    receiver.policy_group.broadcast = broadcast
    model = receiver.model_runner.model
    pointers = [p.data_ptr() for p in model.parameters()]
    try:
        assert receiver.start_policy_weight_update(1, PLAN) == 2
        assert started.wait(5)
        assert not receiver.policy_weight_update_ready(1)
        torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 3.0))
        with pytest.raises(RuntimeError, match="still being received"):
            receiver.commit_policy_weight_update(1)
        with pytest.raises(RuntimeError, match="already pending"):
            receiver.start_policy_weight_update(2, PLAN)
    finally:
        release.set()
        finish_receiving(receiver)
    assert receiver.policy_weight_update_ready(1)
    # Even completely received weights remain invisible until commit.
    torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 3.0))
    assert receiver.commit_policy_weight_update(1) == 2
    torch.testing.assert_close(model(torch.ones(2)), torch.full((2,), 9.0))
    assert [p.data_ptr() for p in model.parameters()] == pointers
    assert receiver._policy_version == 1
    assert torch.accelerator.synchronize.call_count == 2


def test_reuses_staging_storage_and_rejects_stale_or_wrong_versions(receiver):
    receiver.start_policy_weight_update(7, PLAN)
    finish_receiving(receiver)
    pointers = [tensor.data_ptr() for _, tensor in receiver._policy_buffer]
    with pytest.raises(ValueError, match="No pending"):
        receiver.commit_policy_weight_update(8)
    receiver.commit_policy_weight_update(7)
    with pytest.raises(ValueError, match="must exceed"):
        receiver.start_policy_weight_update(7, PLAN)
    receiver.start_policy_weight_update(8, PLAN)
    finish_receiving(receiver)
    assert [tensor.data_ptr() for _, tensor in receiver._policy_buffer] == pointers
    receiver.commit_policy_weight_update(8)
    with pytest.raises(ValueError, match="No pending"):
        receiver.commit_policy_weight_update(8)


@pytest.mark.parametrize("plan", [[], PLAN + PLAN, [["weight", "bad_dtype", [2]]], [["weight", "float32", [-1]]]])
def test_rejects_bad_metadata_before_starting_transfer(receiver, plan):
    receiver.policy_group.broadcast = Mock()
    with pytest.raises(ValueError):
        receiver.start_policy_weight_update(1, plan)
    receiver.policy_group.broadcast.assert_not_called()
    assert receiver._policy_pending is None


def test_transfer_failure_never_changes_live_weights(receiver):
    receiver.policy_group.broadcast = Mock(side_effect=RuntimeError("NCCL failure"))
    receiver.start_policy_weight_update(1, PLAN)
    with pytest.raises(RuntimeError, match="NCCL failure"):
        finish_receiving(receiver)
    with pytest.raises(RuntimeError, match="NCCL failure"):
        receiver.policy_weight_update_ready(1)
    with pytest.raises(RuntimeError, match="NCCL failure"):
        receiver.commit_policy_weight_update(1)
    assert receiver._policy_version is None
    assert torch.accelerator.synchronize.call_count == 0
    torch.testing.assert_close(receiver.model_runner.model.weight, torch.ones(2, 2))


def test_loader_failure_is_fatal_and_never_publishes_version(receiver):
    receiver.start_policy_weight_update(1, PLAN)
    finish_receiving(receiver)
    original_load = receiver.model_runner.model.load_weights

    def fail_second(weights):
        if weights[0][0] == "bias":
            return set()
        return original_load(weights)

    receiver.model_runner.model.load_weights = fail_second
    with pytest.raises(RuntimeError, match="did not load pushed weight: bias"):
        receiver.commit_policy_weight_update(1)
    assert receiver._policy_version is None
    with pytest.raises(RuntimeError, match="restart the actor"):
        receiver.commit_policy_weight_update(1)
    with pytest.raises(RuntimeError, match="restart the actor"):
        receiver.start_policy_weight_update(2, PLAN)


def test_legacy_blocking_receiver_uses_staging(receiver):
    assert receiver.receive_policy_weights(PLAN) == 2
    assert receiver._policy_version == 0
    torch.testing.assert_close(receiver.model_runner.model.weight, torch.full((2, 2), 3.0))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_graph_replay_sees_committed_weights_at_original_addresses():
    device = torch.device("cuda", torch.accelerator.current_device_index())
    receiver = make_receiver(device, torch.cuda.Stream(device=device), lambda tensor: tensor.fill_(3))
    model = receiver.model_runner.model
    x = torch.ones(2, device=device)
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup), torch.inference_mode():
        for _ in range(3):
            model(x)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        output = model(x)
    receiver.start_policy_weight_update(1, PLAN)
    finish_receiving(receiver)
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(output, 3))
    receiver.commit_policy_weight_update(1)
    graph.replay()
    torch.testing.assert_close(output, torch.full_like(output, 9))


def _nccl_policy_process(rank, port, staged, decoded):
    """Two real GPUs: trainer broadcasts while actor replays a captured graph."""
    import time

    from duplexio.opd_link import PolicyGroup

    device = torch.device("cuda", rank)
    torch.accelerator.set_device_index(rank)
    plan = [["weight", "float32", [1024, 1024]], ["bias", "float32", [1024]]]
    if rank == 0:
        group = PolicyGroup("127.0.0.1", port, rank, 2, device, timeout_seconds=60)
        for index in range(2):
            assert staged[index].wait(30)
            assert decoded[index].wait(30)
            for _, _, shape in plan:
                group.broadcast(torch.full(shape, 3.0 + index, device=device))
            torch.accelerator.synchronize(device)
        return

    model = Model(device)
    model.weight = torch.nn.Parameter(torch.ones(1024, 1024, device=device))
    model.bias = torch.nn.Parameter(torch.ones(1024, device=device))
    receiver = PolicyWeightReceiver()
    receiver.device = device
    receiver.model_runner = SimpleNamespace(model=model)
    receiver.join_policy_group("127.0.0.1", port, rank, 2)
    x = torch.ones(1024, device=device)
    warmup = torch.cuda.Stream(device=device)
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup), torch.inference_mode():
        for _ in range(3):
            model(x)
    torch.cuda.current_stream().wait_stream(warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        output = model(x)
    expected = torch.full_like(output, 1025.0)
    pointers = [p.data_ptr() for p in model.parameters()]
    staging_pointers = None
    for index in range(2):
        receiver.start_policy_weight_update(index + 1, plan)
        current_pointers = [t.data_ptr() for _, t in receiver._policy_buffer]
        if staging_pointers is not None:
            assert current_pointers == staging_pointers
        staging_pointers = current_pointers
        staged[index].set()
        assert not receiver.policy_weight_update_ready(index + 1)
        deadline = time.monotonic() + 30
        replays = 0
        while not receiver.policy_weight_update_ready(index + 1):
            assert time.monotonic() < deadline, "Transfer blocked concurrent decode"
            graph.replay()
            torch.testing.assert_close(output, expected)
            replays += 1
            decoded[index].set()
        assert replays > 0
        receiver.commit_policy_weight_update(index + 1)
        assert [p.data_ptr() for p in model.parameters()] == pointers
        graph.replay()
        expected.fill_(1025 * (3 + index))
        torch.testing.assert_close(output, expected)
        print(f"policy version {index + 1}: {replays} graph replays during NCCL receipt", flush=True)


@pytest.mark.skipif(torch.accelerator.device_count() < 2, reason="Two CUDA devices required for NCCL")
def test_nccl_receipt_overlaps_graph_decode_and_reuses_buffer():
    import socket

    pytest.importorskip("duplexio.opd_link")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    context = torch.multiprocessing.get_context("spawn")
    staged = [context.Event() for _ in range(2)]
    decoded = [context.Event() for _ in range(2)]
    torch.multiprocessing.spawn(_nccl_policy_process, args=(port, staged, decoded), nprocs=2, join=True)
