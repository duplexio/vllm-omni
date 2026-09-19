# SPDX-License-Identifier: Apache-2.0
"""Stage NCCL policy updates while decoding, then commit between decode steps.

The worker RPC thread owns the update lifecycle; only the receiver thread writes
staging tensors. Commit uses the ordinary checkpoint loader, preserving parameter
addresses referenced by CUDA graphs. Atomicity is with respect to inference: the
caller must drain its RolloutGate before commit and resume only after success.
A failed commit is fatal (there is no rollback of partially copied parameters).
"""

from __future__ import annotations

from concurrent.futures import Future
from threading import Thread
from typing import Any

import torch


class PolicyWeightReceiver:
    def join_policy_group(self, host: str, port: int, rank: int, world_size: int) -> None:
        from duplexio.opd_link import PolicyGroup

        if getattr(self, "_policy_pending", None) is not None:
            raise RuntimeError("Cannot replace the policy group with an update pending")
        self.policy_group = PolicyGroup(host, port, rank, world_size, self.device)
        self._policy_stream = torch.cuda.Stream(device=self.device)
        self._policy_pending: tuple[int, Future[None]] | None = None
        self._policy_plan: tuple | None = None
        self._policy_buffer: list[tuple[str, torch.Tensor]] = []
        self._policy_version: int | None = None
        self._policy_commit_failed = False

    def start_policy_weight_update(self, version: int, weights: list[list[Any]]) -> int:
        """Start receiving into reusable GPU storage and return without waiting.

        Call before sending READY to the trainer. Only one update may be pending;
        both a failed receive and a failed commit require restarting the actor.
        The additional GPU memory is the size of the transmitted policy tensors.
        """
        if self._policy_commit_failed:
            raise RuntimeError("A policy commit failed; restart the actor")
        if self._policy_pending is not None:
            raise RuntimeError("A policy weight update is already pending")
        if self._policy_version is not None and version <= self._policy_version:
            raise ValueError(f"Policy version {version} must exceed {self._policy_version}")
        plan = []
        names: set[str] = set()
        for name, dtype_name, shape in weights:
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(name, str) or not name or name in names:
                raise ValueError(f"Invalid or duplicate policy weight name: {name!r}")
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"Invalid policy weight dtype: {dtype_name!r}")
            if any(not isinstance(size, int) or size < 0 for size in shape):
                raise ValueError(f"Invalid policy weight shape for {name}: {shape}")
            names.add(name)
            plan.append((name, dtype, tuple(shape)))
        if not plan:
            raise ValueError("A policy weight update must not be empty")
        plan = tuple(plan)
        if plan != self._policy_plan:
            # Allocate before READY so allocation errors cannot strand a trainer
            # that has already started broadcasting. Reuse storage across pushes.
            self._policy_buffer = []
            self._policy_plan = None
            with torch.cuda.device(self.device), torch.cuda.stream(self._policy_stream):
                self._policy_buffer = [
                    (name, torch.empty(shape, dtype=dtype, device=self.device)) for name, dtype, shape in plan
                ]
            self._policy_plan = plan
        future: Future[None] = Future()
        self._policy_pending = (version, future)

        def receive() -> None:
            try:
                with torch.inference_mode(), torch.cuda.device(self.device), torch.cuda.stream(self._policy_stream):
                    for _, tensor in self._policy_buffer:
                        self.policy_group.broadcast(tensor)
                    # NCCL Work.wait establishes ordering on this stream. Wait
                    # only for transfer completion, never synchronize decoding.
                    self._policy_stream.synchronize()
            except BaseException as error:
                future.set_exception(error)
            else:
                future.set_result(None)

        Thread(target=receive, name=f"policy-receive-{version}", daemon=True).start()
        return len(plan)

    def policy_weight_update_ready(self, version: int) -> bool:
        """Nonblocking completion query; propagate receive errors to the actor."""
        if self._policy_pending is None or self._policy_pending[0] != version:
            raise ValueError(f"No pending policy update for version {version}")
        future = self._policy_pending[1]
        if not future.done():
            return False
        future.result()
        return True

    def commit_policy_weight_update(self, version: int) -> int:
        """Publish a completed update under the actor's drained RolloutGate.

        Worker RPCs serialize with model execution. Synchronization also fences
        already-enqueued kernels before overwriting live parameter storage. The
        commit is a bounded local copy/load, not a pointer swap: captured graphs
        must keep seeing the same addresses.
        """
        if self._policy_commit_failed:
            raise RuntimeError("A policy commit failed; restart the actor")
        if not self.policy_weight_update_ready(version):
            raise RuntimeError(f"Policy version {version} is still being received")
        try:
            torch.accelerator.synchronize(self.device)
            with torch.inference_mode():
                # Keep per-tensor validation from the original receiver. The
                # native loader performs packed/sharded projection transforms.
                for name, tensor in self._policy_buffer:
                    if not self.model_runner.model.load_weights([(name, tensor)]):
                        raise RuntimeError(f"Serving model did not load pushed weight: {name}")
            torch.accelerator.synchronize(self.device)
        except BaseException:
            self._policy_commit_failed = True
            raise
        self.model_runner.model.policy_version = version
        self._policy_version = version
        self._policy_pending = None
        return len(self._policy_buffer)

    def receive_policy_weights(self, weights: list[list[Any]]) -> int:
        """Compatibility RPC for callers which already pause for the full transfer."""
        version = 0 if self._policy_version is None else self._policy_version + 1
        self.start_policy_weight_update(version, weights)
        self._policy_pending[1].result()
        return self.commit_policy_weight_update(version)
