# SPDX-License-Identifier: Apache-2.0
"""Worker extension: receive trainer weights over NCCL and load them in place.

Installed through vLLM's `worker_extension_cls`; both methods are reached with
`AsyncOmni.collective_rpc`. Tensors arrive in the native export layout, so the
model's ordinary `load_weights` performs the same fusing and reshaping as a
checkpoint load, and parameter storage (captured by CUDA graphs) is unchanged.
"""

from __future__ import annotations

from typing import Any


class PolicyWeightReceiver:
    def join_policy_group(self, host: str, port: int, rank: int, world_size: int) -> None:
        from duplexio.opd_link import PolicyGroup

        self.policy_group = PolicyGroup(host, port, rank, world_size, self.device)

    def receive_policy_weights(self, weights: list[list[Any]]) -> int:
        from duplexio.opd_link import receive_serving_weights

        receive_serving_weights(self.policy_group, weights, self.model_runner.model.load_weights)
        return len(weights)
