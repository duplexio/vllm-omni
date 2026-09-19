from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DuplexAppendTaskMeta:
    epoch: int
    mode: str
    final: bool
    response_bound: bool


@dataclass
class DuplexSessionTasks:
    """Connection-independent task handles for one resumable session."""

    native_append_tasks: dict[asyncio.Task[bool], DuplexAppendTaskMeta] = field(default_factory=dict)
    native_append_tail: asyncio.Task[bool] | None = None
    active_response_task: asyncio.Task[None] | None = None
    data_plane_task: asyncio.Task[None] | None = None
    data_plane_restart_requested: bool = False
    continuation_owner_id: str | None = None
    continuation_units: int = 0
    pending_silence_task: asyncio.Task[bool] | None = None
    pending_silence_owner_id: str | None = None
    silence_continuation_scheduler: Callable[..., Awaitable[bool]] | None = None

    def clear_continuation(self) -> None:
        """Forget continuation ownership without cancelling an in-flight append."""
        self.continuation_owner_id = None
        self.continuation_units = 0
        self.pending_silence_task = None
        self.pending_silence_owner_id = None

    def track_append_task(
        self,
        task: asyncio.Task[bool],
        *,
        epoch: int,
        mode: str,
        final: bool,
        response_bound: bool,
    ) -> None:
        self.native_append_tasks[task] = DuplexAppendTaskMeta(epoch, mode, final, response_bound)
        task.add_done_callback(self.native_append_tasks.pop)

    def has_response_bound_append_tasks(self) -> bool:
        return any(meta.response_bound for meta in self.native_append_tasks.values())

    async def cancel_append_tasks(self, timeout_s: float = 0.25, *, response_bound_only: bool = False) -> bool:
        tasks = [
            task for task, meta in self.native_append_tasks.items() if not response_bound_only or meta.response_bound
        ]
        if not tasks:
            return False
        for task in tasks:
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=timeout_s)
        except TimeoutError:
            pass
        return True
