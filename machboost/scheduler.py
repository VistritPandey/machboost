from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
import threading
import time
from typing import Any, Callable, Iterator, Optional, Sequence


class RequestAdmissionError(RuntimeError):
    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ReplicaLease:
    index: int
    resource: Any
    queue_wait_seconds: float


class ReplicaPool:
    def __init__(
        self,
        resources: Sequence[Any],
        *,
        max_queue: int = 64,
        queue_timeout: float = 300.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not resources:
            raise ValueError("replica pool requires at least one resource")
        if max_queue < 0:
            raise ValueError("max_queue cannot be negative")
        if math.isnan(float(queue_timeout)):
            raise ValueError("queue_timeout cannot be NaN")
        self.resources = tuple(resources)
        self.max_queue = int(max_queue)
        self.queue_timeout = float(queue_timeout)
        self.clock = clock
        self._condition = threading.Condition(threading.RLock())
        self._available = set(range(len(self.resources)))
        self._active: set[int] = set()
        self._waiters: deque[int] = deque()
        self._next_ticket = 0
        self._closed = False
        self._accepted = 0
        self._completed = 0
        self._rejected = 0
        self._timed_out = 0
        self._max_active = 0
        self._max_queued = 0
        self._queue_wait_total = 0.0
        self._worker_requests = [0 for _ in self.resources]

    @contextmanager
    def slot(
        self,
        *,
        affinity_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[ReplicaLease]:
        lease = self.acquire(affinity_key=affinity_key, timeout=timeout)
        try:
            yield lease
        finally:
            self.release(lease.index)

    def acquire(
        self,
        *,
        affinity_key: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> ReplicaLease:
        started = self.clock()
        wait_timeout = self.queue_timeout if timeout is None else float(timeout)
        deadline = None if wait_timeout < 0 else started + wait_timeout
        preferred = self._preferred_index(affinity_key)
        ticket: Optional[int] = None

        with self._condition:
            if self._closed:
                raise RequestAdmissionError(
                    "model scheduler is shutting down",
                    reason="scheduler_closed",
                )
            if self._waiters or not self._available:
                if len(self._waiters) >= self.max_queue:
                    self._rejected += 1
                    raise RequestAdmissionError(
                        "model request queue is full",
                        reason="queue_full",
                    )
                ticket = self._next_ticket
                self._next_ticket += 1
                self._waiters.append(ticket)
                self._max_queued = max(self._max_queued, len(self._waiters))

                while True:
                    if self._closed:
                        self._remove_waiter(ticket)
                        raise RequestAdmissionError(
                            "model scheduler is shutting down",
                            reason="scheduler_closed",
                        )
                    if self._waiters[0] == ticket and self._available:
                        self._waiters.popleft()
                        break
                    remaining = None if deadline is None else deadline - self.clock()
                    if remaining is not None and remaining <= 0:
                        self._remove_waiter(ticket)
                        self._timed_out += 1
                        self._condition.notify_all()
                        raise RequestAdmissionError(
                            "timed out waiting for an inference replica",
                            reason="queue_timeout",
                        )
                    self._condition.wait(remaining)

            index = self._select_available(preferred)
            self._available.remove(index)
            self._active.add(index)
            self._accepted += 1
            self._worker_requests[index] += 1
            self._max_active = max(self._max_active, len(self._active))
            queue_wait = max(0.0, self.clock() - started) if ticket is not None else 0.0
            self._queue_wait_total += queue_wait
            self._condition.notify_all()
            return ReplicaLease(
                index=index,
                resource=self.resources[index],
                queue_wait_seconds=queue_wait,
            )

    def release(self, index: int) -> None:
        with self._condition:
            if index not in self._active:
                raise RuntimeError(f"replica {index} is not active")
            self._active.remove(index)
            self._available.add(index)
            self._completed += 1
            self._condition.notify_all()

    def close(self, *, wait: bool = True) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
            if wait:
                while self._active:
                    self._condition.wait()

    def try_close_idle(self) -> bool:
        """Close the pool only when no admitted or queued request can use it."""
        with self._condition:
            if self._active or self._waiters:
                return False
            self._closed = True
            self._condition.notify_all()
            return True

    def is_idle(self) -> bool:
        with self._condition:
            return not self._active and not self._waiters

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            accepted = self._accepted
            return {
                "replicas": len(self.resources),
                "active_requests": len(self._active),
                "queued_requests": len(self._waiters),
                "max_queue": self.max_queue,
                "queue_timeout_seconds": self.queue_timeout,
                "accepted_requests": accepted,
                "completed_requests": self._completed,
                "rejected_requests": self._rejected,
                "timed_out_requests": self._timed_out,
                "max_active_requests": self._max_active,
                "max_queued_requests": self._max_queued,
                "mean_queue_wait_seconds": (
                    self._queue_wait_total / accepted if accepted else 0.0
                ),
                "workers": [
                    {
                        "index": index,
                        "busy": index in self._active,
                        "requests": self._worker_requests[index],
                    }
                    for index in range(len(self.resources))
                ],
            }

    def _preferred_index(self, affinity_key: Optional[str]) -> Optional[int]:
        if not affinity_key:
            return None
        digest = hashlib.sha256(str(affinity_key).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % len(self.resources)

    def _select_available(self, preferred: Optional[int]) -> int:
        if preferred is not None and preferred in self._available:
            return preferred
        return min(self._available)

    def _remove_waiter(self, ticket: int) -> None:
        try:
            self._waiters.remove(ticket)
        except ValueError:
            pass
