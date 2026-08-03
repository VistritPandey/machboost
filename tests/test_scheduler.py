from __future__ import annotations

import threading
import time
import unittest

from machboost.scheduler import ReplicaPool, RequestAdmissionError


class ReplicaPoolTests(unittest.TestCase):
    def test_two_replicas_can_be_active_together(self) -> None:
        pool = ReplicaPool(["worker-0", "worker-1"])
        first = pool.acquire()
        second = pool.acquire()

        snapshot = pool.snapshot()

        self.assertNotEqual(first.index, second.index)
        self.assertEqual(snapshot["active_requests"], 2)
        self.assertEqual(snapshot["max_active_requests"], 2)
        pool.release(first.index)
        pool.release(second.index)

    def test_waiting_request_is_admitted_after_release(self) -> None:
        pool = ReplicaPool(["worker"], max_queue=1, queue_timeout=1.0)
        first = pool.acquire()
        acquired = threading.Event()

        def wait_for_slot() -> None:
            with pool.slot():
                acquired.set()

        thread = threading.Thread(target=wait_for_slot)
        thread.start()
        self.assertTrue(_wait_for(lambda: pool.snapshot()["queued_requests"] == 1))

        pool.release(first.index)
        self.assertTrue(acquired.wait(timeout=1.0))
        thread.join(timeout=1.0)
        self.assertFalse(thread.is_alive())
        self.assertEqual(pool.snapshot()["completed_requests"], 2)

    def test_full_queue_rejects_immediately(self) -> None:
        pool = ReplicaPool(["worker"], max_queue=0)
        first = pool.acquire()

        with self.assertRaisesRegex(RequestAdmissionError, "queue is full") as raised:
            pool.acquire()

        self.assertEqual(raised.exception.reason, "queue_full")
        self.assertEqual(pool.snapshot()["rejected_requests"], 1)
        pool.release(first.index)

    def test_queue_timeout_is_reported(self) -> None:
        pool = ReplicaPool(["worker"], max_queue=1, queue_timeout=0.01)
        first = pool.acquire()

        with self.assertRaisesRegex(RequestAdmissionError, "timed out") as raised:
            pool.acquire()

        self.assertEqual(raised.exception.reason, "queue_timeout")
        self.assertEqual(pool.snapshot()["timed_out_requests"], 1)
        pool.release(first.index)

    def test_affinity_prefers_the_same_available_replica(self) -> None:
        pool = ReplicaPool(["worker-0", "worker-1"])

        with pool.slot(affinity_key="account-42") as first:
            first_index = first.index
        with pool.slot(affinity_key="account-42") as second:
            second_index = second.index

        self.assertEqual(first_index, second_index)

    def test_close_waits_for_active_replica(self) -> None:
        pool = ReplicaPool(["worker"])
        lease = pool.acquire()
        closed = threading.Event()

        thread = threading.Thread(
            target=lambda: (pool.close(wait=True), closed.set())
        )
        thread.start()
        self.assertFalse(closed.wait(timeout=0.02))

        pool.release(lease.index)
        self.assertTrue(closed.wait(timeout=1.0))
        thread.join(timeout=1.0)

    def test_try_close_idle_does_not_close_an_active_pool(self) -> None:
        pool = ReplicaPool(["worker"])
        lease = pool.acquire()

        self.assertFalse(pool.try_close_idle())
        pool.release(lease.index)
        self.assertTrue(pool.try_close_idle())
        with self.assertRaises(RequestAdmissionError) as raised:
            pool.acquire()
        self.assertEqual(raised.exception.reason, "scheduler_closed")

    def test_tenant_round_robin_prevents_one_key_from_blocking_another(self) -> None:
        pool = ReplicaPool(["worker"], max_queue=3, queue_timeout=1.0)
        active = pool.acquire(tenant_key="busy-key")
        admitted: list[str] = []

        def wait_for_slot(tenant: str) -> None:
            with pool.slot(tenant_key=tenant):
                admitted.append(tenant)

        threads = []
        for tenant in ("busy-key", "busy-key", "other-key"):
            thread = threading.Thread(target=wait_for_slot, args=(tenant,))
            thread.start()
            threads.append(thread)
            expected = len(threads)
            self.assertTrue(
                _wait_for(lambda: pool.snapshot()["queued_requests"] == expected)
            )

        snapshot = pool.snapshot()
        self.assertEqual(snapshot["queued_by_tenant"]["busy-key"], 2)
        self.assertEqual(snapshot["queued_by_tenant"]["other-key"], 1)

        pool.release(active.index)
        for thread in threads:
            thread.join(timeout=1.0)

        self.assertEqual(admitted[0], "other-key")
        self.assertEqual(sorted(admitted), ["busy-key", "busy-key", "other-key"])


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


if __name__ == "__main__":
    unittest.main()
