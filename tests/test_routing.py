import unittest

from machboost.client import MachBoostAPIError
from machboost.routing import (
    HostTarget,
    MachBoostHostPool,
    expected_delay_score,
)


class FakeHostClient:
    def __init__(
        self,
        endpoint,
        *,
        timeout,
        api_token,
        fixtures,
    ):
        self.endpoint = endpoint
        self.fixture = fixtures[endpoint]
        self.calls = []

    def metrics(self):
        error = self.fixture.get("probe_error")
        if error:
            raise error
        return {
            "operations": {
                "latency_seconds": {"p50": self.fixture.get("latency", 1.0)},
                "generation_tokens_per_second": self.fixture.get("tokens_per_second", 20.0),
            },
            "models": [
                {
                    "model": "coder",
                    "replicas": self.fixture.get("replicas", 1),
                    "scheduler": {
                        "active_requests": self.fixture.get("active", 0),
                        "queued_requests": self.fixture.get("queued", 0),
                    },
                }
            ]
            if self.fixture.get("loaded", True)
            else [],
            "scheduler": {
                "active_requests": self.fixture.get("active", 0),
                "queued_requests": self.fixture.get("queued", 0),
            },
        }

    def catalog(self):
        return [
            {
                "name": "coder",
                "repository": "org/coder",
                "cached": self.fixture.get("cached", True),
                "support": "ready",
            }
        ]

    def chat(self, model, messages, **kwargs):
        self.calls.append(("chat", model))

        def rows():
            error = self.fixture.get("chat_error")
            if error:
                raise error
            for row in self.fixture.get(
                "rows",
                [
                    {"message": {"content": "ok"}, "done": False},
                    {"message": {"content": ""}, "done": True},
                ],
            ):
                yield dict(row)

        return rows()

    def generate(self, model, prompt, **kwargs):
        return self.chat(model, [{"role": "user", "content": prompt}], **kwargs)

    def load(self, model, **kwargs):
        self.calls.append(("load", model))
        error = self.fixture.get("load_error")
        if error:
            raise error
        return {"instance": {"model": model, "backend": "mlx"}}

    def show(self, model, **kwargs):
        return {"model": model}

    def pull(self, model, **kwargs):
        return {"model": model}

    def stop(self, model=None):
        return {"unloaded": 1}


class HostPoolTests(unittest.TestCase):
    def pool(self, fixtures):
        targets = [
            HostTarget(key, key, endpoint, "token")
            for key, endpoint in (
                ("studio", "http://studio:11435"),
                ("laptop", "http://laptop:11435"),
            )
        ]
        clients = {}

        def factory(endpoint, **kwargs):
            client = FakeHostClient(endpoint, fixtures=fixtures, **kwargs)
            clients[endpoint] = client
            return client

        return MachBoostHostPool(targets, client_factory=factory, probe_ttl=0), clients

    def test_expected_delay_accounts_for_queue_capacity_reservations_and_residency(self):
        idle = expected_delay_score(
            round_trip_seconds=0.01,
            service_seconds=1,
            active_requests=0,
            queued_requests=0,
            replicas=1,
            reserved_requests=0,
            model_loaded=True,
        )
        busy = expected_delay_score(
            round_trip_seconds=0.01,
            service_seconds=1,
            active_requests=2,
            queued_requests=3,
            replicas=1,
            reserved_requests=1,
            model_loaded=True,
        )
        cold = expected_delay_score(
            round_trip_seconds=0.01,
            service_seconds=1,
            active_requests=0,
            queued_requests=0,
            replicas=1,
            reserved_requests=0,
            model_loaded=False,
        )

        self.assertEqual(idle, 1.01)
        self.assertGreater(busy, idle)
        self.assertGreater(cold, idle)

    def test_pool_routes_to_loaded_host_with_lower_expected_delay(self):
        pool, _ = self.pool(
            {
                "http://studio:11435": {"active": 4, "queued": 2, "latency": 2.0},
                "http://laptop:11435": {"active": 0, "queued": 0, "latency": 0.5},
            }
        )

        rows = list(pool.chat("coder", [{"role": "user", "content": "hello"}]))

        self.assertEqual(pool.last_route.target.id, "laptop")
        fabric = rows[-1]["machboost"]["fabric"]
        self.assertEqual(fabric["host_id"], "laptop")
        self.assertEqual(fabric["attempts"], 1)

    def test_pool_fails_over_when_selected_host_fails_before_first_event(self):
        pool, clients = self.pool(
            {
                "http://studio:11435": {
                    "latency": 0.1,
                    "chat_error": MachBoostAPIError("overloaded", status=503),
                },
                "http://laptop:11435": {"latency": 1.0},
            }
        )

        rows = list(pool.chat("coder", [{"role": "user", "content": "hello"}]))

        self.assertEqual(pool.last_route.target.id, "laptop")
        self.assertEqual(rows[-1]["machboost"]["fabric"]["attempts"], 2)
        self.assertEqual(clients["http://studio:11435"].calls, [("chat", "coder")])
        self.assertEqual(clients["http://laptop:11435"].calls, [("chat", "coder")])

    def test_pool_does_not_replay_after_streaming_output(self):
        pool, clients = self.pool(
            {
                "http://studio:11435": {
                    "latency": 0.1,
                    "rows": [
                        {"message": {"content": "partial"}, "done": False},
                    ],
                },
                "http://laptop:11435": {"latency": 1.0},
            }
        )
        original = clients["http://studio:11435"].chat

        def interrupted(model, messages, **kwargs):
            def rows():
                yield {"message": {"content": "partial"}, "done": False}
                raise MachBoostAPIError("lost connection", status=503)

            return rows()

        clients["http://studio:11435"].chat = interrupted

        with self.assertRaises(MachBoostAPIError):
            list(pool.chat("coder", [{"role": "user", "content": "hello"}]))

        self.assertEqual(clients["http://laptop:11435"].calls, [])
        clients["http://studio:11435"].chat = original

    def test_route_status_reports_offline_and_unsupported_hosts(self):
        pool, _ = self.pool(
            {
                "http://studio:11435": {
                    "probe_error": MachBoostAPIError("offline", status=503)
                },
                "http://laptop:11435": {"cached": False},
            }
        )

        status = pool.route_status("coder")

        self.assertIsNone(status["selected"])
        by_id = {row["id"]: row for row in status["hosts"]}
        self.assertFalse(by_id["studio"]["online"])
        self.assertFalse(by_id["laptop"]["supports_model"])


if __name__ == "__main__":
    unittest.main()
