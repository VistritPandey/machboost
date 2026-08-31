from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable, Iterator, Optional, Sequence

from .client import MachBoostAPIError, MachBoostClient


@dataclass(frozen=True)
class HostTarget:
    id: str
    name: str
    endpoint: str
    api_token: Optional[str] = None


@dataclass(frozen=True)
class HostProbe:
    target: HostTarget
    online: bool
    supports_model: bool
    model_loaded: bool
    round_trip_seconds: float
    active_requests: int = 0
    queued_requests: int = 0
    replicas: int = 1
    service_seconds: float = 0.75
    generation_tokens_per_second: float = 0.0
    score: float = float("inf")
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.target.id,
            "name": self.target.name,
            "endpoint": self.target.endpoint,
            "online": self.online,
            "supports_model": self.supports_model,
            "model_loaded": self.model_loaded,
            "round_trip_seconds": self.round_trip_seconds,
            "active_requests": self.active_requests,
            "queued_requests": self.queued_requests,
            "replicas": self.replicas,
            "service_seconds": self.service_seconds,
            "generation_tokens_per_second": self.generation_tokens_per_second,
            "score": None if self.score == float("inf") else self.score,
            "error": self.error,
        }


@dataclass
class _HostRuntime:
    client: MachBoostClient
    reserved: int = 0
    failures: int = 0
    cooldown_until: float = 0.0
    rtt_ewma: Optional[float] = None
    probe: Optional[HostProbe] = None
    probe_model: Optional[str] = None
    probe_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


def expected_delay_score(
    *,
    round_trip_seconds: float,
    service_seconds: float,
    active_requests: int,
    queued_requests: int,
    replicas: int,
    reserved_requests: int,
    model_loaded: bool,
) -> float:
    replicas = max(1, int(replicas))
    service = max(0.05, float(service_seconds))
    demand_ahead = max(0, int(queued_requests)) + max(0, int(reserved_requests))
    active_over_capacity = max(0, int(active_requests) - replicas + 1)
    queue_delay = (demand_ahead + active_over_capacity) * service / replicas
    cold_penalty = 0.0 if model_loaded else max(2.0, service * 4.0)
    return max(0.0, float(round_trip_seconds)) + service + queue_delay + cold_penalty


class MachBoostHostPool:
    def __init__(
        self,
        targets: Sequence[HostTarget],
        *,
        timeout: float = 300.0,
        probe_ttl: float = 2.0,
        cooldown_seconds: float = 10.0,
        clock: Callable[[], float] = time.monotonic,
        client_factory: Callable[..., MachBoostClient] = MachBoostClient,
    ) -> None:
        if not targets:
            raise ValueError("at least one MachBoost host is required")
        self.targets = tuple(targets)
        self.timeout = float(timeout)
        self.probe_ttl = max(0.0, float(probe_ttl))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.clock = clock
        self._runtimes = {
            target.id: _HostRuntime(
                client=client_factory(
                    target.endpoint,
                    timeout=self.timeout,
                    api_token=target.api_token,
                )
            )
            for target in self.targets
        }
        self._lock = threading.Lock()
        self.last_route: Optional[HostProbe] = None

    @property
    def endpoint(self) -> str:
        route = self.last_route
        return route.target.endpoint if route is not None else "machboost://auto"

    def is_healthy(self) -> bool:
        return any(probe.online for probe in self.probe())

    def probe(self, model: Optional[str] = None, *, force: bool = False) -> list[HostProbe]:
        now = self.clock()
        pending: list[HostTarget] = []
        result: list[HostProbe] = []
        for target in self.targets:
            runtime = self._runtimes[target.id]
            with runtime.lock:
                cached = runtime.probe
                if (
                    not force
                    and cached is not None
                    and runtime.probe_model == model
                    and now - runtime.probe_at <= self.probe_ttl
                ):
                    result.append(self._rescore(cached, runtime))
                else:
                    pending.append(target)

        if pending:
            with ThreadPoolExecutor(max_workers=min(8, len(pending))) as executor:
                futures = {
                    executor.submit(self._probe_target, target, model): target
                    for target in pending
                }
                for future in as_completed(futures):
                    target = futures[future]
                    probe = future.result()
                    runtime = self._runtimes[target.id]
                    with runtime.lock:
                        runtime.probe = probe
                        runtime.probe_model = model
                        runtime.probe_at = self.clock()
                        if probe.online:
                            runtime.failures = 0
                            runtime.cooldown_until = 0.0
                    result.append(self._rescore(probe, runtime))
        return sorted(result, key=lambda item: (item.score, item.target.name.lower()))

    def ranked(self, model: str, *, exclude: Sequence[str] = ()) -> list[HostProbe]:
        excluded = set(exclude)
        now = self.clock()
        return [
            probe
            for probe in self.probe(model)
            if probe.target.id not in excluded
            and probe.online
            and probe.supports_model
            and self._runtimes[probe.target.id].cooldown_until <= now
        ]

    def chat(self, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Iterator[dict[str, Any]]:
        return self._stream_with_failover("chat", model, messages, kwargs)

    def generate(self, model: str, prompt: str, **kwargs: Any) -> Iterator[dict[str, Any]]:
        return self._stream_with_failover("generate", model, prompt, kwargs)

    def load(self, model: str, **kwargs: Any) -> dict[str, Any]:
        return self._call_with_failover("load", model, model, kwargs)

    def show(self, model: str, **kwargs: Any) -> dict[str, Any]:
        return self._call_with_failover("show", model, model, kwargs, require_support=False)

    def pull(self, model: str, **kwargs: Any) -> Any:
        return self._call_with_failover("pull", model, model, kwargs, require_support=False)

    def stop(self, model: Optional[str] = None) -> dict[str, Any]:
        if model and self.last_route is not None:
            runtime = self._runtimes[self.last_route.target.id]
            return runtime.client.stop(model)
        stopped = 0
        for runtime in self._runtimes.values():
            try:
                stopped += int(runtime.client.stop(model).get("unloaded") or 0)
            except MachBoostAPIError:
                continue
        return {"unloaded": stopped}

    def route_status(self, model: Optional[str] = None) -> dict[str, Any]:
        probes = self.probe(model, force=True)
        selected = next(
            (probe for probe in probes if probe.online and (model is None or probe.supports_model)),
            None,
        )
        return {
            "schema": "machboost.host_pool.v1",
            "model": model,
            "selected": selected.target.id if selected else None,
            "hosts": [probe.to_dict() for probe in probes],
        }

    def _stream_with_failover(
        self,
        method: str,
        model: str,
        request: Any,
        kwargs: dict[str, Any],
    ) -> Iterator[dict[str, Any]]:
        attempted: list[str] = []
        last_error: Optional[Exception] = None
        while True:
            candidates = self.ranked(model, exclude=attempted)
            if not candidates:
                if last_error is not None:
                    raise last_error
                raise MachBoostAPIError(f"no healthy MachBoost host has {model} ready")
            probe = candidates[0]
            attempted.append(probe.target.id)
            runtime = self._runtimes[probe.target.id]
            self._reserve(runtime)
            emitted = False
            started = self.clock()
            try:
                source = getattr(runtime.client, method)(model, request, **kwargs)
                for row in source:
                    if not emitted:
                        emitted = True
                        self._mark_success(runtime, probe, self.clock() - started)
                    if row.get("done"):
                        row = dict(row)
                        extension = dict(row.get("machboost") or {})
                        extension["fabric"] = self._route_payload(probe, attempted)
                        row["machboost"] = extension
                    yield row
                if not emitted:
                    self._mark_success(runtime, probe, self.clock() - started)
                return
            except Exception as exc:
                last_error = exc
                self._mark_failure(runtime)
                if emitted or not _transient_error(exc):
                    raise
            finally:
                self._release(runtime)

    def _call_with_failover(
        self,
        method: str,
        route_model: str,
        argument: Any,
        kwargs: dict[str, Any],
        *,
        require_support: bool = True,
    ) -> dict[str, Any]:
        probes = self.ranked(route_model) if require_support else self.probe(route_model)
        candidates = [probe for probe in probes if probe.online]
        if not candidates:
            raise MachBoostAPIError("no healthy MachBoost host is reachable")
        last_error: Optional[Exception] = None
        for probe in candidates:
            runtime = self._runtimes[probe.target.id]
            self._reserve(runtime)
            started = self.clock()
            try:
                value = getattr(runtime.client, method)(argument, **kwargs)
                self._mark_success(runtime, probe, self.clock() - started)
                return value
            except Exception as exc:
                last_error = exc
                self._mark_failure(runtime)
                if not _transient_error(exc):
                    raise
            finally:
                self._release(runtime)
        assert last_error is not None
        raise last_error

    def _probe_target(self, target: HostTarget, model: Optional[str]) -> HostProbe:
        runtime = self._runtimes[target.id]
        started = self.clock()
        try:
            metrics = runtime.client.metrics()
            catalog = runtime.client.catalog() if model else []
        except Exception as exc:
            return HostProbe(
                target=target,
                online=False,
                supports_model=False,
                model_loaded=False,
                round_trip_seconds=max(0.0, self.clock() - started),
                error=str(exc),
            )
        elapsed = max(0.0001, self.clock() - started)
        with runtime.lock:
            runtime.rtt_ewma = elapsed if runtime.rtt_ewma is None else (
                runtime.rtt_ewma * 0.7 + elapsed * 0.3
            )
            rtt = runtime.rtt_ewma
        models = list(metrics.get("models") or ())
        model_rows = [row for row in models if _model_matches(row, model)] if model else models
        scheduler = _model_scheduler(model_rows) or dict(metrics.get("scheduler") or {})
        operations = dict(metrics.get("operations") or {})
        latency = dict(operations.get("latency_seconds") or {})
        service_seconds = max(0.05, float(latency.get("p50") or 0.75))
        supports = model is None or any(
            _model_matches(row, model)
            and bool(row.get("cached"))
            and str(row.get("support") or "ready") == "ready"
            for row in catalog
        )
        return HostProbe(
            target=target,
            online=True,
            supports_model=supports,
            model_loaded=bool(model_rows),
            round_trip_seconds=rtt,
            active_requests=int(scheduler.get("active_requests") or 0),
            queued_requests=int(scheduler.get("queued_requests") or 0),
            replicas=max(1, sum(int(row.get("replicas") or 1) for row in model_rows) or 1),
            service_seconds=service_seconds,
            generation_tokens_per_second=float(
                operations.get("generation_tokens_per_second") or 0.0
            ),
        )

    def _rescore(self, probe: HostProbe, runtime: _HostRuntime) -> HostProbe:
        score = expected_delay_score(
            round_trip_seconds=probe.round_trip_seconds,
            service_seconds=probe.service_seconds,
            active_requests=probe.active_requests,
            queued_requests=probe.queued_requests,
            replicas=probe.replicas,
            reserved_requests=runtime.reserved,
            model_loaded=probe.model_loaded,
        )
        return HostProbe(**{**probe.__dict__, "score": score})

    def _reserve(self, runtime: _HostRuntime) -> None:
        with runtime.lock:
            runtime.reserved += 1

    def _release(self, runtime: _HostRuntime) -> None:
        with runtime.lock:
            runtime.reserved = max(0, runtime.reserved - 1)

    def _mark_success(self, runtime: _HostRuntime, probe: HostProbe, elapsed: float) -> None:
        with runtime.lock:
            runtime.failures = 0
            runtime.cooldown_until = 0.0
            self.last_route = probe

    def _mark_failure(self, runtime: _HostRuntime) -> None:
        with runtime.lock:
            runtime.failures += 1
            runtime.cooldown_until = self.clock() + min(
                60.0,
                self.cooldown_seconds * (2 ** max(0, runtime.failures - 1)),
            )

    @staticmethod
    def _route_payload(probe: HostProbe, attempted: Sequence[str]) -> dict[str, Any]:
        return {
            "schema": "machboost.fabric.route.v1",
            "host_id": probe.target.id,
            "host_name": probe.target.name,
            "endpoint": probe.target.endpoint,
            "expected_delay_seconds": probe.score,
            "round_trip_seconds": probe.round_trip_seconds,
            "model_loaded": probe.model_loaded,
            "active_requests": probe.active_requests,
            "queued_requests": probe.queued_requests,
            "attempts": len(attempted),
        }


def _model_matches(row: dict[str, Any], model: Optional[str]) -> bool:
    if model is None:
        return True
    requested = str(model).removesuffix(":latest")
    values = {
        str(row.get("name") or "").removesuffix(":latest"),
        str(row.get("model") or "").removesuffix(":latest"),
        str(row.get("repository") or "").removesuffix(":latest"),
    }
    return requested in values


def _model_scheduler(rows: Sequence[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if not rows:
        return None
    schedulers = [dict(row.get("scheduler") or {}) for row in rows]
    return {
        key: sum(int(item.get(key) or 0) for item in schedulers)
        for key in ("active_requests", "queued_requests")
    }


def _transient_error(exc: Exception) -> bool:
    if not isinstance(exc, MachBoostAPIError):
        return isinstance(exc, (ConnectionError, OSError, TimeoutError))
    return exc.status is None or exc.status in {408, 425, 429, 500, 502, 503, 504}
