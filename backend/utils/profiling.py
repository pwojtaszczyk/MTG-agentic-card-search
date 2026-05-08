from __future__ import annotations

import os
import threading
import time
import uuid
from contextvars import ContextVar

_request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_timings: dict[str, dict[str, float]] = {}
_request_timings_lock = threading.Lock()
_db_bucket_ms: dict[str, dict[str, float]] = {}


def profiling_enabled() -> bool:
    return os.getenv("PROFILING_ENABLED", "").strip().lower() in {"1", "true", "yes", "on"}


def current_request_id() -> str:
    return _request_id_var.get() or "unknown"


def start_request_event() -> None:
    """Emit request start event and initialize per-request timing context."""
    if not profiling_enabled():
        return
    request_id = uuid.uuid4().hex[:8]
    now = time.perf_counter()
    _request_id_var.set(request_id)
    with _request_timings_lock:
        _request_timings[request_id] = {"start": now, "last": now}
        _db_bucket_ms[request_id] = {"fts": 0.0, "vector": 0.0, "other": 0.0}
    print(f"[profile][{request_id}] request received", flush=True)


def _timing_prefix(now: float, last_ts: float | None, start_ts: float | None) -> str:
    delta_ms = 0.0 if last_ts is None else (now - last_ts) * 1000.0
    total_ms = 0.0 if start_ts is None else (now - start_ts) * 1000.0
    return f"[+{delta_ms:9.2f} ms | total {total_ms:9.2f} ms]"


def _get_timing_state(request_id: str) -> tuple[float | None, float | None]:
    with _request_timings_lock:
        state = _request_timings.get(request_id)
        if state is None:
            return None, None
        return state.get("last"), state.get("start")


def _set_last_ts(request_id: str, now: float) -> None:
    with _request_timings_lock:
        state = _request_timings.get(request_id)
        if state is None:
            _request_timings[request_id] = {"start": now, "last": now}
            return
        state["last"] = now


def _clear_timing_state(request_id: str) -> None:
    with _request_timings_lock:
        _request_timings.pop(request_id, None)
        _db_bucket_ms.pop(request_id, None)


def record_db_bucket_time(bucket: str, duration_ms: float) -> None:
    """Accumulate measured SQLite time into ``fts``, ``vector``, or ``other`` for the current request."""
    if not profiling_enabled():
        return
    if bucket not in ("fts", "vector", "other"):
        bucket = "other"
    request_id = current_request_id()
    with _request_timings_lock:
        if request_id not in _db_bucket_ms:
            _db_bucket_ms[request_id] = {"fts": 0.0, "vector": 0.0, "other": 0.0}
        _db_bucket_ms[request_id][bucket] += duration_ms


def emit_db_aggregate_summary() -> None:
    """Print three lines: total DB ms for fts search, vector search, and all other queries."""
    if not profiling_enabled():
        return
    request_id = current_request_id()
    with _request_timings_lock:
        totals = _db_bucket_ms.pop(request_id, None)
    if totals is None:
        return
    for msg, key in (
        ("db aggregate: fts search", "fts"),
        ("db aggregate: vector search", "vector"),
        ("db aggregate: all other", "other"),
    ):
        ms = totals[key]
        now = time.perf_counter()
        last_ts, start_ts = _get_timing_state(request_id)
        prefix = _timing_prefix(now, last_ts, start_ts)
        print(
            f"[profile][{request_id}] {prefix} {msg} (queries {ms:.2f} ms)",
            flush=True,
        )
        _set_last_ts(request_id, now)


def emit_event(name: str) -> None:
    """Emit event elapsed time from the previous event for this request."""
    if not profiling_enabled():
        return
    now = time.perf_counter()
    request_id = _request_id_var.get() or "unknown"
    last_ts, start_ts = _get_timing_state(request_id)
    prefix = _timing_prefix(now, last_ts, start_ts)
    print(f"[profile][{request_id}] {prefix} {name}", flush=True)
    _set_last_ts(request_id, now)
    if name == "agent responded to front end":
        _clear_timing_state(request_id)

