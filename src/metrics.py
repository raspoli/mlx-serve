"""
metrics.py — quantitative request metrics and system resource tracking.

Stores recent request metrics (TTFT, TPS, duration) and periodic memory
snapshots in bounded deques.  Also writes to JSONL files for history
that survives restarts.
"""
import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

import events

logger = logging.getLogger("mlx-serve.metrics")

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RequestMetrics:
    request_id: str
    model: str
    endpoint: str
    timestamp: str                      # ISO wall-clock
    started_at: float                   # monotonic (internal only)
    total_duration_ms: float = 0.0
    ttft_ms: float | None = None       # time to first token (streaming)
    tokens_per_second: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    status_code: int = 200
    error: str | None = None
    cold_start: bool = False            # model was loaded on demand


@dataclass
class MemorySnapshot:
    timestamp: str
    # System RAM
    ram_total_gb: float
    ram_used_gb: float
    ram_available_gb: float
    ram_percent: float
    # Swap
    swap_used_gb: float
    swap_percent: float
    # Metal GPU (in-process models only; None for subprocess)
    metal_active_mb: float | None = None
    metal_peak_mb: float | None = None
    metal_cache_mb: float | None = None
    # Subprocess RSS (text/vision models)
    subprocess_rss_mb: float | None = None
    subprocess_pid: int | None = None
    # Context
    active_model: str | None = None
    event: str = "sample"               # "sample", "model_loaded", "model_unloaded"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_requests: deque[RequestMetrics] = deque(maxlen=500)
_memory: deque[MemorySnapshot] = deque(maxlen=360)
_requests_file: Path | None = None
_memory_file: Path | None = None

# Running aggregates per model
_aggregates: dict[str, dict[str, Any]] = {}


def configure(
    log_dir: Path,
    requests_max: int = 500,
    memory_max: int = 360,
) -> None:
    """Set file paths and resize deques."""
    global _requests, _memory, _requests_file, _memory_file
    log_dir.mkdir(parents=True, exist_ok=True)
    _requests_file = log_dir / "requests.jsonl"
    _memory_file = log_dir / "memory.jsonl"
    _requests = deque(_requests, maxlen=requests_max)
    _memory = deque(_memory, maxlen=memory_max)


# ---------------------------------------------------------------------------
# Request recording
# ---------------------------------------------------------------------------


def record_request(m: RequestMetrics) -> None:
    """Append a completed request to history and update aggregates."""
    _requests.append(m)
    _update_aggregates(m)

    # Log to terminal
    parts = [
        f"{m.endpoint} {m.status_code}",
        f"model={m.model}",
        f"{m.total_duration_ms:.0f}ms",
    ]
    if m.ttft_ms is not None:
        parts.append(f"TTFT={m.ttft_ms:.0f}ms")
    if m.tokens_per_second is not None:
        parts.append(f"{m.tokens_per_second:.1f} tok/s")
    if m.prompt_tokens is not None:
        parts.append(f"prompt={m.prompt_tokens}")
    if m.completion_tokens is not None:
        parts.append(f"completion={m.completion_tokens}")
    if m.cold_start:
        parts.append("cold_start=true")
    if m.error:
        parts.append(f"error={m.error}")
    logger.info(" | ".join(parts))

    # Write to JSONL
    if _requests_file is not None:
        try:
            record = asdict(m)
            record.pop("started_at", None)  # internal monotonic, not useful on disk
            with open(_requests_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            pass


def _update_aggregates(m: RequestMetrics) -> None:
    agg = _aggregates.setdefault(m.model, {
        "total_requests": 0,
        "total_errors": 0,
        "total_duration_ms": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "ttft_values": [],
        "tps_values": [],
        "cold_starts": 0,
    })
    agg["total_requests"] += 1
    agg["total_duration_ms"] += m.total_duration_ms
    if m.error:
        agg["total_errors"] += 1
    if m.prompt_tokens:
        agg["total_prompt_tokens"] += m.prompt_tokens
    if m.completion_tokens:
        agg["total_completion_tokens"] += m.completion_tokens
    if m.ttft_ms is not None:
        agg["ttft_values"].append(m.ttft_ms)
        # keep last 200 for percentile calculation
        if len(agg["ttft_values"]) > 200:
            agg["ttft_values"] = agg["ttft_values"][-200:]
    if m.tokens_per_second is not None:
        agg["tps_values"].append(m.tokens_per_second)
        if len(agg["tps_values"]) > 200:
            agg["tps_values"] = agg["tps_values"][-200:]
    if m.cold_start:
        agg["cold_starts"] += 1


# ---------------------------------------------------------------------------
# Memory snapshots
# ---------------------------------------------------------------------------


def take_memory_snapshot(
    active_model: str | None = None,
    event: str = "sample",
    subprocess_pid: int | None = None,
) -> MemorySnapshot:
    """Capture current system + Metal memory state."""
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    metal = get_metal_memory()

    sub_rss = None
    if subprocess_pid is not None:
        try:
            sub_rss = round(psutil.Process(subprocess_pid).memory_info().rss / 1_048_576, 1)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    snap = MemorySnapshot(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ram_total_gb=round(vm.total / 1e9, 1),
        ram_used_gb=round(vm.used / 1e9, 1),
        ram_available_gb=round(vm.available / 1e9, 1),
        ram_percent=vm.percent,
        swap_used_gb=round(swap.used / 1e9, 2),
        swap_percent=swap.percent,
        metal_active_mb=metal.get("active_mb"),
        metal_peak_mb=metal.get("peak_mb"),
        metal_cache_mb=metal.get("cache_mb"),
        subprocess_rss_mb=sub_rss,
        subprocess_pid=subprocess_pid,
        active_model=active_model,
        event=event,
    )
    _memory.append(snap)

    # Write to JSONL (at reduced frequency — caller controls this)
    if _memory_file is not None and event != "sample":
        try:
            with open(_memory_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(snap), default=str) + "\n")
        except OSError:
            pass

    return snap


def get_metal_memory() -> dict[str, float | None]:
    """Read Metal GPU memory from mlx.core.metal. Returns empty values if unavailable."""
    try:
        import mlx.core as mx
        return {
            "active_mb": round(mx.metal.get_active_memory() / 1_048_576, 1),
            "peak_mb": round(mx.metal.get_peak_memory() / 1_048_576, 1),
            "cache_mb": round(mx.metal.get_cache_memory() / 1_048_576, 1),
        }
    except Exception:
        return {"active_mb": None, "peak_mb": None, "cache_mb": None}


def check_memory_pressure() -> str:
    """Classify current memory pressure: nominal, warn, critical."""
    vm = psutil.virtual_memory()
    if vm.percent > 90:
        return "critical"
    if vm.percent > 75:
        return "warn"
    return "nominal"


def check_swap_active() -> tuple[bool, float]:
    """Return (is_swap_used, swap_gb)."""
    swap = psutil.swap_memory()
    gb = round(swap.used / 1e9, 2)
    return gb > 0.1, gb


# ---------------------------------------------------------------------------
# Background sampler
# ---------------------------------------------------------------------------

_last_pressure: str = "nominal"
_last_swap_warned: bool = False
# disk write counter — write every 6th sample (60s at 10s interval)
_disk_write_counter: int = 0


async def start_memory_sampler(
    interval: int = 10,
    get_active_model=None,
    get_subprocess_pid=None,
) -> None:
    """Background task: sample memory every `interval` seconds."""
    import asyncio
    global _last_pressure, _last_swap_warned, _disk_write_counter

    while True:
        await asyncio.sleep(interval)
        try:
            model = get_active_model() if get_active_model else None
            pid = get_subprocess_pid() if get_subprocess_pid else None

            snap = take_memory_snapshot(
                active_model=model, event="sample", subprocess_pid=pid,
            )

            # Write to disk every 60s
            _disk_write_counter += 1
            if _disk_write_counter >= 6 and _memory_file is not None:
                _disk_write_counter = 0
                try:
                    with open(_memory_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(asdict(snap), default=str) + "\n")
                except OSError:
                    pass

            # Check memory pressure transitions
            pressure = check_memory_pressure()
            if pressure != _last_pressure:
                if pressure == "critical":
                    events.emit(
                        events.EventType.MEMORY_PRESSURE_CRITICAL,
                        detail={
                            "ram_percent": snap.ram_percent,
                            "ram_available_gb": snap.ram_available_gb,
                        },
                    )
                    logger.warning(
                        f"Memory pressure CRITICAL: {snap.ram_percent}% used, "
                        f"{snap.ram_available_gb}GB available"
                    )
                elif pressure == "warn":
                    events.emit(
                        events.EventType.MEMORY_PRESSURE_WARN,
                        detail={
                            "ram_percent": snap.ram_percent,
                            "ram_available_gb": snap.ram_available_gb,
                        },
                    )
                _last_pressure = pressure

            # Check swap
            swap_active, swap_gb = check_swap_active()
            if swap_active and not _last_swap_warned:
                events.emit(
                    events.EventType.SWAP_ACTIVE,
                    detail={"swap_used_gb": swap_gb},
                )
                logger.warning(f"Swap memory active: {swap_gb}GB used")
                _last_swap_warned = True
            elif not swap_active:
                _last_swap_warned = False

        except Exception:
            pass  # never crash the sampler


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------


def get_request_history(
    model: str | None = None,
    last_n: int = 50,
) -> list[dict]:
    """Return recent requests, newest first."""
    results = []
    for m in reversed(_requests):
        if model and m.model != model:
            continue
        d = asdict(m)
        d.pop("started_at", None)
        results.append(d)
        if len(results) >= last_n:
            break
    return results


def get_memory_timeline(last_n: int = 60) -> list[dict]:
    """Return recent memory snapshots, newest first."""
    items = list(_memory)[-last_n:]
    return [asdict(s) for s in reversed(items)]


def get_current_memory() -> dict:
    """Return a single current memory snapshot as dict."""
    snap = take_memory_snapshot(event="query")
    _memory.pop()  # don't pollute the timeline with manual queries
    return asdict(snap)


def get_aggregates() -> dict[str, dict]:
    """Return per-model aggregate metrics with computed averages."""
    result = {}
    for model, agg in _aggregates.items():
        ttft_vals = agg["ttft_values"]
        tps_vals = agg["tps_values"]

        entry = {
            "total_requests": agg["total_requests"],
            "total_errors": agg["total_errors"],
            "cold_starts": agg["cold_starts"],
            "total_prompt_tokens": agg["total_prompt_tokens"],
            "total_completion_tokens": agg["total_completion_tokens"],
            "avg_duration_ms": round(agg["total_duration_ms"] / max(agg["total_requests"], 1), 1),
        }

        if ttft_vals:
            sorted_ttft = sorted(ttft_vals)
            entry["avg_ttft_ms"] = round(sum(sorted_ttft) / len(sorted_ttft), 1)
            entry["p50_ttft_ms"] = round(sorted_ttft[len(sorted_ttft) // 2], 1)
            entry["p95_ttft_ms"] = round(sorted_ttft[int(len(sorted_ttft) * 0.95)], 1)

        if tps_vals:
            sorted_tps = sorted(tps_vals)
            entry["avg_tps"] = round(sum(sorted_tps) / len(sorted_tps), 1)
            entry["p50_tps"] = round(sorted_tps[len(sorted_tps) // 2], 1)

        result[model] = entry
    return result
