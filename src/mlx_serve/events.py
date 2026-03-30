"""
events.py — structured lifecycle event log.

Every model lifecycle transition is recorded as a ModelEvent and:
  1. Stored in an in-memory deque (queryable via /v1/events)
  2. Appended to a JSONL file on disk (survives restarts)
  3. Logged to Python logging (appears in terminal)
"""
import json
import logging
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger("mlx-serve.events")

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(str, Enum):
    # Server
    SERVER_START = "server.start"
    SERVER_SHUTDOWN = "server.shutdown"

    # Model lifecycle
    MODEL_LOADING = "model.loading"
    MODEL_READY = "model.ready"
    MODEL_FAILED = "model.failed"
    MODEL_UNLOADED = "model.unloaded"
    MODEL_SWITCH_START = "model.switch.start"
    MODEL_SWITCH_COMPLETE = "model.switch.complete"
    MODEL_INACTIVITY_UNLOAD = "model.inactivity_unload"

    # Subprocess
    SUBPROCESS_SPAWNED = "subprocess.spawned"
    SUBPROCESS_CRASH = "subprocess.crash"

    # System
    OOM_DETECTED = "oom.detected"
    MEMORY_PRESSURE_WARN = "memory.pressure.warn"
    MEMORY_PRESSURE_CRITICAL = "memory.pressure.critical"
    SWAP_ACTIVE = "swap.active"

    # Requests
    REQUEST_COLD_START = "request.cold_start"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ModelEvent:
    timestamp: str
    event: str
    model: str | None = None
    detail: dict = field(default_factory=dict)
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_events: deque[ModelEvent] = deque(maxlen=1000)
_log_file: Path | None = None


def configure(log_dir: Path, max_events: int = 1000) -> None:
    """Set the JSONL file path and resize the deque."""
    global _events, _log_file
    log_dir.mkdir(parents=True, exist_ok=True)
    _log_file = log_dir / "events.jsonl"
    _events = deque(_events, maxlen=max_events)


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------

_LEVEL_MAP = {
    EventType.MODEL_FAILED: logging.ERROR,
    EventType.SUBPROCESS_CRASH: logging.ERROR,
    EventType.OOM_DETECTED: logging.ERROR,
    EventType.MEMORY_PRESSURE_CRITICAL: logging.WARNING,
    EventType.MEMORY_PRESSURE_WARN: logging.WARNING,
    EventType.SWAP_ACTIVE: logging.WARNING,
}


def emit(
    event_type: EventType,
    model: str | None = None,
    detail: dict[str, Any] | None = None,
    duration_ms: float | None = None,
) -> ModelEvent:
    """Record a lifecycle event."""
    ev = ModelEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        event=event_type.value,
        model=model,
        detail=detail or {},
        duration_ms=duration_ms,
    )
    _events.append(ev)

    # Log to terminal
    level = _LEVEL_MAP.get(event_type, logging.INFO)
    parts = [ev.event]
    if model:
        parts.append(f"model={model}")
    if duration_ms is not None:
        parts.append(f"duration={duration_ms:.0f}ms")
    if detail:
        for k, v in detail.items():
            if isinstance(v, list):
                continue  # skip long stderr tails in terminal
            parts.append(f"{k}={v}")
    logger.log(level, " | ".join(parts))

    # Append to JSONL file
    if _log_file is not None:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(ev), default=str) + "\n")
        except OSError:
            pass  # don't crash the server over a log write

    return ev


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def get_events(
    model: str | None = None,
    event_type: str | None = None,
    last_n: int = 100,
) -> list[dict]:
    """Return recent events, newest first, with optional filters."""
    results = []
    for ev in reversed(_events):
        if model and ev.model != model:
            continue
        if event_type and ev.event != event_type:
            continue
        results.append(asdict(ev))
        if len(results) >= last_n:
            break
    return results
