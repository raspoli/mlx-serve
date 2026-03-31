"""Tests for the event logging system."""

from pathlib import Path

import mlx_serve.events as ev_mod
from mlx_serve.events import EventType, configure, emit, get_events


def _clear():
    ev_mod._events.clear()


def test_emit_and_query(log_dir: Path):
    configure(log_dir, max_events=100)
    _clear()

    emit(EventType.MODEL_LOADING, model="test-model", detail={"hf_path": "test/path"})
    emit(EventType.MODEL_READY, model="test-model", duration_ms=5000.0)
    emit(EventType.MODEL_FAILED, model="other-model", detail={"reason": "oom"})

    # All events
    all_events = get_events(last_n=10)
    assert len(all_events) == 3

    # Filter by model
    test_events = get_events(model="test-model")
    assert len(test_events) == 2

    # Filter by type
    failed = get_events(event_type="model.failed")
    assert len(failed) == 1
    assert failed[0]["model"] == "other-model"
    assert failed[0]["detail"]["reason"] == "oom"


def test_event_has_timestamp(log_dir: Path):
    configure(log_dir, max_events=100)
    _clear()

    e = emit(EventType.SERVER_START, detail={"port": 8095})
    assert e.timestamp is not None
    assert "T" in e.timestamp  # ISO format


def test_events_written_to_jsonl(log_dir: Path):
    configure(log_dir, max_events=100)
    _clear()

    emit(EventType.MODEL_LOADING, model="m1")
    emit(EventType.MODEL_READY, model="m1", duration_ms=1000.0)

    jsonl = log_dir / "events.jsonl"
    assert jsonl.exists()
    lines = jsonl.read_text().strip().split("\n")
    assert len(lines) == 2

    import json

    first = json.loads(lines[0])
    assert first["event"] == "model.loading"
    assert first["model"] == "m1"


def test_events_deque_bounded(log_dir: Path):
    configure(log_dir, max_events=5)
    _clear()

    for i in range(10):
        emit(EventType.MODEL_LOADING, model=f"m{i}")

    assert len(ev_mod._events) == 5
    # Newest events kept
    all_events = get_events(last_n=10)
    assert all_events[0]["model"] == "m9"  # newest first


def test_events_newest_first(log_dir: Path):
    configure(log_dir, max_events=100)
    _clear()

    emit(EventType.MODEL_LOADING, model="first")
    emit(EventType.MODEL_READY, model="second")

    results = get_events(last_n=10)
    assert results[0]["model"] == "second"
    assert results[1]["model"] == "first"
