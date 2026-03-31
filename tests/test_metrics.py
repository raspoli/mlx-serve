"""Tests for the metrics collection system."""

from pathlib import Path

import mlx_serve.metrics as met_mod
from mlx_serve.metrics import (
    RequestMetrics,
    check_memory_pressure,
    configure,
    get_aggregates,
    get_current_memory,
    get_metal_memory,
    get_request_history,
    record_request,
)


def _make_request(model: str = "test-model", **kwargs) -> RequestMetrics:
    defaults = dict(
        request_id="req-1",
        model=model,
        endpoint="/v1/chat/completions",
        timestamp="2026-01-01T00:00:00Z",
        started_at=0.0,
        total_duration_ms=1000.0,
        ttft_ms=200.0,
        tokens_per_second=40.0,
        prompt_tokens=100,
        completion_tokens=50,
        status_code=200,
    )
    defaults.update(kwargs)
    return RequestMetrics(**defaults)


def test_record_and_query(log_dir: Path):
    configure(log_dir, requests_max=100)
    met_mod._requests.clear()
    met_mod._aggregates.clear()

    record_request(_make_request(request_id="r1"))
    record_request(_make_request(request_id="r2", model="other"))
    record_request(_make_request(request_id="r3"))

    # All requests
    history = get_request_history(last_n=10)
    assert len(history) == 3

    # Filter by model
    filtered = get_request_history(model="test-model", last_n=10)
    assert len(filtered) == 2

    # Newest first
    assert history[0]["request_id"] == "r3"


def test_aggregates(log_dir: Path):
    configure(log_dir, requests_max=100)
    met_mod._requests.clear()
    met_mod._aggregates.clear()

    record_request(_make_request(ttft_ms=100.0, tokens_per_second=30.0))
    record_request(_make_request(ttft_ms=200.0, tokens_per_second=50.0))
    record_request(_make_request(ttft_ms=300.0, tokens_per_second=40.0, error="timeout"))

    aggs = get_aggregates()
    assert "test-model" in aggs
    m = aggs["test-model"]
    assert m["total_requests"] == 3
    assert m["total_errors"] == 1
    assert m["avg_ttft_ms"] == 200.0
    assert m["avg_tps"] == 40.0


def test_cold_start_tracking(log_dir: Path):
    configure(log_dir, requests_max=100)
    met_mod._requests.clear()
    met_mod._aggregates.clear()

    record_request(_make_request(cold_start=True))
    record_request(_make_request(cold_start=False))

    aggs = get_aggregates()
    assert aggs["test-model"]["cold_starts"] == 1


def test_requests_deque_bounded(log_dir: Path):
    configure(log_dir, requests_max=5)
    met_mod._requests.clear()
    met_mod._aggregates.clear()

    for i in range(10):
        record_request(_make_request(request_id=f"r{i}"))

    assert len(met_mod._requests) == 5


def test_memory_pressure():
    pressure = check_memory_pressure()
    assert pressure in ("nominal", "warn", "critical")


def test_metal_memory_returns_dict():
    metal = get_metal_memory()
    assert "active_mb" in metal
    assert "peak_mb" in metal
    assert "cache_mb" in metal


def test_current_memory_snapshot():
    snap = get_current_memory()
    assert "ram_total_gb" in snap
    assert "ram_used_gb" in snap
    assert "ram_percent" in snap
    assert snap["ram_total_gb"] > 0


def test_requests_written_to_jsonl(log_dir: Path):
    configure(log_dir, requests_max=100)
    met_mod._requests.clear()
    met_mod._aggregates.clear()

    record_request(_make_request())

    jsonl = log_dir / "requests.jsonl"
    assert jsonl.exists()

    import json

    data = json.loads(jsonl.read_text().strip())
    assert data["model"] == "test-model"
    assert "started_at" not in data  # internal field excluded
