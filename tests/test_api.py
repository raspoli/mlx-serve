"""Tests for the FastAPI endpoints (no real MLX models needed)."""

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path: Path):
    """Create a test client with a temporary config."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 18091
manager_port: 18095
inactivity_timeout_seconds: 60
startup_timeout_seconds: 10
models:
  - name: test-text
    type: text
    hf_path: mlx-community/test-text
    context_length: 4096
  - name: test-embed
    type: embedding
    hf_path: mlx-community/test-embed
""")
    os.environ["MLX_SERVE_CONFIG"] = str(config_file)

    # Force reimport with new config
    import importlib
    import mlx_serve.config

    importlib.reload(mlx_serve.config)

    from mlx_serve.main import app

    with TestClient(app) as c:
        yield c

    os.environ.pop("MLX_SERVE_CONFIG", None)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_health_v1(client):
    resp = client.get("/v1/health")
    assert resp.status_code == 200


def test_version(client):
    resp = client.get("/v1/version")
    assert resp.status_code == 200
    assert "version" in resp.json()


def test_list_models(client):
    resp = client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    model_ids = [m["id"] for m in data["data"]]
    assert "test-text" in model_ids
    assert "test-embed" in model_ids


def test_get_model_not_found(client):
    resp = client.get("/v1/models/nonexistent")
    assert resp.status_code == 404


def test_status(client):
    resp = client.get("/status")
    assert resp.status_code == 200
    body = resp.json()
    assert "subprocess" in body
    assert "inline" in body
    assert "memory" in body


def test_metrics_endpoint(client):
    resp = client.get("/v1/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert "memory" in body
    assert "models" in body
    assert "pressure" in body


def test_metrics_requests_endpoint(client):
    resp = client.get("/v1/metrics/requests")
    assert resp.status_code == 200
    assert "requests" in resp.json()


def test_metrics_memory_endpoint(client):
    resp = client.get("/v1/metrics/memory")
    assert resp.status_code == 200
    body = resp.json()
    assert "ram_total_gb" in body


def test_events_endpoint(client):
    resp = client.get("/v1/events")
    assert resp.status_code == 200
    assert "events" in resp.json()


def test_dashboard_endpoint(client):
    resp = client.get("/v1/dashboard")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "memory" in body
    assert "recent_events" in body


def test_chat_completions_model_not_found(client):
    resp = client.post("/v1/chat/completions", json={"model": "nope", "messages": []})
    assert resp.status_code == 404


def test_chat_completions_wrong_type(client):
    resp = client.post("/v1/chat/completions", json={"model": "test-embed", "messages": []})
    assert resp.status_code == 404
    assert "use the correct endpoint" in resp.json()["detail"]["error"]["message"]


def test_embeddings_model_not_found(client):
    resp = client.post("/v1/embeddings", json={"model": "nope", "input": "hello"})
    assert resp.status_code == 404


def test_unload_model(client):
    resp = client.post("/v1/models/unload", json={})
    assert resp.status_code == 200
    assert resp.json()["status"] == "unloaded"
