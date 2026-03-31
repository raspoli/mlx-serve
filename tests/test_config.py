"""Tests for config discovery and parsing."""

import os
from pathlib import Path

import pytest


def test_config_discovery_cwd(tmp_path, monkeypatch):
    """Config is found in the current working directory."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 9091
manager_port: 9095
inactivity_timeout_seconds: 300
startup_timeout_seconds: 60
models: []
""")
    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)

    from mlx_serve.config import _find_config

    found = _find_config()
    assert found == config_file


def test_config_discovery_env_var(tmp_path, monkeypatch):
    """MLX_SERVE_CONFIG env var takes priority."""
    config_file = tmp_path / "custom.yaml"
    config_file.write_text("mlx_port: 9091\nmanager_port: 9095\nmodels: []\n")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    from mlx_serve.config import _find_config

    assert _find_config() == config_file


def test_config_discovery_env_var_missing(tmp_path, monkeypatch):
    """MLX_SERVE_CONFIG pointing to missing file raises FileNotFoundError."""
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(tmp_path / "nope.yaml"))

    from mlx_serve.config import _find_config

    with pytest.raises(FileNotFoundError, match="MLX_SERVE_CONFIG"):
        _find_config()


def test_config_discovery_user_dir(tmp_path, monkeypatch):
    """Config is found in ~/.mlx-serve/models.yaml."""
    fake_home = tmp_path / "home"
    config_dir = fake_home / ".mlx-serve"
    config_dir.mkdir(parents=True)
    config_file = config_dir / "models.yaml"
    config_file.write_text("mlx_port: 9091\nmanager_port: 9095\nmodels: []\n")

    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no models.yaml in CWD
    monkeypatch.setenv("HOME", str(fake_home))

    from mlx_serve.config import _find_config

    assert _find_config() == config_file


def test_config_discovery_bundled_default(tmp_path, monkeypatch):
    """Falls back to bundled _default_models.yaml."""
    monkeypatch.delenv("MLX_SERVE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)  # no models.yaml here
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))  # no ~/.mlx-serve/

    from mlx_serve.config import _find_config

    found = _find_config()
    assert found.name == "_default_models.yaml"


def test_config_parses_models(tmp_path, monkeypatch):
    """Config correctly parses model entries."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 8091
manager_port: 8095
inactivity_timeout_seconds: 600
startup_timeout_seconds: 120
models:
  - name: my-model
    type: text
    hf_path: mlx-community/test
    context_length: 4096
    max_kv_cache_size: 8192
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert "my-model" in cfg.MODELS
    assert cfg.MODELS["my-model"].type == "text"
    assert cfg.MODELS["my-model"].hf_path == "mlx-community/test"
    assert cfg.MODELS["my-model"].context_length == 4096
    assert cfg.MODELS["my-model"].max_kv_cache_size == 8192
    assert cfg.MLX_PORT == 8091
    assert cfg.MANAGER_PORT == 8095


def test_config_invalid_type(tmp_path, monkeypatch):
    """Invalid model type raises ValueError."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
models:
  - name: bad
    type: invalid
    hf_path: test/test
""")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    with pytest.raises(ValueError, match="invalid type"):
        importlib.reload(cfg)


def test_config_monitoring_defaults(tmp_path, monkeypatch):
    """Monitoring config uses sensible defaults when not specified."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("models: []\n")
    monkeypatch.setenv("MLX_SERVE_CONFIG", str(config_file))

    import importlib
    import mlx_serve.config as cfg

    importlib.reload(cfg)

    assert cfg.MONITORING.metrics_history_size == 500
    assert cfg.MONITORING.events_history_size == 1000
    assert cfg.MONITORING.memory_sample_interval == 10
