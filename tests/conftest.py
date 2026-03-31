"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_config(tmp_path: Path):
    """Create a temporary models.yaml and point MLX_SERVE_CONFIG at it."""
    config_file = tmp_path / "models.yaml"
    config_file.write_text("""
mlx_port: 18091
manager_port: 18095
inactivity_timeout_seconds: 60
startup_timeout_seconds: 10

models:
  - name: test-text-model
    type: text
    hf_path: mlx-community/test-text
    context_length: 4096

  - name: test-vision-model
    type: vision
    hf_path: mlx-community/test-vision

  - name: test-embedding-model
    type: embedding
    hf_path: mlx-community/test-embedding

  - name: test-tts-model
    type: tts
    hf_path: mlx-community/test-tts

  - name: test-stt-model
    type: stt
    hf_path: mlx-community/test-stt
""")
    os.environ["MLX_SERVE_CONFIG"] = str(config_file)
    yield config_file
    os.environ.pop("MLX_SERVE_CONFIG", None)


@pytest.fixture()
def log_dir(tmp_path: Path):
    """Provide a temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d
