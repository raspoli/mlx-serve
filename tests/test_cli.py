"""Tests for the CLI module."""

import subprocess
import sys
from pathlib import Path


def test_cli_help():
    """mlx-serve --help exits 0 and shows usage."""
    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mlx-serve" in result.stdout
    assert "start" in result.stdout


def test_cli_init(tmp_path: Path):
    """mlx-serve init creates a models.yaml."""
    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli", "init", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (tmp_path / "models.yaml").exists()
    content = (tmp_path / "models.yaml").read_text()
    assert "mlx_port" in content


def test_cli_init_no_overwrite(tmp_path: Path):
    """mlx-serve init refuses to overwrite without --force."""
    (tmp_path / "models.yaml").write_text("existing")

    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli", "init", "--dir", str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert (tmp_path / "models.yaml").read_text() == "existing"


def test_cli_init_force(tmp_path: Path):
    """mlx-serve init --force overwrites existing file."""
    (tmp_path / "models.yaml").write_text("old")

    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli", "init", "--dir", str(tmp_path), "--force"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    content = (tmp_path / "models.yaml").read_text()
    assert "mlx_port" in content


def test_cli_status_no_server():
    """mlx-serve status fails gracefully when no server is running."""
    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli", "status", "--port", "19999"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "cannot connect" in result.stderr.lower()


def test_cli_no_command():
    """mlx-serve with no command shows help and exits non-zero."""
    result = subprocess.run(
        [sys.executable, "-m", "mlx_serve.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
