"""
config.py — reads models.yaml and exposes typed configuration.

Config discovery order:
  1. MLX_SERVE_CONFIG env var (explicit override)
  2. ./models.yaml  (current working directory)
  3. ~/.mlx-serve/models.yaml  (user config directory)
  4. Bundled _default_models.yaml inside the package
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml


def _find_config() -> Path:
    """Locate models.yaml using a fallback chain."""
    # 1. Explicit env var
    env_path = os.environ.get("MLX_SERVE_CONFIG")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
        raise FileNotFoundError(f"MLX_SERVE_CONFIG points to {p} which does not exist")

    # 2. Current working directory
    cwd_path = Path.cwd() / "models.yaml"
    if cwd_path.exists():
        return cwd_path

    # 3. User config directory
    user_path = Path.home() / ".mlx-serve" / "models.yaml"
    if user_path.exists():
        return user_path

    # 4. Bundled default
    bundled = Path(__file__).parent / "_default_models.yaml"
    if bundled.exists():
        return bundled

    raise FileNotFoundError(
        "No models.yaml found. Searched:\n"
        f"  - {cwd_path}\n"
        f"  - {user_path}\n"
        "Run 'mlx-serve init' to generate one."
    )


_CONFIG_PATH = _find_config()


@dataclass
class ModelConfig:
    name: str
    type: str                    # "text", "vision", "embedding", "tts", or "stt"
    hf_path: str
    context_length: int = 0      # max output tokens per response (--max-tokens); 0 = server default
    max_kv_cache_size: int = 0   # KV cache token capacity for prompt caching (--max-kv-cache-size); 0 = model default


@dataclass
class MonitoringConfig:
    log_dir: Path
    metrics_history_size: int = 500
    events_history_size: int = 1000
    memory_sample_interval: int = 10     # seconds, in-memory
    memory_log_interval: int = 60        # seconds, to disk
    log_retention_mb: int = 50           # per JSONL file


_VALID_TYPES = {"text", "vision", "embedding", "tts", "stt"}


def _load() -> tuple[dict[str, ModelConfig], int, int, int, int, MonitoringConfig]:
    with open(_CONFIG_PATH) as f:
        data = yaml.safe_load(f)

    models = {}
    for entry in data.get("models", []):
        if entry["type"] not in _VALID_TYPES:
            raise ValueError(
                f"Model '{entry['name']}' has invalid type '{entry['type']}'. "
                f"Must be one of: {sorted(_VALID_TYPES)}"
            )
        models[entry["name"]] = ModelConfig(
            name=entry["name"],
            type=entry["type"],
            hf_path=entry["hf_path"],
            context_length=entry.get("context_length", 0),
            max_kv_cache_size=entry.get("max_kv_cache_size", 0),
        )

    # Monitoring settings (optional section in models.yaml)
    mon_raw = data.get("monitoring", {})
    default_log_dir = Path.home() / ".mlx-serve" / "logs"
    log_dir_str = mon_raw.get("log_dir", str(default_log_dir))
    monitoring = MonitoringConfig(
        log_dir=Path(log_dir_str).expanduser(),
        metrics_history_size=mon_raw.get("metrics_history_size", 500),
        events_history_size=mon_raw.get("events_history_size", 1000),
        memory_sample_interval=mon_raw.get("memory_sample_interval", 10),
        memory_log_interval=mon_raw.get("memory_log_interval", 60),
        log_retention_mb=mon_raw.get("log_retention_mb", 50),
    )

    return (
        models,
        data.get("mlx_port", 8091),
        data.get("manager_port", 8095),
        data.get("inactivity_timeout_seconds", 600),
        data.get("startup_timeout_seconds", 120),
        monitoring,
    )


(
    MODELS,
    MLX_PORT,
    MANAGER_PORT,
    INACTIVITY_TIMEOUT,
    STARTUP_TIMEOUT,
    MONITORING,
) = _load()

# Optional bearer token auth. Set MLX_API_KEY env var to enable.
# When empty, all requests are accepted (safe for localhost-only deployments).
API_KEY: str = os.environ.get("MLX_API_KEY", "")
