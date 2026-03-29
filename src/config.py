"""
config.py — reads models.yaml and exposes typed configuration.
"""
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

# models.yaml lives in the project root, one level above src/
_CONFIG_PATH = Path(__file__).parent.parent / "models.yaml"


@dataclass
class ModelConfig:
    name: str
    type: str                    # "text", "vision", "embedding", "tts", or "stt"
    hf_path: str
    context_length: int = 0      # max output tokens per response (--max-tokens); 0 = server default
    max_kv_cache_size: int = 0   # KV cache token capacity for prompt caching (--max-kv-cache-size); 0 = model default


_VALID_TYPES = {"text", "vision", "embedding", "tts", "stt"}


def _load() -> tuple[dict[str, ModelConfig], int, int, int, int]:
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
    return (
        models,
        data.get("mlx_port", 8091),
        data.get("manager_port", 8090),
        data.get("inactivity_timeout_seconds", 600),
        data.get("startup_timeout_seconds", 120),
    )


(
    MODELS,
    MLX_PORT,
    MANAGER_PORT,
    INACTIVITY_TIMEOUT,
    STARTUP_TIMEOUT,
) = _load()

# Optional bearer token auth. Set MLX_API_KEY env var to enable.
# When empty, all requests are accepted (safe for localhost-only deployments).
API_KEY: str = os.environ.get("MLX_API_KEY", "")
