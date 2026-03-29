"""
process_manager.py — subprocess lifecycle for the active mlx model.

State machine: IDLE → LOADING → READY / FAILED → IDLE (on inactivity/shutdown)
"""
import asyncio
import os
import pathlib
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from enum import Enum

import httpx

import config

# Resolve executable paths relative to the running venv so subprocesses
# inherit the same environment that has mlx_lm / mlx_vlm installed.
_VENV_BIN = os.path.dirname(sys.executable)
_MLX_LM_SERVER = os.path.join(_VENV_BIN, "mlx_lm.server")
_MLX_VLM_SERVER = os.path.join(_VENV_BIN, "mlx_vlm.server")

_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "mlx-manager-logs"

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class ModelState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


_state: ModelState = ModelState.IDLE
_active_model: str | None = None
_process: subprocess.Popen | None = None
_ready_event: asyncio.Event = asyncio.Event()
_switch_lock: asyncio.Lock = asyncio.Lock()
_last_request_at: datetime | None = None
_started_at: datetime = datetime.now(timezone.utc)
_stderr_log_path: pathlib.Path | None = None
_stderr_log_handle = None
_inactivity_timeout: int = config.INACTIVITY_TIMEOUT  # overridable per-request


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_command(model_cfg: config.ModelConfig) -> list[str]:
    executable = _MLX_VLM_SERVER if model_cfg.type == "vision" else _MLX_LM_SERVER
    cmd = [
        executable,
        "--model", model_cfg.hf_path,
        "--host", "127.0.0.1",
        "--port", str(config.MLX_PORT),
    ]
    if model_cfg.context_length > 0:
        cmd += ["--max-tokens", str(model_cfg.context_length)]
    if model_cfg.max_kv_cache_size > 0:
        cmd += ["--max-kv-cache-size", str(model_cfg.max_kv_cache_size)]
    return cmd


async def _terminate_current() -> None:
    global _process, _state, _active_model, _stderr_log_handle, _inactivity_timeout
    if _process is not None:
        _process.terminate()
        try:
            await asyncio.wait_for(
                asyncio.get_running_loop().run_in_executor(None, _process.wait),
                timeout=5,
            )
        except asyncio.TimeoutError:
            _process.kill()
            await asyncio.get_running_loop().run_in_executor(None, _process.wait)
        _process = None
    if _stderr_log_handle is not None:
        _stderr_log_handle.close()
        _stderr_log_handle = None
    _state = ModelState.IDLE
    _active_model = None
    _inactivity_timeout = config.INACTIVITY_TIMEOUT  # reset per-request override


async def _health_check_loop() -> None:
    """Poll /health every 2s until READY or startup timeout."""
    global _state
    deadline = datetime.now(timezone.utc) + timedelta(seconds=config.STARTUP_TIMEOUT)
    async with httpx.AsyncClient() as client:
        while datetime.now(timezone.utc) < deadline:
            # Detect early process exit
            if _process is not None and _process.poll() is not None:
                _state = ModelState.FAILED
                _ready_event.set()
                if _stderr_log_path:
                    print(f"[mlx-manager] Model failed to load. Logs: {_stderr_log_path}", flush=True)
                return
            try:
                resp = await client.get(
                    f"http://127.0.0.1:{config.MLX_PORT}/health",
                    timeout=2,
                )
                if resp.status_code == 200:
                    _state = ModelState.READY
                    _ready_event.set()
                    return
            except httpx.RequestError:
                pass
            await asyncio.sleep(2)

    # Timed out
    _state = ModelState.FAILED
    _ready_event.set()
    if _stderr_log_path:
        print(f"[mlx-manager] Model load timed out. Logs: {_stderr_log_path}", flush=True)


async def _switch_model(model_name: str) -> None:
    global _process, _state, _active_model, _stderr_log_path, _stderr_log_handle

    async with _switch_lock:
        # Another coroutine may have already loaded this model while we waited
        if _active_model == model_name and _state == ModelState.READY:
            return

        await _terminate_current()

        model_cfg = config.MODELS[model_name]
        _ready_event.clear()
        _state = ModelState.LOADING
        _active_model = model_name

        _LOG_DIR.mkdir(exist_ok=True)
        _stderr_log_path = _LOG_DIR / f"{model_name}.log"
        _stderr_log_handle = open(_stderr_log_path, "w")

        _process = subprocess.Popen(
            _build_command(model_cfg),
            stdout=subprocess.DEVNULL,
            stderr=_stderr_log_handle,
        )

        asyncio.create_task(_health_check_loop())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ensure_model(model_name: str) -> None:
    """Ensure the named model is READY before the caller proxies a request."""
    global _last_request_at

    if _active_model == model_name and _state == ModelState.READY:
        _last_request_at = datetime.now(timezone.utc)
        return

    if _active_model != model_name or _state in (ModelState.IDLE, ModelState.FAILED):
        await _switch_model(model_name)

    if _state == ModelState.LOADING:
        try:
            await asyncio.wait_for(_ready_event.wait(), timeout=config.STARTUP_TIMEOUT + 5)
        except asyncio.TimeoutError:
            pass

    if _state == ModelState.FAILED:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail=f"Model {model_name} failed to load")

    _last_request_at = datetime.now(timezone.utc)


def set_keep_alive(seconds: int) -> None:
    """Override the inactivity timeout for the current model session."""
    global _inactivity_timeout
    _inactivity_timeout = seconds


async def start_inactivity_watcher() -> None:
    """Background task: unload idle models every 30s check."""
    while True:
        await asyncio.sleep(30)
        if (
            _state == ModelState.READY
            and _last_request_at is not None
            and datetime.now(timezone.utc) - _last_request_at
            > timedelta(seconds=_inactivity_timeout)
        ):
            await _terminate_current()


async def unload() -> None:
    """Terminate the active subprocess and reset to IDLE."""
    await _terminate_current()


# ---------------------------------------------------------------------------
# Status accessors
# ---------------------------------------------------------------------------

def get_status() -> dict:
    unload_at = None
    if _last_request_at is not None and _inactivity_timeout > 0 and _state == ModelState.READY:
        unload_at = (_last_request_at + timedelta(seconds=_inactivity_timeout)).isoformat()
    return {
        "active_model": _active_model,
        "state": _state.value,
        "last_request_at": _last_request_at.isoformat() if _last_request_at else None,
        "unload_at": unload_at,
        "uptime_seconds": int((datetime.now(timezone.utc) - _started_at).total_seconds()),
        "inactivity_timeout_seconds": _inactivity_timeout,
    }
