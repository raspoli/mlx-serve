"""
inline_manager.py — in-process model lifecycle for embedding, TTS, and STT types.

Unlike text/vision models (which spawn a subprocess), these model types are
loaded directly into the FastAPI process via mlx-embeddings, mlx-audio, and
mlx-whisper respectively.
"""

import asyncio
import gc
import io
import logging
import pathlib
import tempfile
import time
import wave
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

from . import config, events, metrics

logger = logging.getLogger("mlx-serve.inline")


class InlineModelState(Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


_state: InlineModelState = InlineModelState.IDLE
_active_model: str | None = None
_model: Any = None
_processor: Any = None
_last_request_at: datetime | None = None
_load_lock: asyncio.Lock = asyncio.Lock()
_inactivity_timeout: int = config.INACTIVITY_TIMEOUT  # overridable per-request


# ---------------------------------------------------------------------------
# Internal loaders (blocking — run in executor)
# ---------------------------------------------------------------------------


_INSTALL_HINTS = {
    "embedding": "pip install mlx-serve[embeddings]",
    "tts": "pip install mlx-serve[tts]",
    "stt": "pip install mlx-serve[stt]",
}


def _load_embedding(hf_path: str) -> tuple[Any, Any]:
    try:
        from mlx_embeddings import load
    except ImportError:
        raise RuntimeError(f"mlx-embeddings is not installed. Run: {_INSTALL_HINTS['embedding']}")
    return load(hf_path)  # returns (model, processor)


def _load_tts(hf_path: str) -> tuple[Any, None]:
    try:
        from mlx_audio.tts.utils import load_model
    except ImportError:
        raise RuntimeError(f"mlx-audio is not installed. Run: {_INSTALL_HINTS['tts']}")
    return load_model(hf_path), None


def _load_stt(hf_path: str) -> tuple[str, None]:
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        raise RuntimeError(f"mlx-whisper is not installed. Run: {_INSTALL_HINTS['stt']}")
    return hf_path, None


# ---------------------------------------------------------------------------
# Inference helpers (blocking — run in executor)
# ---------------------------------------------------------------------------


def _run_embedding(model: Any, processor: Any, inputs: list[dict]) -> list[list[float]]:
    """
    inputs: list of dicts, each with optional "text" and/or "image" keys.
    e.g. [{"text": "hello"}, {"image": "/path/img.jpg"}, {"text": "cap", "image": "url"}]
    """
    import mlx.core as mx

    embeddings = model.process(inputs, processor=processor)
    mx.eval(embeddings)
    return embeddings.tolist()


def _run_tts(model: Any, text: str, speed: float, lang_code: str) -> bytes:
    import numpy as np

    chunks = []
    for result in model.generate(text=text, speed=speed, lang_code=lang_code):
        chunks.append(np.array(result.audio).flatten())

    audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)

    # Encode as 16-bit PCM WAV (24 kHz mono — chatterbox native sample rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _run_stt(hf_path: str, audio_bytes: bytes, language: str | None) -> str:
    import mlx_whisper

    tmp = pathlib.Path(tempfile.mktemp(suffix=".wav"))
    try:
        tmp.write_bytes(audio_bytes)
        kwargs: dict[str, Any] = {"path_or_hf_repo": hf_path}
        if language:
            kwargs["language"] = language
        result = mlx_whisper.transcribe(str(tmp), **kwargs)
        return result.get("text", "").strip()
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def ensure_model(model_name: str) -> tuple[Any, Any]:
    """Return (model, processor), loading it if necessary. Returns cold_start flag via side effect."""
    global _state, _active_model, _model, _processor, _last_request_at

    if _active_model == model_name and _state == InlineModelState.READY:
        _last_request_at = datetime.now(UTC)
        return _model, _processor

    async with _load_lock:
        if _active_model == model_name and _state == InlineModelState.READY:
            _last_request_at = datetime.now(UTC)
            return _model, _processor

        prev_model = _active_model

        if prev_model is not None:
            events.emit(
                events.EventType.MODEL_SWITCH_START,
                model=model_name,
                detail={"from_model": prev_model, "to_model": model_name},
            )
            logger.info(f"Switching inline model: {prev_model} -> {model_name}")

        _unload_unsafe()

        model_cfg = config.MODELS[model_name]
        _state = InlineModelState.LOADING
        _active_model = model_name
        load_start = time.monotonic()

        events.emit(
            events.EventType.MODEL_LOADING,
            model=model_name,
            detail={"hf_path": model_cfg.hf_path, "type": model_cfg.type, "in_process": True},
        )
        logger.info(f"Loading inline model {model_name} ({model_cfg.hf_path})")

        # Snapshot before loading to capture baseline
        metrics.take_memory_snapshot(active_model=model_name, event="model_loading")

        loop = asyncio.get_running_loop()
        try:
            if model_cfg.type == "embedding":
                _model, _processor = await loop.run_in_executor(
                    None, _load_embedding, model_cfg.hf_path
                )
            elif model_cfg.type == "tts":
                _model, _processor = await loop.run_in_executor(None, _load_tts, model_cfg.hf_path)
            elif model_cfg.type == "stt":
                _model, _processor = await loop.run_in_executor(None, _load_stt, model_cfg.hf_path)
            _state = InlineModelState.READY
        except Exception as exc:
            _state = InlineModelState.FAILED
            load_ms = (time.monotonic() - load_start) * 1000
            _active_model = None

            # Diagnose the failure
            reason = "load_error"
            exc_str = str(exc).lower()
            if any(kw in exc_str for kw in ("memoryerror", "out of memory", "malloc")):
                reason = "oom"
                events.emit(
                    events.EventType.OOM_DETECTED, model=model_name, detail={"error": str(exc)}
                )
            events.emit(
                events.EventType.MODEL_FAILED,
                model=model_name,
                detail={"reason": reason, "error": str(exc)},
                duration_ms=load_ms,
            )
            logger.error(f"Failed to load {model_name}: {exc}")
            raise RuntimeError(f"Failed to load {model_name}: {exc}") from exc

        load_ms = (time.monotonic() - load_start) * 1000

        # Snapshot after load to show memory consumed by model
        snap = metrics.take_memory_snapshot(active_model=model_name, event="model_loaded")
        events.emit(
            events.EventType.MODEL_READY,
            model=model_name,
            detail={
                "ram_used_gb": snap.ram_used_gb,
                "metal_active_mb": snap.metal_active_mb,
                "metal_peak_mb": snap.metal_peak_mb,
            },
            duration_ms=load_ms,
        )
        events.emit(
            events.EventType.MODEL_SWITCH_COMPLETE,
            model=model_name,
            duration_ms=load_ms,
        )
        logger.info(
            f"Model {model_name} ready in {load_ms:.0f}ms | "
            f"RAM={snap.ram_used_gb}GB | Metal={snap.metal_active_mb}MB"
        )

        _last_request_at = datetime.now(UTC)
        return _model, _processor


def _unload_unsafe() -> None:
    """Zero out model state. Must only be called while holding _load_lock."""
    global _state, _active_model, _model, _processor, _inactivity_timeout
    prev_model = _active_model
    _model = None
    _processor = None
    _state = InlineModelState.IDLE
    _active_model = None
    _inactivity_timeout = config.INACTIVITY_TIMEOUT  # reset per-request override
    gc.collect()

    if prev_model is not None:
        snap = metrics.take_memory_snapshot(active_model=None, event="model_unloaded")
        events.emit(
            events.EventType.MODEL_UNLOADED,
            model=prev_model,
            detail={
                "ram_used_gb": snap.ram_used_gb,
                "ram_available_gb": snap.ram_available_gb,
                "metal_active_mb": snap.metal_active_mb,
            },
        )


async def unload() -> None:
    """Unload the active inline model. Safe to call from outside the load path."""
    async with _load_lock:
        _unload_unsafe()


def set_keep_alive(seconds: int) -> None:
    """Override the inactivity timeout for the current model session."""
    global _inactivity_timeout
    _inactivity_timeout = seconds


async def generate_embeddings(model_name: str, inputs: list[dict]) -> list[list[float]]:
    """
    inputs: list of dicts with "text" and/or "image" keys.
    Text-only example:  [{"text": "hello"}, {"text": "world"}]
    Multimodal example: [{"text": "a dog", "image": "/path/img.jpg"}]
    """
    model, processor = await ensure_model(model_name)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_embedding, model, processor, inputs)


async def generate_tts(
    model_name: str, text: str, speed: float = 1.0, lang_code: str = "en"
) -> bytes:
    model, _ = await ensure_model(model_name)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_tts, model, text, speed, lang_code)


async def generate_stt(model_name: str, audio_bytes: bytes, language: str | None = None) -> str:
    """Transcribe audio bytes to text using an STT model."""
    hf_path, _ = await ensure_model(model_name)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _run_stt, hf_path, audio_bytes, language)


async def start_inactivity_watcher() -> None:
    """Background task: unload idle inline model every 30 s."""
    while True:
        await asyncio.sleep(30)
        if (
            _state == InlineModelState.READY
            and _last_request_at is not None
            and datetime.now(UTC) - _last_request_at > timedelta(seconds=_inactivity_timeout)
        ):
            idle_secs = int((datetime.now(UTC) - _last_request_at).total_seconds())
            events.emit(
                events.EventType.MODEL_INACTIVITY_UNLOAD,
                model=_active_model,
                detail={"idle_seconds": idle_secs},
            )
            logger.info(f"Inline model {_active_model} idle for {idle_secs}s, unloading")
            await unload()


def get_status() -> dict:
    unload_at = None
    if (
        _last_request_at is not None
        and _inactivity_timeout > 0
        and _state == InlineModelState.READY
    ):
        unload_at = (_last_request_at + timedelta(seconds=_inactivity_timeout)).isoformat()
    return {
        "active_model": _active_model,
        "state": _state.value,
        "last_request_at": _last_request_at.isoformat() if _last_request_at else None,
        "unload_at": unload_at,
        "inactivity_timeout_seconds": _inactivity_timeout,
    }


def get_active_model_name() -> str | None:
    """Return the name of the currently active inline model."""
    return _active_model
