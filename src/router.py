"""
router.py — OpenAI-compatible HTTP endpoints for the MLX model manager.
"""
import importlib.metadata
import os
import pathlib
import sys
import tempfile
import time
from typing import Any

import httpx
import psutil
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

import config
import inline_manager
import process_manager

_HTTP_CLIENT = httpx.AsyncClient(timeout=None)
_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "mlx-manager-logs"
_VENV_BIN = os.path.dirname(sys.executable)

# Maps model type → OpenAI-style capabilities array
_TYPE_CAPABILITIES: dict[str, list[str]] = {
    "text": ["completion"],
    "vision": ["completion", "vision"],
    "embedding": ["embedding"],
    "tts": ["audio_speech"],
    "stt": ["audio_transcription"],
}


async def close_client() -> None:
    """Close the shared HTTP client. Called from app lifespan on shutdown."""
    await _HTTP_CLIENT.aclose()


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(auto_error=False)


async def _require_auth(
    credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
) -> None:
    """Enforce bearer token auth when MLX_API_KEY is set; no-op otherwise."""
    if not config.API_KEY:
        return
    if credentials is None or credentials.credentials != config.API_KEY:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Unauthorized", "code": 401}},
        )


router = APIRouter(dependencies=[Depends(_require_auth)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_keep_alive(value: str | int | float | None) -> int | None:
    """
    Parse a keep_alive value into seconds.
    Accepts: 300 (int), 300.0 (float), "5m", "10s", "1h", "-1" (disable timeout).
    Returns None if value is absent or unparseable.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    value = str(value).strip()
    try:
        if value.endswith("m"):
            return int(value[:-1]) * 60
        if value.endswith("h"):
            return int(value[:-1]) * 3600
        if value.endswith("s"):
            return int(value[:-1])
        return int(value)
    except ValueError:
        return None


_FORWARDED_HEADERS = {"content-type", "authorization", "accept", "x-request-id"}


def _forward_headers(headers) -> dict:
    return {k: v for k, v in headers.items() if k.lower() in _FORWARDED_HEADERS}


def _model_detail(m: config.ModelConfig, cached: bool | None = None) -> dict:
    """Build the full model detail dict for a single ModelConfig."""
    entry: dict[str, Any] = {
        "id": m.name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "mlx-community",
        "type": m.type,
        "hf_path": m.hf_path,
        "capabilities": _TYPE_CAPABILITIES.get(m.type, []),
        "context_length": m.context_length,
        "max_kv_cache_size": m.max_kv_cache_size,
    }
    if cached is not None:
        entry["cached"] = cached
    return entry


def _get_memory_stats() -> dict:
    """Return unified memory pressure stats via psutil."""
    vm = psutil.virtual_memory()
    return {
        "total_gb": round(vm.total / 1_000_000_000, 1),
        "used_gb": round(vm.used / 1_000_000_000, 1),
        "available_gb": round(vm.available / 1_000_000_000, 1),
        "percent_used": vm.percent,
    }


def _scan_cached_hf_paths() -> set[str]:
    try:
        from huggingface_hub import scan_cache_dir
        return {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# GET /health  (also registered at bare /health in main.py — no auth there)
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict:
    """Liveness check. Returns 200 OK when the manager is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /version
# ---------------------------------------------------------------------------

@router.get("/version")
async def version() -> dict:
    """Returns the mlx-manager version."""
    try:
        ver = importlib.metadata.version("mlx-manager")
    except importlib.metadata.PackageNotFoundError:
        ver = "0.1.0"
    return {"version": ver}


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

@router.get("/models")
async def list_models() -> dict:
    """Return all models defined in models.yaml in OpenAI format."""
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community",
                "capabilities": _TYPE_CAPABILITIES.get(m.type, []),
            }
            for m in config.MODELS.values()
        ],
    }


# ---------------------------------------------------------------------------
# GET /models/local  — must be registered BEFORE /models/{model_id}
# ---------------------------------------------------------------------------

@router.get("/models/local")
async def list_local_models() -> dict:
    """Return only the configured models that are present in the HuggingFace cache."""
    cached = _scan_cached_hf_paths()
    return {
        "object": "list",
        "data": [
            {
                "id": m.name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mlx-community",
                "capabilities": _TYPE_CAPABILITIES.get(m.type, []),
                "cached": True,
            }
            for m in config.MODELS.values()
            if m.hf_path in cached
        ],
    }


# ---------------------------------------------------------------------------
# GET /models/{model_id}
# ---------------------------------------------------------------------------

@router.get("/models/{model_id}")
async def get_model(model_id: str) -> dict:
    """Return full detail for a single configured model including cache status."""
    if model_id not in config.MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model not found: {model_id}", "code": 404}},
        )
    m = config.MODELS[model_id]
    cached_paths = _scan_cached_hf_paths()
    return _model_detail(m, cached=m.hf_path in cached_paths)


# ---------------------------------------------------------------------------
# POST /models/pull
# ---------------------------------------------------------------------------

@router.post("/models/pull")
async def pull_model(request: Request) -> StreamingResponse:
    """
    Download a model from HuggingFace. Streams progress as newline-delimited JSON.
    Body: {"model": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
    """
    body = await request.json()
    hf_path: str = body.get("model", "").strip()
    if not hf_path:
        raise HTTPException(status_code=400, detail={"error": {"message": "model field is required"}})

    hf_cli = os.path.join(_VENV_BIN, "huggingface-cli")

    async def stream():
        import asyncio
        import json
        yield json.dumps({"status": "pulling", "model": hf_path}) + "\n"
        try:
            proc = await asyncio.create_subprocess_exec(
                hf_cli, "download", hf_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            async for line in proc.stdout:
                text = line.decode().strip()
                if text:
                    yield json.dumps({"status": "downloading", "detail": text}) + "\n"
            await proc.wait()
            if proc.returncode == 0:
                yield json.dumps({"status": "done", "model": hf_path}) + "\n"
            else:
                yield json.dumps({"status": "error", "error": f"exit code {proc.returncode}"}) + "\n"
        except Exception as exc:
            yield json.dumps({"status": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


# ---------------------------------------------------------------------------
# POST /models/load  — pre-warm a model without inference
# ---------------------------------------------------------------------------

@router.post("/models/load")
async def load_model(request: Request) -> dict:
    """
    Pre-load a model into memory without running inference.
    Useful for warming up before a known workload.
    Body: {"model": "mlx-qwen2.5-7b", "keep_alive": "30m"}
    """
    body = await request.json()
    model_name: str = body.get("model", "")

    if model_name not in config.MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model not found: {model_name}", "code": 404}},
        )

    keep_alive = _parse_keep_alive(body.get("keep_alive"))
    model_cfg = config.MODELS[model_name]

    if model_cfg.type in ("text", "vision"):
        if keep_alive is not None:
            process_manager.set_keep_alive(keep_alive)
        await inline_manager.unload()
        await process_manager.ensure_model(model_name)
    else:
        if keep_alive is not None:
            inline_manager.set_keep_alive(keep_alive)
        await process_manager.unload()
        await inline_manager.ensure_model(model_name)

    return {"model": model_name, "status": "ready"}


# ---------------------------------------------------------------------------
# POST /models/unload  — force-free memory now
# ---------------------------------------------------------------------------

@router.post("/models/unload")
async def unload_model(request: Request) -> dict:
    """
    Immediately unload the active model and free all memory.
    Body: {} or {"model": "mlx-qwen2.5-7b"}  (model name is informational)
    """
    try:
        body = await request.json()
    except Exception:
        body = {}
    model_name: str | None = body.get("model")

    await process_manager.unload()
    await inline_manager.unload()

    return {"status": "unloaded", "model": model_name}


# ---------------------------------------------------------------------------
# DELETE /models/{model_id}  — remove from HuggingFace disk cache
# ---------------------------------------------------------------------------

@router.delete("/models/{model_id}")
async def delete_model(model_id: str) -> dict:
    """
    Delete a model from the local HuggingFace cache to free disk space.
    Automatically unloads the model first if it is currently active.
    """
    if model_id not in config.MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model not found: {model_id}", "code": 404}},
        )

    model_cfg = config.MODELS[model_id]

    # Unload first if currently active — avoids file-in-use issues
    await process_manager.unload()
    await inline_manager.unload()

    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        repos = [r for r in cache_info.repos if r.repo_id == model_cfg.hf_path]
        if not repos:
            raise HTTPException(
                status_code=404,
                detail={"error": {"message": f"Model '{model_id}' is not in the local HuggingFace cache"}},
            )
        commit_hashes = [rev.commit_hash for repo in repos for rev in repo.revisions]
        strategy = cache_info.delete_revisions(*commit_hashes)
        strategy.execute()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={"error": {"message": f"Failed to delete model: {exc}"}},
        ) from exc

    return {"deleted": model_id, "hf_path": model_cfg.hf_path}


# ---------------------------------------------------------------------------
# POST /chat/completions
# ---------------------------------------------------------------------------

@router.post("/chat/completions")
async def chat_completions(request: Request) -> Any:
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail={"error": {"message": "Request body must be valid JSON"}})

    model_name: str = body.get("model", "")
    if model_name not in config.MODELS:
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model not found: {model_name}", "code": 404}},
        )

    model_cfg = config.MODELS[model_name]
    if model_cfg.type not in ("text", "vision"):
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Model '{model_name}' has type '{model_cfg.type}' — use the correct endpoint", "code": 404}},
        )

    keep_alive = _parse_keep_alive(body.get("keep_alive"))
    if keep_alive is not None:
        process_manager.set_keep_alive(keep_alive)

    # Free any in-process model before loading subprocess
    await inline_manager.unload()
    await process_manager.ensure_model(model_name)

    # mlx_lm.server validates the model field against the one it was started with —
    # rewrite it to the HuggingFace path so the request passes through.
    body["model"] = model_cfg.hf_path

    target = f"http://127.0.0.1:{config.MLX_PORT}/v1/chat/completions"

    if body.get("stream"):
        return await _stream_response(target, body, request.headers)
    else:
        return await _proxy_response(target, body, request.headers)


async def _proxy_response(url: str, body: dict, headers) -> Any:
    from fastapi.responses import JSONResponse
    resp = await _HTTP_CLIENT.post(url, json=body, headers=_forward_headers(headers))
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


async def _stream_response(url: str, body: dict, headers) -> StreamingResponse:
    async def generate():
        async with _HTTP_CLIENT.stream("POST", url, json=body, headers=_forward_headers(headers)) as resp:
            async for chunk in resp.aiter_bytes():
                yield chunk

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# POST /embeddings
# ---------------------------------------------------------------------------

@router.post("/embeddings")
async def embeddings(request: Request) -> dict:
    body = await request.json()
    model_name: str = body.get("model", "")
    input_data = body.get("input", [])

    if model_name not in config.MODELS or config.MODELS[model_name].type != "embedding":
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"Embedding model not found: {model_name}", "code": 404}},
        )

    # Normalise input into list[dict] for mlx-embeddings:
    #   "hello"                         → [{"text": "hello"}]
    #   ["a", "b"]                      → [{"text": "a"}, {"text": "b"}]
    #   [{"text": "a", "image": "..."}] → passed through unchanged
    if isinstance(input_data, str):
        inputs = [{"text": input_data}]
    elif isinstance(input_data, list):
        inputs = [
            item if isinstance(item, dict) else {"text": item}
            for item in input_data
        ]
    else:
        inputs = [{"text": str(input_data)}]

    keep_alive = _parse_keep_alive(body.get("keep_alive"))
    if keep_alive is not None:
        inline_manager.set_keep_alive(keep_alive)

    await process_manager.unload()
    vectors = await inline_manager.generate_embeddings(model_name, inputs)

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": model_name,
    }


# ---------------------------------------------------------------------------
# POST /audio/speech  (TTS)
# ---------------------------------------------------------------------------

@router.post("/audio/speech")
async def audio_speech(request: Request) -> Response:
    body = await request.json()
    model_name: str = body.get("model", "")
    text: str = body.get("input", "")
    speed: float = float(body.get("speed", 1.0))
    lang_code: str = body.get("language", "en")

    if not text:
        raise HTTPException(status_code=400, detail={"error": {"message": "input text is required"}})

    if model_name not in config.MODELS or config.MODELS[model_name].type != "tts":
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"TTS model not found: {model_name}", "code": 404}},
        )

    keep_alive = _parse_keep_alive(body.get("keep_alive"))
    if keep_alive is not None:
        inline_manager.set_keep_alive(keep_alive)

    await process_manager.unload()
    wav_bytes = await inline_manager.generate_tts(model_name, text, speed, lang_code)
    return Response(content=wav_bytes, media_type="audio/wav")


# ---------------------------------------------------------------------------
# POST /audio/transcriptions  (STT)
# ---------------------------------------------------------------------------

@router.post("/audio/transcriptions")
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: str | None = Form(None),
) -> dict:
    """
    Transcribe an audio file to text using an STT model.
    Accepts any audio format supported by mlx-whisper (wav, mp3, m4a, flac, etc.)
    """
    if model not in config.MODELS or config.MODELS[model].type != "stt":
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"STT model not found: {model}", "code": 404}},
        )

    audio_bytes = await file.read()

    await process_manager.unload()
    text = await inline_manager.generate_stt(model, audio_bytes, language or None)
    return {"text": text}


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@router.get("/status")
async def status() -> dict:
    return {
        "subprocess": process_manager.get_status(),
        "inline": inline_manager.get_status(),
        "memory": _get_memory_stats(),
    }


# ---------------------------------------------------------------------------
# GET /status/logs/{model_name}
# ---------------------------------------------------------------------------

@router.get("/status/logs/{model_name}")
async def model_logs(model_name: str) -> dict:
    """Return the last 100 lines of the subprocess log for the given model."""
    log_path = _LOG_DIR / f"{model_name}.log"
    if not log_path.exists():
        raise HTTPException(
            status_code=404,
            detail={"error": {"message": f"No logs found for '{model_name}'. Logs are only written for subprocess models (text/vision) that have been loaded."}},
        )
    lines = log_path.read_text().splitlines()
    return {
        "model": model_name,
        "log_path": str(log_path),
        "lines": lines[-100:],
    }
