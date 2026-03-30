"""
router.py — OpenAI-compatible HTTP endpoints for the MLX model manager.
"""
import importlib.metadata
import json
import logging
import os
import pathlib
import sys
import tempfile
import time
import uuid
from typing import Any

import httpx
import psutil
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from . import config, events, inline_manager, metrics, process_manager

logger = logging.getLogger("mlx-serve.router")

_HTTP_CLIENT = httpx.AsyncClient(timeout=None)
_LOG_DIR = pathlib.Path(tempfile.gettempdir()) / "mlx-manager-logs"
_VENV_BIN = os.path.dirname(sys.executable)

# Maps model type -> OpenAI-style capabilities array
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
        ver = importlib.metadata.version("mlx-serve")
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

    logger.info(f"POST /v1/chat/completions model={model_name} stream={body.get('stream', False)}")

    # Free any in-process model before loading subprocess
    await inline_manager.unload()
    request_start = time.monotonic()
    cold_start = await process_manager.ensure_model(model_name)

    if cold_start:
        events.emit(events.EventType.REQUEST_COLD_START, model=model_name)

    # mlx_lm.server validates the model field against the one it was started with —
    # rewrite it to the HuggingFace path so the request passes through.
    body["model"] = model_cfg.hf_path

    target = f"http://127.0.0.1:{config.MLX_PORT}/v1/chat/completions"

    if body.get("stream"):
        return await _instrumented_stream_response(
            target, body, request.headers, model_name, request_start, cold_start,
        )
    else:
        return await _instrumented_proxy_response(
            target, body, request.headers, model_name, request_start, cold_start,
        )


async def _instrumented_proxy_response(
    url: str, body: dict, headers, model_name: str, request_start: float, cold_start: bool,
) -> Any:
    """Forward non-streaming request and record metrics."""
    from fastapi.responses import JSONResponse

    request_id = str(uuid.uuid4())
    resp = await _HTTP_CLIENT.post(url, json=body, headers=_forward_headers(headers))
    end = time.monotonic()

    resp_json = resp.json()

    # Extract token counts from response
    usage = resp_json.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_ms = (end - request_start) * 1000

    # Calculate TPS (generation time approximated as total time for non-streaming)
    tps = None
    if completion_tokens and completion_tokens > 0 and total_ms > 0:
        tps = round(completion_tokens / (total_ms / 1000), 1)

    metrics.record_request(metrics.RequestMetrics(
        request_id=request_id,
        model=model_name,
        endpoint="/v1/chat/completions",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        started_at=request_start,
        total_duration_ms=round(total_ms, 1),
        ttft_ms=round(total_ms, 1),  # for non-streaming, TTFT = total time
        tokens_per_second=tps,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        status_code=resp.status_code,
        error=resp_json.get("error", {}).get("message") if resp.status_code >= 400 else None,
        cold_start=cold_start,
    ))

    return JSONResponse(content=resp_json, status_code=resp.status_code)


async def _instrumented_stream_response(
    url: str, body: dict, headers, model_name: str, request_start: float, cold_start: bool,
) -> StreamingResponse:
    """Forward streaming request, measure TTFT, and record metrics."""
    request_id = str(uuid.uuid4())
    first_token_time: float | None = None
    completion_tokens = 0
    prompt_tokens_from_usage: int | None = None
    completion_tokens_from_usage: int | None = None

    async def generate():
        nonlocal first_token_time, completion_tokens
        nonlocal prompt_tokens_from_usage, completion_tokens_from_usage

        async with _HTTP_CLIENT.stream("POST", url, json=body, headers=_forward_headers(headers)) as resp:
            async for chunk in resp.aiter_lines():
                if not chunk.startswith("data:"):
                    yield chunk + "\n"
                    continue

                data_str = chunk[5:].strip()
                yield chunk + "\n\n"

                if data_str == "[DONE]":
                    continue

                # Track TTFT and token count
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                            completion_tokens += 1

                    # mlx_lm may include usage in the final chunk
                    usage = data.get("usage")
                    if usage:
                        prompt_tokens_from_usage = usage.get("prompt_tokens")
                        completion_tokens_from_usage = usage.get("completion_tokens")
                except (json.JSONDecodeError, KeyError):
                    pass

        # Record metrics after stream completes
        end = time.monotonic()
        total_ms = (end - request_start) * 1000
        ttft_ms = (first_token_time - request_start) * 1000 if first_token_time else None

        # Use usage-reported tokens if available, else our chunk count
        final_completion = completion_tokens_from_usage or completion_tokens
        tps = None
        if final_completion and first_token_time:
            gen_time = end - first_token_time
            if gen_time > 0:
                tps = round(final_completion / gen_time, 1)

        metrics.record_request(metrics.RequestMetrics(
            request_id=request_id,
            model=model_name,
            endpoint="/v1/chat/completions",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            started_at=request_start,
            total_duration_ms=round(total_ms, 1),
            ttft_ms=round(ttft_ms, 1) if ttft_ms else None,
            tokens_per_second=tps,
            prompt_tokens=prompt_tokens_from_usage,
            completion_tokens=final_completion if final_completion else None,
            status_code=200,
            cold_start=cold_start,
        ))

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

    logger.info(f"POST /v1/embeddings model={model_name} inputs={len(inputs)}")
    request_start = time.monotonic()

    await process_manager.unload()
    vectors = await inline_manager.generate_embeddings(model_name, inputs)

    total_ms = (time.monotonic() - request_start) * 1000
    metrics.record_request(metrics.RequestMetrics(
        request_id=str(uuid.uuid4()),
        model=model_name,
        endpoint="/v1/embeddings",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        started_at=request_start,
        total_duration_ms=round(total_ms, 1),
        status_code=200,
    ))

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

    logger.info(f"POST /v1/audio/speech model={model_name} chars={len(text)}")
    request_start = time.monotonic()

    await process_manager.unload()
    wav_bytes = await inline_manager.generate_tts(model_name, text, speed, lang_code)

    total_ms = (time.monotonic() - request_start) * 1000
    metrics.record_request(metrics.RequestMetrics(
        request_id=str(uuid.uuid4()),
        model=model_name,
        endpoint="/v1/audio/speech",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        started_at=request_start,
        total_duration_ms=round(total_ms, 1),
        status_code=200,
    ))

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

    logger.info(f"POST /v1/audio/transcriptions model={model} size={len(audio_bytes)}")
    request_start = time.monotonic()

    await process_manager.unload()
    text = await inline_manager.generate_stt(model, audio_bytes, language or None)

    total_ms = (time.monotonic() - request_start) * 1000
    metrics.record_request(metrics.RequestMetrics(
        request_id=str(uuid.uuid4()),
        model=model,
        endpoint="/v1/audio/transcriptions",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        started_at=request_start,
        total_duration_ms=round(total_ms, 1),
        status_code=200,
    ))

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
        "metal": metrics.get_metal_memory(),
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


# ---------------------------------------------------------------------------
# Monitoring endpoints
# ---------------------------------------------------------------------------

@router.get("/metrics")
async def get_metrics() -> dict:
    """Summary of all metrics: per-model aggregates, current memory, active model."""
    metal = metrics.get_metal_memory()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    return {
        "uptime_seconds": process_manager.get_status()["uptime_seconds"],
        "models": metrics.get_aggregates(),
        "memory": {
            "ram_total_gb": round(vm.total / 1e9, 1),
            "ram_used_gb": round(vm.used / 1e9, 1),
            "ram_available_gb": round(vm.available / 1e9, 1),
            "ram_percent": vm.percent,
            "swap_used_gb": round(swap.used / 1e9, 2),
            "metal_active_mb": metal.get("active_mb"),
            "metal_peak_mb": metal.get("peak_mb"),
            "metal_cache_mb": metal.get("cache_mb"),
        },
        "pressure": metrics.check_memory_pressure(),
        "active_subprocess_model": process_manager.get_active_model_name(),
        "active_inline_model": inline_manager.get_active_model_name(),
    }


@router.get("/metrics/requests")
async def get_request_metrics(model: str | None = None, last_n: int = 50) -> dict:
    """Recent request history with TTFT, TPS, duration."""
    return {"requests": metrics.get_request_history(model=model, last_n=last_n)}


@router.get("/metrics/memory")
async def get_memory_snapshot() -> dict:
    """Current memory snapshot: RAM + Metal + subprocess RSS."""
    return metrics.get_current_memory()


@router.get("/metrics/memory/timeline")
async def get_memory_timeline(last_n: int = 60) -> dict:
    """Memory history over time (sampled every 10s)."""
    return {"snapshots": metrics.get_memory_timeline(last_n=last_n)}


@router.get("/events")
async def get_events(
    model: str | None = None,
    type: str | None = None,
    last_n: int = 100,
) -> dict:
    """Lifecycle events: model loads, failures, switches, memory pressure."""
    return {"events": events.get_events(model=model, event_type=type, last_n=last_n)}


@router.get("/dashboard")
async def dashboard() -> dict:
    """Everything in one call: status, metrics summary, recent events."""
    metal = metrics.get_metal_memory()
    vm = psutil.virtual_memory()

    return {
        "status": {
            "subprocess": process_manager.get_status(),
            "inline": inline_manager.get_status(),
        },
        "memory": {
            "ram_used_gb": round(vm.used / 1e9, 1),
            "ram_available_gb": round(vm.available / 1e9, 1),
            "ram_percent": vm.percent,
            "metal_active_mb": metal.get("active_mb"),
            "metal_peak_mb": metal.get("peak_mb"),
            "pressure": metrics.check_memory_pressure(),
        },
        "metrics_summary": metrics.get_aggregates(),
        "recent_requests": metrics.get_request_history(last_n=5),
        "recent_events": events.get_events(last_n=10),
        "memory_timeline": metrics.get_memory_timeline(last_n=12),
    }
