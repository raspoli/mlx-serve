# Architecture

## Overview

mlx-manager is a thin FastAPI proxy that sits between a client (typically LiteLLM) and local MLX model servers. It manages a single active model at a time, loading and unloading as requests arrive.

```
┌──────────────────────────────────────────────────────────────┐
│  Client  (LiteLLM in Docker / curl / any OpenAI SDK)         │
└──────────────────────────────────────────────────────────────┘
                          │  HTTP  (port 8095)
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  mlx-manager  (FastAPI, Mac host)                            │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  router.py  — validates, coordinates, proxies          │  │
│  └────────────────────────────────────────────────────────┘  │
│          │                              │                    │
│          ▼                              ▼                    │
│  ┌────────────────────┐    ┌────────────────────────────┐   │
│  │  process_manager   │    │  inline_manager            │   │
│  │                    │    │                            │   │
│  │  text / vision     │    │  embedding / tts / stt     │   │
│  │  (subprocess)      │    │  (in-process, executor)    │   │
│  └────────────────────┘    └────────────────────────────┘   │
│          │                                                   │
│          ▼  (port 8091, loopback only)                       │
│  ┌────────────────────┐                                      │
│  │  mlx_lm.server  or │                                      │
│  │  mlx_vlm.server    │                                      │
│  └────────────────────┘                                      │
└──────────────────────────────────────────────────────────────┘
```

---

## Modules

### `src/main.py`
FastAPI application entry point. Owns the app lifespan:
- Starts two background inactivity watcher tasks on startup
- Cancels watchers and unloads all models on shutdown
- Registers routes under `/v1` prefix; `/status` also accessible at root

### `src/config.py`
Reads `models.yaml` at import time and exposes typed constants used across all modules:
- `MODELS` — dict of model name → `ModelConfig`
- `MLX_PORT` — internal subprocess port (default 8091)
- `MANAGER_PORT` — FastAPI listen port (default 8095)
- `INACTIVITY_TIMEOUT` — seconds before idle model is unloaded (default 600)
- `STARTUP_TIMEOUT` — max seconds to wait for model to become ready (default 120)

Validates model types at startup; raises `ValueError` with a clear message on bad config.

### `src/router.py`
HTTP endpoint handlers. The only module that imports both managers. Responsible for cross-manager coordination before delegating:
- Before loading a subprocess model → calls `inline_manager.unload()`
- Before loading an inline model → calls `process_manager.unload()`

This keeps both managers free of knowledge about each other.

### `src/process_manager.py`
Subprocess lifecycle for `text` and `vision` model types. Spawns `mlx_lm.server` or `mlx_vlm.server` as a child process, polls its `/health` endpoint until ready, and kills it on unload or inactivity.

### `src/inline_manager.py`
In-process lifecycle for `embedding`, `tts`, and `stt` model types. Loads models directly into the FastAPI process via `mlx-embeddings`, `mlx-audio`, and `mlx-whisper`. Runs blocking inference in a thread executor to avoid blocking the async event loop.

---

## Execution Model: Subprocess vs In-Process

| Model type | Execution | Reason |
|------------|-----------|--------|
| `text` | subprocess (`mlx_lm.server`) | Large models; isolation prevents memory fragmentation |
| `vision` | subprocess (`mlx_vlm.server`) | Same as text |
| `embedding` | in-process (thread executor) | Small models; no subprocess overhead needed |
| `tts` | in-process (thread executor) | Same as embedding |
| `stt` | in-process (thread executor) | Same as embedding; mlx-whisper loads lazily per-call |

---

## State Machines

Both managers use the same four-state machine.

```
              request arrives
              (different model or IDLE/FAILED)
                      │
    ┌─────────────────▼──────────────────┐
    │               IDLE                 │◄─── inactivity timeout
    └─────────────────┬──────────────────┘     or explicit unload()
                      │
              start loading
                      │
    ┌─────────────────▼──────────────────┐
    │             LOADING                │
    │                                    │
    │  process_manager: health poll /2s  │
    │  inline_manager:  run_in_executor  │
    └───────┬────────────────┬───────────┘
            │                │
       success           failure / timeout
            │                │
    ┌───────▼──────┐  ┌──────▼──────────┐
    │    READY     │  │     FAILED      │
    │              │  │  → 503 to client│
    │  proxy reqs  │  │  next request   │
    │  update      │  │  retries load   │
    │  timestamp   │  └─────────────────┘
    └───────┬──────┘
            │
    inactivity timeout
    (30s check loop)
            │
         IDLE
```

Key behaviours:
- **FAILED → retry**: The next request for a failed model retries the load rather than returning 503 forever.
- **Concurrent requests during LOADING**: All requests for the same model wait on a shared `asyncio.Event`; the model loads exactly once.
- **Model switch**: Acquiring `_switch_lock` serialises switches. A coroutine that waited on the lock re-checks the state on entry — if another coroutine already loaded the requested model, it returns immediately.

---

## Request Flows

### Cold start (text model, first request)

```
Client  →  POST /v1/chat/completions  {model: "mlx-qwen2.5-7b"}
              │
        router validates model name
              │
        inline_manager.unload()        ← no-op if nothing loaded
              │
        process_manager.ensure_model("mlx-qwen2.5-7b")
              │
              ├─ _switch_model() acquires _switch_lock
              ├─ spawns: mlx_lm.server --model ... --port 8091
              ├─ asyncio.create_task(_health_check_loop())
              │
              │  [health check polls GET 127.0.0.1:8091/health every 2s]
              │  [ensure_model awaits _ready_event]
              │
              ├─ health returns 200  →  _state = READY, _ready_event.set()
              │
        router rewrites body["model"] to hf_path
              │
        proxy POST → 127.0.0.1:8091/v1/chat/completions
              │
Client  ←  response (stream or JSON)
```

### Warm request (same model already READY)

```
Client  →  POST /v1/chat/completions  {model: "mlx-qwen2.5-7b"}
              │
        inline_manager.unload()        ← no-op
              │
        process_manager.ensure_model() ← active_model matches, READY → return immediately
              │
        proxy → 127.0.0.1:8091
              │
Client  ←  response
```

### Model switch (different model requested)

```
Client  →  POST /v1/chat/completions  {model: "mlx-qwen3vl-8b"}
              │
        inline_manager.unload()        ← no-op
              │
        process_manager.ensure_model("mlx-qwen3vl-8b")
              │
              ├─ active_model is "mlx-qwen2.5-7b" → _switch_model()
              ├─ _terminate_current(): SIGTERM → wait 5s → SIGKILL if needed
              ├─ spawns: mlx_vlm.server --model ... --port 8091
              │  [health check loop]
              │
        proxy → 127.0.0.1:8091
              │
Client  ←  response
```

### Inactivity unload

```
[background: start_inactivity_watcher() loops every 30s]

  if state == READY
  and now - last_request_at > inactivity_timeout_seconds:
      _terminate_current()   →  state = IDLE
```

---

## Design Decisions

### Single subprocess port (8091)

Only one model runs at a time, so only one port is ever needed. A port registry would add complexity with no benefit.

### No Docker

MLX requires the Apple Silicon Metal GPU. Docker on Mac runs a Linux VM — Metal is not accessible inside it. Running mlx-manager on the host is not a workaround; it is the only correct deployment.

### Router owns cross-manager coordination

`process_manager` and `inline_manager` are pure: neither knows the other exists. The router is the only module that imports both and decides which to unload before loading the other. This keeps the managers independently testable and prevents circular imports.

### YAML model registry

Models are defined in `models.yaml`, not in code. Operators add or remove models by editing the file and restarting the service. No code changes or redeployment of the Python package is required.

### Subprocess stderr → log file

When a subprocess model fails to load, the error is written to `/tmp/mlx-manager-logs/<model-name>.log`. The log path is printed to stdout on failure so the operator knows exactly where to look. Silently discarding stderr (the previous approach) made cold-start failures undebuggable.
