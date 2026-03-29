# mlx-manager

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)
[![CI](https://github.com/your-username/mlx-manager/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mlx-manager/actions/workflows/ci.yml)

An OpenAI-compatible model manager for Apple Silicon Macs. It hot-swaps local MLX models on demand — text, vision, embeddings, TTS, and STT — loading exactly one at a time to stay within unified memory limits.

```
LiteLLM (Docker) → host.docker.internal:8090 → mlx-manager (Mac host) → MLX model
```

---

## The Problem

MLX models consume large amounts of Apple Silicon unified memory. Running two simultaneously exhausts RAM. Most local model tools either load everything at once, use LRU caches that still allow overlap, or don't support the full MLX stack (text + vision + embeddings + TTS).

mlx-manager enforces a strict single-model contract: exactly one model in memory at any moment, automatically freed after a configurable idle period.

---

## Features

- **Hot-swap by model name** — send a request to any configured model; the manager loads it and unloads the previous one automatically
- **OpenAI-compatible API** — drop-in with LiteLLM, any OpenAI SDK, or direct HTTP
- **All five MLX model types** — text (`mlx-lm`), vision (`mlx-vlm`), embeddings (`mlx-embeddings`), TTS (`mlx-audio`), STT (`mlx-whisper`)
- **Subprocess isolation for large models** — text/vision models run as isolated subprocesses to prevent memory fragmentation; embeddings/TTS/STT run in-process
- **Auto-unload on inactivity** — configurable timeout (default 10 min) frees memory when idle; `unload_at` timestamp surfaced in `/status`
- **Per-request `keep_alive`** — override the idle timeout per request (`"keep_alive": "30m"`, `"-1"` for permanent, `0` to unload after response)
- **Prompt caching / KV cache control** — `max_kv_cache_size` per model caps KV cache token capacity (`--max-kv-cache-size`); enables efficient prefix reuse across requests
- **Model management API** — preload (`POST /v1/models/load`), force-unload (`POST /v1/models/unload`), delete from disk (`DELETE /v1/models/{id}`), show detail (`GET /v1/models/{id}`)
- **Model capabilities** — each model exposes a `capabilities` array (`["completion"]`, `["embedding"]`, etc.) for LiteLLM routing
- **Memory stats in `/status`** — unified memory pressure (`used_gb`, `available_gb`, `percent_used`) via psutil
- **Failure recovery** — if a model fails to load, the next request retries automatically
- **Subprocess logs** — stderr from failing models written to `/tmp/mlx-manager-logs/<model>.log`; retrievable via `GET /v1/status/logs/{model}`
- **Optional bearer token auth** — set `MLX_API_KEY` env var to protect all `/v1/*` endpoints
- **YAML model registry** — add models by editing `models.yaml`, no code changes needed
- **Minimal footprint** — ~600 lines of Python, no GUI, runs as a headless daemon

---

## Why Not Ollama?

Ollama is the closest equivalent in terms of feature set and design philosophy — hot-swap, auto-unload, OpenAI-compatible API, single active model. The difference is the runtime.

Ollama uses **llama.cpp** as its inference backend. llama.cpp uses Metal but is not the native Apple MLX framework. MLX is Apple's own machine learning framework, built specifically for the unified memory architecture of Apple Silicon. In practice, MLX models from `mlx-community` typically outperform their llama.cpp equivalents on Apple Silicon in both throughput and memory efficiency.

mlx-manager is what Ollama would be if it were built natively on MLX.

---

## Why Not LM Studio?

LM Studio is closed source and requires a GUI app to be installed and running. It cannot be embedded in a headless pipeline, scripted, or modified. Its MLX support is also newer and less complete than running mlx-lm/mlx-vlm directly.

---

## Why Not mlx-openai-server?

`mlx-openai-server` is a good project. The key architectural difference: it runs all models in-process. For a 0.6B embedding model that's fine. For a 7B+ LLM it causes memory fragmentation over long sessions — the Python process holds onto MLX memory allocations that are never fully released between model switches.

mlx-manager runs text/vision models as **isolated subprocesses**. When a model is unloaded, the subprocess exits and the OS reclaims all its memory cleanly. Embeddings and TTS stay in-process because they are small enough that fragmentation is not a concern.

---

## Why Not Docker?

MLX requires direct access to the Apple Silicon Metal GPU. Docker on Mac runs a Linux VM — Metal is not accessible inside it. Running mlx-manager in Docker would silently fall back to CPU inference.

The correct topology is: stateless services (LiteLLM, databases, monitoring) in Docker; mlx-manager on the Mac host. They communicate via `host.docker.internal`.

---

## Prerequisites

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- MLX models downloaded to your HuggingFace cache

---

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Download models you want to use
#    Text/vision models — example:
uv run huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

#    TTS model:
make download-tts

# 3. Edit models.yaml to list your downloaded models
#    (see docs/configuration.md for the full reference)

# 4. Start the manager
make mlx-start

# 5. Verify
make status
make models
```

The manager listens on `http://0.0.0.0:8090`.

---

## Usage

### Chat completion

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-qwen2.5-7b",
    "messages": [{"role": "user", "content": "What is Apple Silicon?"}]
  }'
```

### Embeddings

```bash
curl http://localhost:8090/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-qwen3-embedding", "input": "Hello world"}'
```

### Text-to-speech

```bash
curl http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-chatterbox", "input": "Hello from Apple Silicon."}' \
  --output speech.wav
```

### Speech-to-text

```bash
curl http://localhost:8090/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=mlx-whisper-turbo"
```

### Streaming

```bash
curl http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-qwen2.5-7b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

---

## LiteLLM Integration

mlx-manager is designed to sit behind LiteLLM in a Docker-on-Mac stack.

```yaml
# litellm/config.yaml
model_list:
  - model_name: mlx-qwen2.5-7b
    litellm_params:
      model: openai/mlx-qwen2.5-7b
      api_base: http://host.docker.internal:8090/v1
      api_key: none

  - model_name: mlx-qwen3-embedding
    litellm_params:
      model: openai/mlx-qwen3-embedding
      api_base: http://host.docker.internal:8090/v1
      api_key: none
```

---

## Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install dependencies |
| `make mlx-start` | Start manager on port 8090 (foreground) |
| `make mlx-dev` | Start with auto-reload on code changes |
| `make mlx-stop` | Kill manager and any active subprocess |
| `make status` | Print current model state as JSON |
| `make models` | List all configured models as JSON |
| `make download-tts` | Download the Chatterbox TTS model |
| `make download-stt` | Download the Whisper STT model |

## Authentication

Set `MLX_API_KEY` in the environment to enable bearer token auth on all `/v1/*` endpoints. When unset, all requests are accepted (safe for localhost-only use).

```bash
export MLX_API_KEY=my-secret-key
make mlx-start
```

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | System design, module map, state machines, request flows, design decisions |
| [docs/configuration.md](docs/configuration.md) | `models.yaml` complete reference, all settings, validation rules |
| [docs/api.md](docs/api.md) | All endpoints, request/response schemas, curl examples, error codes |
| [docs/development.md](docs/development.md) | Setup, make commands, debugging, adding models, LiteLLM integration |

---

## Project Structure

```
mlx-manager/
├── src/
│   ├── main.py              FastAPI app and lifespan
│   ├── config.py            Reads models.yaml, typed config
│   ├── router.py            HTTP endpoints, cross-manager coordination
│   ├── process_manager.py   Subprocess lifecycle (text/vision)
│   └── inline_manager.py    In-process lifecycle (embedding/tts)
├── models.yaml              Model registry — edit this to add models
├── docs/                    Full documentation
├── pyproject.toml
└── Makefile
```

---

## Supported Model Types

| Type | Inference | Endpoint | Capabilities |
|------|-----------|----------|--------------|
| `text` | `mlx_lm.server` subprocess | `/v1/chat/completions` | `["completion"]` |
| `vision` | `mlx_vlm.server` subprocess | `/v1/chat/completions` | `["completion", "vision"]` |
| `embedding` | `mlx-embeddings` in-process | `/v1/embeddings` | `["embedding"]` |
| `tts` | `mlx-audio` in-process | `/v1/audio/speech` | `["audio_speech"]` |
| `stt` | `mlx-whisper` in-process | `/v1/audio/transcriptions` | `["audio_transcription"]` |
