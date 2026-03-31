# mlx-serve

[![PyPI version](https://img.shields.io/pypi/v/mlx-serve)](https://pypi.org/project/mlx-serve/)
[![Python versions](https://img.shields.io/pypi/pyversions/mlx-serve)](https://pypi.org/project/mlx-serve/)
[![License](https://img.shields.io/pypi/l/mlx-serve)](https://github.com/raspoli/mlx-serve/blob/main/LICENSE)
[![CI](https://github.com/raspoli/mlx-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/raspoli/mlx-serve/actions/workflows/ci.yml)
[![Platform](https://img.shields.io/badge/platform-Apple%20Silicon-black?logo=apple)](https://developer.apple.com/metal/)

Local inference server for Apple Silicon that hot-swaps MLX models on demand — text, vision, embeddings, TTS, and STT — loading exactly one at a time to stay within unified memory limits.

```
Client / LiteLLM  -->  mlx-serve (port 8095)  -->  MLX model (one at a time)
```

---

## Install

```bash
pip install mlx-serve[all]

# or pick only what you need:
pip install mlx-serve[text,vision]
pip install mlx-serve[embeddings,tts,stt]
```

> **Requires:** Apple Silicon Mac (M1+), macOS 13+, Python 3.11+

---

## Quick Start

```bash
# 1. Generate a default config
mlx-serve init

# 2. Edit models.yaml to list your models (see docs/configuration.md)

# 3. Start the server
mlx-serve start

# 4. Verify
curl http://localhost:8095/v1/models
```

---

## Why mlx-serve?

| | mlx-serve | Ollama | LM Studio | mlx-openai-server |
|---|---|---|---|---|
| **Runtime** | MLX (native Apple) | llama.cpp (Metal) | Mixed | MLX |
| **Memory model** | One model, subprocess-isolated | One model, in-process | GUI-managed | In-process |
| **Auto-unload** | Configurable timeout | Yes | Manual | No |
| **Model types** | 5 (text, vision, embed, TTS, STT) | 1 (text) | ~2 | ~3 |
| **API** | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible |
| **Headless / scriptable** | Yes | Yes | No (GUI) | Yes |
| **Open source** | MIT | MIT | No | MIT |

**Key differences:**

- **vs Ollama** — Ollama uses llama.cpp. mlx-serve uses Apple's native MLX framework, which typically achieves better throughput and memory efficiency on Apple Silicon. mlx-serve is what Ollama would be if it were built natively on MLX.
- **vs LM Studio** — Closed source, requires a GUI, cannot be embedded in headless pipelines.
- **vs mlx-openai-server** — Runs all models in-process, causing memory fragmentation over long sessions. mlx-serve isolates text/vision models as subprocesses so the OS reclaims all memory cleanly on unload.
- **vs Docker** — MLX requires direct Metal GPU access. Docker on Mac runs a Linux VM without Metal. The correct topology: stateless services in Docker, mlx-serve on the Mac host via `host.docker.internal`.

---

## Features

- **Hot-swap by model name** — send a request to any configured model; the server loads it and unloads the previous one automatically
- **OpenAI-compatible API** — drop-in with LiteLLM, any OpenAI SDK, or direct HTTP
- **All five MLX model types** — text (`mlx-lm`), vision (`mlx-vlm`), embeddings (`mlx-embeddings`), TTS (`mlx-audio`), STT (`mlx-whisper`)
- **Subprocess isolation** — text/vision models run as isolated subprocesses; embeddings/TTS/STT run in-process
- **Auto-unload on inactivity** — configurable timeout (default 10 min) frees memory when idle
- **Per-request `keep_alive`** — override the idle timeout per request (`"keep_alive": "30m"`, `"-1"` for permanent, `0` to unload immediately)
- **Prompt caching** — `max_kv_cache_size` per model caps KV cache token capacity for efficient prefix reuse
- **Model management API** — preload, force-unload, delete from disk, show detail, pull from HuggingFace
- **Observability** — request metrics (TTFT, TPS, latency), memory monitoring, lifecycle event log, dashboard endpoint
- **Optional auth** — set `MLX_API_KEY` to protect all `/v1/*` endpoints
- **YAML config** — add models by editing `models.yaml`, no code changes needed
- **CLI** — `mlx-serve init`, `start`, `stop`, `status`, `logs`

---

## Supported Model Types

| Type | Backend | Endpoint | Capabilities |
|------|---------|----------|--------------|
| `text` | `mlx_lm.server` subprocess | `/v1/chat/completions` | `["completion"]` |
| `vision` | `mlx_vlm.server` subprocess | `/v1/chat/completions` | `["completion", "vision"]` |
| `embedding` | `mlx-embeddings` in-process | `/v1/embeddings` | `["embedding"]` |
| `tts` | `mlx-audio` in-process | `/v1/audio/speech` | `["audio_speech"]` |
| `stt` | `mlx-whisper` in-process | `/v1/audio/transcriptions` | `["audio_transcription"]` |

---

## Usage

### Chat completion

```bash
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-qwen2.5-7b",
    "messages": [{"role": "user", "content": "What is Apple Silicon?"}]
  }'
```

### Streaming

```bash
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-qwen2.5-7b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

### Embeddings

```bash
curl http://localhost:8095/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-qwen3-embedding", "input": "Hello world"}'
```

### Text-to-speech

```bash
curl http://localhost:8095/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-chatterbox", "input": "Hello from Apple Silicon."}' \
  --output speech.wav
```

### Speech-to-text

```bash
curl http://localhost:8095/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=mlx-whisper-turbo"
```

---

## LiteLLM Integration

mlx-serve is designed to sit behind LiteLLM in a Docker-on-Mac stack.

```yaml
# litellm/config.yaml
model_list:
  - model_name: mlx-qwen2.5-7b
    litellm_params:
      model: openai/mlx-qwen2.5-7b
      api_base: http://host.docker.internal:8095/v1
      api_key: none

  - model_name: mlx-qwen3-embedding
    litellm_params:
      model: openai/mlx-qwen3-embedding
      api_base: http://host.docker.internal:8095/v1
      api_key: none
```

---

## Development

```bash
git clone https://github.com/raspoli/mlx-serve.git
cd mlx-serve
make install    # uv sync with all extras
make dev        # start with auto-reload
make test       # run test suite
make lint       # ruff check + format check
```

See [docs/development.md](docs/development.md) for the full guide.

---

## Documentation

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | System design, module map, state machines, request flows |
| [docs/configuration.md](docs/configuration.md) | `models.yaml` complete reference, all settings |
| [docs/api.md](docs/api.md) | All endpoints, request/response schemas, curl examples |
| [docs/development.md](docs/development.md) | Setup, debugging, adding models, contributing |

---

## License

[MIT](LICENSE)
