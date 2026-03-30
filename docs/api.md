# API Reference

mlx-manager exposes an OpenAI-compatible HTTP API on port `8095`. All endpoints are available under the `/v1` prefix. The `/status` endpoint is also accessible at the root without a prefix.

---

## Authentication

Set the `MLX_API_KEY` environment variable to enable bearer token auth. When set, all `/v1/*` endpoints require `Authorization: Bearer <key>`. The bare `/health` endpoint at the root is always unauthenticated (for load balancer probes).

```bash
export MLX_API_KEY=my-secret-key
make mlx-start
```

When `MLX_API_KEY` is unset (default), authentication is disabled — safe for localhost-only deployments.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check (no auth) |
| `GET` | `/v1/health` | Liveness check |
| `GET` | `/v1/version` | Manager version |
| `GET` | `/v1/models` | List all configured models |
| `GET` | `/v1/models/local` | List only downloaded (cached) models |
| `GET` | `/v1/models/{model_id}` | Full detail for one model |
| `POST` | `/v1/models/pull` | Download a model from HuggingFace |
| `POST` | `/v1/models/load` | Pre-warm a model without inference |
| `POST` | `/v1/models/unload` | Force-unload active model, free memory |
| `DELETE` | `/v1/models/{model_id}` | Delete model from local HuggingFace cache |
| `POST` | `/v1/chat/completions` | Chat inference (text and vision models) |
| `POST` | `/v1/embeddings` | Generate embeddings |
| `POST` | `/v1/audio/speech` | Text-to-speech |
| `POST` | `/v1/audio/transcriptions` | Speech-to-text |
| `GET` | `/v1/status` | Manager state + memory stats |
| `GET` | `/v1/status/logs/{model}` | Last 100 lines of a model's subprocess log |
| `GET` | `/status` | Same as `/v1/status` (no prefix, operator convenience) |

---

## GET /health

Liveness check. Returns `200 OK` immediately. Use this for Docker healthchecks, load balancer probes, or monitoring.

```bash
curl http://localhost:8095/health
# → {"status": "ok"}
```

Also available at `/v1/health`.

---

## GET /v1/version

Returns the manager version string.

```bash
curl http://localhost:8095/v1/version
# → {"version": "0.1.0"}
```

---

## GET /v1/models

Returns all models defined in `models.yaml` with their capabilities.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-qwen2.5-7b",
      "object": "model",
      "created": 1743200000,
      "owned_by": "mlx-community",
      "capabilities": ["completion"]
    },
    {
      "id": "mlx-qwen3vl-8b",
      "object": "model",
      "created": 1743200000,
      "owned_by": "mlx-community",
      "capabilities": ["completion", "vision"]
    },
    {
      "id": "mlx-chatterbox",
      "object": "model",
      "created": 1743200000,
      "owned_by": "mlx-community",
      "capabilities": ["audio_speech"]
    }
  ]
}
```

**Capabilities values**

| Value | Meaning |
|-------|---------|
| `completion` | Chat completions via `/v1/chat/completions` |
| `vision` | Accepts image inputs in messages |
| `embedding` | Embeddings via `/v1/embeddings` |
| `audio_speech` | TTS via `/v1/audio/speech` |
| `audio_transcription` | STT via `/v1/audio/transcriptions` |

**Example**

```bash
curl http://localhost:8095/v1/models
# or
make models
```

---

## GET /v1/models/local

Returns only the configured models that are present in the local HuggingFace cache. Use this to check which models are ready to use without triggering a 503.

```bash
curl http://localhost:8095/v1/models/local
```

Response has the same shape as `/v1/models` but only includes cached models, with an added `"cached": true` field.

---

## GET /v1/models/{model_id}

Returns full detail for a single configured model.

**Response**

```json
{
  "id": "mlx-qwen2.5-7b",
  "object": "model",
  "created": 1743200000,
  "owned_by": "mlx-community",
  "type": "text",
  "hf_path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "capabilities": ["completion"],
  "context_length": 32768,
  "max_kv_cache_size": 0,
  "cached": true
}
```

Returns `404` if the model is not in `models.yaml`.

---

## POST /v1/models/load

Pre-warm a model into memory before the first real request. Blocks until the model is `ready` or returns `503` on failure.

**Request body**

```json
{"model": "mlx-qwen2.5-7b", "keep_alive": "30m"}
```

`keep_alive` is optional — see the [keep_alive parameter](#keep_alive-parameter) section.

**Response**

```json
{"model": "mlx-qwen2.5-7b", "status": "ready"}
```

---

## POST /v1/models/unload

Force-unload the active model immediately, freeing all memory. Does not wait for inactivity timeout.

**Request body**

```json
{}
```

**Response**

```json
{"status": "unloaded", "model": null}
```

---

## DELETE /v1/models/{model_id}

Delete a model from the local HuggingFace cache to free disk space. Automatically unloads the model first if it is currently active.

```bash
curl -X DELETE http://localhost:8095/v1/models/mlx-qwen2.5-7b
```

**Response**

```json
{"deleted": "mlx-qwen2.5-7b", "hf_path": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
```

Returns `404` if the model is not cached locally.

> **Note:** This removes the model from disk. Re-downloading requires `POST /v1/models/pull` or `make download-*`.

---

## POST /v1/models/pull

Download a model from HuggingFace. Streams progress as newline-delimited JSON.

**Request body**

```json
{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
```

**Streamed response** (one JSON object per line)

```
{"status": "pulling", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
{"status": "downloading", "detail": "Fetching 12 files..."}
{"status": "done", "model": "mlx-community/Qwen2.5-7B-Instruct-4bit"}
```

On error:
```
{"status": "error", "error": "Repository not found"}
```

**Example**

```bash
curl -X POST http://localhost:8095/v1/models/pull \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-community/Qwen2.5-3B-Instruct-4bit"}'
```

---

## POST /v1/chat/completions

Chat inference endpoint for `text` and `vision` model types. Proxies the request to the underlying `mlx_lm.server` or `mlx_vlm.server` subprocess after ensuring it is loaded.

If the requested model is not currently loaded, this endpoint will block while the model loads (up to `startup_timeout_seconds`). This is expected behaviour on first request or after a model switch.

**Request body** — standard OpenAI chat completions format

```json
{
  "model": "mlx-qwen2.5-7b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512
}
```

Set `"stream": true` to receive a server-sent events stream.

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1743200000,
  "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "Paris."},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": 24, "completion_tokens": 2, "total_tokens": 26}
}
```

> **Note:** The `model` field in the response will contain the HuggingFace path (`hf_path`), not the friendly name from `models.yaml`. This is required by `mlx_lm.server`'s internal validation.

**Example**

```bash
curl http://localhost:8095/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-qwen2.5-7b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

**`keep_alive` parameter**

Add `"keep_alive"` to any request body to override the idle timeout for the current session. Accepts seconds as a number or a duration string (`"5m"`, `"1h"`, `"30s"`). Use `-1` to disable auto-unload.

```json
{"model": "mlx-qwen2.5-7b", "messages": [...], "keep_alive": "30m"}
```

**Header forwarding**

`content-type`, `authorization`, `accept`, and `x-request-id` are forwarded to the subprocess. All other headers are dropped.

---

## POST /v1/embeddings

Generates vector embeddings using an `embedding` type model.

Input is normalised before being passed to `mlx-embeddings`:

| Input format | Normalised to |
|---|---|
| `"hello"` (string) | `[{"text": "hello"}]` |
| `["a", "b"]` (list of strings) | `[{"text": "a"}, {"text": "b"}]` |
| `[{"text": "a", "image": "..."}]` | passed through unchanged |

**Request body**

```json
{
  "model": "mlx-qwen3-embedding",
  "input": ["The quick brown fox", "jumped over the lazy dog"]
}
```

Multimodal input (text + image):

```json
{
  "model": "mlx-qwen3-embedding",
  "input": [
    {"text": "a photo of a dog", "image": "/path/to/image.jpg"}
  ]
}
```

**Response**

```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "index": 0, "embedding": [0.021, -0.034, ...]},
    {"object": "embedding", "index": 1, "embedding": [0.018, -0.041, ...]}
  ],
  "model": "mlx-qwen3-embedding"
}
```

**Example**

```bash
curl http://localhost:8095/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-qwen3-embedding", "input": "Hello world"}'
```

---

## POST /v1/audio/speech

Generates speech from text using a `tts` type model. Returns raw WAV audio.

**Request body**

```json
{
  "model": "mlx-chatterbox",
  "input": "The quick brown fox jumped over the lazy dog.",
  "speed": 1.0,
  "language": "en"
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Must be a configured `tts` model |
| `input` | string | required | Text to synthesise |
| `speed` | float | `1.0` | Playback speed multiplier |
| `language` | string | `"en"` | Language code passed to the TTS model |

**Response**

- Content-Type: `audio/wav`
- Body: 16-bit PCM WAV, 24 kHz, mono

**Example**

```bash
curl http://localhost:8095/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model": "mlx-chatterbox", "input": "Hello, world!"}' \
  --output speech.wav
```

---

## POST /v1/audio/transcriptions

Transcribe an audio file to text using an `stt` type model. Accepts any format supported by mlx-whisper (WAV, MP3, M4A, FLAC, etc.).

**Request** — multipart form data

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | Audio file to transcribe |
| `model` | string | yes | Must be a configured `stt` model |
| `language` | string | no | Language code hint (e.g. `"en"`, `"fr"`). Auto-detected if omitted. |

**Response**

```json
{"text": "The quick brown fox jumped over the lazy dog."}
```

**Example**

```bash
curl http://localhost:8095/v1/audio/transcriptions \
  -F "file=@recording.wav" \
  -F "model=mlx-whisper-turbo"

# With language hint:
curl http://localhost:8095/v1/audio/transcriptions \
  -F "file=@recording.mp3" \
  -F "model=mlx-whisper-large" \
  -F "language=fr"
```

---

## GET /v1/status

Returns the current state of both model managers plus system memory stats. Useful for monitoring and debugging.

**Response**

```json
{
  "subprocess": {
    "active_model": "mlx-qwen2.5-7b",
    "state": "ready",
    "last_request_at": "2026-03-30T10:24:00+00:00",
    "unload_at": "2026-03-30T10:34:00+00:00",
    "uptime_seconds": 3742,
    "inactivity_timeout_seconds": 600
  },
  "inline": {
    "active_model": null,
    "state": "idle",
    "last_request_at": null,
    "unload_at": null,
    "inactivity_timeout_seconds": 600
  },
  "memory": {
    "total_gb": 64.0,
    "used_gb": 38.4,
    "available_gb": 25.6,
    "percent_used": 60.0
  }
}
```

`unload_at` is the ISO 8601 timestamp when the model will auto-unload due to inactivity. `null` when no model is loaded or `keep_alive` was set to `-1`.

**States**

| State | Meaning |
|-------|---------|
| `idle` | No model loaded |
| `loading` | Model is being loaded (subprocess starting or in-process load in progress) |
| `ready` | Model is loaded and accepting requests |
| `failed` | Last load attempt failed; next request will retry |

**Example**

```bash
curl http://localhost:8095/status | python3 -m json.tool
# or
make status
```

---

## GET /v1/status/logs/{model_name}

Returns the last 100 lines of the subprocess log for a text or vision model. Logs are written to `/tmp/mlx-manager-logs/<model>.log` and only exist after the model has been loaded at least once.

```bash
curl http://localhost:8095/v1/status/logs/mlx-qwen2.5-7b
```

**Response**

```json
{
  "model": "mlx-qwen2.5-7b",
  "log_path": "/tmp/mlx-manager-logs/mlx-qwen2.5-7b.log",
  "lines": [
    "Loading model mlx-community/Qwen2.5-7B-Instruct-4bit...",
    "..."
  ]
}
```

Returns `404` if no log exists for the model name.

---

## Error Responses

All errors follow the OpenAI error format.

| Status | Condition |
|--------|-----------|
| `400` | Request body is not valid JSON |
| `400` | `input` field is empty (TTS endpoint) |
| `404` | Model name not found in `models.yaml` |
| `404` | Model found but wrong type (e.g. using a `tts` model on `/chat/completions`) |
| `503` | Model failed to load within `startup_timeout_seconds` |

**Error body**

```json
{
  "error": {
    "message": "Model not found: my-unknown-model",
    "code": 404
  }
}
```
