# Development Guide

## Prerequisites

- Apple Silicon Mac (M1 or later)
- macOS 13+
- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## Setup

```bash
# Clone and enter the project
cd mlx-manager

# Install all dependencies into a local .venv
make install
# equivalent: uv sync
```

---

## Make Commands

| Command | Description |
|---------|-------------|
| `make install` | Install / sync dependencies via `uv sync` |
| `make mlx-start` | Start the manager on port 8095 (foreground, Ctrl+C to stop) |
| `make mlx-dev` | Start with `--reload` (auto-restarts on code changes) |
| `make mlx-stop` | Kill manager and any active subprocess |
| `make status` | Print current model state as formatted JSON |
| `make models` | List all configured models as formatted JSON |
| `make download-tts` | Download the Chatterbox TTS model |
| `make download-stt` | Download the whisper-large-v3-turbo STT model |

---

## Project Layout

```
mlx-manager/
├── src/
│   ├── main.py              FastAPI app, lifespan, route registration
│   ├── config.py            Reads models.yaml, exposes typed constants
│   ├── router.py            HTTP endpoints, cross-manager coordination
│   ├── process_manager.py   Subprocess lifecycle (text/vision)
│   └── inline_manager.py    In-process lifecycle (embedding/tts)
├── models.yaml              Operator-editable model registry
├── docs/                    Documentation
├── pyproject.toml           Project metadata and dependencies
├── Makefile                 Dev commands
└── uv.lock                  Locked dependency tree
```

---

## Adding a New Model

1. Download the model to your local HuggingFace cache:

```bash
uv run huggingface-cli download mlx-community/your-model-name
```

2. Add an entry to `models.yaml`:

```yaml
- name: my-new-model          # the name clients use in API requests
  type: text                  # text | vision | embedding | tts
  hf_path: mlx-community/your-model-name
```

3. Restart the manager:

```bash
make mlx-stop
make mlx-start
```

4. Verify it appears in the model list:

```bash
make models
```

No code changes are required to add a model.

---

## Debugging

### Check manager and model state

```bash
make status
```

Output shows both the subprocess state (text/vision) and inline state (embedding/tts), the currently active model, and when it last received a request.

### Model failed to load

If a model fails to start, the manager prints the log path to stdout:

```
[mlx-manager] Model failed to load. Logs: /tmp/mlx-manager-logs/mlx-qwen2.5-7b.log
```

Inspect the log file for the underlying error (OOM, wrong path, missing weights, etc.):

```bash
cat /tmp/mlx-manager-logs/mlx-qwen2.5-7b.log
```

Each load attempt overwrites the previous log for that model.

### Manager not responding

```bash
make status   # check if it's running
make mlx-stop # kill everything cleanly
make mlx-start
```

### Port already in use

```bash
lsof -i :8095    # find what's using the manager port
lsof -i :8091    # find what's using the subprocess port
```

---

## LiteLLM Integration

mlx-manager is designed to sit behind LiteLLM running in Docker. LiteLLM routes requests to the manager using `host.docker.internal` to reach the Mac host from inside a container.

Example LiteLLM config entry:

```yaml
model_list:
  - model_name: mlx-qwen2.5-7b
    litellm_params:
      model: openai/mlx-qwen2.5-7b
      api_base: http://host.docker.internal:8095/v1
      api_key: none
```

The `model` field in `litellm_params` must match the `name` in `models.yaml`.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `uvicorn[standard]` | ASGI server |
| `httpx` | Async HTTP client for proxying requests to subprocess |
| `pyyaml` | Parse `models.yaml` |
| `mlx-lm` | Text model inference subprocess |
| `mlx-vlm` | Vision model inference subprocess |
| `mlx-audio` | TTS inference (in-process) |
| `mlx-embeddings` | Embedding inference (in-process) |

All MLX packages require Apple Silicon and will not function on x86 hardware.
