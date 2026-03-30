# Contributing to mlx-manager

## Prerequisites

- Apple Silicon Mac (M1 or later) — MLX requires Metal GPU access
- macOS 13+
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Setup

```bash
git clone https://github.com/your-username/mlx-manager
cd mlx-manager
make install
```

## Running locally

```bash
make mlx-start    # start on port 8095
make status       # check state
make mlx-stop     # stop everything
```

Dev mode (auto-reloads on code changes):

```bash
make mlx-dev
```

## Project layout

```
src/
├── main.py             FastAPI app, lifespan, route registration
├── config.py           Reads models.yaml, exposes typed constants
├── router.py           All HTTP endpoints, cross-manager coordination
├── process_manager.py  Subprocess lifecycle (text/vision)
└── inline_manager.py   In-process lifecycle (embedding/tts/stt)
models.yaml             Operator-editable model registry
docs/                   Full documentation
```

Key constraint: `process_manager` and `inline_manager` must not import each other. All cross-manager coordination belongs in `router.py`.

## Adding a new model type

If you want to add support for a new MLX inference library:

1. Add the new type string to `_VALID_TYPES` in `config.py`
2. Add load/run/generate functions to `inline_manager.py` (for in-process) or extend `process_manager.py` (for subprocess)
3. Add a new endpoint in `router.py`
4. Add the type to `_TYPE_CAPABILITIES` in `router.py`
5. Update `models.yaml` comments and `docs/configuration.md`

## Adding a model to models.yaml

```yaml
- name: my-model          # name clients use in API requests
  type: text              # text | vision | embedding | tts | stt
  hf_path: mlx-community/my-model-name
  context_length: 32768   # max output tokens (optional)
  max_kv_cache_size: 0    # KV cache token limit for prompt caching (optional)
```

Download the model first:

```bash
uv run huggingface-cli download mlx-community/my-model-name
```

## Testing

MLX requires Apple Silicon hardware — automated integration tests cannot run on GitHub Actions. The CI workflow validates Python syntax and YAML only.

Manual testing checklist before submitting a PR:

- [ ] Start the manager: `make mlx-start`
- [ ] Exercise the relevant model type with a real request
- [ ] Verify `make status` shows correct state
- [ ] Verify `make mlx-stop` cleanly terminates all processes
- [ ] Check that inactivity unload works (lower `inactivity_timeout_seconds` temporarily)

## Submitting a PR

1. Fork and create a branch from `main`
2. Make your change — keep it focused, no unrelated cleanup
3. Update docs if behavior changes
4. Open a PR using the provided template

## Style notes

- No new abstractions for single-use cases
- No docstrings on functions whose purpose is obvious from the name
- Error responses must follow OpenAI format: `{"error": {"message": "...", "code": N}}`
- Keep the two managers pure — no cross-imports
