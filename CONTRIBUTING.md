# Contributing to mlx-serve

Thank you for your interest in contributing to mlx-serve! This document covers everything you need to get started.

## Prerequisites

- Apple Silicon Mac (M1 or later) — MLX requires Metal GPU access
- macOS 13+
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/mlx-serve.git
cd mlx-serve

# 2. Install all dependencies (including dev + all model backends)
make install

# 3. Verify everything works
make test
make lint
```

## Project Layout

```
src/mlx_serve/
├── __init__.py            Version declaration
├── cli.py                 Command-line interface (init, start, stop, status, logs)
├── config.py              YAML config loading, typed constants
├── router.py              OpenAI-compatible HTTP endpoints
├── process_manager.py     Subprocess lifecycle (text/vision models)
├── inline_manager.py      In-process lifecycle (embedding/TTS/STT models)
├── events.py              Structured lifecycle event log
├── metrics.py             Request metrics and memory monitoring
├── logging_config.py      Logging setup
├── main.py                FastAPI app and lifespan
└── _default_models.yaml   Bundled fallback config
```

Key constraint: `process_manager` and `inline_manager` must not import each other. All cross-manager coordination belongs in `router.py`.

## Development Workflow

### Running locally

```bash
make dev           # start with auto-reload on code changes
make mlx-start     # start on port 8095 (foreground)
make status        # check current model state
make mlx-stop      # stop everything
```

### Testing

```bash
make test          # run full test suite (42 tests)
make lint          # ruff check + format check
```

CI runs lint, format check, and tests on Python 3.11, 3.12, and 3.13 (macOS) for every push and PR.

**Manual testing** — before submitting a PR that changes inference or model loading:

- [ ] Start the server: `mlx-serve start`
- [ ] Exercise the relevant model type with a real request
- [ ] Verify `curl http://localhost:8095/status` shows correct state
- [ ] Verify inactivity unload works (lower `inactivity_timeout_seconds` temporarily)
- [ ] Verify `mlx-serve stop` cleanly terminates all processes

### Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check src/          # lint
uv run ruff format src/ tests/  # format
```

Rules enforced: isort (import sorting), pyupgrade (modern syntax), flake8-comprehensions, flake8-simplify, flake8-use-pathlib. See `pyproject.toml` for the full config.

Style guidelines:
- No new abstractions for single-use cases
- No docstrings on functions whose purpose is obvious from the name
- Error responses must follow OpenAI format: `{"error": {"message": "...", "code": N}}`
- Keep the two managers pure — no cross-imports
- Use `pathlib.Path` over `os.path`
- Use `datetime.UTC` over `timezone.utc`

## Adding a New Model Type

If you want to add support for a new MLX inference library:

1. Add the new type string to `_VALID_TYPES` in `config.py`
2. Add load/run/generate functions to `inline_manager.py` (for in-process) or extend `process_manager.py` (for subprocess)
3. Add a new endpoint in `router.py`
4. Add the type to `_TYPE_CAPABILITIES` in `router.py`
5. Add tests for the new endpoint in `tests/test_api.py`
6. Update `models.yaml` comments and `docs/configuration.md`

## Adding a Model to models.yaml

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

## Pull Request Guidelines

1. Fork and create a branch from `main`
2. Make your change — keep it focused, no unrelated cleanup
3. Add or update tests for new/changed behavior
4. Ensure `make test` and `make lint` pass
5. Update docs if behavior changes
6. Open a PR using the provided template

## Releases and PyPI Publishing

**Only maintainers** should create releases. mlx-serve uses an automated CI/CD pipeline for PyPI deployment.

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):

- **Major (X.0.0)**: Breaking API changes
- **Minor (X.Y.0)**: New features, backward compatible
- **Patch (X.Y.Z)**: Bug fixes, backward compatible

### Release Process

1. **Update the version**:
   ```bash
   # Edit src/mlx_serve/__init__.py
   __version__ = "X.Y.Z"
   ```

2. **Update the changelog**:
   ```bash
   # Add a new section to CHANGELOG.md with the changes
   ```

3. **Commit and push**:
   ```bash
   git add src/mlx_serve/__init__.py CHANGELOG.md
   git commit -m "Bump version to X.Y.Z"
   git push origin main
   ```

4. **Tag and push the tag**:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

5. **Automated deployment**: Pushing the tag triggers the publish workflow which:
   - Verifies the tag matches `__version__`
   - Runs the full test suite
   - Builds the package (wheel + sdist)
   - Publishes to PyPI via Trusted Publishing
   - Creates a GitHub Release with auto-generated notes

### Publish Pipeline

- **Trigger**: Git tag push matching `v*`
- **Workflow**: `.github/workflows/publish.yml`
- **Authentication**: PyPI Trusted Publishing (OIDC) — no tokens needed
- **Jobs**: `build` (test + build) → `publish` (upload to PyPI) → `github-release` (create release)

### Prerequisites for Maintainers

To deploy releases, maintainers need:
1. Write access to the repository
2. Permission to create `v*` tags
3. PyPI Trusted Publisher configured for the `mlx-serve` project

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
