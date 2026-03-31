# Developer Makefile — end users use the `mlx-serve` CLI directly.
# See: mlx-serve --help

.PHONY: install dev stop test lint build publish clean

# Install / sync dependencies
install:
	uv sync

# Start with auto-reload on code changes (dev mode)
dev:
	uv run uvicorn mlx_serve.main:app --host 0.0.0.0 --port 8095 --reload

# Stop any running processes
stop:
	-pkill -f "mlx-serve" 2>/dev/null || true
	-pkill -f "uvicorn mlx_serve" 2>/dev/null || true
	-pkill -f "mlx_lm.server"       2>/dev/null || true
	-pkill -f "mlx_vlm.server"      2>/dev/null || true

# Run tests
test:
	uv run pytest tests/ -v

# Lint and type check
lint:
	uv run ruff check src/
	uv run ruff format --check src/

# Format code
format:
	uv run ruff format src/

# Build wheel and sdist
build:
	uv build

# Publish to PyPI (requires UV_PUBLISH_TOKEN or --token)
publish: build
	uv publish

# Remove build artifacts
clean:
	rm -rf dist/ build/ *.egg-info src/mlx_serve/__pycache__ src/mlx_serve/*.pyc
