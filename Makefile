.PHONY: install mlx-start mlx-dev mlx-stop status models download-tts download-stt

# Install / sync dependencies
install:
	uv sync

# Start the MLX manager (runs in foreground; Ctrl+C to stop)
mlx-start:
	uv run mlx-serve start

# Start with auto-reload on code changes (dev mode)
mlx-dev:
	uv run uvicorn mlx_serve.main:app --host 0.0.0.0 --port 8095 --reload

# Stop any running manager process and any active model subprocesses
mlx-stop:
	-pkill -f "mlx-serve" 2>/dev/null || true
	-pkill -f "uvicorn mlx_serve" 2>/dev/null || true
	-pkill -f "mlx_lm.server"       2>/dev/null || true
	-pkill -f "mlx_vlm.server"      2>/dev/null || true

# Show current model / process status
status:
	@curl -s http://localhost:8095/status | python3 -m json.tool

# List all available models
models:
	@curl -s http://localhost:8095/v1/models | python3 -m json.tool

# Download the MLX-native chatterbox TTS model (one-time setup)
download-tts:
	uv run huggingface-cli download mlx-community/chatterbox-fp16

# Download the whisper-large-v3-turbo STT model (one-time setup)
download-stt:
	uv run huggingface-cli download mlx-community/whisper-large-v3-turbo
