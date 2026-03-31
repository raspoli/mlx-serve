# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-31

### Added

- OpenAI-compatible API endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/audio/speech`, `/v1/audio/transcriptions`)
- Single-model memory enforcement with automatic hot-swapping
- Support for text, vision, embedding, TTS, and STT model types
- Subprocess isolation for text/vision models, in-process for embeddings
- YAML-based model configuration with multi-path discovery
- CLI with `init`, `start`, `stop`, `status`, and `logs` commands
- Observability: request metrics, memory monitoring, event logging
- JSONL audit logs for requests and model lifecycle events
- Dashboard endpoint (`/v1/dashboard`) for at-a-glance status
