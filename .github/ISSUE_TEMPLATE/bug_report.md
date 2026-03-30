---
name: Bug report
about: Something is broken or behaving unexpectedly
labels: bug
---

## Describe the bug

A clear description of what went wrong.

## To reproduce

Steps to reproduce the behavior:

1. Model type and name from `models.yaml`
2. Exact curl command or API call
3. What happened

## Expected behavior

What you expected to happen.

## Logs

If the model failed to load, paste the log:

```bash
curl http://localhost:8095/v1/status/logs/<model-name>
# or
cat /tmp/mlx-manager-logs/<model-name>.log
```

Manager stdout (from the terminal where you ran `make mlx-start`):

```
paste here
```

## Environment

- macOS version:
- Mac model (e.g. M2 Pro, M3 Max):
- Unified memory (GB):
- Python version (`python3 --version`):
- mlx-manager version (`curl http://localhost:8095/v1/version`):
- mlx-lm version (`uv run python -c "import mlx_lm; print(mlx_lm.__version__)"`):
