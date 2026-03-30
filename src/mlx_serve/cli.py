"""
cli.py — mlx-serve command-line interface.

Usage:
    mlx-serve start [--host HOST] [--port PORT]
    mlx-serve status [--port PORT]
    mlx-serve models [--port PORT]
    mlx-serve pull <model> [--port PORT]
    mlx-serve init [--dir DIR] [--force]
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


def cmd_start(args: argparse.Namespace) -> None:
    """Start the uvicorn server."""
    import uvicorn

    from . import config  # triggers config discovery + validation

    port = args.port or config.MANAGER_PORT
    print(f"Starting mlx-serve on {args.host}:{port}")
    uvicorn.run(
        "mlx_serve.main:app",
        host=args.host,
        port=port,
    )


def cmd_status(args: argparse.Namespace) -> None:
    """Hit /status and pretty-print."""
    import httpx

    port = args.port or 8095
    try:
        resp = httpx.get(f"http://localhost:{port}/status", timeout=5)
        print(json.dumps(resp.json(), indent=2))
    except httpx.ConnectError:
        print(f"Error: cannot connect to mlx-serve on port {port}", file=sys.stderr)
        sys.exit(1)


def cmd_models(args: argparse.Namespace) -> None:
    """List configured models from the running server."""
    import httpx

    port = args.port or 8095
    try:
        resp = httpx.get(f"http://localhost:{port}/v1/models", timeout=5)
        data = resp.json()
        models = data.get("data", [])
        if not models:
            print("No models configured.")
            return
        for m in models:
            caps = ", ".join(m.get("capabilities", []))
            print(f"  {m['id']:30s} [{caps}]")
    except httpx.ConnectError:
        print(f"Error: cannot connect to mlx-serve on port {port}", file=sys.stderr)
        sys.exit(1)


def cmd_pull(args: argparse.Namespace) -> None:
    """Download a model via the running server's /v1/models/pull endpoint."""
    import httpx

    port = args.port or 8095
    try:
        with httpx.stream(
            "POST",
            f"http://localhost:{port}/v1/models/pull",
            json={"model": args.model},
            timeout=None,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    detail = data.get("detail", "")
                    if detail:
                        print(f"  [{status}] {detail}")
                    else:
                        print(f"  [{status}] {data.get('model', '')}")
    except httpx.ConnectError:
        print(f"Error: cannot connect to mlx-serve on port {port}", file=sys.stderr)
        print("Start the server first: mlx-serve start", file=sys.stderr)
        sys.exit(1)


def cmd_init(args: argparse.Namespace) -> None:
    """Generate a starter models.yaml."""
    target_dir = Path(args.dir).expanduser() if args.dir else Path.cwd()
    target = target_dir / "models.yaml"

    if target.exists() and not args.force:
        print(f"models.yaml already exists at {target}")
        print("Use --force to overwrite.")
        sys.exit(1)

    bundled = Path(__file__).parent / "_default_models.yaml"
    if not bundled.exists():
        print("Error: bundled default config not found", file=sys.stderr)
        sys.exit(1)

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(bundled, target)
    print(f"Created {target}")
    print("Edit this file to add your models, then run: mlx-serve start")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="mlx-serve",
        description="OpenAI-compatible MLX model manager for Apple Silicon",
    )
    sub = parser.add_subparsers(dest="command")

    # start
    p_start = sub.add_parser("start", help="Start the server")
    p_start.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_start.add_argument("--port", type=int, default=None, help="Port (default: from models.yaml or 8095)")

    # status
    p_status = sub.add_parser("status", help="Show server and model status")
    p_status.add_argument("--port", type=int, default=None, help="Server port (default: 8095)")

    # models
    p_models = sub.add_parser("models", help="List configured models")
    p_models.add_argument("--port", type=int, default=None, help="Server port (default: 8095)")

    # pull
    p_pull = sub.add_parser("pull", help="Download a model from HuggingFace")
    p_pull.add_argument("model", help="HuggingFace model path (e.g. mlx-community/Qwen2.5-7B-Instruct-4bit)")
    p_pull.add_argument("--port", type=int, default=None, help="Server port (default: 8095)")

    # init
    p_init = sub.add_parser("init", help="Generate a starter models.yaml")
    p_init.add_argument("--dir", default=None, help="Target directory (default: current directory)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing models.yaml")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "models": cmd_models,
        "pull": cmd_pull,
        "init": cmd_init,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
