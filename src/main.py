"""
main.py — FastAPI application entry point for the MLX model manager.

Start with:
    make mlx-start
or:
    uvicorn src.main:app --host 0.0.0.0 --port 8090
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

import inline_manager
import process_manager
from router import close_client, health as _bare_health, router, status as _bare_status


@asynccontextmanager
async def lifespan(app: FastAPI):
    watcher_task = asyncio.create_task(process_manager.start_inactivity_watcher())
    inline_watcher_task = asyncio.create_task(inline_manager.start_inactivity_watcher())
    try:
        yield
    finally:
        watcher_task.cancel()
        inline_watcher_task.cancel()
        await process_manager.unload()
        await inline_manager.unload()
        await close_client()


app = FastAPI(
    title="MLX Model Manager",
    description="OpenAI-compatible proxy that manages one mlx-lm/mlx-vlm subprocess at a time.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/v1")

# /status and /health also accessible without /v1 prefix for operator convenience
app.add_api_route("/status", _bare_status, include_in_schema=False)
app.add_api_route("/health", _bare_health, include_in_schema=False)
