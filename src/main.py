"""
main.py — FastAPI application entry point for the MLX model manager.

Start with:
    make mlx-start
or:
    uvicorn main:app --app-dir src --host 0.0.0.0 --port 8090
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import config
import events
import inline_manager
import logging_config
import metrics
import process_manager
from router import close_client, health as _bare_health, router, status as _bare_status
from router import get_metrics as _bare_metrics
from router import get_events as _bare_events
from router import dashboard as _bare_dashboard

# ---------------------------------------------------------------------------
# Initialise logging and monitoring subsystems
# ---------------------------------------------------------------------------

logging_config.setup(config.MONITORING.log_dir)
events.configure(config.MONITORING.log_dir, max_events=config.MONITORING.events_history_size)
metrics.configure(
    config.MONITORING.log_dir,
    requests_max=config.MONITORING.metrics_history_size,
    memory_max=360,
)

logger = logging.getLogger("mlx-serve")


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------


def _get_active_model() -> str | None:
    return process_manager.get_active_model_name() or inline_manager.get_active_model_name()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Server starting | port={config.MANAGER_PORT} | "
        f"models={len(config.MODELS)} | "
        f"inactivity_timeout={config.INACTIVITY_TIMEOUT}s | "
        f"logs={config.MONITORING.log_dir}"
    )
    events.emit(
        events.EventType.SERVER_START,
        detail={
            "port": config.MANAGER_PORT,
            "model_count": len(config.MODELS),
            "models": list(config.MODELS.keys()),
        },
    )

    # Take an initial memory baseline
    snap = metrics.take_memory_snapshot(event="server_start")
    logger.info(
        f"System memory: {snap.ram_total_gb}GB total, {snap.ram_available_gb}GB available "
        f"({snap.ram_percent}% used)"
    )

    watcher_task = asyncio.create_task(process_manager.start_inactivity_watcher())
    inline_watcher_task = asyncio.create_task(inline_manager.start_inactivity_watcher())
    memory_sampler_task = asyncio.create_task(
        metrics.start_memory_sampler(
            interval=config.MONITORING.memory_sample_interval,
            get_active_model=_get_active_model,
            get_subprocess_pid=process_manager.get_process_pid,
        )
    )

    try:
        yield
    finally:
        logger.info("Server shutting down")
        events.emit(events.EventType.SERVER_SHUTDOWN)

        watcher_task.cancel()
        inline_watcher_task.cancel()
        memory_sampler_task.cancel()
        await process_manager.unload()
        await inline_manager.unload()
        await close_client()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="MLX Model Manager",
    description="OpenAI-compatible proxy that manages one mlx-lm/mlx-vlm subprocess at a time.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/v1")

# /status, /health, /metrics, /events, /dashboard also accessible without /v1 prefix
app.add_api_route("/status", _bare_status, include_in_schema=False)
app.add_api_route("/health", _bare_health, include_in_schema=False)
app.add_api_route("/metrics", _bare_metrics, include_in_schema=False)
app.add_api_route("/events", _bare_events, include_in_schema=False)
app.add_api_route("/dashboard", _bare_dashboard, include_in_schema=False)
