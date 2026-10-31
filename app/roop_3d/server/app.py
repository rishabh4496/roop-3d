"""Minimal local service; no model, diffusion or restricted-weight imports."""
from __future__ import annotations

from pathlib import Path
import asyncio
from contextlib import asynccontextmanager
import threading
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.exceptions import HTTPException
from starlette.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from roop_3d import __version__
from roop_3d.core.hardware import Hardware, probe_hardware
from roop_3d.core.integrity import isolation_status
from roop_3d.audit.isolation import AuditError
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import ConfigStore, Settings
from roop_3d.jobs.store import JobStore
from roop_3d.jobs.worker import JobWorker
from roop_3d.jobs.project_store import ProjectStore
from roop_3d.server.workstation import router as workstation_router
from roop_3d.server.workstation_preview import PreviewService, router as preview_router
from roop_3d.server.api import mount_workstation
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog
from roop_3d.server.middleware import IntakeMiddleware
from roop_3d.server.contracts import ErrorEnvelope, Health
from roop_3d.server.routes import router
from roop_3d.server.ws import EventBroker, event_stream


def create_app(settings: Settings | None = None, *, repo_root: Path | None = None,
               worker_enabled: bool | None = None, pipeline_runner=None) -> FastAPI:
    root = repo_root or Path(__file__).absolute().parents[2]
    if worker_enabled is None:
        # Temporary-root API contract tests do not decode their deliberately
        # synthetic queue fixtures. The installed application enables the worker.
        worker_enabled = repo_root is None
    settings = settings or Settings.from_env(ConfigStore(root).load() or Settings())
    started = time.monotonic()
    events = EventBroker()
    jobs = JobStore(root, events.publish)
    catalog = ModelCatalog(cache_root=root, search_paths=lambda: app.state.settings.model_search_paths)
    media = MediaService(root, catalog, lambda: app.state.settings)
    worker = (JobWorker(root, jobs, lambda: app.state.settings, catalog, media,
                        pipeline_runner=pipeline_runner) if worker_enabled else None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await run_in_threadpool(jobs.recover_interrupted)
        if worker is not None:
            worker.start()

        async def telemetry() -> None:
            while True:
                if events.subscriber_count:
                    value = await run_in_threadpool(probe_hardware, app.state.settings)
                    events.publish({"type": "system.telemetry", "hardware": value})
                await asyncio.sleep(2)

        task = asyncio.create_task(telemetry())
        try:
            yield
        finally:
            if worker is not None:
                await run_in_threadpool(worker.stop)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    app = FastAPI(title="Roop 3D", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan,
                  responses={status: {"model": ErrorEnvelope} for status in (400, 403, 404, 409, 413, 416, 422, 500)})
    app.state.settings = settings
    app.state.root = root
    app.state.events = events
    app.state.jobs = jobs
    app.state.models = catalog
    app.state.media = media
    app.state.projects = ProjectStore(root, media, catalog, jobs)
    app.state.worker = worker
    app.state.config_lock = threading.Lock()
    app.state.previews = PreviewService(app.state)

    def envelope(request: Request, code: str, message: str, status: int) -> JSONResponse:
        return JSONResponse({"error": {"code": code, "message": message,
                                       "request_id": getattr(request.state, "request_id", None)}}, status_code=status)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, error: HTTPException) -> JSONResponse:
        return envelope(request, f"HTTP_{error.status_code}", str(error.detail), error.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        return envelope(request, "VALIDATION_ERROR", "Request does not match the endpoint schema", 422)

    @app.exception_handler(Exception)
    async def internal_error(request: Request, error: Exception) -> JSONResponse:
        return envelope(request, "INTERNAL_ERROR", "The local service could not complete the request", 500)

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, error: DomainError) -> JSONResponse:
        return envelope(request, error.code, error.message, error.status)

    @app.exception_handler(AuditError)
    async def boundary_error(request: Request, error: AuditError) -> JSONResponse:
        return envelope(request, error.code, "Filesystem boundary or integrity check failed", 409)

    @app.get("/api/v1/health", response_model=Health)
    def health() -> dict[str, Any]:
        return {"status": "ok", "application": "roop_3d", "version": __version__,
                "pinokio_app_id": "roop-3d", "standalone": True,
                "uptime_seconds": max(0.0, time.monotonic() - started), "isolation": isolation_status(root)}

    @app.get("/api/v1/system/hardware", response_model=Hardware)
    def hardware() -> Hardware:
        return probe_hardware(app.state.settings)

    app.include_router(router)
    app.include_router(workstation_router)
    app.include_router(preview_router)
    app.add_api_websocket_route("/ws/telemetry", event_stream)
    app.add_api_websocket_route("/ws/events", event_stream)

    mount_workstation(app, root)
    if settings.dev_cors:
        app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174"], allow_methods=["GET", "PUT", "POST", "DELETE"],
                           allow_headers=["Content-Type", "Idempotency-Key"])
    app.add_middleware(IntakeMiddleware, settings=settings)
    return app
