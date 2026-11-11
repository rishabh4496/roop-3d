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
from roop_3d.server.engine_gate import EngineBusy, EngineGate
from roop_3d.server.face_extract import FaceExtractService, router as face_extract_router
from roop_3d.server.live_preview import LivePreviewService, busy_response, router as live_preview_router
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog
from roop_3d.server.middleware import ApiAliasMiddleware, IntakeMiddleware
from roop_3d.server.contracts import ErrorEnvelope, Health
from roop_3d.server.pipeline_status import PipelineTracker, TelemetrySnapshot
from roop_3d.server.routes import router
from roop_3d.server.finishing import router as finishing_router
from roop_3d.server.security import ShareMode
from roop_3d.server.ws import EventBroker, event_stream, pipeline_status_stream


def release_media_models(media, worker=None) -> None:
    """Drop the app-level detector / recognizer sessions, and the render runner's own FaceSet reader models.
    Safe to call twice and with partially built services."""
    owners = [getattr(media, "detector", None), getattr(getattr(media, "analyzer", None), "recognizer", None),
              getattr(getattr(worker, "runner", None), "faceset_reader", None)]
    for owner in owners:
        release = getattr(owner, "release_for_build", None) or getattr(owner, "release", None)
        if callable(release):
            release()


def _universal_model_manager(root: Path, events: EventBroker, catalog: ModelCatalog):
    """The manifest-driven downloader. A broken manifest disables its endpoints (503) instead of taking the app down."""
    import logging
    from roop_3d.audit.local_models import UniversalModelDownloadManager
    rescan_lock = threading.Lock()

    def rescan() -> None:
        with rescan_lock:           # transfers finish on worker threads; the catalog scan is not re-entrant
            catalog.rescan_local()

    try:
        return UniversalModelDownloadManager(root, sink=events.publish, on_installed=rescan,
                                             hash_cache_path=root / "roop_3d/.runtime/models/verify_cache.json")
    except (AuditError, OSError, ValueError, KeyError) as error:
        logging.getLogger("roop3d.models").error("Universal model manifest unavailable: %s", error)
        return None


def create_app(settings: Settings | None = None, *, repo_root: Path | None = None,
               worker_enabled: bool | None = None, pipeline_runner=None, share: ShareMode | None = None) -> FastAPI:
    root = repo_root or Path(__file__).absolute().parents[2]
    if worker_enabled is None:
        # Temporary-root API contract tests do not decode their deliberately
        # synthetic queue fixtures. The installed application enables the worker.
        worker_enabled = repo_root is None
    settings = settings or Settings.from_env(ConfigStore(root).load() or Settings())
    started = time.monotonic()
    events = EventBroker()
    pipeline = PipelineTracker(events.publish, lambda: events.sequence)
    events.add_observer(pipeline.observe)
    jobs = JobStore(root, events.publish)
    from roop_3d.queue_manager import QueueManager
    queue = QueueManager(root, events.publish)
    engine_gate = EngineGate()
    catalog = ModelCatalog(cache_root=root, search_paths=lambda: app.state.settings.model_search_paths)
    from roop_3d.models.downloads import ModelDownloadManager
    from roop_3d.models.hub import ModelHubService
    model_downloads = ModelDownloadManager(catalog)
    model_hub = ModelHubService(catalog)
    universal_models = _universal_model_manager(root, events, catalog)
    media = MediaService(root, catalog, lambda: app.state.settings)
    if worker_enabled and pipeline_runner is None:
        from roop_3d.optimized_processor import make_stream_runner
        pipeline_runner = make_stream_runner(root, catalog, media, lambda: app.state.settings, pipeline_tracker=pipeline)
    worker = (JobWorker(root, jobs, lambda: app.state.settings, catalog, media,
                        pipeline_runner=pipeline_runner, engine_gate=engine_gate) if worker_enabled else None)
    if worker is not None:
        model_hub.engines_registry = worker.engines

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from roop_3d.security.log_sanitizer import install_log_sanitizer
        install_log_sanitizer()
        from roop_3d.models.native_libraries import register_native_libraries
        await run_in_threadpool(register_native_libraries)
        # Keep the local model catalog current before serving requests or starting jobs.
        app.state.model_scan = await run_in_threadpool(catalog.rescan_local)
        await run_in_threadpool(jobs.recover_interrupted)
        await run_in_threadpool(queue.recover_on_startup)
        # Stage 18: crash-consistency. Nothing is running yet, so every temp file, half-published output and orphan
        # FFmpeg child left by the previous process is stale by definition.
        from roop_3d.reliability.monitor import MONITOR
        from roop_3d.reliability.ledger import LEDGER
        from roop_3d.reliability.processes import REGISTRY
        from roop_3d.reliability.recovery import reconcile_on_startup
        REGISTRY.configure(root)
        app.state.orphan_sweep = await run_in_threadpool(REGISTRY.sweep_stale)
        try:
            reconciliation = await run_in_threadpool(
                reconcile_on_startup, root, jobs, lambda: app.state.settings,
                runner=worker.runner if worker is not None else None, queue=queue)
            app.state.startup_reconciliation = reconciliation.as_dict()
        except DomainError as error:  # an unreadable job store must be surfaced, not crash the boot
            app.state.startup_reconciliation = {"ok": False, "error": error.code}
        LEDGER.gauge("event_subscribers", lambda: events.subscriber_count)
        if worker is not None:
            LEDGER.gauge("worker_active_jobs", lambda: len(worker._futures))
        MONITOR.start()
        if worker is not None:
            worker.start()

        async def telemetry() -> None:
            while True:
                if events.subscriber_count:
                    value = await run_in_threadpool(probe_hardware, app.state.settings)
                    events.publish({"type": "system.telemetry", "hardware": value})
                    # Heartbeat frame so late joiners and idle dashboards stay current.
                    events.publish(pipeline.snapshot().model_dump(mode="json", exclude={"sequence", "timestamp"}))
                await asyncio.sleep(2)

        task = asyncio.create_task(telemetry())
        try:
            yield
        finally:
            # Jobs share the decoder pool, model catalog and live-frame hub. Let
            # active jobs finish before tearing down any resource they can use.
            await run_in_threadpool(app.state.live_studio.stop)
            if worker is not None:
                await run_in_threadpool(worker.stop)
            await run_in_threadpool(model_downloads.close)
            if universal_models is not None:
                await run_in_threadpool(universal_models.close)
            await run_in_threadpool(media.frames.close)
            # The detector and recognizer own GPU sessions; nothing may keep them alive past the app (a later app in the same process
            # would otherwise start with them still resident). The worker is already stopped, so no job can be using them.
            await run_in_threadpool(release_media_models, media, worker)
            # The live encoder owns a daemon thread (and may own a CUDA stream).
            # Stop it after the worker so no render can publish into a torn-down hub.
            await run_in_threadpool(app.state.live_hub.stop)
            await run_in_threadpool(MONITOR.stop)
            LEDGER.drop_gauge("event_subscribers")
            LEDGER.drop_gauge("worker_active_jobs")
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await run_in_threadpool(app.state.thread_benchmark.close)

    app = FastAPI(title="Roop 3D", version=__version__, docs_url=None, redoc_url=None, lifespan=lifespan,
                  responses={status: {"model": ErrorEnvelope} for status in (400, 403, 404, 409, 413, 416, 422, 429, 500, 503)})
    app.state.settings = settings
    app.state.root = root
    app.state.events = events
    app.state.pipeline = pipeline
    # Construction is metadata-only; compilation is an explicit prepare/build operation.
    from roop_3d.trt_session_builder import TensorRTSessionBuilder
    app.state.trt_builder = TensorRTSessionBuilder(telemetry=pipeline.compilation_event,
                                                 hardware_profile="12gb" if settings.hardware_profile == "12gb" else "6gb")
    app.state.share = share
    app.state.engine_gate = engine_gate
    app.state.jobs = jobs
    app.state.queue = queue
    app.state.models = catalog
    app.state.model_hub = model_hub
    app.state.model_downloads = model_downloads
    app.state.universal_models = universal_models
    app.state.media = media
    app.state.projects = ProjectStore(root, media, catalog, jobs)
    app.state.worker = worker
    app.state.config_lock = threading.Lock()
    from roop_3d.thread_benchmark import ThreadBenchmark
    app.state.thread_benchmark = ThreadBenchmark()
    app.state.previews = PreviewService(app.state)
    app.state.live_previews = LivePreviewService(app.state)
    from roop_3d.server.live_stream import make_hub
    app.state.live_hub = make_hub()
    from roop_3d.server.live_studio import LiveStudio, router as studio_router
    app.state.live_studio = LiveStudio(app.state)
    app.include_router(studio_router)
    if worker is not None and hasattr(worker.runner, "live_hub"):
        worker.runner.live_hub = app.state.live_hub       # finished render frames are offered to the binary stream
    app.state.face_extract = FaceExtractService(app.state)

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

    @app.exception_handler(EngineBusy)
    async def engine_busy(request: Request, error: EngineBusy) -> JSONResponse:
        return busy_response(error.stage)

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

    @app.get("/api/v1/hardware", response_model=Hardware)
    @app.get("/api/v1/system/hardware", response_model=Hardware)
    def hardware() -> Hardware:
        return probe_hardware(app.state.settings)

    @app.get("/api/v1/system/telemetry", response_model=TelemetrySnapshot)
    def telemetry_snapshot() -> TelemetrySnapshot:
        value = probe_hardware(app.state.settings)
        return TelemetrySnapshot(**value.model_dump(), pipeline=app.state.pipeline.snapshot())

    @app.get("/api/v1/meta")
    def meta() -> dict[str, Any]:
        return {"mock": False, "application": "roop_3d", "version": __version__,
                "api_versions": ["v1"], "share_mode": app.state.share is not None}

    app.include_router(router)
    app.include_router(workstation_router)
    app.include_router(preview_router)
    app.include_router(live_preview_router)
    from roop_3d.server.live_stream import preview_stream, router as live_stream_router
    app.include_router(live_stream_router)
    app.include_router(face_extract_router)
    from roop_3d.server.media_frames import router as media_frames_router
    app.include_router(media_frames_router)
    from roop_3d.server.face_bank import router as face_bank_router
    app.include_router(face_bank_router)
    from roop_3d.server.identity_lock import router as identity_lock_router
    app.include_router(identity_lock_router)
    from roop_3d.server.identity_assets import router as identity_assets_router
    app.include_router(identity_assets_router)
    from roop_3d.server.blender_routes import router as blender_router
    app.include_router(blender_router)
    from roop_3d.server.faceset_3d import router as faceset_3d_router
    app.include_router(faceset_3d_router)
    from roop_3d.server.target_lock import router as target_lock_router
    app.include_router(target_lock_router)
    from roop_3d.server.faceset_builder import router as builder_router
    from roop_3d.faceset_builder import FacesetBuilder
    app.state.faceset_builder = FacesetBuilder(app.state)
    app.include_router(builder_router)
    from roop_3d.server.orchestrator import router as orchestrator_router
    app.include_router(orchestrator_router)
    from roop_3d.server.professional_media import router as professional_media_router
    app.include_router(professional_media_router)
    app.include_router(finishing_router)
    from roop_3d.server.queue_routes import router as queue_router
    app.include_router(queue_router)
    from roop_3d.server.installer_routes import router as installer_router
    app.include_router(installer_router)
    from roop_3d.server.model_manifest_routes import router as model_manifest_router
    app.include_router(model_manifest_router)
    from roop_3d.server.reliability_routes import router as reliability_router
    app.include_router(reliability_router)
    from roop_3d.server.api import benchmark_router
    app.include_router(benchmark_router)
    from roop_3d.server.unified_state import router as unified_state_router, state as unified_state
    app.state.unified_state = unified_state
    app.include_router(unified_state_router)
    app.add_api_websocket_route("/ws/telemetry", event_stream)
    app.add_api_websocket_route("/ws/events", event_stream)
    app.add_api_websocket_route("/ws/queue", event_stream)
    app.add_api_websocket_route("/ws/v1/pipeline/status", pipeline_status_stream)
    app.add_api_websocket_route("/ws/v1/preview/stream", preview_stream)

    mount_workstation(app, root)
    if settings.dev_cors:
        app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5174"], allow_methods=["GET", "PUT", "POST", "DELETE"],
                           allow_headers=["Content-Type", "Idempotency-Key"],
                           expose_headers=["Retry-After", "X-Preview-Cache", "X-Preview-Frame-Index", "X-Preview-Config-Hash",
                                           "X-Preview-Media-Id", "X-Preview-Meta"])
    app.add_middleware(ApiAliasMiddleware)
    app.add_middleware(IntakeMiddleware, settings=settings, share=share)
    return app
