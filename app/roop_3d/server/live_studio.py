"""Local camera studio. Host-frame egress is measured and never advertised as zero-copy."""
from __future__ import annotations

import importlib.util
import sys
import threading
import time
from collections import deque
from contextlib import ExitStack
from typing import Literal

import cv2
import numpy as np
from fastapi import APIRouter, Request
from pydantic import Field
from starlette.concurrency import run_in_threadpool

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

router = APIRouter(prefix="/api/v1/studio")


class StudioConfig(Contract):
    source_faceset_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    swapper_model_id: str | None = Field(default=None, pattern=r"^[a-z0-9][a-z0-9_]{0,95}$")
    input: Literal["webcam", "screen"] = "webcam"
    device: int = Field(default=0, ge=0, le=32)
    backend: Literal["auto", "dshow", "msmf", "v4l2"] = "auto"
    width: int = Field(default=640, ge=128, le=1920)
    height: int = Field(default=480, ge=128, le=1080)
    fps: Literal[30, 60] = 30
    virtual_camera: bool = False
    spout: bool = False
    render_mode: Literal["standard", "pose_3d", "reenactment_3d"] = "standard"


class LiveStudio:
    def __init__(self, state):
        self.state = state
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._status = "stopped"
        self.error = None
        self.frames = 0
        self.latencies = deque(maxlen=300)

    def snapshot(self):
        with self._lock:
            samples = list(self.latencies)
            return {"state": self._status, "error": self.error, "frames": self.frames,
                    "capture_to_publish_p95_ms": float(np.percentile(samples, 95)) if samples else None,
                    "latency_target_ms": 35, "end_to_end_verified": False, "zero_copy": False,
                    "transport": "host BGR; browser JPEG upload",
                    "virtual_camera_available": importlib.util.find_spec("pyvirtualcam") is not None,
                    "screen_available": importlib.util.find_spec("mss") is not None,
                    "spout_available": sys.platform == "win32" and importlib.util.find_spec("SpoutGL") is not None,
                    "ndi_available": False}

    def start(self, config):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise DomainError("STUDIO_RUNNING", "Stop the current live studio before starting another", 409)
            if self.state.engine_gate.busy_reason():
                raise DomainError("STUDIO_ENGINE_BUSY", "Render or model compilation owns the engine", 409)
            self._stop.clear()
            self._status, self.error, self.frames = "starting", None, 0
            self.latencies.clear()
            self._thread = threading.Thread(target=self._run, args=(config,), name="roop3d-live-studio", daemon=True)
            self._thread.start()
        return self.snapshot()

    def stop(self):
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
        if thread is not None and thread.is_alive():
            raise DomainError("STUDIO_STOPPING", "Waiting for capture/inference to release; retry stop", 409)
        with self._lock:
            self._status = "stopped"
        return self.snapshot()

    def _run(self, config):
        from roop_3d.engine.pipeline_runner import MasterPipelineRunner
        from roop_3d.engine.volumetric.session import ReferencePipeline
        from roop_3d.engine.runtime import RuntimeConfig
        session = None
        reader = None
        admission = None
        scheduler = self.state.previews.scheduler
        try:
            settings = self.state.settings
            admission = scheduler.try_admit("live-studio", settings)
            if admission is None:
                raise DomainError("STUDIO_CAPACITY", "Insufficient capacity for live inference", 409)
            with ExitStack() as stack:
                # Hold the preview lease through cleanup: model rebinds wait until this session stops.
                stack.enter_context(self.state.engine_gate.preview_section())
                runner = MasterPipelineRunner(self.state.root, self.state.models, self.state.media, lambda: settings)
                reader = runner.faceset_reader
                stack.callback(reader.release)
                source = reader.read(runner._faceset_path(config.source_faceset_id), faceset_id=config.source_faceset_id)
                provider = settings.preferred_provider if admission.profile != "cpu" and settings.preferred_provider != "auto" else admission.provider
                runtime = RuntimeConfig(provider=provider, hardware_profile=admission.profile,
                                        reserve_mib=settings.memory_safety_reserve_mb)
                session = ReferencePipeline(self.state.models, runtime)
                stack.callback(session.release_session)
                session.select_swapper(config.swapper_model_id or settings.swapper_model_id)
                session.prepare_session(source, retarget="off", render_mode=config.render_mode)
                if session.swapper.model is not None:
                    session.swapper.model.morph_factor = settings.dfm_morph_factor
                if config.input == "webcam":
                    if config.backend in ("dshow", "msmf") and sys.platform != "win32" or config.backend == "v4l2" and not sys.platform.startswith("linux"):
                        raise DomainError("STUDIO_BACKEND", "Capture backend does not match this operating system", 422)
                    backend = {"auto": cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_V4L2,
                               "dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "v4l2": cv2.CAP_V4L2}[config.backend]
                    capture = cv2.VideoCapture(config.device, backend)
                    stack.callback(capture.release)
                    if not capture.isOpened():
                        raise DomainError("STUDIO_CAPTURE", "Camera could not be opened", 422)
                    for key, value in ((cv2.CAP_PROP_FRAME_WIDTH, config.width), (cv2.CAP_PROP_FRAME_HEIGHT, config.height),
                                       (cv2.CAP_PROP_FPS, config.fps), (cv2.CAP_PROP_BUFFERSIZE, 1)):
                        capture.set(key, value)
                    def read_frame():
                        ok, frame = capture.read()
                        if not ok:
                            raise DomainError("STUDIO_CAPTURE", "Camera disconnected or returned no frame", 503)
                        return frame
                else:
                    import mss
                    screen = stack.enter_context(mss.mss())
                    if config.device + 1 >= len(screen.monitors):
                        raise DomainError("STUDIO_SCREEN", "Selected screen is unavailable", 422)
                    def read_frame():
                        return np.asarray(screen.grab(screen.monitors[config.device + 1]))[:, :, :3].copy()
                camera = sender = None
                if config.virtual_camera:
                    import pyvirtualcam
                    camera = stack.enter_context(pyvirtualcam.Camera(config.width, config.height, config.fps, fmt=pyvirtualcam.PixelFormat.BGR))
                if config.spout:
                    if sys.platform != "win32":
                        raise DomainError("STUDIO_SPOUT", "Spout is available on Windows", 422)
                    import SpoutGL
                    from SpoutGL.enums import GL_BGR_EXT
                    sender = stack.enter_context(SpoutGL.SpoutSender())
                    sender.setSenderName("Roop 3D")
                with self._lock:
                    self._status = "running"
                while not self._stop.is_set():
                    began = time.perf_counter()
                    frame = cv2.resize(read_frame(), (config.width, config.height))
                    result = session.infer_frame(frame)
                    output = np.ascontiguousarray(session.composite_frame(frame, result))
                    if camera is not None:
                        camera.send(output)
                    if sender is not None:
                        if not sender.sendImage(output, config.width, config.height, GL_BGR_EXT, False, 0):
                            raise DomainError("STUDIO_SPOUT", "Spout failed to send the frame", 503)
                    self.state.live_hub.publish(output, frame_index=self.frames, timestamp_s=self.frames / config.fps)
                    with self._lock:
                        self.frames += 1
                        self.latencies.append((time.perf_counter() - began) * 1000)
                    self._stop.wait(max(0, 1 / config.fps - (time.perf_counter() - began)))
        except Exception as error:
            with self._lock:
                self.error = str(error)[:400]
                self._status = "error"
        finally:
            if admission is not None:
                scheduler.release("live-studio")
            with self._lock:
                if self._status != "error":
                    self._status = "stopped"


@router.get("")
def status(request: Request):
    return request.app.state.live_studio.snapshot()


@router.post("/start")
async def start(config: StudioConfig, request: Request):
    return await run_in_threadpool(request.app.state.live_studio.start, config)


@router.post("/stop")
async def stop(request: Request):
    return await run_in_threadpool(request.app.state.live_studio.stop)
