"""On-demand, bounded true-3D head previews; no sessions or weights at API boot.

A preview is produced only by the connected true-3D full-head renderer (``roop_3d.pipeline.true3d_renderer``, see
``server/true3d.py``): the same ``ReferencePipeline`` path as a render job (tracking replayed to the playhead, MakeHuman head
render, semantic composite), never a substitute. While no renderer is wired ``POST /workstation/preview`` answers HTTP 503
``RENDERER_NOT_WIRED`` ("3D renderer not wired"); an option the renderer does not implement is a structured 422.
"""
from __future__ import annotations

import asyncio
import base64
from contextlib import nullcontext
from threading import Event, Lock
from typing import Any, Literal
import os

import cv2
import numpy as np
from fastapi import APIRouter, Request
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.faceset.identity_blend import IdentityBlendSpec
from roop_3d.jobs.schema import JobRequest
from roop_3d.server.true3d import NOT_WIRED_CODE, NOT_WIRED_MESSAGE, True3DSettings, require_renderer

router = APIRouter(prefix='/api/v1/workstation')
#: Numeric parameters a sweep may vary: the ones that change pixels of the full-frame 3D compositor (mask feather width and
#: Laplacian blend bands). Controls the compositor does not implement (colour match, relighting, specular sheen) are not offered.
AxisParameter = Literal['feather_px', 'pyramid_levels']


class SweepAxis(Contract):
    parameter: AxisParameter
    values: tuple[float, float, float]

    @model_validator(mode='after')
    def bounds(self):
        for value in self.values:
            WorkstationParameters.model_validate({self.parameter: value})
        if len(set(self.values)) != 3:
            raise ValueError('Sweep coordinates must be distinct')
        return self


class PreviewRequest(Contract):
    request: JobRequest
    timestamp: float = Field(ge=0, le=86400)
    frame_index: int | None = Field(default=None, ge=0, le=10_000_000)
    parameters: WorkstationParameters = WorkstationParameters()
    x: SweepAxis | None = None
    y: SweepAxis | None = None
    # Optional multi-FaceSet identity recipe; replaces the mapped FaceSet's canonical identity.
    identity_blend: IdentityBlendSpec | None = None
    # Head topology engine, shader mode and perspective camera for this preview (server defaults when omitted).
    true3d: True3DSettings = True3DSettings()

    @property
    def effective_parameters(self) -> WorkstationParameters:
        return self.parameters if 'parameters' in self.model_fields_set else self.request.workstation_parameters or self.parameters

    @model_validator(mode='after')
    def axes(self):
        if bool(self.x) != bool(self.y) or self.x and self.y and self.x.parameter == self.y.parameter:
            raise ValueError('Supply two different axes or neither')
        if self.request.processing != 'local':
            raise ValueError('Preview processing must be local')
        return self


def encode(image: np.ndarray, max_edge: int = 1920) -> str:
    height, width = image.shape[:2]
    scale = min(1, max_edge / max(height, width))
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise DomainError('PREVIEW_ENCODE_FAILED', 'Preview encoding failed', 500)
    return 'data:image/jpeg;base64,' + base64.b64encode(data).decode('ascii')


def preview_preroll() -> int:
    """Frames before the target that still run the full render (``ROOP3D_PREVIEW_PREROLL``; 0 = every replayed frame)."""
    try:
        return max(0, int(os.environ.get("ROOP3D_PREVIEW_PREROLL", "6")))
    except ValueError:
        return 6


def _pack(image: Any) -> Any:
    return encode(image) if isinstance(image, np.ndarray) else image


class PreviewService:
    MAX_FRAMES = 1200
    TIME_BUDGET = 120

    def __init__(self, state):
        self.state = state
        self.lock = Lock()
        # The scheduler admits GPU work for previews; it is the same one the render worker uses (None-safe for API-only apps).
        self.scheduler = state.worker.scheduler if state.worker is not None else _scheduler()

    def render(self, value: PreviewRequest, cancel: Event, *, raw: bool = False, final_only: bool = False):
        """Run one preview through the connected true-3D renderer. `raw=True` keeps buffers as ndarrays instead of JPEG data URLs.

        Inference is a gate "preview" section: it is refused immediately (EngineBusy, 503) while a job, TensorRT compile or
        Start-button model re-bind owns the engine, and the owner in turn waits for this call to finish before it tears down or
        re-binds models. Without a wired renderer this raises ``RENDERER_NOT_WIRED`` (503) before touching any engine.
        The preview runs inside ``pooled_sessions()``: the models it builds stay warm for the next preview, and the gate owner
        evicts them the moment a job or a re-bind needs the engine.
        """
        renderer = require_renderer()
        from roop_3d.core.ort_session import pooled_sessions
        from roop_3d.engine.registry import EngineRegistry
        EngineRegistry().validate(value.request.engine_id)
        gate = getattr(self.state, 'engine_gate', None)
        if not self.lock.acquire(blocking=False):
            raise DomainError('PREVIEW_BUSY', 'Another preview is running; try again shortly', 409)
        try:
            with gate.preview_section() if gate is not None else nullcontext():
                with pooled_sessions():
                    result = renderer.render_preview(self.state, value, cancel, raw=raw, final_only=final_only)
        finally:
            self.lock.release()
        if not isinstance(result, dict) or 'buffers' not in result:
            raise DomainError('PREVIEW_RENDERER_CONTRACT', 'The 3D renderer returned no buffers', 500)
        if not raw:
            # The renderer packs its own buffers; an ndarray it left raw is packed here so the response is always JSON.
            result = {**result, 'buffers': {key: _pack(image) for key, image in result['buffers'].items()}}
            if 'cells' in result:
                result['cells'] = [{**cell, 'image': _pack(cell['image'])} for cell in result['cells']]
        result.setdefault('true3d', value.true3d.model_dump(mode='json'))
        return result


def _scheduler():
    from roop_3d.engine.scheduler import RuntimeScheduler
    return RuntimeScheduler()


@router.post('/preview')
async def preview(value: PreviewRequest, request: Request):
    # Fail fast and honestly before a worker thread is spent: no connected renderer, no preview.
    try:
        require_renderer()
    except DomainError:
        raise DomainError(NOT_WIRED_CODE, NOT_WIRED_MESSAGE, 503) from None
    cancel = Event()
    work = asyncio.create_task(run_in_threadpool(request.app.state.previews.render, value, cancel))
    try:
        while not work.done():
            if await request.is_disconnected():
                cancel.set()
            await asyncio.wait({work}, timeout=.1)
        return await work
    finally:
        cancel.set()
        # Observe completion even after client cancellation so no worker task escapes its lifecycle.
        if not work.done():
            await asyncio.shield(work)
