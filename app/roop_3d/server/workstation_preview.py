"""On-demand, bounded local inference previews; no sessions or weights at API boot."""
from __future__ import annotations

import asyncio
import base64
from contextlib import closing
from fractions import Fraction
from threading import Event, Lock
import time
from typing import Literal
import uuid
from dataclasses import replace
from roop_3d.reconstruction.pose_blend import reblend

import cv2
import numpy as np
from fastapi import APIRouter, Request
from pydantic import Field, model_validator
from starlette.concurrency import run_in_threadpool

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.scheduler import RuntimeScheduler
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.compositor.restoration import GpenRestorer
from roop_3d.jobs.schema import JobRecord, JobRequest
from roop_3d.media.video_reader import VideoReader
from roop_3d.engine.edit_plan import FrameEditPlan

router = APIRouter(prefix='/api/v1/workstation')
AxisParameter = Literal['restoration_blend', 'feather_px', 'color_strength', 'pose_strength']


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
    parameters: WorkstationParameters = WorkstationParameters()
    x: SweepAxis | None = None
    y: SweepAxis | None = None

    @property
    def effective_parameters(self) -> WorkstationParameters:
        return self.parameters if 'parameters' in self.model_fields_set else self.request.workstation_parameters or self.parameters

    @model_validator(mode='after')
    def axes(self):
        if bool(self.x) != bool(self.y) or self.x and self.y and self.x.parameter == self.y.parameter:
            raise ValueError('Supply two different axes or neither')
        if self.request.processing != 'local':
            raise ValueError('Preview processing must be local')
        if any(axis and axis.parameter == 'pose_strength' for axis in (self.x, self.y)) and self.effective_parameters.pose_mode != 'mesh_blend':
            raise ValueError('Pose sweeps require mesh blend mode')
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


def collect_composite(session, image, result, parameters, *, capture=True, occluder_masks=None):
    """Fresh temporal state per cell prevents cross-cell contamination."""
    session.compositor.temporal.clear()
    buffers = {}

    def observe(stage, crop, inverse, alpha):
        size = (image.shape[1], image.shape[0])
        if stage == 'matte':
            projected = cv2.warpAffine(crop, inverse, size)
            buffers[stage] = np.maximum(buffers.get(stage, np.zeros(image.shape[:2], np.uint8)), projected)
        else:
            projected = cv2.warpAffine(crop, inverse, size)
            weight = cv2.warpAffine(np.ones(crop.shape[:2], np.float32), inverse, size)[:, :, None]
            base = buffers.get(stage, image.copy())
            buffers[stage] = np.clip(projected + base * (1 - weight), 0, 255).astype(np.uint8)

    landmarks = {item.track_id: item.observation.landmarks106 for item in result.observations}
    final = session.compositor.compose(image, result.replacements, landmarks, scene_cut=True,
        options=parameters.composite_options(), observer=observe if capture else None, occluder_masks=occluder_masks)
    buffers['final'] = final
    return buffers


class PreviewService:
    MAX_FRAMES = 1200
    TIME_BUDGET = 120

    def __init__(self, state):
        self.state = state
        self.lock = Lock()
        self.scheduler = state.worker.scheduler if state.worker is not None else RuntimeScheduler()

    def render(self, value: PreviewRequest, cancel: Event):
        if not self.lock.acquire(blocking=False):
            raise DomainError('PREVIEW_BUSY', 'Another preview is running; try again shortly', 409)
        key = 'preview-' + uuid.uuid4().hex
        admission = None
        session = None
        began = time.monotonic()

        def check():
            if cancel.is_set():
                raise DomainError('PREVIEW_CANCELLED', 'Preview cancelled', 409)
            if time.monotonic() - began > self.TIME_BUDGET:
                raise DomainError('PREVIEW_BUDGET', 'Preview exceeded the 120 second budget; use an earlier frame or shorter clip', 422)

        try:
            settings = self.state.settings
            admission = self.scheduler.try_admit(key, settings)
            if admission is None:
                raise DomainError('PREVIEW_CAPACITY', 'Rendering currently owns the available execution capacity', 409)
            runner = MasterPipelineRunner(self.state.root, self.state.models, self.state.media, lambda: settings)
            # Reuse target hashes, inspection identity and FaceSet validation without creating a job/checkpoint.
            target, _, _, sources, _, _ = runner._prepare_inputs(JobRecord(request=value.request), settings)
            edit_plan = FrameEditPlan(runner, value.request, sources)
            runtime = RuntimeConfig(provider=admission.provider, hardware_profile=admission.profile,
                                    reserve_mib=settings.memory_safety_reserve_mb)
            session = ReferencePipeline(self.state.models, runtime)
            session.prepare_session(next(iter(sources.values())), track_ids=set(sources), sources_by_track=sources,
                                    retarget='off' if value.effective_parameters.pose_mode == 'mesh_blend' else 'auto')
            base_parameters = value.effective_parameters
            parameters = base_parameters
            needs_restorer = parameters.composite_options().restoration_blend > 0 or any(
                axis and axis.parameter == 'restoration_blend' and max(axis.values) > 0 for axis in (value.x, value.y))
            needs_restorer = needs_restorer or bool(value.request.parameter_curves and any(key.value > 0 for key in value.request.parameter_curves.tracks.get('restorer_opacity', ())))
            if needs_restorer:
                if settings.face_restoration_policy in ('gfpgan', 'codeformer'):
                    raise DomainError('RESTORER_UNAVAILABLE', 'Selected restoration architecture has no verified local adapter', 503)
                session.compositor.restorer = GpenRestorer(self.state.models, runtime)
            check()
            selected = None
            origin = None
            with closing(VideoReader(target, hardware='software', cancel=cancel).frames()) as frames:
                for frame in frames:
                    check()
                    if frame.index >= self.MAX_FRAMES:
                        raise DomainError('PREVIEW_FRAME_BUDGET', 'Preview replays tracking from the beginning; select a frame within the first 1200 frames', 422)
                    if frame.image.shape[0] * frame.image.shape[1] > 3840 * 2160:
                        raise DomainError('PREVIEW_SIZE_BUDGET', 'Preview input exceeds the 4K frame budget', 422)
                    origin = frame.timestamp.seconds if origin is None else origin
                    edit_plan.apply(session, frame.index)
                    parameters = value.request.parameter_curves.evaluate(frame.index, base_parameters) if value.request.parameter_curves else base_parameters
                    result = session.infer_frame(frame.image, **parameters.inference_options())
                    selected = (frame, result, float(frame.timestamp.seconds - origin))
                    if frame.timestamp.seconds - origin >= Fraction(str(value.timestamp)):
                        break
                else:
                    raise DomainError('PREVIEW_FRAME_MISSING', 'No frame exists at the requested playhead time', 422)
            if selected is None:
                raise DomainError('PREVIEW_FRAME_MISSING', 'Target has no decodable frames', 422)
            frame, result, timestamp = selected
            if not result.replacements and value.request.face_segments is None:
                raise DomainError('PREVIEW_TRACK_MISSING', 'No mapped face is visible at this playhead frame', 422)
            response = {'frame_index': frame.index, 'timestamp': timestamp,
                        'width': frame.image.shape[1], 'height': frame.image.shape[0],
                        'profile': admission.profile, 'provider': admission.provider,
                        'notes': ['Single-frame preview, tracking replayed from frame zero; temporal smoothing resets per variant.',
                                  'Mesh blend: TPS at 0, perspective 3D proxy projection at 1; canonical geometry and camera are estimates.' if parameters.pose_mode == 'mesh_blend' else 'Reference retarget mode: observed dense landmarks; mesh projection is not selected.',
                                  'Matte uses the reference BiSeNet parser with landmark-hull fallback; no XSeg adapter.']}
            if value.x and value.y:
                cells = []
                for row, y in enumerate(value.y.values):
                    for column, x in enumerate(value.x.values):
                        check()
                        coordinates = {value.x.parameter: x, value.y.parameter: y}
                        cell = WorkstationParameters.model_validate({**parameters.model_dump(), **coordinates})
                        variant = result
                        if cell.pose_mode == 'mesh_blend' and cell.pose_strength != parameters.pose_strength:
                            replacements = []
                            for item in result.replacements:
                                if item.projection is None:
                                    raise DomainError('POSE_PROJECTION_MISSING', 'Mesh blend sweep requires geometric endpoint buffers', 409)
                                projection = reblend(item.projection, cell.pose_strength)
                                replacements.append(replace(item, image=projection.image, coverage=projection.coverage, projection=projection))
                            variant = replace(result, replacements=tuple(replacements))
                        buffers = collect_composite(session, frame.image, variant, cell, capture=False, occluder_masks=edit_plan.occluders(frame.image, variant))
                        cells.append({'row': row, 'column': column, 'coordinates': coordinates,
                                      'parameters': cell.model_dump(), 'image': encode(buffers['final'])})
                return {**response, 'cells': cells}
            check()
            buffers = collect_composite(session, frame.image, result, parameters, occluder_masks=edit_plan.occluders(frame.image, result))
            detection, pose = frame.image.copy(), frame.image.copy()
            for tracked in result.observations:
                face = tracked.observation
                x1, y1, x2, y2 = face.box.astype(int)
                cv2.rectangle(detection, (x1, y1), (x2, y2), (80, 245, 180), 2)
                for point in face.landmarks106:
                    cv2.circle(pose, tuple(np.rint(point).astype(int)), 2, (80, 245, 180), -1)
                cv2.putText(pose, 'Y/P/R ' + '/'.join(f'{p:.1f}' for p in face.pose),
                            (max(0, x1), max(20, y1)), cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 245, 180), 2)
            buffers.update(detector=detection, pose=pose)
            projection_info = []
            if parameters.pose_mode == 'mesh_blend':
                for item in result.replacements:
                    projection = getattr(item, 'projection', None)
                    if projection is None:
                        continue
                    points = projection.blend_pixels @ item.inverse[:, :2].T + item.inverse[:, 2]
                    for triangle in projection.visible_faces:
                        cv2.polylines(pose, [np.rint(points[triangle]).astype(np.int32)], True, (255, 180, 65), 1, cv2.LINE_AA)
                    projection_info.append({'track_id': item.track_id, 'geometry': 'ESTIMATED_CANONICAL_PROXY_68',
                        'camera': 'ESTIMATED_CROP_PINHOLE', 'reprojection_error_px': projection.reprojection_error,
                        'mesh_coverage_ratio': float(np.mean(projection.mesh_coverage > 0)), 'weight': parameters.pose_strength})
                response['pose_projection'] = projection_info
            replacement = result.replacements[0] if result.replacements else None
            source = (session.swapper.sources_by_track.get(replacement.track_id, sources.get(replacement.track_id)) if value.request.face_segments and replacement else sources.get(replacement.track_id) if replacement else None)
            if source is None:
                return {**response, 'buffers': {key: encode(image) for key, image in buffers.items()}, 'parameters': parameters.model_dump(), 'latency': result.latency.__dict__}
            view = next((view for view in source.views if view.index == replacement.reference_index), source.views[0])
            source_image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if source_image is not None:
                buffers['source'] = source_image
            return {**response, 'buffers': {key: encode(image) for key, image in buffers.items()},
                    'parameters': parameters.model_dump(), 'latency': result.latency.__dict__}
        finally:
            try:
                if session is not None:
                    session.release_session()
            finally:
                if admission is not None:
                    self.scheduler.release(key)
                self.lock.release()


@router.post('/preview')
async def preview(value: PreviewRequest, request: Request):
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
