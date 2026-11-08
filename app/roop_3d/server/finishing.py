"""Owned local finishing, artifact verification and NLE interchange planning."""
from __future__ import annotations

import base64
from fractions import Fraction
import json
import time
import uuid

import cv2
from fastapi import APIRouter, Request, Query, Response
from fastapi.responses import FileResponse
from pydantic import Field

from roop_3d.audit.isolation import sha256_file
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract, runtime_path
from roop_3d.core.storage import owned
from roop_3d.jobs.schema import JobState
from roop_3d.media.finishing import Finisher, FinishingOptions
from roop_3d.media.provenance import capabilities, verify_manifest
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.video_writer import encode_video

router = APIRouter(prefix='/api/v1')


class FinishRequest(Contract):
    path: str = Field(min_length=1, max_length=1024)
    finishing: FinishingOptions = FinishingOptions()


class FinishPreviewRequest(FinishRequest):
    timestamp: float = Field(default=0, ge=0, le=86400)


@router.get('/finishing/capabilities')
def finishing_capabilities():
    return {'provenance': capabilities(), 'passes': ['alpha', 'occluder', 'normal', 'depth', 'uv', 'relit'],
            'geometry_requires': 'frame-aligned mesh blend projection', 'lut_max_bytes': 1048576,
            'lut_size': [2, 33]}


@router.post('/finishing/preview')
def finishing_preview(value: FinishPreviewRequest, request: Request):
    media = request.app.state.media
    path = media.path(value.path)
    began, origin, selected = time.monotonic(), None, None
    iterator = VideoReader(path, hardware='software').frames()
    try:
        for decoded in iterator:
            if time.monotonic() - began > 15 or decoded.index >= 10000:
                raise DomainError('FINISH_PREVIEW_BUDGET', 'Preview exceeds the local decode budget', 422)
            if origin is None:
                origin = decoded.timestamp.seconds
            if decoded.timestamp.seconds - origin >= Fraction(str(value.timestamp)):
                selected = decoded
                break
    finally:
        iterator.close()
    if selected is None:
        raise DomainError('FRAME_OUT_OF_RANGE', 'Requested preview is beyond the video', 416)
    image, index = selected.image, selected.index
    image = Finisher(value.finishing).apply(image, index)
    ok, data = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise DomainError('FINISH_PREVIEW_FAILED', 'Finishing preview could not be encoded', 422)
    return {'image': 'data:image/jpeg;base64,' + base64.b64encode(data).decode(), 'frame_index': index}


@router.post('/finishing/render', status_code=201)
def finishing_render(value: FinishRequest, request: Request):
    state = request.app.state
    return encode_video(state.media.path(value.path), f'roop3d_finish_{uuid.uuid4().hex}.mp4', root=state.root,
                        output_dir=state.settings.output_root, audio_policy=state.settings.audio_policy,
                        finishing=value.finishing, manifest_metadata={'operation': 'post-production'})


def finished_artifact(request: Request, path: str):
    state = request.app.state
    media = state.media.path(path)
    if not media.is_relative_to(runtime_path(state.settings.output_root, state.root)):
        raise DomainError('FINISH_MANIFEST_DENIED', 'Select a published output', 422)
    manifest = owned(state.root, media.relative_to(state.root).as_posix() + '.manifest.json')
    if not manifest.is_file():
        raise DomainError('FINISH_MANIFEST_MISSING', 'Published render manifest is missing', 404)
    return media, manifest


@router.get('/finishing/manifest')
def finished_manifest(request: Request, path: str = Query(max_length=1024)):
    _, manifest = finished_artifact(request, path)
    return FileResponse(manifest, filename=manifest.name, media_type='application/json')


@router.get('/finishing/verify')
def finished_verify(request: Request, path: str = Query(max_length=1024)):
    media, manifest = finished_artifact(request, path)
    return verify_manifest(media, json.loads(manifest.read_text('utf-8')))


def completed_job(request: Request, job_id: str):
    job = request.app.state.jobs.get(job_id)
    if job.state != JobState.succeeded or job.output is None:
        raise DomainError('EXPORT_JOB_INCOMPLETE', 'Select a completed render', 409)
    return job


@router.get('/exports/{job_id}/manifest')
def export_manifest(job_id: str, request: Request, passes: bool = False):
    job = completed_job(request, job_id)
    relative = job.output.pass_manifest_path if passes else job.output.manifest_path
    if relative is None:
        raise DomainError('EXPORT_PASSES_MISSING', 'This render has no pass manifest', 404)
    return FileResponse(owned(request.app.state.root, relative), filename=relative.split('/')[-1], media_type='application/json')


@router.get('/exports/{job_id}/verify')
def export_verify(job_id: str, request: Request):
    job = completed_job(request, job_id)
    root = request.app.state.root
    if not job.output.baked_video:
        manifest = json.loads(owned(root, job.output.manifest_path).read_text('utf-8'))
        return {'passes': {name: sha256_file(owned(root, path))[1] == manifest['output_sha256'][name]
                           for name, path in job.output.passes.items()}}
    return verify_manifest(owned(root, job.output.output_path), json.loads(owned(root, job.output.manifest_path).read_text('utf-8')))


@router.get('/exports/{job_id}/timeline')
def export_timeline(job_id: str, request: Request, format: str | None = Query(default=None)):
    job = completed_job(request, job_id)
    state, output = request.app.state, job.output
    source = state.media.path(job.request.target_path)
    manifest = json.loads(owned(state.root, output.manifest_path).read_text('utf-8'))
    if sha256_file(source)[1] != manifest.get('input_sha256'):
        raise DomainError('EXPORT_SOURCE_CHANGED', 'The source changed after rendering', 409)
    if sha256_file(state.media.path(output.output_path))[1] != output.output_sha256:
        raise DomainError('EXPORT_OUTPUT_CHANGED', 'The rendered media changed after publication', 409)
    timeline = manifest.get('timeline') or manifest.get('stage12', {}).get('timeline')
    if timeline is None:
        raise DomainError('EXPORT_TIMELINE_MISSING', 'Render with export options to record actual swapped frames and scene cuts', 422)
    info = state.media.probe(job.request.target_path)
    if not info.fps_rational:
        raise DomainError('EXPORT_RATE_UNKNOWN', 'A rational frame rate is required', 422)
    rate = Fraction(info.fps_rational)
    count, origin, selected = 0, None, []
    start = Fraction(str(job.request.trim_start_seconds)) if job.request.trim_start_seconds is not None else None
    end = Fraction(str(job.request.trim_end_seconds)) if job.request.trim_end_seconds is not None else None
    for frame in VideoReader(source, hardware='software').frames():
        at = frame.timestamp.seconds
        if origin is None:
            origin = at
        if abs(at - origin - Fraction(frame.index, 1) / rate) > frame.timestamp.time_base:
            raise DomainError('EXPORT_VFR_UNSUPPORTED', 'NLE export requires CFR media; conform variable-rate footage first', 422)
        if (start is None or at >= start) and (end is None or at < end):
            selected.append(frame.index)
        count += 1
    if not selected or len(selected) != output.frames:
        raise DomainError('EXPORT_FRAME_MAP_UNSUPPORTED', 'Encoded media frame map differs from source; conform before interchange', 422)
    first, last = selected[0], selected[-1] + 1
    ranges = [[max(a, first), min(b, last)] for a, b in timeline['swapped_ranges'] if max(a, first) < min(b, last)]
    inspection = json.loads(owned(state.root, f'roop_3d/.runtime/inspections/{job.request.inspection_id}.json').read_text('utf-8'))
    if output.pass_manifest_path:
        pass_manifest_path = owned(state.root, output.pass_manifest_path)
        expected_manifest_hash = manifest.get('stage12', {}).get('vfx_manifest_sha256')
        if expected_manifest_hash and sha256_file(pass_manifest_path)[1] != expected_manifest_hash:
            raise DomainError('EXPORT_PASS_MANIFEST_CHANGED', 'The pass manifest changed after publication', 409)
        pass_manifest = json.loads(pass_manifest_path.read_text('utf-8'))
        for name, path in output.passes.items():
            if sha256_file(state.media.path(path))[1] != pass_manifest['output_sha256'].get(name):
                raise DomainError('EXPORT_PASS_CHANGED', 'A compositing pass changed after publication', 409)
    plan = {'name': 'Roop 3D composite', 'frameRate': str(rate), 'frameCount': count,
            'width': info.width, 'height': info.height, 'vfr': False, 'mediaOffset': first,
            'sourceUrl': source.as_uri(), 'swappedUrl': owned(state.root, output.output_path).as_uri() if output.baked_video else None,
            'alphaUrl': owned(state.root, output.passes['alpha']).as_uri() if 'alpha' in output.passes else None,
            'sceneCuts': sorted(set(timeline['scene_cuts'] + inspection.get('scene_cut_frames', []))), 'ranges': ranges}
    if format == 'fcpxml':
        from roop_3d.vfx_export import export_fcpxml
        return Response(content=export_fcpxml(plan), media_type='application/xml')
    if format in ('xml', 'xmeml'):
        from roop_3d.vfx_export import export_xmeml
        return Response(content=export_xmeml(plan), media_type='application/xml')
    if format == 'edl':
        from roop_3d.vfx_export import export_edl
        return export_edl(plan)
    return plan
