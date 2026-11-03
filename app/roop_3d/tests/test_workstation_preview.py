"""Preview contract, real compositor buffers and admission/lifecycle regressions.

Inference is replaced by deterministic arrays here; these tests make no neural
fidelity or GPU performance claims.
"""
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from roop_3d.compositor.pipeline import FrameCompositor, CompositeOptions
from roop_3d.core.errors import DomainError
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.engine.volumetric.session import Replacement, StageLatency
from roop_3d.jobs.schema import JobRequest
from roop_3d.server import workstation_preview as module
from roop_3d.server.workstation_preview import PreviewRequest, collect_composite
from roop_3d.tests.test_stage12_worker import _fixture, _Reader


def arrays():
    image = np.full((64, 96, 3), 80, np.uint8)
    crop = np.zeros((32, 32, 3), np.uint8); crop[:] = (35, 95, 180)
    points = np.array([[18, 18], [42, 18], [42, 42], [18, 42], [30, 30]], np.float32)
    face = SimpleNamespace(box=np.array([18, 18, 42, 42]), landmarks106=points, pose=(0, 0, 0))
    result = SimpleNamespace(observations=(SimpleNamespace(track_id=0, observation=face),),
        replacements=(Replacement(0, crop, np.array([[1, 0, 14], [0, 1, 14]], np.float32), 0, 'fixture'),),
        latency=StageLatency(1, 2, 0, 3))
    return image, result


class FixtureSession:
    instances = []

    def __init__(self, catalog, config):
        self.compositor = FrameCompositor(parser=SimpleNamespace(predict=lambda image: None))
        self.released = False
        self.inferred = 0
        self.__class__.instances.append(self)

    def prepare_session(self, *args, **kwargs): pass
    def infer_frame(self, image, **kwargs):
        self.inferred += 1
        return arrays()[1]
    def release_session(self):
        self.released = True
        self.compositor.release()


def preview_fixture(tmp_path, monkeypatch, count: int = 8):
    app, _, payload = _fixture(tmp_path, count=count)
    original = module.MasterPipelineRunner

    def runner(*args, **kwargs):
        return original(*args, **kwargs, faceset_reader=_Reader())

    monkeypatch.setattr(module, 'MasterPipelineRunner', runner)
    monkeypatch.setattr(module, 'ReferencePipeline', FixtureSession)
    return app, {'request': payload, 'timestamp': .08}


def test_capture_is_non_invasive_and_matte_is_real_grayscale():
    image, result = arrays()
    plain = FrameCompositor(parser=SimpleNamespace(predict=lambda image: None))
    expected = plain.compose(image, result.replacements, {0: result.observations[0].observation.landmarks106}, scene_cut=True)
    session = SimpleNamespace(compositor=FrameCompositor(parser=SimpleNamespace(predict=lambda image: None)))
    buffers = collect_composite(session, image, result, WorkstationParameters())
    assert np.array_equal(expected, buffers['final'])
    assert buffers['matte'].ndim == 2
    assert buffers['matte'].max() > buffers['matte'].min()
    assert not np.array_equal(buffers['swapper'], buffers['final'])
    a = collect_composite(session, image, result, WorkstationParameters(feather_px=0))['final']
    b = collect_composite(session, image, result, WorkstationParameters(feather_px=10))['final']
    c = collect_composite(session, image, result, WorkstationParameters(feather_px=0))['final']
    assert np.array_equal(a, c) and not np.array_equal(a, b)


def test_single_batch_produces_nine_coordinate_specific_cells_and_releases(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    payload.update(x={'parameter': 'color_strength', 'values': [.1, .5, .9]},
                   y={'parameter': 'feather_px', 'values': [0, 5, 10]})
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        response = client.post('/api/v1/workstation/preview', json=payload)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['frame_index'] == 2 and len(value['cells']) == 9
    assert [(cell['row'], cell['column']) for cell in value['cells']] == [(y, x) for y in range(3) for x in range(3)]
    for cell in value['cells']:
        assert cell['parameters']['color_strength'] == cell['coordinates']['color_strength']
        assert cell['parameters']['feather_px'] == cell['coordinates']['feather_px']
        assert cell['image'].startswith('data:image/jpeg;base64,')
    assert len({cell['image'] for cell in value['cells']}) > 3
    assert FixtureSession.instances[-1].inferred == 3  # Replay once, compose nine variants.
    assert FixtureSession.instances[-1].released and app.state.previews.scheduler.active_count == 0
    assert app.state.jobs.list() == []


def test_buffers_and_bypass_match_parameter_contract(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        response = client.post('/api/v1/workstation/preview', json=payload)
        assert response.status_code == 200, response.text
        assert {'detector', 'pose', 'swapper', 'matte', 'restorer', 'color', 'final'} <= response.json()['buffers'].keys()
        escaped = {**payload, 'request': {**payload['request'], 'target_path': '../roop-ultimate/private.mp4'}}
        assert client.post('/api/v1/workstation/preview', json=escaped).status_code == 422
    parameters = WorkstationParameters(restoration_blend=.8, color_strength=.9, bypass=('restorer', 'color'))
    assert parameters.composite_options().restoration_blend == parameters.composite_options().color_strength == 0


@pytest.mark.parametrize('extra', [
    {'x': {'parameter': 'feather_px', 'values': [0, 5, 10]}},
    {'x': {'parameter': 'feather_px', 'values': [0, 5, 10]}, 'y': {'parameter': 'feather_px', 'values': [1, 2, 3]}},
    {'parameters': {'feather_px': -1}}, {'parameters': {'color_strength': float('inf')}},
    {'parameters': {'bypass': ['swapper']}},
    {'x': {'parameter': 'color_strength', 'values': [0, .5, 2]}, 'y': {'parameter': 'feather_px', 'values': [1, 2, 3]}},
])
def test_invalid_sweeps_are_rejected_before_inference(tmp_path, monkeypatch, extra):
    _, payload = preview_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValidationError): PreviewRequest.model_validate({**payload, **extra})


def test_cancel_and_failure_release_admission_and_busy_gate(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    cancel = Event(); cancel.set()
    with pytest.raises(DomainError): app.state.previews.render(PreviewRequest.model_validate(payload), cancel)
    assert app.state.previews.scheduler.active_count == 0
    assert app.state.previews.lock.acquire(blocking=False)
    try:
        with pytest.raises(DomainError, match='Another preview'): app.state.previews.render(PreviewRequest.model_validate(payload), Event())
    finally: app.state.previews.lock.release()


def test_new_parameters_affect_request_identity_without_rewriting_old_identity():
    payload = {'target_path': 'roop_3d/.runtime/media/target.mp4', 'inspection_id': 'a' * 32,
               'face_mappings': [{'track_id': 'track_0000', 'faceset_id': 'alice'}]}
    old = JobRequest.model_validate(payload)
    assert 'workstation_parameters' not in old.identity_payload()
    new = JobRequest.model_validate({**payload, 'workstation_parameters': {'feather_px': 10}})
    assert new.identity_payload() != old.identity_payload()


def test_preview_seek_at_later_timestamp_without_budget_exceeded(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch, count=1300)
    payload['timestamp'] = 48.0
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        response = client.post('/api/v1/workstation/preview', json=payload)
    assert response.status_code == 200, response.text
    value = response.json()
    assert value['frame_index'] == 1200
    assert value['timestamp'] >= 48.0
    assert FixtureSession.instances[-1].inferred == 1
    assert 'seeked to playhead keyframe' in value['notes'][0]


def test_six_gigabyte_profile_downscales_preview_frame(tmp_path, monkeypatch):
    import av
    import hashlib
    from fractions import Fraction
    from roop_3d.core.storage import atomic_json
    from roop_3d.media.schema import FaceInspection, FaceTrack, BoundingBox, PoseRange

    target_path = tmp_path / "roop_3d/.runtime/media/target_large.mp4"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(target_path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 800, 600, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 25)
        stream.codec_context.max_b_frames = 0
        for index in range(8):
            image = np.full((600, 800, 3), 100, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)

    digest = hashlib.sha256(target_path.read_bytes()).hexdigest()
    insp = FaceInspection(inspection_id="b" * 32, target_sha256=digest,
        detector_model_id="fixture", detector_sha256="0" * 64,
        tracks=(FaceTrack(track_id="track_0000", persistent_id=0, first_frame=0,
            last_frame=7, first_timestamp_seconds=0, last_timestamp_seconds=7 / 25,
            bounding_box=BoundingBox(x1=10, y1=10, x2=80, y2=80), pose_range=PoseRange(),
            thumbnail_base64="", sampled_detections=2),),
        decoded_frames=8, sampled_frames=8, truncated=False,
        tracking_method="iou_arcface_hungarian")
    atomic_json(tmp_path, f"roop_3d/.runtime/inspections/{insp.inspection_id}.json", insp.model_dump(mode="json"))

    app, payload = preview_fixture(tmp_path, monkeypatch)
    payload["request"]["target_path"] = "roop_3d/.runtime/media/target_large.mp4"
    payload["request"]["inspection_id"] = insp.inspection_id

    class FakeScheduled:
        profile = "6gb"
        provider = "CPUExecutionProvider"
        def release(self): pass
    monkeypatch.setattr(app.state.previews.scheduler, "try_admit", lambda key, settings, **kwargs: FakeScheduled())

    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/workstation/preview", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["profile"] == "6gb"
    assert data["width"] <= 512
    assert data["height"] <= 512
    assert max(data["width"], data["height"]) == 512

    # Now verify 12GB profile retains full native resolution
    FakeScheduled.profile = "12gb"
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response12 = client.post("/api/v1/workstation/preview", json=payload)
    assert response12.status_code == 200, response12.text
    data12 = response12.json()
    assert data12["profile"] == "12gb"
    assert data12["width"] == 800
    assert data12["height"] == 600
