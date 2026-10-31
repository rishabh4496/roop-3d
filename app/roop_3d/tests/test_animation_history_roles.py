"""Frame curves, protected role intervals and non-destructive compositor behavior."""
from types import SimpleNamespace
import base64
import json

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from roop_3d.core.animation import ParameterCurves, FaceSegment
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.core.errors import DomainError
from roop_3d.engine.edit_plan import FrameEditPlan
from roop_3d.jobs.schema import JobRequest
from roop_3d.server import workstation_preview as preview_module
from roop_3d.server.workstation_preview import PreviewRequest, collect_composite
from roop_3d.tests.test_workstation_preview import arrays, FixtureSession, preview_fixture
from roop_3d.compositor.pipeline import FrameCompositor


def test_linear_bezier_hold_and_exact_cut_boundaries():
    for easing, expected in [('linear', .25), ('ease-in-out', .15625), ('hold', 0)]:
        curves = ParameterCurves.model_validate({'tracks': {'swap_weight': [
            {'frame': 0, 'value': 0, 'easing': easing}, {'frame': 100, 'value': 1}]}})
        assert curves.evaluate(25, WorkstationParameters()).swap_weight == expected
        assert curves.evaluate(100, WorkstationParameters()).swap_weight == 1
        assert curves.evaluate(200, WorkstationParameters()).swap_weight == 1
    with pytest.raises(ValidationError): ParameterCurves.model_validate({'tracks': {'swap_weight': [{'frame': 0, 'value': 2}]}})
    with pytest.raises(ValidationError): ParameterCurves.model_validate({'tracks': {'mask_feather': [{'frame': 1, 'value': 2}, {'frame': 1, 'value': 3}]}})


def test_weight_zero_is_exact_source_and_erosion_changes_real_matte():
    image, result = arrays()
    session = SimpleNamespace(compositor=FrameCompositor(parser=SimpleNamespace(predict=lambda crop: None)))
    zero = collect_composite(session, image, result, WorkstationParameters(swap_weight=0))
    assert np.array_equal(zero['final'], image)
    normal = collect_composite(session, image, result, WorkstationParameters(erosion_px=0))
    eroded = collect_composite(session, image, result, WorkstationParameters(erosion_px=4))
    assert eroded['matte'].sum() < normal['matte'].sum()
    mask = np.ones((32, 32), np.float32)
    kept = collect_composite(session, image, result, WorkstationParameters(), occluder_masks={0: mask})
    assert np.array_equal(kept['final'], image)


def test_older_parameter_request_serialization_and_identity_are_stable():
    payload = {'target_path': 'local.mp4', 'inspection_id': 'a' * 32,
        'face_mappings': [{'track_id': 'track_0000', 'faceset_id': 'alice'}], 'workstation_parameters': {'feather_px': 5}}
    before = JobRequest.model_validate(payload)
    saved = before.model_dump(mode='json')
    after = JobRequest.model_validate(saved)
    assert before.identity_payload() == after.identity_payload()
    assert 'swap_weight' not in saved['workstation_parameters']
    assert 'face_segments' not in before.identity_payload()
    assert 'parameter_curves' not in before.identity_payload()


def test_interval_overlap_and_unknown_role_are_rejected():
    payload = {'target_path': 'local.mp4', 'inspection_id': 'a' * 32,
        'face_mappings': [{'track_id': 'track_0000', 'faceset_id': 'alice'}], 'face_segments': [
        {'cluster_id': 'one', 'track_id': 'track_0000', 'start_frame': 0, 'end_frame': 10, 'role': 'exclude'},
        {'cluster_id': 'two', 'track_id': 'track_0000', 'start_frame': 10, 'role': 'swap', 'faceset_id': 'alice'}]}
    with pytest.raises(ValidationError): JobRequest.model_validate(payload)
    payload['face_segments'][1]['start_frame'] = 11
    assert JobRequest.model_validate(payload).face_segments[1].contains(11)
    with pytest.raises(ValidationError): FaceSegment(cluster_id='one', track_id='track_0000', start_frame=0, role='unknown')


def test_role_plan_gates_inference_and_single_face_source_without_resetting_tracker(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    request = JobRequest.model_validate({**payload['request'], 'face_segments': [
        {'cluster_id': 'head', 'track_id': 'track_0000', 'start_frame': 0, 'end_frame': 1, 'role': 'exclude'},
        {'cluster_id': 'tail', 'track_id': 'track_0000', 'start_frame': 2, 'role': 'swap', 'faceset_id': 'alice', 'source_view_index': 0}]})
    runner = preview_module.MasterPipelineRunner(app.state.root, app.state.models, app.state.media, lambda: app.state.settings)
    _, _, _, sources, _, _ = runner._prepare_inputs(SimpleNamespace(request=request), app.state.settings)
    plan = FrameEditPlan(runner, request, sources)
    updates = []
    session = SimpleNamespace(selected_ids={0}, swapper=SimpleNamespace(set_track_source=lambda *args: updates.append(args)), portrait=SimpleNamespace(tracks={0: 'old conditioning'}))
    plan.apply(session, 1); assert session.selected_ids == frozenset() and not updates
    plan.apply(session, 2); assert session.selected_ids == {0} and len(updates) == 1
    assert len(updates[0][1].views) == 1 and not session.portrait.tracks
    plan.apply(session, 3); assert len(updates) == 1


def test_preview_curves_reach_inference_and_saved_jobs_retain_edits(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    strengths = []
    class AnimatedSession(FixtureSession):
        def infer_frame(self, image, **kwargs):
            strengths.append(kwargs['strength'])
            return super().infer_frame(image, **kwargs)
    monkeypatch.setattr(preview_module, 'ReferencePipeline', AnimatedSession)
    payload['request']['parameter_curves'] = {'tracks': {'pose_correction_strength': [
        {'frame': 0, 'value': 0}, {'frame': 2, 'value': 1}], 'swap_weight': [{'frame': 0, 'value': 0}]}}
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        result = client.post('/api/v1/workstation/preview', json=payload)
        assert result.status_code == 200, result.text
        assert strengths == [0, .5, 1]
        assert result.json()['parameters']['swap_weight'] == 0
        saved = client.post('/api/v1/projects', json={'name': 'Animated edit', 'request': payload['request']})
        assert saved.status_code == 201, saved.text
        reopened = client.get('/api/v1/projects/' + saved.json()['project_id']).json()
        assert reopened['request']['parameter_curves'] == saved.json()['request']['parameter_curves']


def test_manual_mask_validation_and_projection(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    runner = preview_module.MasterPipelineRunner(app.state.root, app.state.models, app.state.media, lambda: app.state.settings)
    invalid = JobRequest.model_validate({**payload['request'], 'mask_base64': 'not a png'})
    with pytest.raises(DomainError, match='bounded local PNG'): FrameEditPlan(runner, invalid, {})
    mask = cv2.imencode('.png', np.full((64, 96), 255, np.uint8))[1].tobytes()
    request = JobRequest.model_validate({**payload['request'], 'mask_base64': 'data:image/png;base64,' + base64.b64encode(mask).decode()})
    plan = FrameEditPlan(runner, request, {})
    image, result = arrays(); projected = plan.occluders(image, result)
    assert projected[0].shape == (32, 32) and projected[0].min() == 1
