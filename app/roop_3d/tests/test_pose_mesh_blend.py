"""Geometric endpoints, depth visibility, curve plumbing and real local output."""
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.core.animation import ParameterCurves
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.session import ReferencePipeline, FrameResult, StageLatency, Replacement
from roop_3d.reconstruction.pose_blend import PoseBlendProjector, reblend, rasterize
from roop_3d.tests.test_stage06_pipeline import real_faces
from roop_3d.tests.test_workstation_preview import preview_fixture, FixtureSession, arrays
from roop_3d.server import workstation_preview as preview_module
from roop_3d.compositor.pipeline import FrameCompositor
from roop_3d.server.app import create_app
from roop_3d.core.settings import Settings
from roop_3d.tests.test_stage15_e2e_golden import create_golden_video, _wait_state
from roop_3d.core.storage import owned
from roop_3d.media.video_reader import VideoReader


def checker(size=128):
    y, x = np.indices((size, size))
    return np.stack((x * 255 / (size - 1), y * 255 / (size - 1), ((x // 8 + y // 8) % 2) * 255), -1).astype(np.uint8)


def test_tps_and_perspective_endpoints_and_geometric_midpoint():
    projector = PoseBlendProjector()
    projection = projector.render(checker(), projector.reference * 128, .5)
    zero, one = reblend(projection, 0), reblend(projection, 1)
    assert np.array_equal(zero.image, projection.tps_image)
    assert np.array_equal(one.image, projection.mesh_image)
    assert np.array_equal(zero.coverage, projection.tps_coverage)
    assert np.array_equal(one.coverage, projection.mesh_coverage)
    assert not np.array_equal(zero.image, one.image)
    geometry = projection.geometry
    vertices = (geometry.flat_camera_vertices + geometry.mesh_camera_vertices) * .5
    homogeneous = vertices @ geometry.camera_matrix.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    expected, expected_alpha, _ = rasterize(geometry.texture, geometry.texture_coverage, pixels, vertices,
        geometry.uv, geometry.faces, (128, 128), perspective=True)
    assert np.array_equal(projection.image, expected)
    assert np.array_equal(projection.coverage, expected_alpha)
    assert not np.array_equal(projection.image, ((zero.image.astype(float) + one.image) * .5).astype(np.uint8))
    assert np.all(projection.image[projection.coverage == 0] == 0)


@pytest.mark.parametrize('angle', [-.65, 0, .65])
def test_mesh_depth_and_known_camera_pose_reach_the_rasterizer(angle):
    projector = PoseBlendProjector()
    camera = np.array(((192, 0, 64), (0, 192, 64), (0, 0, 1)), float)
    observed, _ = cv2.projectPoints(projector.anchors, np.array((.1, angle, .05)), np.array((0., 0., .45)), camera, None)
    projection = projector.render(checker(), observed.reshape(5, 2), 1)
    assert projection.reprojection_error < .001
    assert projection.mesh_coverage.max() == 1
    assert len(projection.visible_faces) > 0
    flat = PoseBlendProjector(); flat.vertices[:, 2] = 0
    flat_projection = flat.render(checker(), observed.reshape(5, 2), 1)
    assert not np.array_equal(projection.mesh_image, flat_projection.mesh_image)


def test_z_buffer_backfaces_and_perspective_correct_texture_sampling():
    texture = checker(32)
    points = np.array(((2, 2), (2, 26), (26, 2)), float)
    # Clockwise image winding: front-facing in CV camera coordinates.
    camera = np.column_stack((points, np.ones(3)))
    uv = np.array(((0, 0), (0, 31), (31, 0)), float)
    faces = np.array(((0, 1, 2), (3, 4, 5)))
    vertices = np.vstack((camera * 2, camera))
    image, coverage, visible = rasterize(texture, np.ones((32, 32), np.float32), np.vstack((points, points)), vertices,
        np.vstack((uv, uv)), faces, (32, 32), perspective=True)
    assert visible.tolist() == [[3, 4, 5]]  # Near triangle owns all overlapping pixels.
    _, back, visible = rasterize(texture, np.ones((32, 32), np.float32), points, camera, uv,
        np.array(((2, 1, 0),)), (32, 32), perspective=True)
    assert not back.any() and len(visible) == 0
    varied = camera * np.array((1, 2, 3))[:, None]
    perspective, _, _ = rasterize(texture, np.ones((32, 32), np.float32), points, varied, uv, faces[:1], (32, 32), perspective=True)
    affine, _, _ = rasterize(texture, np.ones((32, 32), np.float32), points, varied, uv, faces[:1], (32, 32), perspective=False)
    assert not np.array_equal(perspective[8, 8], affine[8, 8])


@pytest.mark.parametrize('points,weight', [(np.zeros((5, 2)), .5), (np.ones((106, 2)), .5), (np.full((5, 2), np.nan), .5), (np.ones((5, 2)), 2)])
def test_invalid_geometry_fails_explicitly_instead_of_retarget_fallback(points, weight):
    with pytest.raises(DomainError): PoseBlendProjector().render(checker(), points, weight)


def test_legacy_serialization_and_new_curve_dispatch_stay_separate():
    old = WorkstationParameters(pose_strength=.8)
    assert old.inference_options() == {'strength': .8}
    assert 'pose_mode' not in old.model_dump()
    new = WorkstationParameters(pose_mode='mesh_blend')
    curve = ParameterCurves.model_validate({'tracks': {'pose_correction_strength': [{'frame': 0, 'value': 0}, {'frame': 2, 'value': 1}]}})
    assert [curve.evaluate(frame, new).inference_options() for frame in range(3)] == [{'pose_blend': 0}, {'pose_blend': .5}, {'pose_blend': 1}]


def test_projected_coverage_preserves_unseen_target_pixels_even_after_dilation():
    image, result = arrays()
    coverage = np.zeros((32, 32), np.float32); coverage[8:24, 8:24] = 1
    item = replace(result.replacements[0], coverage=coverage)
    compositor = FrameCompositor(parser=SimpleNamespace(predict=lambda crop: None))
    output = compositor.compose(image, (item,), {0: result.observations[0].observation.landmarks106},
        options=WorkstationParameters(erosion_px=-10).composite_options())
    frame_mask = cv2.warpAffine(coverage, item.inverse, (96, 64))
    assert np.array_equal(output[frame_mask == 0], image[frame_mask == 0])


def test_preview_mesh_curve_and_nine_cell_pose_sweep_use_projection_buffers(tmp_path, monkeypatch):
    app, payload = preview_fixture(tmp_path, monkeypatch)
    calls = []
    projector = PoseBlendProjector()
    class MeshSession(FixtureSession):
        def infer_frame(self, image, **options):
            calls.append(options)
            projection = projector.render(checker(), projector.reference * 128, options['pose_blend'])
            # Fit the actual result to the fixture's full frame via an exact affine.
            item = Replacement(0, projection.image, np.array(((.4, 0, 20), (0, .4, 4))), 0,
                'identity_swap_tps_mesh_blend', projection.coverage, projection)
            observed = arrays()[1].observations
            return FrameResult(observed, (item,), StageLatency(1, 2, 0, 3), False)
    monkeypatch.setattr(preview_module, 'ReferencePipeline', MeshSession)
    payload['parameters'] = {'pose_mode': 'mesh_blend'}
    payload['request']['parameter_curves'] = {'tracks': {'pose_correction_strength': [{'frame': 0, 'value': 0}, {'frame': 2, 'value': 1}]}}
    payload.update(x={'parameter': 'pose_strength', 'values': [0, .5, 1]}, y={'parameter': 'feather_px', 'values': [0, 5, 10]})
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        result = client.post('/api/v1/workstation/preview', json=payload)
    assert result.status_code == 200, result.text
    assert calls == [{'pose_blend': 0}, {'pose_blend': .5}, {'pose_blend': 1}]
    assert len(result.json()['cells']) == 9
    assert len({cell['image'] for cell in result.json()['cells']}) > 3
    assert [cell['parameters']['pose_strength'] for cell in result.json()['cells'][:3]] == [0, .5, 1]


def test_real_identity_inference_produces_mesh_blend_outputs(real_faces, tmp_path):
    source, _, _, target_image = real_faces
    session = ReferencePipeline(config=RuntimeConfig(provider='CPUExecutionProvider', hardware_profile='cpu'))
    outputs = []
    try:
        session.prepare_session(source, retarget='off')
        for index, weight in enumerate((0., .5, 1.)):
            result = session.infer_frame(target_image, pose_blend=weight)
            assert result.replacements
            assert all(item.method == 'identity_swap_tps_mesh_blend' and item.projection is not None for item in result.replacements)
            output = session.composite_frame(target_image, result)
            assert output.shape == target_image.shape and output.dtype == np.uint8
            outputs.append(output)
            assert cv2.imwrite(str(tmp_path / f'pose-{index}.png'), output)
        assert not np.array_equal(outputs[0], outputs[2])
        assert not np.array_equal(outputs[1], outputs[0])
        assert session.capability()['pose_projection']['active']
    finally:
        session.release_session()
    assert session.pose_projector is None


def test_real_api_mesh_preview_and_animated_render_job(real_faces, tmp_path):
    """Real upload/build, inspection, neural swap, projection, queue and video encode."""
    source, _, _, target_image = real_faces
    settings = Settings(hardware_profile='cpu', preferred_provider='CPUExecutionProvider', audio_policy='copy')
    app = create_app(settings, repo_root=tmp_path, worker_enabled=True)
    path = owned(tmp_path, 'roop_3d/.runtime/media/pose-target.mp4')
    create_golden_video(path, target_image, count=3, fps=25)
    import base64
    view = min(source.views, key=lambda item: abs(item.yaw))
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        mime = 'image/png' if view.image_bytes.startswith(b'\x89PNG') else 'image/jpeg'
        built = client.post('/api/v1/facesets/build', json={'name': 'Pose acceptance fixture', 'views': [{
            'image_base64': f'data:{mime};base64,' + base64.b64encode(view.image_bytes).decode(),
            'embedding': list(view.embedding), 'landmarks': [list(point) for point in view.landmarks],
            'bbox': list(view.bbox), 'yaw': view.yaw, 'pitch': view.pitch, 'roll': view.roll}]})
        assert built.status_code == 201, built.text
        inspection = client.post('/api/v1/media/inspect-faces', json={'path': path.relative_to(tmp_path).as_posix()})
        assert inspection.status_code == 200, inspection.text
        assert inspection.json()['tracks']
        request = {'target_path': path.relative_to(tmp_path).as_posix(), 'inspection_id': inspection.json()['inspection_id'],
            'face_mappings': [{'track_id': inspection.json()['tracks'][0]['track_id'], 'faceset_id': built.json()['faceset_id']}],
            'workstation_parameters': {'pose_mode': 'mesh_blend'}, 'parameter_curves': {'tracks': {'pose_correction_strength': [
                {'frame': 0, 'value': 0}, {'frame': 2, 'value': 1}]}}}
        preview = client.post('/api/v1/workstation/preview', json={'request': request, 'timestamp': .04})
        assert preview.status_code == 200, preview.text
        result = preview.json()
        assert result['pose_projection'] and result['pose_projection'][0]['weight'] == .5
        assert {'pose', 'swapper', 'final'} <= result['buffers'].keys()
        created = client.post('/api/v1/jobs', json=request, headers={'Idempotency-Key': 'real-pose-blend-curve'})
        assert created.status_code == 201, created.text
        completed = _wait_state(client, created.json()['job_id'], 'succeeded', timeout=180)
        assert completed['output']['frames'] == 3
        video = owned(tmp_path, completed['output']['output_path'])
        frames = VideoReader(video, hardware='software').frames()
        try:
            images = [frame.image for frame in frames]
        finally:
            frames.close()
        assert len(images) == 3 and not np.array_equal(images[0], images[2])
