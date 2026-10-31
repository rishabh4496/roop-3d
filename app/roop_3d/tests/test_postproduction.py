"""Real codecs and generated pixels, no inference/model or sibling-app imports."""
from fractions import Fraction
import hashlib
import json
import time
from types import SimpleNamespace

import av
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.storage import atomic_json
from roop_3d.jobs.schema import JobRequest, JobState
from roop_3d.engine.pipeline_runner import PipelinePaused
from roop_3d.media.finishing import parse_cube, Finisher, FinishingOptions, ExportOptions
from roop_3d.media.provenance import verify_manifest
from roop_3d.media.vfx import PassCollector, geometry_passes, publish_passes, discard_sequences
from roop_3d.media.video_writer import encode_video
from roop_3d.tests.test_stage05_media import make_video
from roop_3d.tests.test_stage12_worker import _fixture, _Session, _wait_state


def cube(invert=False):
    return 'TITLE "test"\nLUT_3D_SIZE 2\n' + '\n'.join(f'{1-r if invert else r} {1-g if invert else g} {1-b if invert else b}'
        for b in (0, 1) for g in (0, 1) for r in (0, 1))


def test_cube_red_fastest_domain_and_interpolation():
    rgb = np.random.default_rng(42).random((9, 7, 3)).astype(np.float32)
    np.testing.assert_allclose(parse_cube(cube()).apply(rgb), rgb, atol=1e-6)
    np.testing.assert_allclose(parse_cube(cube(True)).apply(rgb), 1-rgb, atol=1e-6)
    domain = cube().replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 2\nDOMAIN_MIN -1 -1 -1\nDOMAIN_MAX 1 1 1')
    np.testing.assert_allclose(parse_cube(domain).apply(np.zeros((1, 1, 3))), .5)


@pytest.mark.parametrize('text', ['LUT_1D_SIZE 2\n0 0 0', 'LUT_3D_SIZE 34', cube() + '\n1 0 0',
    cube().replace('0 0 0', 'nan 0 0'), cube().replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 2\nDOMAIN_MAX 0 1 1'),
    cube().replace('LUT_3D_SIZE 2', 'LUT_3D_SIZE 2\nLUT_3D_SIZE 2')])
def test_cube_rejects_malformed(text):
    with pytest.raises(ValueError):
        parse_cube(text)


def test_grain_is_repeatable_temporal_and_luminance_controlled():
    image = np.full((96, 128, 3), 110, np.uint8)
    finisher = Finisher(FinishingOptions(grain={'intensity': .05, 'size': 2, 'shadows': 1, 'highlights': .2}))
    assert np.array_equal(finisher.apply(image, 7), finisher.apply(image, 7))
    assert not np.array_equal(finisher.apply(image, 7), finisher.apply(image, 8))
    assert np.std(finisher.apply(image, 7).astype(float)-image) > 1
    assert Finisher(FinishingOptions()).apply(image, 1) is image


def test_export_contract_and_old_identity():
    payload = {'target_path': 'roop_3d/.runtime/media/a.mp4', 'inspection_id': 'a'*32,
               'face_mappings': [{'track_id': 'track_0000', 'faceset_id': 'alice'}]}
    assert 'export' not in JobRequest(**payload).identity_payload()
    for value in [{'baked_video': False}, {'passes': ['alpha', 'alpha']}, {'passes': ['invented']},
                  {'baked_video': False, 'passes': ['alpha'], 'finishing': {'cube': cube()}}]:
        with pytest.raises(ValueError):
            ExportOptions.model_validate(value)


def test_finishing_codec_manifest_hmac_and_tampering(tmp_path, monkeypatch):
    monkeypatch.setenv('ROOP3D_PROVENANCE_KEY', 'test-key-'*8)
    source = make_video(tmp_path/'roop_3d/.runtime/media/source.mov', audio=1)
    options = FinishingOptions(cube=cube(True), provenance='hmac-sha256')
    manifest = encode_video(source, 'finished.mp4', root=tmp_path, output_dir='roop_3d/.runtime/outputs',
                            hardware='software', finishing=options)
    output = tmp_path/manifest['output_path']
    assert verify_manifest(output, manifest)['manifest_authenticated'] is True
    assert manifest['finishing']['lut_sha256'] == hashlib.sha256(cube(True).encode()).hexdigest()
    with av.open(str(output)) as container:
        assert 'recipe_sha256' in container.metadata['comment']
        assert len(container.streams.audio) == 1
    modified = {**manifest, 'encoder': 'tampered'}
    assert verify_manifest(output, modified)['valid'] is False
    with output.open('ab') as stream:
        stream.write(b'tampered')
    assert verify_manifest(output, manifest)['media_hash_matches'] is False


def test_pass_precision_alignment_roundtrip_and_retention(tmp_path):
    source = make_video(tmp_path/'roop_3d/.runtime/media/source.mov')
    selected, job_id = ('alpha', 'occluder', 'uv'), 'b'*32
    alpha = np.tile(np.linspace(0, 1, 48, dtype=np.float32), (32, 1))
    inverse = np.array([[1, 0, 7], [0, 1, 9]], np.float32)
    for index in range(8):
        collector = PassCollector((64, 96), selected)
        collector.observe('vfx', np.full((32, 48, 3), 180, np.uint8), inverse, alpha)
        collector.observe('occluder', alpha, inverse, alpha)
        collector.save(tmp_path, job_id, index)
    result = publish_passes(tmp_path, job_id, source, selected, 'roop_3d/.runtime/outputs',
                            start=Fraction(2, 25), end=Fraction(6, 25), check=lambda: None)
    assert len(result['metadata']['frames']) == 4
    sequence = tmp_path/result['metadata']['sequence_root']
    assert sequence.is_dir()
    image = cv2.imread(str(sequence/'alpha/00000002.png'), cv2.IMREAD_UNCHANGED)
    assert image.dtype == np.uint16 and image[9, 7] == 0 and image[9, 54] == 65535
    assert len(np.unique(image)) > 20 and image[0, 0] == 0
    assert (tmp_path/f'roop_3d/.runtime/jobs/{job_id}/passes').exists()
    discard_sequences(tmp_path, result['manifest_path'])
    assert not sequence.exists()


def test_geometry_does_not_invent_missing_projection():
    with pytest.raises(DomainError, match='frame-aligned'):
        geometry_passes(SimpleNamespace(), (10, 10), .5)


def test_geometry_depth_and_normal_are_target_aligned():
    vertices = np.array([[0, 0, 10], [8, 0, 10], [0, 8, 10]], np.float32)
    projection = SimpleNamespace(geometry=SimpleNamespace(flat_camera_vertices=vertices, mesh_camera_vertices=vertices),
        blend_pixels=np.array([[1, 1], [9, 1], [1, 9]], np.float32), visible_faces=np.array([[0, 1, 2]]),
        coverage=np.ones((12, 12), np.float32))
    replacement = SimpleNamespace(projection=projection, image=np.zeros((12, 12, 3), np.uint8),
        inverse=np.array([[1, 0, 5], [0, 1, 3]], np.float32))
    normal, depth = geometry_passes(replacement, (24, 24), .5)
    assert depth[5, 7] == pytest.approx(10/2048)
    np.testing.assert_allclose(normal[5, 7], [1, .5, .5])  # BGR encodes +Z normal
    assert depth[0, 0] == 0 and not normal[0, 0].any()
    vertices[1, 2] = 14  # tilted triangle, then a 90-degree crop-to-frame roll
    replacement.inverse = np.array([[0, -1, 16], [1, 0, 3]], np.float32)
    rotated_normal, _ = geometry_passes(replacement, (24, 24), .5)
    assert rotated_normal[5, 14, 2] == pytest.approx(.5, abs=1e-6)
    assert rotated_normal[5, 14, 1] < .5


class _PassSession(_Session):
    def composite_frame(self, image, result, **kwargs):
        observer = kwargs.get('observer')
        if observer:
            alpha = np.full(image.shape[:2], .31415, np.float32)
            inverse = np.array([[1, 0, 0], [0, 1, 0]], np.float32)
            observer('vfx', 255-image, inverse, alpha)
            observer('occluder', np.full_like(alpha, .1), inverse, alpha)
        return 255-image


@pytest.mark.parametrize('baked', [True, False])
def test_queued_passes_survive_worker_cleanup_and_trim_timeline(tmp_path, baked):
    app, worker, payload = _fixture(tmp_path, session_factory=lambda catalog, config: _PassSession(config))
    payload.update({'export': {'baked_video': baked, 'passes': ['alpha', 'occluder', 'uv']},
                    'trim_start_seconds': .08, 'trim_end_seconds': .24})
    worker.start()
    try:
        with TestClient(app, base_url='http://127.0.0.1:8030') as client:
            created = client.post('/api/v1/jobs', json=payload, headers={'Idempotency-Key': 'passes-test'})
            assert created.status_code == 201, created.text
            job_id = created.json()['job_id']
            result = _wait_state(client, job_id, 'succeeded')
            output = result['output']
            assert output['baked_video'] is baked and output['frames'] == 4
            manifest = json.loads((tmp_path/output['pass_manifest_path']).read_text('utf-8'))
            assert (tmp_path/manifest['sequence_root']/'alpha/00000002.png').is_file()
            assert len(output['passes']) == 3
            plan = client.get(f'/api/v1/exports/{job_id}/timeline')
            assert plan.status_code == 200, plan.text
            assert plan.json()['mediaOffset'] == 2 and plan.json()['ranges'] == [[2, 6]]
            assert bool(plan.json()['swappedUrl']) is baked
            assert plan.json()['alphaUrl'].endswith('_alpha.mkv')
            deadline = time.monotonic() + 3
            checkpoint_dir = tmp_path/f'roop_3d/.runtime/jobs/{job_id}'
            while checkpoint_dir.exists() and time.monotonic() < deadline:
                time.sleep(.01)
            assert not checkpoint_dir.exists()
            assert (tmp_path/manifest['sequence_root']/'alpha/00000002.png').is_file()
    finally:
        worker.stop()


def test_pass_checkpoints_resume_after_pause_during_baked_encoding(tmp_path):
    app, worker, payload = _fixture(tmp_path, session_factory=lambda catalog, config: _PassSession(config))
    payload['export'] = {'baked_video': True, 'passes': ['alpha']}
    real_writer = worker.runner.writer
    attempts = []

    def pause_writer(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            job_id = kwargs['manifest_metadata']['job_id']
            app.state.jobs.transition(job_id, JobState.pausing)
            raise PipelinePaused()
        return real_writer(*args, **kwargs)

    worker.runner.writer = pause_writer
    worker.start()
    try:
        with TestClient(app, base_url='http://127.0.0.1:8030') as client:
            created = client.post('/api/v1/jobs', json=payload, headers={'Idempotency-Key': 'passes-pause'})
            assert created.status_code == 201, created.text
            job_id = created.json()['job_id']
            _wait_state(client, job_id, 'paused')
            assert (tmp_path/f'roop_3d/.runtime/jobs/{job_id}/passes/alpha/00000007.png').is_file()
            assert not (tmp_path/f'roop_3d/.runtime/outputs/roop3d_{job_id}_alpha.mkv').exists()
            assert not (tmp_path/f'roop_3d/.runtime/outputs/roop3d_{job_id}_sequences').exists()
            resumed = client.post(f'/api/v1/jobs/{job_id}/resume', json={})
            assert resumed.status_code == 200, resumed.text
            result = _wait_state(client, job_id, 'succeeded')
            assert result['output']['passes']['alpha'].endswith('_alpha.mkv')
            assert len(attempts) == 2
    finally:
        worker.stop()


def test_queued_export_timeline_and_finishing_endpoints(tmp_path):
    app, worker, payload = _fixture(tmp_path)
    payload['export'] = {'finishing': {'cube': cube(True), 'provenance': 'sha256'}}
    worker.start()
    try:
        with TestClient(app, base_url='http://127.0.0.1:8030') as client:
            created = client.post('/api/v1/jobs', json=payload, headers={'Idempotency-Key': 'postproduction-test'})
            assert created.status_code == 201, created.text
            job_id = created.json()['job_id']
            result = _wait_state(client, job_id, 'succeeded')
            assert result['output']['baked_video'] is True
            plan = client.get(f'/api/v1/exports/{job_id}/timeline')
            assert plan.status_code == 200, plan.text
            assert plan.json()['ranges'] == [[0, 8]] and plan.json()['frameRate'] == '25'
            assert plan.json()['sourceUrl'].startswith('file:')
            assert client.get(f'/api/v1/exports/{job_id}/verify').json()['valid'] is True
            assert client.get(f'/api/v1/exports/{job_id}/manifest').status_code == 200
            preview = client.post('/api/v1/finishing/preview', json={'path': payload['target_path'], 'finishing': {'cube': cube()}})
            assert preview.status_code == 200 and preview.json()['image'].startswith('data:image/jpeg')
            assert client.post('/api/v1/finishing/render', json={'path': '../roop-ultimate/a.mp4'}).status_code == 422
            assert client.post('/api/v1/finishing/render', json={'path': payload['target_path'], 'finishing': {'cube': 'invalid'}}).status_code == 422
    finally:
        worker.stop()
