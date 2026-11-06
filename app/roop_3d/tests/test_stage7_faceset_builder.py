"""Curation invariants independent of installed GPU weights."""
from types import SimpleNamespace
import zipfile
import json
import cv2
import numpy as np
import pytest
from roop_3d.core.errors import DomainError
from roop_3d.faceset.schemas import FaceView, QualityMetrics
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.faceset_builder import FacesetBuilder
from roop_3d.server.face_extract import FaceExtractService


def view(identity=0, yaw=0, score=.9):
    image = np.random.default_rng(1).integers(0, 255, (112,112,3), dtype=np.uint8)
    _, png = cv2.imencode('.png', image)
    embedding = [0.] * 512
    embedding[identity] = 1.
    return FaceView(index=0, image_bytes=png.tobytes(), embedding=tuple(embedding),
        landmarks=((30.,35.),(80.,35.),(55.,60.),(35.,80.),(75.,80.)), bbox=(0.,0.,112.,112.),
        yaw=yaw, pitch=0., roll=0., quality=QualityMetrics(score=score), pose_bin='front_mid', landmarks_trusted=False)


def draft():
    return dict(job_id='a'*32, status='ready', progress=1., scanned=3, error=None, clusters=[], samples={}, centroids=[], counts=[])


def test_rank_and_quality_selection():
    builder = FacesetBuilder(SimpleNamespace())
    job = draft()
    builder.add(job, view(0, score=.8), {'eligible': True})
    builder.add(job, view(1), {'eligible': True})
    builder.add(job, view(0, score=.99), {'eligible': False})
    builder.add(job, view(0, score=.9), {'eligible': True})
    assert [c['rank'] for c in job['clusters']] == [0,1]
    assert [c['detections'] for c in job['clusters']] == [3,1]
    assert len(job['samples']) == 3
    primary = job['clusters'][0]
    assert len(primary['selected_ids']) == 1
    assert job['samples'][primary['selected_ids'][0]][0].quality.score == .9


def test_save_edit_and_reconstruction(tmp_path):
    state = SimpleNamespace(root=tmp_path, models=None)
    state.face_extract = FaceExtractService(state)
    builder = FacesetBuilder(state)
    job = draft()
    builder.jobs[job['job_id']] = job
    for yaw in (-50,0,50):
        builder.add(job, view(yaw=yaw), {'eligible': True})
    selected = job['clusters'][0]['selected_ids']
    first = builder.save(job['job_id'], 0, selected, 'Test')
    path = tmp_path / first['path']
    assert first['reconstruction']['sphere_coverage_count'] == 3
    builder.save(job['job_id'], 0, selected[:1], 'Edited', first['faceset_id'])
    reloaded = FaceSetReader().read(path)
    assert len(reloaded.views) == 1 and reloaded.name == 'Edited'
    assert state.faceset_3d_state['values'][first['faceset_id']]['payload']['sphere_coverage_count'] == 1
    with zipfile.ZipFile(path) as archive:
        assert len(json.loads(archive.read('sphere_coverage.json'))) == 1
        assert archive.read('uv_albedo.png').startswith(b'\x89PNG')
    assert path.read_bytes() == (tmp_path / 'roop_3d/outputs/facesets' / path.name).read_bytes()
    with pytest.raises(DomainError):
        builder.save(job['job_id'], 1, selected[:1], 'Wrong person')


def test_media_paths_cannot_escape_checkout(tmp_path):
    builder = FacesetBuilder(SimpleNamespace(root=tmp_path))
    with pytest.raises(DomainError):
        list(builder._frames(['../private.jpg'], 1, 1))


def test_multipart_photos_and_image_directory(tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from roop_3d.server.faceset_builder import router
    app = FastAPI()
    app.include_router(router)
    app.state.root = tmp_path
    received = []
    def start(paths, **kwargs):
        received.extend(paths)
        return {'job_id': 'a'*32, 'status': 'extracting'}
    app.state.faceset_builder = SimpleNamespace(start=start)
    with TestClient(app) as client:
        result = client.post('/api/v1/faceset/extract-from-media', files=[
            ('files', ('a.png', view().image_bytes, 'image/png')),
            ('files', ('b.png', view().image_bytes, 'image/png'))])
        assert result.status_code == 200, result.text
    assert len(received) == 2
    builder = FacesetBuilder(SimpleNamespace(root=tmp_path))
    frames = list(builder._frames(['roop_3d/.runtime/media'], 10, 1))
    assert len(frames) == 2 and frames[0].shape == (112,112,3)


def test_all_96_bins_fit_in_a_valid_archive(tmp_path):
    state = SimpleNamespace(root=tmp_path, models=None)
    state.face_extract = FaceExtractService(state)
    builder = FacesetBuilder(state)
    job = draft()
    builder.jobs[job['job_id']] = job
    for pitch in range(8):
        for yaw in range(12):
            sample = view(yaw=-82.5+yaw*15).model_copy(update={'pitch': -52.5+pitch*15})
            builder.add(job, sample, {'eligible': True})
    assert job['clusters'][0]['coverage'] == 96
    result = builder.save(job['job_id'], 0, job['clusters'][0]['selected_ids'], '96-bin capacity')
    assert result['reconstruction']['sphere_coverage_count'] == 96
    assert len(FaceSetReader().read(tmp_path / result['path']).views) == 96
