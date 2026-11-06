"""Stage 7 real-model smoke test. Run with an optional checkout-local video/image directory.

Uses SCRFD, ArcFace, 106/3D68 landmarks and neural occlusion; writes an ignored .fsz
under app/facesets, validates it, and reloads it through the Stage 3 HTTP API.
"""
from pathlib import Path
import json
import sys
import time
import zipfile
import onnxruntime as ort
ort.set_default_logger_severity(3)
ROOT = Path(__file__).resolve().parent
APP = ROOT / 'app'
sys.path.insert(0, str(APP))
from fastapi import FastAPI
from fastapi.testclient import TestClient
from roop_3d.core.settings import Settings
from roop_3d.models.catalog import ModelCatalog
from roop_3d.server.face_extract import FaceExtractService
from roop_3d.server.faceset_builder import router
from roop_3d.server.faceset_3d import router as reconstruction_router
from roop_3d.faceset_builder import FacesetBuilder
from roop_3d.faceset.faceset_reader import FaceSetReader


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('media', nargs='?')
    parser.add_argument('--profile', choices=['auto', 'cpu', '6gb', '12gb'], default='auto')
    args = parser.parse_args()
    path = Path(args.media).resolve() if args.media else APP / 'roop_3d/.runtime/media/03de3642d8d64b22bee7ffe543fa2966.mp4'
    assert path.is_relative_to(APP) and path.exists(), 'Supply media inside app/'
    app = FastAPI()
    app.include_router(router)
    app.include_router(reconstruction_router)
    state = app.state
    state.root, state.models, state.settings = APP, ModelCatalog(APP), Settings(hardware_profile=args.profile)
    state.face_extract = FaceExtractService(state)
    state.faceset_builder = FacesetBuilder(state)
    with TestClient(app) as client:
        response = client.post('/api/v1/faceset/extract-from-media', json={'media_path': str(path.relative_to(APP)), 'max_frames': 12, 'frame_stride': 30})
        assert response.status_code == 200, response.text
        key = response.json()['job_id']
        deadline = time.monotonic() + 300
        while True:
            draft = client.get('/api/v1/faceset/extraction/' + key).json()
            if draft['status'] != 'extracting':
                break
            assert time.monotonic() < deadline, 'Extraction timed out'
            time.sleep(.5)
        assert draft['status'] == 'ready', draft
        assert draft['clusters'], draft
        print('Inference: ' + json.dumps(draft.get('inference', {})), flush=True)
        if args.profile in ('6gb', '12gb'):
            assert draft['inference']['landmarks_3d']['provider'] == 'CUDAExecutionProvider', 'Explicit GPU verification must execute 3D landmarks on CUDA'
        cluster = max(draft['clusters'], key=lambda c: c['coverage'])
        assert cluster['selected_ids'], 'No unobstructed face; supply a clearer reference'
        assert [c['rank'] for c in draft['clusters']] == list(range(len(draft['clusters'])))
        selected = cluster['selected_ids']
        print(f"Real media: {draft['scanned']} frames, {len(draft['clusters'])} identities, {cluster['coverage']}/96 bins", flush=True)
        response = client.post('/api/v1/faceset/save-fsz', json={'job_id': key, 'cluster_rank': cluster['rank'], 'selected_ids': selected, 'name': 'Stage 7 verification'})
        assert response.status_code == 200, response.text
        saved = response.json()
        archive_path = APP / saved['path']
        with zipfile.ZipFile(archive_path) as archive:
            assert {'metadata.json', 'sphere_coverage.json', 'curation.json', 'uv_albedo.png'} <= set(archive.namelist())
            extras = json.loads(archive.read('curation.json'))
            assert all(len(v['landmarks_3d_68']) == 68 and len(v['landmarks_106']) == 106 for v in extras.values())
            assert archive.read('uv_albedo.png').startswith(b'\x89PNG')
        document = FaceSetReader(state.models).read(archive_path)
        assert len(document.views) == len(selected) <= 96
        assert all(len(v.embedding) == 512 and len(v.landmarks_68) == 68 for v in document.views)
        response = client.post('/api/v1/faceset/reconstruct-3d', json={'faceset_id': saved['faceset_id']})
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload['sphere_coverage_count'] == cluster['coverage']
        assert payload['mesh_full_3d']['vertices'] and payload['uv_texture_data_url']
        assert saved['faceset_id'] in [v['faceset_id'] for v in client.get('/api/v1/faceset/list').json()]
        # Exercise archive editing without rerunning inference: remove one angle and preserve tensors.
        if len(selected) > 1:
            edit = client.post('/api/v1/faceset/extract-from-media', json={'faceset_id': saved['faceset_id']}).json()
            while edit['status'] == 'extracting':
                assert time.monotonic() < deadline, 'Edit draft timed out'
                time.sleep(.1)
                edit = client.get('/api/v1/faceset/extraction/' + edit['job_id']).json()
            assert edit['status'] == 'ready', edit
            choices = edit['clusters'][0]['selected_ids'][:-1]
            edited = client.post('/api/v1/faceset/save-fsz', json={'job_id': edit['job_id'], 'cluster_rank': 0,
                'selected_ids': choices, 'name': 'Stage 7 verification (edited)', 'faceset_id': saved['faceset_id']})
            assert edited.status_code == 200, edited.text
            current = client.get('/api/v1/faceset/3d-state', params={'faceset_id': saved['faceset_id']}).json()
            assert current['sphere_coverage_count'] == len(choices)
            assert len(FaceSetReader(state.models).read(archive_path).views) == len(choices)
            with zipfile.ZipFile(archive_path) as archive:
                assert all(len(v['landmarks_3d_68']) == 68 for v in json.loads(archive.read('curation.json')).values())
            print(f"Edit round-trip: {len(choices)} angles, reconstruction refreshed", flush=True)
        print('PASS: ' + str(archive_path), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
