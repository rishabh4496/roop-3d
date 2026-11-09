from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import Manifest, ModelRegistry
from roop_3d.tests.test_stage03_registry import artifact


@pytest.mark.parametrize('model_id,name,role', [
    ('hyperswap_1a_256', 'hyperswap_1a_256.onnx', 'identity_swapper'),
    ('gpen_bfr_512', 'gpen_bfr_512.onnx', 'face_restorer'),
    ('scrfd', 'scrfd_10g_bnkps.onnx', 'detector'),
])
def test_catalog_reports_provider_weights_as_installed(tmp_path, model_id, name, role):
    data = b'locally verified fixture'
    record = artifact(data, model_id=model_id, name=name, filenames=[name], role=role)
    registry = ModelRegistry(manifest=Manifest(evidence_scope='Fixture', models=(record,)))
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry,
                           search_paths=lambda: ('roop_3d/.runtime/weights',))
    path = tmp_path / 'roop_3d/.runtime/weights' / name
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    row = next(r for r in catalog.public() if r['id'] == model_id)
    assert row['installed'], 'Weights resolved by the runtime must be selectable in the rack'
    assert row['state'] == 'VERIFIED_LOCAL'
    assert catalog.resolve_local_model(model_id) == path


def test_catalog_does_not_verify_corrupted_provider_weights(tmp_path):
    record = artifact(b'expected', model_id='gpen_bfr_512', name='gpen_bfr_512.onnx',
                      filenames=['gpen_bfr_512.onnx'], role='face_restorer')
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path,
        registry=ModelRegistry(manifest=Manifest(evidence_scope='Fixture', models=(record,))),
        search_paths=lambda: ('roop_3d/.runtime/weights',))
    path = tmp_path / 'roop_3d/.runtime/weights/gpen_bfr_512.onnx'
    path.parent.mkdir(parents=True)
    path.write_bytes(b'corrupt!')
    row = next(r for r in catalog.public() if r['id'] == 'gpen_bfr_512')
    assert not row['installed']
    assert row['state'] != 'VERIFIED_LOCAL'


def test_detector_selection_changes_the_loaded_graph():
    from roop_3d.media.detector import Detector
    selected = ['scrfd']
    loaded = []
    class Catalog:
        def detector(self, model_id='scrfd'):
            return {'model_id': model_id, 'sha256': model_id}
    class Session:
        def __init__(self, model_id): self.model_id = model_id
        def get_inputs(self): return [SimpleNamespace(name='input', shape=(1, 3, 640, 640), type='tensor(float)')]
        def run(self, _names, feeds):
            if self.model_id == 'yoloface':
                row = np.array([320, 320, 100, 120, .9] + [320, 320, 1] * 5, np.float32)
                return [row.reshape(1, 20, 1)]
            return ([np.zeros((640 // s) ** 2 * 2, np.float32) for s in (8, 16, 32)] +
                    [np.zeros(((640 // s) ** 2 * 2, 4), np.float32) for s in (8, 16, 32)] +
                    [np.zeros(((640 // s) ** 2 * 2, 10), np.float32) for s in (8, 16, 32)])
    def factory(record):
        loaded.append(record['model_id'])
        return Session(record['model_id'])
    detector = Detector(Catalog(), factory, model_id=lambda: selected[0])
    for model_id in ('scrfd', 'retinaface', 'yoloface', 'scrfd'):
        selected[0] = model_id
        faces, record = detector.detect(np.zeros((640, 640, 3), np.uint8), .5)
        assert record['model_id'] == model_id
        assert len(faces) == (1 if model_id == 'yoloface' else 0)
    assert loaded == ['scrfd', 'retinaface', 'yoloface', 'scrfd']


def test_selecting_same_person_in_later_frame_reuses_track(tmp_path, monkeypatch):
    from roop_3d.media.service import MediaService
    service = MediaService(tmp_path, ModelCatalog())
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'video fixture')
    box = np.array([20, 20, 80, 90], np.float32)
    points = np.array([[35, 40], [65, 40], [50, 55], [38, 70], [62, 70]], np.float32)
    record = {'model_id': 'fixture', 'sha256': 'a' * 64}
    service.detector = SimpleNamespace(detect=lambda *a, **k: ([(box, points)], record),
                                      catalog=SimpleNamespace(recognizer=lambda: record))
    service.analyzer.recognizer = SimpleNamespace(embed=lambda *a: np.array([1, 0, 0], np.float32))
    monkeypatch.setattr(service, 'path', lambda _: path)
    monkeypatch.setattr(service, 'probe', lambda _: SimpleNamespace(fps=30, width=100, height=100))
    monkeypatch.setattr(service, 'frame', lambda *a: np.zeros((100, 100, 3), np.uint8))
    first = service.detect_frame_faces('clip', frame_index=0)
    second = service.detect_frame_faces('clip', frame_index=30)
    assert second['faces'][0]['track_id'] == first['faces'][0]['track_id']
    assert len(second['inspection']['tracks']) == 1
    assert second['inspection']['tracks'][0]['last_frame'] == 30


def test_live_preview_keeps_swapper_selection_and_invalidates_cache(tmp_path, monkeypatch):
    from roop_3d.tests.test_workstation_preview import preview_fixture
    from roop_3d.tests.test_remediation_stage4 import live_payload
    from roop_3d.server.live_preview import LivePreviewPayload
    app, fixture = preview_fixture(tmp_path, monkeypatch)
    first = LivePreviewPayload.model_validate(live_payload(fixture, swapper_model_id='inswapper_128'))
    second = LivePreviewPayload.model_validate(live_payload(fixture, swapper_model_id='hyperswap_1c_256'))
    service = app.state.live_previews
    assert service._build(second).request.swapper_model_id == 'hyperswap_1c_256'
    assert service.config_hash(first) != service.config_hash(second)


@pytest.mark.parametrize('precision,fp16', [('fp16', True), ('fp32', False)])
def test_local_swapper_passes_selected_precision_to_tensorrt(tmp_path, monkeypatch, precision, fp16):
    import onnx
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.engine.volumetric.swapper_engine import load_local_session
    from roop_3d.core import ort_session
    path = tmp_path / 'fixture.onnx'
    tensor = onnx.helper.make_tensor_value_info('image', onnx.TensorProto.FLOAT, [1, 3, 8, 8])
    graph = onnx.helper.make_graph([onnx.helper.make_node('Identity', ['image'], ['output'])], 'fixture',
        [tensor], [onnx.helper.make_tensor_value_info('output', onnx.TensorProto.FLOAT, [1, 3, 8, 8])])
    onnx.save(onnx.helper.make_model(graph), path)
    seen = {}
    class Manager:
        def __init__(self, **kwargs): seen.update(kwargs)
        def create(self, *args, **kwargs):
            return SimpleNamespace(provider='TensorrtExecutionProvider', session='fixture-session')
    monkeypatch.setattr(ort_session, 'SessionManager', Manager)
    config = RuntimeConfig(provider='TensorrtExecutionProvider', hardware_profile='6gb', precision_policy=precision)
    config.__dict__['gpu_budget_bytes'] = 1024 * 1024 * 1024
    catalog = SimpleNamespace(set_vram_loaded=lambda *args: None)
    assert load_local_session(catalog, 'fixture', path, config) == 'fixture-session'
    assert seen['fp16'] is fp16


def test_preview_runtime_uses_tensorrt_instead_of_cuda_admission():
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner
    from roop_3d.core.settings import Settings
    config = MasterPipelineRunner._runtime_config(SimpleNamespace(profile='6gb', provider='CUDAExecutionProvider'),
        Settings(preferred_provider='TensorrtExecutionProvider', precision_policy='fp16', detector_model_id='yoloface'))
    assert (config.provider, config.precision_policy, config.detector_model_id) == ('TensorrtExecutionProvider', 'fp16', 'yoloface')


def test_compile_route_runs_adapter_with_selected_precision(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from roop_3d.server.app import create_app
    from roop_3d.core.settings import Settings
    from roop_3d.core import hardware
    from roop_3d.server import model_runtime
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    app.state.worker = SimpleNamespace()
    monkeypatch.setattr(hardware, 'probe_hardware', lambda _: SimpleNamespace(gpus=[object()],
        advertised_providers=['CUDAExecutionProvider', 'TensorrtExecutionProvider'], resolved_hardware_profile='6gb'))
    calls = []
    def compile_one(catalog, model_id, config):
        calls.append((model_id, config.precision_policy, config.hardware_profile))
        return {'model_id': model_id, 'status': 'compiled', 'canary_verified': True}
    monkeypatch.setattr(model_runtime, 'compile_model', compile_one)
    with TestClient(app, base_url='http://127.0.0.1:8030') as client:
        response = client.post('/api/v1/models/compile_trt', json={'model_id': 'hyperswap_1a_256', 'precision': 'fp32'})
    assert response.status_code == 200, response.text
    assert response.json()['results'][0]['canary_verified']
    assert calls == [('hyperswap_1a_256', 'fp32', '6gb')]
    assert app.state.engine_gate.owner_count == 0


def test_locked_angle_samples_reach_existing_durable_person_bank(tmp_path, monkeypatch):
    from roop_3d.tests.test_remediation_stage5 import extraction_app, detection, FakeDetector, angle_vector
    from roop_3d.core.pose import Pose3D
    from roop_3d.identity.target_lock import FaceRef, BankEntry
    from roop_3d.server.face_bank import FaceBank, DiscoverBank
    app, service, request = extraction_app(tmp_path, monkeypatch)
    box, points = detection(28, Pose3D(yaw=0, pitch=0, roll=0))
    service.detector = FakeDetector([(box, points)])
    service.recognizer = SimpleNamespace(embed=lambda *args: angle_vector(0))
    image = np.zeros((64, 96, 3), np.uint8)
    monkeypatch.setattr(app.state.media, 'frame', lambda *args: image)
    monkeypatch.setattr(app.state.media.frames, 'read', lambda *args, **kwargs: SimpleNamespace(image=image))
    bank = FaceBank(service)
    value = DiscoverBank(target_media_path=request['target_path'], inspection_id=request['inspection_id'])
    first = bank.discover(value)
    locks = app.state.media.target_locks
    lock = locks.create(request['target_path'], 0, FaceRef(angle_vector(0), 0, 0, 0, box.tolist()))
    cluster = locks.registry(request['target_path']).clusters[lock.cluster_id]
    cluster.track_id = 'track_0000'
    cluster.consider(BankEntry(angle_vector(0), 60, 20, 0, 5, box.tolist(), 'auto'))
    second = bank.discover(value)
    assert len(second['clusters']) == len(first['clusters']) == 1
    assert second['clusters'][0]['id'] == first['clusters'][0]['id']
    assert any(face['frame'] == 5 and face['yaw'] == 60 and face['pitch'] == -20 for face in second['clusters'][0]['faces'])
