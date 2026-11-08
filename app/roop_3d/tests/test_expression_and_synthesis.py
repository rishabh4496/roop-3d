"""Motion invariants, post-enhancement closure, wire parity and archive provenance."""
from dataclasses import replace
from types import SimpleNamespace
import json
import base64
import hashlib
import zipfile
import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roop_3d.core.face_editor import FaceEditor
from roop_3d.core.workstation import WorkstationParameters
from roop_3d.expression_restorer import ExpressionRestorer, eye_aspect_ratio
from roop_3d.engine.volumetric.reference_adapter import Motion
from roop_3d.faceset_synthesizer import SYNTHESIS_POSES, start_synthesis, start_photo_synthesis, angle_uv
from roop_3d.faceset.faceset_reader import FaceSetReader, export_v2
from roop_3d.faceset.pose_selector import PoseSelector
from roop_3d.tests.test_stage7_faceset_builder import view, draft
from roop_3d.server.face_extract import FaceExtractService
from roop_3d.faceset_builder import FacesetBuilder
from roop_3d.compositor.pipeline import FrameCompositor, CompositeOptions
from roop_3d.engine.volumetric.session import Replacement
from roop_3d.core.errors import DomainError


def landmarks(closed_left=False, closed_right=False):
    points = np.full((68, 2), 20, np.float32)
    for offset, cx, closed in ((36, 18, closed_left), (42, 46, closed_right)):
        height = .5 if closed else 4
        points[offset:offset+6] = [(cx-8, 20), (cx-4, 20-height), (cx+4, 20-height),
                                 (cx+8, 20), (cx+4, 20+height), (cx-4, 20+height)]
    points[48], points[54] = (20, 44), (44, 44)
    points[62], points[66] = (32, 41), (32, 47)
    points[60:68] = [(20, 44), (26, 41), (32, 41), (38, 41), (44, 44), (38, 47), (32, 47), (26, 47)]
    return points


class Model:
    def __init__(self, output):
        self.output, self.calls = output, []
    def run(self, feed):
        self.calls.append({k: v.copy() for k, v in feed.items()})
        return self.output


def processor(closed_left=False, closed_right=False):
    models = {
        'appearance': Model({'output': np.zeros((1, 32, 16, 64, 64), np.float32)}),
        'eye': Model({'output': np.zeros((1, 63), np.float32)}),
        'stitch': Model({'output': np.zeros((1, 65), np.float32)}),
        'warp_spade': Model({'out': np.full((1, 3, 512, 512), .4, np.float32)}),
    }
    source = Motion(np.full((1, 21, 3), .1, np.float32), np.full((1, 21, 3), .02, np.float32),
                    np.array([[.1, .2, 0]], np.float32), np.ones((1, 1), np.float32), (4., 8., 2.))
    driving = Motion(np.full((1, 21, 3), 10, np.float32), np.full((1, 21, 3), .04, np.float32),
                     np.ones((1, 3), np.float32), np.full((1, 1), 3, np.float32), (30., 40., 50.))
    portrait = SimpleNamespace(models=models, motion=lambda image: source if image[0, 0, 0] == 10 else driving)
    geometry = SimpleNamespace(landmarks=lambda image: landmarks(closed_left, closed_right) if image[0, 0, 0] == 20 else landmarks())
    return ExpressionRestorer(portrait=portrait, geometry=geometry), source, driving


def test_zero_factor_is_byte_exact_and_never_loads_models():
    restorer = ExpressionRestorer()
    image = np.full((64, 64, 3), 10, np.uint8)
    assert restorer.restore(image, image, factor=0) is image
    assert restorer.portrait is None


@pytest.mark.parametrize('factor', [.25, 1.])
def test_current_frame_expression_delta_locks_identity_pose_scale_translation(factor):
    restorer, source, driving = processor()
    image = np.full((64, 64, 3), 10, np.uint8)
    restorer.restore(image, image*2, factor=factor)
    moved = replace(source, expression=source.expression+factor*(driving.expression-source.expression))
    feed = restorer.portrait.models['warp_spade'].calls[-1]
    np.testing.assert_allclose(feed['kp_source'], source.points())
    np.testing.assert_allclose(feed['kp_driving'], moved.points())
    assert restorer.last_report['identity_canonical_locked']
    assert restorer.last_report['pose_locked']


def test_first_frame_wink_enforces_only_closed_eye_without_temporal_baseline():
    restorer, _, _ = processor(closed_left=True)
    source = np.full((64, 64, 3), 10, np.uint8)
    output = restorer.restore(source, source*2, factor=.01)
    assert restorer.last_report['closed_left'] and not restorer.last_report['closed_right']
    eye_calls = restorer.portrait.models['eye'].calls
    assert eye_calls[0]['input'][0, -1] == 0
    assert eye_calls[1]['input'][0, -1] > .16
    assert not np.array_equal(output[15:26, 10:27], np.full((11, 17, 3), 102, np.uint8))
    assert np.all(output[15:26, 37:55] == 102)


def test_per_eye_manual_blink_and_ear_boundary():
    restorer, _, _ = processor()
    image = np.full((64, 64, 3), 10, np.uint8)
    restorer.restore(image, image*2, edit=FaceEditor(blink_right=1))
    assert not restorer.last_report['closed_left'] and restorer.last_report['closed_right']
    assert eye_aspect_ratio(landmarks()[36:42]) == pytest.approx(.5)
    with pytest.raises(DomainError):
        eye_aspect_ratio(np.zeros((6, 2)))


def test_ear_threshold_is_strict_and_does_not_close_an_eye_at_point_16():
    restorer, _, _ = processor()
    target = landmarks()
    target[36:42] = [(0,20), (25,12), (75,12), (100,20), (75,28), (25,28)]
    assert eye_aspect_ratio(target[36:42]) == .16
    restorer.geometry.landmarks = lambda image: target if image[0,0,0] == 20 else landmarks()
    image = np.full((64,64,3), 10, np.uint8)
    restorer.restore(image, image*2, factor=1)
    assert not restorer.last_report['closed_left']


def test_releasing_the_processor_does_not_release_a_shared_portrait_session():
    calls = []
    restorer, _, _ = processor()
    restorer.portrait.release_session = lambda: calls.append('portrait')
    restorer.geometry.release = lambda: calls.append('geometry')
    restorer.prepare()
    restorer.release()
    assert calls == ['geometry']


def test_explicit_pose_and_morph_edits_do_not_mutate_cached_identity():
    restorer, source, _ = processor()
    before = source.canonical.copy()
    image = np.full((64, 64, 3), 10, np.uint8)
    restorer.restore(image, image*2, edit=FaceEditor(head_yaw=15, jaw_width=.4))
    np.testing.assert_array_equal(source.canonical, before)
    assert not restorer.last_report['pose_locked']
    assert not restorer.last_report['identity_canonical_locked']


def test_stage_25_runs_before_ultramax_and_enforces_after_detail_temporal():
    events = []
    compositor = FrameCompositor(parser=SimpleNamespace(predict=lambda image: None))
    def restore(image, target, **kwargs):
        events.append(('expression', kwargs['edit'].head_yaw))
        return image+20
    compositor.expression_restorer = SimpleNamespace(restore=restore, last_report={}, enforce=lambda image: events.append(('enforce', int(image[0, 0, 0]))) or image)
    compositor.restorer = SimpleNamespace(restore=lambda image: events.append(('ultramax', int(image[0, 0, 0]))) or image)
    # Inspection labels can differ from persistent numeric tracker ids.
    compositor.track_labels = {7: 'track_0002'}
    replacement = Replacement(7, np.full((64,64,3), 30, np.uint8), np.eye(2,3), 0, 'fixture')
    options = CompositeOptions(restoration_blend=1., color_strength=0, expression_restore_factor=.5,
                               face_expression_overrides={'track_0002': {'head_yaw': 9}})
    crop = replacement.image
    result, _, _ = compositor._enhance(replacement, crop, crop, np.ones((64,64), np.float32),
        probabilities=None, skin=None, options=options, observer=None, lip_result=None, frame_index=0)
    compositor._stabilize_detail(7, result, crop, crop, np.ones((64,64), np.float32), frame_index=0, scene_cut=True)
    assert events[:2] == [('expression', 9), ('ultramax', 50)]
    assert events[-1][0] == 'enforce'


def test_validation_and_legacy_serialization():
    assert 'expression_restore_factor' not in WorkstationParameters().model_dump()
    assert 'face_expression_overrides' not in WorkstationParameters().model_dump()
    for value in (-.01, 1.01, float('nan')):
        with pytest.raises(ValidationError):
            WorkstationParameters(expression_restore_factor=value)
    with pytest.raises(ValidationError):
        WorkstationParameters(face_expression_overrides={'wrong': {'head_yaw': 1}})
    with pytest.raises(ValidationError):
        WorkstationParameters(face_expression_overrides={'track_0000': {'blink_left': 2}})


def test_36_angle_schedule_preserves_every_requested_grid_pose():
    assert len(SYNTHESIS_POSES) == len(set(SYNTHESIS_POSES)) == 36
    assert all(-75 <= yaw <= 75 and -30 <= pitch <= 30 for yaw, pitch in SYNTHESIS_POSES)
    assert {(float(y), float(p)) for p in (-30,0,30) for y in range(-75,76,15)} <= set(SYNTHESIS_POSES)


def test_synthetic_flags_uvs_embeddings_and_provenance_round_trip(tmp_path):
    state = SimpleNamespace(root=tmp_path, models=None)
    service = FaceExtractService(state)
    samples = tuple(view(yaw=yaw).model_copy(update={'index': i, 'is_synthetic': True, 'uv_map_path': f'uv/{i}.png'}) for i, yaw in enumerate((-60,0,60)))
    document = service._document('generated', 'Generated', samples, {'synthesis': {'view_count': 3}})
    path = tmp_path/'roop_3d/outputs/facesets/generated.fsz'
    export_v2(document, path, app_root=tmp_path)
    with zipfile.ZipFile(path, 'a') as archive:
        for sample in samples:
            archive.writestr(sample.uv_map_path, sample.image_bytes)
    loaded = FaceSetReader().read(path)
    assert all(v.is_synthetic for v in loaded.views)
    assert [v.uv_map_path for v in loaded.views] == [v.uv_map_path for v in samples]
    np.testing.assert_array_equal(loaded.canonical_embedding, document.canonical_embedding)
    assert loaded.metadata['provenance']['synthesis']['view_count'] == 3
    assert all(v.uv_map_bytes for v in loaded.views)
    second = path.with_name('second.fsz')
    export_v2(loaded, second, app_root=tmp_path)
    twice = FaceSetReader().read(second)
    assert twice.metadata['provenance']['synthesis']['view_count'] == 3
    assert twice.views[0].uv_map_bytes == loaded.views[0].uv_map_bytes
    assert PoseSelector(loaded).select_best_reference(0, 0).is_synthetic


def test_generated_uv_contains_only_visible_samples():
    buffers = SimpleNamespace(uv=np.array([[[0,0], [1,1]], [[0,1],[1,0]]], np.float32),
                              silhouette=np.array([[1,0], [0,0]], np.float32))
    image = np.full((2,2,3), 50, np.uint8)
    uv = cv2.imdecode(np.frombuffer(angle_uv(image, buffers, size=8), np.uint8), cv2.IMREAD_UNCHANGED)
    assert np.count_nonzero(uv[:,:,3]) == 1
    assert tuple(uv[7,0]) == (50,50,50,255)


def test_generated_views_never_establish_observed_depth_or_texture():
    from roop_3d.tests import identity_world as world
    from roop_3d.reconstruction.identity_asset import build_identity_asset, IdentityAssetSettings, compute_build_key
    from roop_3d.reconstruction.geometry_types import ConfidenceState
    from roop_3d.reconstruction.faceset_3d import observe_poses
    subject = world.make_subject('generated_fixture', world.LAYOUT_68, 141, subdivisions=1)
    original = world.full_set(subject, (0, -40, 40))
    generated = original.model_copy(update={'views': tuple(v.model_copy(update={'is_synthetic': True}) for v in original.views)})
    settings = IdentityAssetSettings(uv_size=64, subdivisions=1, holdout_validation=False)
    assert compute_build_key(original, settings) != compute_build_key(generated, settings)
    asset = build_identity_asset(generated, settings)
    assert not asset.manifest.claims['depth_triangulated']
    assert not asset.manifest.claims['texture_from_observations_only']
    assert asset.manifest.reconstruction_level == 'generated_view_prior'
    assert not np.any(asset.texel_state == int(ConfidenceState.OBSERVED))
    assert not np.any(asset.vertex_state == int(ConfidenceState.OBSERVED))
    assert all(point['is_synthetic'] for point in observe_poses(generated))


def test_synthesis_rejects_cross_identity_and_synthetic_inputs_before_models_load():
    builder = FacesetBuilder(SimpleNamespace())
    job = draft(); builder.jobs[job['job_id']] = job
    builder.add(job, view(), {})
    ids = list(job['samples'])
    with pytest.raises(DomainError, match='chosen|person|photos'):
        start_synthesis(builder, job['job_id'], 1, ids, 'Wrong person')
    pair = job['samples'][ids[0]]
    job['samples'][ids[0]] = (pair[0].model_copy(update={'is_synthetic': True}), pair[1])
    with pytest.raises(DomainError):
        start_synthesis(builder, job['job_id'], 0, ids, 'Repeated synthesis')


def test_direct_photo_action_rejects_invalid_base64_without_files(tmp_path):
    builder = FacesetBuilder(SimpleNamespace(root=tmp_path))
    with pytest.raises(DomainError):
        start_photo_synthesis(builder, ['not an image'], 'Invalid')
    assert not list(tmp_path.rglob('*.png'))


def test_tensorrt_generator_derivative_has_static_batch_and_lowered_shape_metadata(tmp_path, monkeypatch):
    import onnx
    from onnx import helper, TensorProto
    import roop_3d.models.stage16 as stage16
    source = tmp_path/'reference.onnx'
    graph = helper.make_graph([helper.make_node('GridSample', ['features', 'grid'], ['out'], mode='bilinear')], 'portrait',
        [helper.make_tensor_value_info('features', TensorProto.FLOAT, ['batch',2,3,4,5]),
         helper.make_tensor_value_info('grid', TensorProto.FLOAT, ['batch',2,2,2,3])],
        [helper.make_tensor_value_info('out', TensorProto.FLOAT, ['batch',2,2,2,2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('',16)])
    onnx.save(model, source)
    original = source.read_bytes()
    record = SimpleNamespace(name='liveportrait_warping_spade.onnx', locator=str(source), model_id='fixture',
                             expected_sha256=hashlib.sha256(original).hexdigest())
    catalog = SimpleNamespace(registry=SimpleNamespace(records=lambda: [record]), resolve_for_inference=lambda key: source)
    monkeypatch.setattr(stage16, 'application_root', lambda: tmp_path)
    path, digest, size = stage16.prepare_generator(catalog, batch_one=True)
    derived = onnx.load(path)
    assert all(v.type.tensor_type.shape.dim[0].dim_value == 1 for v in derived.graph.input)
    inferred = {v.name: len(v.type.tensor_type.shape.dim) for v in derived.graph.value_info}
    assert all(n.output[0] in inferred for n in derived.graph.node if n.op_type == 'Unsqueeze')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest and path.stat().st_size == size
    assert source.read_bytes() == original
    assert stage16.prepare_generator(catalog, batch_one=True) == (path, digest, size)


def test_full_synthesis_job_verifies_and_publishes_all_36_views_with_uvs(tmp_path, monkeypatch):
    import roop_3d.faceset_synthesizer as module
    import roop_3d.expression_restorer as expression
    import roop_3d.compositor.restoration as restoration
    import roop_3d.engine.detection.face_tracker as tracking
    import roop_3d.reconstruction.full_head as head
    from roop_3d.engine.runtime import RuntimeConfig
    counts = {'neural': 0, 'ultramax': 0, 'recognition': 0, 'released': 0}
    sample = view()
    class Neural:
        def __init__(self, *args):
            self.geometry = SimpleNamespace(landmarks=lambda image, box: landmarks())
            self.portrait = SimpleNamespace(capability=lambda: {'fixture': True})
        def prepare(self): pass
        def generate_pose(self, source, **kwargs):
            counts['neural'] += 1
            return cv2.imdecode(np.frombuffer(sample.image_bytes, np.uint8), 1)
        def release(self): counts['released'] += 1
    class Restorer:
        def __init__(self, *args): pass
        def restore_hires(self, image, points):
            counts['ultramax'] += 1
            return image
        def capability(self): return {'fixture': True}
        def release(self): counts['released'] += 1
    class Tracker:
        def __init__(self, *args, **kwargs): pass
        def observe(self, image):
            counts['recognition'] += 1
            return [SimpleNamespace(embedding=np.asarray(sample.embedding, np.float32),
                landmarks5=np.asarray(sample.landmarks, np.float32), box=np.array([0,0,112,112]),
                landmarks106=np.random.default_rng(counts['recognition']).uniform(5,110,(106,2)).astype(np.float32))]
        def release(self): counts['released'] += 1
    class Rasterizer:
        def render_buffers(self, *args, **kwargs):
            uv = np.random.default_rng(1).uniform(0,1,(8,8,2)).astype(np.float32)
            return SimpleNamespace(uv=uv, silhouette=np.ones((8,8), np.float32))
        def release(self): counts['released'] += 1
    monkeypatch.setattr(expression, 'ExpressionRestorer', Neural)
    monkeypatch.setattr(restoration, 'UltraMaxPipelineRestorer', Restorer)
    monkeypatch.setattr(tracking, 'FaceTracker', Tracker)
    monkeypatch.setattr(head, 'GPUHeadRasterizer', Rasterizer)
    monkeypatch.setattr(module, 'register_uv', lambda buffers, *args: buffers)
    mesh = {'vertices': np.random.default_rng(1).uniform(-.1,.1,(68,3)).tolist(),
            'indices': [[0,1,2]], 'uvs': np.zeros((68,2)).tolist(),
            'texture_base64': base64.b64encode(sample.image_bytes).decode()}
    monkeypatch.setattr(module, 'reconstruct_faceset', lambda document: {'mesh_full_3d': mesh, 'observed_pose_points': []})
    state = SimpleNamespace(root=tmp_path, models=None)
    state.face_extract = FaceExtractService(state)
    builder = FacesetBuilder(state)
    job = {'status': 'synthesizing', 'scanned': 0}
    module._run(builder, job, (sample,), '36-angle fixture', RuntimeConfig())
    assert job['status'] == 'ready', job
    assert counts == {'neural': 36, 'ultramax': 36, 'recognition': 72, 'released': 4}
    result = job['result']
    path = tmp_path/result['path']
    loaded = FaceSetReader().read(path)
    assert len(loaded.views) == 36 and all(v.is_synthetic for v in loaded.views)
    with zipfile.ZipFile(path) as archive:
        assert all(v.uv_map_path in archive.namelist() for v in loaded.views)
        assert len(json.loads(archive.read('synthesis.json'))['poses']) == 36
    assert path.read_bytes() == (tmp_path/'roop_3d/outputs/facesets'/path.name).read_bytes()
