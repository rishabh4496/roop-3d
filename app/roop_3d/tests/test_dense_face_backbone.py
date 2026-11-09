"""Analytical geometry and neural-routing regressions; no pretrained weights claimed."""
from types import SimpleNamespace
import hashlib
import json

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.dense_correspondence import (
    ReferenceSurfaceRasterizer, GPUSurfaceRasterizer, correspondence_field,
)
from roop_3d.reconstruction.morphable_backbone import FaceTopology, MorphableBackbone, TOPOLOGIES, sample_uv_displacement


@pytest.mark.parametrize("family", TOPOLOGIES)
def test_semantic_cut_physically_removes_excluded_vertices_and_boundary_triangles(tmp_path, family):
    count = TOPOLOGIES[family]
    mask = np.zeros(count, bool)
    mask[:6] = True
    excluded = np.zeros(count, bool)
    excluded[3] = True
    parts = np.zeros((8, count), bool)
    parts[:, :6] = True
    path = tmp_path / "topology.npz"
    np.savez(path, vertices_m=np.zeros((count, 3), np.float32), uvs=np.zeros((count, 2)),
             faces=np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]]), face_surface=mask,
             scalp_hair=np.zeros(count, bool), ears=excluded, neck=np.zeros(count, bool), semantic_parts=parts)
    topology = FaceTopology.load(family, path)
    assert 3 not in topology.vertex_ids
    assert len(topology.faces) == 2
    assert np.max(topology.faces) < len(topology.vertex_ids)
    assert np.all(topology.vertex_ids[topology.faces] != 3)


def test_missing_bundle_fails_with_actionable_reason(tmp_path):
    from roop_3d.engine.runtime import RuntimeConfig
    with pytest.raises(DomainError) as error:
        MorphableBackbone.load(None, RuntimeConfig(), tmp_path / "missing.json")
    assert error.value.code == "DENSE_FACE_ASSETS_MISSING"


def test_bundle_signature_invalidates_canonical_previews_after_model_install_or_replacement(tmp_path, monkeypatch):
    import roop_3d.reconstruction.morphable_backbone as module
    monkeypatch.setattr(module, "application_root", lambda: tmp_path)
    directory = tmp_path / "roop_3d/.runtime/models/dense_face"
    directory.mkdir(parents=True)
    missing = module.bundle_signature()
    asset = directory / "coarse.onnx"
    asset.write_bytes(b"first checkpoint identity")
    bundle = directory / "bundle.json"
    bundle.write_text(json.dumps({"assets": {"coarse": {"file": "coarse.onnx"}}}))
    installed = module.bundle_signature()
    assert installed != missing
    asset.write_bytes(b"a different checkpoint identity with different bytes")
    assert module.bundle_signature() != installed


def test_native_manifest_selects_native_decoders_without_custom_graph_wrappers(tmp_path, monkeypatch):
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.reconstruction.native_backbone import NativeMorphableBackbone, NATIVE_SCHEMA
    path = tmp_path / "bundle.json"
    path.write_text(json.dumps({"schema": NATIVE_SCHEMA}))
    expected = object()
    monkeypatch.setattr(NativeMorphableBackbone, "load", lambda *args: expected)
    assert MorphableBackbone.load(None, RuntimeConfig(), path) is expected


@pytest.mark.parametrize("bad", ["hash", "path"])
def test_bundle_integrity_fails_before_loading_any_graph(tmp_path, bad):
    from roop_3d.engine.runtime import RuntimeConfig
    directory = tmp_path / "bundle"
    directory.mkdir()
    payload = b"invalid asset must never reach a graph loader"
    asset = directory / "asset.bin" if bad == "hash" else tmp_path / "outside.bin"
    asset.write_bytes(payload)
    record = {"file": "asset.bin" if bad == "hash" else "../outside.bin",
              "sha256": "0"*64 if bad == "hash" else hashlib.sha256(payload).hexdigest()}
    spec = {"schema": "roop3d.dense_face_bundle.v1", "topology": "bfm35709",
            "coordinates": "opencv_metres", "uv_origin": "bottom_left", "crop": "arcface_128_rgb_0_1",
            "detail_topology": "bfm35709", "mask_provenance": "test", "expression_mapping_evidence": "test",
            "detail_crop_size": 512,
            "expression_basis": "arkit52", "assets": {role: record for role in ("topology", "coarse", "expression", "detail")}}
    bundle = directory / "bundle.json"
    bundle.write_text(json.dumps(spec))
    with pytest.raises(DomainError) as error:
        MorphableBackbone.load(None, RuntimeConfig(), bundle)
    assert error.value.code == "DENSE_FACE_BUNDLE_INVALID"


def test_unavailable_backbone_releases_session_and_never_uses_legacy_cg(monkeypatch):
    import roop_3d.engine.volumetric.session as module
    from roop_3d.reconstruction.neural_face import NeuralFaceEngine
    session = module.ReferencePipeline()
    released = []
    monkeypatch.setattr(session.swapper, "prepare_session", lambda *a, **k: None)
    monkeypatch.setattr(module, "FaceTracker", lambda *a, **k:
                        SimpleNamespace(release=lambda: released.append("tracker")))
    def unavailable(*args, **kwargs):
        raise DomainError("DENSE_FACE_ASSETS_MISSING", "Missing test bundle", 503)
    monkeypatch.setattr(NeuralFaceEngine, "load", unavailable)
    with pytest.raises(DomainError) as error:
        session.prepare_session(SimpleNamespace(faceset_id="test"), render_mode="pose_3d")
    assert error.value.code == "DENSE_FACE_ASSETS_MISSING"
    assert released == ["tracker"]
    assert not session.prepared and session.neural_face is None
    assert not session.full_head_engines


def test_three_tiers_preserve_asymmetry_and_apply_metric_displacement():
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    neutral = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0]], np.float32)
    topology = FaceTopology("flame2023open", 3, np.arange(3), np.array([[0, 2, 1]]),
                            np.array([[.1, .1], [.9, .1], [.1, .9]], np.float32))
    expressed = neutral.copy()
    expressed[0, 1] = .02  # one-sided expression, deliberately not mirrored
    mid = np.full((1, 3, 3), .001, np.float32)
    coefficients = np.zeros((1, 52), np.float32)
    coefficients[0, 0] = .8
    class Graph:
        def __init__(self, output, expected):
            self.output, self.expected = output, expected
        def run(self, inputs):
            assert set(inputs) == self.expected
            return self.output
    coarse = Graph({"identity_vertices_m": neutral[None], "rotation": np.eye(3)[None],
        "translation_m": np.array([[0, 0, 1]]), "scale": np.ones((1, 1)),
        "intrinsics": np.diag([224, 224, 1])[None]}, {"rgb"})
    expression = Graph({"coefficients": coefficients, "expressed_vertices_m": expressed[None]},
                       {"rgb", "identity_vertices_m"})
    detail = Graph({"mid_offsets_m": mid, "displacement_uv_m": np.full((1, 1, 64, 64), .005)},
                   {"rgb", "rgb_hd", "vertices_m"})
    engine = MorphableBackbone(topology, coarse, expression, detail, expression_basis="arkit52", evidence={})
    points = ARCFACE_REFERENCE_5PTS.copy()
    points[:, 0] += 8
    fitted = engine.fit(np.zeros((128, 128, 3), np.uint8), points)
    expected = expressed + mid[0]
    expected[:, 2] -= .005  # outward-facing normal points toward camera
    np.testing.assert_allclose(fitted.vertices, expected, atol=1e-7)
    assert fitted.expression[0] == pytest.approx(.8)
    assert fitted.expression[1] == 0
    np.testing.assert_allclose(fitted.camera_vertices, expected + (0, 0, 1), atol=1e-7)
    coarse.output["rotation"] = np.zeros((1, 3, 3))
    with pytest.raises(ValueError, match="SO\\(3\\)"):
        engine.fit(np.zeros((128, 128, 3), np.uint8), points)


def test_high_frequency_sampling_supports_all_35709_vertices_and_uv_origin():
    u = np.linspace(0, 1, 35709, dtype=np.float32)
    uv = np.column_stack((u, 1-u))
    y, x = np.mgrid[:64, :64]
    displacement = (x+2*y).astype(np.float32)/63*.001
    sampled = sample_uv_displacement(displacement, uv)
    assert sampled.shape == (35709,)
    np.testing.assert_allclose(sampled, 3*u*.001, atol=1e-9)


def test_identity_correspondence_samples_original_photographic_pixels():
    raster = ReferenceSurfaceRasterizer()
    vertices = np.array([[2, 2, 1], [29, 2, 1], [2, 29, 1]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    k = np.eye(3, dtype=np.float32)
    depth = raster.rasterize(vertices, faces, k, vertices, (32, 32)).depth
    field = correspondence_field(raster, faces, vertices, k, vertices, k, (32, 32), depth, (32, 32))
    y, x = np.mgrid[:32, :32]
    photograph = np.stack((x*7, y*7, (x+y)*3), -1).astype(np.uint8)
    sampled = field.sample(photograph)
    valid = field.confidence > 0
    assert valid.sum() > 300
    np.testing.assert_array_equal(sampled[valid], photograph[valid])
    assert np.all(field.source_xy[~valid] == -1)
    supported = correspondence_field(raster, faces, vertices, k, vertices, k, (32, 32), depth,
                                     (32, 32), source_support=np.zeros((32, 32), np.float32))
    assert supported.confidence.sum() == 0


def test_perspective_correspondence_matches_independent_camera_ray_solution():
    raster = ReferenceSurfaceRasterizer()
    target = np.array([[0, 0, 1], [2, 0, 2], [0, 2, 2]], np.float32)
    source = target + np.array([.2, .1, 1], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    k = np.diag([20, 20, 1]).astype(np.float32)
    depth = raster.rasterize(source, faces, k, source, (32, 32)).depth
    field = correspondence_field(raster, faces, source, k, target, k, (32, 32), depth, (32, 32), tolerance_m=.2)
    # Target plane z=1+(x+y)/2. Ray at pixel(5,5): (u*z/f,v*z/f,z).
    z = 1 / (1-(5+5)/40)
    hit = np.array([5*z/20, 5*z/20, z]) + (.2, .1, 1)
    expected = hit[:2]/hit[2]*20
    assert field.confidence[5, 5] == 1
    np.testing.assert_allclose(field.source_xy[5, 5], expected, atol=1e-5)


def test_source_occlusion_rejects_surface_even_when_visible_in_target():
    raster = ReferenceSurfaceRasterizer()
    front = np.array([[2, 2, 1], [18, 2, 1], [2, 18, 1]], np.float32)
    source = np.vstack((front, front*2))
    target = source.copy()
    target[3:, 0] += 40
    faces = np.array([[0, 1, 2], [3, 4, 5]], np.int32)
    k = np.eye(3, dtype=np.float32)
    depth = raster.rasterize(source, faces, k, source, (64, 32)).depth
    field = correspondence_field(raster, faces, source, k, target, k, (64, 32), depth, (64, 32))
    assert field.confidence[4, 4] == 1
    assert field.confidence[4, 24] == 0  # hidden source triangle, visible target triangle


def test_gpu_geometry_interpolation_matches_cpu_reference():
    gpu = GPUSurfaceRasterizer()
    target = np.array([[.1, .1, 1], [1.5, .1, 2], [.1, 1.5, 2]], np.float32)
    faces = np.array([[0, 1, 2]], np.int32)
    k = np.diag([32, 32, 1]).astype(np.float32)
    try:
        try:
            actual = gpu.rasterize(target, faces, k, target + (.2, .1, 1), (32, 32))
        except DomainError as error:
            pytest.skip(str(error))
        expected = ReferenceSurfaceRasterizer().rasterize(target, faces, k, target + (.2, .1, 1), (32, 32))
        inside = cv2.erode(expected.valid.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        assert inside.sum() > 20
        assert actual.valid[inside].all()
        # Hardware triangle setup uses subpixel fixed-point coordinates. Bound
        # its metric interpolation error (0.2mm), rather than require bit parity.
        np.testing.assert_allclose(actual.attributes[inside], expected.attributes[inside], atol=2e-4)
        np.testing.assert_allclose(actual.depth[inside], expected.depth[inside], atol=2e-4)
    finally:
        gpu.release()


@pytest.mark.parametrize("native_size,template", [(128, "arcface_112"), (256, "arcface_112"),
                                                 (512, "arcface_112"), (512, "ffhq_512")])
def test_full_neural_output_and_ultramax_own_all_replacement_pixels(native_size, template):
    from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper
    from roop_3d.engine.volumetric.preprocess import aligned_crop
    from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
    from roop_3d.reconstruction.neural_face import NeuralFaceEngine
    points = ARCFACE_REFERENCE_5PTS.copy()
    points[:, 0] += 8
    image = np.full((128, 128, 3), 45, np.uint8)
    vertices = np.array([[.1, .1, 1], [.9, .1, 1], [.1, .9, 1]], np.float32)
    faces = np.array([[0, 2, 1]], np.int32)
    class Backbone:
        topology = SimpleNamespace(faces=faces)
        def fit(self, image, landmarks):
            return SimpleNamespace(camera_vertices=vertices, intrinsics=np.diag([224, 224, 1]),
                rotation=np.eye(3), alignment=aligned_crop(image, np.asarray(landmarks), 128))
    class Enhancer:
        def restore_hires(self, crop, landmarks):
            assert crop.shape == (native_size, native_size, 3)
            assert np.all(crop == 190), "photographic warp bypassed the neural swapper"
            return np.full((256, 256, 3), 210, np.uint8)
    class Model:
        resolution = native_size
        spec = {"alignment_template": template}
        def infer(self, crop, embedding, *, source_view):
            assert embedding.shape == (512,)
            assert np.any(crop == 45)
            return np.full_like(crop, 190)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    view = SimpleNamespace(index=0, image_bytes=encoded.tobytes(), landmarks=points,
        is_synthetic=False, landmarks_trusted=True, yaw=0, pitch=0, quality=SimpleNamespace(score=1))
    document = SimpleNamespace(faceset_id="test", views=(view,))
    swapper = IdentitySwapper.__new__(IdentitySwapper)
    swapper.model, swapper.source, swapper.sources_by_track, swapper.projection = Model(), document, {}, None
    swapper.reference_embedding = lambda *args: (np.ones(512, np.float32), 0, 1)
    engine = NeuralFaceEngine(Backbone(), ReferenceSurfaceRasterizer(), Enhancer())
    engine.prepare(document)
    observed = engine.sources[document.faceset_id][0].observed
    observed[:, :observed.shape[1]//2] = 0  # source appearance is missing on half the facial surface
    result = engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper, track_id=0)
    assert np.all(result.image == 210)
    assert result.coverage.sum() > 100
    assert result.coverage[0, 0] == 0
    assert len(engine.last_fields) == 1
    native_coverage = cv2.resize(result.coverage, (native_size, native_size), interpolation=cv2.INTER_NEAREST)
    assert ((engine.last_fields[0].confidence == 0) & (native_coverage > .9)).sum() > 10, "source holes retained target identity instead of neural skin"
    recovered_points = result.landmarks5 @ result.inverse[:, :2].T + result.inverse[:, 2]
    np.testing.assert_allclose(recovered_points, points, atol=1e-5)
    observed[:] = 0
    with pytest.raises(DomainError) as unsupported:
        engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper, track_id=0)
    assert unsupported.value.code == "DENSE_FACE_NO_CORRESPONDENCE"
    swapper.model = None
    with pytest.raises(DomainError) as error:
        engine.process(image, SimpleNamespace(landmarks5=points, pose=(0, 0, 0)), document, swapper, track_id=0)
    assert error.value.code == "DENSE_FACE_NEURAL_REQUIRED"
