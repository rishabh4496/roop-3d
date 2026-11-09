"""Full-head regression: the MakeHuman topology must survive the runtime rig."""
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.reconstruction.makehuman_template import makehuman_template
from roop_3d.reconstruction.reenactment import IdentityRig


def head_asset():
    mesh = makehuman_template()
    return SimpleNamespace(
        manifest=SimpleNamespace(schema_id="roop.identity_asset.v1", build_key="test",
            settings={"template": "makehuman"}, sources=[], claims={}),
        vertices=mesh.vertices, faces=mesh.faces, normals=mesh.normals, uvs=mesh.uvs,
        texture=np.full((64, 64, 3), 180, np.uint8),
        texel_confidence=np.ones((64, 64), np.float32),
        texel_state=np.ones((64, 64), np.uint8))


def test_makehuman_runtime_preserves_entire_head():
    asset = head_asset()
    rig = IdentityRig(asset)
    assert len(rig.faces) == len(asset.faces)
    np.testing.assert_allclose(rig.V0[rig.faces], asset.vertices[asset.faces])
    assert np.ptp(rig.V0[:, 1]) > 0.20


@pytest.mark.parametrize("yaw", range(-60, 61, 15))
def test_gpu_full_head_silhouette_and_roi(yaw):
    import cv2
    from roop_3d.reconstruction.full_head import GPUHeadRasterizer, HeadPose, head_roi
    from roop_3d.core.pose import rotation_from_euler
    asset = head_asset()
    rig = IdentityRig(asset)
    pose = HeadPose(rotation_from_euler(yaw, 0, 0), np.array([0., 0., .9]),
                    np.array([[800., 0, 320], [0, 800., 320], [0, 0, 1]]))
    pixels = pose.project(rig.V0)
    inverse = head_roi(pixels)
    local = cv2.transform(pixels.astype(np.float32)[None], cv2.invertAffineTransform(inverse))[0]
    renderer = GPUHeadRasterizer()
    try:
        rendered, alpha = renderer.render(rig.V0, rig.faces, asset.uvs, asset.texture,
            np.ones((64, 64), np.float32), pose, inverse)
    finally:
        renderer.release()
    yy, xx = np.where(alpha > .5)
    assert len(xx) > 10000
    # Independent projected convex hull: a closed head must reach its forehead,
    # lower contour and both profile extrema within raster pixel tolerance.
    for actual, expected in zip((xx.min(), yy.min(), xx.max(), yy.max()),
                                (*local.min(0), *local.max(0))):
        assert abs(actual - expected) <= 3
    # No inner-square seams: central scanlines continue past the middle 128px.
    assert (xx.max() - xx.min()) > 256
    assert yy.max() - yy.min() > 400
    assert rendered[alpha > .9].mean() == pytest.approx(180, abs=1)


def test_view_texture_selects_nearest_pose_and_preserves_unseen():
    from roop_3d.reconstruction.full_head import view_texture
    asset = head_asset()
    asset.view_poses = np.array([[-60, 0, 0], [60, 0, 0]], np.float32)
    asset.view_textures = np.stack([np.full((64, 64, 3), value, np.uint8) for value in (30, 230)])
    asset.view_support = np.ones((2, 64, 64), np.float32)
    assert view_texture(asset, (-60, 0, 0))[0].mean() < 35
    assert view_texture(asset, (60, 0, 0))[0].mean() > 225
    asset.view_support[:] = 0
    assert np.array_equal(view_texture(asset, (60, 0, 0))[0], asset.texture)


def test_dense_pnp_tracks_source_head_without_neural_inner_crop():
    from roop_3d.reconstruction.full_head import FullHeadEngine, HeadPose
    from roop_3d.core.pose import rotation_from_euler
    rig = IdentityRig(head_asset())
    image = np.zeros((640, 640, 3), np.uint8)
    k = np.array([[960., 0, 320], [0, 960., 320], [0, 0, 1]])
    engine = FullHeadEngine(rig)
    try:
        for yaw in (-60, 0, 60):
            pose = HeadPose(rotation_from_euler(yaw, 0, 0), np.array([0., 0., .9]), k)
            face = SimpleNamespace(landmarks5=pose.project(rig.anchors5))
            rendered, alpha, inverse = engine.process(image, face,
                pose.project(rig.V0[rig.landmark_indices]), 0., scene_cut=True, animate=False)
            assert rendered.shape == (512, 512, 3)
            assert alpha.shape == (512, 512)
            assert alpha.sum() > 1000
            expected = pose.project(rig.V0)
            import cv2
            local = cv2.transform(expected[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
            assert local.min() > 20 and local.max() < 492
    finally:
        engine.rasterizer.release()


def test_full_head_compositor_keeps_forehead_and_jaw_and_vetoes_foreground():
    from roop_3d.compositor.pipeline import FrameCompositor, CompositeOptions, RobustOptions
    from roop_3d.compositor.occlusion_analysis import FaceAnalysis
    from roop_3d.engine.volumetric.session import Replacement
    class Analyzer:
        def retain(self, tracks):
            pass
        def analyze(self, crop, *args, **kwargs):
            side = crop.shape[0]
            inner = np.zeros((side, side), np.float32)
            inner[side//3:2*side//3, side//3:2*side//3] = 1
            front = np.zeros_like(inner)
            front[:, side//2:side//2+12] = 1
            return FaceAnalysis(inner, front, inner, None, None, 1., "test")
    frame = np.zeros((256, 256, 3), np.uint8)
    crop = np.full_like(frame, 200)
    coverage = np.zeros((256, 256), np.float32)
    coverage[24:232, 50:206] = 1
    replacement = Replacement(1, crop, np.eye(2, 3, dtype=np.float32), 0, "full_head_gpu",
                              coverage=coverage, full_head=True)
    compositor = FrameCompositor(analyzer=Analyzer())
    options = CompositeOptions(color_strength=0, feather_px=0, robust=RobustOptions(
        refine_matte=False, edge_decontamination=False, sharpness_matching=False,
        local_illumination=0, texture_strength=0))
    result = compositor.compose(frame, [replacement], options=options)
    assert result[40, 80].mean() > 100, "forehead was clipped by the target inner hull"
    assert result[216, 80].mean() > 100, "jaw was clipped by the target inner hull"
    assert result[100, 134].max() == 0, "foreground occluder was overwritten"


def test_session_3d_route_never_calls_inner_face_swapper(monkeypatch):
    from roop_3d.engine.volumetric.session import ReferencePipeline
    from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
    from roop_3d.reconstruction.full_head import HeadPose
    rig = IdentityRig(head_asset())
    pose = HeadPose(np.eye(3), np.array([0., 0., .9]),
                    np.array([[960., 0, 320], [0, 960., 320], [0, 0, 1]]))
    face = FaceObservation(np.array([200., 200., 440., 440.]), pose.project(rig.anchors5),
        np.zeros((106, 2)), np.ones(512), (0., 0., 0.),
        landmarks68=pose.project(rig.V0[rig.landmark_indices]))
    tracked = TrackedFace(1, face, 1, 0)
    session = ReferencePipeline()
    session.tracker = SimpleNamespace(process=lambda *args, **kwargs: (tracked,),
        active_ids=frozenset({1}), scene_cut=False, timestamp_seconds=0., release=lambda: None)
    session.prepared = True
    session.render_mode = "pose_3d"
    session.default_source = SimpleNamespace(faceset_id="test")
    session.neural_face = SimpleNamespace(sources={"test": ()}, release=lambda: None,
        process=lambda *a, **k: SimpleNamespace(image=np.full((512, 512, 3), 210, np.uint8),
            inverse=np.eye(2, 3), coverage=np.ones((512, 512), np.float32), landmarks5=face.landmarks5))
    def inner_crop_forbidden(*args, **kwargs):
        pytest.fail("3D mode called the inner-face swapper")
    monkeypatch.setattr(session.swapper, "infer_frame", inner_crop_forbidden)
    try:
        result = session.infer_frame(np.zeros((640, 640, 3), np.uint8))
        replacement, = result.replacements
        assert not replacement.full_head and replacement.fallback_reason is None
        assert replacement.method == "dense_3d_correspondence_neural_ultramax"
        assert np.all(replacement.image == 210)
        assert replacement.image.shape == (512, 512, 3)
    finally:
        session.release_session()


def test_neural_detail_bridge_spans_head_and_preserves_coverage():
    from roop_3d.reconstruction.full_head import bridge_neural_detail
    rendered = np.full((512, 512, 3), 150, np.uint8)
    neural = rendered.copy()
    neural[40:460:4, :, :] = 200
    coverage = np.zeros((512, 512), np.float32)
    coverage[30:480, 80:440] = 1
    refined = bridge_neural_detail(rendered, neural, coverage)
    assert np.array_equal(refined[coverage == 0], rendered[coverage == 0])
    assert refined[40, 120].mean() > 150  # forehead, well above inner-face box
    assert refined[448, 120].mean() > 150  # lower jaw


def test_archive_preserves_all_dense_layouts_and_embeddings(tmp_path):
    from roop_3d.faceset.faceset_reader import FaceSetReader, export_v2
    from roop_3d.faceset_loader import enrich_document
    from roop_3d.tests import identity_world as W
    subject = W.make_subject("dense", "68pt", 123)
    document = W.full_set(subject, (-60, 0, 60))
    views = tuple(view.model_copy(update={
        "landmarks_106": tuple((float(i), float(i + 1)) for i in range(106)),
        "landmarks_468": tuple((float(i), float(i + 2)) for i in range(468))}) for view in document.views)
    document = document.model_copy(update={"views": views})
    path = tmp_path / "roop_3d/outputs/facesets/dense.fsz"
    export_v2(document, path, app_root=tmp_path)
    loaded = enrich_document(FaceSetReader(cache_root=tmp_path / "cache").read(path), None, cache=False)
    assert len(loaded.document.views) == 3
    for before, after in zip(document.views, loaded.document.views):
        assert before.landmarks_68 == after.landmarks_68
        assert before.landmarks_106 == after.landmarks_106
        assert before.landmarks_468 == after.landmarks_468
        np.testing.assert_allclose(before.embedding, after.embedding, atol=1e-6)
