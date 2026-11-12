"""Full-head coverage, perspective visibility and old-jaw removal regressions.

Synthetic camera/texture fixtures give exact geometric ground truth. Real-photo
acceptance is separate: synthetic poses do not prove landmark accuracy on video.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.tests.test_full_head import head_asset
from roop_3d.reconstruction.full_head import GPUHeadRasterizer, HeadPose, head_roi, view_texture
from roop_3d.reconstruction.reenactment import IdentityRig
from roop_3d.core.pose import rotation_from_euler


def test_view_bank_observation_restores_canonical_coverage():
    asset = head_asset()
    asset.texel_state[:] = 0
    asset.texel_confidence[:] = 0
    asset.view_poses = np.array([[80, 0, 0]], np.float32)
    asset.view_textures = np.full((1, 64, 64, 3), 210, np.uint8)
    asset.view_support = np.ones((1, 64, 64), np.float32)
    image, coverage = view_texture(asset, (80, 0, 0))
    assert np.all(coverage == 1), "observed profile texels became transparent"
    assert np.all(image == 210)


def test_exact_view_does_not_import_other_hemisphere_texture():
    asset = head_asset()
    asset.view_poses = np.array([[80, 0, 0], [60, 0, 0], [-80, 0, 0]], np.float32)
    asset.view_textures = np.stack([np.full((64, 64, 3), v, np.uint8) for v in (200, 50, 0)])
    asset.view_support = np.ones((3, 64, 64), np.float32)
    assert np.all(view_texture(asset, (80, 0, 0))[0] == 200)


def test_slimmer_head_removes_old_jaw_and_preserves_foreground():
    from roop_3d.compositor.pipeline import FrameCompositor, CompositeOptions, RobustOptions
    from roop_3d.compositor.occlusion_analysis import FaceAnalysis
    side = 128
    frame = np.full((side, side, 3), (30, 60, 90), np.uint8)
    target = np.zeros((side, side), np.float32)
    target[20:112, 20:108] = 1
    frame[target > 0] = (180, 140, 100)
    source = np.zeros_like(target)
    source[20:112, 36:92] = 1
    front = np.zeros_like(target)
    front[72:88, 20:108] = 1
    frame[front > 0] = (10, 220, 10)
    class Analyzer:
        def retain(self, tracks): pass
        def release(self): pass
        def analyze(self, crop, *args, **kwargs):
            resize = lambda m: cv2.resize(m, (crop.shape[1], crop.shape[0]))
            return FaceAnalysis(resize(target), resize(front), resize(target), None, None, 1., "test")
    item = SimpleNamespace(track_id=1, image=np.full_like(frame, 210), inverse=np.eye(2, 3),
                           coverage=source, full_head=True, target_silhouette=target)
    compositor = FrameCompositor(analyzer=Analyzer())
    options = CompositeOptions(color_strength=0, feather_px=0, robust=RobustOptions(
        analysis_size=128, refine_matte=False, edge_decontamination=False, sharpness_matching=False,
        local_illumination=0, texture_strength=0))
    try:
        result = compositor.compose(frame, [item], options=options)
    finally:
        compositor.release()
    assert np.linalg.norm(result[100, 26].astype(float) - frame[0, 0]) < 8, "old wide jaw survived"
    np.testing.assert_array_equal(result[front > 0], frame[front > 0])
    np.testing.assert_array_equal(result[target == 0], frame[target == 0])
    assert result[100, 60].mean() > 150


@pytest.mark.parametrize("yaw", [-85, -80, -60, -35, 0, 35, 60, 80, 85])
@pytest.mark.parametrize("pitch", [-60, 0, 60])
def test_full_mesh_silhouette_survives_extreme_camera(yaw, pitch):
    asset = head_asset()
    rig = IdentityRig(asset)
    pose = HeadPose(rotation_from_euler(yaw, pitch, 0), np.array([0., 0., .9]),
                    np.array([[800., 0, 320], [0, 800., 320], [0, 0, 1]]))
    inverse = head_roi(pose.project(rig.V0), side=256)
    renderer = GPUHeadRasterizer()
    try:
        image, alpha = renderer.render(rig.V0, rig.faces, asset.uvs, asset.texture,
                                      np.ones((64, 64), np.float32), pose, inverse, 256)
    finally:
        renderer.release()
    local = cv2.transform(pose.project(rig.V0)[None].astype(np.float32), cv2.invertAffineTransform(inverse))[0]
    yy, xx = np.where(alpha > .5)
    for actual, expected in zip((xx.min(), yy.min(), xx.max(), yy.max()), (*local.min(0), *local.max(0))):
        assert abs(actual - expected) <= 3
    assert np.all(image[alpha > .9] == 180)


@pytest.mark.parametrize("yaw", [-85, -80, 80, 85])
def test_gpu_depth_culls_hidden_texture_without_horizontal_smearing(yaw):
    # Two exactly overlapping surfaces. Far-side UVs contain a bright marker
    # that must never leak through the foreground, regardless of draw order.
    pose = HeadPose(rotation_from_euler(yaw, 0, 0), np.array([0., 0., 2.]),
                    np.array([[100., 0, 32], [0, 100., 32], [0, 0, 1]]))
    camera = np.array([[-.4, -.4, 1.], [-.4, .4, 1.], [.4, -.4, 1.]])
    camera = np.vstack((camera, camera * 1.5))
    vertices = ((camera - pose.translation) @ pose.rotation) * (1, 1, -1)
    uv = np.array([[.25, .5]] * 3 + [[.75, .5]] * 3)
    texture = np.zeros((32, 32, 3), np.uint8)
    texture[:, :16] = (25, 80, 120)
    texture[:, 16:] = (255, 0, 255)
    renderer = GPUHeadRasterizer()
    try:
        image, alpha = renderer.render(vertices, np.array([[0, 1, 2], [3, 4, 5]]), uv,
            texture, np.ones((32, 32), np.float32), pose, np.eye(2, 3), 64)
        assert alpha.sum() > 100
        np.testing.assert_array_equal(image[alpha > .9], np.tile((25, 80, 120), ((alpha > .9).sum(), 1)))
        centers = np.stack((vertices[:3].mean(0), vertices[3:].mean(0)))
        assert renderer.visibility(centers, pose, np.eye(2, 3)).tolist() == [True, False]
    finally:
        renderer.release()


@pytest.mark.parametrize("yaw", [-85, -80, -60, -35, 0, 35, 60, 80, 85])
@pytest.mark.parametrize("pitch", [-60, 0, 60])
def test_dense_pnp_and_composite_modify_full_forehead_cheek_and_jaw(yaw, pitch):
    from roop_3d.reconstruction.full_head import FullHeadEngine
    from roop_3d.compositor.pipeline import FrameCompositor, CompositeOptions, RobustOptions
    from roop_3d.compositor.occlusion_analysis import FaceAnalysis
    asset = head_asset()
    # Missing forehead/profile observation must not reveal the original actor.
    asset.texel_state[:24] = 0
    asset.texel_confidence[:24] = 0
    asset.texture[:24] = 0
    rig = IdentityRig(asset)
    k = np.array([[960., 0, 320], [0, 960., 320], [0, 0, 1]])
    expected = HeadPose(rotation_from_euler(yaw, pitch, 0), np.array([0., 0., .9]), k)
    face = SimpleNamespace(landmarks5=expected.project(rig.anchors5))
    engine = FullHeadEngine(rig, side=256)
    frame = np.zeros((640, 640, 3), np.uint8)
    class Analyzer:
        def retain(self, tracks): pass
        def release(self): pass
        def analyze(self, crop, *args, **kwargs):
            z = np.zeros(crop.shape[:2], np.float32)
            return FaceAnalysis(z, None, z, None, None, 1., "test")
    compositor = FrameCompositor(analyzer=Analyzer())
    try:
        rendered, alpha, inverse = engine.process(frame, face,
            expected.project(rig.V0[rig.landmark_indices]), 0., animate=False, camera_matrix=k)
        np.testing.assert_allclose(engine.last_pose.projection_matrix, expected.projection_matrix, atol=.05)
        # Source coverage now includes a soft 8 mm submental transition ribbon.
        assert np.all(alpha <= engine.source_silhouette)
        assert alpha.sum() > engine.source_silhouette.sum() * .85
        # Measured oral apertures now retain the target's cavity shadows. This
        # coverage gate measures skin, which must never reveal the old actor.
        skin_alpha = alpha * (1 - engine.detail.mouth)
        assert np.all(rendered[skin_alpha > .99] > 100)
        item = SimpleNamespace(track_id=0, image=rendered, inverse=inverse, coverage=alpha,
            source_silhouette=engine.source_silhouette, target_silhouette=engine.target_silhouette, full_head=True)
        options = CompositeOptions(color_strength=0, feather_px=0, robust=RobustOptions(
            analysis_size=256, refine_matte=False, edge_decontamination=False, sharpness_matching=False,
            texture_strength=0, local_illumination=0))
        result = compositor.compose(frame, [item], options=options)
        support = cv2.warpAffine(skin_alpha, inverse, (640, 640)) > .99
        yy, xx = np.where(support)
        # Bands span the actual projected skull, including pitch-dependent extrema.
        for lo, hi in ((0, .2), (.35, .65), (.8, 1.)):
            band = support.copy()
            band[:int(yy.min() + (yy.max()-yy.min()) * lo)] = False
            band[int(yy.min() + (yy.max()-yy.min()) * hi)+1:] = False
            assert band.any()
            assert np.mean(np.any(result[band] != frame[band], axis=1)) > .98
    finally:
        engine.rasterizer.release()
        compositor.release()


def test_spherical_bank_top_two_and_roll_invariance():
    from roop_3d.reconstruction.view_texture_bank import nearest_views, BUCKETS, pose_bucket
    poses = np.array([[-60, 0, 0], [0, 0, 0], [60, 0, 0]])
    ids, weights = nearest_views(poses, (30, 0, 0))
    assert set(ids) == {1, 2}
    np.testing.assert_allclose(weights, [.5, .5])
    other, same = nearest_views(poses, (30, 0, 170))
    np.testing.assert_array_equal(ids, other)
    np.testing.assert_allclose(weights, same)
    assert {pose_bucket(y, p) for y, p in ((0, 0), (-30, 0), (30, 0), (-80, 0), (80, 0),
        (0, 30), (0, -30), (-45, 30), (45, -30))} == set(BUCKETS)


def test_selected_neural_model_refines_entire_head_without_alignment(monkeypatch):
    from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper
    import roop_3d.engine.volumetric.identity_swapper as module
    swapper = IdentitySwapper()
    swapper.source = SimpleNamespace(views=[SimpleNamespace(index=0)])
    swapper.reference_embedding = lambda *a: (np.ones(512, np.float32) / np.sqrt(512), 0, 1)
    calls = []
    def infer(image, embedding, **kwargs):
        calls.append(image.copy())
        return image
    swapper.model = SimpleNamespace(resolution=256, infer=infer)
    monkeypatch.setattr(module, "roll_compensated_aligned_crop", lambda *a, **k: pytest.fail("5pt head alignment"))
    rendered = np.full((512, 512, 3), 180, np.uint8)
    rendered[:60] = 90
    output = swapper.refine_head_texture(rendered, np.ones((512, 512), np.float32), pose=(80, 0, 0))
    assert calls[0][0].mean() == 90 and calls[0][-1].mean() == 180
    assert output.shape == rendered.shape


@pytest.mark.parametrize("yaw,pitch", [(35, 0), (80, 0), (-80, 0), (0, 30), (0, -30)])
def test_standard_session_routes_extreme_pose_to_mesh(monkeypatch, yaw, pitch):
    from roop_3d.engine.volumetric.session import ReferencePipeline
    from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
    rig = IdentityRig(head_asset())
    pose = HeadPose(rotation_from_euler(yaw, pitch, 0), np.array([0., 0., .9]),
                    np.array([[960., 0, 320], [0, 960., 320], [0, 0, 1]]))
    face = FaceObservation(np.array([200., 200., 440., 440.]), pose.project(rig.anchors5),
        np.zeros((106, 2)), np.ones(512), (yaw, -pitch, 0),
        landmarks68=pose.project(rig.V0[rig.landmark_indices]))
    tracked = TrackedFace(0, face, 1, 0)
    session = ReferencePipeline()
    session.tracker = SimpleNamespace(process=lambda *a, **k: (tracked,), active_ids={0},
        scene_cut=False, timestamp_seconds=0., release=lambda: None)
    session.prepared = True
    session.default_source = SimpleNamespace(faceset_id="test")
    session.neural_face = SimpleNamespace(sources={"test": ()}, release=lambda: None, retain_tracks=lambda ids: None, last_diag=None,
        process=lambda *a, **k: SimpleNamespace(image=np.full((512, 512, 3), 210, np.uint8),
            inverse=np.eye(2, 3), coverage=np.ones((512, 512), np.float32), landmarks5=face.landmarks5))
    monkeypatch.setattr(session.swapper, "infer_frame", lambda *a, **k: pytest.fail("extreme target reached 5pt swapper"))
    try:
        result = session.infer_frame(np.zeros((640, 640, 3), np.uint8))
        item, = result.replacements
        assert not item.full_head and item.render_mode == "pose_3d"
        assert item.method == "dense_3d_correspondence_neural_ultramax"
    finally:
        session.release_session()


def test_explicit_468_correspondences_and_calibrated_camera_without_five_points():
    from roop_3d.reconstruction.full_head import FullHeadEngine
    rig = IdentityRig(head_asset())
    indices = np.linspace(0, len(rig.V0)-1, 468, dtype=np.int64)
    k = np.array([[810., 0, 313], [0, 850., 307], [0, 0, 1]])
    pose = HeadPose(rotation_from_euler(80, -25, 12), np.array([.01, -.02, .9]), k)
    points = pose.project(rig.V0[indices])
    # Large detector outliers must not steer a profile's camera.
    points[::40] += (75, -90)
    engine = FullHeadEngine(rig, side=256)
    try:
        engine.process(np.zeros((640, 640, 3), np.uint8), SimpleNamespace(), points, 0.,
            animate=False, camera_matrix=k, landmark_indices=indices)
        np.testing.assert_allclose(engine.last_pose.projection_matrix, pose.projection_matrix, atol=.1)
        assert engine.target_silhouette is None  # no invented iBUG jaw ordering
    finally:
        engine.rasterizer.release()


def test_mesh_forehead_keeps_confident_hair_during_parser_warmup():
    from roop_3d.compositor.occlusion_analysis import OcclusionAnalyzer
    scores = np.zeros((128, 128, 19), np.float32)
    scores[..., 0] = 1
    scores[:24, :, 0] = 0
    scores[:24, :, 17] = 1
    analyzer = OcclusionAnalyzer(parser=SimpleNamespace(predict=lambda crop: scores), net_names=())
    analysis = analyzer.analyze(np.zeros((128, 128, 3), np.uint8), None,
                                mesh_support=np.ones((128, 128), np.float32), track_id=1)
    assert not analysis.parser_trusted
    assert np.all(analysis.occluder[:24] == 1)
    assert np.all(analysis.occluder[24:] == 0)


@pytest.mark.parametrize("yaw", [-85, -75, 75, 85])
def test_uv_acquisition_never_bakes_visible_texture_into_hidden_surface(yaw):
    from roop_3d.reconstruction.geometry_types import Mesh3D, WeakPerspectiveCamera
    from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation
    rotation = rotation_from_euler(yaw, 0, 0)
    # Both triangles face the source camera and overlap exactly in projection;
    # normal culling alone cannot distinguish the occluded rear UV island.
    near = np.array([[-.06, -.06, .04], [.06, -.06, .04], [-.06, .06, .04]])
    far = near.copy(); far[:, 2] = 0
    vertices = np.vstack((near, far)) @ rotation
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    uv = np.array([[0, 0], [.45, 0], [0, 1], [.55, 0], [1, 0], [.55, 1]])
    mesh = Mesh3D(vertices, faces, uvs=uv)
    camera = WeakPerspectiveCamera(rotation=tuple(map(tuple, rotation)), scale=800, tx=64, ty=64)
    synth = UVSynthesizer(mesh, uv_size=64, max_interpolation_fraction=0, visibility_backend="gpu")
    result = synth.synthesize([ViewObservation(image=np.full((128, 128, 3), 200, np.uint8), camera=camera)])
    ys, xs = np.indices((64, 64))
    rear = synth.face_texel_mask & (xs > 36)
    front = synth.face_texel_mask & (xs < 25)
    assert rear.sum() > 500
    assert np.all(result.confidence_mask[rear] == 0)
    assert np.all(result.texture_map[rear] == 0)
    assert np.any(result.confidence_mask[front] == 2)


@pytest.mark.parametrize("yaw", [0, 35, 60, 80])
def test_real_target_frames(yaw):
    """Opt-in captured-photo acceptance; missing real profiles are explicit skips.

    fixtures/extreme_pose_real/ contains source.fsz and targets.fsz. Targets
    must be captured photos with measured dense landmarks and actual pose labels,
    not frontal photographs transformed into nominal profile-angle fixtures.
    """
    from pathlib import Path
    from roop_3d.faceset.faceset_reader import FaceSetReader
    from roop_3d.faceset_loader import LandmarkEngine, enrich_document
    from roop_3d.reconstruction.identity_asset import build_identity_asset, IdentityAssetSettings
    from roop_3d.reconstruction.full_head import FullHeadEngine
    from roop_3d.audit.project_paths import application_root
    root = Path(__file__).parent / "fixtures/extreme_pose_real"
    if not (root / "source.fsz").is_file() or not (root / "targets.fsz").is_file():
        pytest.skip("Captured real-pose source.fsz and targets.fsz fixtures are not installed")
    source, targets = (FaceSetReader().read(root / name) for name in ("source.fsz", "targets.fsz"))
    view = min(targets.views, key=lambda v: abs(v.yaw-yaw) + abs(v.pitch))
    assert abs(view.yaw-yaw) <= 5 and abs(view.pitch) <= 10, f"Missing real {yaw} degree target"
    landmarks = LandmarkEngine(application_root())
    source = enrich_document(source, landmarks).document
    asset = build_identity_asset(source, IdentityAssetSettings(template="makehuman", uv_size=1024))
    rig = IdentityRig(asset)
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    dense = np.asarray(view.landmarks_68) if view.landmarks_68 is not None else landmarks.landmarks68(image, view.bbox)
    assert dense is not None, "Real target needs measured dense landmarks"
    face = SimpleNamespace(landmarks5=np.asarray(view.landmarks))
    engine = FullHeadEngine(rig)
    try:
        rendered, alpha, inverse = engine.process(image, face, dense, 0., animate=False)
        assert np.all(alpha <= engine.source_silhouette)
        assert alpha.sum() > engine.source_silhouette.sum() * .85
        target = cv2.warpAffine(image, cv2.invertAffineTransform(inverse), (512, 512))
        yy, _ = np.where(alpha > .9)
        for lo, hi in ((0, .2), (.35, .65), (.8, 1.)):
            band = alpha > .9
            band[:int(yy.min() + np.ptp(yy)*lo)] = False
            band[int(yy.min() + np.ptp(yy)*hi)+1:] = False
            assert band.any()
            assert np.mean(np.any(rendered[band] != target[band], axis=1)) > .95
    finally:
        engine.rasterizer.release()
