"""Clean-room regression coverage for the Stage 18 expression, UV, and atlas additions."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.core.face_editor import FaceEditor
from roop_3d.reconstruction.arkit_bridge import ARKIT52_NAMES, ARKit52To3DMMBridge
from roop_3d.reconstruction.shape_keys_plus import AsymmetricShapeKeyManager, ShapeKeyMode
from roop_3d.tracking.video_tracker import CameraPose, FrameEvidence, TrackerSettings, VideoHeadTracker3D
from roop_3d.rendering.uv_baker import BakedPBRAtlas, atlas_manifest, bake_pbr_atlas
from roop_3d.rendering.uv_gaussian_map import ContinuousUVLoD, UVGaussianMicroDetailMap
from roop_3d.reconstruction.uv_synthesizer import ConfidenceState


def test_arkit_bridge_has_52_named_controls_and_100d_bidirectional_contract():
    bridge = ARKit52To3DMMBridge()
    assert len(ARKIT52_NAMES) == len(set(ARKIT52_NAMES)) == 52
    values = np.zeros(52, np.float32)
    values[ARKIT52_NAMES.index("jawOpen")] = 0.8
    values[ARKIT52_NAMES.index("eyeLookUpLeft")] = 0.5
    mapped = bridge.to_3dmm(values)
    assert mapped.psi.shape == (100,)
    assert mapped.theta_jaw.shape == (3,) and mapped.theta_jaw[0] == pytest.approx(0.48)
    assert mapped.theta_eyes.shape == (6,) and mapped.gaze_angles.shape == (4,)
    assert mapped.gaze_angles[1] == pytest.approx(0.15)
    assert bridge.to_arkit(mapped.psi).shape == (52,)
    assert len(bridge.slider_definitions()) == 52


def test_arkit_bridge_fits_bidirectional_linear_and_quadratic_residuals():
    rng = np.random.default_rng(1801)
    x = rng.uniform(0, 1, (64, 52)).astype(np.float32)
    psi = np.zeros((64, 100), np.float32)
    psi[:, :52] = x
    psi[:, 52] = x[:, 0] * x[:, 1]
    jaw = np.column_stack((x[:, 24] * 0.6, x[:, 23] * 0.15, x[:, 22] * 0.1)).astype(np.float32)
    bridge = ARKit52To3DMMBridge.fit(x, psi, jaw_samples=jaw, ridge=1e-6)
    converted = bridge.to_3dmm(x[:3])
    restored = bridge.to_arkit(converted.psi)
    assert converted.psi.shape == (3, 100) and converted.theta_jaw.shape == (3, 3)
    assert np.max(np.abs(restored - x[:3])) < 0.01
    assert np.max(np.abs(converted.psi[:, 52] - x[:3, 0] * x[:3, 1])) < 0.01
    assert bridge.regularize(np.zeros(100, np.float32), x[0], strength=0.4) == pytest.approx(
        0.4 * bridge.to_3dmm(x[0]).psi, abs=1e-5)


def test_video_tracker_uses_arkit_bridge_for_filtered_psi_jaw_and_gaze():
    controls = np.zeros(52, np.float32)
    controls[ARKIT52_NAMES.index("jawOpen")] = 0.7
    controls[ARKIT52_NAMES.index("eyeLookOutLeft")] = 0.4
    evidence = FrameEvidence(timestamp=1.0, landmarks5=None, arkit52=controls)
    tracker = VideoHeadTracker3D(TrackerSettings(lighting_enabled=False))
    mapped = tracker.arkit_bridge.to_3dmm(controls)
    psi = tracker._psi(evidence, 1.0, mapped)
    jaw, source = tracker._jaw(evidence, None, 1.0, mapped)
    camera = CameraPose(0, 0, 0, 1, 0, 0, np.eye(3))
    gaze = tracker._gaze(evidence, camera, 1.0, mapped)
    assert psi.shape == (100,)
    assert jaw == pytest.approx(0.42) and source == "arkit52_bridge"
    assert gaze is not None and gaze.source == "arkit52_bridge"
    assert gaze.angles[0] == pytest.approx(0.14)


def test_face_editor_validates_sparse_arkit_slider_values():
    editor = FaceEditor(arkit_blendshapes={"mouthPucker": 0.75, "eyeBlinkLeft": 1.0})
    assert editor.active()
    with pytest.raises(ValueError, match="Unknown ARKit"):
        FaceEditor(arkit_blendshapes={"notAControl": 0.5})
    with pytest.raises(ValueError, match=r"in \[0,1\]"):
        FaceEditor(arkit_blendshapes={"jawOpen": 1.5})


def test_uv_detail_map_is_nchw_12_channel_and_resizes_with_face_lod():
    plane = np.ones((20, 24), np.float32)
    detail = UVGaussianMicroDetailMap.from_channels({"offset_z": plane, "pore": plane * 2,
                                                       "hair": plane * 0.5}, size=180)
    assert detail.shape == (1, 12, 180, 180)
    assert detail.tensor[0, 2].mean() == pytest.approx(1.0)
    assert detail.tensor[0, 9].max() == 1.0
    lod = ContinuousUVLoD()
    for width, expected in ((80, 64), (200, 180), (420, 320)):
        selected, state = lod.rasterize(detail, (0, 0, width, width))
        assert state.raster_resolution == expected
        assert selected.shape == (1, 12, expected, expected)
    intermediate = ContinuousUVLoD().select(140, 140)
    assert 64 < intermediate.raster_resolution < 180 and 0 < intermediate.blend < 1
    assert ContinuousUVLoD().resolution_for_bbox((80, 80)) == 64
    with pytest.raises(ValueError, match=r"shape \[B, 12"):
        UVGaussianMicroDetailMap(np.zeros((1, 11, 64, 64), np.float32))


def test_asymmetric_shape_key_masks_modes_and_solo_selection():
    vertices = np.array([[-1, 0, 0], [-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0], [1, 0, 0]], np.float32)
    manager = AsymmetricShapeKeyManager(vertices, midline_x=0, blend_width=0.2)
    left, right = manager.left_weights, manager.right_weights
    assert left + right == pytest.approx(np.ones(5))
    assert left[2] == pytest.approx(0.5) and right[2] == pytest.approx(0.5)
    basis = np.ones((5, 3, 60), np.float32)
    beta = np.zeros(60, np.float32); beta[2] = 1
    symmetric = manager.deform(basis, beta)
    left_only = manager.deform(basis, beta, mode=ShapeKeyMode.LEFT_ONLY)
    right_only = manager.deform(basis, beta, mode="Right Only")
    solo = manager.deform(basis, np.ones(60, np.float32), mode="Solo Active Shape Key", active_index=2, active_value=0.5)
    assert symmetric - vertices == pytest.approx(np.ones((5, 3)))
    assert left_only - vertices == pytest.approx(left[:, None] * np.ones((5, 3)))
    assert right_only - vertices == pytest.approx(right[:, None] * np.ones((5, 3)))
    assert solo - vertices == pytest.approx(np.ones((5, 3)) * 0.5)
    assert np.isfinite(manager.deform(np.zeros((5, 3, 100), np.float32), np.zeros(100), mode="Mirror Flip")).all()


def _small_asset():
    color = np.full((16, 16, 3), (60, 100, 180), np.uint8)
    state = np.full((16, 16), int(ConfidenceState.OBSERVED), np.uint8)
    state[:, 8:12] = int(ConfidenceState.INTERPOLATED)
    state[:, 12:] = int(ConfidenceState.UNSEEN)
    relief = np.zeros((16, 16), np.float32); relief[7:9, 7:9] = 0.001
    return SimpleNamespace(texture=color, texel_state=state, displacement_uv_m=relief)


def test_pbr_atlas_preserves_observed_albedo_and_emits_trust_normal_orm_and_cavity_maps():
    baked = bake_pbr_atlas(_small_asset(), resolution=64)
    assert baked.basecolor_bgra.shape == (64, 64, 4)
    assert baked.basecolor_bgra[8, 8, 3] == 255
    assert 0 < baked.basecolor_bgra[8, 40, 3] < 255
    assert baked.basecolor_bgra[8, 55, 3] == 0
    assert baked.relief_available and set(baked.encoded()) == {"basecolor", "normal", "orm", "cavity"}
    orm = baked.orm_bgra
    assert np.all(orm[..., 0] == 0)  # metallic B channel for dielectric skin
    assert np.all(orm[..., 1] > 0) and np.all(orm[..., 2] > 0)
    manifest = atlas_manifest(baked, baked.encoded(), faceset_id="faceset_a", build_key="a" * 64,
                              source_sha256="b" * 64)
    assert manifest["albedo"]["diffuse_is_unbaked"] is True
    assert manifest["claims"]["25_percent_render_speedup_benchmarked"] is False


def test_pbr_baker_derives_pore_relief_from_observed_high_frequency_albedo():
    checker = (np.indices((32, 32)).sum(axis=0) % 2 * 24 + 112).astype(np.uint8)
    asset = SimpleNamespace(texture=np.repeat(checker[..., None], 3, axis=2),
                            texel_state=np.full((32, 32), int(ConfidenceState.OBSERVED), np.uint8),
                            displacement_uv_m=np.zeros((0, 0), np.float32))
    baked = bake_pbr_atlas(asset, resolution=64)
    assert baked.relief_available
    assert np.any(baked.normal_bgra[..., 1] != 128) or np.any(baked.normal_bgra[..., 2] != 128)


def test_bake_atlas_api_is_registered_and_writes_to_the_content_addressed_faceset_asset_dir(tmp_path, monkeypatch):
    from roop_3d.server import identity_assets

    asset = _small_asset()

    class FakeStore:
        app_root = tmp_path

        def get_or_build(self, document, settings):
            return asset, True

        def _dir(self, faceset_id, build_key):
            return tmp_path / "roop_3d" / ".runtime" / "identity_assets" / faceset_id / build_key[:24]

    tiny = np.zeros((8, 8, 4), np.uint8)
    output = BakedPBRAtlas(2048, tiny, tiny, tiny, np.zeros((8, 8), np.uint8), np.ones((8, 8), np.float32),
                           (16, 16), 0.001, True)
    monkeypatch.setattr(identity_assets, "_store", FakeStore)
    monkeypatch.setattr(identity_assets, "_settings", lambda template="canonical": object())
    import roop_3d.reconstruction.identity_asset as identity_asset
    monkeypatch.setattr(identity_asset, "compute_build_key", lambda document, settings: "c" * 64)
    import roop_3d.rendering.uv_baker as uv_baker
    monkeypatch.setattr(uv_baker, "bake_pbr_atlas", lambda asset, resolution: output)
    document = SimpleNamespace(faceset_id="local_face", source_sha256="d" * 64)
    result = identity_assets._bake_atlas(document, 2048)
    assert result["asset_reused"] and set(result["maps"]) == {"basecolor", "normal", "orm", "cavity"}
    directory = tmp_path / "roop_3d" / ".runtime" / "identity_assets" / "local_face" / ("c" * 24)
    assert (directory / "atlas_2k_basecolor.png").is_file()
    assert (directory / "atlas_2k_manifest.json").is_file()
    registered = next(route for route in identity_assets.router.routes if route.path.endswith("/bake-atlas"))
    assert "POST" in registered.methods
