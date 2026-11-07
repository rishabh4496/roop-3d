"""Stage 3 Phase C: identity texture atlas. Atlas texels are compared with the KNOWN synthetic ground-truth texture."""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.reconstruction.geometry_types import ConfidenceState, Mesh3D, WeakPerspectiveCamera  # noqa: E402
from roop_3d.reconstruction.identity_asset import IdentityAssetSettings, build_identity_asset  # noqa: E402
from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation, remap_points  # noqa: E402

OBS, INTERP, UNSEEN = (int(ConfidenceState.OBSERVED), int(ConfidenceState.INTERPOLATED), int(ConfidenceState.UNSEEN))
SETTINGS = IdentityAssetSettings(uv_size=W.ATLAS)


@pytest.fixture(scope="module")
def subject():
    return W.make_subject("tex", W.LAYOUT_68, 61)


def _mae(asset, subject, mask=None):
    sel = (asset.texel_state == OBS) if mask is None else mask & (asset.texel_state == OBS)
    assert sel.any()
    return float(np.abs(asset.texture[sel].astype(np.float32) - subject.texture[sel].astype(np.float32)).mean())


def _atlas_uv(asset):
    return np.column_stack([asset.uvs[:, 0], 1.0 - asset.uvs[:, 1]])


def _gt_normal_z(asset, yaw, pitch=0.0, roll=0.0):
    """Ground-truth-camera facing of every texel (z of the rotated surface normal), from the asset's own geometry."""
    synth = UVSynthesizer(Mesh3D(asset.vertices, asset.faces, uvs=_atlas_uv(asset)), uv_size=W.ATLAS)
    from roop_3d.core.pose import rotation_from_euler
    rot = rotation_from_euler(yaw, pitch, roll)
    return synth.norm_map @ rot.T[:, 2], synth.face_texel_mask


@pytest.fixture(scope="module")
def three_view(subject):
    return build_identity_asset(W.full_set(subject, (0, -40, 40)), SETTINGS)


def test_atlas_reproduces_the_ground_truth_texture(three_view, subject):
    assert three_view.manifest.statistics["texel_observed_ratio"] > 0.95
    assert _mae(three_view, subject) < 22.0
    assert float(three_view.texel_confidence[three_view.texel_state == OBS].mean()) > 0.6
    # every observed texel knows which source view dominates it, and how many contributed
    obs = three_view.texel_state == OBS
    assert (three_view.texel_view_count[obs] >= 1).all() and set(np.unique(three_view.texel_source_view[obs])) <= {0, 1, 2}


def test_unseen_regions_stay_empty_and_are_never_labelled_observed(subject):
    """One strongly turned view: the far side of the face was never seen. It must be UNSEEN with zero colour."""
    asset = build_identity_asset(W.make_faceset("one_profile", [W.make_view(subject, 0, 62.0)]), SETTINGS)
    normal_z, footprint = _gt_normal_z(asset, 62.0)
    hidden = footprint & (normal_z < -0.05)                       # surface facing away from the real camera
    assert hidden.sum() > 500
    assert not (asset.texel_state[hidden] == OBS).any(), "a texel facing away from the camera cannot be OBSERVED"
    unseen = asset.texel_state == UNSEEN
    assert (asset.texture[unseen] == 0).all() and (asset.texel_confidence[unseen] == 0).all()
    assert asset.manifest.statistics["texel_unseen_ratio"] > 0.15
    # interpolation never reaches far from real observations
    dist = cv2.distanceTransform((asset.texel_state != OBS).astype(np.uint8), cv2.DIST_L2, 3)
    interp = asset.texel_state == INTERP
    assert (dist[interp] <= SETTINGS.max_interpolation_fraction * W.ATLAS + 1.0).all()
    assert (asset.texel_confidence[interp] <= 0.31).all()
    assert asset.manifest.claims["texture_from_observations_only"] and not asset.manifest.claims["back_of_head_recovered"]


def test_self_occlusion_stops_the_nose_painting_the_cheek(subject, monkeypatch):
    doc = W.make_faceset("occl", [W.make_view(subject, 0, 55.0)])
    with_zbuffer = build_identity_asset(doc, SETTINGS)
    monkeypatch.setattr(UVSynthesizer, "_visible_by_zbuffer",
                        lambda self, cam, depth, xy: (np.ones(len(depth), bool), np.ones(len(depth))))
    without = build_identity_asset(doc, SETTINGS)
    assert _mae(with_zbuffer, subject) + 1.0 < _mae(without, subject)


def test_gain_compensation_handles_different_lighting(subject, monkeypatch):
    lights = (1.0, 0.55, 1.15)      # a 1.35x view would clip >35% of its pixels and rightly be excluded
    doc = W.make_faceset("lighting", [W.make_view(subject, i, y, light=l, cast=c)
                                      for i, (y, l, c) in enumerate(zip((0, -30, 30), lights, ((1, 1, 1), (0.9, 1.0, 1.1), (1.1, 1.0, 0.9))))])
    compensated = build_identity_asset(doc, SETTINGS)
    gains = compensated.manifest.statistics["view_gains"]
    mean_gain = {int(k): float(np.mean(v)) for k, v in gains.items()}
    assert mean_gain[1] > mean_gain[0] > mean_gain[2]            # the dark view is brightened, the bright one darkened
    monkeypatch.setattr(UVSynthesizer, "_solve_gains", staticmethod(lambda colors, weights: np.ones((colors.shape[0], 3))))
    naive = build_identity_asset(doc, SETTINGS)

    def seam(asset):  # std-dev of the brightness error: lower = consistent exposure across the atlas
        sel = asset.texel_state == OBS
        diff = asset.texture[sel].astype(np.float32).mean(axis=1) - subject.texture[sel].astype(np.float32).mean(axis=1)
        return float(diff.std())
    assert seam(compensated) < seam(naive)


def test_partial_occlusion_is_rejected_by_consensus_and_by_mask(subject):
    views = [W.make_view(subject, 0, 0.0, occlude_eyes=True), W.make_view(subject, 1, -22.0), W.make_view(subject, 2, 22.0),
             W.make_view(subject, 3, 9.0)]
    doc = W.make_faceset("occluded", views)
    eyes = np.zeros((W.ATLAS, W.ATLAS), bool)
    probe = build_identity_asset(doc, SETTINGS)
    uv = (np.column_stack([probe.uvs[:, 0], 1.0 - probe.uvs[:, 1]]) * (W.ATLAS - 1)).round().astype(int)
    for u, v in uv[36:48]:
        eyes[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = True
    consensus = probe
    assert consensus.manifest.statistics["rejected_photometric_samples"] > 0
    assert _mae(consensus, subject, eyes) < 25.0, "black occluder leaked into the eye texels"
    # supplying the occluder mask removes the occluded samples before blending, so nothing needs rejecting
    mask = np.full((W.IMG_H, W.IMG_W), 255, np.uint8)
    pts = np.asarray(views[0].landmarks_68, np.float64)[36:48]
    x0, y0 = (pts.min(axis=0) - 14).astype(int)
    x1, y1 = (pts.max(axis=0) + 14).astype(int)
    mask[y0:y1, x0:x1] = 0
    masked = build_identity_asset(doc, SETTINGS, occlusion_masks={0: mask})
    assert _mae(masked, subject, eyes) < 25.0
    assert masked.manifest.build_key != consensus.manifest.build_key      # masks are part of the content address


def test_interpolation_is_bounded_and_labelled(subject):
    asset = build_identity_asset(W.make_faceset("front", [W.make_view(subject, 0, 0.0)]), SETTINGS)
    assert asset.manifest.statistics["texel_interpolated_ratio"] < 0.15
    unseen_face = asset.texel_state == UNSEEN
    assert (asset.texture[unseen_face] == 0).all()


def test_remap_points_handles_more_than_32767_samples():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    xs = np.random.default_rng(0).uniform(0, 99, 70000)
    ys = np.random.default_rng(1).uniform(0, 99, 70000)
    out = remap_points(img, xs, ys, cv2.INTER_NEAREST)
    assert out.shape == (70000, 3)
    ref = img[np.round(ys).astype(int), np.round(xs).astype(int)]
    assert (np.abs(out.astype(int) - ref.astype(int)).max()) <= 90          # nearest-neighbour rounding differences only


def test_observation_and_camera_validation():
    img = np.zeros((10, 10, 3), np.uint8)
    with pytest.raises(ValueError):
        ViewObservation(image=img)                               # neither a camera nor pose + intrinsics
    with pytest.raises(ValueError):
        WeakPerspectiveCamera(rotation=((2, 0, 0), (0, 1, 0), (0, 0, 1)), scale=100.0, tx=0.0, ty=0.0)
    cam = WeakPerspectiveCamera(rotation=((1, 0, 0), (0, 1, 0), (0, 0, 1)), scale=100.0, tx=5.0, ty=6.0)
    assert np.allclose(cam.project(np.array([[0.1, 0.2, 0.3]])), [[15.0, 26.0]]) and cam.depth(np.array([[0, 0, 0.3]]))[0] == pytest.approx(0.3)
    mesh = W.identity_template(1) if hasattr(W, "identity_template") else None
    synth = UVSynthesizer(mesh, uv_size=64)
    other = ViewObservation(image=img, camera=cam)
    legacy = ViewObservation(image=img, pose=__import__("roop_3d.reconstruction.geometry_types", fromlist=["Pose3D"]).Pose3D(
        rvec=(0, 0, 0), tvec=(0, 0, 1), yaw=0, pitch=0, roll=0),
        intrinsics=__import__("roop_3d.reconstruction.geometry_types", fromlist=["CameraIntrinsics"]).CameraIntrinsics.from_fov(60, 10, 10))
    with pytest.raises(DomainError) as info:
        synth.synthesize([other, legacy])
    assert info.value.code == "MIXED_CAMERA_MODELS"


# ----------------------------------------------------------------------------- real-photo regressions (found on real FaceSets in Stage 4)
def test_grazing_surfaces_are_never_observed(subject):
    """Texture seen more than ~75 deg off the surface normal is stretched beyond use: it must not count as observed."""
    asset = build_identity_asset(W.make_faceset("grazing", [W.make_view(subject, 0, 78.0)]), SETTINGS)
    normal_z, _ = _gt_normal_z(asset, 78.0)
    observed = asset.texel_state == OBS
    assert observed.any() and float(normal_z[observed].min()) > 0.2


def test_loosely_registered_views_are_not_averaged_into_ghosts(subject, monkeypatch):
    base = [W.make_view(subject, i, y) for i, y in enumerate((0, -25, 25))]
    # the 5-point landmarks of two views are off by a few pixels, as they are on real photographs
    shifted = [base[0]] + [v.model_copy(update={"landmarks_68": tuple((x + 3.5, y - 3.0) for x, y in v.landmarks_68),
                                                "landmarks": tuple((x + 3.5, y - 3.0) for x, y in v.landmarks)}) for v in base[1:]]
    doc = W.make_faceset("ghost", shifted)
    new = build_identity_asset(doc, SETTINGS)
    monkeypatch.setattr(UVSynthesizer, "DOMINANCE_FRACTION", 0.0)
    monkeypatch.setattr(UVSynthesizer, "WEIGHT_POWER", 2.0)
    old = build_identity_asset(doc, SETTINGS)
    # error against the KNOWN texture: smearing misregistered views is strictly worse than following the best one
    assert _mae(new, subject) + 0.3 < _mae(old, subject)


def test_near_profile_views_are_down_weighted_but_still_count_where_nothing_else_sees(subject):
    views = [W.make_view(subject, i, y) for i, y in enumerate((0, -40, 40, 86))]
    asset = build_identity_asset(W.make_faceset("profile_weight", views), SETTINGS)
    observed = asset.texel_state == OBS
    share = float((asset.texel_source_view[observed] == 3).mean())
    assert share < 0.05, f"a poorly registered 86 deg view dominates {share:.0%} of the atlas"
    from roop_3d.reconstruction.identity_asset import ALGORITHM_VERSION
    assert asset.manifest.algorithm_version == ALGORITHM_VERSION
