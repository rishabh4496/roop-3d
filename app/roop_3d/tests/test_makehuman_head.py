"""MakeHuman (CC0) head model: pinned sources, asset integrity, frame conventions and multi-view fitting."""
from __future__ import annotations

import json
import re
import shutil

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction import makehuman_build as mb
from roop_3d.reconstruction.makehuman_head import (
    MakeHumanHeadModel,
    ViewObservation,
    fit_multiview,
    intrinsics_for,
    _pseudo_huber,
)

ASSET = mb.DEFAULT_ASSET_DIR / mb.ASSET_NAME
needs_asset = pytest.mark.skipif(not ASSET.exists(), reason="run `python -m roop_3d.reconstruction.makehuman_build` first")


@pytest.fixture(scope="module")
def model() -> MakeHumanHeadModel:
    return MakeHumanHeadModel.load()


def test_lock_file_pins_every_upstream_file_and_declares_cc0():
    lock = mb.load_lock()
    assert lock["license"] == "CC0-1.0"
    assert lock["commit"] == mb.COMMIT and re.fullmatch(r"[0-9a-f]{40}", mb.COMMIT)
    assert set(lock["files"]) == set(mb._all_paths())
    assert all(re.fullmatch(r"[0-9a-f]{64}", f["sha256"]) and f["bytes"] > 0 for f in lock["files"].values())


def test_only_the_pinned_upstream_repository_is_downloadable(monkeypatch):
    monkeypatch.setattr(mb, "RAW_BASE", "https://evil.example/makehuman/")
    with pytest.raises(DomainError) as err:
        mb._download("3dobjs/base.obj")
    assert err.value.code == "MAKEHUMAN_SOURCE_FORBIDDEN"


def test_target_parser_ignores_comments_and_out_of_range_rows():
    idx, delta = mb.parse_target(b"# header\n\n3 0.1 0.2 0.3\n99999 1 1 1\n7 -1 0 0\n", n_vertices=100)
    assert idx.tolist() == [3, 7] and delta.tolist() == [[0.1, 0.2, 0.3], [-1.0, 0.0, 0.0]]


def test_frame_conversion_is_a_proper_rotation_in_metres():
    p = mb._to_head_frame(np.array([[1.0, 2.0, 3.0]]))
    assert np.allclose(p, [[0.1, -0.2, -0.3]])
    assert np.linalg.det(np.diag([1.0, -1.0, -1.0])) == pytest.approx(1.0)


def test_pseudo_huber_is_linear_near_zero_and_sublinear_for_outliers():
    r = np.array([0.01, 5.0, -5.0])
    out = _pseudo_huber(r, 1.0)
    assert out[0] == pytest.approx(0.01, rel=1e-2)
    assert abs(out[1]) < 5.0 and out[1] == -out[2]


def test_missing_asset_gives_actionable_error(tmp_path):
    with pytest.raises(DomainError) as err:
        MakeHumanHeadModel.load(tmp_path / "nope.npz")
    assert err.value.code == "MAKEHUMAN_ASSET_MISSING"


@needs_asset
def test_asset_shape_and_frame_conventions(model):
    v, li = model.base_mesh.vertices, model.landmark_indices
    assert np.all(np.isfinite(v)) and model.shape_basis.shape == (len(v), 3, 50) and model.expr_basis.shape[2] == 10
    right_eye, left_eye, nose, mouth_r, mouth_l = v[li]
    assert right_eye[0] < 0 < left_eye[0] and mouth_r[0] < 0 < mouth_l[0]     # detector order: image-left (character right) first
    assert right_eye[1] < mouth_r[1]                                            # y down: eyes above the mouth
    assert nose[2] < min(right_eye[2], left_eye[2])                             # nose points at the camera at R = I
    assert 0.050 < np.linalg.norm(right_eye - left_eye) < 0.070                 # human inter-ocular distance
    assert 0.040 < np.linalg.norm(mouth_r - mouth_l) < 0.060
    assert 0.15 < np.ptp(v[:, 0]) < 0.20                                         # head width in metres
    assert model.base_mesh.uvs is not None and model.base_mesh.face_uvs.shape == model.base_mesh.faces.shape


@needs_asset
def test_frontal_surface_winding_faces_the_camera(model):
    m = model.base_mesh
    centroid = m.vertices[m.faces].mean(axis=1)
    front = centroid[:, 2] < (m.vertices[:, 2].min() + 0.03)                     # the ~3 cm nearest the camera
    v0, v1, v2 = (m.vertices[m.faces[:, i]] for i in range(3))
    cross = np.cross(v1 - v0, v2 - v0)
    area = np.linalg.norm(cross, axis=1)
    toward_camera = cross[:, 2] < 0
    # A flipped mesh would score ~0.2; concave regions (nostrils, eye sockets, lips) keep the correct one below 1.
    assert front.sum() > 50 and area[front & toward_camera].sum() / area[front].sum() > 0.75


@needs_asset
def test_forward_equals_landmark_shortcut_and_expression_moves_the_mouth(model):
    rng = np.random.default_rng(3)
    beta, psi = rng.normal(0, 0.5, model.num_shape), rng.normal(0, 0.3, model.num_expr)
    full = model.forward(beta, psi)[model.landmark_indices]
    assert np.allclose(full, model.landmarks_for(beta, psi), atol=1e-6)
    neutral = model.landmarks_for(np.zeros(model.num_shape))
    opened = model.landmarks_for(np.zeros(model.num_shape), np.eye(model.num_expr)[model.expr_names.index("mouth_open")])
    assert np.linalg.norm(opened[3:] - neutral[3:], axis=1).max() > 0.002          # mouth corners shift when it opens


@needs_asset
def test_corrupt_asset_is_rejected(tmp_path):
    copy = tmp_path / ASSET.name
    shutil.copy(ASSET, copy)
    shutil.copy(ASSET.with_name(ASSET.name + ".json"), tmp_path / (ASSET.name + ".json"))
    with np.load(copy, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["vertices"] = arrays["vertices"] + np.float32(0.001)
    np.savez_compressed(copy, **arrays)
    with pytest.raises(DomainError) as err:
        MakeHumanHeadModel.load(copy)
    assert err.value.code == "MAKEHUMAN_ASSET_CORRUPT"


@needs_asset
def test_tampered_source_file_fails_hash_check(tmp_path):
    src = mb.DEFAULT_ASSET_DIR / "sources"
    if not (src / mb.BASE_OBJ).exists():
        pytest.skip("source cache not present")
    shutil.copytree(src, tmp_path / "s")
    target = tmp_path / "s" / mb._all_paths()[3]
    target.write_bytes(target.read_bytes() + b"\n# tampered\n")
    with pytest.raises(DomainError) as err:
        mb.fetch_sources(tmp_path / "s")
    assert err.value.code == "MAKEHUMAN_HASH_MISMATCH"


def _synthetic_views(model, beta, yaws, size=512, fov=45.0):
    views, truth = [], []
    for yaw in yaws:
        obs = ViewObservation(np.zeros((5, 2)), size, size)
        k = intrinsics_for(obs, fov).matrix()
        rvec = np.array([0.03, np.radians(yaw), 0.02])
        tvec = np.array([0.01, -0.005, 0.45])
        pts, _ = cv2.projectPoints(model.landmarks_for(beta), rvec.reshape(3, 1), tvec.reshape(3, 1), k, None)
        obs.landmarks = pts.reshape(5, 2)
        obs.name = f"yaw{yaw}"
        views.append(obs)
        truth.append(yaw)
    return views, truth


@needs_asset
def test_multiview_fit_recovers_poses_and_predicts_held_out_views(model):
    beta = np.zeros(model.num_shape)
    beta[model.shape_names.index("mouth_scale_horiz")] = 0.6
    beta[model.shape_names.index("nose_scale_depth")] = -0.5
    beta[model.shape_names.index("head_scale_horiz")] = 0.4
    yaws = [-60.0, -25.0, 0.0, 20.0, 55.0]
    views, truth = _synthetic_views(model, beta, yaws)
    result = fit_multiview(model, views)
    assert result.converged
    for r, yaw in zip(result.views, truth):
        assert r.rms_px < 0.6, (r.name, r.rms_px)
        assert abs(r.pose.yaw - yaw) < 2.0, (r.name, r.pose.yaw)
        assert r.holdout_rms_px < 1.5, (r.name, r.holdout_rms_px)
        assert r.rms_px <= r.neutral_rms_px + 1e-6                                 # fitting identity never hurts


@needs_asset
def test_multiview_fit_rejects_bad_input(model):
    with pytest.raises(DomainError):
        fit_multiview(model, [])
    with pytest.raises(DomainError):
        fit_multiview(model, [ViewObservation(np.zeros((68, 2)), 512, 512)])


@needs_asset
def test_robust_to_one_wild_landmark(model):
    views, _ = _synthetic_views(model, np.zeros(model.num_shape), [-30.0, 0.0, 30.0, 10.0])
    views[1].landmarks = views[1].landmarks.copy()
    views[1].landmarks[2] += np.array([40.0, -35.0])                                # nose detector failure
    result = fit_multiview(model, views)
    by_name = {r.name: r for r in result.views}
    assert by_name["yaw0.0"].rejected and not any(r.rejected for n, r in by_name.items() if n != "yaw0.0")
    clean = [r for n, r in by_name.items() if n != "yaw0.0"]
    assert max(r.rms_px for r in clean) < 1.0
    for name, yaw in (("yaw-30.0", -30.0), ("yaw30.0", 30.0), ("yaw10.0", 10.0)):
        assert abs(by_name[name].pose.yaw - yaw) < 3.0, (name, by_name[name].pose.yaw)


def test_lock_json_is_valid_json():
    json.loads(mb.LOCK_PATH.read_text("utf-8"))
