"""G1: dense-landmark PnP recovers extreme poses; flip resolution; observation wiring; occlusion at the same poses."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import rotation_from_euler
from roop_3d.engine.detection.dense_landmarks import POSE_SOURCE_GEOMETRIC, observation_pose
from roop_3d.mesh_raycaster import MeshRaycaster
from roop_3d.reconstruction import pose_bins
from roop_3d.reconstruction.pose_solver import (
    _FLIP, default_intrinsics_matrix, rotation_angle_deg, rotation_from_standard, solve_head_pose_6dof)
from roop_3d.reconstruction.tps_fallback import build_canonical_head_mesh, get_canonical_3d_landmarks

W, H = 1280, 960
K = default_intrinsics_matrix(W, H)
CANON = get_canonical_3d_landmarks(68).astype(np.float64)


def project(model_repo, yaw, pitch, roll, t=(0.03, -0.02, 0.6)):
    rotation = rotation_from_standard(rotation_from_euler(yaw, pitch, roll))
    pts = cv2.projectPoints(model_repo @ _FLIP, cv2.Rodrigues(rotation)[0], np.array(t, np.float64).reshape(3, 1), K, None)
    return pts[0].reshape(-1, 2), rotation


@pytest.mark.parametrize("yaw", [-90, -75, -60, 60, 75, 90])
@pytest.mark.parametrize("pitch", [-35, 35])
def test_pnp_recovers_extreme_pose_with_half_pixel_noise(yaw, pitch):
    rng = np.random.default_rng(1000 + yaw * 7 + pitch)
    pts, rotation = project(CANON, yaw, pitch, 8.0)
    pose = solve_head_pose_6dof(pts + rng.normal(0, 0.5, pts.shape), image_size=(W, H))
    assert abs(pose.yaw_deg - yaw) <= 1.5
    assert abs(pose.pitch_deg - pitch) <= 2.0 and abs(pose.roll_deg - 8.0) <= 2.0
    assert pose.source == "pnp_dense" and pose.inliers == 68 and pose.reproj_rms_px < 1.2
    assert rotation_angle_deg(pose.R, rotation) < 2.5
    assert pose.t[2] == pytest.approx(0.6, rel=0.02)                      # metric depth recovered
    assert pose.quat[0] >= 0 and np.linalg.norm(pose.quat) == pytest.approx(1.0)
    assert np.allclose(pose.R @ pose.R.T, np.eye(3), atol=1e-6) and np.linalg.det(pose.R) == pytest.approx(1.0)
    assert pose.spherical_bin == pose_bins.classify(yaw, pitch).name


def test_outlier_landmarks_are_rejected_by_ransac():
    rng = np.random.default_rng(5)
    pts, _ = project(CANON, 40.0, 10.0, 0.0)
    noisy = pts + rng.normal(0, 0.4, pts.shape)
    noisy[[3, 20, 40, 55, 61]] += 80.0                                   # five gross detector failures
    pose = solve_head_pose_6dof(noisy, image_size=(W, H))
    assert pose.inliers <= 63 and abs(pose.yaw_deg - 40.0) < 1.5 and abs(pose.pitch_deg - 10.0) < 1.5


def test_flip_resolution_on_a_flat_model_uses_the_temporal_prior():
    """A planar head model viewed from far away has two PnP solutions (mirrored depth). The mirrored hypothesis is
    evaluated, and the previous rotation selects the right one at a near-profile yaw."""
    flat = CANON.copy()
    flat[:, 2] = 0.0
    yaw, pitch = 70.0, 20.0
    wrong = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        pts, _ = project(flat, yaw, pitch, 0.0, t=(0, 0, 3.0))
        noisy = pts + rng.normal(0, 0.3, pts.shape)
        prior = rotation_from_standard(rotation_from_euler(yaw + 3, pitch - 2, 1.0))
        pose = solve_head_pose_6dof(noisy, image_size=(W, H), model_points=flat, prev_rotation=prior)
        assert pose.flip_resolved, "the mirrored-depth hypothesis must have been evaluated and found distinct"
        wrong += abs(pose.yaw_deg - yaw) > 3.0 or abs(pose.pitch_deg - pitch) > 3.0
    assert wrong == 0


def test_telemetry_matches_the_pose_6dof_contract():
    pts, _ = project(CANON, 45.0, 0.0, 0.0)
    telemetry = solve_head_pose_6dof(pts, image_size=(W, H)).to_telemetry()
    assert set(telemetry) == {"yaw_deg", "pitch_deg", "roll_deg", "tx_m", "ty_m", "tz_m", "quat", "reproj_rms_px",
                              "inliers", "source", "spherical_bin"}
    assert telemetry["spherical_bin"] == "yaw+45/pitch0" and telemetry["source"] == "pnp_dense"
    assert len(telemetry["quat"]) == 4


def test_unusable_landmarks_raise_named_errors():
    with pytest.raises(DomainError) as count:
        solve_head_pose_6dof(np.zeros((30, 2)), image_size=(W, H))
    assert count.value.code == "POSE_LANDMARK_COUNT"
    with pytest.raises(DomainError) as nan:
        solve_head_pose_6dof(np.full((68, 2), np.nan), image_size=(W, H))
    assert nan.value.code == "POSE_LANDMARKS_INVALID"


def test_observation_pose_uses_pnp_when_dense_landmarks_exist_and_reports_fallbacks():
    pts, _ = project(CANON, 75.0, -20.0, 0.0)
    five = np.vstack([pts[36:42].mean(0), pts[42:48].mean(0), pts[30], pts[48], pts[54]])
    dense = observation_pose(five, landmarks68=pts, image_size=(W, H))
    assert dense.source == "pnp_dense" and dense.fallback_reason is None and dense.pose6dof is not None
    yaw, legacy_pitch, _ = dense.pose
    assert abs(yaw - 75.0) < 1.5 and abs(legacy_pitch - 20.0) < 1.5          # legacy frame: pitch positive = down
    missing = observation_pose(five)
    assert missing.source == POSE_SOURCE_GEOMETRIC and missing.fallback_reason == "no_dense_landmarks" and missing.pose6dof is None
    broken = observation_pose(five, landmarks68=np.full((68, 2), 5.0), image_size=(W, H))
    assert broken.source == POSE_SOURCE_GEOMETRIC and broken.fallback_reason.startswith("pnp_")


@pytest.mark.parametrize("yaw", [-75, 75, -90, 90])
def test_visibility_excludes_the_occluded_cheek_at_the_solved_pose(yaw):
    pts, _ = project(CANON, yaw, 0.0, 0.0)
    pose = solve_head_pose_6dof(pts, image_size=(W, H))
    mesh = build_canonical_head_mesh(68)
    visible = MeshRaycaster().compute_visibility_map(mesh, pose, resolution=(H, W)).vertex_visible
    far_cheek, near_cheek = (15, 1) if yaw > 0 else (1, 15)
    assert not visible[far_cheek] and visible[near_cheek]
