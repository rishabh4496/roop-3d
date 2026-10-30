"""Acceptance tests for Stage 08 2.5D landmark PnP solver and TPS/Delaunay warper."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    Mesh3D,
    Pose3D,
    euler_to_rotation_matrix,
    rotation_matrix_to_rodrigues,
)
from roop_3d.reconstruction.tps_fallback import (
    ThinPlateSplineWarper,
    build_canonical_head_mesh,
    get_canonical_3d_landmarks,
    solve_pose_pnp,
    warp_delaunay_piecewise_affine,
)


def test_canonical_landmarks_generation():
    for count in (5, 68, 106, 203):
        pts = get_canonical_3d_landmarks(count)
        assert pts.shape == (count, 3)
        assert pts.dtype == np.float32
        assert np.all(np.isfinite(pts))
        width = np.max(pts[:, 0]) - np.min(pts[:, 0])
        if count == 5:
            # 5-point landmarks span between eye corners (~64mm)
            assert 0.05 <= width <= 0.10
        else:
            # 68/106/203 points span ear-to-ear jaw (~150mm)
            assert 0.10 <= width <= 0.25

    with pytest.raises(DomainError):
        get_canonical_3d_landmarks(42)


def test_solve_pose_pnp_ground_truth_recovery():
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=640, height=480)
    canonical = get_canonical_3d_landmarks(68)

    # Known ground-truth pose
    gt_yaw, gt_pitch, gt_roll = 15.0, -10.0, 5.0
    gt_tvec = np.array([0.05, -0.02, 1.2], dtype=np.float64)
    gt_r_mat = euler_to_rotation_matrix(gt_yaw, gt_pitch, gt_roll, degrees=True)
    gt_rvec = rotation_matrix_to_rodrigues(gt_r_mat)

    # Transform 3D canonical points into camera frame
    transformed_3d = (canonical @ gt_r_mat.T) + gt_tvec
    observed_2d = intrinsics.project(transformed_3d)

    # Solve pose via PnP
    recovered = solve_pose_pnp(observed_2d, intrinsics, canonical_3d=canonical)

    assert recovered.yaw == pytest.approx(gt_yaw, abs=0.5)
    assert recovered.pitch == pytest.approx(gt_pitch, abs=0.5)
    assert recovered.roll == pytest.approx(gt_roll, abs=0.5)
    np.testing.assert_allclose(recovered.tvec, gt_tvec, atol=0.01)
    assert recovered.reprojection_error < 0.1  # Sub-pixel accuracy


def test_build_canonical_head_mesh():
    mesh = build_canonical_head_mesh(landmark_count=68)
    assert isinstance(mesh, Mesh3D)
    assert len(mesh.vertices) >= 68
    assert len(mesh.faces) > 50
    assert mesh.uvs is not None
    assert mesh.uvs.shape == (len(mesh.vertices), 2)
    assert np.all((mesh.uvs >= 0.0) & (mesh.uvs <= 1.0))
    assert mesh.normals.shape == mesh.vertices.shape


def test_tps_warper_identity_and_points():
    src_pts = np.array([
        [10.0, 10.0],
        [50.0, 10.0],
        [50.0, 50.0],
        [10.0, 50.0],
        [30.0, 30.0],
    ], dtype=np.float64)

    # Test identity mapping
    warper_ident = ThinPlateSplineWarper(src_pts, src_pts)
    transformed_ident = warper_ident.transform_points(src_pts)
    np.testing.assert_allclose(transformed_ident, src_pts, atol=1e-3)

    # Test shifting center point
    dst_pts = src_pts.copy()
    dst_pts[4] = [35.0, 25.0]
    warper_shifted = ThinPlateSplineWarper(src_pts, dst_pts)
    transformed_shifted = warper_shifted.transform_points(src_pts)
    np.testing.assert_allclose(transformed_shifted, dst_pts, atol=1e-3)


def test_tps_image_warping():
    # 64x64 synthetic gradient image
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :, 0] = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))
    img[:, :, 1] = np.tile(np.linspace(0, 255, 64, dtype=np.uint8)[:, None], (1, 64))

    src_pts = np.array([
        [0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0], [32.0, 32.0]
    ], dtype=np.float64)
    dst_pts = np.array([
        [0.0, 0.0], [63.0, 0.0], [63.0, 63.0], [0.0, 63.0], [36.0, 28.0]
    ], dtype=np.float64)

    warper = ThinPlateSplineWarper(src_pts, dst_pts)
    warped = warper.warp_image(img, output_shape=(64, 64))

    assert warped.shape == (64, 64, 3)
    assert warped.dtype == np.uint8
    assert not np.all(warped == 0)


def test_delaunay_mesh_warper():
    img = np.ones((60, 60, 3), dtype=np.uint8) * 128
    src = np.array([[10, 10], [50, 10], [30, 50]], dtype=np.float32)
    dst = np.array([[15, 12], [48, 15], [32, 45]], dtype=np.float32)

    warped = warp_delaunay_piecewise_affine(img, src, dst, output_shape=(60, 60))
    assert warped.shape == (60, 60, 3)
