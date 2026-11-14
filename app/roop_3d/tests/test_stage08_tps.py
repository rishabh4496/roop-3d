"""Acceptance tests for the Stage 08 perspective landmark PnP solver (the 2.5D mesh / TPS / Delaunay warpers were removed in Phase 0)."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    euler_to_rotation_matrix,
    rotation_matrix_to_rodrigues,
)
from roop_3d.reconstruction.tps_fallback import (
    get_canonical_3d_landmarks,
    solve_pose_pnp,
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
