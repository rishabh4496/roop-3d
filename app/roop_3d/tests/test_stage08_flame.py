"""Acceptance tests for Stage 08 FLAME license gating, robust losses, and parametric fitting."""
from __future__ import annotations

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.flame_fitter import (
    FLAMEFitter,
    SyntheticFLAMEModel,
    geman_mcclure_weights,
    huber_weights,
    verify_flame_license,
)
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, Pose3D


def test_flame_license_gate_blocks_unauthorized_activation():
    # FLAME 2020 is blocked in Stage 03 ModelRegistry by default
    with pytest.raises(DomainError) as exc_info:
        verify_flame_license(accepted_license=False)
    assert exc_info.value.code in ("FLAME_MODEL_BLOCKED", "LICENSE_ACCEPTANCE_REQUIRED", "MODEL_BLOCKED")

    # Even if accepted_license is True, blocked status in registry refuses activation
    with pytest.raises(DomainError) as exc_info:
        verify_flame_license(accepted_license=True)
    assert exc_info.value.code in ("FLAME_MODEL_BLOCKED", "MODEL_BLOCKED")


def test_robust_loss_weights_downweight_outliers():
    # Inlier residual: 1.0 pixel
    # Outlier residual: 50.0 pixels
    residuals = np.array([[1.0, 0.0], [50.0, 0.0]], dtype=np.float64)

    huber_w = huber_weights(residuals, delta=2.0)
    assert huber_w[0] == pytest.approx(1.0)
    assert huber_w[1] < 0.1  # Outlier downweighted

    gm_w = geman_mcclure_weights(residuals, sigma=5.0)
    assert gm_w[0] > gm_w[1]
    assert gm_w[1] < 0.01  # Extreme outlier strongly suppressed


def test_synthetic_flame_model_forward():
    model = SyntheticFLAMEModel(num_shape=4, num_expr=3)
    base_v = model.base_mesh.vertices

    # Shape deformation
    shape = np.array([1.5, 0.0, 0.0, 0.0], dtype=np.float32)
    expr = np.zeros(3, dtype=np.float32)
    v_shaped = model.forward(shape, expr)
    assert not np.allclose(v_shaped, base_v)
    assert v_shaped.shape == base_v.shape

    # Expression deformation
    shape_zero = np.zeros(4, dtype=np.float32)
    expr_smile = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    v_expr = model.forward(shape_zero, expr_smile)
    assert not np.allclose(v_expr, base_v)

    # Jaw pose deformation
    jaw = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    v_jaw = model.forward(shape_zero, expr_zero := np.zeros(3, dtype=np.float32), jaw_pose=jaw)
    # Jaw vertices (chin) should move down (+Y in OpenCV)
    chin_idx = 8
    assert v_jaw[chin_idx, 1] > base_v[chin_idx, 1]


def test_flame_fitter_convergence():
    intrinsics = CameraIntrinsics.from_fov(fov_degrees=60.0, width=640, height=480)
    model = SyntheticFLAMEModel(num_shape=4, num_expr=3)

    # Generate synthetic target with known shape and expression deformation
    true_shape = np.array([1.0, -0.5, 0.0, 0.0], dtype=np.float32)
    true_expr = np.array([0.5, 0.0, 0.0], dtype=np.float32)
    v_target = model.forward(true_shape, true_expr)
    lm_3d = model.get_landmarks(v_target)

    # Position at distance z = 1.2 meters
    tvec = np.array([0.0, 0.0, 1.2], dtype=np.float32)
    rvec = np.zeros(3, dtype=np.float32)
    proj_pts = intrinsics.project(lm_3d + tvec)

    # Add an outlier to test robust Huber loss
    proj_pts_noisy = proj_pts.copy()
    proj_pts_noisy[0] += np.array([40.0, -30.0])  # Gross outlier on jaw corner

    fitter = FLAMEFitter(
        model=model,
        loss_type="huber",
        huber_delta=2.0,
        max_iterations=25,
    )

    initial_pose = Pose3D(
        rvec=(0.0, 0.0, 0.0),
        tvec=(0.0, 0.0, 1.2),
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
    )

    result, fitted_mesh = fitter.fit(proj_pts_noisy, intrinsics, initial_pose=initial_pose)

    assert result.iterations > 0
    assert len(result.convergence_history) > 0
    # Final error should be lower or bounded despite outlier
    assert result.final_error_px <= result.initial_error_px + 1.0
    assert len(result.shape_params) == 4
    assert len(result.expr_params) == 3
    assert fitted_mesh.vertices.shape == model.base_mesh.vertices.shape
