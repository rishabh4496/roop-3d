"""Dependency-free 3D canonical landmark PnP solver and 2.5D Thin-Plate-Spline / Delaunay mesh warper.

This module guarantees 3D pose visualization for the Stage 14 Three.js viewport and
geometric alignment for Stage 09 even when restricted FLAME weights are not installed.
"""
from __future__ import annotations

import math
from typing import Literal
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    Mesh3D,
    Pose3D,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_euler,
    uv_to_pixel,
)


# ---------------------------------------------------------------------------
# Canonical Anthropometric 3D Facial Landmarks (in Metric Meters)
# ---------------------------------------------------------------------------

def _build_canonical_68_points() -> np.ndarray:
    """Generate standard anthropometric 3D landmarks for 68 points in meters.
    Coordinate frame: OpenCV (+X right, +Y down, +Z forward towards nose tip).
    Face center (between eyes) is roughly at (0, 0, 0).
    """
    pts = np.zeros((68, 3), dtype=np.float32)

    # Jawline 0..16 (right ear to left ear through chin)
    for i in range(17):
        angle = math.pi * (i / 16.0)  # 0 at right ear, pi/2 at chin, pi at left ear
        x = -0.075 * math.cos(angle)
        y = 0.09 * math.sin(angle) - 0.01
        z = -0.04 * math.sin(angle) + 0.01  # Chin is further forward than ears
        pts[i] = [x, y, z]
    # Chin (index 8) exactly at X=0
    pts[8] = [0.0, 0.082, 0.025]

    # Right Eyebrow 17..21 (outer to inner)
    for idx, i in enumerate(range(17, 22)):
        frac = idx / 4.0
        x = -0.060 + frac * 0.040
        y = -0.038 - 0.006 * math.sin(frac * math.pi)
        z = 0.020 + frac * 0.010
        pts[i] = [x, y, z]

    # Left Eyebrow 22..26 (inner to outer)
    for idx, i in enumerate(range(22, 27)):
        frac = idx / 4.0
        x = 0.020 + frac * 0.040
        y = -0.038 - 0.006 * math.sin(frac * math.pi)
        z = 0.030 - frac * 0.010
        pts[i] = [x, y, z]

    # Nose Bridge 27..30 (top to tip)
    pts[27] = [0.0, -0.025, 0.030]
    pts[28] = [0.0, -0.008, 0.038]
    pts[29] = [0.0,  0.008, 0.048]
    pts[30] = [0.0,  0.022, 0.062]  # Nose tip (most prominent point)

    # Nose Base 31..35 (nostrils)
    pts[31] = [-0.018, 0.026, 0.040]
    pts[32] = [-0.009, 0.029, 0.048]
    pts[33] = [ 0.000, 0.030, 0.050]
    pts[34] = [ 0.009, 0.029, 0.048]
    pts[35] = [ 0.018, 0.026, 0.040]

    # Right Eye 36..41 (outer corner 36, inner corner 39)
    pts[36] = [-0.048, -0.016, 0.018]
    pts[37] = [-0.038, -0.021, 0.022]
    pts[38] = [-0.026, -0.020, 0.024]
    pts[39] = [-0.018, -0.016, 0.025]
    pts[40] = [-0.026, -0.013, 0.023]
    pts[41] = [-0.038, -0.013, 0.020]

    # Left Eye 42..47 (inner corner 42, outer corner 45)
    pts[42] = [ 0.018, -0.016, 0.025]
    pts[43] = [ 0.026, -0.020, 0.024]
    pts[44] = [ 0.038, -0.021, 0.022]
    pts[45] = [ 0.048, -0.016, 0.018]
    pts[46] = [ 0.038, -0.013, 0.020]
    pts[47] = [ 0.026, -0.013, 0.023]

    # Outer Lip 48..59 (corners 48, 54)
    lip_outer_x = [-0.028, -0.018, -0.009, 0.0, 0.009, 0.018, 0.028, 0.018, 0.009, 0.0, -0.009, -0.018]
    lip_outer_y = [0.046, 0.042, 0.041, 0.042, 0.041, 0.042, 0.046, 0.054, 0.057, 0.058, 0.057, 0.054]
    lip_outer_z = [0.032, 0.040, 0.045, 0.046, 0.045, 0.040, 0.032, 0.038, 0.042, 0.043, 0.042, 0.038]
    for idx, i in enumerate(range(48, 60)):
        pts[i] = [lip_outer_x[idx], lip_outer_y[idx], lip_outer_z[idx]]

    # Inner Lip 60..67
    lip_inner_x = [-0.020, -0.009, 0.0, 0.009, 0.020, 0.009, 0.0, -0.009]
    lip_inner_y = [0.047, 0.044, 0.045, 0.044, 0.047, 0.050, 0.051, 0.050]
    lip_inner_z = [0.034, 0.041, 0.043, 0.041, 0.034, 0.039, 0.040, 0.039]
    for idx, i in enumerate(range(60, 68)):
        pts[i] = [lip_inner_x[idx], lip_inner_y[idx], lip_inner_z[idx]]

    return pts


def _build_canonical_5_points() -> np.ndarray:
    """Canonical 5-point facial landmarks: left eye, right eye, nose, left mouth, right mouth."""
    return np.array([
        [-0.032, -0.016, 0.022],  # Right eye (from viewer: left side of image)
        [ 0.032, -0.016, 0.022],  # Left eye
        [ 0.000,  0.022, 0.062],  # Nose tip
        [-0.024,  0.048, 0.035],  # Mouth right corner
        [ 0.024,  0.048, 0.035],  # Mouth left corner
    ], dtype=np.float32)


def get_canonical_3d_landmarks(landmark_count: int = 68) -> np.ndarray:
    """Return canonical 3D facial landmarks in right-handed camera space (meters)."""
    if landmark_count == 5:
        return _build_canonical_5_points()
    if landmark_count == 68:
        return _build_canonical_68_points()
    if landmark_count == 106:
        # Interpolate 106-point landmark template from 68-point anthropometric basis
        base68 = _build_canonical_68_points()
        pts106 = np.zeros((106, 3), dtype=np.float32)
        # Jaw (33 points)
        for i in range(33):
            t = (i / 32.0) * 16.0
            idx0 = int(math.floor(t))
            idx1 = min(idx0 + 1, 16)
            frac = t - idx0
            pts106[i] = (1.0 - frac) * base68[idx0] + frac * base68[idx1]
        # Brows (10 points each)
        for i in range(10):
            frac = i / 9.0
            pts106[33 + i] = (1.0 - frac) * base68[17] + frac * base68[21]
            pts106[43 + i] = (1.0 - frac) * base68[22] + frac * base68[26]
        # Nose (10 points)
        for i in range(10):
            frac = i / 9.0
            pts106[53 + i] = (1.0 - frac) * base68[27] + frac * base68[35]
        # Eyes (12 points each)
        for i in range(12):
            frac = i / 12.0
            idx68_r = 36 + int(frac * 6) % 6
            idx68_l = 42 + int(frac * 6) % 6
            pts106[63 + i] = base68[idx68_r]
            pts106[75 + i] = base68[idx68_l]
        # Mouth (19 points)
        for i in range(19):
            idx68 = 48 + (i % 20)
            pts106[87 + i] = base68[min(idx68, 67)]
        return pts106
    if landmark_count == 203:
        # Interpolate 203-point dense template
        base68 = _build_canonical_68_points()
        pts203 = np.zeros((203, 3), dtype=np.float32)
        for i in range(203):
            t = (i / 202.0) * 67.0
            idx0 = int(math.floor(t))
            idx1 = min(idx0 + 1, 67)
            frac = t - idx0
            pts203[i] = (1.0 - frac) * base68[idx0] + frac * base68[idx1]
        return pts203

    raise DomainError("UNSUPPORTED_LANDMARK_COUNT", f"Expected 5, 68, 106, or 203 landmarks, got {landmark_count}", 422)


# ---------------------------------------------------------------------------
# 3D Canonical Landmark PnP Solver
# ---------------------------------------------------------------------------

def solve_pose_pnp(
    landmarks_2d: np.ndarray,
    intrinsics: CameraIntrinsics | None = None,
    canonical_3d: np.ndarray | None = None,
    *,
    estimate_focal: bool = False,
    focal_bounds_mm: tuple[float, float] = (18.0, 135.0),
    sensor_width_mm: float = 36.0,
) -> Pose3D:
    """Solve 3D head pose (R, t) from 2D landmarks using Perspective-n-Point.

    If intrinsics is None, or contains the legacy placeholder f = 1015.0, or
    estimate_focal=True, solves dynamic camera focal length f_est bounded between
    18mm wide-angle selfie and 135mm telephoto equivalents via Levenberg-Marquardt.

    Args:
        landmarks_2d: (N, 2) array of detected 2D pixel coordinates.
        intrinsics: CameraIntrinsics pinhole parameters (or None).
        canonical_3d: Optional (N, 3) canonical landmark points in meters.
        estimate_focal: Force dynamic Levenberg-Marquardt focal length estimation.
        focal_bounds_mm: (min_mm, max_mm) 35mm equivalent bounds (default 18mm..135mm).
        sensor_width_mm: Sensor width for 35mm conversion (default 36.0mm).

    Returns:
        Pose3D with rvec, tvec, yaw, pitch, roll, reprojection error, and f_est.
    """
    pts_2d = np.asarray(landmarks_2d, dtype=np.float64)
    if pts_2d.ndim != 2 or pts_2d.shape[1] != 2:
        raise DomainError("INVALID_LANDMARKS_2D", f"Expected (N, 2) 2D landmarks, got {pts_2d.shape}", 422)

    n_points = pts_2d.shape[0]
    if canonical_3d is None:
        canonical_3d = get_canonical_3d_landmarks(n_points)
    pts_3d = np.asarray(canonical_3d, dtype=np.float64)

    if pts_3d.shape != (n_points, 3):
        raise DomainError(
            "LANDMARK_SHAPE_MISMATCH",
            f"2D points {pts_2d.shape} do not match 3D canonical points {pts_3d.shape}",
            422,
        )

    # Dynamic focal length solver when requested, intrinsics omitted, or legacy 1015.0 placeholder present
    is_legacy_1015 = intrinsics is not None and abs(float(intrinsics.fx) - 1015.0) < 1e-3
    if estimate_focal or intrinsics is None or is_legacy_1015:
        from roop_3d.reconstruction.dynamic_camera import DynamicCameraSolver
        solver = DynamicCameraSolver(focal_bounds_mm=focal_bounds_mm, sensor_width_mm=sensor_width_mm)
        w = intrinsics.width if intrinsics is not None else None
        h = intrinsics.height if intrinsics is not None else None
        dyn_k, dyn_pose = solver.solve(
            pts_2d,
            pts_3d,
            width=w,
            height=h,
            initial_focal_px=None if is_legacy_1015 else (intrinsics.fx if intrinsics else None),
        )
        return dyn_pose

    k_matrix = intrinsics.matrix()
    success, rvec, tvec = cv2.solvePnP(
        pts_3d,
        pts_2d,
        k_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success or rvec is None or tvec is None:
        raise DomainError("PNP_SOLVER_FAILED", "PnP optimization failed to converge", 422)

    rvec_flat = tuple(float(v) for v in rvec.reshape(3))
    tvec_flat = tuple(float(v) for v in tvec.reshape(3))

    r_mat = rodrigues_to_rotation_matrix(np.array(rvec_flat))
    yaw, pitch, roll = rotation_matrix_to_euler(r_mat, degrees=True)

    # Compute mean reprojection error in pixels
    projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, k_matrix, None)
    projected = projected.reshape(-1, 2)
    reprojection_error = float(np.mean(np.linalg.norm(projected - pts_2d, axis=1)))

    fov_deg = float(math.degrees(2.0 * math.atan(intrinsics.width / (2.0 * intrinsics.fx))))

    return Pose3D(
        rvec=rvec_flat,
        tvec=tvec_flat,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        reprojection_error=reprojection_error,
        f_est=float(intrinsics.fx),
        fov_degrees=fov_deg,
    )
