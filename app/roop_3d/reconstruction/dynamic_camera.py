"""Dynamic Camera Intrinsic Focal-Length (FOV) Solver.

Replaces hardcoded camera focal lengths (e.g. f = 1015.0 or fixed 55 FOV) in the
3D Perspective-n-Point (PnP) solver with a dynamic focal-length estimator
(f_est bounded between 18mm wide-angle selfie and 135mm telephoto equivalents).

Solves K_target = [[f_est, 0, cx], [0, f_est, cy], [0, 0, 1]] via Levenberg-Marquardt
reprojection minimization over 3D canonical landmarks so close-up wide-angle selfies
and distant telephoto shots have 100% matching perspective foreshortening.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    Pose3D,
    euler_to_rotation_matrix,
    rodrigues_to_rotation_matrix,
    rotation_matrix_to_euler,
    rotation_matrix_to_rodrigues,
)


SENSOR_WIDTH_35MM: float = 36.0  # Standard 35mm full-frame sensor width in mm
DEFAULT_FOCAL_BOUNDS_MM: tuple[float, float] = (18.0, 135.0)  # 18mm selfie to 135mm telephoto


def mm_to_focal_pixels(f_mm: float, image_dimension: int, sensor_dimension_mm: float = SENSOR_WIDTH_35MM) -> float:
    """Convert 35mm equivalent focal length in millimeters to pixel focal length."""
    if sensor_dimension_mm <= 0:
        raise ValueError("Sensor dimension must be positive")
    return float(f_mm * (image_dimension / sensor_dimension_mm))


def focal_pixels_to_mm(f_px: float, image_dimension: int, sensor_dimension_mm: float = SENSOR_WIDTH_35MM) -> float:
    """Convert pixel focal length to 35mm equivalent focal length in millimeters."""
    if image_dimension <= 0:
        raise ValueError("Image dimension must be positive")
    return float(f_px * (sensor_dimension_mm / image_dimension))


def compute_focal_bounds_pixels(
    width: int,
    height: int,
    focal_bounds_mm: tuple[float, float] = DEFAULT_FOCAL_BOUNDS_MM,
    sensor_width_mm: float = SENSOR_WIDTH_35MM,
) -> tuple[float, float]:
    """Compute (f_min_px, f_max_px) for an image size from 35mm equivalent focal bounds."""
    span = max(width, height)
    min_mm, max_mm = focal_bounds_mm
    f_min_px = mm_to_focal_pixels(min_mm, span, sensor_width_mm)
    f_max_px = mm_to_focal_pixels(max_mm, span, sensor_width_mm)
    return f_min_px, f_max_px


class DynamicCameraSolver:
    """Levenberg-Marquardt dynamic focal length and 3D pose solver."""

    def __init__(
        self,
        focal_bounds_mm: tuple[float, float] = DEFAULT_FOCAL_BOUNDS_MM,
        sensor_width_mm: float = SENSOR_WIDTH_35MM,
        default_fov_degrees: float = 50.0,
    ) -> None:
        self.focal_bounds_mm = focal_bounds_mm
        self.sensor_width_mm = sensor_width_mm
        self.default_fov_degrees = default_fov_degrees

    def solve(
        self,
        landmarks_2d: np.ndarray,
        canonical_3d: np.ndarray,
        width: int | None = None,
        height: int | None = None,
        initial_focal_px: float | None = None,
        principal_point: tuple[float, float] | None = None,
    ) -> tuple[CameraIntrinsics, Pose3D]:
        """Jointly solve focal length f_est and 3D camera pose (R, t) via Levenberg-Marquardt.

        Args:
            landmarks_2d: (N, 2) detected 2D landmarks in image pixels.
            canonical_3d: (N, 3) 3D canonical landmarks in metric meters.
            width: Image width in pixels (inferred from landmarks if None).
            height: Image height in pixels (inferred from landmarks if None).
            initial_focal_px: Initial focal guess, or None for 50mm portrait default.
            principal_point: (cx, cy) or None for image center (width/2, height/2).

        Returns:
            (K_target_intrinsics, pose_3d) with optimal perspective foreshortening.
        """
        pts_2d = np.asarray(landmarks_2d, dtype=np.float64)
        pts_3d = np.asarray(canonical_3d, dtype=np.float64)

        if pts_2d.ndim != 2 or pts_2d.shape[1] != 2:
            raise DomainError("INVALID_LANDMARKS_2D", f"Expected (N, 2) 2D landmarks, got {pts_2d.shape}", 422)
        if pts_3d.ndim != 2 or pts_3d.shape[1] != 3:
            raise DomainError("INVALID_LANDMARKS_3D", f"Expected (N, 3) 3D landmarks, got {pts_3d.shape}", 422)
        if pts_2d.shape[0] != pts_3d.shape[0]:
            raise DomainError(
                "LANDMARK_COUNT_MISMATCH",
                f"2D count {pts_2d.shape[0]} does not match 3D count {pts_3d.shape[0]}",
                422,
            )

        n_pts = pts_2d.shape[0]
        if n_pts < 5:
            raise DomainError("INSUFFICIENT_LANDMARKS", f"Need at least 5 points for PnP fit, got {n_pts}", 422)

        # Incur dimensions
        if width is None or height is None:
            max_x = float(np.max(pts_2d[:, 0]))
            max_y = float(np.max(pts_2d[:, 1]))
            w = int(max(width or 0, max_x * 1.1, 224))
            h = int(max(height or 0, max_y * 1.1, 224))
        else:
            w, h = int(width), int(height)

        if principal_point is None:
            cx = w / 2.0
            cy = h / 2.0
        else:
            cx, cy = float(principal_point[0]), float(principal_point[1])

        # Compute focal bounds in pixels: 18mm to 135mm
        f_min_px, f_max_px = compute_focal_bounds_pixels(w, h, self.focal_bounds_mm, self.sensor_width_mm)

        # Initial focal guess
        if initial_focal_px is not None and np.isfinite(initial_focal_px) and initial_focal_px > 0:
            # If initial focal is legacy 1015 on non-224 or out-of-bounds, clamp gracefully
            f0 = float(np.clip(initial_focal_px, f_min_px, f_max_px))
        else:
            # Default to 50mm portrait equivalent
            f0 = float(mm_to_focal_pixels(50.0, max(w, h), self.sensor_width_mm))
            f0 = float(np.clip(f0, f_min_px, f_max_px))

        # 1. Initial rigid pose using iterative PnP with K(f0)
        k0 = np.array([[f0, 0.0, cx], [0.0, f0, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        success, rvec0, tvec0 = cv2.solvePnP(
            pts_3d,
            pts_2d,
            k0,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success or rvec0 is None or tvec0 is None:
            # Try SQPNP or EPNP fallback
            success, rvec0, tvec0 = cv2.solvePnP(
                pts_3d,
                pts_2d,
                k0,
                None,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not success or rvec0 is None or tvec0 is None:
                raise DomainError("PNP_SOLVER_FAILED", "Initial PnP convergence failed", 422)

        rvec0 = rvec0.reshape(3)
        tvec0 = tvec0.reshape(3)

        # 2. Levenberg-Marquardt joint reprojection error minimization over (rx, ry, rz, tx, ty, tz, f_est)
        def _reprojection_residuals(params: np.ndarray) -> np.ndarray:
            rvec = params[0:3]
            tvec = params[3:6]
            f_val = params[6]
            k_mat = np.array([[f_val, 0.0, cx], [0.0, f_val, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
            projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, k_mat, None)
            projected = projected.reshape(-1, 2)
            residuals = (projected - pts_2d).ravel()
            return residuals

        init_params = np.array([rvec0[0], rvec0[1], rvec0[2], tvec0[0], tvec0[1], tvec0[2], f0], dtype=np.float64)

        # Parameter bounds
        lower_bounds = np.array([-np.pi, -np.pi, -np.pi, -100.0, -100.0, 0.01, f_min_px], dtype=np.float64)
        upper_bounds = np.array([np.pi, np.pi, np.pi, 100.0, 100.0, 100.0, f_max_px], dtype=np.float64)

        # Run Trust Region Reflective (Levenberg-Marquardt with box bounds)
        try:
            res = least_squares(
                _reprojection_residuals,
                init_params,
                bounds=(lower_bounds, upper_bounds),
                method="trf",
                ftol=1e-6,
                xtol=1e-6,
                gtol=1e-6,
                max_nfev=60,
            )
            opt_rvec = res.x[0:3]
            opt_tvec = res.x[3:6]
            f_est = float(np.clip(res.x[6], f_min_px, f_max_px))
        except Exception:
            # Fall back to initial f0 and rvec0, tvec0
            opt_rvec = rvec0
            opt_tvec = tvec0
            f_est = f0

        # Construct final K_target calibration matrix
        k_target = np.array([
            [f_est, 0.0, cx],
            [0.0, f_est, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        # Refine (R, t) with OpenCV Levenberg-Marquardt given K_target
        try:
            refined_rvec, refined_tvec = cv2.solvePnPRefineLM(
                pts_3d,
                pts_2d,
                k_target,
                None,
                opt_rvec.reshape(3, 1),
                opt_tvec.reshape(3, 1),
            )
            if refined_rvec is not None and refined_tvec is not None:
                opt_rvec = refined_rvec.reshape(3)
                opt_tvec = refined_tvec.reshape(3)
        except Exception:
            pass

        rvec_flat = tuple(float(v) for v in opt_rvec)
        tvec_flat = tuple(float(v) for v in opt_tvec)

        r_mat = rodrigues_to_rotation_matrix(np.array(rvec_flat))
        yaw, pitch, roll = rotation_matrix_to_euler(r_mat, degrees=True)

        # Compute mean reprojection error in pixels
        final_proj, _ = cv2.projectPoints(pts_3d, opt_rvec, opt_tvec, k_target, None)
        final_proj = final_proj.reshape(-1, 2)
        reprojection_error = float(np.mean(np.linalg.norm(final_proj - pts_2d, axis=1)))

        # FOV in degrees
        fov_deg = float(math.degrees(2.0 * math.atan(w / (2.0 * f_est))))

        intrinsics = CameraIntrinsics(
            fx=f_est,
            fy=f_est,
            cx=cx,
            cy=cy,
            width=w,
            height=h,
        )

        pose = Pose3D(
            rvec=rvec_flat,
            tvec=tvec_flat,
            yaw=yaw,
            pitch=pitch,
            roll=roll,
            reprojection_error=reprojection_error,
            f_est=f_est,
            fov_degrees=fov_deg,
        )

        return intrinsics, pose


def solve_dynamic_camera_intrinsics(
    landmarks_2d: np.ndarray,
    canonical_3d: np.ndarray | None = None,
    width: int | None = None,
    height: int | None = None,
    focal_bounds_mm: tuple[float, float] = DEFAULT_FOCAL_BOUNDS_MM,
    sensor_width_mm: float = SENSOR_WIDTH_35MM,
    initial_focal_px: float | None = None,
) -> tuple[CameraIntrinsics, Pose3D]:
    """Convenience functional interface for dynamic focal length estimation."""
    from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks

    n_points = landmarks_2d.shape[0]
    if canonical_3d is None:
        canonical_3d = get_canonical_3d_landmarks(n_points)

    solver = DynamicCameraSolver(focal_bounds_mm=focal_bounds_mm, sensor_width_mm=sensor_width_mm)
    return solver.solve(
        landmarks_2d=landmarks_2d,
        canonical_3d=canonical_3d,
        width=width,
        height=height,
        initial_focal_px=initial_focal_px,
    )
