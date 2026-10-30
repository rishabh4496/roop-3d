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
    intrinsics: CameraIntrinsics,
    canonical_3d: np.ndarray | None = None,
) -> Pose3D:
    """Solve 3D head pose (R, t) from 2D landmarks using Perspective-n-Point.

    Args:
        landmarks_2d: (N, 2) array of detected 2D pixel coordinates.
        intrinsics: CameraIntrinsics pinhole parameters.
        canonical_3d: Optional (N, 3) canonical landmark points in meters.

    Returns:
        Pose3D with rvec, tvec, yaw, pitch, roll, and reprojection error.
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

    return Pose3D(
        rvec=rvec_flat,
        tvec=tvec_flat,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        reprojection_error=reprojection_error,
    )


# ---------------------------------------------------------------------------
# Canonical Head Mesh for Stage 14 Three.js Viewport
# ---------------------------------------------------------------------------

def build_canonical_head_mesh(landmark_count: int = 68) -> Mesh3D:
    """Create a canonical 3D face mesh using Delaunay triangulation of canonical landmarks.

    Guarantees counter-clockwise winding, valid UV coordinates, and vertex normals
    ready for Stage 14's Three.js viewport.
    """
    pts_3d = get_canonical_3d_landmarks(landmark_count)

    # Add cranial/perimeter boundary points to form a closed convex facial surface
    boundary = np.array([
        [-0.08, -0.07, -0.02],
        [-0.04, -0.09, -0.01],
        [ 0.00, -0.10,  0.00],
        [ 0.04, -0.09, -0.01],
        [ 0.08, -0.07, -0.02],
        [-0.08,  0.02, -0.03],
        [ 0.08,  0.02, -0.03],
        [ 0.00,  0.11,  0.01],
    ], dtype=np.float32)

    all_verts = np.vstack([pts_3d, boundary])

    # Delaunay triangulation on (X, Y) plane
    pts_2d = all_verts[:, :2]
    # Shift and scale to [10, 990] for cv2.Subdiv2D
    min_xy = np.min(pts_2d, axis=0)
    max_xy = np.max(pts_2d, axis=0)
    range_xy = np.maximum(max_xy - min_xy, 1e-6)
    norm_2d = (pts_2d - min_xy) / range_xy
    subdiv_pts = norm_2d * 800.0 + 100.0

    rect = (0, 0, 1000, 1000)
    subdiv = cv2.Subdiv2D(rect)
    for p in subdiv_pts:
        subdiv.insert((float(p[0]), float(p[1])))

    triangle_list = subdiv.getTriangleList()
    faces_list = []

    for t in triangle_list:
        pt1 = np.array([t[0], t[1]])
        pt2 = np.array([t[2], t[3]])
        pt3 = np.array([t[4], t[5]])

        # Find matching vertex indices
        d1 = np.linalg.norm(subdiv_pts - pt1, axis=1)
        d2 = np.linalg.norm(subdiv_pts - pt2, axis=1)
        d3 = np.linalg.norm(subdiv_pts - pt3, axis=1)

        idx1, idx2, idx3 = int(np.argmin(d1)), int(np.argmin(d2)), int(np.argmin(d3))
        if d1[idx1] < 1.0 and d2[idx2] < 1.0 and d3[idx3] < 1.0:
            if idx1 != idx2 and idx2 != idx3 and idx3 != idx1:
                # Check winding order in (X, Y): cross product of (v2 - v1) and (v3 - v1)
                v1_2d, v2_2d, v3_2d = pts_2d[idx1], pts_2d[idx2], pts_2d[idx3]
                cross_z = (v2_2d[0] - v1_2d[0]) * (v3_2d[1] - v1_2d[1]) - (v2_2d[1] - v1_2d[1]) * (v3_2d[0] - v1_2d[0])
                if cross_z < 0:
                    idx2, idx3 = idx3, idx2
                faces_list.append([idx1, idx2, idx3])

    if not faces_list:
        raise DomainError("TRIANGULATION_FAILED", "Failed to build Delaunay surface mesh", 500)

    faces = np.array(faces_list, dtype=np.int32)

    # UV coordinates normalized to [0, 1] in standard texture space (u: left to right, v: bottom to top)
    u = (all_verts[:, 0] - min_xy[0]) / range_xy[0]
    # Invert Y for standard bottom-left UV convention
    v = 1.0 - ((all_verts[:, 1] - min_xy[1]) / range_xy[1])
    uvs = np.column_stack([np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0)]).astype(np.float32)

    return Mesh3D(vertices=all_verts, faces=faces, uvs=uvs)


# ---------------------------------------------------------------------------
# 2.5D Thin-Plate-Spline (TPS) Warper
# ---------------------------------------------------------------------------

class ThinPlateSplineWarper:
    """Dependency-free 2D Thin-Plate-Spline (TPS) warper minimizing bending energy."""

    def __init__(
        self,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        regularization: float = 1e-4,
    ):
        self.src_points = np.asarray(src_points, dtype=np.float64)
        self.dst_points = np.asarray(dst_points, dtype=np.float64)
        self.reg = float(regularization)

        if self.src_points.shape != self.dst_points.shape:
            raise DomainError("TPS_POINT_MISMATCH", "Source and destination points must have identical shapes", 422)
        if self.src_points.ndim != 2 or self.src_points.shape[1] != 2:
            raise DomainError("TPS_INVALID_SHAPE", f"Expected (N, 2) points, got {self.src_points.shape}", 422)
        if len(self.src_points) < 4:
            raise DomainError("TPS_TOO_FEW_POINTS", "TPS requires at least 4 non-collinear point pairs", 422)

        self._fit()

    @staticmethod
    def _radial_basis(r: np.ndarray) -> np.ndarray:
        """TPS kernel U(r) = r^2 * ln(r) for r > 0, 0 for r == 0."""
        r2 = r * r
        result = np.zeros_like(r2)
        mask = r > 1e-8
        result[mask] = r2[mask] * np.log(r[mask])
        return result

    def _fit(self) -> None:
        """Solve TPS linear system:
        [[K + lambda*I, P],
         [P^T,          0]] * [[W], [A]] = [[Y], [0]]
        """
        n = len(self.src_points)
        diff = self.src_points[:, None, :] - self.src_points[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        k_mat = self._radial_basis(dist) + self.reg * np.eye(n, dtype=np.float64)

        p_mat = np.column_stack([np.ones(n, dtype=np.float64), self.src_points])

        l_mat = np.zeros((n + 3, n + 3), dtype=np.float64)
        l_mat[:n, :n] = k_mat
        l_mat[:n, n:] = p_mat
        l_mat[n:, :n] = p_mat.T

        y_mat = np.zeros((n + 3, 2), dtype=np.float64)
        y_mat[:n] = self.dst_points

        try:
            # Solve system
            params = np.linalg.solve(l_mat, y_mat)
        except np.linalg.LinAlgError:
            params, _, _, _ = np.linalg.lstsq(l_mat, y_mat, rcond=None)

        self.w = params[:n]   # (N, 2)
        self.a = params[n:]   # (3, 2) [a0, ax, ay]

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Apply TPS deformation to arbitrary (M, 2) points."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise DomainError("TPS_TRANSFORM_SHAPE", f"Expected (M, 2) points, got {pts.shape}", 422)

        diff = pts[:, None, :] - self.src_points[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        k_eval = self._radial_basis(dist)  # (M, N)

        p_eval = np.column_stack([np.ones(len(pts), dtype=np.float64), pts])  # (M, 3)

        warped = (k_eval @ self.w) + (p_eval @ self.a)
        return warped.astype(np.float32)

    def warp_image(
        self,
        image: np.ndarray,
        output_shape: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Warp image using backward TPS mapping with bilinear interpolation.

        Backward mapping fits dst_points -> src_points to sample source image.
        """
        h_src, w_src = image.shape[:2]
        h_out, w_out = output_shape if output_shape is not None else (h_src, w_src)

        # Build inverse warper (dst -> src)
        inv_warper = ThinPlateSplineWarper(
            src_points=self.dst_points,
            dst_points=self.src_points,
            regularization=self.reg,
        )

        grid_y, grid_x = np.indices((h_out, w_out), dtype=np.float32)
        flat_grid = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        mapped_src = inv_warper.transform_points(flat_grid)
        map_x = mapped_src[:, 0].reshape(h_out, w_out).astype(np.float32)
        map_y = mapped_src[:, 1].reshape(h_out, w_out).astype(np.float32)

        return cv2.remap(
            image,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )


# ---------------------------------------------------------------------------
# 2.5D Delaunay Mesh Warper
# ---------------------------------------------------------------------------

def warp_delaunay_piecewise_affine(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    triangles: np.ndarray | None = None,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Piecewise affine warping over Delaunay triangles."""
    h_src, w_src = image.shape[:2]
    h_out, w_out = output_shape if output_shape is not None else (h_src, w_src)

    src_pts = np.asarray(src_points, dtype=np.float32)
    dst_pts = np.asarray(dst_points, dtype=np.float32)

    if triangles is None:
        # Build Delaunay triangulation on dst_points
        rect = (0, 0, w_out, h_out)
        subdiv = cv2.Subdiv2D(rect)
        for p in dst_pts:
            subdiv.insert((float(p[0]), float(p[1])))

        tri_list = subdiv.getTriangleList()
        tri_indices = []
        for t in tri_list:
            p1, p2, p3 = np.array([t[0], t[1]]), np.array([t[2], t[3]]), np.array([t[4], t[5]])
            i1 = int(np.argmin(np.linalg.norm(dst_pts - p1, axis=1)))
            i2 = int(np.argmin(np.linalg.norm(dst_pts - p2, axis=1)))
            i3 = int(np.argmin(np.linalg.norm(dst_pts - p3, axis=1)))
            if i1 != i2 and i2 != i3 and i3 != i1:
                tri_indices.append([i1, i2, i3])
        triangles = np.array(tri_indices, dtype=np.int32)

    output = np.zeros((h_out, w_out, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)

    for tri in triangles:
        t_src = src_pts[tri]
        t_dst = dst_pts[tri]

        # Bounding box of destination triangle
        r_dst = cv2.boundingRect(t_dst)
        x_dst, y_dst, w_dst, h_dst = r_dst
        if w_dst <= 0 or h_dst <= 0:
            continue

        # Offset points by top-left corner of bounding box
        t_dst_local = t_dst - np.array([x_dst, y_dst], dtype=np.float32)
        affine_mat = cv2.getAffineTransform(t_src.astype(np.float32), t_dst_local)

        # Warp source image onto local bounding patch
        patch = cv2.warpAffine(
            image,
            affine_mat,
            (w_dst, h_dst),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Triangle mask for blending patch
        mask = np.zeros((h_dst, w_dst), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(t_dst_local), 255)

        # Place patch into output
        y_slice = slice(y_dst, min(y_dst + h_dst, h_out))
        x_slice = slice(x_dst, min(x_dst + w_dst, w_out))
        patch_y_slice = slice(0, y_slice.stop - y_slice.start)
        patch_x_slice = slice(0, x_slice.stop - x_slice.start)

        mask_slice = mask[patch_y_slice, patch_x_slice]
        out_region = output[y_slice, x_slice]
        patch_region = patch[patch_y_slice, patch_x_slice]

        if output.ndim == 3:
            mask_3d = mask_slice[..., None] > 0
            np.copyto(out_region, patch_region, where=mask_3d)
        else:
            np.copyto(out_region, patch_region, where=mask_slice > 0)

    return output.squeeze() if image.ndim == 2 else output
