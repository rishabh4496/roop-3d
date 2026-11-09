"""Authoritative 3D geometry contracts, coordinate conventions, and typed representations.

COORDINATE CONVENTIONS & GRAPHICS CONTRACTS
-------------------------------------------
1. Camera Coordinate System (Right-Handed):
   - Primary Computer Vision Frame (OpenCV convention):
     * +X axis points to the RIGHT of the camera image.
     * +Y axis points DOWNWARDS (matching image row indices).
     * +Z axis points FORWARD along the optical axis into the scene.
   - Graphics & Viewport Frame (OpenGL / Three.js convention):
     * +X axis points to the RIGHT.
     * +Y axis points UPWARDS.
     * +Z axis points BACKWARDS towards the viewer (-Z is forward viewing direction).
   - Transformation between frames:
     * P_gl = diag(1, -1, -1) * P_cv
     * P_cv = diag(1, -1, -1) * P_gl

2. Metric Scale:
   - Primary metric scale: meters (1.0 = 1 meter).
   - Facial models (such as FLAME and canonical anthropometric templates) are defined in meters:
     * Standard human inter-pupillary distance: ~0.063 - 0.065 m (63 - 65 mm).
     * Standard head height: ~0.22 - 0.25 m.
     * Standard head width: ~0.14 - 0.16 m.

3. Rotation Representations & Tait-Bryan Angles:
   - Rodrigues Vector r = theta * u in R^3, where theta = ||r|| is angle in radians.
   - Rotation Matrix R in SO(3): 3x3 orthogonal matrix with det(R) = +1, R * R^T = I.
   - Unit Quaternion q = (w, x, y, z): normalized 4-vector where w = cos(theta/2),
     (x, y, z) = u * sin(theta/2), satisfying w^2 + x^2 + y^2 + z^2 = 1.
   - Euler Angles (Yaw, Pitch, Roll):
     * Yaw (psi): rotation around Y axis (horizontal turning, left/right).
     * Pitch (theta): rotation around X axis (nodding, up/down).
     * Roll (phi): rotation around Z axis (tilting side-to-side).
     * Defined with explicit conversion between degrees and radians.

4. Camera Intrinsics (Pinhole Model):
   - Intrinsics matrix K = [[f_x,   0, c_x],
                            [  0, f_y, c_y],
                            [  0,   0,   1]]
   - Projection: (u, v) = (f_x * X / Z + c_x, f_y * Y / Z + c_y).
   - Unprojection: (X, Y, Z) = ((u - c_x) * Z / f_x, (v - c_y) * Z / f_y, Z).

5. Mesh Topology & Winding Order:
   - Triangles: 0-indexed integer triplets (v0, v1, v2).
   - Face Winding: Counter-Clockwise (CCW) when viewed from the outside (front-facing).
   - Normal Vector n = normalized((v1 - v0) x (v2 - v0)).

6. UV Origin Conventions:
   - Standard Texture Space (OBJ / OpenGL / glTF):
     * u in [0, 1] horizontally from left to right.
     * v in [0, 1] vertically from bottom to top (v = 0 is bottom edge, v = 1 is top edge).
   - Image Raster Space (NumPy array indexing):
     * col in [0, W - 1] from left to right.
     * row in [0, H - 1] from top to bottom (row = 0 is top edge).
   - Explicit bidirectional mappings are provided and enforced.
"""
from __future__ import annotations

from enum import Enum, IntEnum
import math
from typing import Any, Literal
import cv2
import numpy as np
from pydantic import Field, field_validator, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

# Metric scale constants
METRIC_SCALE_METERS: float = 1.0
METRIC_SCALE_MILLIMETERS: float = 0.001


class UVOrigin(str, Enum):
    BOTTOM_LEFT = "bottom_left"  # OpenGL / glTF / OBJ convention (v=0 at bottom)
    TOP_LEFT = "top_left"        # Image raster convention (v=0 at top)


class ConfidenceState(IntEnum):
    UNSEEN = 0        # Back-of-head, occluded or out-of-frame; must NEVER be blended
    INTERPOLATED = 1  # Missing observation smoothly filled from adjacent observed regions
    OBSERVED = 2      # Directly visible by camera with positive normal dot product


class CameraIntrinsics(Contract):
    """Pinhole camera intrinsics matrix and image resolution."""
    fx: float = Field(gt=0.0, description="Focal length along X axis in pixels")
    fy: float = Field(gt=0.0, description="Focal length along Y axis in pixels")
    cx: float = Field(ge=0.0, description="Principal point X coordinate in pixels")
    cy: float = Field(ge=0.0, description="Principal point Y coordinate in pixels")
    width: int = Field(gt=0, description="Image width in pixels")
    height: int = Field(gt=0, description="Image height in pixels")

    @classmethod
    def from_fov(cls, fov_degrees: float, width: int, height: int) -> "CameraIntrinsics":
        """Construct intrinsics from horizontal field of view in degrees."""
        if not (1.0 <= fov_degrees <= 170.0):
            raise DomainError("INVALID_FOV", f"Field of view must be in [1.0, 170.0], got {fov_degrees}", 422)
        fov_rad = math.radians(fov_degrees)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx  # Square pixels by default
        cx = width / 2.0
        cy = height / 2.0
        return cls(fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

    def matrix(self) -> np.ndarray:
        """Return 3x3 camera calibration matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def project(self, points_3d: np.ndarray) -> np.ndarray:
        """Project (N, 3) 3D points in camera frame to (N, 2) 2D pixel coordinates."""
        pts = np.asarray(points_3d, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[-1] != 3:
            raise DomainError("INVALID_POINT_DIM", f"Expected (N, 3) points, got {pts.shape}", 422)
        z = np.maximum(pts[:, 2], 1e-6)
        u = self.fx * (pts[:, 0] / z) + self.cx
        v = self.fy * (pts[:, 1] / z) + self.cy
        return np.column_stack((u, v)).astype(np.float32)

    def unproject(self, points_2d: np.ndarray, depths: np.ndarray) -> np.ndarray:
        """Unproject (N, 2) 2D pixel coordinates and (N,) depths to (N, 3) 3D points."""
        pts = np.asarray(points_2d, dtype=np.float64)
        z = np.asarray(depths, dtype=np.float64).reshape(-1)
        if pts.shape[0] != z.shape[0]:
            raise DomainError("DIMENSION_MISMATCH", "Number of 2D points and depths must match", 422)
        x = (pts[:, 0] - self.cx) * z / self.fx
        y = (pts[:, 1] - self.cy) * z / self.fy
        return np.column_stack((x, y, z)).astype(np.float32)


class Pose3D(Contract):
    """Calibrated 3D rigid pose representation."""
    rvec: tuple[float, float, float] = Field(description="Rodrigues rotation vector (axis * angle in radians)")
    tvec: tuple[float, float, float] = Field(description="Translation vector (tx, ty, tz) in meters")
    yaw: float = Field(description="Yaw angle in degrees")
    pitch: float = Field(description="Pitch angle in degrees")
    roll: float = Field(description="Roll angle in degrees")
    reprojection_error: float = Field(ge=0.0, default=0.0, description="Mean reprojection error in pixels")
    f_est: float | None = Field(default=None, description="Estimated dynamic focal length in pixels")
    fov_degrees: float | None = Field(default=None, description="Estimated horizontal field of view in degrees")

    def rotation_matrix(self) -> np.ndarray:
        """Return 3x3 orthogonal rotation matrix."""
        r = np.array(self.rvec, dtype=np.float64)
        return rodrigues_to_rotation_matrix(r)

    def quaternion(self) -> tuple[float, float, float, float]:
        """Return unit quaternion (w, x, y, z)."""
        return rotation_matrix_to_quaternion(self.rotation_matrix())

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform (N, 3) points from model coordinates to camera coordinates: P_cam = R * P + t."""
        pts = np.asarray(points, dtype=np.float64)
        r_mat = self.rotation_matrix()
        t = np.array(self.tvec, dtype=np.float64)
        transformed = (pts @ r_mat.T) + t
        return transformed.astype(np.float32)


class Mesh3D:
    """Validated 3D polygonal mesh with counter-clockwise winding and UV attributes."""

    def __init__(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray | None = None,
        normals: np.ndarray | None = None,
        face_uvs: np.ndarray | None = None,
    ):
        self.vertices = np.asarray(vertices, dtype=np.float32)
        self.faces = np.asarray(faces, dtype=np.int32)
        self.uvs = np.asarray(uvs, dtype=np.float32) if uvs is not None else None
        self.face_uvs = np.asarray(face_uvs, dtype=np.int32) if face_uvs is not None else None

        self.validate()

        if normals is not None:
            self.normals = np.asarray(normals, dtype=np.float32)
        else:
            self.normals = self.compute_vertex_normals()

    def validate(self) -> None:
        """Validate topological integrity, absence of NaNs, and index bounds."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise DomainError("INVALID_VERTICES", f"Vertices must have shape (N, 3), got {self.vertices.shape}", 422)
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise DomainError("INVALID_FACES", f"Faces must have shape (M, 3), got {self.faces.shape}", 422)
        if not np.all(np.isfinite(self.vertices)):
            raise DomainError("NON_FINITE_VERTICES", "Mesh vertices contain NaN or Inf values", 422)
        if len(self.vertices) == 0:
            raise DomainError("EMPTY_MESH", "Mesh must contain at least one vertex", 422)
        if len(self.faces) == 0:
            raise DomainError("EMPTY_FACES", "Mesh must contain at least one face", 422)

        max_idx = np.max(self.faces)
        min_idx = np.min(self.faces)
        if min_idx < 0 or max_idx >= len(self.vertices):
            raise DomainError("FACE_INDEX_OUT_OF_BOUNDS", f"Face index out of bounds [0, {len(self.vertices) - 1}]", 422)

        if self.uvs is not None:
            if self.uvs.ndim != 2 or self.uvs.shape[1] != 2:
                raise DomainError("INVALID_UVS", f"UVs must have shape (K, 2), got {self.uvs.shape}", 422)
            if not np.all(np.isfinite(self.uvs)):
                raise DomainError("NON_FINITE_UVS", "Mesh UVs contain NaN or Inf values", 422)

        if self.face_uvs is not None and self.uvs is not None:
            if self.face_uvs.shape != self.faces.shape:
                raise DomainError("FACE_UV_SHAPE_MISMATCH", "face_uvs shape must match faces shape", 422)
            if np.max(self.face_uvs) >= len(self.uvs) or np.min(self.face_uvs) < 0:
                raise DomainError("FACE_UV_INDEX_OUT_OF_BOUNDS", "Face UV index out of bounds", 422)

    def compute_face_normals(self) -> np.ndarray:
        """Compute area-weighted face normals with counter-clockwise winding."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (cross / norms).astype(np.float32)

    def compute_vertex_normals(self) -> np.ndarray:
        """Compute normalized per-vertex normals by accumulating adjacent face normals."""
        normals = np.zeros_like(self.vertices, dtype=np.float32)
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        face_normals = np.cross(v1 - v0, v2 - v0)

        for i in range(3):
            np.add.at(normals, self.faces[:, i], face_normals)

        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return (normals / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinate Frame Conversion Utilities
# ---------------------------------------------------------------------------

def opencv_to_opengl_camera(points: np.ndarray) -> np.ndarray:
    """Convert points or vectors from OpenCV camera (+X right, +Y down, +Z fwd)
    to OpenGL/Three.js camera (+X right, +Y up, -Z fwd).
    P_gl = [X, -Y, -Z].
    """
    pts = np.asarray(points, dtype=np.float32)
    converted = pts.copy()
    converted[..., 1] = -converted[..., 1]
    converted[..., 2] = -converted[..., 2]
    return converted


def opengl_to_opencv_camera(points: np.ndarray) -> np.ndarray:
    """Convert points or vectors from OpenGL/Three.js camera to OpenCV camera."""
    return opencv_to_opengl_camera(points)  # Involution: diag(1, -1, -1) is its own inverse


# ---------------------------------------------------------------------------
# Rotation Representation Conversions (Rodrigues, Matrix, Quaternion, Euler)
# ---------------------------------------------------------------------------

def rodrigues_to_rotation_matrix(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rotation vector (3,) to 3x3 orthogonal rotation matrix."""
    r = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    r_mat, _ = cv2.Rodrigues(r)
    return r_mat.astype(np.float64)


def rotation_matrix_to_rodrigues(r_mat: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to Rodrigues rotation vector (3,)."""
    r_mat = np.asarray(r_mat, dtype=np.float64)
    rvec, _ = cv2.Rodrigues(r_mat)
    return rvec.reshape(3).astype(np.float64)


def rotation_matrix_to_quaternion(r_mat: np.ndarray) -> tuple[float, float, float, float]:
    """Convert 3x3 rotation matrix to unit quaternion (w, x, y, z) using Shepperd's algorithm."""
    r = np.asarray(r_mat, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]

    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r[2, 1] - r[1, 2]) * s
        y = (r[0, 2] - r[2, 0]) * s
        z = (r[1, 0] - r[0, 1]) * s
    elif (r[0, 0] > r[1, 1]) and (r[0, 0] > r[2, 2]):
        s = 2.0 * math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = 2.0 * math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return float(w / norm), float(x / norm), float(y / norm), float(z / norm)


def quaternion_to_rotation_matrix(q: tuple[float, float, float, float] | np.ndarray) -> np.ndarray:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = (float(v) for v in q)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < 1e-12:
        raise DomainError("DEGENERATE_QUATERNION", "Quaternion has near-zero norm", 422)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)]
    ], dtype=np.float64)


def rotation_matrix_to_euler(r_mat: np.ndarray, degrees: bool = True) -> tuple[float, float, float]:
    """Extract (yaw, pitch, roll) in standard camera frame:
    - Yaw (around Y axis)
    - Pitch (around X axis)
    - Roll (around Z axis)
    """
    r = np.asarray(r_mat, dtype=np.float64)
    # Pitch: rotation around X
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0

    if degrees:
        return math.degrees(yaw), math.degrees(pitch), math.degrees(roll)
    return yaw, pitch, roll


def euler_to_rotation_matrix(yaw: float, pitch: float, roll: float, degrees: bool = True) -> np.ndarray:
    """Construct 3x3 rotation matrix from yaw (Y), pitch (X), roll (Z) angles."""
    if degrees:
        yaw = math.radians(yaw)
        pitch = math.radians(pitch)
        roll = math.radians(roll)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(pitch), -math.sin(pitch)],
        [0.0, math.sin(pitch), math.cos(pitch)]
    ], dtype=np.float64)

    ry = np.array([
        [math.cos(yaw), 0.0, math.sin(yaw)],
        [0.0, 1.0, 0.0],
        [-math.sin(yaw), 0.0, math.cos(yaw)]
    ], dtype=np.float64)

    rz = np.array([
        [math.cos(roll), -math.sin(roll), 0.0],
        [math.sin(roll), math.cos(roll), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    # Combined rotation: R = Rz * Ry * Rx
    return (rz @ ry @ rx).astype(np.float64)


# ---------------------------------------------------------------------------
# UV and Pixel Coordinate Mapping
# ---------------------------------------------------------------------------

def uv_to_pixel(
    uv: np.ndarray,
    width: int,
    height: int,
    origin: UVOrigin = UVOrigin.BOTTOM_LEFT
) -> np.ndarray:
    """Convert (N, 2) UV coordinates in [0, 1] to (N, 2) image pixel coordinates [x, y]."""
    pts = np.asarray(uv, dtype=np.float32)
    u = pts[..., 0]
    v = pts[..., 1]

    x = u * (width - 1.0)
    if origin == UVOrigin.BOTTOM_LEFT:
        y = (1.0 - v) * (height - 1.0)
    else:
        y = v * (height - 1.0)

    return np.column_stack((x, y)).astype(np.float32)


def pixel_to_uv(
    pixel: np.ndarray,
    width: int,
    height: int,
    origin: UVOrigin = UVOrigin.BOTTOM_LEFT
) -> np.ndarray:
    """Convert (N, 2) image pixel coordinates [x, y] to (N, 2) UV coordinates in [0, 1]."""
    pts = np.asarray(pixel, dtype=np.float32)
    x = pts[..., 0]
    y = pts[..., 1]

    u = x / max(width - 1.0, 1.0)
    if origin == UVOrigin.BOTTOM_LEFT:
        v = 1.0 - (y / max(height - 1.0, 1.0))
    else:
        v = y / max(height - 1.0, 1.0)

    return np.column_stack((u, v)).astype(np.float32)


# ---------------------------------------------------------------------------
# Scaled-orthographic (weak-perspective) camera for multi-view identity fitting
# ---------------------------------------------------------------------------

class WeakPerspectiveCamera(Contract):
    """Scaled-orthographic camera in the project's mesh frame (x right, y down, z toward the viewer).

    This is the same projection family as ``core.pose.estimate_pose_5pt``: image = scale * (R @ X)[:2] + t.
    ``depth`` is the rotated z, so a LARGER value is CLOSER to the viewer (the frame is left-handed on
    purpose: it matches stored FaceSet poses and the canonical mesh, whose outward normals point +z).
    """
    rotation: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
    scale: float = Field(gt=0.0, description="Pixels per metre")
    tx: float
    ty: float

    @field_validator("rotation")
    @classmethod
    def _orthonormal(cls, value):
        r = np.asarray(value, dtype=np.float64)
        if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or abs(np.linalg.det(r) - 1.0) > 1e-3:
            raise ValueError("rotation must be a proper orthonormal matrix")
        return value

    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float64)

    def project(self, points: np.ndarray) -> np.ndarray:
        """(N, 3) model points -> (N, 2) pixel coordinates."""
        rotated = np.asarray(points, dtype=np.float64).reshape(-1, 3) @ self.rotation_matrix().T
        return np.column_stack((self.scale * rotated[:, 0] + self.tx, self.scale * rotated[:, 1] + self.ty))

    def depth(self, points: np.ndarray) -> np.ndarray:
        """(N,) toward-the-viewer depth of model points (larger = closer)."""
        return (np.asarray(points, dtype=np.float64).reshape(-1, 3) @ self.rotation_matrix().T)[:, 2]

    def rotate_normals(self, normals: np.ndarray) -> np.ndarray:
        return np.asarray(normals, dtype=np.float64).reshape(-1, 3) @ self.rotation_matrix().T
