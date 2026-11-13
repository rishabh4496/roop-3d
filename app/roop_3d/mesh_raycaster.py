"""Vectorised depth-buffered self-occlusion of a head mesh (blueprint 03 §3.3).

Not a BVH/Embree ray caster: it is a CPU z-buffer. Vertices are projected with ``u = K (R X + t)``; every front-facing
triangle is sampled on a barycentric lattice dense enough (<= ~0.7 px spacing) to leave no pixel gaps and its
perspective-correct depth is splatted into a float buffer with ``np.minimum.at`` (nearest z wins). A vertex is *visible*
iff ``z_v <= z_buffer(u_v) + eps`` (default 8 mm) **and** its outward normal makes less than 80 degrees with the direction
to the camera. A soft image mask is the filled union of triangles whose three vertices are visible, feathered by 3 px.

Frames: mesh vertices/winding use the repo mesh frame (x right, y down, z toward the viewer, counter-clockwise faces with
outward normals, e.g. ``Mesh3D`` canonical/MakeHuman heads). The pose is a ``HeadPose6DoF`` (OpenCV head frame, exact), or
a dict/``yaw_override`` in the standard frame of ``core.pose`` (yaw + = nose to image right, pitch + = up). The z-buffer
covers the image plus one frame of margin on each side so vertices just outside the frame are still depth-tested;
vertices beyond that window are judged by the normal test only and flagged ``vertex_depth_tested = False``.
Limit: very large triangles (> ~45 px edges) are sampled at coarser than 0.7 px and may leave isolated depth holes.
"""
from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import CameraIntrinsics, Mesh3D
from roop_3d.reconstruction.pose_solver import HeadPose6DoF, rotation_from_standard
from roop_3d.core.pose import rotation_from_euler

# Register compatibility alias so legacy module paths map to roop_3d
sys.modules.setdefault("roop.mesh_raycaster", sys.modules[__name__])

EPSILON_M = 0.008
MAX_GRAZING_DEG = 80.0
FEATHER_PX = 3
DEFAULT_DISTANCE_M = 0.8
_SAMPLE_SPACING_PX = 0.7
_MAX_LATTICE = 64
_FLIP = np.diag([1.0, 1.0, -1.0])


@dataclass(frozen=True)
class VisibilityResult:
    """Per-vertex visibility, soft image mask and depth buffer of one pose."""
    vertex_visible: np.ndarray          # (N,) bool
    vertex_depth_tested: np.ndarray     # (N,) bool; False -> judged by the normal test only (outside the depth window)
    visibility_map: np.ndarray          # (H, W) float32 in [0, 1]: feathered fill of visible triangles
    z_buffer: np.ndarray                # (H, W) float32 camera depth in metres, +inf where nothing projects
    self_occluded_fraction: float       # fraction of vertices not visible
    yaw_degrees: float                  # standard-frame yaw of the evaluated pose
    execution_time_ms: float

    @property
    def occluded_indices(self) -> np.ndarray:
        return np.flatnonzero(~self.vertex_visible)

    def to_telemetry(self) -> dict[str, float]:
        """The ``visibility`` object of the pipeline.status frame (blueprint 03 §4)."""
        return {"self_occluded_fraction": round(self.self_occluded_fraction, 4)}


def _unpack_mesh(mesh: Any) -> Mesh3D:
    if isinstance(mesh, Mesh3D):
        return mesh
    if isinstance(mesh, tuple) and len(mesh) == 2:
        return Mesh3D(np.asarray(mesh[0], np.float32), np.asarray(mesh[1], np.int32))
    raise DomainError("VISIBILITY_FACES_REQUIRED",
                      "Self-occlusion needs a triangle mesh (Mesh3D or (vertices, faces)); bare vertices cannot occlude", 422)


def _resolve_pose(pose: Any, yaw_override: float | None) -> tuple[np.ndarray, np.ndarray, float]:
    """(R in the OpenCV head frame, t, standard yaw)."""
    t = np.array([0.0, 0.0, DEFAULT_DISTANCE_M])
    if isinstance(pose, HeadPose6DoF):
        rotation, t, yaw = pose.R, pose.t, pose.yaw_deg
    else:
        values = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
        if isinstance(pose, dict):
            values.update({k: float(pose[k]) for k in values if k in pose})
            t = np.array([float(pose.get("tx", 0.0)), float(pose.get("ty", 0.0)), float(pose.get("tz", DEFAULT_DISTANCE_M))])
        elif pose is not None:
            raise DomainError("VISIBILITY_POSE_INVALID", "pose must be HeadPose6DoF, a yaw/pitch/roll dict or None", 422)
        if yaw_override is not None:
            values["yaw"] = float(yaw_override)
        rotation = rotation_from_standard(rotation_from_euler(values["yaw"], values["pitch"], values["roll"]))
        yaw = values["yaw"]
    if yaw_override is not None and isinstance(pose, HeadPose6DoF):
        raise DomainError("VISIBILITY_POSE_INVALID", "yaw_override cannot be combined with a HeadPose6DoF", 422)
    if t[2] <= 0:
        raise DomainError("VISIBILITY_POSE_INVALID", "Head translation must have t_z > 0", 422)
    return np.asarray(rotation, np.float64), np.asarray(t, np.float64), float(yaw)


class MeshRaycaster:
    """Depth-buffered per-vertex self-occlusion for a posed head mesh."""

    def __init__(self, default_resolution: tuple[int, int] = (256, 256), epsilon_m: float = EPSILON_M):
        self.default_resolution = default_resolution
        self.epsilon_m = epsilon_m

    def compute_visibility_map(self, mesh: Mesh3D | tuple[np.ndarray, np.ndarray], pose: Any = None,
                               intrinsics: CameraIntrinsics | None = None, resolution: tuple[int, int] | None = None,
                               yaw_override: float | None = None) -> VisibilityResult:
        t0 = time.perf_counter()
        mesh = _unpack_mesh(mesh)
        if resolution is not None:
            height, width = resolution
        elif intrinsics is not None:
            height, width = intrinsics.height, intrinsics.width
        else:
            height, width = self.default_resolution
        intr = intrinsics or CameraIntrinsics(fx=500.0, fy=500.0, cx=width / 2.0, cy=height / 2.0, width=width, height=height)
        rotation, t, yaw = _resolve_pose(pose, yaw_override)

        model = np.asarray(mesh.vertices, np.float64) @ _FLIP
        cam = model @ rotation.T + t
        z = cam[:, 2]
        if (z <= 1e-4).any():
            raise DomainError("VISIBILITY_BEHIND_CAMERA", "Mesh vertices lie at or behind the camera plane", 422)
        u = intr.fx * cam[:, 0] / z + intr.cx
        v = intr.fy * cam[:, 1] / z + intr.cy
        count = len(z)

        # Outward normals as vectors: reflect into the head frame (not re-derived from reflected winding), then rotate.
        face_normals = mesh.compute_face_normals().astype(np.float64) @ _FLIP @ rotation.T
        vertex_normals = np.asarray(mesh.normals, np.float64) @ _FLIP @ rotation.T
        faces = mesh.faces.astype(np.int64)
        cos_limit = math.cos(math.radians(MAX_GRAZING_DEG))

        centroid = cam[faces].mean(axis=1)
        to_camera = -centroid / np.linalg.norm(centroid, axis=1, keepdims=True)
        front = np.einsum("ij,ij->i", face_normals, to_camera) > 0.0

        # Depth window: the frame plus one frame of margin each side.
        x_lo, y_lo = -width, -height
        win_w, win_h = 3 * width, 3 * height
        zbuf = np.full(win_h * win_w, np.inf, np.float64)
        self._splat(zbuf, win_w, win_h, x_lo, y_lo, faces[front], u, v, z)
        zbuf = zbuf.reshape(win_h, win_w)

        px = np.rint(u).astype(np.int64) - x_lo
        py = np.rint(v).astype(np.int64) - y_lo
        tested = (px >= 0) & (px < win_w) & (py >= 0) & (py < win_h)
        surface = np.full(count, np.inf)
        surface[tested] = zbuf[py[tested], px[tested]]
        depth_ok = np.where(tested, z <= surface + self.epsilon_m, True)
        vertex_to_camera = -cam / np.linalg.norm(cam, axis=1, keepdims=True)
        facing_ok = np.einsum("ij,ij->i", vertex_normals, vertex_to_camera) > cos_limit
        visible = depth_ok & facing_ok

        mask = np.zeros((height, width), np.uint8)
        tri_visible = front & visible[faces].all(axis=1)
        if tri_visible.any():
            polys = np.rint(np.stack([u[faces[tri_visible]], v[faces[tri_visible]]], axis=-1)).astype(np.int32)
            cv2.fillPoly(mask, list(polys), 255)
        soft = mask.astype(np.float32) / 255.0
        if FEATHER_PX > 0:
            soft = cv2.GaussianBlur(soft, (0, 0), FEATHER_PX / 2.0)
        frame_z = zbuf[-y_lo:-y_lo + height, -x_lo:-x_lo + width].astype(np.float32)
        return VisibilityResult(
            vertex_visible=visible, vertex_depth_tested=tested, visibility_map=np.clip(soft, 0.0, 1.0), z_buffer=frame_z,
            self_occluded_fraction=float(1.0 - visible.mean()), yaw_degrees=yaw,
            execution_time_ms=(time.perf_counter() - t0) * 1000.0)

    @staticmethod
    def _splat(zbuf: np.ndarray, win_w: int, win_h: int, x_lo: int, y_lo: int, faces: np.ndarray,
               u: np.ndarray, v: np.ndarray, z: np.ndarray) -> None:
        """Perspective-correct nearest-z splat of barycentric samples of ``faces`` into the flat buffer."""
        if len(faces) == 0:
            return
        fu, fv, fz = u[faces], v[faces], z[faces]
        edge = np.maximum.reduce([np.hypot(fu[:, 0] - fu[:, 1], fv[:, 0] - fv[:, 1]),
                                  np.hypot(fu[:, 1] - fu[:, 2], fv[:, 1] - fv[:, 2]),
                                  np.hypot(fu[:, 2] - fu[:, 0], fv[:, 2] - fv[:, 0])])
        need = np.clip(np.ceil(edge / _SAMPLE_SPACING_PX), 1, _MAX_LATTICE)
        level = np.minimum(2 ** np.ceil(np.log2(need)), _MAX_LATTICE).astype(np.int64)
        for n in np.unique(level):
            sel = level == n
            i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
            keep = i + j <= n
            bary = np.stack([(n - i - j)[keep], i[keep], j[keep]], axis=1) / float(n)   # (S, 3)
            inv = np.einsum("sk,fk->fs", bary, 1.0 / fz[sel])
            su = np.einsum("sk,fk->fs", bary, fu[sel])
            sv = np.einsum("sk,fk->fs", bary, fv[sel])
            ix = np.rint(su).astype(np.int64) - x_lo
            iy = np.rint(sv).astype(np.int64) - y_lo
            ok = (ix >= 0) & (ix < win_w) & (iy >= 0) & (iy < win_h)
            np.minimum.at(zbuf, (iy[ok] * win_w + ix[ok]), 1.0 / inv[ok])
