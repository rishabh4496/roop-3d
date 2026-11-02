"""TPS / perspective-mesh pose correction on an explicit estimated 3D proxy.

No fitted identity geometry, calibrated camera or unseen texture is claimed.
All texture comes from the current swapped crop. Rasterization uses a z-buffer,
back-face rejection and perspective-correct sampling; unsupported pixels have
zero coverage and must preserve the target in the compositor.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.runtime import image_bgr
from roop_3d.faceset.recognizer import ARCFACE_REFERENCE_5PTS
from roop_3d.reconstruction.tps_fallback import (ThinPlateSplineWarper,
    build_canonical_head_mesh, get_canonical_3d_landmarks)


@dataclass(frozen=True)
class PoseBlendGeometry:
    texture: np.ndarray
    texture_coverage: np.ndarray
    flat_camera_vertices: np.ndarray
    mesh_camera_vertices: np.ndarray
    camera_matrix: np.ndarray
    uv: np.ndarray
    faces: np.ndarray
    flat_pixels: np.ndarray
    flat_visible_faces: np.ndarray
    mesh_visible_faces: np.ndarray


@dataclass(frozen=True)
class PoseBlendResult:
    image: np.ndarray
    coverage: np.ndarray
    tps_image: np.ndarray
    tps_coverage: np.ndarray
    mesh_image: np.ndarray
    mesh_coverage: np.ndarray
    mesh_pixels: np.ndarray
    visible_faces: np.ndarray
    reprojection_error: float
    geometry: PoseBlendGeometry
    blend_pixels: np.ndarray


def reblend(projection: PoseBlendResult, weight: float) -> PoseBlendResult:
    """Morph camera-space geometry, never cross-fade two misaligned faces."""
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise DomainError('POSE_BLEND_INVALID', 'Pose blend must be within 0..1', 422)
    tps, mesh = projection.tps_image, projection.mesh_image
    geometry = projection.geometry
    if weight == 0:
        image, alpha = tps.copy(), projection.tps_coverage.copy()
        pixels, faces = geometry.flat_pixels, geometry.flat_visible_faces
    elif weight == 1:
        image, alpha = mesh.copy(), projection.mesh_coverage.copy()
        pixels, faces = projection.mesh_pixels, geometry.mesh_visible_faces
    else:
        vertices = geometry.flat_camera_vertices * (1 - weight) + geometry.mesh_camera_vertices * weight
        homogeneous = vertices @ geometry.camera_matrix.T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        image, alpha, faces = rasterize(geometry.texture, geometry.texture_coverage, pixels, vertices,
            geometry.uv, geometry.faces, tps.shape[:2], perspective=True)
    image[alpha <= 0] = 0
    return replace(projection, image=image, coverage=alpha, blend_pixels=pixels, visible_faces=faces)


def rasterize(texture: np.ndarray, texture_coverage: np.ndarray, pixels: np.ndarray,
              camera_vertices: np.ndarray, uv: np.ndarray, faces: np.ndarray,
              shape: tuple[int, int], *, perspective: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bounded triangle rasterizer. UVs use image origin (top-left), in pixels."""
    h, w = shape
    depth = np.full((h, w), np.inf, np.float64)
    map_x = np.full((h, w), -1, np.float32)
    map_y = np.full((h, w), -1, np.float32)
    owner = np.full((h, w), -1, np.int32)
    for face_index, face in enumerate(faces):
        points = pixels[face].astype(np.float64)
        vertices = camera_vertices[face].astype(np.float64)
        if not np.isfinite(points).all() or not np.isfinite(vertices).all() or np.any(vertices[:, 2] <= 1e-4):
            continue
        if perspective:
            normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
            if np.dot(normal, -vertices.mean(0)) <= 1e-12:
                continue
        x0, x1 = max(0, int(np.floor(points[:, 0].min()))), min(w - 1, int(np.ceil(points[:, 0].max())))
        y0, y1 = max(0, int(np.floor(points[:, 1].min()))), min(h - 1, int(np.ceil(points[:, 1].max())))
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = points
        determinant = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(determinant) < 1e-8:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        p = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / determinant
        q = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / determinant
        barycentric = np.stack((p, q, 1 - p - q), axis=-1)
        inside = np.all(barycentric >= -1e-5, axis=-1)
        weighted = barycentric / vertices[:, 2] if perspective else barycentric
        denominator = weighted.sum(-1)
        z = 1 / np.maximum(denominator, 1e-12) if perspective else np.ones_like(p)
        region = depth[y0:y1 + 1, x0:x1 + 1]
        valid = inside & (denominator > 1e-12) & (z < region)
        if not valid.any():
            continue
        sampled = (weighted @ uv[face]) / np.maximum(denominator[..., None], 1e-12)
        region[valid] = z[valid]
        owner[y0:y1 + 1, x0:x1 + 1][valid] = face_index
        map_x[y0:y1 + 1, x0:x1 + 1][valid] = sampled[..., 0][valid]
        map_y[y0:y1 + 1, x0:x1 + 1][valid] = sampled[..., 1][valid]
    image = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    coverage = cv2.remap(texture_coverage, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    coverage *= np.isfinite(depth)
    visible = np.unique(owner[owner >= 0])
    return image, np.clip(coverage, 0, 1).astype(np.float32), faces[visible]


class PoseBlendProjector:
    MAX_SIZE = 512

    def __init__(self, *, vertices: np.ndarray | None = None, faces: np.ndarray | None = None,
                 anchors: np.ndarray | None = None, geometry: str = "ESTIMATED_CANONICAL_PROXY_68") -> None:
        # Default: the generic canonical proxy (unchanged behaviour). Stage 4 "3D Pose" mode supplies the SOURCE
        # identity's own geometry and 5 anchors (same mesh frame: x right, y down, z toward the viewer).
        if (vertices is None) != (faces is None) or (vertices is None) != (anchors is None):
            raise DomainError('POSE_GEOMETRY_INVALID', 'vertices, faces and anchors must be supplied together', 422)
        self.geometry = geometry
        if vertices is None:
            mesh = build_canonical_head_mesh(68)
            self.vertices = mesh.vertices.copy()
            self.faces = mesh.faces[:, [0, 2, 1]].copy()  # Front normals toward camera.
            self.anchors = get_canonical_3d_landmarks(5).copy()
        else:
            self.vertices = np.asarray(vertices, np.float32).copy()
            self.faces = np.asarray(faces, np.int32)[:, [0, 2, 1]].copy()
            self.anchors = np.asarray(anchors, np.float32).copy()
            if (self.vertices.ndim != 2 or self.vertices.shape[1] != 3 or self.anchors.shape != (5, 3)
                    or not np.isfinite(self.vertices).all() or not np.isfinite(self.anchors).all()):
                raise DomainError('POSE_GEOMETRY_INVALID', 'Identity geometry must be finite (N,3) vertices and (5,3) anchors', 422)
        self.vertices[:, 2] *= -1  # Nose toward camera in OpenCV (+Z into scene).
        self.anchors[:, 2] *= -1
        reference = ARCFACE_REFERENCE_5PTS.astype(np.float64).copy()
        reference[:, 0] += 8
        self.reference = reference / 128
        # Explicit correspondence only for the known five-point model. Do not
        # reinterpret a detector's 106/203 ordering as an anatomical topology.
        model_xy = self.anchors[:, :2].astype(np.float64) * 5
        atlas = ThinPlateSplineWarper(model_xy, self.reference, regularization=1e-6)
        self.uv = atlas.transform_points(self.vertices[:, :2] * 5)

    @classmethod
    def from_identity_asset(cls, asset) -> "PoseBlendProjector":
        """Same renderer, driven by the Stage 3 source identity's geometry (coarse proxy-resolution mesh)."""
        from roop_3d.reconstruction.multiview_fit import identity_template
        coarse = identity_template(0)                       # same topology as the canonical proxy; indices 0..N-1 are shared
        count = len(coarse.vertices)
        vertices = asset.vertices[:count].astype(np.float64)
        anchors = np.vstack([vertices[36:42].mean(axis=0), vertices[42:48].mean(axis=0), vertices[30], vertices[48], vertices[54]])
        return cls(vertices=vertices, faces=coarse.faces, anchors=anchors, geometry="IDENTITY_ASSET_" + asset.manifest.build_key[:12])

    def render(self, crop: np.ndarray, observed: np.ndarray, weight: float) -> PoseBlendResult:
        image_bgr(crop)
        h, w = crop.shape[:2]
        points = np.asarray(observed, np.float64)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise DomainError('POSE_BLEND_INVALID', 'Pose blend must be within 0..1', 422)
        if min(h, w) < 16 or max(h, w) > self.MAX_SIZE:
            raise DomainError('POSE_BLEND_SIZE', 'Pose projection requires a 16–512 pixel crop', 422)
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise DomainError('POSE_LANDMARKS_INVALID', 'Pose projection requires five finite observed landmarks', 422)
        eye_distance = float(np.linalg.norm(points[1] - points[0]))
        if eye_distance < 2 or cv2.contourArea(cv2.convexHull(points.astype(np.float32))) < 4:
            raise DomainError('POSE_LANDMARKS_INVALID', 'Observed landmarks cannot constrain pose projection', 422)
        scale = np.array((w, h), np.float64)
        normalized = points / scale
        # Neutralize the observed crop with a backward TPS. Evaluate in strips
        # so temporary TPS matrices stay bounded at the 512px crop limit.
        backward = ThinPlateSplineWarper(self.reference, normalized, regularization=1e-6)
        map_x, map_y = np.empty((h, w), np.float32), np.empty((h, w), np.float32)
        for start in range(0, h, 32):
            yy, xx = np.mgrid[start:min(h, start + 32), 0:w]
            samples = backward.transform_points(np.column_stack((xx.ravel() / w, yy.ravel() / h))) * scale
            map_x[start:start + len(yy)] = samples[:, 0].reshape(yy.shape)
            map_y[start:start + len(yy)] = samples[:, 1].reshape(yy.shape)
        texture = cv2.remap(crop, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        valid = cv2.remap(np.ones((h, w), np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        forward = ThinPlateSplineWarper(self.reference, normalized, regularization=1e-6)
        flat_pixels = forward.transform_points(self.uv) * scale
        # Camera is an explicit estimate for this crop, not supplied calibration.
        focal = max(w, h) * 1.5
        camera = np.array(((focal, 0, w / 2), (0, focal, h / 2), (0, 0, 1)), np.float64)
        try:
            ok, rvec, tvec = cv2.solvePnP(self.anchors.astype(np.float64), points, camera, None, flags=cv2.SOLVEPNP_SQPNP)
            if not ok:
                raise ValueError()
            rvec, tvec = cv2.solvePnPRefineLM(self.anchors.astype(np.float64), points, camera, None, rvec, tvec)
            rotation, _ = cv2.Rodrigues(rvec)
            camera_vertices = self.vertices @ rotation.T + tvec.reshape(3)
            projected, _ = cv2.projectPoints(self.vertices, rvec, tvec, camera, None)
            fitted, _ = cv2.projectPoints(self.anchors, rvec, tvec, camera, None)
            error = float(np.linalg.norm(fitted.reshape(-1, 2) - points, axis=1).mean())
            if not np.isfinite(camera_vertices).all() or np.any(camera_vertices[:, 2] <= 1e-4) or error > eye_distance * .4:
                raise ValueError()
        except (cv2.error, ValueError):
            raise DomainError('POSE_FIT_FAILED', 'Observed landmarks do not support a bounded front-facing proxy fit', 422) from None
        mesh_pixels = projected.reshape(-1, 2)
        flat_camera = np.column_stack((flat_pixels, np.ones(len(flat_pixels))))
        uv = self.uv * scale
        tps_image, tps_alpha, flat_visible = rasterize(texture, valid, flat_pixels, flat_camera, uv, self.faces, (h, w), perspective=False)
        mesh_image, mesh_alpha, visible = rasterize(texture, valid, mesh_pixels, camera_vertices, uv, self.faces, (h, w), perspective=True)
        if not np.any(tps_alpha > .5) or not np.any(mesh_alpha > .5):
            raise DomainError('POSE_VISIBILITY_EMPTY', 'No observed face surface is available for projection', 422)
        depth = float(np.mean(camera_vertices[:, 2]))
        flat_camera_vertices = np.column_stack(((flat_pixels[:, 0] - w / 2) / focal * depth,
                                               (flat_pixels[:, 1] - h / 2) / focal * depth,
                                               np.full(len(flat_pixels), depth)))
        geometry = PoseBlendGeometry(texture, valid, flat_camera_vertices, camera_vertices,
            camera, uv, self.faces, flat_pixels, flat_visible, visible)
        return reblend(PoseBlendResult(tps_image, tps_alpha, tps_image, tps_alpha,
            mesh_image, mesh_alpha, mesh_pixels, visible, error, geometry, mesh_pixels), weight)
