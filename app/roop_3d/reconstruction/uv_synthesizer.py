"""Visibility-weighted multi-view UV texture synthesis with strict confidence gating.

Projects observed multi-angle FaceSet images onto canonical UV space weighted by surface
normal dot-product (camera visibility), image sharpness, and occlusion masks.
Explicitly tracks a per-texel uv_confidence_mask distinguishing OBSERVED, INTERPOLATED,
and UNSEEN regions. Never blends unseen back-of-head or occluded background pixels as
valid skin texture.
"""
from __future__ import annotations

import math
from typing import Any, Sequence
import cv2
import numpy as np
from pydantic import ConfigDict, Field, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    Pose3D,
    UVOrigin,
    WeakPerspectiveCamera,
    uv_to_pixel,
)


def remap_points(image: np.ndarray, xs: np.ndarray, ys: np.ndarray, interpolation: int = cv2.INTER_LINEAR,
                 *, device: str = "cpu") -> np.ndarray:
    """Sample `image` at N sub-pixel points. `cv2.remap` rejects maps with >= 32767 rows, so the points are laid
    out as a (ceil(N/1024), 1024) grid; returns (N,) or (N, C). CUDA uses
    ``torch.grid_sample`` when available and otherwise keeps the CPU reference path."""
    if device not in ("cpu", "cuda"):
        raise ValueError("Sampling device must be cpu or cuda")
    xs, ys = np.asarray(xs, np.float32).reshape(-1), np.asarray(ys, np.float32).reshape(-1)
    n = len(xs)
    if n == 0:
        return np.empty((0, *np.asarray(image).shape[2:]), dtype=np.asarray(image).dtype)
    width = 1024
    rows = max(1, -(-n // width))
    gx = np.zeros(rows * width, np.float32)
    gy = np.zeros(rows * width, np.float32)
    gx[:n], gy[:n] = xs, ys
    source = np.asarray(image)
    if device == "cuda":
        try:
            import torch
            import torch.nn.functional as functional
            if torch.cuda.is_available():
                height, image_width = source.shape[:2]
                x = np.zeros_like(gx) if image_width <= 1 else gx * (2.0 / (image_width - 1)) - 1.0
                y = np.zeros_like(gy) if height <= 1 else gy * (2.0 / (height - 1)) - 1.0
                x[n:], y[n:] = 2.0, 2.0
                grid = torch.as_tensor(np.stack((x, y), axis=-1).reshape(1, rows, width, 2),
                                       dtype=torch.float32, device="cuda")
                pixels = torch.as_tensor(source, dtype=torch.float32, device="cuda")
                if pixels.ndim == 2:
                    pixels = pixels[None, None]
                    single_channel = True
                else:
                    pixels = pixels.permute(2, 0, 1)[None]
                    single_channel = False
                mode = "nearest" if interpolation == cv2.INTER_NEAREST else "bilinear"
                sampled = functional.grid_sample(pixels, grid, mode=mode, padding_mode="zeros",
                                                align_corners=True)[0].permute(1, 2, 0).cpu().numpy()
                if single_channel:
                    sampled = sampled[..., 0]
                if source.dtype == np.uint8:
                    sampled = np.clip(np.rint(sampled), 0, 255).astype(np.uint8)
                else:
                    sampled = sampled.astype(source.dtype, copy=False)
                return sampled.reshape(rows * width, *sampled.shape[2:])[:n]
        except (ImportError, RuntimeError, ValueError):
            pass
    out = cv2.remap(source, gx.reshape(rows, width), gy.reshape(rows, width), interpolation,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape(rows * width, *out.shape[2:])[:n]


class ViewObservation(Contract):
    """Observation frame with camera pose, intrinsics, and quality metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.ndarray = Field(description="BGR image array (H, W, 3)")
    pose: Pose3D | None = Field(default=None, description="Calibrated head pose in camera frame (legacy path)")
    intrinsics: CameraIntrinsics | None = Field(default=None, description="Pinhole camera parameters (legacy path)")
    sharpness: float = Field(default=1.0, ge=0.0, description="Image sharpness metric")
    occlusion_mask: np.ndarray | None = Field(default=None, description="Optional binary face mask (H, W)")
    camera: WeakPerspectiveCamera | None = Field(
        default=None, description="Scaled-orthographic camera fitted to this view (identity path)")
    view_quality: float = Field(default=1.0, ge=0.0, le=1.0,
                                description="0..1 trust in this view (landmark consistency x image quality)")
    view_index: int = Field(default=-1, description="Source FaceSet view index, reported in per-texel provenance")

    @model_validator(mode="after")
    def _has_camera_model(self) -> "ViewObservation":
        if self.camera is None and (self.pose is None or self.intrinsics is None):
            raise ValueError("a ViewObservation needs either `camera` or both `pose` and `intrinsics`")
        return self


class UVSynthesisResult(Contract):
    """Output synthesized UV texture map with per-texel confidence mask."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texture_map: np.ndarray = Field(description="Synthesized BGR texture map (H_uv, W_uv, 3) uint8")
    confidence_mask: np.ndarray = Field(description="Per-texel confidence state mask (H_uv, W_uv) uint8")
    observed_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels directly observed")
    interpolated_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels interpolated")
    unseen_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels remaining unseen")
    # Identity (camera) path only; None for the legacy pose/intrinsics path.
    confidence_score: np.ndarray | None = Field(default=None, description="Per-texel float32 confidence in [0, 1]")
    view_count: np.ndarray | None = Field(default=None, description="Per-texel uint8 number of contributing views")
    dominant_view: np.ndarray | None = Field(default=None, description="Per-texel int32 view_index of the top contributor (-1 none)")
    view_gains: dict[int, tuple[float, float, float]] | None = Field(default=None, description="Per-view BGR exposure gain applied")
    rejected_samples: int = Field(default=0, ge=0, description="Photometric outlier samples excluded from blending")


class UVSynthesizer:
    """Visibility-weighted UV texture synthesizer."""

    def __init__(
        self,
        mesh: Mesh3D,
        uv_size: int = 512,
        visibility_threshold: float = 0.05,
        max_interpolation_fraction: float | None = None,
        blend_mode: str = "weighted",
        visibility_backend: str = "cpu",
    ):
        if blend_mode not in ("weighted", "multiband"):
            raise DomainError("INVALID_BLEND_MODE", f"blend_mode must be 'weighted' or 'multiband', got {blend_mode!r}", 422)
        if uv_size < 64 or uv_size > 4096:
            raise DomainError("INVALID_UV_SIZE", f"UV size must be in [64, 4096], got {uv_size}", 422)
        if mesh.uvs is None:
            raise DomainError("MISSING_UVS", "Mesh must have valid UV coordinates for texture synthesis", 422)

        self.mesh = mesh
        self.uv_size = uv_size
        # 'multiband': Laplacian-pyramid blend of the per-view layers (identity/camera path only); 'weighted': per-texel average.
        self.blend_mode = blend_mode
        if visibility_backend not in ("cpu", "gpu"):
            raise ValueError("Visibility backend must be cpu or gpu")
        self.visibility_backend = visibility_backend
        self.vis_thresh = float(visibility_threshold)
        # Identity path only: interpolation is bounded to this distance (in texels) from an observed texel.
        self.max_interp_px = None if max_interpolation_fraction is None else max(1.0, max_interpolation_fraction * uv_size)

        # Precompute rasterized 3D surface positions and normals in UV space
        self._rasterize_uv_geometry()

    def _rasterize_uv_geometry(self) -> None:
        """Rasterize mesh 3D positions and normals onto UV pixel grid."""
        h = w = self.uv_size
        self.pos_map = np.zeros((h, w, 3), dtype=np.float32)
        self.norm_map = np.zeros((h, w, 3), dtype=np.float32)
        self.face_texel_mask = np.zeros((h, w), dtype=bool)

        verts = self.mesh.vertices
        normals = self.mesh.normals
        uvs = self.mesh.uvs
        faces = self.mesh.faces

        # Convert UV coordinates to raster pixel space (origin: TOP_LEFT for image indexing)
        uv_pixels = uv_to_pixel(uvs, w, h, origin=UVOrigin.TOP_LEFT)

        for tri in faces:
            p0_uv = uv_pixels[tri[0]]
            p1_uv = uv_pixels[tri[1]]
            p2_uv = uv_pixels[tri[2]]

            # Bounding box of triangle in UV grid
            min_x = max(0, int(math.floor(min(p0_uv[0], p1_uv[0], p2_uv[0]))))
            max_x = min(w - 1, int(math.ceil(max(p0_uv[0], p1_uv[0], p2_uv[0]))))
            min_y = max(0, int(math.floor(min(p0_uv[1], p1_uv[1], p2_uv[1]))))
            max_y = min(h - 1, int(math.ceil(max(p0_uv[1], p1_uv[1], p2_uv[1]))))

            if min_x > max_x or min_y > max_y:
                continue

            # Barycentric denominator: (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
            denom = (p1_uv[1] - p2_uv[1]) * (p0_uv[0] - p2_uv[0]) + (p2_uv[0] - p1_uv[0]) * (p0_uv[1] - p2_uv[1])
            if abs(denom) < 1e-8:
                continue

            # Grid of coordinates in bounding box
            grid_y, grid_x = np.mgrid[min_y:max_y + 1, min_x:max_x + 1]

            w0 = ((p1_uv[1] - p2_uv[1]) * (grid_x - p2_uv[0]) + (p2_uv[0] - p1_uv[0]) * (grid_y - p2_uv[1])) / denom
            w1 = ((p2_uv[1] - p0_uv[1]) * (grid_x - p2_uv[0]) + (p0_uv[0] - p2_uv[0]) * (grid_y - p2_uv[1])) / denom
            w2 = 1.0 - w0 - w1

            inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not np.any(inside):
                continue

            w0_in = w0[inside, None]
            w1_in = w1[inside, None]
            w2_in = w2[inside, None]

            # Interpolate 3D position and normal
            pos_interp = w0_in * verts[tri[0]] + w1_in * verts[tri[1]] + w2_in * verts[tri[2]]
            norm_interp = w0_in * normals[tri[0]] + w1_in * normals[tri[1]] + w2_in * normals[tri[2]]
            norm_len = np.linalg.norm(norm_interp, axis=-1, keepdims=True)
            norm_interp = norm_interp / np.maximum(norm_len, 1e-12)

            ys = grid_y[inside]
            xs = grid_x[inside]

            self.pos_map[ys, xs] = pos_interp
            self.norm_map[ys, xs] = norm_interp
            self.face_texel_mask[ys, xs] = True

    def synthesize(
        self,
        observations: Sequence[ViewObservation],
    ) -> UVSynthesisResult:
        """Synthesize UV texture from multi-angle observations."""
        if not observations:
            raise DomainError("EMPTY_OBSERVATIONS", "At least one view observation is required for UV synthesis", 422)
        if any(obs.camera is not None for obs in observations):
            if any(obs.camera is None for obs in observations):
                raise DomainError("MIXED_CAMERA_MODELS", "Either all or none of the observations may carry a camera", 422)
            return self._synthesize_cameras(observations)

        h = w = self.uv_size
        accum_color = np.zeros((h, w, 3), dtype=np.float64)
        accum_weight = np.zeros((h, w), dtype=np.float64)
        observed_mask = np.zeros((h, w), dtype=bool)

        for obs in observations:
            img = obs.image
            h_img, w_img = img.shape[:2]
            r_mat = obs.pose.rotation_matrix()
            t_vec = np.array(obs.pose.tvec, dtype=np.float64)
            intrinsics = obs.intrinsics

            # Only process valid face texels
            valid_texels = self.face_texel_mask
            if not np.any(valid_texels):
                continue

            pts_3d = self.pos_map[valid_texels]  # (K, 3)
            norms_3d = self.norm_map[valid_texels]  # (K, 3)

            # Transform normals to camera frame: N_cam = R * N
            norms_cam = norms_3d @ r_mat.T  # (K, 3)

            # Camera visibility: dot product of surface normal with viewing direction towards viewer (+Z)
            # Surfaces with positive Z-normal in camera frame face the viewer
            vis_dot = norms_cam[:, 2]  # (K,)

            # Filter out surface facing away from camera (back-of-head)
            facing_camera = vis_dot > self.vis_thresh
            if not np.any(facing_camera):
                continue

            # Transform points to camera frame: P_cam = R * P + t
            pts_cam = (pts_3d @ r_mat.T) + t_vec

            # Filter points in front of camera
            in_front = pts_cam[:, 2] > 0.05
            valid_subset = facing_camera & in_front
            if not np.any(valid_subset):
                continue

            # Project into image coordinates
            pts_sub = pts_cam[valid_subset]
            proj_x = intrinsics.fx * (pts_sub[:, 0] / pts_sub[:, 2]) + intrinsics.cx
            proj_y = intrinsics.fy * (pts_sub[:, 1] / pts_sub[:, 2]) + intrinsics.cy

            # Bounds check inside image frame
            in_bounds = (proj_x >= 0.0) & (proj_x <= w_img - 1.0) & (proj_y >= 0.0) & (proj_y <= h_img - 1.0)
            if not np.any(in_bounds):
                continue

            # Map back to full valid_texels indices
            subset_indices = np.where(valid_subset)[0][in_bounds]
            px_valid = proj_x[in_bounds].astype(np.float32)
            py_valid = proj_y[in_bounds].astype(np.float32)

            # Occlusion mask check
            if obs.occlusion_mask is not None:
                mask_val = remap_points(obs.occlusion_mask.astype(np.float32), px_valid, py_valid, cv2.INTER_NEAREST)
                unoccluded = mask_val > 128
                subset_indices = subset_indices[unoccluded]
                px_valid = px_valid[unoccluded]
                py_valid = py_valid[unoccluded]
                if len(subset_indices) == 0:
                    continue

            # Sample color from image with bilinear interpolation
            sampled_colors = remap_points(img, px_valid, py_valid, cv2.INTER_LINEAR).reshape(-1, 3).astype(np.float64)

            # Weight = dot product * image sharpness
            sub_vis = vis_dot[subset_indices]
            weights = sub_vis * max(obs.sharpness, 0.1)

            # Accumulate onto UV grid
            all_valid_ys, all_valid_xs = np.where(valid_texels)
            target_ys = all_valid_ys[subset_indices]
            target_xs = all_valid_xs[subset_indices]

            for c in range(3):
                np.add.at(accum_color[:, :, c], (target_ys, target_xs), sampled_colors[:, c] * weights)
            np.add.at(accum_weight, (target_ys, target_xs), weights)
            observed_mask[target_ys, target_xs] = True

        # Synthesize base texture from observed texels
        texture_map = np.zeros((h, w, 3), dtype=np.uint8)
        has_weight = accum_weight > 1e-6
        for c in range(3):
            texture_map[has_weight, c] = np.clip(
                accum_color[has_weight, c] / accum_weight[has_weight],
                0,
                255,
            ).astype(np.uint8)

        # Classify confidence states
        confidence_mask = np.zeros((h, w), dtype=np.uint8)
        confidence_mask[observed_mask] = int(ConfidenceState.OBSERVED)

        # Identify holes on the face mesh (covered by face UV geometry but unobserved)
        face_unobserved = self.face_texel_mask & (~observed_mask)

        # Smoothly interpolate missing face texels using inpainting
        if np.any(face_unobserved) and np.any(observed_mask):
            inpaint_mask = np.zeros((h, w), dtype=np.uint8)
            inpaint_mask[face_unobserved] = 255
            inpainted = cv2.inpaint(texture_map, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

            # Apply only to face geometry holes
            texture_map[face_unobserved] = inpainted[face_unobserved]
            confidence_mask[face_unobserved] = int(ConfidenceState.INTERPOLATED)

        # Texels outside face geometry or back of head without observations remain strictly UNSEEN
        unseen_mask = ~self.face_texel_mask
        confidence_mask[unseen_mask] = int(ConfidenceState.UNSEEN)
        # Never blend unseen regions: set pixel colors strictly to zero
        texture_map[unseen_mask] = 0

        # Calculate coverage metrics over face geometry
        total_face = max(int(np.sum(self.face_texel_mask)), 1)
        n_obs = int(np.sum(confidence_mask == int(ConfidenceState.OBSERVED)))
        n_interp = int(np.sum(confidence_mask == int(ConfidenceState.INTERPOLATED)))
        n_unseen = total_face - n_obs - n_interp

        return UVSynthesisResult(
            texture_map=texture_map,
            confidence_mask=confidence_mask,
            observed_ratio=float(n_obs / total_face),
            interpolated_ratio=float(n_interp / total_face),
            unseen_ratio=float(max(0, n_unseen) / total_face),
        )

    # ------------------------------------------------------------------ identity (camera) path
    OBSERVED_MIN_WEIGHT = 0.02        # total blend weight a texel needs to count as OBSERVED
    # Real-data finding (Stage 4): with 5-point geometry the views are only loosely registered, so averaging several views
    # (or trusting surfaces seen at a grazing angle, where texture is stretched) produces ghosts and seams. A texel must
    # face the camera within ~75 deg, weights are sharpened (vis^3) and a secondary view carrying < 25 % of the best
    # view's weight is not blended in at all.
    GRAZING_MIN_COS = 0.25
    WEIGHT_POWER = 3.0
    DOMINANCE_FRACTION = 0.25
    OUTLIER_COLOR_DISTANCE = 55.0     # BGR L2 distance from the per-texel median beyond which a sample is rejected
    DEPTH_TOLERANCE_M = 0.008         # z-buffer slack: covers a triangle's own depth range

    SILHOUETTE_FEATHER = 0.04         # fraction of the face extent over which weights ramp up from the silhouette

    def _visible_by_zbuffer(self, camera: WeakPerspectiveCamera, texel_depth: np.ndarray,
                            texel_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(visible, silhouette_feather) per texel. `visible` is False where the surface point is hidden behind other
        front-facing geometry; the feather ramps 0 -> 1 with distance from the projected silhouette, so texels at
        the edge of the mask (where the photo shows hair, neck or background) carry little weight."""
        if self.visibility_backend == "gpu":
            return self._visible_on_gpu(camera, texel_xy)
        verts = self.mesh.vertices.astype(np.float64)
        vxy, vdepth = camera.project(verts), camera.depth(verts)
        face_n = camera.rotate_normals(self.mesh.compute_face_normals())[:, 2]
        front = np.where(face_n > 0.0)[0]
        if len(front) == 0:
            return np.zeros(len(texel_depth), dtype=bool), np.zeros(len(texel_depth))
        tri = self.mesh.faces[front]
        lo, hi = vxy[tri].reshape(-1, 2).min(axis=0), vxy[tri].reshape(-1, 2).max(axis=0)
        extent = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
        bs = float(np.clip(512.0 / extent, 0.25, 1.0))
        size = (int(math.ceil((hi[1] - lo[1]) * bs)) + 4, int(math.ceil((hi[0] - lo[0]) * bs)) + 4)
        zbuf = np.full(size, -np.inf, dtype=np.float32)
        order = np.argsort(vdepth[tri].mean(axis=1))          # far first, near paints over
        poly_xy = (vxy[tri] - lo) * bs + 2.0
        for k in order:
            cv2.fillConvexPoly(zbuf, np.round(poly_xy[k]).astype(np.int32), float(vdepth[tri[k]].mean()))
        bx = np.clip(np.round((texel_xy[:, 0] - lo[0]) * bs + 2.0).astype(np.int64), 0, size[1] - 1)
        by = np.clip(np.round((texel_xy[:, 1] - lo[1]) * bs + 2.0).astype(np.int64), 0, size[0] - 1)
        visible = texel_depth >= zbuf[by, bx] - self.DEPTH_TOLERANCE_M
        footprint = np.isfinite(zbuf).astype(np.uint8)
        distance = cv2.distanceTransform(footprint, cv2.DIST_L2, 3) / bs      # image pixels
        feather = np.clip(distance[by, bx] / max(self.SILHOUETTE_FEATHER * extent, 1.0), 0.0, 1.0)
        return visible, feather

    def _visible_on_gpu(self, camera, texel_xy):
        """Depth-test every UV surface sample, not average-depth triangle painting."""
        from roop_3d.reconstruction.full_head import GPUHeadRasterizer, head_roi
        class OrthographicPose:
            perspective = False
            rotation = np.diag([1., 1., -1.]) @ np.asarray(camera.rotation) @ np.diag([1., 1., -1.])
            def project(self, points):
                return camera.project(points)
            def camera_vertices(self, points):
                return np.column_stack((np.zeros((len(points), 2)), 2. - camera.depth(points)))
        pose = OrthographicPose()
        side = 1024
        inverse = head_roi(camera.project(self.mesh.vertices), side)
        ys, xs = np.where(self.face_texel_mask)
        points = self.pos_map[ys, xs].astype(np.float64)
        renderer = GPUHeadRasterizer()
        try:
            buffers = renderer.render_buffers(self.mesh.vertices, self.mesh.faces,
                np.column_stack((self.mesh.uvs[:, 0], 1 - self.mesh.uvs[:, 1])),
                np.zeros((64, 64, 3), np.uint8), np.ones((64, 64), np.float32), pose, inverse, side)
            visible = renderer.visibility(points, pose, inverse, buffers, tolerance=.0015)
        finally:
            renderer.release()
        local = cv2.transform(np.asarray(texel_xy, np.float32)[None], cv2.invertAffineTransform(inverse))[0]
        xx, yy = np.rint(local).astype(np.int64).T
        inside = (xx >= 0) & (xx < side) & (yy >= 0) & (yy < side)
        xx, yy = np.clip(xx, 0, side-1), np.clip(yy, 0, side-1)
        distance = cv2.distanceTransform((buffers.silhouette > .5).astype(np.uint8), cv2.DIST_L2, 3)
        feather = np.clip(distance[yy, xx] / (self.SILHOUETTE_FEATHER * side), 0, 1)
        return visible & inside, feather

    @staticmethod
    def _local_sharpness(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return cv2.GaussianBlur(np.abs(cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)), (0, 0), 3.0)

    @staticmethod
    def _solve_gains(colors: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Per-view, per-channel exposure gains from overlap statistics (gain compensation); (V, 3)."""
        n_views = colors.shape[0]
        gains = np.ones((n_views, 3))
        present = weights > 0.02
        pairs = []
        for i in range(n_views):
            for j in range(i + 1, n_views):
                m = present[i] & present[j]
                n = int(m.sum())
                if n >= 40:
                    pairs.append((i, j, n, colors[i][m].mean(axis=0), colors[j][m].mean(axis=0)))
        if not pairs:
            return gains
        ridge = 0.3 * float(np.mean([math.sqrt(p[2]) for p in pairs]))
        for c in range(3):
            rows, rhs = [], []
            for i, j, n, a, b in pairs:
                row = np.zeros(n_views)
                row[i], row[j] = a[c], -b[c]
                rows.append(row * math.sqrt(n) / max(a[c], b[c], 1.0))
                rhs.append(0.0)
            for i in range(n_views):
                row = np.zeros(n_views)
                row[i] = ridge
                rows.append(row)
                rhs.append(ridge)
            gains[:, c] = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)[0]
        gains = np.clip(gains, 0.7, 1.4)
        counts = present.sum(axis=1).astype(np.float64) + 1e-9
        return gains / ((gains * counts[:, None]).sum(axis=0) / counts.sum())

    def _synthesize_cameras(self, observations: Sequence[ViewObservation]) -> UVSynthesisResult:
        h = w = self.uv_size
        ys, xs = np.where(self.face_texel_mask)
        k_total = len(ys)
        if k_total == 0:
            raise DomainError("EMPTY_UV_LAYOUT", "The mesh UV layout covers no texels", 422)
        pts = self.pos_map[ys, xs].astype(np.float64)
        nrm = self.norm_map[ys, xs].astype(np.float64)
        n_views = len(observations)
        colors = np.zeros((n_views, k_total, 3), np.float32)
        weights = np.zeros((n_views, k_total), np.float32)
        sharp = np.zeros((n_views, k_total), np.float32)
        view_ids = np.array([o.view_index for o in observations], np.int32)

        for i, obs in enumerate(observations):
            cam, img = obs.camera, obs.image
            assert cam is not None
            if img.ndim != 3 or img.shape[2] != 3:
                raise DomainError("INVALID_IMAGE", "Observation images must be BGR (H, W, 3)", 422)
            h_img, w_img = img.shape[:2]
            vis = cam.rotate_normals(nrm)[:, 2]
            xy = cam.project(pts)
            in_img = (xy[:, 0] >= 1) & (xy[:, 0] <= w_img - 2) & (xy[:, 1] >= 1) & (xy[:, 1] <= h_img - 2)
            ok = (vis > max(self.vis_thresh, self.GRAZING_MIN_COS)) & in_img
            if not ok.any():
                continue
            unoccluded, silhouette = self._visible_by_zbuffer(cam, cam.depth(pts), xy)
            ok &= unoccluded
            if obs.occlusion_mask is not None:
                clear = remap_points(obs.occlusion_mask.astype(np.float32), xy[:, 0], xy[:, 1], cv2.INTER_NEAREST) > 128
                ok &= clear
            if not ok.any():
                continue
            colors[i, ok] = remap_points(img, xy[ok, 0], xy[ok, 1], cv2.INTER_LINEAR).reshape(-1, 3)
            sharp[i, ok] = remap_points(self._local_sharpness(img), xy[ok, 0], xy[ok, 1], cv2.INTER_LINEAR).reshape(-1)
            edge = np.minimum.reduce([xy[ok, 0], xy[ok, 1], w_img - 1 - xy[ok, 0], h_img - 1 - xy[ok, 1]])
            feather = np.clip(edge / max(0.03 * min(h_img, w_img), 1.0), 0.0, 1.0)
            weights[i, ok] = ((vis[ok] ** self.WEIGHT_POWER) * feather * silhouette[ok] * max(obs.view_quality, 0.05)
                              * max(obs.sharpness, 0.1) ** 0.5)

        # relative local sharpness: the sharpest observation of a texel gets full weight, blurrier ones down to 25%
        best = sharp.max(axis=0, keepdims=True)
        weights *= (0.25 + 0.75 * (sharp + 1e-3) / (best + 1e-3)) * (weights > 0)

        # dominance: drop views that contribute far less than the best view of the same texel (loosely registered
        # secondary views would only smear it)
        weights *= weights >= self.DOMINANCE_FRACTION * weights.max(axis=0, keepdims=True)
        gains = self._solve_gains(colors, weights) if n_views > 1 else np.ones((1, 3))
        colors = np.clip(colors * gains[:, None, :].astype(np.float32), 0, 255)

        # photometric outlier rejection (specular / occluder / residual lighting) against the per-texel median
        rejected = 0
        counts = (weights > 0).sum(axis=0)
        multi = counts >= 3
        if multi.any():
            masked = np.where((weights > 0)[:, multi, None], colors[:, multi], np.nan)
            with np.errstate(all="ignore"):
                median = np.nanmedian(masked, axis=0)
            dist = np.linalg.norm(colors[:, multi] - median[None], axis=2)
            bad = (weights[:, multi] > 0) & (dist > self.OUTLIER_COLOR_DISTANCE)
            rejected = int(bad.sum())
            sub = weights[:, multi]
            sub[bad] = 0.0
            weights[:, multi] = sub

        total = weights.sum(axis=0)
        observed_flat = total >= self.OBSERVED_MIN_WEIGHT
        mean_color = np.zeros((k_total, 3), np.float64)
        denom = np.maximum(total, 1e-9)
        for c in range(3):
            mean_color[:, c] = (weights * colors[:, :, c]).sum(axis=0) / denom

        if self.blend_mode == "multiband" and int((weights.max(axis=1) > 0).sum()) > 1:
            from roop_3d.reconstruction.uv_completion import multiband_blend
            used = np.nonzero(weights.max(axis=1) > 0)[0]
            layer_c = np.zeros((len(used), h, w, 3), np.float32)
            layer_w = np.zeros((len(used), h, w), np.float32)
            layer_c[:, ys, xs] = colors[used]
            layer_w[:, ys, xs] = weights[used]
            mean_color = multiband_blend(layer_c, layer_w)[ys, xs].astype(np.float64)

        texture = np.zeros((h, w, 3), np.uint8)
        state = np.zeros((h, w), np.uint8)
        score = np.zeros((h, w), np.float32)
        view_count = np.zeros((h, w), np.uint8)
        dominant = np.full((h, w), -1, np.int32)
        oy, ox = ys[observed_flat], xs[observed_flat]
        texture[oy, ox] = np.clip(mean_color[observed_flat], 0, 255).astype(np.uint8)
        state[oy, ox] = int(ConfidenceState.OBSERVED)
        score[oy, ox] = (1.0 - np.exp(-total[observed_flat] / 0.5)).astype(np.float32)
        view_count[oy, ox] = (weights[:, observed_flat] > 0).sum(axis=0).astype(np.uint8)
        dominant[oy, ox] = view_ids[np.argmax(weights[:, observed_flat], axis=0)]

        # bounded interpolation: only short gaps next to real observations; everything else stays UNSEEN
        observed = state == int(ConfidenceState.OBSERVED)
        limit = self.max_interp_px if self.max_interp_px is not None else max(2.0, 0.02 * self.uv_size)
        if observed.any():
            dist = cv2.distanceTransform((~observed).astype(np.uint8), cv2.DIST_L2, 3)
            candidates = self.face_texel_mask & ~observed & (dist <= limit)
            if candidates.any():
                fill = cv2.inpaint(texture, (~observed).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)
                texture[candidates] = fill[candidates]
                state[candidates] = int(ConfidenceState.INTERPOLATED)
                score[candidates] = (0.3 * (1.0 - dist[candidates] / (limit + 1e-6))).astype(np.float32)

        texture[~self.face_texel_mask] = 0
        state[~self.face_texel_mask] = int(ConfidenceState.UNSEEN)
        score[state == int(ConfidenceState.UNSEEN)] = 0.0
        total_face = max(int(self.face_texel_mask.sum()), 1)
        n_obs = int((state == int(ConfidenceState.OBSERVED)).sum())
        n_interp = int((state == int(ConfidenceState.INTERPOLATED)).sum())
        return UVSynthesisResult(
            texture_map=texture, confidence_mask=state,
            observed_ratio=n_obs / total_face, interpolated_ratio=n_interp / total_face,
            unseen_ratio=max(0, total_face - n_obs - n_interp) / total_face,
            confidence_score=score, view_count=view_count, dominant_view=dominant,
            view_gains={int(view_ids[i]): tuple(float(g) for g in gains[i]) for i in range(n_views)},
            rejected_samples=rejected,
        )
