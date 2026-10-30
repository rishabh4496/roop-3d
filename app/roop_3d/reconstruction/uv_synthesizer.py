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
from pydantic import ConfigDict, Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.reconstruction.geometry_types import (
    CameraIntrinsics,
    ConfidenceState,
    Mesh3D,
    Pose3D,
    UVOrigin,
    uv_to_pixel,
)


class ViewObservation(Contract):
    """Observation frame with camera pose, intrinsics, and quality metrics."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image: np.ndarray = Field(description="BGR image array (H, W, 3)")
    pose: Pose3D = Field(description="Calibrated head pose in camera frame")
    intrinsics: CameraIntrinsics = Field(description="Pinhole camera parameters")
    sharpness: float = Field(default=1.0, ge=0.0, description="Image sharpness metric")
    occlusion_mask: np.ndarray | None = Field(default=None, description="Optional binary face mask (H, W)")


class UVSynthesisResult(Contract):
    """Output synthesized UV texture map with per-texel confidence mask."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    texture_map: np.ndarray = Field(description="Synthesized BGR texture map (H_uv, W_uv, 3) uint8")
    confidence_mask: np.ndarray = Field(description="Per-texel confidence state mask (H_uv, W_uv) uint8")
    observed_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels directly observed")
    interpolated_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels interpolated")
    unseen_ratio: float = Field(ge=0.0, le=1.0, description="Ratio of face texels remaining unseen")


class UVSynthesizer:
    """Visibility-weighted UV texture synthesizer."""

    def __init__(
        self,
        mesh: Mesh3D,
        uv_size: int = 512,
        visibility_threshold: float = 0.05,
    ):
        if uv_size < 64 or uv_size > 4096:
            raise DomainError("INVALID_UV_SIZE", f"UV size must be in [64, 4096], got {uv_size}", 422)
        if mesh.uvs is None:
            raise DomainError("MISSING_UVS", "Mesh must have valid UV coordinates for texture synthesis", 422)

        self.mesh = mesh
        self.uv_size = uv_size
        self.vis_thresh = float(visibility_threshold)

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
                mask_val = cv2.remap(
                    obs.occlusion_mask.astype(np.float32),
                    px_valid[:, None],
                    py_valid[:, None],
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                ).reshape(-1)
                unoccluded = mask_val > 128
                subset_indices = subset_indices[unoccluded]
                px_valid = px_valid[unoccluded]
                py_valid = py_valid[unoccluded]
                if len(subset_indices) == 0:
                    continue

            # Sample color from image with bilinear interpolation
            sampled_colors = cv2.remap(
                img,
                px_valid[:, None],
                py_valid[:, None],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            ).reshape(-1, 3).astype(np.float64)

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
