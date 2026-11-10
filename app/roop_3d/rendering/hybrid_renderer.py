# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Clean-room mesh + boundary point appearance primitives.

The mesh remains the owner of facial skin, visibility and depth. Sparse Gaussian
points are accepted only for explicitly labelled high-frequency boundary parts.
All tensor operations use upstream PyTorch APIs; no research project kernels,
checkpoints or non-commercial 3D Gaussian Splatting code are used here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


_BOUNDARY_REGIONS = frozenset({"eyebrow", "eyelash", "beard_stubble", "inner_mouth"})
_TORCH_MISSING = object()
_TORCH_CACHE: Any = None


def _optional_torch():
    global _TORCH_CACHE
    if _TORCH_CACHE is None:
        try:
            import torch
            _TORCH_CACHE = torch
        except ImportError:
            _TORCH_CACHE = _TORCH_MISSING
    return None if _TORCH_CACHE is _TORCH_MISSING else _TORCH_CACHE


def _torch_module():
    """Import optional PyTorch at the point a tensor path is requested."""
    torch = _optional_torch()
    if torch is None:
        raise RuntimeError("This rendering path requires the optional PyTorch runtime")
    return torch


def _is_tensor(value: Any) -> bool:
    torch = _optional_torch()
    return torch is not None and torch.is_tensor(value)


def _numpy_value(value: Any) -> np.ndarray:
    if _is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _directions_numpy(poses_ypr: np.ndarray) -> np.ndarray:
    angles = np.deg2rad(np.asarray(poses_ypr, np.float32)[..., :2])
    yaw, pitch = angles[..., 0], angles[..., 1]
    return np.stack((np.sin(yaw) * np.cos(pitch), np.sin(pitch),
                     np.cos(yaw) * np.cos(pitch)), axis=-1)


@dataclass(frozen=True)
class CanonicalAppearance:
    """Fused UV albedo in BGR [0, 1], confidence, and per-view attention."""

    albedo_bgr: Any                  # HxWx3; numpy or torch tensor
    confidence: Any                  # HxW; numpy or torch tensor
    attention: Any                   # ViewsxHxW; numpy or torch tensor


class MultiViewCanonicalAggregator:
    """Pose- and surface-conditioned cross-attention over UV-projected FaceSet views.

    Inputs are one observed UV layer per reference photograph. Use
    :meth:`project_faceset_views` to create those layers with the repository's
    visibility-tested ``UVSynthesizer``. Attention is deterministic: it uses
    view direction, UV support, and canonical surface normals, so it needs no
    learned checkpoint and a real profile view wins over frontal stretching
    wherever its projection has measured support.
    """

    def __init__(self, *, temperature: float = 0.18, normal_query_weight: float = 0.35) -> None:
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive and finite")
        if not np.isfinite(normal_query_weight) or not 0 <= normal_query_weight <= 1:
            raise ValueError("normal_query_weight must be within [0, 1]")
        self.temperature = float(temperature)
        self.normal_query_weight = float(normal_query_weight)

    @staticmethod
    def project_faceset_views(document, mesh, cameras: Mapping[int, Any], *, uv_size: int = 256,
                              visibility_backend: str = "cpu") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Project decoded ``FaceSetDocument`` photographs into separate UV layers.

        ``cameras`` maps FaceView.index to its fitted WeakPerspectiveCamera.
        Profiles remain separate observations instead of being averaged into a
        frontal atlas before view-conditioned rendering.
        """
        from roop_3d.reconstruction.uv_synthesizer import UVSynthesizer, ViewObservation

        if not document.views:
            raise ValueError("FaceSet must contain at least one view")
        uv_mesh = mesh
        synthesizer = UVSynthesizer(uv_mesh, uv_size=uv_size, max_interpolation_fraction=0,
                                    visibility_backend=visibility_backend)
        layers, supports, poses, qualities = [], [], [], []
        for view in document.views:
            camera = cameras.get(view.index)
            if camera is None:
                continue
            image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"FaceSet view {view.index} could not be decoded")
            observation = ViewObservation(
                image=image, camera=camera, sharpness=max(float(view.quality.sharpness), 0.1),
                view_quality=float(np.clip(view.quality.score, 0, 1)), view_index=int(view.index))
            layer = synthesizer.synthesize([observation])
            support = layer.confidence_score
            if support is None:
                support = (layer.confidence_mask > 0).astype(np.float32)
            layers.append(layer.texture_map)
            supports.append(np.asarray(support, np.float32))
            poses.append((float(view.yaw), float(view.pitch), float(view.roll)))
            qualities.append(float(np.clip(view.quality.score, 0, 1)))
        if not layers:
            raise ValueError("No FaceSet view had a fitted camera")
        return (np.stack(layers), np.stack(supports), np.asarray(poses, np.float32),
                np.asarray(qualities, np.float32))

    def aggregate(
        self,
        view_textures,
        view_support,
        view_poses_ypr,
        target_pose_ypr,
        *,
        qualities=None,
        uv_normals=None,
        base_albedo=None,
        base_confidence=None,
    ) -> CanonicalAppearance:
        """Cross-attend observed view layers for every UV texel.

        Supports numpy inputs for existing CPU render paths and torch inputs for
        differentiable GPU texture/pose optimization. Texture samples may be
        uint8 or float; output is always float BGR in [0, 1].
        """
        if any(_is_tensor(value) for value in (view_textures, view_support, view_poses_ypr,
                                                target_pose_ypr, uv_normals, base_albedo, base_confidence)
               if value is not None):
            return self._aggregate_torch(view_textures, view_support, view_poses_ypr, target_pose_ypr,
                                         qualities, uv_normals, base_albedo, base_confidence)
        return self._aggregate_numpy(view_textures, view_support, view_poses_ypr, target_pose_ypr,
                                     qualities, uv_normals, base_albedo, base_confidence)

    def _validate_numpy(self, textures, support, poses, qualities, normals, base, base_confidence):
        tex = np.asarray(textures)
        sup = np.asarray(support, np.float32)
        p = np.asarray(poses, np.float32)
        if tex.ndim != 4 or tex.shape[-1] != 3 or len(tex) == 0:
            raise ValueError("view_textures must have shape (views, H, W, 3)")
        if sup.shape != tex.shape[:3] or not np.isfinite(sup).all():
            raise ValueError("view_support must be finite with shape (views, H, W)")
        if p.ndim != 2 or p.shape[0] != len(tex) or p.shape[1] not in (2, 3) or not np.isfinite(p).all():
            raise ValueError("view_poses_ypr must have shape (views, 2|3)")
        q = np.ones(len(tex), np.float32) if qualities is None else np.asarray(qualities, np.float32)
        if q.shape != (len(tex),) or not np.isfinite(q).all() or np.any(q < 0):
            raise ValueError("qualities must be finite nonnegative values, one per view")
        n = None if normals is None else np.asarray(normals, np.float32)
        if n is not None and (n.shape != (*tex.shape[1:3], 3) or not np.isfinite(n).all()):
            raise ValueError("uv_normals must have shape (H, W, 3)")
        b = None if base is None else np.asarray(base)
        if b is not None and b.shape != tex.shape[1:]:
            raise ValueError("base_albedo must match one view texture")
        bc = None if base_confidence is None else np.asarray(base_confidence, np.float32)
        if bc is not None and bc.shape != tex.shape[1:3]:
            raise ValueError("base_confidence must match the UV dimensions")
        return tex, np.clip(sup, 0, 1), p, q, n, b, None if bc is None else np.clip(bc, 0, 1)

    def _aggregate_numpy(self, textures, support, poses, target, qualities, normals, base, base_confidence):
        tex, sup, poses, quality, normal, base, base_conf = self._validate_numpy(
            textures, support, poses, qualities, normals, base, base_confidence)
        target = np.asarray(target, np.float32).reshape(-1)
        if target.size not in (2, 3) or not np.isfinite(target).all():
            raise ValueError("target_pose_ypr must contain finite yaw, pitch[, roll]")
        rgb = tex.astype(np.float32) / (255.0 if tex.dtype == np.uint8 else 1.0)
        rgb = np.clip(rgb, 0, 1)
        source_dirs = _directions_numpy(poses)
        target_dir = _directions_numpy(target[None, :2])[0]
        target_dir /= max(float(np.linalg.norm(target_dir)), 1e-8)
        query_dir = np.broadcast_to(target_dir, (*sup.shape[1:], 3)).copy()
        if normal is not None:
            norm = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8)
            query_dir = (1 - self.normal_query_weight) * query_dir + self.normal_query_weight * norm
            query_dir /= np.maximum(np.linalg.norm(query_dir, axis=-1, keepdims=True), 1e-8)
        target_score = np.einsum("hwc,nc->nhw", query_dir, source_dirs, optimize=True)
        visible = (np.maximum(np.einsum("hwc,nc->nhw", normal / np.maximum(
            np.linalg.norm(normal, axis=-1, keepdims=True), 1e-8), source_dirs, optimize=True), 0)
                   if normal is not None else np.ones_like(sup))
        reliability = sup * quality[:, None, None] * visible
        valid = reliability > 1e-8
        logits = target_score / self.temperature + np.log(np.maximum(reliability, 1e-12))
        logits = np.where(valid, logits, -1e9)
        logits -= logits.max(axis=0, keepdims=True)
        weights = np.exp(logits) * valid
        weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-12)
        fused = np.einsum("nhw,nhwc->hwc", weights, rgb, optimize=True)
        confidence = np.clip(reliability, 0, 1).max(axis=0)
        if base is not None:
            base_float = base.astype(np.float32) / (255.0 if base.dtype == np.uint8 else 1.0)
            base_float = np.clip(base_float, 0, 1)
            old_confidence = np.zeros_like(confidence) if base_conf is None else base_conf
            use_views = (np.clip(confidence / 0.1, 0, 1) if base_conf is None else
                         confidence / np.maximum(confidence + old_confidence, 1e-8))
            fused = fused * use_views[..., None] + base_float * (1 - use_views[..., None])
            confidence = np.maximum(confidence, old_confidence)
        return CanonicalAppearance(np.clip(fused, 0, 1).astype(np.float32),
                                   np.clip(confidence, 0, 1).astype(np.float32),
                                   weights.astype(np.float32))

    def _aggregate_torch(self, textures, support, poses, target, qualities, normals, base, base_confidence):
        torch = _torch_module()
        reference = next(value for value in (textures, support, poses, target, normals, base, base_confidence)
                         if _is_tensor(value))
        device = reference.device
        tex = torch.as_tensor(textures, device=device)
        sup = torch.as_tensor(support, dtype=torch.float32, device=device).clamp(0, 1)
        p = torch.as_tensor(poses, dtype=torch.float32, device=device)
        q = torch.ones(tex.shape[0], device=device) if qualities is None else torch.as_tensor(
            qualities, dtype=torch.float32, device=device)
        if tex.ndim != 4 or tex.shape[-1] != 3 or tex.shape[0] == 0 or sup.shape != tex.shape[:3]:
            raise ValueError("view_textures and view_support must have compatible (views, H, W, ...) shapes")
        if p.shape[0] != tex.shape[0] or p.ndim != 2 or p.shape[1] not in (2, 3):
            raise ValueError("view_poses_ypr must have shape (views, 2|3)")
        if q.shape != (tex.shape[0],) or torch.any(q < 0):
            raise ValueError("qualities must contain one nonnegative value per view")
        target = torch.as_tensor(target, dtype=torch.float32, device=device).reshape(-1)
        if target.numel() not in (2, 3) or not bool(torch.isfinite(target).all()):
            raise ValueError("target_pose_ypr must contain yaw, pitch[, roll]")
        rgb = tex.to(torch.float32) / (255.0 if tex.dtype == torch.uint8 else 1.0)
        rgb = rgb.clamp(0, 1)
        def direction(value):
            angles = torch.deg2rad(value[..., :2])
            yaw, pitch = angles[..., 0], angles[..., 1]
            return torch.stack((torch.sin(yaw) * torch.cos(pitch), torch.sin(pitch),
                                torch.cos(yaw) * torch.cos(pitch)), dim=-1)
        source_dirs = direction(p)
        target_dir = torch.nn.functional.normalize(direction(target[None])[0], dim=0)
        query_dir = target_dir.view(1, 1, 3).expand(tex.shape[1], tex.shape[2], 3)
        normal = None if normals is None else torch.as_tensor(normals, dtype=torch.float32, device=device)
        if normal is not None:
            if normal.shape != (*tex.shape[1:3], 3):
                raise ValueError("uv_normals must have shape (H, W, 3)")
            normal = torch.nn.functional.normalize(normal, dim=-1)
            query_dir = torch.nn.functional.normalize((1 - self.normal_query_weight) * query_dir
                + self.normal_query_weight * normal, dim=-1)
        score = torch.einsum("hwc,nc->nhw", query_dir, source_dirs) / self.temperature
        visible = (torch.einsum("hwc,nc->nhw", normal, source_dirs).clamp_min(0)
                   if normal is not None else torch.ones_like(sup))
        reliability = sup * q[:, None, None] * visible
        valid = reliability > 1e-8
        logits = score + reliability.clamp_min(1e-12).log()
        logits = logits.masked_fill(~valid, -1e9)
        weights = torch.softmax(logits, dim=0) * valid.to(logits.dtype)
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(1e-12)
        fused = torch.einsum("nhw,nhwc->hwc", weights, rgb)
        confidence = reliability.clamp(0, 1).amax(dim=0)
        if base is not None:
            base = torch.as_tensor(base, device=device)
            base_float = base.to(torch.float32) / (255.0 if base.dtype == torch.uint8 else 1.0)
            old_conf = torch.zeros_like(confidence) if base_confidence is None else torch.as_tensor(
                base_confidence, dtype=torch.float32, device=device).clamp(0, 1)
            if base_float.shape != fused.shape or old_conf.shape != confidence.shape:
                raise ValueError("base albedo/confidence must match the UV dimensions")
            if not bool(torch.isfinite(base_float).all()) or not bool(torch.isfinite(old_conf).all()):
                raise ValueError("base albedo and confidence must be finite")
            use_views = ((confidence / 0.1).clamp(0, 1) if base_confidence is None else
                         confidence / (confidence + old_conf).clamp_min(1e-8))
            fused = fused * use_views[..., None] + base_float.clamp(0, 1) * (1 - use_views[..., None])
            confidence = torch.maximum(confidence, old_conf)
        return CanonicalAppearance(fused.clamp(0, 1), confidence.clamp(0, 1), weights)

    @staticmethod
    def sample_vertices(albedo_bgr, confidence, uvs):
        """Sample fused UV values at mesh UV coordinates; torch gradients are kept."""
        if _is_tensor(albedo_bgr) or _is_tensor(uvs):
            torch = _torch_module()
            import torch.nn.functional as functional
            reference = albedo_bgr if _is_tensor(albedo_bgr) else uvs
            atlas = torch.as_tensor(albedo_bgr, dtype=torch.float32, device=reference.device)
            uv = torch.as_tensor(uvs, dtype=torch.float32, device=reference.device)
            conf = torch.as_tensor(confidence, dtype=torch.float32, device=reference.device)
            grid = torch.stack((2 * uv[:, 0] - 1, 1 - 2 * uv[:, 1]), -1).view(1, 1, -1, 2)
            color = functional.grid_sample(atlas.permute(2, 0, 1)[None], grid, align_corners=True)[0, :, 0].T
            support = functional.grid_sample(conf[None, None], grid, align_corners=True)[0, 0, 0]
            return color, support
        atlas, uv = np.asarray(albedo_bgr, np.float32), np.asarray(uvs, np.float32)
        h, w = atlas.shape[:2]
        x = np.clip(uv[:, 0], 0, 1) * (w - 1)
        y = (1 - np.clip(uv[:, 1], 0, 1)) * (h - 1)
        return (cv2.remap(atlas, x.reshape(-1, 1), y.reshape(-1, 1), cv2.INTER_LINEAR).reshape(-1, 3),
                cv2.remap(np.asarray(confidence, np.float32), x.reshape(-1, 1), y.reshape(-1, 1),
                          cv2.INTER_LINEAR).reshape(-1))


@dataclass(frozen=True)
class ReducedBlendshapeBank:
    """At most 20 observed linear appearance bases and their expression map."""

    neutral_bgr: np.ndarray               # HxWx3 float32 [0, 1]
    banks_bgr: np.ndarray                 # 20xHxWx3 signed deltas
    expression_map: np.ndarray            # D x 20; normalized psi -> bank coefficients
    expression_center: np.ndarray         # D
    expression_scale: np.ndarray          # D
    sample_count: int

    BANK_COUNT = 20

    @classmethod
    def fit(cls, expression_codes, appearance_samples, *, neutral_index: int | None = None,
            ridge: float = 1e-3) -> "ReducedBlendshapeBank":
        """Compress pose-normalized observed atlases into <=20 linear banks.

        This is a closed-form PCA plus ridge fit, not a trained RGBAvatar model.
        A caller should pass aligned appearance samples from expression-varying
        references; missing expression evidence produces zero-valued bank slots.
        """
        psi = np.asarray(expression_codes, np.float32)
        images = np.asarray(appearance_samples)
        if psi.ndim != 2 or len(psi) < 2 or psi.shape[1] < 1 or not np.isfinite(psi).all():
            raise ValueError("expression_codes must be a finite (samples, channels) array with at least two rows")
        if images.ndim != 4 or len(images) != len(psi) or images.shape[-1] != 3:
            raise ValueError("appearance_samples must have shape (samples, H, W, 3)")
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0
        else:
            images = np.asarray(images, np.float32)
        if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1):
            raise ValueError("appearance samples must be finite BGR values in [0, 1]")
        if not np.isfinite(ridge) or ridge < 0:
            raise ValueError("ridge must be finite and nonnegative")
        if neutral_index is None:
            neutral_index = int(np.argmin(np.linalg.norm(psi, axis=1)))
        if not 0 <= neutral_index < len(psi):
            raise ValueError("neutral_index is outside the sample range")
        neutral = images[neutral_index].copy()
        centered_psi = psi - psi[neutral_index]
        scale = centered_psi.std(axis=0)
        scale[scale < 1e-5] = 1.0
        design = centered_psi / scale
        delta = (images - neutral).reshape(len(images), -1).astype(np.float64)

        # The sample Gram matrix avoids a large pixel-by-pixel covariance matrix.
        gram = delta @ delta.T
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1]
        rank = min(cls.BANK_COUNT, len(psi) - 1, int(np.count_nonzero(eigenvalues > 1e-10)))
        banks = np.zeros((cls.BANK_COUNT, *neutral.shape), np.float32)
        if rank:
            chosen = order[:rank]
            singular = np.sqrt(np.maximum(eigenvalues[chosen], 1e-10))
            components = (eigenvectors[:, chosen].T @ delta) / singular[:, None]
            scores = eigenvectors[:, chosen] * singular[None, :]
            lhs = design.astype(np.float64).T @ design.astype(np.float64)
            rhs = design.astype(np.float64).T @ scores
            mapping = np.linalg.solve(lhs + max(float(ridge), 1e-8) * np.eye(lhs.shape[0]), rhs)
            banks[:rank] = components.reshape(rank, *neutral.shape).astype(np.float32)
            expression_map = np.zeros((psi.shape[1], cls.BANK_COUNT), np.float32)
            expression_map[:, :rank] = mapping.astype(np.float32)
        else:
            expression_map = np.zeros((psi.shape[1], cls.BANK_COUNT), np.float32)
        return cls(neutral.astype(np.float32), banks, expression_map,
                   psi[neutral_index].astype(np.float32), scale.astype(np.float32), len(psi))

    def evaluate(self, psi_t, *, base_albedo=None, clamp: bool = True):
        """Evaluate a batch of expression vectors without expanding beyond 20 bases."""
        if _is_tensor(psi_t):
            torch = _torch_module()
            value = psi_t
            if value.shape[-1] != self.expression_map.shape[0]:
                raise ValueError("psi_t channel count does not match the fitted bank")
            device, dtype = value.device, value.dtype
            center = torch.as_tensor(self.expression_center, device=device, dtype=dtype)
            scale = torch.as_tensor(self.expression_scale, device=device, dtype=dtype)
            mapping = torch.as_tensor(self.expression_map, device=device, dtype=dtype)
            banks = torch.as_tensor(self.banks_bgr, device=device, dtype=dtype)
            neutral = torch.as_tensor(self.neutral_bgr if base_albedo is None else base_albedo,
                                      device=device, dtype=dtype)
            if neutral.shape != self.neutral_bgr.shape:
                raise ValueError("base_albedo must match the fitted bank atlas dimensions")
            coefficients = ((value - center) / scale) @ mapping
            delta = torch.einsum("...k,khwc->...hwc", coefficients, banks)
            result = neutral + delta
            return result.clamp(0, 1) if clamp else result
        value = np.asarray(psi_t, np.float32)
        if value.shape[-1] != self.expression_map.shape[0] or not np.isfinite(value).all():
            raise ValueError("psi_t must be finite and match the fitted expression channel count")
        coefficients = ((value - self.expression_center) / self.expression_scale) @ self.expression_map
        delta = np.einsum("...k,khwc->...hwc", coefficients, self.banks_bgr, optimize=True)
        neutral = np.asarray(self.neutral_bgr if base_albedo is None else base_albedo, np.float32)
        if neutral.shape != self.neutral_bgr.shape:
            raise ValueError("base_albedo must match the fitted bank atlas dimensions")
        result = neutral + delta
        return np.clip(result, 0, 1) if clamp else result


@dataclass(frozen=True)
class BoundarySplats:
    """Sparse camera-space appearance samples; regions are intentionally restricted."""

    points_camera: Any                 # Nx3, metric camera coordinates (+Z away)
    colors_bgr: Any                    # Nx3 linear [0, 1]
    opacity: Any                       # N [0, 1]
    sigma_px: Any                      # Nx2 or N
    regions: Sequence[str]             # eyebrow | eyelash | beard_stubble | inner_mouth


@dataclass(frozen=True)
class SplatLayer:
    color_bgr: Any                     # HxWx3 linear color
    alpha: Any                         # HxW


class BoundaryGaussianSplatRenderer:
    """Differentiable, depth-tested 2D Gaussian projection for sparse boundary points.

    A small fixed support window keeps memory bounded. The CUDA path uses only
    PyTorch operations, so gradients flow to point centers, colors, opacity and
    scale. No Gaussian point can claim skin coverage or a non-boundary region.
    """

    def __init__(self, *, max_radius_px: int = 12, depth_tolerance_m: float = 0.002) -> None:
        if not 1 <= max_radius_px <= 32 or not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0:
            raise ValueError("invalid splat radius or depth tolerance")
        self.max_radius_px = int(max_radius_px)
        self.depth_tolerance_m = float(depth_tolerance_m)

    def render(self, splats: BoundarySplats, camera_matrix, mesh_depth, *, device: str | None = None) -> SplatLayer:
        torch = _torch_module()
        import torch.nn.functional as functional
        if any(str(region) not in _BOUNDARY_REGIONS for region in splats.regions):
            raise ValueError("Gaussian splats are limited to eyebrow, eyelash, beard_stubble, and inner_mouth regions")
        reference = next((v for v in (splats.points_camera, splats.colors_bgr, mesh_depth) if _is_tensor(v)), None)
        if device is None:
            device = str(reference.device) if reference is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        depth = torch.as_tensor(mesh_depth, dtype=torch.float32, device=device)
        points = torch.as_tensor(splats.points_camera, dtype=torch.float32, device=device)
        colors = torch.as_tensor(splats.colors_bgr, dtype=torch.float32, device=device)
        opacity = torch.as_tensor(splats.opacity, dtype=torch.float32, device=device).reshape(-1).clamp(0, 1)
        sigma = torch.as_tensor(splats.sigma_px, dtype=torch.float32, device=device)
        matrix = torch.as_tensor(camera_matrix, dtype=torch.float32, device=device)
        if points.ndim != 2 or points.shape[1] != 3 or colors.shape != (len(points), 3):
            raise ValueError("points_camera and colors_bgr must have shapes (N,3)")
        if len(splats.regions) != len(points) or opacity.shape != (len(points),):
            raise ValueError("splat regions and opacity must have one entry per point")
        if sigma.ndim == 1:
            sigma = sigma[:, None].expand(-1, 2)
        if sigma.shape != (len(points), 2) or matrix.shape != (3, 3) or depth.ndim != 2:
            raise ValueError("sigma_px must be (N,) or (N,2), camera_matrix 3x3, mesh_depth HxW")
        h, w = depth.shape
        if len(points) == 0:
            return SplatLayer(torch.zeros((h, w, 3), device=device), torch.zeros((h, w), device=device))
        z = points[:, 2]
        projected = points @ matrix.T
        centers = projected[:, :2] / projected[:, 2:3].clamp_min(1e-6)
        ix, iy = centers[:, 0].detach().floor().long(), centers[:, 1].detach().floor().long()
        sigma = sigma.clamp(0.35, self.max_radius_px / 3.0)
        # Fixed support avoids a device synchronization to inspect tensor radii.
        radius = self.max_radius_px
        offsets = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
        oy, ox = torch.meshgrid(offsets, offsets, indexing="ij")
        px = ix[:, None, None] + ox
        py = iy[:, None, None] + oy
        dx = px.to(torch.float32) + 0.5 - centers[:, 0, None, None]
        dy = py.to(torch.float32) + 0.5 - centers[:, 1, None, None]
        gaussian = torch.exp(-0.5 * ((dx / sigma[:, 0, None, None]) ** 2
                                     + (dy / sigma[:, 1, None, None]) ** 2))
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        surface_z = depth[py.clamp(0, h - 1), px.clamp(0, w - 1)]
        visible = (~torch.isfinite(surface_z) | (z[:, None, None] <= surface_z + self.depth_tolerance_m))
        alpha = gaussian * opacity[:, None, None] * (valid & visible & (z[:, None, None] > 0))
        linear_index = (py.clamp(0, h - 1) * w + px.clamp(0, w - 1)).reshape(-1)
        flat_alpha = alpha.reshape(-1)
        alpha_sum = torch.zeros(h * w, dtype=torch.float32, device=device).index_add(0, linear_index, flat_alpha)
        color_values = (alpha[..., None] * colors[:, None, None, :]).reshape(-1, 3)
        color_sum = torch.zeros((h * w, 3), dtype=torch.float32, device=device).index_add(0, linear_index, color_values)
        color = color_sum / alpha_sum[:, None].clamp_min(1e-8)
        out_alpha = 1 - torch.exp(-alpha_sum)
        return SplatLayer(color.reshape(h, w, 3), out_alpha.reshape(h, w).clamp(0, 1))


@dataclass(frozen=True)
class HybridRenderResult:
    """Mesh-owned physical buffers plus an optional boundary-only point layer."""

    physical: Any
    splats: SplatLayer | None
    linear_bgr: Any
    coverage: Any
    mode: str


class FastUVFallbackRenderer:
    """Perspective depth-tested mesh-to-UV warp for low VRAM environments."""

    @staticmethod
    def render(vertices, faces, uvs, texture_bgr, uv_confidence, *, project, transform, shape,
               displacement_uv_m=None, vertex_normals=None, displacement_gain: float = 1.0,
               sampling_device: str = "cpu", perspective: bool = True):
        """Warp the canonical atlas through the posed mesh; relief displaces mesh vertices first."""
        from roop_3d.reconstruction.pose_blend import rasterize
        from roop_3d.reconstruction.uv_synthesizer import remap_points

        mesh_vertices = _numpy_value(vertices).astype(np.float32, copy=True)
        triangles = _numpy_value(faces).astype(np.int32, copy=False)
        uv = _numpy_value(uvs).astype(np.float32, copy=False)
        texture_bgr = _numpy_value(texture_bgr)
        uv_confidence = _numpy_value(uv_confidence).astype(np.float32, copy=False)
        if mesh_vertices.ndim != 2 or mesh_vertices.shape[1] != 3 or uv.shape != (len(mesh_vertices), 2):
            raise ValueError("vertices and uvs must have shapes (V,3) and (V,2)")
        if displacement_uv_m is not None:
            relief = _numpy_value(displacement_uv_m).astype(np.float32, copy=False)
            if relief.ndim != 2 or not np.isfinite(relief).all() or not np.isfinite(displacement_gain):
                raise ValueError("displacement_uv_m must be a finite 2D UV map")
            if np.any(np.abs(relief) > 1e-10):
                if vertex_normals is None:
                    tri = mesh_vertices[triangles]
                    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    normals = np.zeros_like(mesh_vertices)
                    for corner in range(3):
                        np.add.at(normals, triangles[:, corner], face_normals)
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
                else:
                    normals = _numpy_value(vertex_normals).astype(np.float32, copy=True)
                    if normals.shape != mesh_vertices.shape:
                        raise ValueError("vertex_normals must match vertices")
                    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
                texel_x = uv[:, 0] * (relief.shape[1] - 1)
                texel_y = (1.0 - uv[:, 1]) * (relief.shape[0] - 1)
                offsets = remap_points(relief, texel_x, texel_y, cv2.INTER_LINEAR)
                mesh_vertices += normals * (offsets[:, None] * float(displacement_gain))
        pixels = np.asarray(project(mesh_vertices), np.float32)
        camera_vertices = np.asarray(transform(mesh_vertices), np.float32)
        atlas_uv = np.column_stack((uv[:, 0] * (texture_bgr.shape[1] - 1),
                                    (1 - uv[:, 1]) * (texture_bgr.shape[0] - 1))).astype(np.float32)
        image, coverage, visible_faces = rasterize(texture_bgr, uv_confidence, pixels, camera_vertices,
            atlas_uv, triangles, tuple(shape), perspective=bool(perspective), sampling_device=sampling_device)
        return image, coverage, visible_faces


class HybridMeshGaussianRenderer:
    """Select the high-detail mesh + sparse-splat pass or the fast UV fallback.

    ``PhysicalBufferRenderer`` owns the primary skin surface. The optional
    expression bank modulates the UV appearance before rasterization. Boundary
    splats are depth tested and composited only inside mesh coverage.
    """

    def __init__(self, *, mode: str | None = None, profile: str = "12gb", physical_renderer=None,
                 splat_renderer: BoundaryGaussianSplatRenderer | None = None) -> None:
        if mode is None:
            mode = "low_vram" if profile.lower() in {"6gb", "low_vram", "lightweight"} else "hybrid"
        if mode not in {"hybrid", "low_vram"}:
            raise ValueError("mode must be 'hybrid' or 'low_vram'")
        from roop_3d.rendering.physical_buffers import PhysicalBufferRenderer
        self.mode = mode
        self.profile = profile
        self.physical_renderer = physical_renderer or PhysicalBufferRenderer()
        self.splat_renderer = splat_renderer or BoundaryGaussianSplatRenderer()

    def render(self, *, vertices, faces, uvs, albedo_uv, uv_confidence, pose, inverse,
               sh_coefficients, side: int = 512, expression_bank: ReducedBlendshapeBank | None = None,
               psi_t=None, boundary_splats: BoundarySplats | None = None, camera_matrix=None,
               fallback: Mapping[str, Any] | None = None, specular_strength: float = 0.25,
               shininess: float = 32.0, splat_device: str | None = None,
               projected_pixels=None, camera_vertices_override=None) -> HybridRenderResult:
        """Render one frame. ``fallback`` provides project/transform/shape callables for low VRAM."""
        if self.mode == "low_vram":
            if fallback is None:
                raise ValueError("low_vram mode requires fallback project, transform, and shape values")
            texture = albedo_uv
            if expression_bank is not None:
                if psi_t is None:
                    raise ValueError("psi_t is required when an expression appearance bank is supplied")
                texture = np.asarray(expression_bank.evaluate(_numpy_value(psi_t)), np.float32)
                if texture.ndim == 4:
                    if texture.shape[0] != 1:
                        raise ValueError("low_vram renderer handles one expression bank sample per call")
                    texture = texture[0]
                texture = np.rint(np.clip(texture, 0, 1) * 255).astype(np.uint8)
            elif _numpy_value(texture).dtype != np.uint8:
                texture = np.rint(np.clip(_numpy_value(texture), 0, 1) * 255).astype(np.uint8)
            image, coverage, _ = FastUVFallbackRenderer.render(
                vertices, faces, uvs, texture, uv_confidence,
                displacement_uv_m=fallback.get("displacement_uv_m"),
                vertex_normals=fallback.get("vertex_normals"),
                displacement_gain=fallback.get("displacement_gain", 1.0),
                sampling_device=fallback.get("sampling_device", "cpu"),
                perspective=fallback.get("perspective", True),
                project=fallback["project"], transform=fallback["transform"], shape=fallback["shape"])
            from roop_3d.rendering.skin_calibration import srgb_to_linear
            linear = srgb_to_linear(image.astype(np.float32) / 255.0)
            return HybridRenderResult(None, None, linear.astype(np.float32), coverage, "low_vram_uv_warp")

        texture = albedo_uv
        if expression_bank is not None:
            if psi_t is None:
                raise ValueError("psi_t is required when an expression appearance bank is supplied")
            expression_texture = expression_bank.evaluate(psi_t)
            if len(expression_texture.shape) == 4:
                if expression_texture.shape[0] != 1:
                    raise ValueError("PhysicalBufferRenderer renders one expression bank sample per call")
                expression_texture = expression_texture[0]
            texture = expression_texture
        if len(texture.shape) == 4:
            raise ValueError("albedo_uv must be one HxWx3 atlas")
        if tuple(texture.shape[-1:]) != (3,):
            raise ValueError("albedo_uv must be HxWx3 BGR")
        if tuple(texture.shape[:2]) != tuple(albedo_uv.shape[:2]):
            if _is_tensor(texture):
                torch = _torch_module()
                import torch.nn.functional as functional
                texture = functional.interpolate(texture.permute(2, 0, 1)[None],
                    size=tuple(albedo_uv.shape[:2]), mode="bilinear", align_corners=True)[0].permute(1, 2, 0)
            else:
                texture = cv2.resize(np.asarray(texture, np.float32),
                                     (int(albedo_uv.shape[1]), int(albedo_uv.shape[0])),
                                     interpolation=cv2.INTER_LINEAR)
        if _is_tensor(texture):
            physical = self.physical_renderer.render_torch(
                vertices, faces, uvs, texture, uv_confidence, pose, inverse, sh_coefficients,
                side=side, specular_strength=specular_strength, shininess=shininess,
                projected_pixels=projected_pixels, camera_vertices_override=camera_vertices_override)
        else:
            texture = np.asarray(texture)
            if expression_bank is not None:
                expression_texture = np.asarray(texture, np.float32)
                texture = np.rint(np.clip(expression_texture, 0, 1) * 255).astype(np.uint8)
            physical = self.physical_renderer.render(
                vertices, faces, uvs, texture, uv_confidence, pose, inverse, sh_coefficients,
                side=side, specular_strength=specular_strength, shininess=shininess,
                projected_pixels=projected_pixels, camera_vertices_override=camera_vertices_override)
        splat_layer = None
        linear = physical.sh_shaded.clone() if _is_tensor(physical.sh_shaded) else physical.sh_shaded.copy()
        if boundary_splats is not None:
            if camera_matrix is None:
                raise ValueError("camera_matrix is required to project boundary splats")
            splat_layer = self.splat_renderer.render(boundary_splats, camera_matrix, physical.z_depth,
                                                    device=splat_device)
            if _is_tensor(linear):
                alpha = splat_layer.alpha * physical.coverage
                linear = linear * (1 - alpha[..., None]) + splat_layer.color_bgr * alpha[..., None]
            else:
                alpha = np.asarray(splat_layer.alpha, np.float32) * np.asarray(physical.coverage, np.float32)
                color = np.asarray(splat_layer.color_bgr, np.float32)
                linear = np.asarray(linear, np.float32) * (1 - alpha[..., None]) + color * alpha[..., None]
        if not _is_tensor(linear):
            linear = linear.astype(np.float32)
        return HybridRenderResult(physical, splat_layer, linear, physical.coverage, "hybrid_mesh_gaussian")

    def release(self) -> None:
        self.physical_renderer.release()


__all__ = [
    "BoundaryGaussianSplatRenderer", "BoundarySplats", "CanonicalAppearance",
    "FastUVFallbackRenderer", "HybridMeshGaussianRenderer", "HybridRenderResult",
    "MultiViewCanonicalAggregator", "ReducedBlendshapeBank", "SplatLayer",
]
