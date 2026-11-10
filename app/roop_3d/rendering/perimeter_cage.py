# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Perimeter cage warp (Stage 6B): make the target's hair, ears and neck hug a different skull.

Swapping the whole head geometry changes the silhouette. Along the hairline, temples, ear attachments and neck base the new
perimeter is somewhere else than the old one; ``DeltaP = Project(V_source_perimeter) - Project(V_target_perimeter)`` is how far
the content there has to travel. This module turns those sparse displacements into a dense, smooth, fold-free warp.

Method (a Gaussian-RBF analogue of the ROME Laplacian cage, evaluated in 2D):

* the displacement is interpolated from the perimeter points with a regularised Gaussian RBF (a thin-plate-like smooth field with no
  extrapolation spikes);
* it is multiplied by an exponential falloff in the distance to the perimeter's convex hull, so the field is 1 at and inside the
  perimeter and decays away from it - the rest of the frame is not dragged along;
* it is limited per point (``max_displacement_fraction`` of the head extent) and then scaled down globally, by bisection, until the
  Jacobian determinant of the sampling map stays above ``min_jacobian`` everywhere - the same no-fold guarantee ``rbf_deform`` gives
  the 3D mesh. A field that cannot be made fold-free is dropped (identity), never applied broken.

The warp is a BACKWARD map: ``out(x) = in(x - DeltaP(x))``, so content at the target perimeter lands on the source perimeter.

Inputs are plain pixel arrays, so any source of correspondences works: matching vertex ids of the source- and target-shaped meshes
(``mesh_perimeter_pairs``, topology-identical heads such as ``ArticulatedHeadLayer`` with ``beta_source`` / ``beta_target``) or,
when no mesh is at hand, the two silhouettes themselves (``perimeter_from_masks``, matched along rays from a common centre).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class CageOptions:
    sigma_fraction: float = 0.30            # RBF kernel width, fraction of the perimeter's extent
    falloff_fraction: float = 0.18          # exponential falloff length outside the perimeter hull, fraction of the extent
    regularisation: float = 1e-2
    max_displacement_fraction: float = 0.25
    min_jacobian: float = 0.25
    min_motion_px: float = 0.75             # below this the field is the identity and nothing is warped
    min_points: int = 6
    grid_max_side: int = 160                # the field is solved on a grid of at most this size, then upsampled

    def __post_init__(self) -> None:
        if not (0.02 <= self.sigma_fraction <= 2 and 0.01 <= self.falloff_fraction <= 2 and self.regularisation >= 0
                and 0 < self.max_displacement_fraction <= 1 and 0 < self.min_jacobian < 1 and self.min_points >= 3
                and self.grid_max_side >= 16 and self.min_motion_px >= 0):
            raise ValueError("invalid cage options")


@dataclass(frozen=True)
class CageReport:
    points: int = 0
    identity: bool = True
    reason: str = ""
    max_displacement_px: float = 0.0
    control_residual_px: float = 0.0        # mean |field(control point) - requested| after clamping and fold relief
    fold_scale: float = 1.0                 # global share of the field kept to stay fold-free (1 = untouched)
    min_jacobian: float = 1.0
    extent_px: float = 0.0


@dataclass(frozen=True)
class CageField:
    dx: np.ndarray                          # (H, W) float32, out(x) = in(x + d(x))
    dy: np.ndarray
    report: CageReport

    @property
    def identity(self) -> bool:
        return self.report.identity


def _identity(shape, report: CageReport) -> CageField:
    zeros = np.zeros(shape, np.float32)
    return CageField(zeros, zeros.copy(), report)


class PerimeterCageWarper:
    def __init__(self, options: CageOptions | None = None) -> None:
        self.options = options or CageOptions()

    def estimate(self, source_xy, target_xy, shape: tuple[int, int], *, weights=None) -> CageField:
        """Dense field moving content at ``target_xy`` onto ``source_xy`` (both (N, 2) pixels of an ``shape`` = (H, W) image)."""
        opt = self.options
        src = np.asarray(source_xy, np.float64)
        dst = np.asarray(target_xy, np.float64)
        if src.ndim != 2 or src.shape[1] != 2 or src.shape != dst.shape or not (np.isfinite(src).all() and np.isfinite(dst).all()):
            raise DomainError("CAGE_POINTS_INVALID", "Perimeter points must be matching finite (N, 2) arrays", 422)
        h, w = int(shape[0]), int(shape[1])
        weight = np.ones(len(src)) if weights is None else np.asarray(weights, np.float64).reshape(-1)
        if weight.shape != (len(src),) or not np.isfinite(weight).all() or (weight < 0).any():
            raise DomainError("CAGE_POINTS_INVALID", "Weights must be one finite non-negative value per point", 422)
        keep = weight > 1e-6
        src, dst, weight = src[keep], dst[keep], weight[keep]
        if len(src) < opt.min_points:
            return _identity((h, w), CageReport(points=len(src), reason="TOO_FEW_PERIMETER_POINTS"))
        everything = np.vstack((src, dst))
        extent = float(np.hypot(*np.ptp(everything, axis=0)))
        if extent < 4.0:
            return _identity((h, w), CageReport(points=len(src), reason="DEGENERATE_PERIMETER"))
        delta = dst - src                                   # backward sampling offset = -DeltaP
        cap = opt.max_displacement_fraction * extent
        norm = np.linalg.norm(delta, axis=1, keepdims=True)
        delta = delta * np.minimum(1.0, cap / np.maximum(norm, 1e-9))
        if float(np.linalg.norm(delta, axis=1).max()) < opt.min_motion_px:
            return _identity((h, w), CageReport(points=len(src), reason="SILHOUETTES_ALREADY_MATCH", extent_px=extent))

        sigma = max(opt.sigma_fraction * extent, 1.0)
        d2 = ((src[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        kernel = np.exp(-d2 / (2 * sigma * sigma)) + np.diag(opt.regularisation / np.maximum(weight, 1e-3))
        try:
            coefficients = np.linalg.solve(kernel, delta)
        except np.linalg.LinAlgError:
            return _identity((h, w), CageReport(points=len(src), reason="CAGE_SINGULAR", extent_px=extent))

        cell = max(1.0, max(h, w) / opt.grid_max_side)
        gh, gw = max(2, int(np.ceil(h / cell))), max(2, int(np.ceil(w / cell)))
        gx = (np.arange(gw) + 0.5) * (w / gw)
        gy = (np.arange(gh) + 0.5) * (h / gh)
        grid = np.stack(np.meshgrid(gx, gy), -1).reshape(-1, 2)
        field = (np.exp(-((grid[:, None, :] - src[None]) ** 2).sum(-1) / (2 * sigma * sigma)) @ coefficients)
        hull = cv2.convexHull(everything.astype(np.float32))
        inside = np.zeros((gh, gw), np.uint8)
        cv2.fillConvexPoly(inside, np.round(hull[:, 0, :] / np.array([w / gw, h / gh])).astype(np.int32), 1)
        outside_distance = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 3) * (w / gw)
        falloff = np.exp(-outside_distance / max(opt.falloff_fraction * extent, 1.0)).reshape(-1, 1)
        field = (field * falloff).reshape(gh, gw, 2)

        scale, min_det = self._fold_free_scale(field, w / gw, h / gh)
        if scale <= 0.0:
            return _identity((h, w), CageReport(points=len(src), reason="CAGE_WOULD_FOLD", extent_px=extent, min_jacobian=min_det))
        field = field * scale
        at_points = self._sample(field, src, w / gw, h / gh)
        residual = float(np.linalg.norm(at_points - delta, axis=1).mean())
        dense = cv2.resize(field.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        report = CageReport(points=len(src), identity=False, reason="", max_displacement_px=float(np.linalg.norm(dense, axis=2).max()),
                            control_residual_px=residual, fold_scale=float(scale), min_jacobian=float(min_det), extent_px=extent)
        return CageField(np.ascontiguousarray(dense[..., 0]), np.ascontiguousarray(dense[..., 1]), report)

    @staticmethod
    def _sample(field: np.ndarray, points: np.ndarray, cw: float, ch: float) -> np.ndarray:
        gh, gw = field.shape[:2]
        u = np.clip(points[:, 0] / cw - 0.5, 0, gw - 1.001)
        v = np.clip(points[:, 1] / ch - 0.5, 0, gh - 1.001)
        x0, y0 = u.astype(int), v.astype(int)
        fx, fy = (u - x0)[:, None], (v - y0)[:, None]
        return ((1 - fx) * (1 - fy) * field[y0, x0] + fx * (1 - fy) * field[y0, x0 + 1]
                + (1 - fx) * fy * field[y0 + 1, x0] + fx * fy * field[y0 + 1, x0 + 1])

    def _fold_free_scale(self, field: np.ndarray, cw: float, ch: float) -> tuple[float, float]:
        """Largest global scale s in (0, 1] with min det(I + s * d(field)) >= min_jacobian (bisection)."""
        dxdx = np.gradient(field[..., 0], cw, axis=1)
        dxdy = np.gradient(field[..., 0], ch, axis=0)
        dydx = np.gradient(field[..., 1], cw, axis=1)
        dydy = np.gradient(field[..., 1], ch, axis=0)

        def min_det(s: float) -> float:
            return float(((1 + s * dxdx) * (1 + s * dydy) - (s * dxdy) * (s * dydx)).min())

        target = self.options.min_jacobian
        if min_det(1.0) >= target:
            return 1.0, min_det(1.0)
        if min_det(1e-3) < target:
            return 0.0, min_det(1e-3)
        lo, hi = 1e-3, 1.0
        for _ in range(12):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if min_det(mid) >= target else (lo, mid)
        return lo, min_det(lo)

    def warp(self, field: CageField, image: np.ndarray, *, border: int = cv2.BORDER_REPLICATE) -> np.ndarray:
        """Resample ``image`` (uint8 / float, 2D or 3D) through the field; identity fields return a copy."""
        array = np.asarray(image)
        if array.shape[:2] != field.dx.shape:
            raise DomainError("CAGE_IMAGE_INVALID", "Image must match the field's size", 422)
        if field.identity:
            return array.copy()
        h, w = field.dx.shape
        xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        return cv2.remap(array, xs + field.dx, ys + field.dy, cv2.INTER_LINEAR, borderMode=border)


# =========================================================================== correspondences
def perimeter_vertex_ids(weights: np.ndarray, faces: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Vertices of the swap region (weight > threshold) that touch a vertex outside it: the region's boundary ring.

    ``weights`` is ``SwapMask3D.weights`` (the full craniofacial union by default), so the ring follows the hairline, temples,
    jawline, chin underside and neck transition of the mask rather than an arbitrary landmark polyline."""
    w = np.asarray(weights, np.float64).reshape(-1)
    tri = np.asarray(faces, np.int64)
    if tri.ndim != 2 or tri.shape[1] != 3 or (tri.size and (tri.min() < 0 or tri.max() >= len(w))):
        raise DomainError("CAGE_MESH_INVALID", "Faces must be (F, 3) indices into the weights", 422)
    inside = w > threshold
    mixed = inside[tri].any(axis=1) & ~inside[tri].all(axis=1)
    ring = np.zeros(len(w), bool)
    ring[tri[mixed].ravel()] = True
    return np.nonzero(ring & inside)[0]


def mesh_perimeter_pairs(ids: np.ndarray, source_pixels: np.ndarray, target_pixels: np.ndarray, *, visible=None,
                         max_points: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """``(Project(V_source_perimeter), Project(V_target_perimeter))`` for the same vertex ids of two topology-identical heads.

    ``visible`` (V,) bool drops vertices hidden from the camera in either head (their projection is not an outline)."""
    ids = np.asarray(ids, np.int64).reshape(-1)
    s, t = np.asarray(source_pixels, np.float64), np.asarray(target_pixels, np.float64)
    if s.shape != t.shape or s.ndim != 2 or s.shape[1] != 2 or (ids.size and (ids.min() < 0 or ids.max() >= len(s))):
        raise DomainError("CAGE_POINTS_INVALID", "Pixel arrays must be matching (V, 2) and ids inside them", 422)
    if visible is not None:
        keep = np.asarray(visible, bool).reshape(-1)[ids]
        ids = ids[keep]
    if len(ids) > max_points:
        ids = ids[np.linspace(0, len(ids) - 1, max_points).astype(int)]
    return s[ids], t[ids]


def perimeter_from_masks(source_mask, target_mask, *, rays: int = 48, center=None) -> tuple[np.ndarray, np.ndarray]:
    """Silhouette-only correspondences: along ``rays`` rays from a common centre, the outermost source-mask pixel and the
    outermost target-mask pixel. Needs both outlines star-shaped about the centre (true for a head without a hat brim);
    rays where either mask is empty are dropped. Returns (source_xy, target_xy)."""
    src = np.asarray(source_mask, np.float32) > 0.5
    tgt = np.asarray(target_mask, np.float32) > 0.5
    if src.shape != tgt.shape or src.ndim != 2:
        raise DomainError("CAGE_MASK_INVALID", "Masks must be matching 2D planes", 422)
    if not (src.any() and tgt.any()):
        return np.zeros((0, 2)), np.zeros((0, 2))
    if center is None:
        ys, xs = np.nonzero(tgt)
        center = (float(xs.mean()), float(ys.mean()))
    h, w = src.shape
    reach = float(np.hypot(h, w))
    radii = np.arange(0.0, reach, 0.5)
    angles = np.linspace(0, 2 * np.pi, rays, endpoint=False)
    out_s, out_t = [], []
    for theta in angles:
        px = np.round(center[0] + radii * np.cos(theta)).astype(int)
        py = np.round(center[1] + radii * np.sin(theta)).astype(int)
        ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        px, py, r = px[ok], py[ok], radii[ok]
        hit_s, hit_t = np.nonzero(src[py, px])[0], np.nonzero(tgt[py, px])[0]
        if not len(hit_s) or not len(hit_t):
            continue
        for hits, sink in ((hit_s, out_s), (hit_t, out_t)):
            i = hits[-1]
            sink.append((center[0] + r[i] * np.cos(theta), center[1] + r[i] * np.sin(theta)))
    return np.asarray(out_s, np.float64).reshape(-1, 2), np.asarray(out_t, np.float64).reshape(-1, 2)
