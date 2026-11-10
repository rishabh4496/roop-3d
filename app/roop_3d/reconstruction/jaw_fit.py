# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Per-frame jaw-width fit of the canonical proxy head used by ``PoseBlendProjector`` (opt-in).

The proxy is a generic 76-vertex head (vertices 0-67 are the iBUG landmarks). Its footprint is fitted to the target by five
points, so its jaw outline is the *generic* jaw: on a target with a wider or narrower jaw the swap either spills past the real
jaw or leaves a rim of the target's own jaw around it. This module measures that mismatch and moves the proxy's jaw contour
(``rendering.swap_region_3d.adapt_jaw``, a biharmonic Laplacian field on the proxy mesh) onto the target's.

Measurement (reference-free and pose-aware): the proxy's jaw is projected through the SAME fitted camera as the rest of the
proxy, so foreshortening cancels; the target's outline is the detector's 106-point face contour. Both are compared as lateral
width at matching heights of the lower face, and the median ratio is the jaw width ratio. No population mean is assumed.

Honest limits: the 106 contour and the iBUG jaw are two annotations of the same outline, not identical definitions, so a small
systematic bias is possible (hence a dead band and a clamp); the comparison is only made on near-frontal faces; and this was
validated on synthetic outlines and the canonical mesh, not on measured real-footage seams.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.rendering.swap_region_3d import SwapRegionConfig, WeldedGraph, adapt_jaw, jaw_target_from_ratios

JAW_POINTS = 17
CONTOUR_POINTS = 33
MIN_LEVELS = 4
DEAD_BAND = 0.03              # below +-3 % the proxy jaw is left alone (annotation bias and landmark noise live here)
RATIO_LIMITS = (0.85, 1.18)   # a measurement outside this is more likely a bad contour than a real jaw
MIN_FORESHORTENING = 0.75     # cos(yaw) of the PnP fit: beyond ~41 degrees one side of the outline is not the jaw
CACHE_STEP = 0.01             # adaptations are cached per 1 % of width ratio
INNER_FACE_LANDMARKS = tuple(range(17, 68))   # brows, nose, eyes, mouth: held still, because PnP fits the camera to them


@dataclass(frozen=True)
class JawFitReport:
    applied: bool
    reason: str                       # adapted | within_dead_band | contour_invalid | too_few_levels | pose_too_oblique | adaptation_failed
    measured_ratio: float | None      # target jaw width / proxy jaw width at matching heights (before dead band and clamp)
    ratio_used: float | None
    levels: int = 0
    folded_triangles: int = 0


def _chain_widths(points: np.ndarray, levels: np.ndarray) -> np.ndarray:
    """Lateral extent of a polyline at each horizontal level (NaN where it does not cross the level at least twice)."""
    a, b = points[:-1], points[1:]
    out = np.full(len(levels), np.nan)
    for k, y in enumerate(levels):
        lo, hi = np.minimum(a[:, 1], b[:, 1]), np.maximum(a[:, 1], b[:, 1])
        hit = (lo <= y) & (y < hi)
        if hit.sum() < 2:
            continue
        t = (y - a[hit, 1]) / (b[hit, 1] - a[hit, 1])
        x = a[hit, 0] + t * (b[hit, 0] - a[hit, 0])
        out[k] = x.max() - x.min()
    return out


def measure_jaw_ratio(proxy_jaw: np.ndarray, contour: np.ndarray, *, levels: int = 9) -> tuple[float | None, int, str]:
    """(ratio, levels used, reason). ``proxy_jaw`` (17, 2) and ``contour`` (33, 2) are in the same image coordinates."""
    proxy = np.asarray(proxy_jaw, np.float64)
    outline = np.asarray(contour, np.float64)
    if (proxy.shape != (JAW_POINTS, 2) or outline.shape != (CONTOUR_POINTS, 2) or not np.isfinite(proxy).all()
            or not np.isfinite(outline).all()):
        return None, 0, "contour_invalid"
    top = 0.5 * (proxy[0, 1] + proxy[-1, 1])
    chin = float(proxy[:, 1].max())
    if chin - top < 8.0 or np.ptp(outline[:, 1]) < 8.0 or np.ptp(outline[:, 0]) < 8.0:
        return None, 0, "contour_invalid"
    ys = top + (chin - top) * np.linspace(0.2, 0.85, levels)       # the lower face: past the ear, short of the chin tip
    ratio = _chain_widths(outline, ys) / _chain_widths(proxy, ys)
    ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
    if len(ratio) < MIN_LEVELS:
        return None, len(ratio), "too_few_levels"
    return float(np.median(ratio)), len(ratio), "ok"


class ProxyJawAdapter:
    """Adapts a proxy head's jaw to a measured width ratio; one instance per proxy mesh, results cached per 1 %."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        self.vertices = np.asarray(vertices, np.float64)
        self.faces = np.asarray(faces, np.int64)
        self.config = SwapRegionConfig(include_jawline=True)
        self._graph = WeldedGraph(self.vertices, self.faces)
        self._landmarks = np.arange(68)
        self._cache: dict[int, tuple[np.ndarray, int]] = {}

    def adapted(self, ratio: float) -> tuple[np.ndarray, int]:
        """(vertices in the proxy's own frame, folded triangle count) for a jaw width ratio."""
        key = int(round(ratio / CACHE_STEP))
        hit = self._cache.get(key)
        if hit is None:
            width = key * CACHE_STEP
            target = jaw_target_from_ratios(self.vertices, self._landmarks, width_ratio=width, config=self.config)
            result = adapt_jaw(self.vertices, self.faces, self._landmarks, target, self.config, graph=self._graph,
                               pinned=INNER_FACE_LANDMARKS)
            hit = self._cache[key] = (result.vertices.astype(np.float64), result.report.folded_triangles)
        return hit


def decide(measured: float | None, levels: int, reason: str, foreshortening: float) -> tuple[float | None, JawFitReport | None]:
    """(ratio to apply or None, report when nothing is applied). Applies the pose gate, dead band and clamp."""
    if measured is None:
        return None, JawFitReport(False, reason, None, None, levels)
    if foreshortening < MIN_FORESHORTENING:
        return None, JawFitReport(False, "pose_too_oblique", measured, None, levels)
    used = float(np.clip(measured, *RATIO_LIMITS))
    if abs(used - 1.0) < DEAD_BAND:
        return None, JawFitReport(False, "within_dead_band", measured, 1.0, levels)
    return used, None
