"""Canonical 68-point landmark correspondence on the MakeHuman (CC0) head template.

The MakeHuman asset ships only the five detector landmarks (``mesh.landmark_vertices["5pt"]``). The standard 68-point
layout (iBUG order: jaw 0-16, brows 17-26, nose 27-35, eyes 36-47, mouth 48-67) has no hand-annotated vertex set on this
mesh, so it is DERIVED, deterministically, from the anthropometric 68-point model the rest of the pipeline already uses:

1. a proper-rotation similarity transform (Umeyama) registers the canonical 5-point subset onto the template's five
   landmark vertices (both frames are +x image-right, +y down, +z toward the viewer);
2. every canonical landmark is carried through that transform and snapped to the nearest template vertex that faces the
   viewer (outward normal not pointing away), which makes the index a real vertex of the deformable surface.

This is a prior, not an annotation: the registration residual on the five anchors (about 5 mm) and the snap distance (a
few mm) are both reported by :func:`correspondence_report` so a caller never has to trust it blindly.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.multiview_fit import five_from_68
from roop_3d.reconstruction.tps_fallback import get_canonical_3d_landmarks

# iBUG 68 groups, used for per-region weighting and reporting.
REGIONS_68: dict[str, tuple[int, int]] = {
    "jaw": (0, 17), "brows": (17, 27), "nose": (27, 36), "eyes": (36, 48), "mouth": (48, 68)}


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (scale, proper rotation, translation) minimising |s R src + t - dst|."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    s, d = src - mu_s, dst - mu_d
    u, sing, vt = np.linalg.svd(d.T @ s)
    flip = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        flip[2, 2] = -1.0
    rot = u @ flip @ vt
    scale = float((sing * np.diag(flip)).sum() / (s ** 2).sum())
    return scale, rot, mu_d - scale * rot @ mu_s


def _derive(vertices: np.ndarray, normals: np.ndarray, five: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    canon68 = get_canonical_3d_landmarks(68).astype(np.float64)
    scale, rot, trans = _umeyama(five_from_68(canon68), five)
    carried = scale * canon68 @ rot.T + trans
    anchors = scale * five_from_68(canon68) @ rot.T + trans
    facing = normals[:, 2] > -0.3                      # the viewer-facing hemisphere (jaw ends sit on the sides)
    indices = np.empty(68, np.int64)
    snap = np.empty(68, np.float64)
    for k, point in enumerate(carried):
        dist = np.where(facing, np.linalg.norm(vertices - point, axis=1), np.inf)
        indices[k] = int(np.argmin(dist))
        snap[k] = float(dist[indices[k]])
    report = {"anchor_rms_mm": float(1000 * np.sqrt(((anchors - five) ** 2).sum(axis=1).mean())),
              "scale": scale, "snap_mean_mm": float(1000 * snap.mean()), "snap_max_mm": float(1000 * snap.max())}
    return indices, report


@lru_cache(maxsize=4)
def _cached(digest: bytes, vertices: bytes, normals: bytes, five: bytes, n: int) -> tuple[np.ndarray, dict[str, float]]:
    v = np.frombuffer(vertices, np.float64).reshape(n, 3)
    nrm = np.frombuffer(normals, np.float64).reshape(n, 3)
    return _derive(v, nrm, np.frombuffer(five, np.float64).reshape(5, 3))


def landmark_68_vertices(mesh: Mesh3D) -> np.ndarray:
    """Template vertex index (into ``mesh.vertices``) of each of the 68 canonical landmarks (deterministic, cached)."""
    return correspondence(mesh)[0].copy()


def correspondence(mesh: Mesh3D) -> tuple[np.ndarray, dict[str, float]]:
    five_idx = np.asarray(mesh.landmark_vertices["5pt"], np.int64)
    v = np.ascontiguousarray(mesh.vertices, np.float64)
    nrm = np.ascontiguousarray(mesh.normals, np.float64)
    five = np.ascontiguousarray(v[five_idx])
    return _cached(b"", v.tobytes(), nrm.tobytes(), five.tobytes(), len(v))


def correspondence_report(mesh: Mesh3D) -> dict[str, float]:
    return dict(correspondence(mesh)[1])
