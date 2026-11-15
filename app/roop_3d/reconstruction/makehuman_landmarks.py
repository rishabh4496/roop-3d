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

from scipy.optimize import linear_sum_assignment

from roop_3d.core.errors import DomainError

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


def eye_apertures(vertices: np.ndarray, faces: np.ndarray) -> list[dict[str, np.ndarray]]:
    """The two eyelid apertures of the MakeHuman head, image-left eye first.

    The template models each eye socket as its own UV island (an orbital cavity) sewn to the face island along the lid
    margin. The vertices of the face island that coincide with that seam are the *anatomical* eyelid margin loop. Returns,
    per eye, ``socket`` (every cavity vertex id), ``rim`` (margin vertex ids ordered around the loop) and ``landmarks`` (the
    six iBUG eye-contour vertex ids: corner, two upper-lid, corner, two lower-lid, in iBUG order for that eye).
    """
    from roop_3d.reconstruction.makehuman_template import _uv_islands
    from roop_3d.reconstruction.rbf_deform import weld_ids

    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    islands = _uv_islands(f, len(v))
    weld = weld_ids(v)
    sizes = {int(i): int((islands == i).sum()) for i in np.unique(islands)}
    main = max(sizes, key=sizes.get)
    # The cavities are the compact islands whose weld group touches the main face island at the eye height.
    candidates = []
    for island, size in sizes.items():
        if island == main or size > 800:
            continue
        members = islands == island
        centre = v[members].mean(axis=0)
        welded = np.isin(weld, np.unique(weld[members]))
        rim = np.flatnonzero(welded & (islands == main))
        if len(rim) >= 8 and abs(centre[0]) > 0.01 and centre[1] < 0.0:      # above the nose tip, off the midline
            candidates.append((float(centre[0]), island, members, rim))
    if len(candidates) != 2:
        raise DomainError("MAKEHUMAN_EYE_APERTURE_NOT_FOUND",
                          f"Expected two orbital-cavity islands sewn to the face island, found {len(candidates)}", 422)
    result = []
    for side, (_, island, members, rim) in enumerate(sorted(candidates, key=lambda item: item[0])):
        points = v[rim]
        centre = points[:, :2].mean(axis=0)
        order = np.argsort(np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0]))
        ring = rim[order]
        xs = v[ring, 0]
        left, right = int(np.argmin(xs)), int(np.argmax(xs))
        chord = v[ring[right], :2] - v[ring[left], :2]
        # corner-to-corner chord splits the loop into two arcs; image y grows downward, so the arc with the
        # smaller mean y is the upper lid
        normal = np.array((chord[1], -chord[0])) / max(float(np.linalg.norm(chord)), 1e-12)
        height = (v[ring, :2] - v[ring[left], :2]) @ normal
        first, second = ring[height < 0], ring[height >= 0]
        upper, lower = (first, second) if v[first, 1].mean() < v[second, 1].mean() else (second, first)
        upper = upper[np.argsort(v[upper, 0])]
        lower = lower[np.argsort(v[lower, 0])]

        def pick(arc: np.ndarray, fraction: float) -> int:
            path = np.vstack((v[ring[left]], v[arc], v[ring[right]]))
            ids = np.concatenate(([ring[left]], arc, [ring[right]]))
            length = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
            return int(ids[int(np.argmin(np.abs(length - fraction * length[-1])))])

        lo, hi = int(ring[left]), int(ring[right])
        if side == 0:      # image-left eye: 36 outer corner, 37-38 upper lid, 39 inner corner, 40-41 lower lid
            marks = [lo, pick(upper, 1 / 3), pick(upper, 2 / 3), hi, pick(lower, 2 / 3), pick(lower, 1 / 3)]
        else:              # image-right eye: 42 inner corner, 43-44 upper lid, 45 outer corner, 46-47 lower lid
            marks = [lo, pick(upper, 1 / 3), pick(upper, 2 / 3), hi, pick(lower, 2 / 3), pick(lower, 1 / 3)]
        result.append({"socket": np.flatnonzero(members).astype(np.int64), "rim": ring.astype(np.int64),
                       "landmarks": np.asarray(marks, np.int64)})
    return result


def _derive(vertices: np.ndarray, normals: np.ndarray, five: np.ndarray,
            faces: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    canon68 = get_canonical_3d_landmarks(68).astype(np.float64)
    scale, rot, trans = _umeyama(five_from_68(canon68), five)
    carried = scale * canon68 @ rot.T + trans
    anchors = scale * five_from_68(canon68) @ rot.T + trans
    facing = normals[:, 2] > -0.3                      # the viewer-facing hemisphere (jaw ends sit on the sides)

    if np.count_nonzero(facing) < 68:
        raise DomainError("MAKEHUMAN_LANDMARK_CORRESPONDENCE_FAILED",
                          "MakeHuman template has fewer than 68 viewer-facing vertices", 422)
    # The anthropometric eye contour lands several millimetres off the mesh's real lid margin, so the twelve eye points
    # are carried to the anatomical aperture loop instead of being snapped to whatever vertex is nearest.
    apertures = eye_apertures(vertices, faces)
    eye_vertices = np.concatenate([item["landmarks"] for item in apertures])
    distances = np.linalg.norm(vertices[None, :, :] - carried[:, None, :], axis=2)
    distances[:, ~facing] = np.inf
    distances[:, eye_vertices] = np.inf                 # eye-contour vertices belong to the eye points only
    distances[36:48] = np.inf
    for slot, vertex in zip(range(36, 48), eye_vertices):
        distances[slot, vertex] = 0.0
    rows, assigned = linear_sum_assignment(distances)
    indices = np.empty(68, np.int64)
    indices[rows] = assigned
    snap = np.linalg.norm(vertices[indices] - carried, axis=1)
    eye_shift = np.linalg.norm(vertices[indices[36:48]] - carried[36:48], axis=1)
    report = {"anchor_rms_mm": float(1000 * np.sqrt(((anchors - five) ** 2).sum(axis=1).mean())),
              "scale": scale, "snap_mean_mm": float(1000 * np.delete(snap, np.s_[36:48]).mean()),
              "snap_max_mm": float(1000 * np.delete(snap, np.s_[36:48]).max()),
              "eye_rim_shift_mean_mm": float(1000 * eye_shift.mean()), "eye_rim_shift_max_mm": float(1000 * eye_shift.max())}
    return indices, report


@lru_cache(maxsize=4)
def _cached(digest: bytes, vertices: bytes, normals: bytes, five: bytes, n: int,
            faces: bytes) -> tuple[np.ndarray, dict[str, float]]:
    v = np.frombuffer(vertices, np.float64).reshape(n, 3)
    nrm = np.frombuffer(normals, np.float64).reshape(n, 3)
    return _derive(v, nrm, np.frombuffer(five, np.float64).reshape(5, 3), np.frombuffer(faces, np.int64).reshape(-1, 3))


def correspondence(mesh: Mesh3D) -> tuple[np.ndarray, dict[str, float]]:
    five_idx = np.asarray(mesh.landmark_vertices["5pt"], np.int64)
    v = np.ascontiguousarray(mesh.vertices, np.float64)
    nrm = np.ascontiguousarray(mesh.normals, np.float64)
    five = np.ascontiguousarray(v[five_idx])
    faces = np.ascontiguousarray(mesh.faces, np.int64)
    indices, report = _cached(b"", v.tobytes(), nrm.tobytes(), five.tobytes(), len(v), faces.tobytes())
    return indices.copy(), dict(report)


def correspondence_report(mesh: Mesh3D) -> dict[str, float]:
    return dict(correspondence(mesh)[1])
