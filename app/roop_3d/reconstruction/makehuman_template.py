"""The MakeHuman (CC0) head as an identity-asset template: a drop-in alternative to the canonical Delaunay template.

Produces a ``Mesh3D`` in the identity-asset mesh frame (metres; +x image-right, +y down, +z toward the viewer, outward
CCW normals), with per-vertex UVs (vertices are split at UV seams) normalised to fill the atlas, and the five detector
landmark vertices exposed as ``mesh.landmark_vertices["5pt"]`` in detector order (image-left eye, image-right eye, nose
tip, image-left mouth corner, image-right mouth corner). There is no 68-point correspondence for this template.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.geometry_types import Mesh3D
from roop_3d.reconstruction.makehuman_build import ASSET_NAME, BUILD_VERSION, COMMIT, DEFAULT_ASSET_DIR
from roop_3d.reconstruction.makehuman_head import MakeHumanHeadModel

TEMPLATE_KIND = "makehuman"
TEMPLATE_VERSION = f"makehuman-cc0-head-v{BUILD_VERSION}@{COMMIT[:7]}"


UV_PAD = 0.008  # atlas padding between islands (about 4 texels at 512), so dilation / bilinear taps never bleed across islands


def _uv_islands(faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Island id per vertex (vertices are already split at seams, so shared vertices <=> same island)."""
    parent = np.arange(n_vertices)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for tri in faces:
        for k in (1, 2):
            ra, rb = find(int(tri[0])), find(int(tri[k]))
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(n_vertices)])


def _repack_uv_islands(faces: np.ndarray, uvs: np.ndarray) -> np.ndarray:
    """MakeHuman's UVs address a whole-body atlas, so the cropped head covers only ~22 % of it. Repack the islands (uniform
    scale, no rotation, so island shape and texel aspect are preserved): the largest island on the left, the rest in columns
    beside it, at the largest scale that fits [0, 1]^2 with padding."""
    ids = _uv_islands(faces, len(uvs))
    roots = np.unique(ids)
    boxes = []
    for r in roots:
        pts = uvs[ids == r]
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        boxes.append((float(np.prod(hi - lo)), r, lo, hi - lo))
    boxes.sort(key=lambda b: (-b[0], int(b[1])))
    main, rest = boxes[0], sorted(boxes[1:], key=lambda b: (-b[3][1], int(b[1])))

    def layout(s: float):
        placed = {int(main[1]): np.array([UV_PAD, UV_PAD])}
        x0 = UV_PAD + main[3][0] * s + UV_PAD
        x, y, col_w = x0, UV_PAD, 0.0
        for _, r, _, size in rest:
            w, h = size * s
            if y + h > 1.0 - UV_PAD and y > UV_PAD:   # next column
                x, y, col_w = x + col_w + UV_PAD, UV_PAD, 0.0
            placed[int(r)] = np.array([x, y])
            y += h + UV_PAD
            col_w = max(col_w, w)
        right = max([UV_PAD + main[3][0] * s] + [placed[int(r)][0] + size[0] * s for _, r, _, size in rest])
        bottom = max([UV_PAD + main[3][1] * s] + [placed[int(r)][1] + size[1] * s for _, r, _, size in rest])
        return (placed, right, bottom) if right <= 1.0 - UV_PAD and bottom <= 1.0 - UV_PAD else None

    lo_s, hi_s = 0.0, 4.0
    for _ in range(40):
        mid = 0.5 * (lo_s + hi_s)
        lo_s, hi_s = (mid, hi_s) if layout(mid) is not None else (lo_s, mid)
    s = lo_s
    placed, _, _ = layout(s)
    out = np.empty_like(uvs)
    for _, r, lo, _ in boxes:
        sel = ids == r
        out[sel] = (uvs[sel] - lo) * s + placed[int(r)]
    return out


def _arrays(path: str) -> dict[str, np.ndarray]:
    model = MakeHumanHeadModel.load(path)  # verifies the recorded asset digest
    base = model.base_mesh
    verts = base.vertices.astype(np.float64) * np.array([1.0, 1.0, -1.0])   # head frame (z away) -> viewer frame (z toward)
    faces, face_uvs = base.faces.astype(np.int64), base.face_uvs.astype(np.int64)
    # The z mirror reverses handedness: restore CCW-outward winding (and make sure, whatever the source winding was).
    tri = verts[faces]
    outward = float(np.einsum("fc,fc->f", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), tri.mean(axis=1) - verts.mean(axis=0)).sum())
    if outward < 0:
        faces, face_uvs = faces[:, [0, 2, 1]], face_uvs[:, [0, 2, 1]]

    # Split vertices at UV seams so every vertex has exactly one UV (the atlas code addresses uvs[vertex]).
    pairs = np.stack([faces.ravel(), face_uvs.ravel()], axis=1)
    unique, inverse = np.unique(pairs, axis=0, return_inverse=True)
    new_faces = inverse.reshape(-1, 3).astype(np.int32)
    new_verts = verts[unique[:, 0]]
    uvs = base.uvs.astype(np.float64)[unique[:, 1]]

    uvs = _repack_uv_islands(new_faces, uvs)

    first_copy = {int(orig): i for i, orig in reversed(list(enumerate(unique[:, 0])))}   # lowest new index per original vertex
    landmarks = np.array([first_copy[int(i)] for i in model.landmark_indices], dtype=np.int64)
    return {"vertices": new_verts, "faces": new_faces, "uvs": uvs, "landmarks": landmarks}


@lru_cache(maxsize=2)
def _cached(path: str, mtime_ns: int) -> dict[str, np.ndarray]:  # mtime keys the cache so a rebuilt asset is picked up
    return _arrays(path)


def makehuman_template(path: Path | str | None = None) -> Mesh3D:
    """The MakeHuman head as a ``Mesh3D`` template (a fresh copy on every call)."""
    target = Path(path) if path else DEFAULT_ASSET_DIR / ASSET_NAME
    if not target.exists():
        raise DomainError(
            "MAKEHUMAN_ASSET_MISSING",
            f"The MakeHuman head model is not built ({target}). Build it once: python -m roop_3d.reconstruction.makehuman_build "
            "(downloads ~130 small CC0 files from a pinned commit, SHA-256 verified)", 404)
    a = _cached(str(target.resolve()), target.stat().st_mtime_ns)
    mesh = Mesh3D(a["vertices"].copy(), a["faces"].copy(), uvs=a["uvs"].copy())
    mesh.landmark_vertices = {"5pt": a["landmarks"].copy()}      # consumed by multiview_fit.template_landmarks / landmark_normals
    mesh.template_kind = TEMPLATE_KIND
    return mesh


def template_hash(path: Path | str | None = None) -> str:
    """Content hash of the template geometry, topology and UVs (part of the identity-asset build key)."""
    mesh = makehuman_template(path)
    h = hashlib.sha256(TEMPLATE_VERSION.encode())
    for arr in (mesh.vertices, mesh.faces, mesh.uvs, mesh.landmark_vertices["5pt"]):
        h.update(str(arr.dtype).encode() + str(arr.shape).encode() + np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()
