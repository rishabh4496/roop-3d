"""Non-rigid surface deformation of a head template onto fitted 3D landmarks (RBF + Laplacian regularisation).

* **RBF interpolation** of the landmark displacements with a compactly supported Wendland C2 kernel
  ``phi(r) = (1 - r/R)^4 (4 r/R + 1)``. The ridge term is scaled by 1 / confidence, so a landmark the views do not
  constrain pulls the surface less. Compact support means the displacement field decays to exactly zero away from the
  controls: skull, neck and ears keep the template prior instead of being extrapolated.
* **Laplacian smoothing** of the displacement field on the position-welded mesh graph (a uniform umbrella operator).
  Control vertices are re-pinned after smoothing, so landmarks keep their fitted positions while the field between them
  is regularised.
* **Welded vertex normals**: the MakeHuman template is split at UV seams, so plain per-vertex normals crease at every
  seam. Normals are recomputed from area-weighted face normals accumulated per *position*, then copied back.
"""
from __future__ import annotations

import numpy as np

from roop_3d.reconstruction.geometry_types import Mesh3D

WELD_DECIMALS = 6          # metres: coincident seam copies agree to a micrometre


def weld_ids(vertices: np.ndarray) -> np.ndarray:
    """Id per vertex such that vertices at the same position (UV-seam copies) share an id."""
    key = np.round(np.asarray(vertices, np.float64), WELD_DECIMALS)
    _, inverse = np.unique(key, axis=0, return_inverse=True)
    return inverse.reshape(-1).astype(np.int64)


def welded_vertex_normals(vertices: np.ndarray, faces: np.ndarray, weld: np.ndarray | None = None) -> np.ndarray:
    """Smooth unit normals (N, 3) that are identical across UV-seam duplicates."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    weld = weld_ids(v) if weld is None else weld
    face_n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])      # |n| = 2 * area: area weighting for free
    acc = np.zeros((int(weld.max()) + 1, 3))
    for corner in range(3):
        np.add.at(acc, weld[f[:, corner]], face_n)
    length = np.linalg.norm(acc, axis=1, keepdims=True)
    return (acc / np.maximum(length, 1e-12))[weld].astype(np.float32)


def wendland(r: np.ndarray, support: float) -> np.ndarray:
    q = np.clip(np.asarray(r, np.float64) / support, 0.0, 1.0)
    return (1.0 - q) ** 4 * (4.0 * q + 1.0)


def default_support(n_controls: int) -> float:
    """Support radius (m): wide for the five sparse controls, tighter for the dense set."""
    return 0.13 if n_controls <= 8 else 0.07


def laplacian_smooth_field(field: np.ndarray, faces: np.ndarray, weld: np.ndarray, *, iterations: int = 6,
                           strength: float = 0.5, pinned: np.ndarray | None = None) -> np.ndarray:
    """Umbrella-operator smoothing of a per-vertex vector field on the welded graph; pinned vertices keep their value."""
    f = np.asarray(faces, np.int64)
    n = int(weld.max()) + 1
    a = np.concatenate([weld[f[:, 0]], weld[f[:, 1]], weld[f[:, 2]], weld[f[:, 1]], weld[f[:, 2]], weld[f[:, 0]]])
    b = np.concatenate([weld[f[:, 1]], weld[f[:, 2]], weld[f[:, 0]], weld[f[:, 0]], weld[f[:, 1]], weld[f[:, 2]]])
    keep = a != b
    a, b = a[keep], b[keep]
    degree = np.maximum(np.bincount(a, minlength=n).astype(np.float64), 1.0)
    first = np.zeros(n, np.int64)
    first[weld] = np.arange(len(weld))                      # a representative vertex per welded id
    value = np.asarray(field, np.float64)[first]
    pin = np.zeros(n, bool)
    if pinned is not None and len(pinned):
        pin[weld[np.asarray(pinned, np.int64)]] = True
    start = value.copy()
    for _ in range(iterations):
        neigh = np.zeros_like(value)
        np.add.at(neigh, a, value[b])
        value = (1.0 - strength) * value + strength * neigh / degree[:, None]
        value[pin] = start[pin]
    return value[weld]


def deform_template_rbf(
    template: Mesh3D,
    control_template: np.ndarray,
    control_target: np.ndarray,
    control_confidence: np.ndarray,
    control_vertices: np.ndarray | None = None,
    *,
    support: float | None = None,
    ridge: float = 5e-2,
    max_displacement_m: float = 0.03,
    smooth_iterations: int = 12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Carry ``template`` onto the fitted landmarks. Returns (vertices, support in 0..1, applied strength).

    Strength is the mean applied fraction of the displacement over the vertices that move; < 1 means folded front-facing
    triangles were relieved locally.
    """
    ctrl = np.asarray(control_template, np.float64)
    disp = np.asarray(control_target, np.float64) - ctrl
    conf = np.clip(np.asarray(control_confidence, np.float64), 0.0, 1.0)
    radius = default_support(len(ctrl)) if support is None else support
    pair = np.linalg.norm(ctrl[:, None] - ctrl[None], axis=2)
    kmat = wendland(pair, radius)
    coeff = np.linalg.solve(kmat + ridge * np.diag(1.0 / np.maximum(conf, 0.02)), disp * conf[:, None])
    verts = template.vertices.astype(np.float64)
    kv = wendland(np.linalg.norm(verts[:, None] - ctrl[None], axis=2), radius)
    field = kv @ coeff
    weld = weld_ids(verts)
    if smooth_iterations > 0:
        field = laplacian_smooth_field(field, template.faces, weld, iterations=smooth_iterations,
                                       pinned=control_vertices)
    norms = np.linalg.norm(field, axis=1, keepdims=True)
    field = field * np.minimum(1.0, max_displacement_m / np.maximum(norms, 1e-12))
    support_map = (kv * conf[None, :]).max(axis=1)

    base_n = template.compute_face_normals()
    front = base_n[:, 2] > 0.3
    faces = np.asarray(template.faces, np.int64)

    def folded_faces(scale: np.ndarray) -> np.ndarray:
        trial = Mesh3D(verts + scale[:, None] * field, template.faces, uvs=template.uvs)
        return front & (trial.compute_face_normals()[:, 2] < 0.05)

    # Local fold relief: only the vertices of folded front-facing triangles (and, via smoothing, their neighbourhood) are
    # pulled back towards the template, so one badly constrained landmark cannot flatten the whole fitted face.
    scale = np.ones(len(verts))
    for _ in range(14):
        bad = folded_faces(scale)
        if not bad.any():
            break
        hit = np.zeros(len(verts), bool)
        hit[faces[bad].ravel()] = True
        scale[hit] *= 0.7
        scale = laplacian_smooth_field(scale[:, None], faces, weld, iterations=2, strength=0.5)[:, 0]
        scale = np.minimum(scale, 1.0)
    else:
        if folded_faces(scale).any():                       # could not be relieved locally: fall back to a global scale
            global_scale = 1.0
            for _ in range(8):
                global_scale *= 0.6
                if not folded_faces(scale * global_scale).any():
                    break
            else:
                global_scale = 0.0
            scale = scale * global_scale
    moved = np.linalg.norm(field, axis=1) > 1e-6
    strength = float(scale[moved].mean()) if moved.any() else 1.0
    return verts + scale[:, None] * field, np.clip(support_map, 0.0, 1.0), strength
