"""Semantic face-shape sliders on top of a linear identity basis (clean-room; no learned or third-party semantic model).

Six intuitive sliders (jaw width, cheekbone prominence, chin projection, nose bridge, brow arch, face fullness) are turned
into shape-coefficient offsets ``delta_beta`` for a topology's linear basis ``V = V0 + S beta``.

Method. Every slider has a *measure*: a fixed linear functional of the surface (distances between 68-point landmarks, ray
probes on the frontal surface, or a mean outward displacement over a skin region), so ``measure(V0 + S beta) = m0 + J beta``
exactly, with ``J = C S`` a small matrix. Three nuisance measures (head width, face height, face depth) are held fixed so a
slider never silently rescales the head (the MakeHuman basis has axes, gender and head scale, that move the whole head by tens
of millimetres per unit). For slider ``j`` the direction is the minimum-norm solution of

    J delta = u_j * e_j          (u_j: the slider's unit in millimetres; every other measure, nuisance included, stays at 0)

i.e. ``delta_j = J^T (J J^T + lambda I)^-1 (u_j e_j)``. Because the model is linear, moving several sliders is just the sum, and
the sliders are *decoupled in measure space* by construction (verified: leakage onto the other measures is reported). They are
not claimed to be orthogonal in coefficient space; the cosine matrix between directions is reported too.

Honest limits: a slider is only as good as the landmark correspondence behind it (derived, not annotated, for both topologies;
see ``makehuman_landmarks`` and the bridge report), and a measure the basis cannot move (reported as ``reachable_fraction < 1``)
is clamped, never faked. Slider units are physical (mm of the measure), not population standard deviations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.mesh_bridge import closest_points_on_mesh
from roop_3d.reconstruction.topology import MorphableTopologyRegistry, TopologyMesh

# Frame: +x image-right, +y DOWN, +z toward the viewer (identity-asset frame). "up" is therefore -y.
UP = np.array([0.0, -1.0, 0.0])
RIGHT = np.array([1.0, 0.0, 0.0])
FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class SemanticSlider:
    key: str
    label: str
    description: str
    unit_mm: float          # change of the measure per slider unit, millimetres
    minimum: float = -3.0
    maximum: float = 3.0

    def public(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label, "description": self.description, "unit_mm": self.unit_mm,
                "minimum": self.minimum, "maximum": self.maximum, "default": 0.0}


SLIDERS: tuple[SemanticSlider, ...] = (
    SemanticSlider("jaw_width", "Jaw width", "Bigonial width: distance between the jaw angles (landmarks 4-12 and 5-11).", 4.0),
    SemanticSlider("cheekbone_prominence", "Cheekbone prominence",
                   "How far the cheek surface projects forward of the eye-corner / mouth-corner plane.", 2.0),
    SemanticSlider("chin_projection", "Chin projection", "How far the chin tip projects forward of the lower lip.", 3.0),
    SemanticSlider("nose_bridge", "Nose bridge", "Height of the nose bridge above the inner eye corners.", 2.0),
    SemanticSlider("brow_arch", "Brow arch", "Rise of each eyebrow's peak above the line joining its two ends.", 1.5),
    SemanticSlider("face_fullness", "Face fullness", "Mean outward displacement of the cheek and jowl skin.", 1.5),
)
SLIDER_KEYS = tuple(s.key for s in SLIDERS)
NUISANCE_KEYS = ("head_width", "face_height", "face_depth")
DEFAULT_COEFF_LIMIT = {"makehuman_cc0": 1.0, "flame2023_open": 3.0}   # MakeHuman sliders are [-1, 1]; FLAME is +-3 sigma


def slider_definitions() -> list[dict[str, Any]]:
    return [s.public() for s in SLIDERS]


# ----------------------------------------------------------------------------------------------------- measure building
def _frontal_weights(vertices: np.ndarray, faces: np.ndarray, xy: np.ndarray) -> sparse.csr_matrix:
    """Barycentric row of the most frontal (largest z) surface point above/below ``xy`` (a ray cast along the view axis)."""
    tri = vertices[faces]
    k = min(32, len(faces))
    _, near = cKDTree(tri[:, :, :2].mean(axis=1)).query(xy, k=k)
    best_z, best = -np.inf, None
    for f in np.atleast_1d(near):
        a, b, c = tri[f]
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-14:
            continue
        w0 = ((b[1] - c[1]) * (xy[0] - c[0]) + (c[0] - b[0]) * (xy[1] - c[1])) / det
        w1 = ((c[1] - a[1]) * (xy[0] - c[0]) + (a[0] - c[0]) * (xy[1] - c[1])) / det
        w = np.array([w0, w1, 1.0 - w0 - w1])
        if (w >= -1e-9).all() and w @ tri[f][:, 2] > best_z:
            best_z, best = float(w @ tri[f][:, 2]), (int(f), w)
    if best is None:                                  # xy falls in a hole: nearest surface point to the front of it
        ids, bary, _ = closest_points_on_mesh(np.array([[xy[0], xy[1], vertices[:, 2].max()]]), vertices, faces)
        best = (int(ids[0]), bary[0])
    row = sparse.csr_matrix((best[1], ([0, 0, 0], faces[best[0]])), shape=(1, len(vertices)))
    return row


class _Builder:
    """Assembles the dense measure matrix ``C`` (one row of length ``3V`` per measure)."""

    def __init__(self, rest: np.ndarray, faces: np.ndarray, landmarks: sparse.csr_matrix):
        self.rest, self.faces, self.lm = rest, faces, landmarks
        self.v = len(rest)

    def pt(self, i: int) -> np.ndarray:
        """Landmark ``i`` as a (V,) vertex weight row."""
        return np.asarray(self.lm[i].todense()).ravel()

    def along(self, weights: np.ndarray, axis: np.ndarray) -> np.ndarray:
        return (weights[:, None] * axis[None, :]).ravel()

    def pos(self, weights: np.ndarray) -> np.ndarray:
        return weights @ self.rest

    def frontal_probe(self, centre_weights: np.ndarray) -> np.ndarray:
        xy = self.pos(centre_weights)[:2]
        return np.asarray(_frontal_weights(self.rest, self.faces, xy).todense()).ravel()

    def build(self) -> dict[str, np.ndarray]:
        pt, along = self.pt, self.along
        def mid(*ids):
            return sum(pt(i) for i in ids) / len(ids)

        c: dict[str, np.ndarray] = {}
        # jaw width: bigonial span, averaged over two landmark pairs
        c["jaw_width"] = 0.5 * (along(pt(12) - pt(4), RIGHT) + along(pt(11) - pt(5), RIGHT))
        # cheekbone prominence: frontal cheek surface point (above the midpoint of outer eye corner, mouth corner and contour)
        # relative to the plane through the eye and mouth corners
        ref_z = along(mid(36, 45, 48, 54), FORWARD)
        cheeks = []
        for eye, mouth, contour in ((36, 48, 1), (45, 54, 15)):
            cheeks.append(self.frontal_probe(0.45 * pt(eye) + 0.35 * pt(mouth) + 0.20 * pt(contour)))
        c["cheekbone_prominence"] = along(0.5 * (cheeks[0] + cheeks[1]), FORWARD) - ref_z
        # chin projection: chin tip against the lower lip, along the view axis
        c["chin_projection"] = along(pt(8) - pt(57), FORWARD)
        # nose bridge: frontal bridge point against the inner eye corners
        bridge = self.frontal_probe(mid(28, 29))
        c["nose_bridge"] = along(bridge - mid(39, 42), FORWARD)
        # brow arch: peak above the mean of the two ends, both brows, measured along "up"
        arch = 0.0
        for outer, peak, inner in ((17, 19, 21), (26, 24, 22)):
            arch = arch + along(pt(peak) - 0.5 * (pt(outer) + pt(inner)), UP)
        c["brow_arch"] = arch / 2.0
        c["face_fullness"] = self._fullness()
        # nuisance measures: overall head size must not move with a slider
        c["head_width"] = along(pt(16) - pt(0), RIGHT)
        c["face_height"] = along(mid(36, 45) - pt(8), UP)
        c["face_depth"] = along(pt(30) - mid(0, 16), FORWARD)
        return c

    def _fullness(self) -> np.ndarray:
        """Mean outward (rest-normal) displacement over frontal cheek/jowl skin: below the eye line, above the chin, outside the nose."""
        pt, rest = self.pt, self.rest
        eye_y = float(self.pos(0.5 * (pt(36) + pt(45)))[1])
        chin_y = float(self.pos(pt(8))[1])
        nose_half = 0.6 * float(self.pos(pt(35))[0] - self.pos(pt(31))[0])
        jaw_half = 0.5 * float(self.pos(pt(12))[0] - self.pos(pt(4))[0])
        normals = TopologyMesh("_", rest, self.faces).normals()
        region = ((rest[:, 1] > eye_y) & (rest[:, 1] < chin_y) & (np.abs(rest[:, 0]) > nose_half)
                  & (np.abs(rest[:, 0]) < jaw_half * 1.25) & (normals[:, 2] > 0.2))
        if region.sum() < 20:
            raise DomainError("SEMANTIC_REGION_EMPTY", "Cannot locate the cheek region on this topology", 422)
        row = np.zeros((self.v, 3))
        row[region] = normals[region] / region.sum()
        return row.ravel()


# ---------------------------------------------------------------------------------------------------------- controller
@dataclass
class SemanticResult:
    delta_beta: np.ndarray
    requested: dict[str, float]
    achieved_mm: dict[str, float]          # change of each slider's measure, millimetres
    leakage_mm: dict[str, float]           # change of the OTHER measures (sliders + nuisance), worst case per slider
    limit_scale: float                     # < 1 when the coefficient limit forced the whole move to shrink
    clamped: bool

    def public(self) -> dict[str, Any]:
        return {"requested": self.requested, "achieved_mm": {k: round(v, 3) for k, v in self.achieved_mm.items()},
                "leakage_mm": {k: round(v, 3) for k, v in self.leakage_mm.items()}, "limit_scale": round(self.limit_scale, 4),
                "clamped": self.clamped, "coefficients_used": int(np.count_nonzero(self.delta_beta)),
                "max_abs_delta_beta": round(float(np.abs(self.delta_beta).max(initial=0.0)), 4)}


class SemanticShapeController:
    def __init__(self, rest_vertices: np.ndarray, faces: np.ndarray, shape_basis: np.ndarray, landmarks68: sparse.csr_matrix, *,
                 coeff_limit: float, ridge: float = 1e-6, topology_id: str = "custom"):
        if shape_basis is None or shape_basis.ndim != 3 or shape_basis.shape[:2] != rest_vertices.shape:
            raise DomainError("SEMANTIC_BASIS_MISSING", "A linear shape basis (V, 3, K) matching the vertices is required", 422)
        if landmarks68.shape[0] < 68:
            raise DomainError("SEMANTIC_LANDMARKS_MISSING", "A 68-point landmark correspondence is required", 422)
        self.topology_id, self.coeff_limit = topology_id, float(coeff_limit)
        self.rest = np.asarray(rest_vertices, np.float64)
        self.basis = np.asarray(shape_basis, np.float32)
        builder = _Builder(self.rest, np.asarray(faces, np.int64), landmarks68.tocsr())
        self._c = builder.build()
        self.keys = SLIDER_KEYS + NUISANCE_KEYS
        cmat = np.stack([self._c[k] for k in self.keys])                                  # (M, 3V)
        self._base = cmat @ self.rest.ravel()                                              # measures of the rest mesh (m)
        flat = self.basis.reshape(-1, self.basis.shape[2]).astype(np.float64)              # (3V, K)
        self._jac = cmat @ flat                                                            # (M, K) metres per unit coefficient
        self._directions = self._solve(ridge)                                              # (6, K) delta_beta per slider unit
        self._cmat, self._flat = cmat, flat

    # ------------------------------------------------------------------ construction
    @classmethod
    def for_topology(cls, registry: MorphableTopologyRegistry, topology_id: str) -> "SemanticShapeController":
        """Controller for a registered topology. Landmarks of the non-default topology are carried across the bridge."""
        default = registry.default_id
        base = registry.prepared(default)
        lm68 = base.landmarks.get("68pt")
        if lm68 is None or len(lm68) != 68:
            raise DomainError("SEMANTIC_LANDMARKS_MISSING", f"{default} has no 68-point correspondence", 422)
        limit = DEFAULT_COEFF_LIMIT.get(topology_id, 1.0)
        if topology_id == default:
            rows = sparse.csr_matrix((np.ones(68), (np.arange(68), lm68)), shape=(68, base.vertex_count))
            return cls(base.vertices, base.faces, base.shape_basis, rows, coeff_limit=limit, topology_id=topology_id)
        bridge = registry.bridge(default, topology_id)
        target = bridge.target
        if target.shape_basis is None:
            raise DomainError("SEMANTIC_BASIS_MISSING", f"{topology_id} has no shape basis", 422)
        basis = bridge.target_vectors_to_shared(target.shape_basis.reshape(-1, 3)).reshape(target.shape_basis.shape).astype(np.float32)
        return cls(bridge.target_in_source, target.faces, basis, bridge.point_weights(lm68), coeff_limit=limit, topology_id=topology_id)

    def _solve(self, ridge: float) -> np.ndarray:
        j = self._jac                                                                      # (M, K)
        gram = j @ j.T
        lam = ridge * float(np.trace(gram)) / len(gram)
        pinv = j.T @ np.linalg.inv(gram + lam * np.eye(len(gram)))                         # (K, M)
        out = np.zeros((len(SLIDERS), j.shape[1]))
        for n, slider in enumerate(SLIDERS):
            target = np.zeros(len(self.keys))
            target[n] = slider.unit_mm / 1000.0
            out[n] = pinv @ target
        return out

    # ------------------------------------------------------------------ use
    def measure(self, vertices: np.ndarray) -> dict[str, float]:
        """Every measure (millimetres) of a mesh in this topology's vertex order and the shared frame."""
        values = self._cmat @ np.asarray(vertices, np.float64).ravel()
        return {k: float(v * 1000.0) for k, v in zip(self.keys, values)}

    def solve(self, values: Mapping[str, float]) -> SemanticResult:
        unknown = sorted(set(values) - set(SLIDER_KEYS))
        if unknown:
            raise DomainError("SEMANTIC_SLIDER_UNKNOWN", f"Unknown semantic slider(s): {', '.join(unknown)}", 422)
        requested = {s.key: float(values.get(s.key, 0.0)) for s in SLIDERS}
        for s in SLIDERS:
            if not (s.minimum <= requested[s.key] <= s.maximum) or not np.isfinite(requested[s.key]):
                raise DomainError("SEMANTIC_SLIDER_RANGE", f"{s.key} must be in [{s.minimum:g}, {s.maximum:g}]", 422)
        delta = np.array([requested[s.key] for s in SLIDERS]) @ self._directions
        peak = float(np.abs(delta).max(initial=0.0))
        scale = min(1.0, self.coeff_limit / peak) if peak > 0 else 1.0
        delta = delta * scale
        achieved = {k: float(v * 1000.0) for k, v in zip(self.keys, self._jac @ delta)}
        leakage = {}
        for n, s in enumerate(SLIDERS):
            if requested[s.key] == 0.0:
                continue
            col = (self._jac @ (self._directions[n] * requested[s.key] * scale)) * 1000.0
            leakage[s.key] = float(np.abs(np.delete(col, n)).max())
        return SemanticResult(delta, requested, {k: achieved[k] for k in SLIDER_KEYS}, leakage, scale, scale < 1.0 - 1e-9)

    def displacement(self, delta_beta: np.ndarray) -> np.ndarray:
        """(V, 3) vertex displacement (shared frame) for a coefficient offset."""
        return np.einsum("vck,k->vc", self.basis, np.asarray(delta_beta, np.float32)).astype(np.float64)

    def apply(self, vertices: np.ndarray, values: Mapping[str, float]) -> tuple[np.ndarray, SemanticResult]:
        result = self.solve(values)
        return np.asarray(vertices, np.float64) + self.displacement(result.delta_beta), result

    # ------------------------------------------------------------------ evidence
    def diagnostics(self) -> dict[str, Any]:
        """How decoupled and how reachable the sliders are on this basis (computed, not assumed)."""
        unit = np.stack([self._jac @ self._directions[n] * 1000.0 for n in range(len(SLIDERS))])      # (6, M) mm per unit
        diag = np.array([unit[n, n] for n in range(len(SLIDERS))])
        off = np.array([np.abs(np.delete(unit[n], n)).max() for n in range(len(SLIDERS))])
        norms = np.linalg.norm(self._directions, axis=1)
        cos = (self._directions @ self._directions.T) / np.maximum(np.outer(norms, norms), 1e-12)
        sing = np.linalg.svd(self._jac, compute_uv=False)
        return {"topology": self.topology_id, "basis_axes": int(self.basis.shape[2]), "coeff_limit": self.coeff_limit,
                "reachable_fraction": {s.key: round(float(diag[n] / s.unit_mm), 4) for n, s in enumerate(SLIDERS)},
                "max_leakage_mm_per_unit": {s.key: round(float(off[n]), 4) for n, s in enumerate(SLIDERS)},
                "coefficient_norm_per_unit": {s.key: round(float(norms[n]), 4) for n, s in enumerate(SLIDERS)},
                "max_offdiag_cosine": round(float(np.abs(cos - np.eye(len(cos))).max()), 4),
                "jacobian_condition": round(float(sing[0] / max(sing[-1], 1e-12)), 2),
                "rest_measures_mm": {k: round(v, 3) for k, v in self.measure(self.rest).items()},
                "frame": "identity-asset frame: +x image-right, +y down, +z toward the viewer"}
