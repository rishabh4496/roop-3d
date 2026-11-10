"""Barycentric surface bridge between two head topologies (positions, normals, UV albedo, in both directions).

Clean-room implementation of the idea behind BFM->FLAME style conversion: register the two rest surfaces once, then express
every vertex of one topology as a fixed *surface point* (triangle + barycentric weights) on the other. Everything after that is
a sparse matrix product, so converting a deformed mesh costs a few milliseconds:

* positions   ``p_B[k] = sum_i w_ki * p_A[tri_ki,i]  +  d_k * n_A(k)``  (``d_k``: the rest-pose signed offset of B's vertex
              from A's surface, carried along A's interpolated normal so B keeps its own relief)
* normals     barycentric (Phong) blend, renormalised
* UV albedo   each target texel is located on the target surface, projected to the source surface, and the source atlas is
              sampled at that surface point's UV (so UV seams on either side are handled by geometry, never by interpolating
              UVs across a seam)

Honest limits (all reported by :meth:`BarycentricMeshBridge.report`): the registration is a trimmed similarity ICP (plus a
low-order non-rigid TPS when that measurably helps), the correspondence is closest-point, not anatomical, and regions one
topology has and the other lacks are *clamped* to the nearest boundary point and flagged in ``matched`` rather than invented.
Lateral residuals between the rest surfaces are dropped; only the normal offset is carried.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

from roop_3d.core.errors import DomainError
from roop_3d.reconstruction.topology import TopologyMesh

Direction = Literal["target", "source"]   # the topology a conversion produces


# ------------------------------------------------------------------------------------------- closest point on a mesh
def _closest_bary(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Barycentric weights of the point of triangle abc closest to p (Ericson, Real-Time Collision Detection 5.1.5), batched."""
    def dot(x, y):
        return np.einsum("...c,...c->...", x, y)

    ab, ac, ap, bp, cp = b - a, c - a, p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = dot(ab, ap), dot(ac, ap), dot(ab, bp), dot(ac, bp), dot(ab, cp), dot(ac, cp)
    vc, vb, va = d1 * d4 - d3 * d2, d5 * d2 - d1 * d6, d3 * d6 - d5 * d4

    def safe(x):
        return np.where(np.abs(x) < 1e-30, 1e-30, x)

    total = safe(va + vb + vc)
    v_in, w_in = vb / total, vc / total
    w = np.stack([1 - v_in - w_in, v_in, w_in], axis=-1)                               # interior (lowest priority)
    t = (d4 - d3) / safe((d4 - d3) + (d5 - d6))
    w = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], np.stack([0 * t, 1 - t, t], axis=-1), w)   # edge BC
    t = d2 / safe(d2 - d6)
    w = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], np.stack([1 - t, 0 * t, t], axis=-1), w)               # edge AC
    w = np.where(((d6 >= 0) & (d5 <= d6))[..., None], np.array([0.0, 0.0, 1.0]), w)                                       # vertex C
    t = d1 / safe(d1 - d3)
    w = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], np.stack([1 - t, t, 0 * t], axis=-1), w)               # edge AB
    w = np.where(((d3 >= 0) & (d4 <= d3))[..., None], np.array([0.0, 1.0, 0.0]), w)                                       # vertex B
    w = np.where(((d1 <= 0) & (d2 <= 0))[..., None], np.array([1.0, 0.0, 0.0]), w)                                       # vertex A
    return np.clip(w, 0.0, 1.0)


def closest_points_on_mesh(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray, *, k: int = 16,
                           chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closest surface point of every query: ``(triangle id, barycentric weights, distance)``.

    Candidates are the ``k`` triangles whose centroids are nearest the query, which is exact for the roughly uniform
    triangulations used here (tested against brute force on both head meshes). ``faces`` may be any subset of a mesh.
    """
    points = np.asarray(points, np.float64).reshape(-1, 3)
    vertices = np.asarray(vertices, np.float64)
    faces = np.asarray(faces, np.int64)
    if len(faces) == 0:
        raise DomainError("MESH_EMPTY", "Cannot project onto a mesh with no triangles", 422)
    tri = vertices[faces]                                           # (F, 3, 3)
    k = int(min(k, len(faces)))
    _, near = cKDTree(tri.mean(axis=1)).query(points, k=k)
    near = near.reshape(len(points), k)
    ids = np.empty(len(points), np.int64)
    bary = np.empty((len(points), 3))
    dist = np.empty(len(points))
    for start in range(0, len(points), chunk):
        sl = slice(start, start + chunk)
        p, cand = points[sl], near[sl]
        a, b, c = tri[cand, 0], tri[cand, 1], tri[cand, 2]
        w = _closest_bary(p[:, None, :], a, b, c)                    # (n, k, 3)
        q = w[..., 0:1] * a + w[..., 1:2] * b + w[..., 2:3] * c
        d = np.linalg.norm(q - p[:, None, :], axis=-1)
        d = np.where(np.isfinite(d), d, np.inf)
        best = d.argmin(axis=1)
        rows = np.arange(len(p))
        ids[sl], bary[sl], dist[sl] = cand[rows, best], w[rows, best], d[rows, best]
    return ids, bary, dist


# ------------------------------------------------------------------------------------------------------- registration
def umeyama(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity (scale, proper rotation, translation) minimising ``|s R source + t - target|``."""
    ms, mt = source.mean(axis=0), target.mean(axis=0)
    xs, xt = source - ms, target - mt
    u, sing, vt = np.linalg.svd(xt.T @ xs / len(source))
    flip = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        flip[2, 2] = -1.0
    rot = u @ flip @ vt
    scale = float((sing * np.diag(flip)).sum() / xs.var(axis=0).sum())
    return scale, rot, mt - scale * rot @ ms


@dataclass
class Registration:
    """Maps the *target* topology's rest surface into the *source* frame: ``x -> s R x + t`` then an optional TPS warp."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    warp: Any = None
    stats: dict[str, Any] = field(default_factory=dict)

    def apply(self, x: np.ndarray) -> np.ndarray:
        y = self.scale * np.asarray(x, np.float64) @ self.rotation.T + self.translation
        return y + self.warp(y) if self.warp is not None else y


def register_surfaces(source: TopologyMesh, target: TopologyMesh, *, iterations: int = 40, trim: float = 0.8,
                      nonrigid: bool = True) -> Registration:
    """Trimmed similarity ICP of ``target`` onto ``source`` (nose tips give the starting translation), then an optional
    smooth TPS refinement driven by closest-point pairs. Both meshes must already share the identity-asset frame."""
    sv, sf = source.vertices, source.faces
    tip_s, tip_t = sv[np.argmax(sv[:, 2])], target.vertices[np.argmax(target.vertices[:, 2])]
    scale, rot, trans = 1.0, np.eye(3), tip_s - tip_t
    history = []
    for _ in range(iterations):
        moved = scale * target.vertices @ rot.T + trans
        ids, bary, dist = closest_points_on_mesh(moved, sv, sf)
        q = np.einsum("nc,ncd->nd", bary, sv[sf[ids]])
        keep = dist <= np.quantile(dist, trim)
        s2, r2, t2 = umeyama(moved[keep], q[keep])
        scale, rot, trans = s2 * scale, r2 @ rot, s2 * trans @ r2.T + t2
        history.append(float(dist[keep].mean()))
    reg = Registration(scale, rot, trans)
    moved = reg.apply(target.vertices)
    _, _, d0 = closest_points_on_mesh(moved, sv, sf)
    reg.stats = {"icp_iterations": iterations, "trim": trim, "similarity_scale": float(scale),
                 "rigid_mean_mm": float(d0.mean() * 1000), "rigid_p95_mm": float(np.percentile(d0, 95) * 1000),
                 "rigid_trimmed_mean_mm": float(history[-1] * 1000)}
    if nonrigid:
        _refine_nonrigid(reg, source, target)
    return reg


def _refine_nonrigid(reg: Registration, source: TopologyMesh, target: TopologyMesh, *, controls: int = 400,
                     smoothing: float = 5e-3, max_shift_mm: float = 8.0) -> None:
    """Low-order TPS from the rigidly aligned target to the source surface. Kept only if it lowers the trimmed residual
    without folding the surface (fraction of triangles whose normal flips) - otherwise the rigid result stands."""
    from scipy.interpolate import RBFInterpolator
    sv, sf = source.vertices, source.faces
    moved = reg.apply(target.vertices)
    ids, bary, dist = closest_points_on_mesh(moved, sv, sf)
    q = np.einsum("nc,ncd->nd", bary, sv[sf[ids]])
    ok = np.flatnonzero(dist <= np.quantile(dist, 0.8))
    rng = np.random.default_rng(0)
    pick = rng.choice(ok, size=min(controls, len(ok)), replace=False)
    shift = np.clip(q[pick] - moved[pick], -max_shift_mm / 1000, max_shift_mm / 1000)
    warp = RBFInterpolator(moved[pick], shift, kernel="thin_plate_spline", smoothing=smoothing, degree=1)

    def flipped(vertices):
        tri_t, tri_n = target.vertices[target.faces], vertices[target.faces]
        n0 = np.cross(tri_t[:, 1] - tri_t[:, 0], tri_t[:, 2] - tri_t[:, 0])
        n1 = np.cross(tri_n[:, 1] - tri_n[:, 0], tri_n[:, 2] - tri_n[:, 0])
        return float((np.einsum("fc,fc->f", n0, n1) < 0).mean())

    after = moved + warp(moved)
    _, _, d1 = closest_points_on_mesh(after, sv, sf)
    before = float(dist[dist <= np.quantile(dist, 0.8)].mean())
    trimmed = float(d1[d1 <= np.quantile(d1, 0.8)].mean())
    fold = flipped(after)
    reg.stats.update({"nonrigid_trimmed_mean_mm": trimmed * 1000, "nonrigid_folded_fraction": fold})
    if trimmed < 0.85 * before and fold < 0.005:
        reg.warp = warp
        reg.stats["nonrigid_applied"] = True
    else:
        reg.stats["nonrigid_applied"] = False


# --------------------------------------------------------------------------------------------------------- UV handling
def _face_rings(faces: np.ndarray, vertex_count: int, rings: int) -> sparse.csr_matrix:
    """Boolean triangle x triangle reachability within ``rings`` vertex-sharing steps (stays inside one UV island)."""
    n = len(faces)
    incidence = sparse.csr_matrix((np.ones(3 * n), (np.repeat(np.arange(n), 3), faces.ravel())), shape=(n, vertex_count))
    step = (incidence @ incidence.T).astype(bool).astype(np.int8)
    reach = sparse.identity(n, dtype=np.int8, format="csr")
    for _ in range(rings):
        reach = (reach @ step).astype(bool).astype(np.int8)
    return reach.tocsr()


def derive_uv_layout(donor: TopologyMesh, receiver: TopologyMesh, receiver_in_donor_frame: np.ndarray, *, rings: int = 3,
                     uv_tolerance: float = 2e-4) -> tuple[TopologyMesh, dict]:
    """Give ``receiver`` a UV layout (and seam-split vertices) by transferring the donor's atlas across the surface.

    Each receiver triangle is anchored on the donor triangle nearest its centroid, and all three of its corners are projected
    onto the donor triangles within ``rings`` vertex-steps of that anchor (so a triangle that spans a cavity opening is not
    stretched across the atlas, which independent per-corner projection does). A receiver vertex whose incident triangles
    disagree about its UV by more than ``uv_tolerance`` is duplicated; otherwise it stays shared. The result is a derived
    layout, not an official one for the receiver model.
    """
    if donor.uvs is None:
        raise DomainError("UV_DONOR_MISSING", "The donor topology has no UV layout to transfer", 422)
    pos, rf = np.asarray(receiver_in_donor_frame, np.float64), receiver.faces
    dv, df = donor.vertices, donor.faces
    anchor, _, _ = closest_points_on_mesh(pos[rf].mean(axis=1), dv, df)
    reach = _face_rings(df, len(dv), rings)
    corner_uv = np.zeros((len(rf), 3, 2))
    corner_dist = np.zeros((len(rf), 3))
    for start in range(0, len(rf), 512):
        stop = min(start + 512, len(rf))
        sets = [reach.indices[reach.indptr[a]:reach.indptr[a + 1]] for a in anchor[start:stop]]
        width = max(len(s) for s in sets)
        cand = np.stack([np.pad(s, (0, width - len(s)), mode="edge") for s in sets])            # (n, K) donor triangle ids
        a, b, c = dv[df[cand, 0]], dv[df[cand, 1]], dv[df[cand, 2]]                              # (n, K, 3)
        for corner in range(3):
            p = pos[rf[start:stop, corner]][:, None, :]
            w = _closest_bary(p, a, b, c)
            q = w[..., 0:1] * a + w[..., 1:2] * b + w[..., 2:3] * c
            d = np.linalg.norm(q - p, axis=-1)
            best = np.where(np.isfinite(d), d, np.inf).argmin(axis=1)
            rows = np.arange(stop - start)
            wb = w[rows, best]
            corner_uv[start:stop, corner] = np.einsum("nc,ncd->nd", wb, donor.uvs[df[cand[rows, best]]])
            corner_dist[start:stop, corner] = d[rows, best]

    flat_vertex = rf.ravel()
    key = np.concatenate([flat_vertex[:, None], np.round(corner_uv.reshape(-1, 2) / uv_tolerance).astype(np.int64)], axis=1)
    unique, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    new_faces = inverse.reshape(-1, 3).astype(np.int64)
    uvs = corner_uv.reshape(-1, 2)[first]
    source_index = unique[:, 0].astype(np.int64)
    split = TopologyMesh(
        topology_id=receiver.topology_id, vertices=receiver.vertices[source_index], faces=new_faces, uvs=uvs,
        shape_basis=None if receiver.shape_basis is None else receiver.shape_basis[source_index],
        shape_names=receiver.shape_names, landmarks={},
        source_index=source_index if receiver.source_index is None else receiver.source_index[source_index],
        uv_provenance=f"derived from {donor.topology_id} atlas by surface transfer", meta=dict(receiver.meta))

    def uv_area(uv, f):
        a, b, c = uv[f[:, 0]], uv[f[:, 1]], uv[f[:, 2]]
        return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))

    # Includes receiver triangles that face the opposite way to the donor surface they sit on (eyeballs, nostril and mouth
    # interiors), which no surface transfer can orient consistently.
    donor_sign = np.sign(uv_area(donor.uvs, df)[anchor])
    mismatch = float((np.sign(uv_area(uvs, new_faces)) != donor_sign).mean())
    return split, {"rings": rings, "vertices_before": int(len(receiver.vertices)), "vertices_after": int(len(source_index)),
                   "uv_distance_mean_mm": float(corner_dist.mean() * 1000), "uv_distance_p95_mm": float(np.percentile(corner_dist, 95) * 1000),
                   "uv_distance_max_mm": float(corner_dist.max() * 1000), "uv_orientation_mismatch_fraction": mismatch,
                   "uv_overlap_fraction": uv_overlap_fraction(uvs, new_faces)}


def _triangle_texels(uvs: np.ndarray, faces: np.ndarray, size: int):
    """Yield ``(face, rows, cols, barycentric (n, 3))`` for the texels whose centre lies inside each UV triangle.
    UV is v-up, so the image row of a point is ``(1 - v) * size``."""
    px, py = uvs[:, 0] * size, (1.0 - uvs[:, 1]) * size
    eps = 1e-6
    for f, (i, j, k) in enumerate(faces):
        x0, x1, x2, y0, y1, y2 = px[i], px[j], px[k], py[i], py[j], py[k]
        det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(det) < 1e-12:
            continue
        lo_x, hi_x = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), size - 1)
        lo_y, hi_y = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), size - 1)
        if hi_x < lo_x or hi_y < lo_y:
            continue
        gx, gy = np.meshgrid(np.arange(lo_x, hi_x + 1) + 0.5, np.arange(lo_y, hi_y + 1) + 0.5)
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / det
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / det
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -eps) & (l1 >= -eps) & (l2 >= -eps)
        if inside.any():
            ys, xs = np.nonzero(inside)
            yield f, lo_y + ys, lo_x + xs, np.stack([l0[inside], l1[inside], l2[inside]], axis=-1)


def rasterize_uv(uvs: np.ndarray, faces: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per texel: covering triangle id (-1 outside every island) and its barycentric weights (later triangles win overlaps)."""
    tri = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    for f, rows, cols, w in _triangle_texels(uvs, faces, size):
        tri[rows, cols] = f
        bary[rows, cols] = w
    return tri, bary


def uv_overlap_fraction(uvs: np.ndarray, faces: np.ndarray, size: int = 256) -> float:
    """Fraction of covered texels claimed by more than one triangle: 0 for a valid (injective) atlas. A texel that two surface
    regions share can only hold one of their colours, so this is the measured validity of a derived layout."""
    count = np.zeros((size, size), np.int32)
    for _, rows, cols, _ in _triangle_texels(uvs, faces, size):
        count[rows, cols] += 1
    covered = int((count > 0).sum())
    return float((count > 1).sum() / covered) if covered else 0.0


def pad_texels(image: np.ndarray, valid: np.ndarray, iterations: int = 8) -> np.ndarray:
    """Grow valid texels outward so bilinear taps and mip levels near island borders never see uncovered texels."""
    out = image.astype(np.float32)
    mask = valid.astype(np.float32)
    for _ in range(iterations):
        count = cv2.boxFilter(mask, -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        total = cv2.boxFilter(out * mask[..., None], -1, (3, 3), normalize=False, borderType=cv2.BORDER_CONSTANT)
        grow = (mask == 0) & (count > 0)
        if not grow.any():
            break
        out[grow] = total[grow] / count[grow][:, None]
        mask = np.where(grow, 1.0, mask)
    return out


# --------------------------------------------------------------------------------------------------------- the bridge
def _csr(ids: np.ndarray, bary: np.ndarray, faces: np.ndarray, columns: int) -> sparse.csr_matrix:
    n = len(ids)
    rows = np.repeat(np.arange(n), 3)
    return sparse.csr_matrix((bary.ravel(), (rows, faces[ids].ravel())), shape=(n, columns))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class BarycentricMeshBridge:
    """Bidirectional surface correspondence between a *source* and a *target* topology (see the module docstring).

    ``source`` is the topology deformed meshes live on (the identity-asset frame); positions passed in and returned are in
    that shared frame. ``target`` is the other topology, with a UV layout derived from the source's when it had none.
    """

    def __init__(self, source: TopologyMesh, target: TopologyMesh, registration: Registration, *, matched_mm: float):
        self.source, self.target, self.registration, self.matched_mm = source, target, registration, matched_mm
        self.target_in_source = registration.apply(target.vertices)             # target rest surface, shared frame
        self._maps: dict[Direction, dict[str, Any]] = {}
        self._texture_maps: dict[tuple[Direction, int], tuple[np.ndarray, np.ndarray]] = {}
        self.uv_stats: dict[str, Any] = {}

    @classmethod
    def build(cls, source: TopologyMesh, target: TopologyMesh, *, matched_mm: float = 6.0) -> "BarycentricMeshBridge":
        registration = register_surfaces(source, target)
        uv_stats: dict[str, Any] = {}
        if target.uvs is None and source.uvs is not None:
            target, uv_stats = derive_uv_layout(source, target, registration.apply(target.vertices))
        bridge = cls(source, target, registration, matched_mm=matched_mm)
        bridge.uv_stats = uv_stats
        bridge._build_maps()
        return bridge

    # ------------------------------------------------------------------ operators
    def _build_maps(self) -> None:
        s, t, t_in_s = self.source, self.target, self.target_in_source
        # target vertex -> point on the source surface (produces target-topology meshes from source-topology ones)
        ids, bary, dist = closest_points_on_mesh(t_in_s, s.vertices, s.faces)
        q = np.einsum("nc,ncd->nd", bary, s.vertices[s.faces[ids]])
        n_q = _unit(np.einsum("nc,ncd->nd", bary, s.normals()[s.faces[ids]]))
        self._maps["target"] = {"matrix": _csr(ids, bary, s.faces, len(s.vertices)), "offset": np.einsum("nc,nc->n", t_in_s - q, n_q),
                                "distance": dist, "faces": s.faces, "out_faces": t.faces}
        # source vertex -> point on the (registered) target surface
        t_normals = TopologyMesh(t.topology_id, t_in_s, t.faces).normals()
        ids, bary, dist = closest_points_on_mesh(s.vertices, t_in_s, t.faces)
        q = np.einsum("nc,ncd->nd", bary, t_in_s[t.faces[ids]])
        n_q = _unit(np.einsum("nc,ncd->nd", bary, t_normals[t.faces[ids]]))
        self._maps["source"] = {"matrix": _csr(ids, bary, t.faces, len(t.vertices)), "offset": np.einsum("nc,nc->n", s.vertices - q, n_q),
                                "distance": dist, "faces": t.faces, "out_faces": s.faces}

    def _check(self, values: np.ndarray, to: Direction) -> dict[str, Any]:
        if to not in ("target", "source"):
            raise DomainError("BRIDGE_DIRECTION_INVALID", "direction must be 'target' or 'source'", 422)
        m = self._maps[to]
        if values.shape[0] != m["matrix"].shape[1]:
            raise DomainError("BRIDGE_SHAPE_MISMATCH", f"expected {m['matrix'].shape[1]} vertices on the "
                              f"{'source' if to == 'target' else 'target'} topology, got {values.shape[0]}", 422)
        return m

    def matched(self, to: Direction) -> np.ndarray:
        """Per output vertex: True when the other surface lies within ``matched_mm`` (otherwise the point is clamped)."""
        return self._maps[to]["distance"] * 1000.0 <= self.matched_mm

    def transfer(self, values: np.ndarray, to: Direction = "target") -> np.ndarray:
        """Barycentric interpolation of any per-vertex attribute (colours, weights, masks) onto the other topology."""
        values = np.asarray(values, np.float64)
        return self._check(values, to)["matrix"] @ values

    def convert_normals(self, normals: np.ndarray, to: Direction = "target") -> np.ndarray:
        return _unit(self.transfer(normals, to))

    def convert_positions(self, positions: np.ndarray, to: Direction = "target") -> np.ndarray:
        """Surface points of a deformed mesh on one topology, expressed as vertices of the other (shared frame)."""
        positions = np.asarray(positions, np.float64)
        m = self._check(positions, to)
        in_faces = self.source.faces if to == "target" else self.target.faces
        normals = self.convert_normals(TopologyMesh("_", positions, in_faces).normals(), to)
        return m["matrix"] @ positions + m["offset"][:, None] * normals

    def point_weights(self, source_vertices: np.ndarray) -> sparse.csr_matrix:
        """Rows that express the given *source* vertices as surface points of the target: ``(n, V_target)``, so a landmark set
        defined on the source topology becomes a linear functional of target vertices."""
        return self._maps["source"]["matrix"][np.asarray(source_vertices, np.int64)].tocsr()

    def target_vectors_to_shared(self, vectors: np.ndarray) -> np.ndarray:
        """Rotate/scale displacement vectors of the target's own frame (e.g. its shape basis) into the shared frame."""
        return self.registration.scale * np.asarray(vectors, np.float64) @ self.registration.rotation.T

    # ------------------------------------------------------------------ UV albedo
    def _texture_map(self, to: Direction, size: int) -> tuple[np.ndarray, np.ndarray]:
        key = (to, size)
        if key not in self._texture_maps:
            out_mesh, src_mesh = (self.target, self.source) if to == "target" else (self.source, self.target)
            out_pos = self.target_in_source if to == "target" else self.source.vertices
            src_pos = self.source.vertices if to == "target" else self.target_in_source
            if out_mesh.uvs is None or src_mesh.uvs is None:
                raise DomainError("UV_LAYOUT_MISSING", "Both topologies need a UV layout to transfer a texture", 422)
            tri, bary = rasterize_uv(out_mesh.uvs, out_mesh.faces, size)
            covered = tri >= 0
            points = np.einsum("nc,ncd->nd", bary[covered], out_pos[out_mesh.faces[tri[covered]]])
            ids, b, dist = closest_points_on_mesh(points, src_pos, src_mesh.faces)
            uv = np.einsum("nc,ncd->nd", b, src_mesh.uvs[src_mesh.faces[ids]])
            uv_map = np.zeros((size, size, 2), np.float32)
            uv_map[covered] = uv
            valid = np.zeros((size, size), bool)
            valid[covered] = dist * 1000.0 <= self.matched_mm
            self._texture_maps[key] = (uv_map, np.stack([covered, valid], axis=-1))
        return self._texture_maps[key]

    def convert_texture(self, texture: np.ndarray, to: Direction = "target", *, out_size: int | None = None,
                        texel_state: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Resample a UV albedo atlas of the other topology onto this one's layout.

        Returns ``texture`` (uint8, island borders padded), ``covered`` (texel lies inside a UV island) and ``matched`` (its
        surface point has a counterpart on the source surface); with ``texel_state`` also ``texel_state`` (nearest-sampled;
        unmatched texels are forced to 0 = UNSEEN so a viewer greys them instead of showing clamped colour).
        """
        texture = np.asarray(texture)
        if texture.ndim != 3:
            raise DomainError("TEXTURE_SHAPE_INVALID", "texture must be (H, W, C)", 422)
        size = int(out_size or texture.shape[0])
        uv_map, flags = self._texture_map(to, size)
        covered, matched = flags[..., 0], flags[..., 1]
        h, w = texture.shape[:2]
        map_x = (uv_map[..., 0] * w - 0.5).astype(np.float32)
        map_y = ((1.0 - uv_map[..., 1]) * h - 0.5).astype(np.float32)
        sampled = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        sampled = np.where(matched[..., None], sampled, 0)
        padded = pad_texels(sampled, matched)
        result = {"texture": np.clip(np.rint(padded), 0, 255).astype(texture.dtype) if texture.dtype == np.uint8 else padded.astype(texture.dtype),
                  "covered": covered, "matched": matched}
        if texel_state is not None:
            state = cv2.remap(np.asarray(texel_state), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            result["texel_state"] = np.where(matched, state, 0).astype(texel_state.dtype)
        return result

    # ------------------------------------------------------------------ evidence
    def report(self) -> dict[str, Any]:
        def block(to: Direction) -> dict[str, Any]:
            d = self._maps[to]["distance"] * 1000.0
            return {"vertices": int(len(d)), "mean_mm": round(float(d.mean()), 3), "p95_mm": round(float(np.percentile(d, 95)), 3),
                    "max_mm": round(float(d.max()), 3), "matched_fraction": round(float((d <= self.matched_mm).mean()), 4)}
        return {"source": self.source.topology_id, "target": self.target.topology_id, "matched_mm": self.matched_mm,
                "to_target": block("target"), "to_source": block("source"),
                "registration": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.registration.stats.items()},
                "uv": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.uv_stats.items()},
                "uv_provenance": self.target.uv_provenance,
                "note": "Closest-point surface correspondence, not anatomical; unmatched regions are clamped to the nearest "
                        "boundary point and flagged, lateral rest-pose residuals are dropped (only the normal offset is carried). "
                        "uv.uv_overlap_fraction is the layout's injectivity check; uv.uv_orientation_mismatch_fraction counts "
                        "triangles wound opposite to the donor surface they sit on (eyeballs and nostril/mouth interiors face the "
                        "other way) plus any UV fold-overs."}
