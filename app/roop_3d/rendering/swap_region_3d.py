# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""3D swap mask beyond the inner face oval, and the Laplacian field that carries a jaw onto a different jaw width.

ROME-inspired (mesh-based head with a face region that is *extended* past the facial oval to the jaw, chin and ears rather
than cropped at it). The inner oval is the default and stays the default: every extension is opt-in.

SwapMask3D
    A soft per-vertex weight on the welded head mesh. ``inner`` is the frontal region enclosed by the jaw contour
    (landmarks 0-16) and the brow line lifted over the forehead. Optional components extend it:
    ``jawline``      a geodesic band around the jaw contour, both sides of it (the old oval cut the jaw in half);
    ``chin_underside`` down-facing surface under the chin (visible when the head tilts up);
    ``ears``         a geodesic disc around each ear anchor.

Jaw adaptation (``adapt_jaw``)
    When the source and target jaws differ in width, a swapped face either spills past the target's jaw or leaves a rim of
    target jaw around it - the two seam types the oval cut hides only at one jaw width. The jaw contour (and, opt-in, chin
    and ear anchors) are moved onto the target's, and the displacement is interpolated through a free zone as a
    **biharmonic** field on the welded mesh graph (minimise |L d|^2, d = delta at the handles, d = 0 outside the zone),
    which is C1 at the handles - a harmonic field would put a log spike at every landmark. Front-facing triangles that would
    flip are relieved locally, the same guarantee as ``rbf_deform``. Seam copies (UV islands) move identically.

What this is not: the 68-point correspondence on the MakeHuman head is DERIVED, not annotated (see ``makehuman_landmarks``),
and ear anchors are the lateral extremum of the eye-to-mouth band, a geometric prior; both are reported, never presented as
measured. A jaw *target* has to come from somewhere: ``jaw_target_from_ratios`` (a user dial), or
``jaw_target_from_landmarks`` (a fitted target head). Nothing here estimates it from pixels.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import Field
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.reconstruction.mesh_bridge import umeyama
from roop_3d.reconstruction.rbf_deform import laplacian_smooth_field, weld_ids, welded_vertex_normals
from roop_3d.rendering.vertex_raster import RasterBuffers, rasterize_vertex_attributes

JAW_LANDMARKS = tuple(range(17))
CHIN_LANDMARKS = (5, 6, 7, 8, 9, 10, 11)
CHIN_HANDLES = (6, 7, 8, 9, 10)
BROW_LANDMARKS = tuple(range(17, 27))
EYE_LANDMARKS = tuple(range(36, 48))
MOUTH_LANDMARKS = tuple(range(48, 68))
COMPONENTS = ("inner", "jawline", "chin_underside", "ears")


class SwapRegionConfig(Contract):
    include_jawline: bool = False
    include_chin_underside: bool = False
    include_ears: bool = False
    inner_margin_m: float = Field(default=0.004, ge=0.0, le=0.03, description="Inner oval is shrunk this far inside the jaw contour")
    forehead_lift: float = Field(default=2.0, ge=0.0, le=4.0, description="Brow line is lifted this many brow-to-eye distances")
    feather_m: float = Field(default=0.008, gt=0.0, le=0.03)
    jaw_band_m: float = Field(default=0.014, gt=0.0, le=0.05, description="Geodesic reach of the jawline component past the contour")
    inner_blend_m: float = Field(default=0.02, ge=0.0, le=0.05, description="Deformation fall-off on the inner side of the contour")
    underside_depth_m: float = Field(default=0.03, gt=0.0, le=0.08)
    underside_min_normal_y: float = Field(default=0.3, ge=0.0, le=0.9)
    ear_radius_m: float = Field(default=0.03, gt=0.0, le=0.08)
    ear_follow: float = Field(default=0.5, ge=0.0, le=1.0, description="Share of the jaw width change the ears follow")
    max_displacement_m: float = Field(default=0.025, gt=0.0, le=0.06)

    @property
    def extended(self) -> bool:
        return self.include_jawline or self.include_chin_underside or self.include_ears


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(x, np.float64) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# =========================================================================== welded mesh graph
class WeldedGraph:
    """Position-welded edge graph of a mesh: geodesics, shortest paths and the uniform graph Laplacian."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray) -> None:
        v = np.asarray(vertices, np.float64)
        f = np.asarray(faces, np.int64)
        if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not np.isfinite(v).all():
            raise DomainError("SWAP_MESH_INVALID", "Mesh vertices and faces must be finite (V,3) and (F,3)", 422)
        if f.size and (f.min() < 0 or f.max() >= len(v)):
            raise DomainError("SWAP_MESH_INVALID", "Face indices are outside the vertex range", 422)
        self.weld = weld_ids(v)
        self.n = int(self.weld.max()) + 1
        first = np.zeros(self.n, np.int64)
        first[self.weld] = np.arange(len(v))
        self.position = v[first]
        wf = self.weld[f]
        edges = np.sort(np.concatenate([wf[:, [0, 1]], wf[:, [1, 2]], wf[:, [2, 0]]]), axis=1)
        edges = np.unique(edges[edges[:, 0] != edges[:, 1]], axis=0)
        length = np.linalg.norm(self.position[edges[:, 0]] - self.position[edges[:, 1]], axis=1)
        rows, cols = np.concatenate([edges[:, 0], edges[:, 1]]), np.concatenate([edges[:, 1], edges[:, 0]])
        self.adjacency = sparse.csr_matrix((np.concatenate([length, length]), (rows, cols)), shape=(self.n, self.n))
        ones = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n, self.n))
        degree = np.asarray(ones.sum(axis=1)).reshape(-1)
        self.laplacian = (sparse.diags(degree) - ones).tocsr()

    def geodesic(self, vertex_ids, limit: float = np.inf) -> np.ndarray:
        """Per-VERTEX surface distance (m) to the nearest of ``vertex_ids`` (inf beyond ``limit``)."""
        nodes = np.unique(self.weld[np.asarray(vertex_ids, np.int64)])
        dist = csgraph.dijkstra(self.adjacency, directed=False, indices=nodes, min_only=True, limit=limit)
        return np.asarray(dist)[self.weld]

    def path(self, a: int, b: int) -> np.ndarray:
        """Welded node ids along the shortest surface path between two vertices."""
        src, dst = int(self.weld[a]), int(self.weld[b])
        _, pred = csgraph.dijkstra(self.adjacency, directed=False, indices=src, return_predecessors=True)
        nodes = [dst]
        while nodes[-1] != src:
            step = int(pred[nodes[-1]])
            if step < 0:
                raise DomainError("SWAP_MESH_DISCONNECTED", "Landmarks are not connected on the mesh surface", 422)
            nodes.append(step)
        return np.asarray(nodes[::-1], np.int64)

    def representative(self, nodes: np.ndarray) -> np.ndarray:
        first = np.zeros(self.n, np.int64)
        first[self.weld] = np.arange(len(self.weld))
        return first[np.asarray(nodes, np.int64)]


# =========================================================================== geometry helpers
def _landmark_ids(landmarks68) -> np.ndarray:
    ids = np.asarray(landmarks68, np.int64).reshape(-1)
    if ids.shape != (68,):
        raise DomainError("SWAP_LANDMARKS_INVALID", "Expected the 68-point landmark vertex indices", 422)
    return ids


def jaw_width(points: np.ndarray) -> float:
    """Lateral extent (m) of the jaw contour points (x axis of the mesh frame)."""
    return float(np.ptp(np.asarray(points, np.float64)[:, 0]))


def polygon_signed_distance(points_xy: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    """Distance (positive inside) from each point to the boundary of a simple polygon (V,2)."""
    p = np.asarray(points_xy, np.float64)
    poly = np.asarray(polygon, np.float64)
    a, b = poly, np.roll(poly, -1, axis=0)
    ab = b - a
    ap = p[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-18)[None], 0.0, 1.0)
    nearest = a[None] + t[..., None] * ab[None]
    dist = np.linalg.norm(p[:, None, :] - nearest, axis=-1).min(axis=1)
    py = p[:, None, 1]
    crosses = ((a[None, :, 1] > py) != (b[None, :, 1] > py))
    with np.errstate(divide="ignore", invalid="ignore"):
        x_at = a[None, :, 0] + (py - a[None, :, 1]) * ab[None, :, 0] / np.where(ab[None, :, 1] == 0, 1e-18, ab[None, :, 1])
    inside = (crosses & (p[:, None, 0] < x_at)).sum(axis=1) % 2 == 1
    return np.where(inside, dist, -dist)


def derive_ear_anchors(vertices: np.ndarray, landmarks68) -> np.ndarray:
    """Vertex ids [x<0 side, x>0 side] at the lateral extremum of the eye-to-mouth band: a geometric prior, not an annotation."""
    v = np.asarray(vertices, np.float64)
    ids = _landmark_ids(landmarks68)
    eye_y = float(v[ids[list(EYE_LANDMARKS)], 1].mean())
    mouth_y = float(v[ids[list(MOUTH_LANDMARKS)], 1].mean())
    band = (v[:, 1] >= eye_y - 0.01) & (v[:, 1] <= mouth_y)
    anchors = []
    for side in (-1.0, 1.0):
        candidate = np.nonzero(band & (side * v[:, 0] > 0))[0]
        if not len(candidate):
            raise DomainError("SWAP_EARS_NOT_FOUND", "No lateral surface found for the ear anchors", 422)
        anchors.append(int(candidate[np.argmax(side * v[candidate, 0])]))
    return np.asarray(anchors, np.int64)


# =========================================================================== swap mask
@dataclass(frozen=True)
class SwapMask3D:
    weights: np.ndarray                            # (V,) float32 in 0..1, max over the enabled components
    components: dict[str, np.ndarray]              # name -> (V,) float32 (only the enabled ones plus ``inner``)
    config: SwapRegionConfig
    contour_vertices: np.ndarray                   # representative vertex ids along the geodesic jaw contour
    ear_anchors: np.ndarray | None = None          # (2,) vertex ids, or None when ears are not enabled
    notes: tuple[str, ...] = field(default_factory=tuple)

    def area_fraction(self, name: str | None = None) -> float:
        w = self.weights if name is None else self.components[name]
        return float((w > 0.5).mean())


def build_swap_mask_3d(vertices: np.ndarray, faces: np.ndarray, landmarks68, config: SwapRegionConfig | None = None, *,
                       graph: WeldedGraph | None = None) -> SwapMask3D:
    cfg = config or SwapRegionConfig()
    v = np.asarray(vertices, np.float64)
    ids = _landmark_ids(landmarks68)
    graph = graph or WeldedGraph(v, faces)
    normals = welded_vertex_normals(v, faces).astype(np.float64)
    lm = v[ids]
    brow_y = float(lm[list(BROW_LANDMARKS), 1].mean())
    eye_y = float(lm[list(EYE_LANDMARKS), 1].mean())
    lift = cfg.forehead_lift * abs(eye_y - brow_y)
    brow_line = lm[list(BROW_LANDMARKS)][::-1][:, :2] - np.array([0.0, lift])
    polygon = np.vstack([lm[list(JAW_LANDMARKS), :2], brow_line])
    signed = polygon_signed_distance(v[:, :2], polygon)
    facing = _smoothstep(normals[:, 2], -0.05, 0.15)
    inner = (_smoothstep(signed - cfg.inner_margin_m, 0.0, cfg.feather_m) * facing).astype(np.float32)
    components = {"inner": inner}
    notes = ["jaw/chin landmarks are the derived 68-point correspondence, not annotations"]

    contour_nodes = np.unique(np.concatenate([graph.path(int(ids[k]), int(ids[k + 1])) for k in range(16)]))
    contour_vertices = graph.representative(contour_nodes)
    if cfg.include_jawline:
        g = graph.geodesic(contour_vertices, limit=cfg.jaw_band_m + cfg.feather_m)
        band = 1.0 - _smoothstep(g, cfg.jaw_band_m - cfg.feather_m, cfg.jaw_band_m)
        components["jawline"] = (np.where(np.isfinite(g), band, 0.0)).astype(np.float32)
    if cfg.include_chin_underside:
        chin_nodes = np.unique(np.concatenate([graph.path(int(ids[k]), int(ids[k + 1])) for k in range(5, 11)]))
        g = graph.geodesic(graph.representative(chin_nodes), limit=cfg.underside_depth_m + cfg.feather_m)
        chin_y = float(lm[list(CHIN_HANDLES), 1].mean()) - 0.01
        half_width = 0.5 * jaw_width(lm[list(JAW_LANDMARKS)]) * 1.05
        underside = (_smoothstep(normals[:, 1], cfg.underside_min_normal_y * 0.5, cfg.underside_min_normal_y + 0.2)
                     * (1.0 - _smoothstep(g, cfg.underside_depth_m - cfg.feather_m, cfg.underside_depth_m))
                     * (v[:, 1] >= chin_y) * (np.abs(v[:, 0]) <= half_width))
        components["chin_underside"] = np.where(np.isfinite(g), underside, 0.0).astype(np.float32)
    ears = None
    if cfg.include_ears:
        ears = derive_ear_anchors(v, ids)
        g = graph.geodesic(ears, limit=cfg.ear_radius_m + cfg.feather_m)
        disc = 1.0 - _smoothstep(g, cfg.ear_radius_m - cfg.feather_m, cfg.ear_radius_m)
        components["ears"] = np.where(np.isfinite(g), disc, 0.0).astype(np.float32)
        notes.append("ear anchors are the lateral extremum of the eye-to-mouth band (geometric prior)")
    weights = np.max(np.stack(list(components.values())), axis=0).astype(np.float32)
    return SwapMask3D(weights, components, cfg, contour_vertices, ears, tuple(notes))


def rasterize_swap_mask(mask: SwapMask3D, pixels: np.ndarray, depth: np.ndarray, faces: np.ndarray,
                        shape: tuple[int, int], *, component: str | None = None) -> tuple[np.ndarray, RasterBuffers]:
    """Depth-tested 2D swap mask (H, W) from the per-vertex weights; also returns the raster buffers (depth, face ids)."""
    weights = mask.weights if component is None else mask.components[component]
    raster = rasterize_vertex_attributes(pixels, depth, faces, weights, shape)
    return np.clip(raster.attributes[..., 0], 0.0, 1.0).astype(np.float32), raster


# =========================================================================== jaw target
@dataclass(frozen=True)
class JawTarget:
    jaw_points: np.ndarray                         # (17, 3) desired positions of landmarks 0-16, in the SOURCE mesh frame
    ear_points: np.ndarray | None = None           # (2, 3) desired ear anchors [x<0 side, x>0 side]


def jaw_target_from_ratios(vertices: np.ndarray, landmarks68, *, width_ratio: float = 1.0, length_ratio: float = 1.0,
                           config: SwapRegionConfig | None = None) -> JawTarget:
    """Scale the jaw contour: width about the mid-line, length (chin drop) about the mouth line; ears follow ``ear_follow``."""
    if not (np.isfinite(width_ratio) and np.isfinite(length_ratio) and 0.5 <= width_ratio <= 1.6 and 0.6 <= length_ratio <= 1.5):
        raise DomainError("SWAP_JAW_TARGET_INVALID", "Jaw ratios must be finite within width 0.5-1.6 and length 0.6-1.5", 422)
    cfg = config or SwapRegionConfig()
    v = np.asarray(vertices, np.float64)
    ids = _landmark_ids(landmarks68)
    jaw = v[ids[list(JAW_LANDMARKS)]].copy()
    mid_x = 0.5 * (jaw[:, 0].min() + jaw[:, 0].max())
    mouth_y = float(v[ids[list(MOUTH_LANDMARKS)], 1].mean())
    out = jaw.copy()
    out[:, 0] = mid_x + (jaw[:, 0] - mid_x) * width_ratio
    below = jaw[:, 1] > mouth_y
    out[below, 1] = mouth_y + (jaw[below, 1] - mouth_y) * length_ratio
    ears = None
    if cfg.include_ears:
        anchors = v[derive_ear_anchors(v, ids)].copy()
        anchors[:, 0] = mid_x + (anchors[:, 0] - mid_x) * (1.0 + cfg.ear_follow * (width_ratio - 1.0))
        ears = anchors
    return JawTarget(out, ears)


def jaw_target_from_landmarks(source_landmarks3d: np.ndarray, target_landmarks3d: np.ndarray, *,
                              target_ears3d: np.ndarray | None = None,
                              source_ears3d: np.ndarray | None = None) -> JawTarget:
    """Target jaw from a fitted target head: both (68, 3) sets are registered on the NON-jaw landmarks (a similarity), so
    only the jaw's *difference in shape* remains and the face's own scale/pose offset is not mistaken for a jaw change."""
    src, dst = np.asarray(source_landmarks3d, np.float64), np.asarray(target_landmarks3d, np.float64)
    if src.shape != (68, 3) or dst.shape != (68, 3) or not (np.isfinite(src).all() and np.isfinite(dst).all()):
        raise DomainError("SWAP_JAW_TARGET_INVALID", "Source and target landmarks must be finite (68, 3)", 422)
    inner = list(range(17, 68))
    scale, rot, trans = umeyama(dst[inner], src[inner])
    to_source = lambda p: scale * np.asarray(p, np.float64) @ rot.T + trans
    ears = None
    if target_ears3d is not None:
        ears = to_source(target_ears3d)
    return JawTarget(to_source(dst[list(JAW_LANDMARKS)]), ears)


# =========================================================================== Laplacian jaw adaptation
@dataclass(frozen=True)
class JawAdaptationReport:
    regions: tuple[str, ...]
    handles: int
    free_vertices: int
    jaw_width_before_m: float
    jaw_width_after_m: float
    jaw_width_target_m: float
    handle_residual_m: float                       # mean distance of the handles from their targets AFTER fold relief
    max_displacement_m: float
    strength: float                                # mean applied share of the field over moved vertices (< 1: folds relieved)
    folded_triangles: int                          # triangles flipped relative to rest after adaptation (0 expected)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class JawAdaptation:
    vertices: np.ndarray
    field: np.ndarray                              # (V, 3) applied displacement
    mask: SwapMask3D
    report: JawAdaptationReport


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    n = np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]], vertices[faces[:, 2]] - vertices[faces[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _relieve_folds(vertices: np.ndarray, faces: np.ndarray, field: np.ndarray, weld: np.ndarray,
                   watch: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale the field down, locally first, until no watched triangle has flipped against its own rest orientation."""
    rest = _face_normals(vertices, faces)

    def flipped(scale: np.ndarray) -> np.ndarray:
        trial = vertices + scale[:, None] * field
        return watch & (np.einsum("ij,ij->i", _face_normals(trial, faces), rest) < 0.15)

    scale = np.ones(len(vertices))
    for _ in range(14):
        bad = flipped(scale)
        if not bad.any():
            break
        hit = np.zeros(len(vertices), bool)
        hit[faces[bad].ravel()] = True
        scale[hit] *= 0.7
        scale = np.minimum(laplacian_smooth_field(scale[:, None], faces, weld, iterations=2, strength=0.5)[:, 0], 1.0)
    else:
        if flipped(scale).any():
            global_scale = 1.0
            for _ in range(8):
                global_scale *= 0.6
                if not flipped(scale * global_scale).any():
                    break
            else:
                global_scale = 0.0
            scale = scale * global_scale
    moved = np.linalg.norm(field, axis=1) > 1e-9
    return scale, float(scale[moved].mean()) if moved.any() else 1.0


def adapt_jaw(vertices: np.ndarray, faces: np.ndarray, landmarks68, target: JawTarget,
              config: SwapRegionConfig | None = None, *, graph: WeldedGraph | None = None,
              pinned=()) -> JawAdaptation:
    """Carry the jaw (and opt-in chin underside / ears) of ``vertices`` onto ``target`` with a biharmonic Laplacian field.

    The inner face is untouched except for the ``inner_blend_m`` collar inside the contour. With no extension enabled the
    field is zero and the mesh is returned unchanged (the inner-oval behaviour). ``pinned`` vertices are held at zero
    displacement (the field decays to zero at them): on a coarse mesh, where one edge spans several centimetres, the collar
    alone cannot keep the eyes or the mouth still.
    """
    cfg = config or SwapRegionConfig()
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    ids = _landmark_ids(landmarks68)
    graph = graph or WeldedGraph(v, f)
    mask = build_swap_mask_3d(v, f, ids, cfg, graph=graph)
    jaw_now = v[ids[list(JAW_LANDMARKS)]]
    jaw_pts = np.asarray(target.jaw_points, np.float64)
    if jaw_pts.shape != (17, 3) or not np.isfinite(jaw_pts).all():
        raise DomainError("SWAP_JAW_TARGET_INVALID", "JawTarget.jaw_points must be finite (17, 3)", 422)
    if cfg.include_ears and (target.ear_points is None or np.asarray(target.ear_points).shape != (2, 3)):
        raise DomainError("SWAP_JAW_TARGET_INVALID", "Ear adaptation needs JawTarget.ear_points of shape (2, 3)", 422)

    handles: dict[int, list[np.ndarray]] = {}
    handle_goal: dict[int, np.ndarray] = {}

    def add_handle(vertex: int, goal: np.ndarray) -> None:
        node = int(graph.weld[vertex])
        handles.setdefault(node, []).append(goal - v[vertex])
        handle_goal[node] = goal

    zone = np.zeros(graph.n, bool)
    regions: list[str] = []
    if cfg.include_jawline:
        regions.append("jawline")
        for k in JAW_LANDMARKS:
            add_handle(int(ids[k]), jaw_pts[k])
        reach = cfg.jaw_band_m + cfg.inner_blend_m
        dist = graph.geodesic(mask.contour_vertices, limit=reach)
        zone |= np.isfinite(dist)[graph.representative(np.arange(graph.n))]
    elif cfg.include_chin_underside:
        for k in CHIN_HANDLES:
            add_handle(int(ids[k]), jaw_pts[k])
    if cfg.include_chin_underside:
        regions.append("chin_underside")
        chin_nodes = np.unique(np.concatenate([graph.path(int(ids[k]), int(ids[k + 1])) for k in range(5, 11)]))
        dist = graph.geodesic(graph.representative(chin_nodes), limit=cfg.underside_depth_m)
        zone |= np.isfinite(dist)[graph.representative(np.arange(graph.n))]
    if cfg.include_ears:
        regions.append("ears")
        for vertex, goal in zip(mask.ear_anchors, np.asarray(target.ear_points, np.float64)):
            add_handle(int(vertex), goal)
        dist = graph.geodesic(mask.ear_anchors, limit=cfg.ear_radius_m)
        zone |= np.isfinite(dist)[graph.representative(np.arange(graph.n))]

    field_nodes = np.zeros((graph.n, 3))
    notes: list[str] = []
    free_count = 0
    if handles:
        h_nodes = np.fromiter(handles.keys(), np.int64)
        h_delta = np.vstack([np.mean(handles[int(n)], axis=0) for n in h_nodes])
        zone[h_nodes] = False
        pinned_ids = np.asarray(list(pinned), np.int64).reshape(-1)
        if pinned_ids.size:
            if pinned_ids.min() < 0 or pinned_ids.max() >= len(v):
                raise DomainError("SWAP_PINNED_INVALID", "Pinned vertex ids are outside the mesh", 422)
            clash = np.intersect1d(graph.weld[pinned_ids], h_nodes)
            if clash.size:
                raise DomainError("SWAP_PINNED_INVALID", "A pinned vertex is also a handle", 422)
            zone[graph.weld[pinned_ids]] = False                  # outside the free set => held at d = 0
        free = np.nonzero(zone)[0]
        free_count = len(free)
        field_nodes[h_nodes] = h_delta
        if len(free):
            q = (graph.laplacian.T @ graph.laplacian).tocsr()
            q_ff = q[free][:, free] + 1e-9 * sparse.identity(len(free), format="csr")
            rhs = -(q[free][:, h_nodes] @ h_delta)
            field_nodes[free] = splu(q_ff.tocsc()).solve(rhs)
        else:
            notes.append("no free vertices around the handles: only the handle vertices move")
    raw = field_nodes[graph.weld]
    magnitude = np.linalg.norm(raw, axis=1, keepdims=True)
    raw = raw * np.minimum(1.0, cfg.max_displacement_m / np.maximum(magnitude, 1e-12))
    if raw.any():
        watch = np.zeros(len(f), bool)
        touched = np.linalg.norm(raw, axis=1) > 1e-9
        watch = touched[f].any(axis=1)
        scale, strength = _relieve_folds(v, f, raw, graph.weld, watch)
        applied = raw * scale[:, None]
    else:
        applied, strength = raw, 1.0
    out = v + applied
    folded = int((np.einsum("ij,ij->i", _face_normals(out, f), _face_normals(v, f)) < 0.0).sum())
    goals = np.vstack([handle_goal[n] for n in handles]) if handles else np.zeros((0, 3))
    reps = graph.representative(np.fromiter(handles.keys(), np.int64)) if handles else np.zeros(0, np.int64)
    residual = float(np.linalg.norm(out[reps] - goals, axis=1).mean()) if handles else 0.0
    jaw_after = out[ids[list(JAW_LANDMARKS)]]
    report = JawAdaptationReport(
        regions=tuple(regions), handles=len(handles), free_vertices=free_count,
        jaw_width_before_m=jaw_width(jaw_now), jaw_width_after_m=jaw_width(jaw_after), jaw_width_target_m=jaw_width(jaw_pts),
        handle_residual_m=residual, max_displacement_m=float(np.linalg.norm(applied, axis=1).max()) if len(applied) else 0.0,
        strength=float(strength), folded_triangles=folded, notes=tuple(notes))
    return JawAdaptation(out.astype(np.float32), applied.astype(np.float32), mask, report)
