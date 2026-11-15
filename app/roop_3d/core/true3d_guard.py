"""True-3D architecture guard.

roop-3d renders a COMPLETE head: a watertight surface (cranium, forehead dome, temples, cheekbones, mandible,
submental neck, eyeballs, oral cavity) seen by a perspective pin-hole camera K[R|t] and drawn with a Z-buffered
rasterizer using PBR + subsurface scattering or 3D Gaussian shading. 2D face swappers, 5-point crop-paste, 2.5D frontal
strip meshes, weak-perspective / orthographic projectors and screen-space restoration were removed from the code base;
this module keeps them out at runtime.

Public API (all raise a subclass of :class:`True3DArchitectureError`, which is a ``DomainError``)::

    validate_head_mesh(vertices, faces, *, regions=None, parts=None, ...)  -> HeadMeshReport
    validate_camera(K, R, t, *, vertices=None, ...)                         -> CameraReport
    validate_render_config(config)                                          -> RenderConfigReport
    assert_true_3d(mesh=..., camera=..., config=...)                        -> True3DReport

    require_true3d_renderer(what)   # raises True3DRendererUnavailable unless the full-head render path is wired
    close_open_boundaries(V, F)     # caps neck / mouth loops so a head surface becomes watertight
    solve_dynamic_intrinsics(...)   # dynamic focal-length (FOV) solver (delegates to reconstruction.dynamic_camera)

Head frame convention (the MakeHuman builder / ``HeadPose`` OpenCV head frame): x image-right, y DOWN, the face points to
-z. Pass ``face_direction`` / ``up_direction`` when a mesh uses another frame (the stored identity-asset mesh has +z
toward the viewer: ``face_direction=(0, 0, 1)``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from roop_3d.core.errors import DomainError

RENDERER_MODULE = "roop_3d.pipeline.true3d_renderer"
NOT_WIRED_CODE = "RENDERER_NOT_WIRED"
NOT_WIRED_MESSAGE = "3D renderer not wired"


# --------------------------------------------------------------------------- exception hierarchy
class True3DArchitectureError(DomainError):
    """Base: something would violate the true-3D architecture (HTTP 422)."""

    def __init__(self, message: str, code: str = "TRUE3D_ARCHITECTURE_VIOLATION", status: int = 422,
                 details: Mapping[str, Any] | None = None):
        super().__init__(code, message, status)
        self.details = dict(details or {})


class HeadMeshError(True3DArchitectureError):
    def __init__(self, message: str, code: str = "HEAD_MESH_INVALID", details=None):
        super().__init__(message, code, 422, details)


class NotWatertightError(HeadMeshError):
    def __init__(self, message: str, details=None):
        super().__init__(message, "HEAD_MESH_NOT_WATERTIGHT", details)


class IncompleteHeadTopologyError(HeadMeshError):
    """A frontal strip / face-only mesh: missing cranium, temples, mandible, neck, eyeballs or oral cavity."""

    def __init__(self, message: str, details=None):
        super().__init__(message, "HEAD_TOPOLOGY_INCOMPLETE", details)


class CameraModelError(True3DArchitectureError):
    def __init__(self, message: str, code: str = "CAMERA_MODEL_INVALID", details=None):
        super().__init__(message, code, 422, details)


class AffineCameraError(CameraModelError):
    """Orthographic / weak-perspective / affine camera rows are banned."""

    def __init__(self, message: str, details=None):
        super().__init__(message, "CAMERA_AFFINE_REJECTED", details)


class CameraPoseError(CameraModelError):
    def __init__(self, message: str, details=None):
        super().__init__(message, "CAMERA_POSE_INVALID", details)


class RenderConfigError(True3DArchitectureError):
    def __init__(self, message: str, code: str = "RENDER_CONFIG_INVALID", details=None):
        super().__init__(message, code, 422, details)


class True3DRendererUnavailable(True3DArchitectureError):
    """The full-head render path is not wired. Raised instead of ever falling back to a 2D / 2.5D path (HTTP 503)."""

    def __init__(self, message: str = NOT_WIRED_MESSAGE, details=None):
        super().__init__(message, NOT_WIRED_CODE, 503, details)
FRAME_PIPELINE_STAGES = (
    "3D Head Pose & Volumetric Mesh Fitting",
    "3D UV Identity Texture Projection",
    "3D Mesh Rasterization & Lighting",
    "Hair-Excluding Semantic Mask",
    "Soft Feathered Edge Blend",
)


class True3DFramePipelineError(True3DArchitectureError):
    """One face did not complete the required full-head rendering stages in order."""

    def __init__(self, message: str, details: Mapping[str, Any]):
        super().__init__(message, "TRUE3D_FRAME_PIPELINE_INVALID", 422, details)


def require_true3d_frame_pipeline(stages: Iterable[str], *, through: str | None = None) -> tuple[str, ...]:
    """Require an exact ordered prefix (or the complete sequence) of the true-3D frame stages."""
    actual = tuple(stages)
    if through is None:
        expected = FRAME_PIPELINE_STAGES
    else:
        try:
            expected = FRAME_PIPELINE_STAGES[:FRAME_PIPELINE_STAGES.index(through) + 1]
        except ValueError:
            raise ValueError(f"Unknown true-3D pipeline stage: {through}") from None
    if actual != expected:
        raise True3DFramePipelineError(
            "Frame skipped: the required true-3D pipeline stages are missing or out of order",
            {"expected": list(expected), "observed": list(actual), "next_stage": expected[len(actual)] if len(actual) < len(expected) else None},
        )
    return actual




# --------------------------------------------------------------------------- policy
REQUIRED_REGIONS = ("cranium", "forehead_dome", "temples", "cheekbones", "mandible", "submental_neck")
REQUIRED_PARTS = ("eyeballs", "oral_cavity")
BANNED_PROJECTIONS = frozenset({
    "affine", "orthographic", "ortho", "weak_perspective", "weak-perspective", "weakperspective", "scaled_orthographic",
    "screen_space", "screen-space", "2d", "2.5d", "2_5d", "crop_paste", "frontal_strip"})
BANNED_CONFIG_KEYS = frozenset({
    "swapper", "swapper_model_id", "restorer", "restorer_model", "face_restoration_policy", "crop_paste", "aligned_crop",
    "weak_perspective", "orthographic", "pose_blend", "mesh_blend", "tps", "frontalization"})


@dataclass(frozen=True)
class GuardPolicy:
    min_vertices: int = 1500              # a frontal landmark strip has tens of vertices; FLAME 5023, MakeHuman head > 10k
    min_region_vertices: int = 12
    min_region_fraction: float = 0.004
    min_depth_to_height: float = 0.45     # a head is ~0.8-1.1 deep relative to its height (strip: ~0.05)
    min_back_fraction: float = 0.18       # share of vertices behind the head's depth centre (cranium / occiput)
    min_azimuth_bins: int = 13            # of 16 around the vertical axis must hold surface area
    min_planarity: float = 0.30           # smallest / largest principal extent
    min_depth_m: float = 1e-3             # every vertex must be this far in front of the camera
    max_skew_ratio: float = 1e-3          # |K01| / fx
    eyeball_radius_m: tuple[float, float] = (0.007, 0.020)
    eyeball_separation_m: tuple[float, float] = (0.040, 0.090)
    min_oral_vertices: int = 24
    perspective_shaders: tuple[str, ...] = ("pbr", "gaussian", "spherical_harmonics")


DEFAULT_POLICY = GuardPolicy()


# --------------------------------------------------------------------------- reports
@dataclass
class HeadMeshReport:
    vertex_count: int
    face_count: int
    boundary_edges: int
    non_manifold_edges: int
    watertight: bool
    region_counts: dict[str, int] = field(default_factory=dict)
    parts: dict[str, str] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class CameraReport:
    fx: float
    fy: float
    cx: float
    cy: float
    horizontal_fov_deg: float | None
    min_depth: float | None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class RenderConfigReport:
    projection: str
    rasterizer: str
    shader: str
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class True3DReport:
    mesh: HeadMeshReport | None
    camera: CameraReport | None
    config: RenderConfigReport | None


# --------------------------------------------------------------------------- mesh topology helpers
def _as_mesh(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces)
    if v.ndim != 2 or v.shape[1] != 3:
        raise HeadMeshError(f"V must have shape (N, 3); got {v.shape} (a 2D / 2.5D vertex layout is rejected)")
    if not np.isfinite(v).all():
        raise HeadMeshError("V contains non-finite coordinates")
    if f.ndim != 2 or f.shape[1] != 3 or not np.issubdtype(f.dtype, np.integer):
        raise HeadMeshError(f"F must be an integer (M, 3) triangle array; got {f.shape} {f.dtype}")
    if f.size and (f.min() < 0 or f.max() >= len(v)):
        raise HeadMeshError("F references vertices outside V")
    return v, f.astype(np.int64)


def edge_statistics(faces: np.ndarray, vertex_count: int) -> tuple[int, int, int]:
    """(boundary_edges, non_manifold_edges, edge_count): watertight means every undirected edge is shared by exactly 2 faces."""
    f = np.asarray(faces, dtype=np.int64)
    if not len(f):
        return 0, 0, 0
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    edges.sort(axis=1)
    keys = edges[:, 0] * np.int64(vertex_count) + edges[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    return int(np.count_nonzero(counts == 1)), int(np.count_nonzero(counts > 2)), int(len(counts))


def _directed_edge_balance(faces: np.ndarray, vertex_count: int) -> int:
    """Edges whose two faces traverse them in the SAME direction (inconsistent winding)."""
    f = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    keys = directed[:, 0] * np.int64(vertex_count) + directed[:, 1]
    _, counts = np.unique(keys, return_counts=True)
    return int(np.count_nonzero(counts > 1))


def close_open_boundaries(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Cap every open boundary loop (neck base, mouth opening, eye sockets) with a centroid fan; returns (V', F').

    The caps are consistently wound with the surface, so a head whose only defects are open loops becomes watertight.
    """
    v, f = _as_mesh(vertices, faces)
    directed = {(int(a), int(b)) for tri in f for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))}
    open_edges = [(a, b) for (a, b) in directed if (b, a) not in directed]
    if not open_edges:
        return v, f
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in open_edges:
        parent[find(a)] = find(b)
    loops: dict[int, set[int]] = {}
    for a, b in open_edges:
        loops.setdefault(find(a), set()).update((a, b))
    new_vertices, new_faces = [v], [f]
    centers: dict[int, int] = {}
    cursor = len(v)
    for root, members in loops.items():
        centers[root] = cursor
        new_vertices.append(v[sorted(members)].mean(axis=0, keepdims=True))
        cursor += 1
    caps = np.array([(b, a, centers[find(a)]) for a, b in open_edges], dtype=np.int64)
    new_faces.append(caps)
    return np.vstack(new_vertices), np.vstack(new_faces)


# --------------------------------------------------------------------------- regions
def infer_regions(vertices, *, face_direction=(0.0, 0.0, -1.0), up_direction=(0.0, -1.0, 0.0)) -> dict[str, np.ndarray]:
    """Geometric fallback labelling in head-relative coordinates, for meshes without MakeHuman / FLAME vertex groups.

    Heights are measured from the crown over the head+neck extent ``L``; ``d`` is depth along the face direction relative
    to the depth centre; ``w`` is the absolute lateral offset. Real asset labels (``regions=``) always take precedence.
    """
    v = np.asarray(vertices, dtype=np.float64)
    up = np.asarray(up_direction, np.float64)
    face = np.asarray(face_direction, np.float64)
    up, face = up / np.linalg.norm(up), face / np.linalg.norm(face)
    right = np.cross(up, face)
    h, d, w = v @ up, v @ face, np.abs(v @ right)
    top, extent = float(h.max()), max(float(np.ptp(h)), 1e-9)
    drop = (top - h) / extent                          # 0 at the crown, 1 at the lowest neck vertex
    d_mid = 0.5 * (float(d.max()) + float(d.min()))
    w_max = max(float(w.max()), 1e-9)
    front, lateral = d > d_mid, w > 0.55 * w_max
    return {
        "cranium": np.flatnonzero(drop < 0.28),
        "forehead_dome": np.flatnonzero((drop < 0.34) & front & (w < 0.65 * w_max)),
        "temples": np.flatnonzero((drop > 0.14) & (drop < 0.42) & (w > 0.6 * w_max)),
        "cheekbones": np.flatnonzero((drop > 0.34) & (drop < 0.60) & front & lateral),
        "mandible": np.flatnonzero((drop > 0.50) & (drop < 0.86) & (front | lateral)),
        "submental_neck": np.flatnonzero(drop > 0.78),
    }


def _part_vertices(part) -> np.ndarray | None:
    """Vertex array of an anatomical part given as (V, F), an object with ``vertices``, or a plain (N, 3) array."""
    if part is None:
        return None
    if hasattr(part, "vertices"):
        part = part.vertices
    elif isinstance(part, tuple) and part and not np.isscalar(part[0]):
        part = part[0]
    arr = np.asarray(part, dtype=np.float64)
    return arr if arr.ndim == 2 and arr.shape[1] == 3 and len(arr) else None


def _check_eyeballs(parts: Mapping[str, Any], policy: GuardPolicy) -> list[str]:
    problems: list[str] = []
    eyes = parts.get("eyeballs")
    if eyes is None:
        return ["eyeballs: missing (two eyeball spheres are required)"]
    spheres = list(eyes) if isinstance(eyes, (list, tuple)) and len(eyes) == 2 else None
    if spheres is None:
        # One combined mesh: split by the sign of the lateral coordinate (x in the head frame).
        pts = _part_vertices(eyes)
        if pts is None:
            return ["eyeballs: not a vertex set"]
        left = pts[pts[:, 0] < np.median(pts[:, 0])]
        right = pts[pts[:, 0] >= np.median(pts[:, 0])]
        spheres = [left, right]
    arrays = [_part_vertices(s) for s in spheres]
    if len(arrays) != 2 or any(a is None or len(a) < 8 for a in arrays):
        return ["eyeballs: exactly two eyeball spheres with >= 8 vertices each are required"]
    centres = []
    for index, pts in enumerate(arrays):
        centre = pts.mean(axis=0)
        radii = np.linalg.norm(pts - centre, axis=1)
        mean = float(radii.mean())
        if not policy.eyeball_radius_m[0] <= mean <= policy.eyeball_radius_m[1]:
            problems.append(f"eyeball {index}: radius {mean * 1000:.1f} mm is outside "
                            f"{policy.eyeball_radius_m[0] * 1000:.0f}-{policy.eyeball_radius_m[1] * 1000:.0f} mm")
        if float(radii.std() / max(mean, 1e-9)) > 0.15:
            problems.append(f"eyeball {index}: not spherical (radius spread {radii.std() / mean:.2f})")
        centres.append(centre)
    separation = float(np.linalg.norm(centres[0] - centres[1]))
    if not policy.eyeball_separation_m[0] <= separation <= policy.eyeball_separation_m[1]:
        problems.append(f"eyeballs: inter-ocular distance {separation * 1000:.0f} mm is implausible")
    return problems


def _check_oral_cavity(parts: Mapping[str, Any], head_vertices: np.ndarray, policy: GuardPolicy) -> list[str]:
    cavity = parts.get("oral_cavity")
    if cavity is None:
        # Teeth + tongue submeshes together are an oral cavity in the anatomical pack.
        pieces = [_part_vertices(parts.get(name)) for name in ("upper_teeth", "lower_teeth", "tongue")]
        pieces = [p for p in pieces if p is not None]
        cavity = np.vstack(pieces) if len(pieces) == 3 else None
    pts = _part_vertices(cavity)
    if pts is None or len(pts) < policy.min_oral_vertices:
        return ["oral_cavity: missing (teeth/tongue/cavity geometry with at least "
                f"{policy.min_oral_vertices} vertices is required)"]
    lo, hi = head_vertices.min(axis=0), head_vertices.max(axis=0)
    margin = 0.02 * (hi - lo)
    if np.any(pts.min(axis=0) < lo - margin) or np.any(pts.max(axis=0) > hi + margin):
        return ["oral_cavity: lies outside the head volume"]
    return []


# --------------------------------------------------------------------------- guard
class True3DArchitectureGuard:
    """Runtime enforcement of the true-3D architecture; construct once, call from every render entry point."""

    def __init__(self, policy: GuardPolicy = DEFAULT_POLICY):
        self.policy = policy

    # -- (a) head mesh -------------------------------------------------------
    def inspect_head_mesh(self, vertices, faces, *, regions: Mapping[str, Iterable[int]] | None = None,
                          parts: Mapping[str, Any] | None = None, face_direction=(0.0, 0.0, -1.0),
                          up_direction=(0.0, -1.0, 0.0), require_parts: bool = True,
                          require_watertight: bool = True) -> HeadMeshReport:
        """Diagnose without raising (readiness / describe); structural errors in V/F still raise HeadMeshError."""
        policy = self.policy
        v, f = _as_mesh(vertices, faces)
        boundary, non_manifold, _ = edge_statistics(f, len(v))
        winding = _directed_edge_balance(f, len(v)) if len(f) else 0
        report = HeadMeshReport(len(v), len(f), boundary, non_manifold, boundary == 0 and non_manifold == 0 and winding == 0)
        problems = report.problems
        if len(v) < policy.min_vertices:
            problems.append(f"only {len(v)} vertices (< {policy.min_vertices}); a landmark strip or face-only mesh is not a head")
        if len(f) < 2 * policy.min_vertices - 8 and len(v) >= policy.min_vertices:
            problems.append(f"only {len(f)} faces for {len(v)} vertices")
        referenced = np.zeros(len(v), bool)
        referenced[f.ravel()] = True
        if not referenced.all():
            problems.append(f"{int((~referenced).sum())} vertices are not referenced by any face")
        if require_watertight and not report.watertight:
            problems.append(f"not watertight: {boundary} boundary edge(s), {non_manifold} non-manifold edge(s), "
                            f"{winding} inconsistently wound edge(s); every edge must be shared by exactly two faces")
        # Extents: a frontal strip is flat, a head is a solid.
        extents = np.sort(np.ptp(v, axis=0))[::-1]
        if extents[0] <= 0 or extents[2] / extents[0] < policy.min_planarity:
            problems.append("flat / strip-like geometry: smallest principal extent is "
                            f"{extents[2] / max(extents[0], 1e-12):.2f} of the largest")
        up = np.asarray(up_direction, float) / np.linalg.norm(up_direction)
        face = np.asarray(face_direction, float) / np.linalg.norm(face_direction)
        height = float(np.ptp(v @ up))
        depth = float(np.ptp(v @ face))
        if height <= 0 or depth / height < policy.min_depth_to_height:
            problems.append(f"head depth/height = {depth / max(height, 1e-12):.2f} (< {policy.min_depth_to_height}): "
                            "no cranium behind the face plane")
        d = v @ face
        back = float(np.mean(d < 0.5 * (d.max() + d.min())))
        if back < policy.min_back_fraction:
            problems.append(f"only {back:.0%} of vertices are behind the depth centre (< {policy.min_back_fraction:.0%}): "
                            "no back of the head")
        # Surface-normal azimuth coverage around the vertical axis (area weighted), plus upward and downward faces.
        if len(f) and not problems_has_structural(problems):
            tri = v[f]
            normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(normals, axis=1)
            keep = area > 1e-14
            unit = normals[keep] / area[keep, None]
            right = np.cross(up, face)
            azimuth = np.arctan2(unit @ right, unit @ face)
            hist = np.histogram(azimuth, bins=16, range=(-math.pi, math.pi), weights=area[keep])[0]
            occupied = int(np.count_nonzero(hist > 0.003 * hist.sum()))
            if occupied < policy.min_azimuth_bins:
                problems.append(f"surface covers {occupied}/16 azimuth sectors (< {policy.min_azimuth_bins}): not a closed head")
            elevation = unit @ up
            if not (np.any(elevation > 0.5) and np.any(elevation < -0.3)):
                problems.append("surface has no upward-facing (crown) or downward-facing (chin / neck underside) region")
        # Anatomical regions: labels from the FLAME / MakeHuman vertex groups, else geometric inference.
        labelled = {k: np.asarray(list(val), dtype=np.int64) for k, val in (regions or {}).items()}
        found = {name: labelled[name] for name in REQUIRED_REGIONS if name in labelled}
        if len(found) < len(REQUIRED_REGIONS):
            inferred = infer_regions(v, face_direction=face_direction, up_direction=up_direction)
            for name in REQUIRED_REGIONS:
                found.setdefault(name, inferred[name])
        floor = max(policy.min_region_vertices, int(policy.min_region_fraction * len(v)))
        for name in REQUIRED_REGIONS:
            idx = found[name]
            if len(idx) and (idx.min() < 0 or idx.max() >= len(v)):
                problems.append(f"region '{name}' indexes outside V")
                idx = idx[(idx >= 0) & (idx < len(v))]
            report.region_counts[name] = int(len(idx))
            if len(idx) < floor:
                problems.append(f"region '{name}' has {len(idx)} vertices (< {floor}): missing from the head surface")
        if report.region_counts.get("cranium", 0) and "cranium" in found:
            idx = found["cranium"][(found["cranium"] >= 0) & (found["cranium"] < len(v))]
            if len(idx) and float(np.mean(d[idx] < 0.5 * (d.max() + d.min()))) < 0.15:
                problems.append("cranium has no back-of-head vertices")
        if require_parts:
            supplied = dict(parts or {})
            eye_problems = _check_eyeballs(supplied, policy)
            oral_problems = _check_oral_cavity(supplied, v, policy)
            report.parts["eyeballs"] = "ok" if not eye_problems else "; ".join(eye_problems)
            report.parts["oral_cavity"] = "ok" if not oral_problems else "; ".join(oral_problems)
            problems.extend(eye_problems + oral_problems)
        return report

    def validate_head_mesh(self, vertices, faces, **kwargs) -> HeadMeshReport:
        report = self.inspect_head_mesh(vertices, faces, **kwargs)
        if report.problems:
            details = {"vertex_count": report.vertex_count, "boundary_edges": report.boundary_edges,
                       "non_manifold_edges": report.non_manifold_edges, "region_counts": report.region_counts}
            text = "; ".join(report.problems)
            if all(p.startswith("not watertight") for p in report.problems):
                raise NotWatertightError(text, details)
            raise IncompleteHeadTopologyError(text, details)
        return report

    # -- (b) camera ----------------------------------------------------------
    def validate_camera(self, K, R=None, t=None, *, vertices=None, mesh_z_toward_viewer: bool = False,
                        image_size: tuple[int, int] | None = None, projection_matrix=None) -> CameraReport:
        """K: full 3x3 perspective intrinsics; (R, t): rigid SE(3) pose; every vertex must be in front of the camera.

        ``projection_matrix`` (3x4 P) is accepted for callers that hold P = K[R|t]; an affine P (last row (0,0,0,*)) is rejected.
        ``mesh_z_toward_viewer`` mirrors ``HeadPose.camera_vertices`` for meshes stored with +z toward the viewer.
        """
        policy = self.policy
        k = np.asarray(K, dtype=np.float64)
        if k.shape in ((2, 3), (2, 2), (3, 4)) or k.ndim != 2:
            raise AffineCameraError(f"camera must be a full 3x3 perspective K, got shape {k.shape} "
                                    "(2x3 scaled-orthographic / affine matrices are rejected)")
        if k.shape != (3, 3) or not np.isfinite(k).all():
            raise CameraModelError(f"K must be a finite 3x3 matrix, got {k.shape}")
        if not (abs(k[2, 0]) < 1e-9 and abs(k[2, 1]) < 1e-9 and abs(k[2, 2] - 1.0) < 1e-9) or abs(k[1, 0]) > 1e-9:
            raise AffineCameraError(f"K bottom row must be (0, 0, 1) and K[1,0] zero; got {k[2].tolist()} - not a pin-hole matrix")
        fx, fy, cx, cy, skew = float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2]), float(k[0, 1])
        report = CameraReport(fx, fy, cx, cy, None, None)
        if fx <= 0 or fy <= 0:
            raise CameraModelError(f"focal lengths must be positive (fx={fx:g}, fy={fy:g})")
        if abs(skew) > policy.max_skew_ratio * fx:
            raise CameraModelError(f"skew {skew:g} is not allowed (pixels are square; |skew|/fx must be <= {policy.max_skew_ratio})")
        if not 0.5 <= fx / fy <= 2.0:
            raise CameraModelError(f"fx/fy = {fx / fy:.2f} is not a plausible pixel aspect ratio")
        if image_size is not None:
            width, height = image_size
            if not (0 < cx < width and 0 < cy < height):
                raise CameraModelError(f"principal point ({cx:g}, {cy:g}) lies outside the {width}x{height} image")
            report.horizontal_fov_deg = math.degrees(2.0 * math.atan(width / (2.0 * fx)))
        if projection_matrix is not None:
            p = np.asarray(projection_matrix, dtype=np.float64)
            if p.shape != (3, 4) or not np.isfinite(p).all():
                raise AffineCameraError(f"projection matrix must be a finite 3x4 K[R|t], got {p.shape}")
            if np.linalg.norm(p[2, :3]) < 1e-9:
                raise AffineCameraError("projection matrix last row has no rotation part: this is an affine (orthographic) camera")
        if R is not None or t is not None or vertices is not None:
            if R is None or t is None:
                raise CameraPoseError("a rigid pose needs both R (3x3) and t (3,)")
            r, tv = np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(-1)
            if r.shape != (3, 3) or tv.shape != (3,) or not (np.isfinite(r).all() and np.isfinite(tv).all()):
                raise CameraPoseError("R must be finite 3x3 and t a finite 3-vector (SE(3))")
            if not np.allclose(r @ r.T, np.eye(3), atol=1e-4) or abs(np.linalg.det(r) - 1.0) > 1e-3:
                raise CameraPoseError("R must be orthonormal with determinant +1 (a scaled or reflected R is not SE(3))")
            if vertices is not None:
                v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
                if mesh_z_toward_viewer:
                    v = v * np.array([1.0, 1.0, -1.0])
                z = (v @ r.T + tv)[:, 2]
                report.min_depth = float(z.min())
                if not np.isfinite(z).all() or z.min() <= policy.min_depth_m:
                    raise CameraPoseError(f"head intersects or lies behind the camera (min Z = {z.min():.4f} m); "
                                          "every vertex needs positive depth")
        return report

    # -- (c) rasterizer / shader --------------------------------------------
    def validate_render_config(self, config) -> RenderConfigReport:
        """Perspective, Z-buffered rasterization with 3D surface shading; nothing screen-space or affine."""
        data = _config_dict(config)
        projection = str(data.get("projection", data.get("camera_projection", ""))).strip().lower()
        rasterizer = str(data.get("rasterizer", data.get("rasterization", ""))).strip().lower()
        shader = str(data.get("shader", data.get("shader_mode", ""))).strip().lower()
        report = RenderConfigReport(projection, rasterizer, shader)
        space = str(data.get("space", data.get("render_space", "world"))).strip().lower()
        for label, value in (("projection", projection), ("rasterizer", rasterizer), ("space", space), ("shader", shader)):
            if value in BANNED_PROJECTIONS or any(token in value for token in ("affine", "orthographic", "screen_space", "screen-space")):
                raise RenderConfigError(f"{label}='{value}' is banned: only perspective Z-buffered 3D rendering is allowed",
                                        "RENDER_CONFIG_BANNED_MODE", {"field": label, "value": value})
        banned = sorted(key for key in data if key in BANNED_CONFIG_KEYS and data[key] not in (None, False, "", "off"))
        if banned:
            raise RenderConfigError(f"2D / 2.5D controls are not part of a true-3D render config: {banned}",
                                    "RENDER_CONFIG_BANNED_KEY", {"keys": banned})
        if projection != "perspective":
            raise RenderConfigError(f"projection must be 'perspective', got '{projection or 'unset'}'", "RENDER_CONFIG_PROJECTION")
        if not ("z_buffer" in rasterizer or "zbuffer" in rasterizer or "depth_buffer" in rasterizer) or data.get("depth_test") is False:
            raise RenderConfigError("rasterizer must be a Z-buffered (depth-tested) 3D rasterizer", "RENDER_CONFIG_RASTERIZER")
        pbr_sss = "pbr" in shader and ("subsurface" in shader or "sss" in shader)
        gaussian = "gaussian" in shader
        spherical_harmonics = shader in ("spherical_harmonics", "perspective_sh_lambertian")
        if not (pbr_sss or gaussian or spherical_harmonics):
            raise RenderConfigError("shader must be spherical-harmonics diffuse, PBR with subsurface scattering, or 3D Gaussian",
                                    "RENDER_CONFIG_SHADER", {"shader": shader})
        return report

    # -- aggregate -----------------------------------------------------------
    def assert_true_3d(self, *, mesh: Mapping[str, Any] | None = None, camera: Mapping[str, Any] | None = None,
                       config=None) -> True3DReport:
        """``mesh`` = kwargs of :meth:`validate_head_mesh` (vertices, faces, ...); ``camera`` = kwargs of :meth:`validate_camera`.
        Whatever is supplied is validated; at least one of the three must be supplied."""
        if mesh is None and camera is None and config is None:
            raise True3DArchitectureError("assert_true_3d needs a mesh, a camera or a render config to check")
        mesh_report = self.validate_head_mesh(**dict(mesh)) if mesh is not None else None
        camera_args = dict(camera) if camera is not None else None
        if camera_args is not None and mesh is not None and "vertices" not in camera_args:
            camera_args["vertices"] = mesh["vertices"]
        camera_report = self.validate_camera(**camera_args) if camera_args is not None else None
        config_report = self.validate_render_config(config) if config is not None else None
        return True3DReport(mesh_report, camera_report, config_report)

    def describe(self) -> dict[str, Any]:
        p = self.policy
        return {"required_regions": list(REQUIRED_REGIONS), "required_parts": list(REQUIRED_PARTS),
                "watertight": "every edge shared by exactly two consistently wound faces",
                "min_vertices": p.min_vertices, "camera": "full 3x3 K + rigid SE(3) [R|t], positive depth for all vertices",
                "projection": "perspective", "rasterizer": "Z-buffer",
                "shaders": ["Spherical harmonics diffuse", "PBR + subsurface scattering", "3D Gaussian"],
                "banned": sorted(BANNED_PROJECTIONS)}


def problems_has_structural(problems: Sequence[str]) -> bool:
    return any("flat / strip-like" in p or p.startswith("only") and "vertices" in p for p in problems)


def _config_dict(config) -> dict[str, Any]:
    if config is None:
        raise RenderConfigError("a render config is required")
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "__dict__"):
        return {k: v for k, v in vars(config).items() if not k.startswith("_")}
    raise RenderConfigError(f"unsupported render config type {type(config).__name__}")


# --------------------------------------------------------------------------- module-level API
_GUARD = True3DArchitectureGuard()


def validate_head_mesh(vertices, faces, **kwargs) -> HeadMeshReport:
    return _GUARD.validate_head_mesh(vertices, faces, **kwargs)


def validate_camera(K, R=None, t=None, **kwargs) -> CameraReport:
    return _GUARD.validate_camera(K, R, t, **kwargs)


def validate_render_config(config) -> RenderConfigReport:
    return _GUARD.validate_render_config(config)


def assert_true_3d(*, mesh=None, camera=None, config=None) -> True3DReport:
    return _GUARD.assert_true_3d(mesh=mesh, camera=camera, config=config)


def renderer_is_wired() -> bool:
    try:
        module = importlib.import_module(RENDERER_MODULE)
        return bool(module.is_wired())
    except Exception:
        return False


def require_true3d_renderer(what: str = "render") -> None:
    """Every live / offline render entry point calls this first: no 3D renderer, no output, no silent 2D fallback."""
    if not renderer_is_wired():
        raise True3DRendererUnavailable(
            f"{NOT_WIRED_MESSAGE}: cannot run '{what}'. The 2D / 2.5D swap and restoration paths were removed and the "
            "FullHeadEngine render path (perspective K[R|t], Z-buffer, PBR-SSS / 3D Gaussian) is not connected yet.",
            {"renderer_module": RENDERER_MODULE, "operation": what})


def solve_dynamic_intrinsics(landmarks_2d, *, width: int, height: int, canonical_3d=None,
                             focal_bounds_mm: tuple[float, float] = (18.0, 135.0), initial_focal_px: float | None = None):
    """Dynamic focal-length (FOV) solver: reuses ``reconstruction.dynamic_camera`` and returns ``(CameraIntrinsics, Pose3D)``.

    The solved K is passed through :func:`validate_camera` so an implausible solution never reaches the renderer.
    """
    from roop_3d.reconstruction.dynamic_camera import solve_dynamic_camera_intrinsics
    intrinsics, pose = solve_dynamic_camera_intrinsics(
        np.asarray(landmarks_2d, dtype=np.float64), canonical_3d=canonical_3d, width=width, height=height,
        focal_bounds_mm=focal_bounds_mm, initial_focal_px=initial_focal_px)
    validate_camera(intrinsics.matrix(), image_size=(width, height))
    return intrinsics, pose


def intrinsics_from_fov(horizontal_fov_deg: float, width: int, height: int) -> np.ndarray:
    """K for a horizontal field of view (square pixels, principal point at the image centre)."""
    if not 1.0 <= horizontal_fov_deg <= 170.0:
        raise CameraModelError(f"horizontal FOV {horizontal_fov_deg} deg is outside 1..170")
    fx = width / (2.0 * math.tan(math.radians(horizontal_fov_deg) / 2.0))
    return np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)


__all__ = [
    "AffineCameraError", "CameraModelError", "CameraPoseError", "CameraReport", "GuardPolicy", "HeadMeshError",
    "HeadMeshReport", "IncompleteHeadTopologyError", "NotWatertightError", "REQUIRED_PARTS", "REQUIRED_REGIONS",
    "RenderConfigError", "RenderConfigReport", "True3DArchitectureError", "True3DArchitectureGuard", "True3DReport",
    "True3DRendererUnavailable", "assert_true_3d", "close_open_boundaries", "edge_statistics", "infer_regions",
    "intrinsics_from_fov", "renderer_is_wired", "require_true3d_renderer", "solve_dynamic_intrinsics", "validate_camera",
    "validate_head_mesh", "validate_render_config",
]
