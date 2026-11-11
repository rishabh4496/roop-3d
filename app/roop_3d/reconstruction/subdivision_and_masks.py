# SPDX-License-Identifier: MIT
"""Procedural craniofacial regions and silhouette-focused triangle subdivision.

The region definitions are derived from mesh coordinates and optional landmark
anchors. They do not ship or load third-party vertex-mask resources. Triangle
meshes use one Loop subdivision step; a view-dependent contour is smoothed as
a boundary while interior vertices keep their fitted positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import cv2
import numpy as np


REGION_NAMES = (
    "scalp", "forehead", "temples_ears", "periorbital_eyes", "nose",
    "malar_cheeks", "lips_mouth", "mandible_jaw_chin", "submental_neck",
)

# Roughness is perceptual linear roughness in the common metallic/roughness PBR
# convention. The T-zone is smoother; cheek and neck values retain softer scatter.
REGION_PBR: dict[str, dict[str, float]] = {
    "scalp": {"roughness": 0.43, "specular": 0.42, "subsurface": 0.10},
    "forehead": {"roughness": 0.32, "specular": 0.52, "subsurface": 0.10},
    "temples_ears": {"roughness": 0.40, "specular": 0.46, "subsurface": 0.14},
    "periorbital_eyes": {"roughness": 0.36, "specular": 0.48, "subsurface": 0.12},
    "nose": {"roughness": 0.28, "specular": 0.56, "subsurface": 0.08},
    "malar_cheeks": {"roughness": 0.48, "specular": 0.40, "subsurface": 0.18},
    "lips_mouth": {"roughness": 0.34, "specular": 0.50, "subsurface": 0.12},
    "mandible_jaw_chin": {"roughness": 0.44, "specular": 0.42, "subsurface": 0.14},
    "submental_neck": {"roughness": 0.52, "specular": 0.36, "subsurface": 0.20},
}

DEFAULT_REGION_FEATHER_PX: dict[str, float] = {
    "scalp": 3.0, "forehead": 2.0, "temples_ears": 4.0,
    "periorbital_eyes": 1.5, "nose": 1.0, "malar_cheeks": 3.0,
    "lips_mouth": 1.5, "mandible_jaw_chin": 4.0, "submental_neck": 6.0,
}


@dataclass(frozen=True)
class AnatomicalVertexGroups:
    """A complete partition plus soft per-region weights in input vertex order."""

    topology_id: str
    groups: Mapping[str, np.ndarray]  # region -> indices, disjoint and exhaustive
    weights: np.ndarray               # (V, 9), columns follow REGION_NAMES
    labels: np.ndarray                # (V,), integer region column
    roughness: np.ndarray             # (V,), interpolated by soft weights
    specular: np.ndarray              # (V,), interpolated by soft weights
    subsurface: np.ndarray            # (V,), interpolated by soft weights
    feather_px: Mapping[str, float]

    def weight(self, region: str) -> np.ndarray:
        try:
            return self.weights[:, REGION_NAMES.index(region)]
        except ValueError:
            raise KeyError(region) from None


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _band(value: np.ndarray, low: float, high: float, softness: float) -> np.ndarray:
    return (_smoothstep(low - softness, low + softness, value)
            * (1.0 - _smoothstep(high - softness, high + softness, value)))


def _landmark_y(vertices: np.ndarray, landmarks: Mapping[str, np.ndarray] | None) -> tuple[float | None, float | None, float | None]:
    if not landmarks:
        return None, None, None
    # MakeHuman's five-point order is right eye, left eye, nose, right mouth, left mouth.
    for key in ("5pt", "68pt"):
        raw = landmarks.get(key)
        if raw is None:
            continue
        indices = np.asarray(raw, np.int64).reshape(-1)
        if key == "5pt" and len(indices) >= 5 and np.all((indices >= 0) & (indices < len(vertices))):
            return (float(vertices[indices[:2], 1].mean()), float(vertices[indices[2], 1]),
                    float(vertices[indices[3:5], 1].mean()))
        if key == "68pt" and len(indices) >= 68 and np.all((indices >= 0) & (indices < len(vertices))):
            return (float(vertices[indices[36:48], 1].mean()), float(vertices[30, 1]),
                    float(vertices[48:68, 1].mean()))
    return None, None, None


def segment_head_vertices(
    vertices: np.ndarray,
    *,
    topology_id: str = "makehuman_cc0",
    landmarks: Mapping[str, np.ndarray] | None = None,
    feather_px: Mapping[str, float] | None = None,
) -> AnatomicalVertexGroups:
    """Assign every vertex to one of nine soft anatomical regions.

    Coordinate windows are anchored to the eye, nose and mouth landmarks when
    available; otherwise they use robust mesh bounds. The x/y frame is the
    repository's image frame (x right, y down), shared by both supported heads.
    """
    v = np.asarray(vertices, np.float64)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 9 or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite (V, 3) array with at least nine vertices")
    if topology_id not in {"makehuman_cc0", "flame2023_open"}:
        raise ValueError("topology_id must be makehuman_cc0 or flame2023_open")
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]
    x_lo, x_hi = np.quantile(x, [0.02, 0.98])
    y_lo, y_hi = np.quantile(y, [0.01, 0.99])
    width = max(float(x_hi - x_lo), 1e-6)
    height = max(float(y_hi - y_lo), 1e-6)
    x_mid = float(np.median(x))
    xn = np.clip((x - x_mid) / (width * 0.5), -1.5, 1.5)

    eye_y, nose_y, mouth_y = _landmark_y(v, landmarks)
    if eye_y is None or mouth_y is None or mouth_y <= eye_y:
        eye_y, nose_y, mouth_y = (float(y_lo + 0.43 * height), float(y_lo + 0.57 * height),
                                  float(y_lo + 0.68 * height))
    face_unit = max(float(mouth_y - eye_y), height * 0.12)
    nose_y = float(nose_y if nose_y is not None and eye_y < nose_y < mouth_y + face_unit * 0.2
                   else eye_y + 0.52 * face_unit)
    # Anchored forehead and chin limits leave the back of the cranium and neck in
    # deterministic groups while matching facial proportions on both topologies.
    forehead_top = max(float(y_lo), eye_y - 0.92 * face_unit)
    chin_y = min(float(y_hi), mouth_y + 0.90 * face_unit)
    yn = (y - forehead_top) / max(chin_y - forehead_top, height * 0.18)
    # Make the viewer-facing plane contribute to the nose score without using a
    # signed z convention (both supported topologies can be transformed upstream).
    zlo, zhi = np.quantile(z, [0.05, 0.95])
    front = np.clip((zhi - z) / max(float(zhi - zlo), 1e-6), 0.0, 1.0)
    soft_y = max(0.025, 0.045 * height / max(chin_y - forehead_top, 1e-6))
    ax = np.abs(xn)

    # Scores overlap intentionally; normalized scores become feather-ready vertex
    # weights and argmax gives a non-overlapping conventional vertex-group layout.
    scores = np.column_stack((
        0.03 + (1.0 - _smoothstep(-0.02, 0.10, yn)),
        _band(yn, 0.03, 0.30, soft_y) * (1.0 - _smoothstep(0.75, 1.05, ax)),
        _band(yn, 0.06, 0.86, soft_y) * _smoothstep(0.58, 0.92, ax),
        np.exp(-0.5 * ((yn - 0.33) / 0.11) ** 2) * _band(ax, 0.16, 0.74, 0.12),
        _band(yn, 0.27, 0.62, soft_y) * (1.0 - _smoothstep(0.20, 0.39, ax)) * (0.65 + 0.35 * front),
        _band(yn, 0.37, 0.67, soft_y) * _band(ax, 0.20, 0.84, 0.15),
        np.exp(-0.5 * ((yn - 0.64) / 0.09) ** 2) * (1.0 - _smoothstep(0.22, 0.48, ax)),
        _band(yn, 0.59, 1.07, soft_y) * (1.0 - _smoothstep(0.84, 1.12, ax)),
        _smoothstep(0.91, 1.14, yn) * (1.0 - _smoothstep(0.55, 0.93, ax)),
    ))
    scores = np.maximum(scores, 1e-5)
    # The scalp score is a catch-all only above the forehead; reduce it elsewhere
    # so it cannot dominate facial labels on vertices near a region boundary.
    scores[:, 0] *= (1.0 - _smoothstep(0.03, 0.16, yn)) + 1e-4
    weights = scores / scores.sum(axis=1, keepdims=True)
    labels = np.argmax(weights, axis=1).astype(np.int32)
    groups = {name: np.flatnonzero(labels == i).astype(np.int64) for i, name in enumerate(REGION_NAMES)}

    config = dict(DEFAULT_REGION_FEATHER_PX)
    if feather_px:
        unknown = set(feather_px) - set(REGION_NAMES)
        if unknown:
            raise ValueError(f"unknown region feather settings: {sorted(unknown)}")
        for name, amount in feather_px.items():
            if not np.isfinite(amount) or amount < 0 or amount > 64:
                raise ValueError("per-region feathering must be finite and within [0, 64] pixels")
            config[name] = float(amount)
    channels = {key: np.asarray([REGION_PBR[n][key] for n in REGION_NAMES], np.float64) for key in ("roughness", "specular", "subsurface")}
    return AnatomicalVertexGroups(
        topology_id=topology_id, groups=groups, weights=weights.astype(np.float32), labels=labels,
        roughness=(weights @ channels["roughness"]).astype(np.float32),
        specular=(weights @ channels["specular"]).astype(np.float32),
        subsurface=(weights @ channels["subsurface"]).astype(np.float32), feather_px=config,
    )


def segment_topology_mesh(mesh, *, feather_px: Mapping[str, float] | None = None) -> AnatomicalVertexGroups:
    """Convenience adapter for a loaded ``TopologyMesh`` from reconstruction.topology."""
    topology_id = getattr(mesh, "topology_id", None)
    if topology_id not in {"makehuman_cc0", "flame2023_open"}:
        raise ValueError("mesh must be a registered MakeHuman CC0 or FLAME 2023 Open topology")
    return segment_head_vertices(mesh.vertices, topology_id=topology_id,
                                 landmarks=getattr(mesh, "landmarks", None), feather_px=feather_px)


def feather_region_swap_masks(
    masks: Mapping[str, np.ndarray],
    feather_px: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Gaussian-soften image masks with independent per-region feather radii."""
    config = dict(DEFAULT_REGION_FEATHER_PX)
    if feather_px:
        for name, value in feather_px.items():
            if name not in REGION_NAMES or not np.isfinite(value) or not 0 <= value <= 64:
                raise ValueError("feather_px contains an unknown region or invalid radius")
            config[name] = float(value)
    output: dict[str, np.ndarray] = {}
    shape = None
    for name, raw in masks.items():
        if name not in REGION_NAMES:
            raise ValueError(f"unknown anatomical region {name!r}")
        mask = np.asarray(raw, np.float32)
        if mask.ndim != 2 or not np.isfinite(mask).all():
            raise ValueError("region swap masks must be finite two-dimensional arrays")
        if shape is None:
            shape = mask.shape
        elif shape != mask.shape:
            raise ValueError("all region swap masks must share one image shape")
        mask = np.clip(mask, 0.0, 1.0)
        radius = config[name]
        if radius:
            # A Gaussian has a cosine-like soft shoulder without introducing a
            # hard radius discontinuity into small per-region facial masks.
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(0.25, radius * 0.5),
                                    sigmaY=max(0.25, radius * 0.5), borderType=cv2.BORDER_REPLICATE)
        output[name] = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return output


def _edge_table(faces: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int], list[int]]]:
    adjacent: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for a, b in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge = (int(min(a, b)), int(max(a, b)))
            adjacent.setdefault(edge, []).append(face_index)
    return np.asarray(list(adjacent), np.int64), adjacent


def _silhouette_edges(vertices: np.ndarray, faces: np.ndarray, edge_indices: np.ndarray,
                      adjacent: Mapping[tuple[int, int], Sequence[int]], view_direction: np.ndarray) -> np.ndarray:
    tri = vertices[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    facing = normals @ (view_direction / max(float(np.linalg.norm(view_direction)), 1e-12))
    selected = []
    for edge in edge_indices:
        incident = adjacent[(int(edge[0]), int(edge[1]))]
        if len(incident) == 1 or (min(facing[incident]) <= 0 <= max(facing[incident])):
            selected.append((int(edge[0]), int(edge[1])))
    return np.asarray(selected, np.int64).reshape(-1, 2)


@dataclass(frozen=True)
class SubdividedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    vertex_weights: np.ndarray | None = None
    silhouette_edges: np.ndarray | None = None
    device: str = "cpu"
    uvs: np.ndarray | None = None


def subdivide_silhouette(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    silhouette_edges: np.ndarray | None = None,
    view_direction: Sequence[float] = (0.0, 0.0, -1.0),
    vertex_weights: np.ndarray | None = None,
    uvs: np.ndarray | None = None,
    device: str = "auto",
) -> SubdividedMesh:
    """Apply one Loop step, using a smooth boundary rule on the outer contour.

    ``device='cuda'`` uses Torch tensor kernels when CUDA is available;
    ``device='auto'`` selects CUDA when present and otherwise remains NumPy-only.
    All triangles are split to avoid T-junctions, while old non-contour vertices
    stay fixed so fitted interior details do not shrink.
    """
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("subdivision expects vertices (V,3) and triangular faces (F,3)")
    if not len(v) or not len(f) or not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("subdivision mesh is empty, non-finite, or has invalid face indices")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    use_cuda = False
    if device in {"auto", "cuda"}:
        try:
            import torch
            use_cuda = bool(torch.cuda.is_available())
        except ImportError:
            if device == "cuda":
                raise RuntimeError("CUDA subdivision requires the optional Torch package") from None
    if device == "cuda" and not use_cuda:
        raise RuntimeError("CUDA subdivision was requested but no CUDA device is available")

    edge_indices, adjacent = _edge_table(f)
    if silhouette_edges is None:
        view = np.asarray(view_direction, np.float64)
        if view.shape != (3,) or not np.isfinite(view).all() or np.linalg.norm(view) <= 1e-8:
            raise ValueError("view_direction must be a nonzero finite 3-vector")
        contour = _silhouette_edges(v, f, edge_indices, adjacent, view)
    else:
        contour = np.asarray(silhouette_edges, np.int64).reshape(-1, 2)
        if len(contour) and (contour.min() < 0 or contour.max() >= len(v)):
            raise ValueError("silhouette edge index is outside the vertex array")
    contour_set = {tuple(sorted((int(a), int(b)))) for a, b in contour}
    contour_neighbors: dict[int, set[int]] = {}
    for a, b in contour_set:
        contour_neighbors.setdefault(a, set()).add(b)
        contour_neighbors.setdefault(b, set()).add(a)
    updated = v.copy()
    for index, neighbors in contour_neighbors.items():
        if len(neighbors) == 2:
            a, b = sorted(neighbors)
            updated[index] = 0.75 * v[index] + 0.125 * (v[a] + v[b])

    # GPU path handles edge interpolation and old-vertex contour updates; the
    # compact combinatorial index construction above stays on the CPU.
    if use_cuda:
        import torch
        vt = torch.as_tensor(v, device="cuda")
        oldt = torch.as_tensor(updated, device="cuda")
        edge_t = torch.as_tensor(edge_indices, device="cuda", dtype=torch.long)
        edge_points = (vt[edge_t[:, 0]] + vt[edge_t[:, 1]]) * 0.5
        # Interior edges use the Loop 3/8 + 1/8 rule. Contour edges use the
        # straight midpoint to preserve the exact smooth boundary curve.
        for eidx, pair in enumerate(edge_indices):
            incident = adjacent[(int(pair[0]), int(pair[1]))]
            if tuple(sorted((int(pair[0]), int(pair[1])))) in contour_set or len(incident) != 2:
                continue
            opposite = [int(k) for face_id in incident for k in f[face_id]
                        if k != pair[0] and k != pair[1]]
            if len(opposite) == 2:
                edge_points[eidx] = (3.0 * (vt[pair[0]] + vt[pair[1]]) + vt[opposite[0]] + vt[opposite[1]]) / 8.0
        points = torch.cat((oldt, edge_points), dim=0).cpu().numpy()
    else:
        edge_points = (v[edge_indices[:, 0]] + v[edge_indices[:, 1]]) * 0.5
        for eidx, pair in enumerate(edge_indices):
            incident = adjacent[(int(pair[0]), int(pair[1]))]
            if tuple(sorted((int(pair[0]), int(pair[1])))) in contour_set or len(incident) != 2:
                continue
            opposite = [int(k) for face_id in incident for k in f[face_id]
                        if k != pair[0] and k != pair[1]]
            if len(opposite) == 2:
                edge_points[eidx] = (3.0 * (v[pair[0]] + v[pair[1]]) + v[opposite[0]] + v[opposite[1]]) / 8.0
        points = np.concatenate((updated, edge_points), axis=0)

    edge_id = {tuple(map(int, edge)): len(v) + i for i, edge in enumerate(edge_indices)}
    quads: list[tuple[int, int, int, int]] = []
    for a, b, c in f:
        ab = edge_id[(min(int(a), int(b)), max(int(a), int(b)))]
        bc = edge_id[(min(int(b), int(c)), max(int(b), int(c)))]
        ca = edge_id[(min(int(c), int(a)), max(int(c), int(a)))]
        quads.extend(((int(a), ab, ca), (int(b), bc, ab), (int(c), ca, bc), (ab, bc, ca)))
    new_faces = np.asarray(quads, np.int64)

    new_weights = None
    if vertex_weights is not None:
        weights = np.asarray(vertex_weights, np.float32)
        if weights.ndim != 2 or weights.shape[0] != len(v) or not np.isfinite(weights).all():
            raise ValueError("vertex_weights must be a finite (V, K) array")
        ew = (weights[edge_indices[:, 0]] + weights[edge_indices[:, 1]]) * 0.5
        new_weights = np.concatenate((weights, ew), axis=0)
        total = new_weights.sum(axis=1, keepdims=True)
        new_weights = np.divide(new_weights, total, out=np.zeros_like(new_weights), where=total > 1e-8)
    new_uvs = None
    if uvs is not None:
        uv = np.asarray(uvs, np.float32)
        if uv.ndim != 2 or uv.shape[0] != len(v) or uv.shape[1] not in (2, 3) or not np.isfinite(uv).all():
            raise ValueError("uvs must be a finite (V, 2) or (V, 3) array")
        new_uvs = np.concatenate((uv, (uv[edge_indices[:, 0]] + uv[edge_indices[:, 1]]) * 0.5), axis=0)
    return SubdividedMesh(points.astype(np.float32), new_faces, new_weights, contour.copy(),
                          "cuda" if use_cuda else "cpu", new_uvs)
