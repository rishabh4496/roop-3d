"""Deterministic refinement of an already-pruned anatomical face surface.

No new anatomy, learned detail or appearance is invented here. Shared edge
midpoints give source and target the same correspondence indices, then genuine
UV displacement is sampled at every refined vertex instead of only at native
3DMM vertices. The native BFM/FLAME topology and its masks remain unchanged.
"""
from __future__ import annotations

import numpy as np

from roop_3d.reconstruction.geometry_types import Mesh3D


def sample_uv_displacement(displacement, uvs):
    """Bilinear bottom-left UV sampling without OpenCV's 32767-row limit."""
    displacement = np.asarray(displacement, np.float32)
    uvs = np.asarray(uvs, np.float32)
    if (displacement.ndim != 2 or min(displacement.shape) < 2
            or not np.isfinite(displacement).all()):
        raise ValueError("Displacement must be a finite 2D map of at least 2x2")
    if (uvs.ndim != 2 or uvs.shape[1] != 2 or not np.isfinite(uvs).all()
            or np.any((uvs < 0) | (uvs > 1))):
        raise ValueError("Displacement UVs must be finite [N,2] in [0,1]")
    height, width = displacement.shape
    x = uvs[:, 0] * (width-1)
    y = (1-uvs[:, 1]) * (height-1)
    x0, y0 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    a, b = x-x0, y-y0
    return ((1-a)*(1-b)*displacement[y0, x0] + a*(1-b)*displacement[y0, x1]
            + (1-a)*b*displacement[y1, x0] + a*b*displacement[y1, x1]).astype(np.float32)


def detail_levels(requested, profile):
    """Bound refinement independently of the inference provider/VRAM arena."""
    if profile not in ("cpu", "6gb", "12gb"):
        raise ValueError("Unsupported dense face hardware profile")
    if type(requested) is not int or not 0 <= requested <= 2:
        raise ValueError("detail_subdivisions must be an integer in 0..2")
    return min(requested, 2 if profile == "12gb" else 1)


class DisplacementSurface:
    """Reusable shared-edge subdivision plan for a compact face mesh.

    Linear subdivision preserves the coarse anatomy and oriented boundaries.
    It never closes holes, welds UV seams, extrudes a neck or adds skull faces.
    Input UVs must already be normalized to the model's per-vertex chart, as
    required by the bundle contract. Refinement is not a substitute for a
    higher-resolution or more accurate learned displacement map.
    """
    def __init__(self, faces, uvs, *, levels=0):
        if type(levels) is not int or not 0 <= levels <= 2:
            raise ValueError("Face displacement refinement must be in 0..2")
        self.uvs = np.asarray(uvs, np.float32).copy()
        if (self.uvs.ndim != 2 or self.uvs.shape[1] != 2 or not len(self.uvs)
                or not np.isfinite(self.uvs).all() or np.any((self.uvs < 0) | (self.uvs > 1))):
            raise ValueError("Refinement requires finite per-vertex UVs in [0,1]")
        raw_faces = np.asarray(faces)
        if (raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces)
                or raw_faces.dtype.kind not in "iu" or np.any(raw_faces < 0)
                or np.any(raw_faces >= len(self.uvs))
                or np.any(np.sort(raw_faces, axis=1)[:, 1:] == np.sort(raw_faces, axis=1)[:, :-1])):
            raise ValueError("Refinement requires valid non-repeated triangular indices")
        if len(raw_faces) * 4**levels > 2_000_000 or len(self.uvs) > 100_000:
            raise ValueError("Face displacement refinement exceeds the geometry budget")
        self.base_faces = raw_faces.astype(np.int32, copy=True)
        self.faces = self.base_faces
        self.base_vertex_count = len(self.uvs)
        self.levels = levels
        self.edges = []
        for _ in range(levels):
            # np.unique gives a stable order independent of source/target shape.
            edges = np.sort(np.concatenate((self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                                             self.faces[:, [2, 0]])), axis=1)
            edges, inverse = np.unique(edges, axis=0, return_inverse=True)
            ab, bc, ca = inverse.reshape(3, -1) + len(self.uvs)
            a, b, c = self.faces.T
            self.faces = np.concatenate((np.column_stack((a, ab, ca)), np.column_stack((ab, b, bc)),
                                         np.column_stack((ca, bc, c)), np.column_stack((ab, bc, ca)))).astype(np.int32)
            self.uvs = np.concatenate((self.uvs, self.uvs[edges].mean(1)))
            self.edges.append(edges)
        self.vertex_count = len(self.uvs)

    def interpolate(self, values):
        values = np.asarray(values, np.float32)
        if (values.ndim < 2 or len(values) != self.base_vertex_count
                or not np.isfinite(values).all()):
            raise ValueError("Refinement values must match the compact native surface")
        for edges in self.edges:
            values = np.concatenate((values, values[edges].mean(1)))
        return values

    def displace(self, vertices_m, displacement_uv_m):
        """Sample real detail along interpolated smooth canonical normals."""
        vertices_m = np.asarray(vertices_m, np.float32)
        if vertices_m.shape != (self.base_vertex_count, 3) or not np.isfinite(vertices_m).all():
            raise ValueError("Displacement vertices must match the compact native surface")
        normals = self.interpolate(Mesh3D(vertices_m, self.base_faces).normals)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, norms, out=np.zeros_like(normals), where=norms > 1e-12)
        displacement = sample_uv_displacement(displacement_uv_m, self.uvs)
        return self.interpolate(vertices_m) + normals * displacement[:, None]
