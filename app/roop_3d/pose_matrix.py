"""Stage 13 frozen, continuously interpolated source-pose identities.

The long-lived ``faceset.pose_matrix`` remains the Stage 0 compatibility format. This
module builds the render-session view of that data: nine yaw/pitch cells plus optional
96-bin sphere samples, pre-projected INSwapper inputs, and roll compensation rotations.
All arrays exposed by a matrix are immutable. Build the matrix before frame 0 and use
``lookup`` for every subsequent frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import Pose3D, as_pose3d
from roop_3d.faceset.identity_blend import IdentityBlendSpec, blend_embeddings
from roop_3d.faceset.schemas import FaceSetDocument

YAW_EDGES = (-25.0, 25.0)
PITCH_EDGES = (-15.0, 15.0)
YAW_ANCHORS = (-75.0, 0.0, 75.0)
PITCH_ANCHORS = (-35.0, 0.0, 35.0)
from roop_3d.reconstruction.pose_bins import COVERAGE_PITCH_BINS as PITCH_BINS, COVERAGE_YAW_BINS as YAW_BINS, coverage_bin
_ROLL_DEGREES = np.arange(-180, 181, dtype=np.int16)


def _unit(value: Sequence[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (512,) or not np.isfinite(vector).all() or norm < 1e-9:
        raise DomainError("IDENTITY_EMBEDDING_INVALID", "Pose identity must be a finite nonzero 512-D vector", 422)
    return vector / norm


def _readonly(value: np.ndarray, dtype=np.float32) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=dtype)
    result.setflags(write=False)
    return result


def _slerp(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
    """Stable spherical interpolation for unit identity vectors."""
    t = float(np.clip(amount, 0.0, 1.0))
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1e-7:
        return _unit((1.0 - t) * a + t * b)
    if cosine < -0.9999:
        # Embeddings should not be antipodal; choose a deterministic orthogonal path if corrupt input reaches here.
        axis = np.zeros_like(a)
        axis[int(np.argmin(np.abs(a)))] = 1.0
        tangent = axis - float(axis @ a) * a
        tangent = _unit(tangent)
        return _unit(math.cos(math.pi * t) * a + math.sin(math.pi * t) * tangent)
    return _unit((math.sin((1.0 - t) * angle) * a + math.sin(t * angle) * b) / math.sin(angle))


def _pose_group(yaw: float, pitch: float) -> tuple[int, int]:
    yi = 0 if yaw < YAW_EDGES[0] else 2 if yaw > YAW_EDGES[1] else 1
    # Public / standard pitch: negative = looking down, positive = looking up.
    pi = 0 if pitch < PITCH_EDGES[0] else 2 if pitch > PITCH_EDGES[1] else 1
    return yi, pi


def _sphere_bin(yaw: float, pitch: float) -> int:
    return coverage_bin(float(yaw), float(pitch))


def _project_many(vectors: np.ndarray, projection: np.ndarray | None, device: str) -> tuple[np.ndarray, str]:
    if projection is None:
        return _readonly(vectors), "identity"
    matrix = np.asarray(projection, np.float32)
    if matrix.shape != (512, 512) or not np.isfinite(matrix).all():
        raise DomainError("SWAPPER_PROJECTION_INVALID", "EMAP must be a finite 512x512 matrix", 503)
    actual_device = "cpu"
    result = None
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.device("cuda")
                projected = torch.as_tensor(vectors, dtype=torch.float32, device=device_name) @ torch.as_tensor(
                    matrix, dtype=torch.float32, device=device_name)
                projected = torch.nn.functional.normalize(projected, dim=1)
                result = projected.detach().cpu().numpy()
                actual_device = "cuda"
        except (ImportError, RuntimeError, ValueError):
            result = None
    if result is None:
        result = vectors @ matrix
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-9)
    return _readonly(result), actual_device


@dataclass(frozen=True)
class FrozenPoseCell:
    yaw_index: int
    pitch_index: int
    pose_selected_vector: np.ndarray
    projected_vector: np.ndarray
    view_index: int
    support: int
    confidence: float
    filled_from: tuple[int, int] | None = None


@dataclass(frozen=True)
class FrozenPoseSelection:
    vector: np.ndarray
    pose_selected_vector: np.ndarray
    projected_vector: np.ndarray
    view_index: int
    confidence: float
    weights: tuple[tuple[int, int, float], ...]
    source: str


class FrozenSourcePoseMatrix:
    """Pre-frozen source identities with smooth bilinear spherical lookup.

    ``from_document`` accepts an already validated ``.fsz`` document. For multi-source
    identity recipes, use ``from_documents`` so blending happens once per pose cell.
    If ``device='cuda'`` and local PyTorch exposes CUDA, EMAP products are computed on
    that device during construction; CPU-only installs retain the same frozen contract.
    """

    def __init__(self, *, faceset_id: str, canonical: np.ndarray, cell_vectors: np.ndarray,
                 cell_views: np.ndarray, cell_support: np.ndarray, cell_confidence: np.ndarray,
                 sphere_vectors: np.ndarray, sphere_views: np.ndarray, sphere_support: np.ndarray,
                 projection: np.ndarray | None = None, reference_blend: float = 0.15, device: str = "cpu",
                 mode: str = "single"):
        if not 0.0 <= reference_blend <= 0.25:
            raise ValueError("reference_blend must be within 0..0.25")
        if device not in ("cpu", "cuda"):
            raise ValueError("device must be cpu or cuda")
        self.faceset_id, self.mode = str(faceset_id), str(mode)
        self.canonical = _readonly(_unit(canonical))
        self.reference_blend = float(reference_blend)
        self.projection = None if projection is None else _readonly(np.asarray(projection, np.float32))
        if self.projection is not None and self.projection.shape != (512, 512):
            raise DomainError("SWAPPER_PROJECTION_INVALID", "EMAP must have shape 512x512", 503)

        raw_cells = np.asarray(cell_vectors, np.float32).reshape(3, 3, 512)
        raw_sphere = np.asarray(sphere_vectors, np.float32).reshape(PITCH_BINS, YAW_BINS, 512)
        selected_cells, selected_sphere = [], []
        for yi in range(3):
            for pi in range(3):
                t = float(reference_blend) * float(cell_confidence[yi, pi])
                selected_cells.append(_slerp(self.canonical, _unit(raw_cells[yi, pi]), t))
        for pi in range(PITCH_BINS):
            for yi in range(YAW_BINS):
                t = float(reference_blend) * float(self._sphere_confidence(sphere_support[pi, yi]))
                selected_sphere.append(_slerp(self.canonical, _unit(raw_sphere[pi, yi]), t))
        selected_cells = np.asarray(selected_cells, np.float32)
        selected_sphere = np.asarray(selected_sphere, np.float32)
        projected_cells, actual_device = _project_many(selected_cells, self.projection, device)
        projected_sphere, sphere_device = _project_many(selected_sphere, self.projection, device)
        self.projection_device = "cuda" if actual_device == sphere_device == "cuda" else actual_device

        cells = []
        for yi in range(3):
            for pi in range(3):
                index = yi * 3 + pi
                cells.append(FrozenPoseCell(yi, pi, _readonly(selected_cells[index]),
                    _readonly(projected_cells[index]), int(cell_views[yi, pi]), int(cell_support[yi, pi]),
                    float(cell_confidence[yi, pi]), None))
        self.cells = tuple(cells)
        self._cell_vectors = _readonly(raw_cells.reshape(9, 512))
        self._cell_selected = _readonly(selected_cells)
        self._cell_projected = _readonly(projected_cells)
        self._cell_views = _readonly(cell_views, np.int32)
        self._cell_confidence = _readonly(cell_confidence)
        self._sphere_vectors = _readonly(raw_sphere.reshape(96, 512))
        self._sphere_selected = _readonly(selected_sphere)
        self._sphere_projected = _readonly(projected_sphere)
        self._sphere_views = _readonly(sphere_views, np.int32)
        self._sphere_support = _readonly(sphere_support, np.int32)
        self._sphere_confidence_values = _readonly(np.where(np.asarray(sphere_support) > 0, 1.0, 0.5))
        self.fine_bin_count = int(np.count_nonzero(np.asarray(sphere_support) > 0))
        self.uses_fine_bins = self.fine_bin_count >= 3
        self.roll_rotations = self._build_roll_rotations()
        digest = hashlib.sha256()
        digest.update(self.faceset_id.encode())
        digest.update(self.canonical.tobytes())
        digest.update(self._cell_selected.tobytes())
        digest.update(self._sphere_selected.tobytes())
        digest.update(self.roll_rotations.tobytes())
        if self.projection is not None:
            digest.update(self.projection.tobytes())
        self.fingerprint = digest.hexdigest()

    @staticmethod
    def _sphere_confidence(support: int) -> float:
        return 1.0 if int(support) > 0 else 0.5

    @staticmethod
    def _build_roll_rotations() -> np.ndarray:
        rotations = []
        for degrees in _ROLL_DEGREES:
            radians = math.radians(-float(degrees))
            c, s = math.cos(radians), math.sin(radians)
            # OpenCV image coordinates (x right, y down), counter-rotation by target roll.
            rotations.append(((c, s), (-s, c)))
        return _readonly(np.asarray(rotations), np.float64)

    @classmethod
    def from_document(cls, document: FaceSetDocument, *, projection: np.ndarray | None = None,
                      reference_blend: float = 0.15, device: str = "cpu") -> "FrozenSourcePoseMatrix":
        if not document.views:
            raise DomainError("FACESET_EMPTY", "A frozen pose matrix needs observed source views", 422)
        canonical = _unit(document.canonical_embedding)
        groups: dict[tuple[int, int], list[Any]] = {(yi, pi): [] for yi in range(3) for pi in range(3)}
        fine: dict[int, list[Any]] = {index: [] for index in range(96)}
        for view in document.views:
            pose = as_pose3d((view.yaw, view.pitch, view.roll))  # FaceView uses legacy pitch.
            cell = _pose_group(pose.yaw, pose.pitch)
            groups[cell].append(view)
            fine[_sphere_bin(pose.yaw, pose.pitch)].append(view)

        # Identity recipes already carry the result of their per-cell blend in Stage 0 metadata.
        stored = document.metadata.get("stage0") if document.metadata.get("identity_blend") else None
        if isinstance(stored, dict):
            try:
                by_index = {int(item["cell"]): np.asarray(item["vector"], np.float32) for item in stored["cells"]}
                if set(by_index) == set(range(9)):
                    for yi in range(3):
                        for pi in range(3):
                            # Stored Stage 0 uses pitch-major indexing; remap by its yaw/pitch indices.
                            groups[(yi, pi)] = []
                    cell_vectors = np.stack([by_index[pi * 3 + yi] for yi in range(3) for pi in range(3)]).reshape(3, 3, 512)
                else:
                    cell_vectors = None
            except (KeyError, TypeError, ValueError):
                cell_vectors = None
        else:
            cell_vectors = None

        if cell_vectors is None:
            populated = [key for key, members in groups.items() if members]
            if not populated:
                raise DomainError("FACESET_EMPTY", "No pose-indexable source views were found", 422)
            vectors, views, support, confidence = np.zeros((3, 3, 512), np.float32), np.full((3, 3), -1), np.zeros((3, 3), np.int32), np.zeros((3, 3), np.float32)
            for yi in range(3):
                for pi in range(3):
                    members = groups[(yi, pi)]
                    if not members:
                        continue
                    weights = np.asarray([max(float(v.quality.score), 1e-3) for v in members], np.float64)
                    vectors[yi, pi] = _unit(sum(w * _unit(v.embedding) for w, v in zip(weights, members)))
                    best = max(members, key=lambda v: (v.quality.score, -abs(v.roll)))
                    views[yi, pi], support[yi, pi] = best.index, len(members)
                    confidence[yi, pi] = 0.6 + 0.4 * float(np.clip(best.quality.score, 0, 1))
            _fill_empty_cells(vectors, views, support, confidence)
        else:
            views, support, confidence = np.zeros((3, 3), np.int32), np.zeros((3, 3), np.int32), np.zeros((3, 3), np.float32)
            for yi in range(3):
                for pi in range(3):
                    item = stored["cells"][pi * 3 + yi]
                    support[yi, pi] = int(item.get("support", 0))
                    confidence[yi, pi] = float(item.get("confidence", 0.5 if not support[yi, pi] else 1.0))
                    members = groups[(yi, pi)]
                    if members:
                        views[yi, pi] = max(members, key=lambda v: v.quality.score).index
                    else:
                        views[yi, pi] = int(item["view_index"])

        if cell_vectors is not None:
            # The archive's stored Stage 0 matrix is the trusted per-cell Identity Blender result.
            # Its view images are from the dominant source, so do not let them replace blended cells
            # in the fine sphere lookup until original recipe members are available as documents.
            sphere_vectors = np.zeros((8, 12, 512), np.float32)
            sphere_views = np.zeros((8, 12), np.int32)
            sphere_support = np.zeros((8, 12), np.int32)
            for pi in range(8):
                pitch = -60.0 + (pi + 0.5) * 15.0
                for yi in range(12):
                    yaw = -90.0 + (yi + 0.5) * 15.0
                    cell_yaw, cell_pitch = _pose_group(yaw, pitch)
                    sphere_vectors[pi, yi] = vectors[cell_yaw, cell_pitch]
                    sphere_views[pi, yi] = views[cell_yaw, cell_pitch]
        else:
            sphere_vectors, sphere_views, sphere_support = _build_sphere(fine, vectors, views)
        return cls(faceset_id=document.faceset_id, canonical=canonical, cell_vectors=vectors, cell_views=views,
                   cell_support=support, cell_confidence=confidence, sphere_vectors=sphere_vectors,
                   sphere_views=sphere_views, sphere_support=sphere_support, projection=projection,
                   reference_blend=reference_blend, device=device,
                   mode="identity_blend" if document.metadata.get("identity_blend") else "single")

    @classmethod
    def from_documents(cls, documents: Sequence[FaceSetDocument], recipe: IdentityBlendSpec, *,
                       projection: np.ndarray | None = None, reference_blend: float = 0.15,
                       device: str = "cpu") -> "FrozenSourcePoseMatrix":
        if len(documents) != len(recipe.source_face_ids) or not documents:
            raise ValueError("One source document per identity recipe entry is required")
        if tuple(document.faceset_id for document in documents) != tuple(recipe.source_face_ids):
            raise ValueError("Frozen source documents must follow the recipe's declared FaceSet order")
        bases = [cls.from_document(doc, reference_blend=0.0) for doc in documents]
        weights, dials = recipe.normalised_weights(), recipe.dials()
        dominant = int(np.argmax(weights))
        cells = np.stack([blend_embeddings([base._cell_vectors[i] for base in bases], weights,
            method=recipe.method, dials=dials) for i in range(9)]).reshape(3, 3, 512)
        sphere = np.stack([blend_embeddings([base._sphere_vectors[i] for base in bases], weights,
            method=recipe.method, dials=dials) for i in range(96)])
        canonical = blend_embeddings([doc.canonical_embedding for doc in documents], weights,
                                     method=recipe.method, dials=dials)
        cell_views = bases[dominant]._cell_views.reshape(3, 3)
        cell_support = np.stack([base._cell_views.reshape(3, 3) >= 0 for base in bases]).sum(0).astype(np.int32)
        cell_conf = np.where(cell_support > 0, 1.0, 0.5).astype(np.float32)
        sphere_views = bases[dominant]._sphere_views
        sphere_support = np.stack([base._sphere_support for base in bases]).sum(0).astype(np.int32)
        return cls(faceset_id=documents[dominant].faceset_id, canonical=canonical, cell_vectors=cells,
                   cell_views=cell_views, cell_support=cell_support, cell_confidence=cell_conf,
                   sphere_vectors=sphere, sphere_views=sphere_views, sphere_support=sphere_support,
                   projection=projection, reference_blend=reference_blend, device=device,
                   mode=f"blend:{recipe.method}")

    def lookup(self, pose: Any) -> FrozenPoseSelection:
        target: Pose3D = as_pose3d(pose)
        if self.uses_fine_bins:
            corners = _sphere_corners(target.yaw, target.pitch)
            vectors, selected, projected, views = (self._sphere_vectors, self._sphere_selected,
                                                    self._sphere_projected, self._sphere_views.reshape(96))
            source = "sphere96"
        else:
            corners = _matrix_corners(target.yaw, target.pitch)
            vectors, selected, projected, views = self._cell_vectors, self._cell_selected, self._cell_projected, self._cell_views.reshape(9)
            source = "matrix3x3"
        latent = _bilinear(vectors, corners)
        frozen = _bilinear(selected, corners)
        projected_vector = _bilinear(projected, corners)
        winner = max(corners, key=lambda item: item[2])
        confidence_values = self._sphere_confidence_values.reshape(96) if self.uses_fine_bins else self._cell_confidence.reshape(9)
        confidence = float(sum(confidence_values[index] * weight for index, _row, weight in corners))
        return FrozenPoseSelection(_readonly(latent), _readonly(frozen), _readonly(projected_vector),
                                   int(views[winner[0]]), confidence,
                                   tuple((int(i), int(j), float(w)) for i, j, w in corners), source)

    def roll_affine(self, roll_degrees: float, center: tuple[float, float]) -> np.ndarray:
        """Interpolate the precomputed counter-roll rotation and place it around the eye midpoint."""
        if not np.isfinite(roll_degrees) or len(center) != 2 or not np.isfinite(center).all():
            raise ValueError("Roll and rotation center must be finite")
        roll = float(np.clip(roll_degrees, -180.0, 180.0))
        coordinate = roll + 180.0
        lower = int(math.floor(coordinate))
        upper = min(360, lower + 1)
        t = coordinate - lower
        linear = (1.0 - t) * self.roll_rotations[lower] + t * self.roll_rotations[upper]
        point = np.asarray(center, np.float64)
        offset = point - linear @ point
        return np.column_stack((linear, offset)).astype(np.float64)

    def summary(self) -> dict[str, Any]:
        return {"faceset_id": self.faceset_id, "mode": self.mode, "fingerprint": self.fingerprint[:16],
                "cells": 9, "yaw_edges": list(YAW_EDGES), "pitch_edges": list(PITCH_EDGES),
                "sphere_bins": self.fine_bin_count, "uses_fine_bins": self.uses_fine_bins,
                "roll_rotations": 361, "projection_device": self.projection_device}


def _fill_empty_cells(vectors, views, support, confidence) -> None:
    populated = [(yi, pi) for yi in range(3) for pi in range(3) if support[yi, pi] > 0]
    anchors = np.asarray(YAW_ANCHORS), np.asarray(PITCH_ANCHORS)
    for yi in range(3):
        for pi in range(3):
            if support[yi, pi] > 0:
                continue
            sy, sp = anchors[0][yi], anchors[1][pi]
            donor = min(populated, key=lambda cell: ((anchors[0][cell[0]] - sy) / 50) ** 2
                                                    + ((anchors[1][cell[1]] - sp) / 25) ** 2)
            vectors[yi, pi] = vectors[donor]
            views[yi, pi] = views[donor]
            confidence[yi, pi] = confidence[donor] * 0.5


def _build_sphere(groups, cell_vectors, cell_views):
    known = {index: members for index, members in groups.items() if members}
    if not known:
        raise DomainError("FACESET_EMPTY", "No pose samples are available for the 96-bin lookup", 422)
    vectors, views, support = np.zeros((8, 12, 512), np.float32), np.full((8, 12), -1, np.int32), np.zeros((8, 12), np.int32)
    for index, members in known.items():
        pi, yi = divmod(index, 12)
        weights = np.asarray([max(float(v.quality.score), 1e-3) for v in members], np.float64)
        vectors[pi, yi] = _unit(sum(w * _unit(v.embedding) for w, v in zip(weights, members)))
        views[pi, yi] = max(members, key=lambda v: v.quality.score).index
        support[pi, yi] = len(members)
    known_coords = [(index // 12, index % 12) for index in known]
    for pi in range(8):
        for yi in range(12):
            if support[pi, yi] > 0:
                continue
            nearest = min(known_coords, key=lambda p: (p[0] - pi) ** 2 + (p[1] - yi) ** 2)
            vectors[pi, yi] = vectors[nearest]
            views[pi, yi] = views[nearest]
    return vectors, views, support


def _matrix_corners(yaw: float, pitch: float):
    x0, x1, tx = _bracket(float(np.clip(yaw, -75, 75)), YAW_ANCHORS)
    y0, y1, ty = _bracket(float(np.clip(pitch, -35, 35)), PITCH_ANCHORS)
    return ((x0 * 3 + y0, y0, (1 - tx) * (1 - ty)), (x1 * 3 + y0, y0, tx * (1 - ty)),
            (x0 * 3 + y1, y1, (1 - tx) * ty), (x1 * 3 + y1, y1, tx * ty))


def _sphere_corners(yaw: float, pitch: float):
    x = float(np.clip(yaw, -90, 90)) / 15.0 + 5.5
    y = float(np.clip(pitch, -60, 60)) / 15.0 + 3.5
    x0, x1 = int(np.clip(math.floor(x), 0, 11)), int(np.clip(math.ceil(x), 0, 11))
    y0, y1 = int(np.clip(math.floor(y), 0, 7)), int(np.clip(math.ceil(y), 0, 7))
    tx, ty = (0.0 if x0 == x1 else x - x0), (0.0 if y0 == y1 else y - y0)
    return ((y0 * 12 + x0, y0, (1 - tx) * (1 - ty)), (y0 * 12 + x1, y0, tx * (1 - ty)),
            (y1 * 12 + x0, y1, (1 - tx) * ty), (y1 * 12 + x1, y1, tx * ty))


def _bracket(value: float, knots: Sequence[float]) -> tuple[int, int, float]:
    if value <= knots[0]:
        return 0, 0, 0.0
    if value >= knots[-1]:
        last = len(knots) - 1
        return last, last, 0.0
    upper = next(index for index, knot in enumerate(knots) if knot >= value)
    lower = upper - 1
    return lower, upper, (value - knots[lower]) / (knots[upper] - knots[lower])


def _bilinear(vectors: np.ndarray, corners) -> np.ndarray:
    by_index: dict[int, float] = {}
    for index, _row, weight in corners:
        by_index[index] = by_index.get(index, 0.0) + weight
    total = sum(by_index.values())
    if total <= 1e-12:
        return vectors[min(by_index, default=0)].copy()
    # Spherical mean of the four corner vectors using the bilinear weights.
    result = _unit(vectors[next(iter(by_index))])
    accumulated = by_index[next(iter(by_index))] / total
    for index, weight in list(by_index.items())[1:]:
        t = weight / total / max(accumulated + weight / total, 1e-12)
        result = _slerp(result, _unit(vectors[index]), t)
        accumulated += weight / total
    return result.astype(np.float32)
