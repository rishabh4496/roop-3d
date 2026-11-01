"""Stage 0 pre-flight: a frozen 3x3 (yaw x pitch) matrix of 512-D identity vectors per FaceSet.

Before a render or preview session starts, every FaceSet's observed views are folded into nine
pose cells (left/center/right x down/level/up, see `core.pose`). Each cell's identity vector is
computed once, frozen (read-only array + fingerprint), and afterwards a target face's pose maps to
its cell in O(1) with no per-frame distance search. Empty cells are filled from the nearest populated
cell and flagged; no identity vector is ever synthesised.

Identity guard: every frozen vector must stay within `GUARD_MAX_COSINE_DISTANCE` (cosine distance,
1 - cos) of its anchor. For a single FaceSet the anchor is its canonical identity, so an off-identity
view is excluded. For an identity-mixer blend (`identity_blend`) the anchor is the dominant source's
vector in the same cell; tangent-plane dials are scaled back until the blend is inside the boundary,
and a blend that cannot satisfy it even without dials is rejected.

`ROOP_STAGE0=0` (also `false`/`off`) disables the pre-flight; callers then fall back to the existing
per-frame `PoseSelector`. FaceSets whose views carry no usable pose spread ("un-indexed custom angle
portfolios") are not indexed either and take the same fallback.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import logging
import os
import threading
from typing import Any, Mapping, Sequence

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.core.pose import ALL_CELLS, PoseCell, as_pose3d, pose_cell
from roop_3d.faceset.identity_blend import IdentityBlendSpec, blend_embeddings
from roop_3d.faceset.schemas import FaceSetDocument

log = logging.getLogger("roop3d.stage0")

ENV_FLAG = "ROOP_STAGE0"
GUARD_MAX_COSINE_DISTANCE = 0.80
_FILL_PENALTY = 0.5


def stage0_enabled() -> bool:
    return os.environ.get(ENV_FLAG, "1").strip().lower() not in {"0", "false", "off", "no"}


def _unit(vector: Sequence[float]) -> np.ndarray:
    value = np.asarray(vector, np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1e-9:
        raise DomainError("IDENTITY_EMBEDDING_INVALID", "Identity embedding must have nonzero norm", 422)
    return value / norm


def _freeze(vector: np.ndarray) -> np.ndarray:
    frozen = np.ascontiguousarray(vector, dtype=np.float32)
    frozen.setflags(write=False)
    return frozen


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.dot(_unit(a), _unit(b)))


@dataclass(frozen=True)
class CellEntry:
    cell: PoseCell
    vector: np.ndarray          # frozen float32 unit vector
    view_index: int             # FaceView.index used as the pose reference
    roll: float                 # roll of the reference view (legacy-frame degrees), the 3rd alignment axis
    support: int                # real views that fell in this cell (0 = filled)
    confidence: float
    filled_from: PoseCell | None = None
    rejected: int = 0           # views excluded from this cell by the identity guard

    @property
    def filled(self) -> bool:
        return self.filled_from is not None


class PoseMatrix:
    """Nine frozen cell entries, indexed by `PoseCell.index` for O(1) lookup."""

    def __init__(self, entries: Mapping[PoseCell, CellEntry], *, faceset_id: str, anchor: np.ndarray,
                 guard: Mapping[str, Any], mode: str = "single"):
        if set(entries) != set(ALL_CELLS):
            raise DomainError("POSE_MATRIX_INCOMPLETE", "A pose matrix needs all nine cells", 500)
        self._cells: tuple[CellEntry, ...] = tuple(entries[cell] for cell in ALL_CELLS)
        self.faceset_id, self.mode = faceset_id, mode
        self.anchor = _freeze(anchor)
        self.guard = dict(guard)
        self.fingerprint = self._digest()

    def _digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.anchor.tobytes())
        for entry in self._cells:
            h.update(entry.vector.tobytes())
            h.update(f"{entry.cell.index}:{entry.view_index}:{entry.support}".encode())
        return h.hexdigest()

    def verify(self) -> bool:
        """True while the frozen vectors still hash to the pre-flight fingerprint."""
        return self._digest() == self.fingerprint

    def lookup(self, pose: Any, frame: str = "legacy") -> CellEntry:
        """O(1): threshold the pose into its cell and return that cell's frozen entry."""
        return self._cells[pose_cell(pose, frame).index]  # type: ignore[arg-type]

    def entry(self, cell: PoseCell) -> CellEntry:
        return self._cells[cell.index]

    @property
    def cells(self) -> tuple[CellEntry, ...]:
        return self._cells

    @property
    def populated(self) -> int:
        return sum(1 for entry in self._cells if entry.support > 0)

    @property
    def filled(self) -> int:
        return sum(1 for entry in self._cells if entry.filled)

    def summary(self) -> dict[str, Any]:
        return {"faceset_id": self.faceset_id, "mode": self.mode, "fingerprint": self.fingerprint[:16],
                "cells": len(self._cells), "populated": self.populated, "filled": self.filled,
                "guard": self.guard,
                "matrix": {entry.cell.name: {"view_index": entry.view_index, "support": entry.support,
                                             "filled_from": entry.filled_from.name if entry.filled_from else None,
                                             "confidence": round(entry.confidence, 4)} for entry in self._cells}}

    def to_json(self) -> dict[str, Any]:
        return {"version": 1, "faceset_id": self.faceset_id, "mode": self.mode, "fingerprint": self.fingerprint,
                "anchor": [float(x) for x in self.anchor], "guard": self.guard,
                "cells": [{"cell": e.cell.index, "vector": [float(x) for x in e.vector], "view_index": e.view_index,
                           "roll": e.roll, "support": e.support, "confidence": e.confidence, "rejected": e.rejected,
                           "filled_from": e.filled_from.index if e.filled_from else None} for e in self._cells]}

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> "PoseMatrix":
        by_index = {cell.index: cell for cell in ALL_CELLS}
        try:
            entries = {by_index[item["cell"]]: CellEntry(
                by_index[item["cell"]], _freeze(_unit(item["vector"])), int(item["view_index"]), float(item["roll"]),
                int(item["support"]), float(item["confidence"]),
                by_index[item["filled_from"]] if item["filled_from"] is not None else None, int(item.get("rejected", 0)))
                for item in data["cells"]}
            matrix = cls(entries, faceset_id=str(data["faceset_id"]), anchor=_unit(data["anchor"]),
                         guard=data.get("guard", {}), mode=str(data.get("mode", "single")))
        except (KeyError, TypeError, ValueError):
            raise DomainError("POSE_MATRIX_INVALID", "Stored Stage 0 pose matrix is malformed", 422) from None
        if matrix.fingerprint != data.get("fingerprint"):
            raise DomainError("POSE_MATRIX_INVALID", "Stage 0 pose matrix fingerprint mismatch", 422)
        return matrix


# --------------------------------------------------------------------------- building
def view_pose(view) -> Any:
    """A FaceView's pose as `Pose3D` (FaceView stores the legacy frame)."""
    return as_pose3d((view.yaw, view.pitch, view.roll))


def cell_members(document: FaceSetDocument) -> dict[str, list[int]]:
    """All nine cell names mapped to the indices of the views that fall in them (the archive's pose-matrix index)."""
    members: dict[str, list[int]] = {cell.name: [] for cell in ALL_CELLS}
    for view in document.views:
        members[pose_cell(view_pose(view), "standard").name].append(view.index)
    return members


def is_indexable(document: FaceSetDocument) -> bool:
    """False for portfolios whose angles carry no usable spread (custom/unposed sets)."""
    if not document.views:
        return False
    if document.metadata.get("stage0_indexed") is False or document.metadata.get("custom_angles"):
        return False
    if len(document.views) > 1:
        poses = {(round(v.yaw, 3), round(v.pitch, 3)) for v in document.views}
        if poses == {(0.0, 0.0)}:  # every view claims dead-centre: pose was never measured
            return False
    return True


def _nearest_populated(cell: PoseCell, populated: Sequence[PoseCell]) -> PoseCell:
    cy, cp = cell.center

    def distance(other: PoseCell) -> tuple[float, int]:
        oy, op = other.center
        return (((cy - oy) / 30.0) ** 2 + ((cp - op) / 20.0) ** 2, other.index)

    return min(populated, key=distance)


def build_pose_matrix(document: FaceSetDocument) -> PoseMatrix:
    if not document.views:
        raise DomainError("FACESET_EMPTY", "Stage 0 needs at least one observed view", 422)
    anchor = _unit(document.canonical_embedding)
    groups: dict[PoseCell, list[Any]] = {cell: [] for cell in ALL_CELLS}
    rejected: dict[PoseCell, int] = {cell: 0 for cell in ALL_CELLS}
    worst = 0.0
    for view in document.views:
        cell = pose_cell(view_pose(view), "standard")
        distance = cosine_distance(np.asarray(view.embedding), anchor)
        if distance > GUARD_MAX_COSINE_DISTANCE:
            rejected[cell] += 1
            continue
        groups[cell].append(view)
        worst = max(worst, distance)
    populated = [cell for cell in ALL_CELLS if groups[cell]]
    if not populated:
        raise DomainError("IDENTITY_GUARD_VIOLATION",
                          f"Every view lies beyond the {GUARD_MAX_COSINE_DISTANCE:.2f} cosine-distance identity guard", 422)
    entries: dict[PoseCell, CellEntry] = {}
    for cell in populated:
        members = groups[cell]
        best = max(members, key=lambda v: (v.quality.score, -abs(v.roll)))
        weights = np.array([max(v.quality.score, 1e-3) for v in members], np.float64)
        mean = _unit(sum(w * _unit(v.embedding) for w, v in zip(weights, members)))
        vector = mean if cosine_distance(mean, anchor) <= GUARD_MAX_COSINE_DISTANCE else _unit(best.embedding)
        entries[cell] = CellEntry(cell, _freeze(vector), best.index, float(best.roll), len(members),
                                  float(0.6 + 0.4 * min(1.0, max(0.0, best.quality.score))), None, rejected[cell])
    for cell in ALL_CELLS:
        if cell in entries:
            continue
        donor = entries[_nearest_populated(cell, populated)]
        entries[cell] = CellEntry(cell, donor.vector, donor.view_index, donor.roll, 0,
                                  donor.confidence * _FILL_PENALTY, donor.cell, rejected[cell])
    guard = {"threshold": GUARD_MAX_COSINE_DISTANCE, "max_distance": round(worst, 6), "rejected_views": sum(rejected.values())}
    return PoseMatrix(entries, faceset_id=document.faceset_id, anchor=anchor, guard=guard)


def build_blended_matrix(documents: Sequence[FaceSetDocument], spec: IdentityBlendSpec) -> PoseMatrix:
    """Blend the per-cell vectors of several FaceSets (Slerp/Linear + tangent dials) under the identity guard."""
    if len(documents) != len(spec.source_face_ids):
        raise ValueError("One document per blend source is required")
    matrices = [build_pose_matrix(document) for document in documents]
    weights, dials = spec.normalised_weights(), spec.dials()
    dominant = int(np.argmax(weights))
    entries: dict[PoseCell, CellEntry] = {}
    worst, clamped = 0.0, []
    for cell in ALL_CELLS:
        vectors = [m.entry(cell).vector for m in matrices]
        reference = vectors[dominant]
        scale, accepted = 1.0, None
        for _ in range(14):
            candidate = blend_embeddings(vectors, weights, method=spec.method, dials=dials * scale)
            if cosine_distance(candidate, reference) <= GUARD_MAX_COSINE_DISTANCE:
                accepted = candidate
                break
            scale = 0.0 if scale < 1e-3 else scale * 0.5
        if accepted is None:
            raise DomainError("IDENTITY_GUARD_VIOLATION",
                              f"Identity blend leaves the {GUARD_MAX_COSINE_DISTANCE:.2f} cosine-distance guard in cell {cell.name}", 422)
        if scale < 1.0:
            clamped.append({"cell": cell.name, "dial_scale": round(scale, 4)})
        worst = max(worst, cosine_distance(accepted, reference))
        base = matrices[dominant].entry(cell)
        entries[cell] = CellEntry(cell, _freeze(accepted), base.view_index, base.roll, base.support, base.confidence,
                                  base.filled_from, base.rejected)
    anchor = blend_embeddings([m.anchor for m in matrices], weights, method=spec.method, dials=dials)
    guard = {"threshold": GUARD_MAX_COSINE_DISTANCE, "max_distance": round(worst, 6), "clamped_cells": clamped,
             "rejected_views": sum(m.guard["rejected_views"] for m in matrices)}
    return PoseMatrix(entries, faceset_id=documents[dominant].faceset_id, anchor=anchor, guard=guard,
                      mode=f"blend:{spec.method}")


# --------------------------------------------------------------------------- pre-flight + cache
_CACHE: "OrderedDict[tuple, PoseMatrix]" = OrderedDict()
_LOCK = threading.Lock()
_CACHE_MAX = 32


def _key(document: FaceSetDocument) -> tuple:
    return (document.faceset_id, document.source_sha256, len(document.views),
            hashlib.sha1(np.asarray(document.canonical_embedding, np.float32).tobytes()).hexdigest())


def invalidate(faceset_id: str | None = None) -> None:
    """Drop memoised matrices (after a FaceSet changes). `None` drops everything."""
    with _LOCK:
        for key in [k for k in _CACHE if faceset_id is None or k[0] == faceset_id]:
            del _CACHE[key]


def preflight(document: FaceSetDocument, *, label: str | None = None) -> PoseMatrix | None:
    """Run (or reuse) Stage 0 for `document`; None means "use the per-frame fallback"."""
    name = label or document.faceset_id
    if not stage0_enabled():
        log.info("[Stage0] disabled (%s=0) faceset=%s: per-frame pose selection", ENV_FLAG, name)
        return None
    stored = document.metadata.get("stage0")
    if isinstance(stored, Mapping):
        matrix = PoseMatrix.from_json(stored)
        source = "blend"
    elif not is_indexable(document):
        log.info("[Stage0] faceset=%s has no usable pose spread (custom angles): per-frame pose selection", name)
        return None
    else:
        key = _key(document)
        with _LOCK:
            matrix = _CACHE.get(key)
            if matrix is not None:
                _CACHE.move_to_end(key)
        source = "cache"
        if matrix is None:
            matrix = build_pose_matrix(document)
            with _LOCK:
                _CACHE[key] = matrix
                while len(_CACHE) > _CACHE_MAX:
                    _CACHE.popitem(last=False)
            source = "built"
    log.info("[Stage0] pre-flight faceset=%s mode=%s cells=%d populated=%d filled=%d guard_max=%.4f/%.2f fingerprint=%s (%s)",
             name, matrix.mode, len(matrix.cells), matrix.populated, matrix.filled,
             matrix.guard.get("max_distance", 0.0), GUARD_MAX_COSINE_DISTANCE, matrix.fingerprint[:12], source)
    return matrix
