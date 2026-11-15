"""Multi-angle identity reference bank of one FaceSet: quality-scored ArcFace embeddings + head pose per view.

One bank feeds both consumers of a FaceSet:

* **target matching** - ``similarity`` scores a target face's 512-D embedding against the normalised identity *centroid*
  and against the references nearest to the target's own pose, because an ArcFace embedding of a 3/4 or profile view sits
  far from the frontal mean even for the right person (a single centroid alone would reject the very frames the 3D head
  fit needs);
* **multi-view UV texture reconstruction** - ``uv_views`` orders the usable views for pose coverage (best view first, then
  the view that adds the largest pose gap) and ``apply`` writes the bank's quality score and centroid back into the
  ``FaceSetDocument`` the asset builders read, so the same measured quality decides which view paints which texels.

Quality is measured, not assumed: detector confidence, face sharpness, face size, landmark trust (dense 68-point fit,
PnP reprojection error) and exposure each contribute; a view whose landmarks are fabricated stand-ins is excluded, as is a
view whose embedding is not the same person as the rest (cosine to the centroid below the FaceSet identity-link floor).
Nothing is invented: a view that cannot be scored or embedded is reported in ``rejected`` with the reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, TYPE_CHECKING

import numpy as np

from roop_3d.faceset.schemas import FaceSetDocument, FaceView, QualityMetrics

if TYPE_CHECKING:                                                   # pragma: no cover
    from roop_3d.faceset_loader import FaceSample

#: ArcFace cosine below which a view is not the same person as the rest of the set (shared with the FaceSet quality gate).
IDENTITY_FLOOR = 0.30
MIN_QUALITY = 0.20
FACE_SIDE_REF_PX = 160.0
SHARPNESS_FLOOR = 0.3                     # a soft but usable photograph keeps 30 % of the sharpness term
REPROJECTION_SCALE_PX = 8.0


@dataclass(frozen=True)
class BankEntry:
    view_index: int
    embedding: np.ndarray                 # (512,) float32, unit length
    yaw: float                            # degrees, standard frame (pitch + = up)
    pitch: float
    roll: float
    quality: float                        # 0..1, product of the components below
    components: dict[str, float]
    consistency: float                    # cosine of the embedding to the final identity centroid
    pose_source: str                      # "1k3d68+2d106det" | "1k3d68" | "archive_5pt"

    def summary(self) -> dict[str, Any]:
        return {"view_index": self.view_index, "yaw": round(self.yaw, 2), "pitch": round(self.pitch, 2),
                "roll": round(self.roll, 2), "quality": round(self.quality, 4),
                "components": {k: round(v, 4) for k, v in self.components.items()},
                "centroid_cosine": round(self.consistency, 4), "pose_source": self.pose_source}


@dataclass
class ReferenceBank:
    faceset_id: str
    centroid: np.ndarray                  # (512,) float32, unit length; quality-weighted mean of the usable embeddings
    entries: list[BankEntry]              # usable references, best quality first
    rejected: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ pose
    @staticmethod
    def _gap(yaw: float, pitch: float, entry: BankEntry) -> float:
        return math.hypot(yaw - entry.yaw, pitch - entry.pitch)

    def nearest(self, yaw: float, pitch: float, count: int = 2) -> list[BankEntry]:
        """The ``count`` references whose head pose is closest to (yaw, pitch) (standard frame, degrees)."""
        return sorted(self.entries, key=lambda e: self._gap(yaw, pitch, e))[:max(1, count)]

    # -------------------------------------------------------------- matching
    def similarity(self, embedding: np.ndarray, yaw: float | None = None, pitch: float | None = None) -> dict[str, Any]:
        """Identity score of a target face: centroid cosine, best-reference cosine and (given its pose) the pose-matched cosine.

        ``score`` = the larger of the centroid cosine and the mean cosine to the two references nearest the target's pose,
        so a turned head is judged against turned references. All cosines are on unit vectors."""
        vector = np.asarray(embedding, np.float32).reshape(-1)
        if vector.shape != (512,) or not np.isfinite(vector).all():
            raise ValueError("embedding must be a finite 512-vector")
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            raise ValueError("embedding has zero length")
        vector = vector / norm
        centroid = float(vector @ self.centroid)
        cosines = [float(vector @ e.embedding) for e in self.entries]
        best = int(np.argmax(cosines))
        pose_matched = None
        if yaw is not None and pitch is not None:
            near = self.nearest(yaw, pitch, 2)
            pose_matched = float(np.mean([float(vector @ e.embedding) for e in near]))
        return {"centroid_cosine": centroid, "best_cosine": cosines[best], "best_view_index": self.entries[best].view_index,
                "pose_matched_cosine": pose_matched, "score": max(centroid, pose_matched if pose_matched is not None else centroid)}

    # ------------------------------------------------------------------- UV
    def uv_views(self, limit: int | None = None) -> list[int]:
        """View indices in the order a multi-view texture builder should consume them.

        Best-quality view first, then repeatedly the usable view farthest in pose from those already chosen (ties by
        quality): every added view widens the covered pose range before a near-duplicate pose is spent."""
        remaining = list(self.entries)
        order: list[BankEntry] = []
        while remaining and (limit is None or len(order) < limit):
            if not order:
                pick = max(remaining, key=lambda e: e.quality)
            else:
                pick = max(remaining, key=lambda e: (min(self._gap(e.yaw, e.pitch, o) for o in order), e.quality))
            order.append(pick)
            remaining.remove(pick)
        return [e.view_index for e in order]

    def coverage(self) -> dict[str, Any]:
        if not self.entries:
            return {"views": 0, "yaw_span_degrees": 0.0, "pitch_span_degrees": 0.0}
        yaws, pitches = [e.yaw for e in self.entries], [e.pitch for e in self.entries]
        return {"views": len(self.entries), "yaw_range_degrees": [round(min(yaws), 2), round(max(yaws), 2)],
                "pitch_range_degrees": [round(min(pitches), 2), round(max(pitches), 2)],
                "yaw_span_degrees": round(max(yaws) - min(yaws), 2), "pitch_span_degrees": round(max(pitches) - min(pitches), 2)}

    def summary(self) -> dict[str, Any]:
        return {"faceset_id": self.faceset_id, "centroid_norm": round(float(np.linalg.norm(self.centroid)), 6),
                "usable": len(self.entries), "rejected": self.rejected, "coverage": self.coverage(),
                "uv_view_order": self.uv_views(), "references": [e.summary() for e in self.entries]}

    # ---------------------------------------------------------------- apply
    def apply(self, document: FaceSetDocument) -> FaceSetDocument:
        """The document with the bank's centroid, medoid and per-view quality scores (rejected views score 0)."""
        from roop_3d.faceset.faceset_reader import build_pose_bins
        by_index = {e.view_index: e for e in self.entries}
        views: list[FaceView] = []
        for view in document.views:
            entry = by_index.get(view.index)
            score = entry.quality if entry is not None else 0.0
            views.append(view.model_copy(update={"quality": view.quality.model_copy(update={"score": round(score, 4)})}))
        medoid = max(self.entries, key=lambda e: float(sum(e.embedding @ o.embedding for o in self.entries))).embedding
        return document.model_copy(update={
            "canonical_embedding": tuple(float(x) for x in self.centroid),
            "medoid_embedding": tuple(float(x) for x in medoid),
            "views": tuple(views), "pose_bins": build_pose_bins(views)})


def _unit(vector: np.ndarray) -> np.ndarray | None:
    value = np.asarray(vector, np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if value.shape == (512,) and np.isfinite(value).all() and norm > 1e-9 else None


def _exposure(crop_rgb: np.ndarray) -> float:
    """1 for a well exposed face, falling as pixels clip to black or white.

    Only the central window of the aligned crop is measured: a turned head's similarity alignment leaves black border
    fill at the sides, which is padding, not under-exposure."""
    side = crop_rgb.shape[0]
    centre = crop_rgb[side // 5: side * 4 // 5, side // 5: side * 4 // 5]
    gray = centre.astype(np.float32).mean(axis=2)
    clipped = float(((gray <= 5.0) | (gray >= 250.0)).mean())
    return float(max(0.0, 1.0 - 3.0 * clipped))


def score_view(view: FaceView, sample: "FaceSample") -> tuple[float, dict[str, float]]:
    """Measured quality 0..1 of one view and the components it is made of."""
    quality: QualityMetrics = view.quality
    face_side = math.sqrt(max(quality.face_pixels, 1.0))
    landmarks = (1.0 if sample.landmarks_68 is not None else 0.8) * math.exp(-max(sample.reprojection_px, 0.0) / REPROJECTION_SCALE_PX)
    components = {
        "detector": float(min(1.0, max(0.0, quality.detector_confidence))),
        "sharpness": SHARPNESS_FLOOR + (1.0 - SHARPNESS_FLOOR) * float(min(1.0, max(0.0, quality.sharpness))),
        "size": float(min(1.0, face_side / FACE_SIDE_REF_PX)),
        "landmarks": float(landmarks),
        "exposure": _exposure(sample.crop_rgb_512),
    }
    weights = {"detector": 0.25, "sharpness": 0.30, "size": 0.15, "landmarks": 0.15, "exposure": 0.15}
    value = 1.0
    for name, weight in weights.items():
        value *= max(components[name], 1e-6) ** weight
    return float(value), components


def build_reference_bank(document: FaceSetDocument, samples: "list[FaceSample]") -> ReferenceBank:
    """Score every view, reject the unusable and the not-this-person ones, and form the identity centroid."""
    by_index = {v.index: v for v in document.views}
    candidates: list[tuple[FaceView, Any, np.ndarray, float, dict[str, float]]] = []
    rejected: list[dict[str, Any]] = []
    for sample in samples:
        view = by_index[sample.view_index]
        embedding = _unit(sample.normed_embedding)
        if embedding is None:
            rejected.append({"view_index": view.index, "reason": "embedding_invalid"})
            continue
        if not sample.trusted:
            rejected.append({"view_index": view.index, "reason": "untrusted_landmarks"})
            continue
        score, components = score_view(view, sample)
        if score < MIN_QUALITY:
            rejected.append({"view_index": view.index, "reason": "low_quality", "quality": round(score, 4)})
            continue
        candidates.append((view, sample, embedding, score, components))
    if not candidates:
        from roop_3d.core.errors import DomainError
        raise DomainError("FACESET_NO_USABLE_VIEW",
                          f"FaceSet {document.faceset_id} has no view with trusted landmarks and usable quality: "
                          + "; ".join(f"view {r['view_index']}: {r['reason']}" for r in rejected), 422)

    def centroid_of(items) -> np.ndarray:
        total = sum(item[3] * item[2] for item in items)
        return total / float(np.linalg.norm(total))

    kept = list(candidates)
    for _ in range(3):                                    # drop not-this-person views, then re-centre without them
        centroid = centroid_of(kept)
        outliers = [item for item in kept if float(item[2] @ centroid) < IDENTITY_FLOOR]
        if not outliers or len(kept) - len(outliers) < 1 or len(kept) < 3:
            break
        # Drop the single worst view per pass: the first outlier skews the centroid the others are judged against.
        worst = min(outliers, key=lambda item: float(item[2] @ centroid))
        kept.remove(worst)
        rejected.append({"view_index": worst[0].index, "reason": "identity_outlier",
                         "centroid_cosine": round(float(worst[2] @ centroid), 4)})
    centroid = centroid_of(kept).astype(np.float32)
    entries = [BankEntry(view_index=view.index, embedding=embedding.astype(np.float32), yaw=float(sample.yaw),
                         pitch=float(sample.pitch), roll=float(sample.roll), quality=score, components=components,
                         consistency=float(embedding @ centroid), pose_source=sample.landmark_source)
               for view, sample, embedding, score, components in kept]
    entries.sort(key=lambda e: -e.quality)
    return ReferenceBank(document.faceset_id, centroid, entries, rejected)
