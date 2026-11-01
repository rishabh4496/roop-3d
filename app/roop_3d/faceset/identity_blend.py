"""Identity-embedding blend recipes: Linear or Slerp (spherical mean) plus tangent-plane dials.

Operates on L2-normalised ArcFace-style vectors. The blended vector replaces the
canonical embedding of a primary FaceSet; its observed views (pose references) are
left untouched, so pose selection still uses real captured views.
"""
from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from pydantic import Field, model_validator

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.faceset.schemas import FaceSetDocument

_ID = r"^[A-Za-z0-9_-]{1,64}$"


class IdentityBlendSpec(Contract):
    method: Literal["slerp", "linear"] = "slerp"
    source_face_ids: tuple[str, ...] = Field(min_length=2, max_length=8)
    weights: tuple[float, ...] = ()
    # One dial per source in [-1, 1]; +1 slides the blend fully toward that source along
    # the sphere's tangent plane at the blended point, -1 slides it directly away.
    tangent_dials: tuple[float, ...] = ()

    @model_validator(mode="after")
    def consistent(self) -> "IdentityBlendSpec":
        import re
        count = len(self.source_face_ids)
        if any(not re.fullmatch(_ID, item) for item in self.source_face_ids) or len(set(self.source_face_ids)) != count:
            raise ValueError("Blend sources must be distinct FaceSet identifiers")
        if self.weights and (len(self.weights) != count or any(w < 0 for w in self.weights) or not any(w > 0 for w in self.weights)):
            raise ValueError("Blend weights must be nonnegative, not all zero, one per source")
        if self.tangent_dials and (len(self.tangent_dials) != count or any(not -1 <= d <= 1 for d in self.tangent_dials)):
            raise ValueError("Tangent dials must lie within [-1, 1], one per source")
        return self

    def normalised_weights(self) -> np.ndarray:
        raw = np.asarray(self.weights or (1.0,) * len(self.source_face_ids), np.float64)
        return raw / raw.sum()

    def dials(self) -> np.ndarray:
        return np.asarray(self.tangent_dials or (0.0,) * len(self.source_face_ids), np.float64)


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-9:
        raise DomainError("IDENTITY_EMBEDDING_INVALID", "Identity embedding must have nonzero norm", 422)
    return vector / norm


def _log(base: np.ndarray, point: np.ndarray) -> np.ndarray:
    cosine = float(np.clip(base @ point, -1.0, 1.0))
    angle = np.arccos(cosine)
    if angle < 1e-9:
        return np.zeros_like(base)
    direction = point - cosine * base
    length = np.linalg.norm(direction)
    if length < 1e-12:  # antipodal: no unique geodesic, contribute nothing
        return np.zeros_like(base)
    return direction * (angle / length)


def _exp(base: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(tangent)
    if length < 1e-12:
        return base
    return _unit(np.cos(length) * base + np.sin(length) * tangent / length)


def blend_embeddings(embeddings: Sequence[Sequence[float]], weights: Sequence[float], *,
                     method: str = "slerp", dials: Sequence[float] | None = None) -> np.ndarray:
    """Return a unit vector: weighted Linear/Slerp mean, then optional tangent-plane dial offsets."""
    points = np.stack([_unit(np.asarray(item)) for item in embeddings])
    w = np.asarray(weights, np.float64)
    if points.shape[0] != w.shape[0] or (w < 0).any() or w.sum() <= 0:
        raise ValueError("One nonnegative weight per embedding is required")
    w = w / w.sum()
    mean = _unit(w @ points)
    if method == "slerp":
        # Weighted Frechet (Karcher) mean on the sphere; for two points this is exactly slerp(t=w[1]).
        for _ in range(32):
            step = sum(weight * _log(mean, point) for weight, point in zip(w, points))
            mean = _exp(mean, step)
            if np.linalg.norm(step) < 1e-10:
                break
    elif method != "linear":
        raise ValueError("Blend method must be slerp or linear")
    if dials is not None and np.any(np.asarray(dials) != 0):
        offset = sum(float(d) * _log(mean, point) for d, point in zip(dials, points))
        mean = _exp(mean, offset)
    return mean.astype(np.float32)


def blend_documents(documents: Sequence[FaceSetDocument], spec: IdentityBlendSpec) -> FaceSetDocument:
    """A document whose canonical/medoid identity is the recipe's blend; views come from the dominant source."""
    if len(documents) != len(spec.source_face_ids):
        raise ValueError("One document per blend source is required")
    weights, dials = spec.normalised_weights(), spec.dials()
    canonical = blend_embeddings([d.canonical_embedding for d in documents], weights, method=spec.method, dials=dials)
    medoid = blend_embeddings([d.medoid_embedding for d in documents], weights, method=spec.method, dials=dials)
    primary = documents[int(np.argmax(weights))]
    return primary.model_copy(update={
        "canonical_embedding": tuple(float(x) for x in canonical),
        "medoid_embedding": tuple(float(x) for x in medoid),
        "metadata": {**primary.metadata, "identity_blend": spec.model_dump(mode="json")}})
