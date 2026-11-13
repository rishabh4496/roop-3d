"""The ``temporal`` block of the ``pipeline.status`` telemetry frame (blueprint §4), as a plain dict.

``jitter_px``  mean per-point distance between the filtered and the raw landmarks of the frame (how far the temporal
               filters moved the measurement; 0 when no landmarks are given).
``one_euro_cutoff_hz``  mean effective cutoff (``min_cutoff + beta |dx/dt|``) of the supplied One-Euro filters.
``scene_cut`` / ``flash`` / ``cuts_total``  from the frame's ``scene_cut.CutEvent``.
``mask_flow_confidence``  mean forward-backward flow confidence of the matte stabiliser (0 after a reset).
``stability_score``  ``exp(-jitter_px / 2) * (0.5 + 0.5 * mask_flow_confidence)``: 1 = steady, low = visibly restless.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from roop_3d.temporal import OneEuroFilter
from roop_3d.tracking.scene_cut import CutEvent


def temporal_telemetry(*, raw_landmarks=None, filtered_landmarks=None, cut_event: CutEvent | None = None,
                       filters: Iterable[OneEuroFilter] = (), mask_flow_confidence: float = 0.0) -> dict:
    jitter = 0.0
    if raw_landmarks is not None and filtered_landmarks is not None:
        raw, filtered = np.asarray(raw_landmarks, np.float64), np.asarray(filtered_landmarks, np.float64)
        if raw.shape != filtered.shape or raw.ndim < 2 or not (np.isfinite(raw).all() and np.isfinite(filtered).all()):
            raise ValueError("raw and filtered landmarks must be finite arrays of equal shape (N, 2+)")
        jitter = float(np.linalg.norm(raw - filtered, axis=-1).mean())
    cutoffs = [float(item.cutoff) for item in filters]
    confidence = float(np.clip(mask_flow_confidence, 0.0, 1.0))
    event = cut_event.to_telemetry() if cut_event is not None else {"scene_cut": False, "flash": False, "cuts_total": 0}
    return {"jitter_px": round(jitter, 4),
            "one_euro_cutoff_hz": round(float(np.mean(cutoffs)), 4) if cutoffs else 0.0,
            **event,
            "mask_flow_confidence": round(confidence, 4),
            "stability_score": round(float(np.exp(-jitter / 2.0) * (0.5 + 0.5 * confidence)), 4)}
