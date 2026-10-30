"""Dynamic pose-bin coverage classifier and nearest-view selector for FaceSets.

Provides reference view selection conditioning for LivePortrait and identity swapper,
returning the closest real observed source crop, landmarks, and confidence score.
"""
from __future__ import annotations

import math
from typing import Any
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, ReferenceMatch


def classify_pose_bin(yaw: float, pitch: float) -> tuple[str, str, str]:
    """Classify yaw and pitch into discrete bins."""
    if yaw < -15.0:
        yaw_bin = "left"
    elif yaw > 15.0:
        yaw_bin = "right"
    else:
        yaw_bin = "center"

    if pitch < -12.0:
        pitch_bin = "up"
    elif pitch > 12.0:
        pitch_bin = "down"
    else:
        pitch_bin = "center"

    bin_name = f"{yaw_bin}_{pitch_bin}"
    return bin_name, yaw_bin, pitch_bin


class PoseSelector:
    """Selects the best real observed source view matching target pose and expression."""

    def __init__(self, document: FaceSetDocument):
        if not document.views:
            raise DomainError("FACESET_EMPTY", "FaceSet contains no face views for pose selection", 422)
        self.document = document
        self._views: list[FaceView] = list(document.views)

    @property
    def view_count(self) -> int:
        return len(self._views)

    def coverage(self) -> dict[str, int]:
        """Return the count of views in each active pose bin."""
        counts: dict[str, int] = {}
        for view in self._views:
            counts[view.pose_bin] = counts.get(view.pose_bin, 0) + 1
        return counts

    def select_best_reference(
        self,
        target_yaw: float,
        target_pitch: float,
        target_roll: float = 0.0,
        target_expression: float | dict[str, float] | None = None,
    ) -> ReferenceMatch:
        """Query the best matching real observed source view for conditioning.

        Args:
            target_yaw: Target head yaw in degrees [-90, +90].
            target_pitch: Target head pitch in degrees [-60, +60].
            target_roll: Target head roll in degrees [-180, +180].
            target_expression: Openness float (0.0 to 1.0) or expression dictionary.

        Returns:
            ReferenceMatch containing crop bytes, landmarks, match confidence, and metadata.
        """
        # Clamp target angles to physical bounds
        clamped_yaw = max(-90.0, min(90.0, float(target_yaw)))
        clamped_pitch = max(-60.0, min(60.0, float(target_pitch)))
        clamped_roll = max(-180.0, min(180.0, float(target_roll)))

        target_openness: float = 0.0
        has_expression = False
        if target_expression is not None:
            has_expression = True
            if isinstance(target_expression, dict):
                target_openness = float(target_expression.get("openness", 0.0))
            else:
                target_openness = float(target_expression)
            target_openness = max(0.0, min(1.0, target_openness))

        best_view: FaceView | None = None
        lowest_cost: float = float("inf")
        best_confidence: float = 0.0

        for view in self._views:
            # 1. Angular deltas
            d_yaw = abs(view.yaw - clamped_yaw)
            d_pitch = abs(view.pitch - clamped_pitch)

            # Circular roll difference
            d_roll = abs((view.roll - clamped_roll + 180.0) % 360.0 - 180.0)

            # 2. Expression delta
            d_exp = 0.0
            if has_expression:
                view_openness = float(view.expression.get("openness", 0.0))
                d_exp = abs(view_openness - target_openness)

            # 3. Weighted normalized distance
            norm_yaw = d_yaw / 90.0
            norm_pitch = d_pitch / 60.0
            norm_roll = d_roll / 45.0
            w_yaw = 1.0
            w_pitch = 1.0
            w_roll = 0.3
            w_exp = 0.5 if has_expression else 0.0

            dist_sq = (
                w_yaw * (norm_yaw ** 2)
                + w_pitch * (norm_pitch ** 2)
                + w_roll * (norm_roll ** 2)
                + w_exp * (d_exp ** 2)
            )
            pose_dist = math.sqrt(dist_sq)

            # 4. Quality score bonus (favors sharper, higher-confidence crops when poses are close)
            quality_score = max(0.0, min(1.0, view.quality.score))
            cost = pose_dist - 0.15 * quality_score

            # 5. Confidence score in [0.0, 1.0]
            confidence = max(0.0, min(1.0, (1.0 / (1.0 + pose_dist)) * (0.6 + 0.4 * quality_score)))

            if cost < lowest_cost:
                lowest_cost = cost
                best_view = view
                best_confidence = confidence

        if best_view is None:
            raise DomainError("FACESET_EMPTY", "Failed to select reference from empty FaceSet", 422)

        return ReferenceMatch(
            view_index=best_view.index,
            crop_bytes=best_view.image_bytes,
            landmarks=best_view.landmarks,
            confidence=round(best_confidence, 4),
            yaw=best_view.yaw,
            pitch=best_view.pitch,
            roll=best_view.roll,
            expression=dict(best_view.expression),
            quality_score=round(best_view.quality.score, 4),
            pose_bin=best_view.pose_bin,
            is_synthetic=False,
        )
