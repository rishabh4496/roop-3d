"""Expression measurement for 3D head animation: landmark-driven expression parameters, neutral calibration, filters.

The former 2.5D rig / painter's-order renderer (``IdentityRig`` / ``ReenactmentEngine``) was removed with the canonical
frontal-strip mesh. What stays is pure measurement (no rendering, no warping): ``measure_expression`` turns 5/106-point
landmarks into inter-ocular-normalised expression deltas, ``NeutralCalibrator`` establishes the track's own neutral and
``params_from_measure`` yields ``ExpressionParams`` that a full-head model (FullHeadEngine) consumes.
"""
from __future__ import annotations

from roop_3d.reconstruction.landmark_layout import group

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.temporal import OneEuroFilter

# 68-landmark vertex groups of the template (indices 0..67 are preserved by every subdivision level)
EYE_A, EYE_B = range(36, 42), range(42, 48)       # image-left / image-right eye of a frontal head
BROW_A, BROW_B = range(17, 22), range(22, 27)
NOSE_TIP, MOUTH_A, MOUTH_B = 30, 48, 54
INNER_LIP = range(60, 68)


# =========================================================================== settings / data
class ReenactmentSettings(Contract):
    # gate
    enter_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    exit_confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    enter_frames: int = Field(default=3, ge=1, le=30)
    exit_frames: int = Field(default=2, ge=1, le=30)
    ramp_frames: int = Field(default=4, ge=1, le=30, description="Frames over which the engine cross-fades in / out")
    max_abs_yaw_deg: float = Field(default=80.0, ge=10.0, le=90.0)
    expression_max_yaw_deg: float = Field(default=60.0, ge=10.0, le=85.0)
    max_pose_rate_deg_s: float = Field(default=1400.0, gt=0.0, description="Physical limit; faster is a measurement error")
    min_pose_support: float = Field(default=0.35, ge=0.0, le=1.0, description="Observed-texture share of the rendered footprint")
    coverage_margin_deg: float = Field(default=10.0, ge=0.0, le=45.0, description="Extrapolation allowed beyond the source views' pose range at full confidence")
    coverage_ramp_deg: float = Field(default=10.0, gt=0.0, le=45.0, description="...then confidence falls to 0 over this many further degrees")
    max_pose_rms_iod: float = Field(default=0.12, gt=0.0)
    max_alignment_iod: float = Field(default=0.25, gt=0.0)
    min_sharpness_ratio: float = Field(default=0.25, ge=0.0, le=1.0)
    calibration_frames: int = Field(default=8, ge=2, le=120)
    calibration_window: int = Field(default=120, ge=8, le=1000)
    # temporal filters (One-Euro): pose is smooth, expression keeps blink-speed events
    pose_min_cutoff: float = Field(default=1.5, gt=0.0)
    pose_beta: float = Field(default=0.03, ge=0.0)
    pose_d_cutoff: float = Field(default=8.0, gt=0.0)
    expression_min_cutoff: float = Field(default=6.0, gt=0.0)
    expression_beta: float = Field(default=1.0, ge=0.0)
    expression_d_cutoff: float = Field(default=15.0, gt=0.0)
    anchor_min_cutoff: float = Field(default=3.0, gt=0.0)
    anchor_beta: float = Field(default=0.05, ge=0.0)
    anchor_d_cutoff: float = Field(default=8.0, gt=0.0)
    # expression transfer
    expression_gain: float = Field(default=1.0, ge=0.0, le=1.5)
    mouth_hole_open_iod: float = Field(default=0.10, ge=0.0, description="Mouth interior is released to the target above this opening")
    eye_hole_blink: float = Field(default=0.45, ge=0.0, le=1.0)
    max_snap_iod: float = Field(default=0.15, ge=0.0, le=0.5)
    snap_enabled: bool = True


@dataclass(frozen=True)
class TargetObservation:
    """One tracked target face in canvas (crop) coordinates."""
    landmarks5: np.ndarray | None                 # (5, 2) image-left eye, image-right eye, nose, image-left mouth corner, image-right mouth corner
    landmarks106: np.ndarray | None               # (106, 2) 2d106 layout (block structure of expression_tps.BLOCKS)
    timestamp: float
    scene_cut: bool = False
    sharpness: float | None = None                # focus measure of the face crop; None = not measured


@dataclass
class ExpressionParams:
    """Target motion as inter-ocular-normalised deltas from the track's own neutral. All zero = neutral."""
    mouth_open: float = 0.0     # >= 0, lower-lip / jaw drop
    mouth_width: float = 0.0    # + wider, - pursed
    smile: float = 0.0          # + mouth corners up, - down
    blink_a: float = 0.0        # 0 open .. 1 closed, image-left eye
    blink_b: float = 0.0
    brow_a: float = 0.0         # + raised
    brow_b: float = 0.0

    NAMES = ("mouth_open", "mouth_width", "smile", "blink_a", "blink_b", "brow_a", "brow_b")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.NAMES], np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "ExpressionParams":
        return cls(**{n: float(x) for n, x in zip(cls.NAMES, v)})

    def clamped(self) -> "ExpressionParams":
        return ExpressionParams(
            mouth_open=float(np.clip(self.mouth_open, 0.0, 0.55)), mouth_width=float(np.clip(self.mouth_width, -0.25, 0.30)),
            smile=float(np.clip(self.smile, -0.12, 0.22)), blink_a=float(np.clip(self.blink_a, 0.0, 1.0)),
            blink_b=float(np.clip(self.blink_b, 0.0, 1.0)), brow_a=float(np.clip(self.brow_a, -0.12, 0.16)),
            brow_b=float(np.clip(self.brow_b, -0.12, 0.16)))


class _EuroSmoother:
    """Velocity-adaptive One-Euro smoothing of one array-valued channel (the engine tracks a single head, so there is one track).

    The Kalman predictor that de1e3ad put behind these channels extrapolated every step a second time and overshot a 70 degree yaw
    step to 88 degrees, so the engine could not engage on a quick turn; smoothing is a pure low-pass again, as it was when the engine's
    lag, jump and recovery bounds were set.
    """

    def __init__(self, min_cutoff: float, beta: float, derivative_cutoff: float) -> None:
        self._filter = OneEuroFilter(min_cutoff, beta, derivative_cutoff)

    def filter(self, track_id: int, channel: str, value, timestamp: float) -> np.ndarray:
        return self._filter.update(value, timestamp)


# =========================================================================== measurement
@dataclass
class ExpressionMeasure:
    mouth_open: float
    mouth_width: float
    smile: float
    eye_h_a: float          # vertical eye opening / foreshortening-corrected inter-ocular distance
    eye_h_b: float
    brow_a: float
    brow_b: float
    iod_px: float

    NAMES = ("mouth_open", "mouth_width", "smile", "eye_h_a", "eye_h_b", "brow_a", "brow_b")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, n) for n in self.NAMES], np.float64)


def measure_expression(lm5: np.ndarray, lm106: np.ndarray, yaw_deg: float) -> ExpressionMeasure:
    """Block-level expression ratios in the face's own (rolled) frame, normalised by the foreshortened inter-ocular distance.

    Uses only the block STRUCTURE of the 106-point layout (which points belong to mouth / eyes / brows), never a
    point-by-point correspondence, because a detector's ordering is not trusted beyond what ``layout_is_plausible`` checks.
    """
    lm5 = np.asarray(lm5, np.float64)
    lm106 = np.asarray(lm106, np.float64)
    across = lm5[1] - lm5[0]
    iod_img = float(np.linalg.norm(across))
    if iod_img < 2.0:
        raise DomainError("EXPRESSION_UNMEASURABLE", "Inter-ocular distance too small to measure expression", 422)
    across = across / iod_img
    down = np.array([-across[1], across[0]])
    iod_eff = iod_img / max(math.cos(math.radians(yaw_deg)), 0.5)
    origin = lm5[:2].mean(axis=0)
    coords = lambda pts: np.column_stack([(pts - origin) @ across, (pts - origin) @ down])
    mouth = coords(group(lm106, "mouth"))
    eyes = sorted((coords(group(lm106, "eye_a")), coords(group(lm106, "eye_b"))), key=lambda c: c[:, 0].mean())
    brows = sorted((coords(group(lm106, "brow_a")), coords(group(lm106, "brow_b"))), key=lambda c: c[:, 0].mean())
    u, v = mouth[:, 0], mouth[:, 1]
    corner_v = 0.5 * (v[np.argmin(u)] + v[np.argmax(u)])
    ar = [float(np.ptp(e[:, 1]) / iod_eff) for e in eyes]
    brow = [float((e[:, 1].mean() - b[:, 1].mean()) / iod_eff) for e, b in zip(eyes, brows)]
    return ExpressionMeasure(
        mouth_open=float(np.ptp(v) / iod_eff), mouth_width=float(np.ptp(u) / iod_img),
        smile=float((np.median(v) - corner_v) / iod_eff), eye_h_a=ar[0], eye_h_b=ar[1], brow_a=brow[0], brow_b=brow[1],
        iod_px=iod_img)


class NeutralCalibrator:
    """Per-track baselines: the track's own neutral, taken as robust percentiles over a sliding window."""

    def __init__(self, window: int, min_frames: int):
        self.min_frames = min_frames
        self.history: deque[np.ndarray] = deque(maxlen=window)

    def push(self, vector: np.ndarray) -> None:
        self.history.append(np.asarray(vector, np.float64))

    @property
    def calibrated(self) -> bool:
        return len(self.history) >= self.min_frames

    def baseline(self) -> np.ndarray:
        data = np.stack(self.history)
        base = np.median(data, axis=0)
        base[0] = np.percentile(data[:, 0], 10)       # mouth_open: neutral = nearly closed
        base[3] = np.percentile(data[:, 3], 90)       # eye openings: neutral = open
        base[4] = np.percentile(data[:, 4], 90)
        return base


def params_from_measure(m: ExpressionMeasure, base: np.ndarray) -> ExpressionParams:
    b = dict(zip(ExpressionMeasure.NAMES, base))
    def blink(opening: float, ref: float) -> float:
        # A few percent of eye-height variation is landmark noise, not a blink.
        # Preserve the full-closure endpoint while removing that neutral dead band.
        return float(np.clip((1.0 - opening / max(ref, 1e-6) - 0.03) / 0.57, 0.0, 1.0)) if ref > 0.01 else 0.0
    return ExpressionParams(
        mouth_open=max(0.0, m.mouth_open - b["mouth_open"]), mouth_width=m.mouth_width - b["mouth_width"],
        smile=m.smile - b["smile"], blink_a=blink(m.eye_h_a, b["eye_h_a"]), blink_b=blink(m.eye_h_b, b["eye_h_b"]),
        brow_a=m.brow_a - b["brow_a"], brow_b=m.brow_b - b["brow_b"])
