"""Shot-boundary and flash discrimination (blueprint §3.5).

Per frame three measurements against the previous frame, all on a fixed 64x64 thumbnail (resolution independent):

* ``H``  Bhattacharyya distance of the 8x8x8 HSV histograms (0 identical .. 1 disjoint);
* ``S``  ``1 - SSIM`` of the 64x64 luma (Gaussian-window SSIM implemented here, no skimage);
* ``V``  optional face pose velocity in degrees/second supplied by the caller (the tracker's rotation channel already
  resets at 900 deg/s, which is the default spike).

Decision, in this order:

1. **flash** -- the global luma jumped by more than ``flash_luma_delta`` while the *structure* is preserved
   (zero-mean luma correlation >= ``flash_structure``). A photographic flash, a lightning frame or the frame after it
   keeps the scene's layout but changes exposure. Consumers flush their filters (colour/detail history is wrong) but
   KEEP tracks: the same faces are still there.
2. **cut** -- ``(H > 0.45 and S > 0.35) or (H > 0.30 and S > 0.30 and V > spike)``.
3. otherwise **none**.

Limits (stated, not hidden): a flash so strong that it saturates the frame has no structure left, so it is
indistinguishable from a cut on the frame itself and is reported as a cut; a slow fade/dissolve changes every
measurement gradually and is reported as neither. The first frame after ``reset`` is never an event.

What the older detectors remain used for: ``video_tracker.ShotBoundaryDetector`` (32x32 colour difference with
``TrackerSettings`` thresholds) keeps serving the offline single-head tracker whose thresholds are part of its saved
settings; ``FaceTracker.process`` and ``FrameCompositor.compose`` use this detector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

CutKind = Literal["none", "cut", "flash"]
THUMBNAIL_SIDE = 64
_SSIM_C1, _SSIM_C2 = (0.01 * 255.0) ** 2, (0.03 * 255.0) ** 2


@dataclass(frozen=True)
class SceneCutConfig:
    cut_hist: float = 0.45
    cut_ssim_drop: float = 0.35
    soft_cut_hist: float = 0.30
    soft_cut_ssim_drop: float = 0.30
    pose_velocity_spike_dps: float = 900.0
    flash_luma_delta: float = 0.18       # normalised (0..1) jump of the mean luma
    flash_structure: float = 0.75        # zero-mean luma correlation above which the layout is "preserved"

    def __post_init__(self) -> None:
        values = (self.cut_hist, self.cut_ssim_drop, self.soft_cut_hist, self.soft_cut_ssim_drop,
                  self.pose_velocity_spike_dps, self.flash_luma_delta, self.flash_structure)
        if not np.isfinite(values).all() or min(values) <= 0 or self.flash_structure >= 1:
            raise ValueError("Invalid scene cut configuration")


@dataclass(frozen=True)
class CutEvent:
    kind: CutKind
    hist_distance: float
    ssim_drop: float
    pose_velocity: float | None
    frame_index: int
    cuts_total: int = 0

    def to_telemetry(self) -> dict:
        """The ``scene_cut`` / ``flash`` / ``cuts_total`` keys of the ``temporal`` block (blueprint §4)."""
        return {"scene_cut": self.kind == "cut", "flash": self.kind == "flash", "cuts_total": int(self.cuts_total)}


def ssim(first: np.ndarray, second: np.ndarray) -> float:
    """Mean SSIM of two equally sized single-channel images (11x11 Gaussian window, sigma 1.5, 0..255 range)."""
    a, b = np.asarray(first, np.float32), np.asarray(second, np.float32)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("SSIM needs two equally sized 2-D images")

    def blur(image: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(image, (11, 11), 1.5, borderType=cv2.BORDER_REFLECT)

    mu_a, mu_b = blur(a), blur(b)
    var_a, var_b = blur(a * a) - mu_a * mu_a, blur(b * b) - mu_b * mu_b
    covariance = blur(a * b) - mu_a * mu_b
    score = ((2 * mu_a * mu_b + _SSIM_C1) * (2 * covariance + _SSIM_C2)
             / ((mu_a * mu_a + mu_b * mu_b + _SSIM_C1) * (var_a + var_b + _SSIM_C2)))
    return float(score.mean())


def _structure_correlation(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a - a.mean(), b - b.mean()
    denominator = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denominator) if denominator > 1e-6 else 0.0


class SceneCutDetector:
    def __init__(self, config: SceneCutConfig = SceneCutConfig()):
        self.config = config
        self.cuts_total = 0
        self._luma: np.ndarray | None = None
        self._hist: np.ndarray | None = None
        self._frames = 0

    def reset(self) -> None:
        """Forget the previous frame (new clip). ``cuts_total`` is kept; it counts for the whole session."""
        self._luma = self._hist = None
        self._frames = 0

    @staticmethod
    def _signature(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        image = np.asarray(image_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8 or not image.size:
            raise ValueError("Scene cut detection needs a non-empty uint8 BGR frame")
        thumb = cv2.resize(image, (THUMBNAIL_SIDE, THUMBNAIL_SIDE), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([cv2.cvtColor(thumb, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, [8, 8, 8],
                            [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_L1)
        return cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).astype(np.float32), hist

    def observe(self, image_bgr: np.ndarray, *, pose_velocity: float | None = None,
                frame_index: int | None = None) -> CutEvent:
        luma, hist = self._signature(image_bgr)
        index = self._frames if frame_index is None else int(frame_index)
        velocity = None if pose_velocity is None else float(pose_velocity)
        if velocity is not None and not np.isfinite(velocity):
            raise ValueError("pose_velocity must be finite")
        previous_luma, previous_hist = self._luma, self._hist
        self._luma, self._hist, self._frames = luma, hist, self._frames + 1
        if previous_luma is None:
            return CutEvent("none", 0.0, 0.0, velocity, index, self.cuts_total)
        distance = float(cv2.compareHist(previous_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        drop = float(np.clip(1.0 - ssim(previous_luma, luma), 0.0, 2.0))
        c = self.config
        luma_jump = abs(float(luma.mean()) - float(previous_luma.mean())) / 255.0
        kind: CutKind = "none"
        if luma_jump > c.flash_luma_delta and _structure_correlation(previous_luma, luma) >= c.flash_structure:
            kind = "flash"
        elif (distance > c.cut_hist and drop > c.cut_ssim_drop) or (
                distance > c.soft_cut_hist and drop > c.soft_cut_ssim_drop
                and velocity is not None and velocity > c.pose_velocity_spike_dps):
            kind = "cut"
            self.cuts_total += 1
        return CutEvent(kind, distance, drop, velocity, index, self.cuts_total)
