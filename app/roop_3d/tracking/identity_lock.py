"""Per-shot target identity lock: robust weighted-median pooling of a per-frame shape descriptor.

The descriptor is whichever of these the frame can honestly provide, and the shot keeps using the first kind it sees:

  * ``flame_beta``                  the first 100 FLAME shape coefficients of a supplied encoder (a true bone-shape beta);
  * ``landmark_proportions_106``    five pose-corrected facial ratios from the 5- and 106-point landmarks;
  * ``landmark_proportions_5``      two ratios from the 5-point landmarks alone.

The landmark kinds are proportions of the visible face, NOT FLAME coefficients, and are reported as such. They carry no
depth information. Ratios that involve the nose tip are deliberately absent: measured on the synthetic performer they move
by about 50 % over +/-20 degrees of pitch because the nose protrudes, which would read as a change of person. The rest move by
a few percent over +/-20 degrees, so frames are weighted to near-frontal poses (``pose_weight``) and the outlier scale has a
relative floor. The descriptor therefore detects a LARGE change of proportions, not a subtle one; it exists to give a shot a
stable identity reference, to notice an undetected change of person, and as a personalisation target for
``SceneTrackerAdaptation``.

Pooling: frames are weighted by how frontal and how neutral they are, a frame far (in robust-sigma units) from the pooled
median is an outlier and is not pooled, and the lock is declared only once enough frames agree and the median has stopped
moving. A locked beta is frozen. A run of consecutive frames far from a locked (or still-collecting) estimate is an
``identity_break``: the caller starts a new shot.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Literal

import numpy as np

from roop_3d.reconstruction.expression_tps import group

LockState = Literal["collecting", "provisional", "locked"]
KIND_FLAME, KIND_106, KIND_5 = "flame_beta", "landmark_proportions_106", "landmark_proportions_5"
PROPORTION_NAMES_106 = ("eye_mouth", "face_width", "face_height", "eye_width", "nose_width")
PROPORTION_NAMES_5 = ("eye_mouth", "mouth_width")
SIGMA_ABSOLUTE = {KIND_FLAME: 0.05, KIND_106: 0.01, KIND_5: 0.01}   # robust sigma is never below this ...
SIGMA_RELATIVE = {KIND_FLAME: 0.0, KIND_106: 0.03, KIND_5: 0.03}    # ... plus this fraction of the descriptor's own value
IDENTITY_YAW_DEG, IDENTITY_PITCH_DEG = 22.0, 16.0                   # weight reaches 0 at these angles


def pose_weight(yaw_deg: float, pitch_deg: float) -> float:
    """0..1: near-frontal frames describe the face best; at the limits the descriptor is not trusted at all."""
    return max(0.0, 1.0 - (yaw_deg / IDENTITY_YAW_DEG) ** 2) * max(0.0, 1.0 - (pitch_deg / IDENTITY_PITCH_DEG) ** 2)


def sigma_floor(kind: str | None, beta: np.ndarray) -> np.ndarray:
    return SIGMA_ABSOLUTE.get(kind, 0.01) + SIGMA_RELATIVE.get(kind, 0.03) * np.abs(beta)


def landmark_proportions(lm5, lm106, yaw_deg: float, pitch_deg: float) -> tuple[str, np.ndarray] | None:
    """Pose-corrected facial ratios in the face's own (rolled) frame, normalised by the foreshortened inter-ocular distance.

    Horizontal distances are divided by cos(yaw), vertical ones by cos(pitch): exact for a planar face, an approximation
    for a real one, which is why the lock weights frames by frontality instead of trusting extreme poses."""
    five = np.asarray(lm5, np.float64)
    across = five[1] - five[0]
    iod = float(np.linalg.norm(across))
    cy, cp = math.cos(math.radians(yaw_deg)), math.cos(math.radians(pitch_deg))
    if iod < 2.0 or cy < 0.25 or cp < 0.25:
        return None
    across = across / iod
    down = np.array((-across[1], across[0]))
    iod_eff = iod / cy
    eye_mid = five[:2].mean(axis=0)
    mouth_mid = five[3:].mean(axis=0)
    eye_mouth = abs(float((mouth_mid - eye_mid) @ down)) / cp / iod_eff
    if lm106 is None:
        return KIND_5, np.array((eye_mouth, abs(float((five[4] - five[3]) @ across)) / cy / iod_eff))
    pts = np.asarray(lm106, np.float64)
    span = lambda block, axis, scale: float(np.ptp((group(pts, block) - eye_mid) @ axis)) / scale
    eyes = 0.5 * (span("eye_a", across, cy) + span("eye_b", across, cy))
    return KIND_106, np.array((eye_mouth, span("contour", across, cy) / iod_eff, span("contour", down, cp) / iod_eff,
                               eyes / iod_eff, span("nose", across, cy) / iod_eff))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per-column weighted median of (N, D) values."""
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    cumulative = np.cumsum(weights[order], axis=0)
    index = (cumulative < 0.5 * cumulative[-1]).sum(axis=0)
    return sorted_values[index, np.arange(values.shape[1])]


@dataclass(frozen=True)
class IdentitySnapshot:
    state: LockState
    kind: str | None
    frames: int                         # frames currently pooled
    rejected: int                       # frames refused (low weight, other kind) or discarded as outliers
    beta: np.ndarray | None             # pooled median; frozen once locked
    sigma: np.ndarray | None            # robust per-dimension spread of the pool
    spread: float | None                # mean robust sigma / descriptor scale: how consistent the pool is
    distance: float | None              # robust distance of the latest accepted frame from the estimate (~1 = typical)
    break_detected: bool = False
    locked_at_frame: int | None = None
    prior_distance: float | None = None
    names: tuple[str, ...] = field(default_factory=tuple)


class ShotIdentityLock:
    def __init__(self, *, min_frames: int = 10, lock_frames: int = 30, window: int = 240, min_weight: float = 0.35,
                 outlier_distance: float = 3.5, break_run: int = 12, stable_tolerance: float = 0.25):
        if not (1 <= min_frames <= lock_frames <= window) or break_run < 2 or not 0 < min_weight <= 1 \
                or outlier_distance <= 1 or stable_tolerance <= 0:
            raise ValueError("Invalid identity-lock configuration")
        self.min_frames, self.lock_frames, self.window = min_frames, lock_frames, window
        self.min_weight, self.outlier_distance = min_weight, outlier_distance
        self.break_run, self.stable_tolerance = break_run, stable_tolerance
        self.prior: tuple[str, np.ndarray, np.ndarray] | None = None     # (kind, beta, sigma) from SceneTrackerAdaptation
        self.reset()

    def reset(self) -> None:
        self.kind: str | None = None
        self._pool: deque[tuple[np.ndarray, float]] = deque(maxlen=self.window)
        self._beta: np.ndarray | None = None
        self._sigma: np.ndarray | None = None
        self._history: deque[np.ndarray] = deque(maxlen=8)
        self.state: LockState = "collecting"
        self.rejected = 0
        self._bad_run = 0
        self._frames_seen = 0
        self._locked_at: int | None = None
        self._distance: float | None = None

    def set_prior(self, kind: str, beta: np.ndarray, sigma: np.ndarray) -> None:
        self.prior = (kind, np.asarray(beta, np.float64).copy(), np.asarray(sigma, np.float64).copy())

    # ------------------------------------------------------------------ pooling
    def _distance_to(self, vector: np.ndarray, beta: np.ndarray, sigma: np.ndarray) -> float:
        return float(np.sqrt(np.mean(((vector - beta) / sigma) ** 2)))

    def _estimate(self) -> None:
        values = np.stack([item[0] for item in self._pool])
        weights = np.array([item[1] for item in self._pool])
        beta = weighted_median(values, weights)
        mad = weighted_median(np.abs(values - beta), weights)
        self._beta, self._sigma = beta, np.maximum(1.4826 * mad, sigma_floor(self.kind, beta))

    def observe(self, kind: str, vector, weight: float) -> IdentitySnapshot:
        """Pool one frame's descriptor with its 0..1 weight (frontality x neutrality x confidence)."""
        vector = np.asarray(vector, np.float64)
        self._frames_seen += 1
        if self.kind is None:
            self.kind = kind
        if (kind != self.kind or not np.isfinite(vector).all() or not np.isfinite(weight) or weight < self.min_weight
                or (self._beta is not None and vector.shape != self._beta.shape)):
            self.rejected += 1
            return self.snapshot()
        weight = float(min(weight, 1.0))
        if self.state == "locked":
            self._distance = self._distance_to(vector, self._beta, self._sigma)
            self._bad_run = self._bad_run + 1 if self._distance > self.outlier_distance else 0
            return self.snapshot()
        if len(self._pool) >= self.min_frames:
            self._distance = self._distance_to(vector, self._beta, self._sigma)
            if self._distance > self.outlier_distance:
                self._bad_run += 1
                self.rejected += 1
                return self.snapshot()
        self._bad_run = 0
        self._pool.append((vector, weight))
        self._estimate()
        self._history.append(self._beta.copy())
        if len(self._pool) >= self.min_frames and self.state == "collecting":
            self.state = "provisional"
        if self.state == "provisional" and len(self._pool) >= self.lock_frames and len(self._history) == self._history.maxlen:
            # stable: the median moved by less than a fraction of the pool's own spread over the last frames
            shift = self._distance_to(self._history[-1], self._history[0], self._sigma)
            if shift <= self.stable_tolerance:
                self.state, self._locked_at = "locked", self._frames_seen
        return self.snapshot()

    def snapshot(self) -> IdentitySnapshot:
        prior_distance = None
        if self.prior is not None and self._beta is not None and self.prior[0] == self.kind and self.prior[1].shape == self._beta.shape:
            prior_distance = self._distance_to(self._beta, self.prior[1], np.maximum(self.prior[2], sigma_floor(self.kind, self.prior[1])))
        names = {KIND_106: PROPORTION_NAMES_106, KIND_5: PROPORTION_NAMES_5}.get(self.kind, ())
        scale = float(np.mean(np.abs(self._beta))) if self._beta is not None else 0.0
        return IdentitySnapshot(
            state=self.state, kind=self.kind, frames=len(self._pool), rejected=self.rejected,
            beta=None if self._beta is None or self.state == "collecting" else self._beta.copy(),
            sigma=None if self._sigma is None else self._sigma.copy(),
            spread=None if self._sigma is None or scale <= 0 else float(np.mean(self._sigma) / scale),
            distance=self._distance, break_detected=self._bad_run >= self.break_run, locked_at_frame=self._locked_at,
            prior_distance=prior_distance, names=tuple(names))
