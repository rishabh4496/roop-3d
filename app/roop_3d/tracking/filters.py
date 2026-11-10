"""Velocity-aware One-Euro + Kalman smoothing that cannot overshoot, and SO(3) head-rotation smoothing.

Why this is not ``temporal.TrackTemporalStabilizer.filter``: that stage runs a constant-velocity Kalman predictor
behind the One-Euro filter and extrapolates a second time. Measured here, a 70 degree yaw step overshoots to 108 degrees
and a 60 degree turn that stops overshoots to 87 degrees (``reenactment._EuroSmoother`` documents the same defect).

``SmoothedChannel`` keeps both stages but constrains the second one:

  1. The One-Euro filter (velocity-adaptive cut-off) produces the smoothed measurement ``z``.
  2. A constant-velocity Kalman filter, with an innovation-adaptive measurement noise, proposes the next output. A large
     innovation is a real movement rather than noise, so the gain rises towards 1; at rest the gain is small and the
     output is smoothed further.
  3. The output step is then confined to the segment between the previous output and ``z`` (a convex combination), so the
     Kalman stage can slow the response down but can never run past, or away from, the measurement.

The one place the velocity state extrapolates is ``coast``: a frame without a usable measurement is bridged by the
prediction, with the velocity decaying, for at most ``max_coast_s``. Gaps, non-monotonic time and shape changes
reinitialise the channel at the measurement, exactly like the One-Euro filter does.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np

from roop_3d.temporal import OneEuroFilter, TrackTemporalStabilizer

GAP_RESET_S = 0.5     # same horizon as OneEuroFilter / TrackTemporalStabilizer


@dataclass(frozen=True)
class FilterSpec:
    """Per-channel smoothing. ``noise`` and ``accel`` are in the channel's own unit (and unit/s^2)."""
    min_cutoff: float
    beta: float
    d_cutoff: float
    noise: float                    # standard deviation of the One-Euro output around the truth
    accel: float                    # expected acceleration standard deviation (process noise)
    max_coast_s: float = 0.25
    velocity_tau_s: float = 0.15    # coasting velocity decays with this time constant

    def __post_init__(self):
        values = (self.min_cutoff, self.beta, self.d_cutoff, self.noise, self.accel, self.max_coast_s, self.velocity_tau_s)
        if not np.isfinite(values).all() or min(self.min_cutoff, self.d_cutoff, self.noise, self.accel, self.velocity_tau_s) <= 0 \
                or self.beta < 0 or self.max_coast_s < 0:
            raise ValueError("Invalid filter specification")


# Units: translation is in shot-reference inter-ocular distances, expression in inter-ocular units, angles in radians.
# Pose is smooth; expression and gaze keep blink / saccade speed (cf. ReenactmentSettings); lighting is nearly static.
DEFAULT_SPECS: dict[str, FilterSpec] = {
    "translation": FilterSpec(1.5, 0.6, 8.0, noise=0.004, accel=6.0),
    "log_scale": FilterSpec(1.5, 4.0, 8.0, noise=0.004, accel=2.0),
    "jaw": FilterSpec(4.0, 2.0, 12.0, noise=0.01, accel=60.0),
    "expression": FilterSpec(6.0, 1.0, 15.0, noise=0.006, accel=40.0),
    "psi": FilterSpec(2.0, 0.3, 10.0, noise=0.05, accel=10.0),           # FLAME coefficients are in sigma units
    "gaze": FilterSpec(1.5, 1.5, 10.0, noise=0.01, accel=150.0, max_coast_s=0.15),
    "lighting": FilterSpec(0.4, 0.0, 1.0, noise=0.02, accel=0.5, max_coast_s=1.0, velocity_tau_s=0.5),
}
ROTATION_SPEC = dict(min_cutoff=1.5, beta=0.03, derivative_cutoff=8.0, rotation_reset_speed=1400.0)   # degrees / second


class SmoothedChannel:
    """One array-valued channel: One-Euro, then a no-overshoot Kalman stage."""

    def __init__(self, spec: FilterSpec):
        self.spec = spec
        self._euro = OneEuroFilter(spec.min_cutoff, spec.beta, spec.d_cutoff)
        self.reset()

    def reset(self) -> None:
        self._euro.reset()
        self.x = self.v = self.pxx = self.pxv = self.pvv = None
        self.time: float | None = None
        self.coasted = 0.0

    @property
    def value(self) -> np.ndarray | None:
        return None if self.x is None else self.x.copy()

    def _predict(self, dt: float):
        s = self.spec
        decay = math.exp(-dt / s.velocity_tau_s)
        q = s.accel * s.accel
        x_p, v_p = self.x + self.v * dt, self.v * decay
        pxx = self.pxx + 2 * dt * self.pxv + dt * dt * self.pvv + q * dt ** 3 / 3
        pxv = decay * (self.pxv + dt * self.pvv) + q * dt * dt / 2
        pvv = decay * decay * self.pvv + q * dt
        return x_p, v_p, pxx, pxv, pvv

    def _start(self, z: np.ndarray, t: float) -> np.ndarray:
        s = self.spec
        self._euro.reset()
        self._euro.update(z, t)
        self.x, self.v = z.copy(), np.zeros_like(z)
        self.pxx = np.full_like(z, s.noise ** 2)
        self.pxv = np.zeros_like(z)
        self.pvv = np.full_like(z, (0.3 * s.accel) ** 2)
        self.time, self.coasted = float(t), 0.0
        return z.copy()

    def update(self, value, timestamp: float) -> np.ndarray:
        raw = np.asarray(value, dtype=np.float64)
        if raw.size == 0 or not np.isfinite(raw).all() or not np.isfinite(timestamp):
            raise ValueError("Filter inputs must be finite and non-empty")
        if (self.x is None or raw.shape != self.x.shape or timestamp <= self.time
                or timestamp - self.time > GAP_RESET_S):
            return self._start(raw, timestamp)
        dt = float(timestamp - self.time)
        z = self._euro.update(raw, timestamp)
        x_p, v_p, pxx, pxv, pvv = self._predict(dt)
        r = self.spec.noise ** 2
        innovation = z - x_p
        surprise = innovation * innovation / (pxx + r)
        r_eff = r / (1.0 + np.maximum(0.0, surprise - 1.0))     # a surprising move is a movement, not noise
        s_inn = pxx + r_eff
        kx, kv = pxx / s_inn, pxv / s_inn
        x, v = x_p + kx * innovation, v_p + kv * innovation
        self.pxx, self.pxv, self.pvv = np.maximum((1 - kx) * pxx, 1e-12), (1 - kx) * pxv, np.maximum(pvv - kv * pxv, 1e-12)
        # Confine the step to [previous output, z]: slower than the One-Euro output is allowed, past it is not.
        previous, span = self.x, z - self.x
        moving = np.abs(span) > 1e-12
        fraction = np.clip(np.divide(x - previous, span, out=np.ones_like(span), where=moving), 0.0, 1.0)
        confined = np.where(moving, previous + fraction * span, z)
        self.v = np.where(np.abs(confined - x) > 1e-12, (confined - previous) / dt, v)
        self.x = confined
        self.time, self.coasted = float(timestamp), 0.0
        return self.x.copy()

    def coast(self, timestamp: float) -> np.ndarray | None:
        """Bridge a frame without a measurement from the prediction; None when there is nothing (left) to bridge with."""
        if self.x is None or not np.isfinite(timestamp) or timestamp <= self.time:
            return None
        dt = float(timestamp - self.time)
        if self.coasted + dt > self.spec.max_coast_s:
            return None
        self.x, self.v, self.pxx, self.pxv, self.pvv = self._predict(dt)
        self.time, self.coasted = float(timestamp), self.coasted + dt
        return self.x.copy()


class TrackFilterBank:
    """All channels of one track, with one ``reset`` for a shot boundary or a lost track."""

    def __init__(self, specs: Mapping[str, FilterSpec] | None = None):
        self.specs = dict(DEFAULT_SPECS if specs is None else specs)
        self._channels = {name: SmoothedChannel(spec) for name, spec in self.specs.items()}
        self._rotation = TrackTemporalStabilizer(**ROTATION_SPEC)
        self._rotation_time: float | None = None
        self._rotation_value: np.ndarray | None = None

    def reset(self) -> None:
        for channel in self._channels.values():
            channel.reset()
        self._rotation.reset()
        self._rotation_time = self._rotation_value = None

    def update(self, name: str, value, timestamp: float) -> np.ndarray:
        return self._channels[name].update(value, timestamp)

    def coast(self, name: str, timestamp: float) -> np.ndarray | None:
        return self._channels[name].coast(timestamp)

    def value(self, name: str) -> np.ndarray | None:
        return self._channels[name].value

    def rotation(self, matrix, timestamp: float) -> np.ndarray:
        """SO(3) One-Euro (quaternion slerp): a pure low-pass, so it cannot overshoot, and it has no Euler wrap."""
        result = self._rotation.filter_rotation_matrix(0, "head", matrix, timestamp)
        self._rotation_time, self._rotation_value = float(timestamp), result.copy()
        return result

    def coast_rotation(self, timestamp: float, max_coast_s: float = 0.25) -> np.ndarray | None:
        """A rotation has no velocity state to coast on, so it is held for at most ``max_coast_s``."""
        if self._rotation_value is None or timestamp - self._rotation_time > max_coast_s:
            return None
        return self._rotation_value.copy()
