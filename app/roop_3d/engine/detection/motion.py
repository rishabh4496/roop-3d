"""Motion models for face association: GIoU, constant-velocity Kalman predictors and a short head-pose trajectory.

* :func:`giou` ranks candidate boxes even when they no longer overlap (a fast or sparsely sampled move).
* :class:`BoxMotion` is a constant-velocity Kalman filter over ``(cx, cy, log size, log aspect)``. Noise is relative to the face
  size, time is the *frame number* (not the call count), so a sparse adaptive scan predicts across its gaps correctly, and a track
  that is not detected coasts on a decaying velocity instead of freezing or running away.
* :class:`CenterMotion3D` is the same filter over the metric head-centre translation of the 6-DoF pose (metres, camera frame).
  Its Mahalanobis distance is a depth-aware plausibility gate for a spatial link.
* :class:`PoseTrajectory` keeps the last few ``(frame, yaw, pitch, roll)`` samples so a head that is turning into profile is
  recognised *before* its embedding similarity collapses.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# chi-square 0.9999 quantile, 3 degrees of freedom: a metric-centre jump beyond this is not the same head moving.
CENTER_GATE_CHI2 = 21.1


def giou(left, right) -> float:
    """Generalised IoU in [-1, 1]: IoU minus the share of the enclosing box that neither box covers."""
    lx1, ly1, lx2, ly2 = (float(v) for v in left[:4])
    rx1, ry1, rx2, ry2 = (float(v) for v in right[:4])
    inter = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(0.0, min(ly2, ry2) - max(ly1, ry1))
    union = max(lx2 - lx1, 0.0) * max(ly2 - ly1, 0.0) + max(rx2 - rx1, 0.0) * max(ry2 - ry1, 0.0) - inter
    enclosing = (max(lx2, rx2) - min(lx1, rx1)) * (max(ly2, ry2) - min(ly1, ry1))
    if union <= 0.0 or enclosing <= 0.0:
        return -1.0
    return inter / union - (enclosing - union) / enclosing


class ConstantVelocityKalman:
    """Kalman filter with state ``[z, dz/dt]``; ``dt`` is in frames and may be fractional or span a gap.

    ``accel_std`` / ``measurement_std`` are per-dimension, multiplied by the ``scale`` given at each step (face size in pixels
    for position dimensions, 1 for logarithmic ones).
    """

    def __init__(self, z, *, accel_std, measurement_std, velocity_decay: float = 0.93, velocity_std=None):
        z = np.asarray(z, np.float64).ravel()
        self.d = z.size
        self.accel = np.asarray(accel_std, np.float64).ravel()
        self.meas = np.asarray(measurement_std, np.float64).ravel()
        if self.accel.size != self.d or self.meas.size != self.d or not 0.0 < velocity_decay <= 1.0:
            raise ValueError("Kalman dimensions or decay are invalid")
        self.decay = float(velocity_decay)
        self.x = np.concatenate([z, np.zeros(self.d)])
        velocity = np.asarray(self.accel if velocity_std is None else velocity_std, np.float64).ravel()
        self.P = np.diag(np.concatenate([self.meas ** 2, velocity ** 2]))
        self._initial_scale = None

    def _gain(self, dt: float) -> float:
        """Displacement per unit of initial velocity over ``dt`` frames while the velocity decays geometrically."""
        if dt <= 0:
            return 0.0
        return float(dt) if self.decay == 1.0 else (1.0 - self.decay ** dt) / (1.0 - self.decay)

    def _transition(self, dt: float, scale) -> tuple[np.ndarray, np.ndarray]:
        d = self.d
        F = np.eye(2 * d)
        F[:d, d:] = np.eye(d) * self._gain(dt)
        F[d:, d:] = np.eye(d) * (self.decay ** dt)
        q = (self.accel * np.asarray(scale, np.float64)) ** 2
        Q = np.zeros((2 * d, 2 * d))
        Q[:d, :d] = np.diag(q * dt ** 3 / 3.0)
        Q[:d, d:] = Q[d:, :d] = np.diag(q * dt ** 2 / 2.0)
        Q[d:, d:] = np.diag(q * dt)
        return F, Q

    def peek(self, dt: float) -> np.ndarray:
        """Predicted measurement ``dt`` frames ahead; the filter is not changed."""
        return self.x[:self.d] + self._gain(dt) * self.x[self.d:]

    def predict(self, dt: float, scale) -> None:
        if dt <= 0:
            return
        F, Q = self._transition(dt, scale)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z, scale, noise_gain: float = 1.0) -> None:
        z = np.asarray(z, np.float64).ravel()
        H = np.hstack([np.eye(self.d), np.zeros((self.d, self.d))])
        R = np.diag((self.meas * np.asarray(scale, np.float64) * noise_gain) ** 2)
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - H @ self.x)
        self.P = (np.eye(2 * self.d) - K @ H) @ self.P

    def distance2(self, z, dt: float, scale) -> float:
        """Squared Mahalanobis distance of measurement ``z`` to the prediction ``dt`` frames ahead."""
        z = np.asarray(z, np.float64).ravel()
        if dt > 0:
            F, Q = self._transition(dt, scale)
            P = F @ self.P @ F.T + Q
        else:
            P = self.P
        H = np.hstack([np.eye(self.d), np.zeros((self.d, self.d))])
        S = H @ P @ H.T + np.diag((self.meas * np.asarray(scale, np.float64)) ** 2)
        innovation = z - self.peek(max(dt, 0.0))
        return float(innovation @ np.linalg.solve(S, innovation))


def _box_to_z(box) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    width, height = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    size = float(np.sqrt(width * height))
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, np.log(size), np.log(width / height)]), size


def _z_to_box(z) -> np.ndarray:
    size, aspect = float(np.exp(z[2])), float(np.exp(z[3]))
    width, height = size * np.sqrt(aspect), size / np.sqrt(aspect)
    return np.array([z[0] - width / 2, z[1] - height / 2, z[0] + width / 2, z[1] + height / 2], np.float32)


class BoxMotion:
    """Kalman-predicted face box in frame-number time."""

    def __init__(self, box, frame: float):
        z, size = _box_to_z(box)
        self.filter = ConstantVelocityKalman(
            z, accel_std=(0.03, 0.03, 0.012, 0.006), measurement_std=(0.05, 0.05, 0.04, 0.05), velocity_decay=0.93)
        self.frame = float(frame)
        self.size = size

    def _scale(self) -> np.ndarray:
        return np.array([self.size, self.size, 1.0, 1.0])

    def predict_box(self, frame: float) -> np.ndarray:
        return _z_to_box(self.filter.peek(max(0.0, float(frame) - self.frame)))

    def coast(self, frame: float) -> None:
        dt = float(frame) - self.frame
        if dt > 0:
            self.filter.predict(dt, self._scale())
            self.frame = float(frame)
            self.size = float(np.exp(self.filter.x[2]))

    def update(self, box, frame: float, noise_gain: float = 1.0) -> None:
        self.coast(frame)
        z, _ = _box_to_z(box)
        self.filter.update(z, self._scale(), noise_gain)
        self.size = float(np.exp(self.filter.x[2]))

    def reset(self, box, frame: float) -> None:
        self.__init__(box, frame)


class CenterMotion3D:
    """Kalman-predicted metric head centre (camera frame, metres); depth is much less certain than the image plane."""

    def __init__(self, translation, frame: float):
        self.filter = ConstantVelocityKalman(
            translation, accel_std=(0.012, 0.012, 0.04), measurement_std=(0.02, 0.02, 0.12), velocity_decay=0.93)
        self.frame = float(frame)

    def coast(self, frame: float) -> None:
        dt = float(frame) - self.frame
        if dt > 0:
            self.filter.predict(dt, np.ones(3))
            self.frame = float(frame)

    def update(self, translation, frame: float) -> None:
        self.coast(frame)
        self.filter.update(translation, np.ones(3))

    def distance2(self, translation, frame: float) -> float:
        return self.filter.distance2(translation, max(0.0, float(frame) - self.frame), np.ones(3))


class PoseTrajectory:
    """Last few head-pose samples of one track (frame number, yaw, pitch, roll in degrees)."""

    def __init__(self, maxlen: int = 24):
        self.samples: deque[tuple[float, float, float, float]] = deque(maxlen=maxlen)

    def add(self, frame: float, yaw: float, pitch: float, roll: float) -> None:
        self.samples.append((float(frame), float(yaw), float(pitch), float(roll)))

    @property
    def last_yaw(self) -> float | None:
        return self.samples[-1][1] if self.samples else None

    def predicted_yaw(self, frame: float, window: int = 6) -> float | None:
        """Linear extrapolation of the recent yaw to ``frame`` (limited to one second of extrapolation at 30 fps)."""
        recent = list(self.samples)[-window:]
        if not recent:
            return None
        if len(recent) < 2 or recent[-1][0] - recent[0][0] <= 0:
            return recent[-1][1]
        frames = np.array([item[0] for item in recent])
        yaws = np.array([item[1] for item in recent])
        slope = float(np.polyfit(frames, yaws, 1)[0])
        ahead = min(max(float(frame) - recent[-1][0], 0.0), 30.0)
        return float(np.clip(recent[-1][1] + slope * ahead, -120.0, 120.0))

    def turning(self, frame: float, yaw: float, *, steep: float, step: float) -> bool:
        """True when this observation is steep, jumps by ``step`` degrees against the last sample, or the trend heads to steep."""
        if abs(yaw) >= steep:
            return True
        last = self.last_yaw
        if last is not None and abs(yaw - last) >= step:
            return True
        predicted = self.predicted_yaw(frame)
        return predicted is not None and abs(predicted) >= steep
