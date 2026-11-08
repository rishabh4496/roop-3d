"""Track-local temporal filters for observations, coefficients, and segmentation masks.

All state is bounded by the active tracks supplied by the caller. Timestamp gaps,
shape changes, and scene cuts reset the affected trajectory instead of blending
unrelated frames.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import cv2
import numpy as np


class OneEuroFilter:
    """Array-valued One-Euro filter with velocity-adaptive cutoff."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = .035, derivative_cutoff: float = 1.0):
        if (not np.isfinite([min_cutoff, beta, derivative_cutoff]).all()
                or min_cutoff <= 0 or beta < 0 or derivative_cutoff <= 0):
            raise ValueError("Invalid One-Euro filter configuration")
        self.min_cutoff, self.beta, self.derivative_cutoff = float(min_cutoff), float(beta), float(derivative_cutoff)
        self.time: float | None = None
        self.raw: np.ndarray | None = None
        self.value: np.ndarray | None = None
        self.derivative: np.ndarray | None = None

    def reset(self) -> None:
        self.time = self.raw = self.value = self.derivative = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, dt: float):
        tau = 1.0 / (2 * math.pi * np.maximum(cutoff, 1e-4))
        return 1.0 / (1.0 + tau / dt)

    def update(self, value, timestamp: float) -> np.ndarray:
        current = np.asarray(value, dtype=np.float64)
        if current.size == 0 or not np.isfinite(current).all() or not np.isfinite(timestamp):
            raise ValueError("Filter inputs must be finite and non-empty")
        if self.time is None or current.shape != self.raw.shape or timestamp <= self.time or timestamp - self.time > .5:
            self.time, self.raw, self.value = float(timestamp), current.copy(), current.copy()
            self.derivative = np.zeros_like(current)
            return current.copy()
        dt = float(timestamp - self.time)
        a_d = self._alpha(self.derivative_cutoff, dt)
        derivative = a_d * (current - self.raw) / dt + (1 - a_d) * self.derivative
        a = self._alpha(self.min_cutoff + self.beta * np.abs(derivative), dt)
        filtered = a * current + (1 - a) * self.value
        self.time, self.raw = float(timestamp), current.copy()
        self.value, self.derivative = filtered, derivative
        return filtered.copy()


@dataclass
class _KalmanChannel:
    euro: OneEuroFilter
    position: np.ndarray | None = None
    velocity: np.ndarray | None = None
    position_variance: float = 1.0
    velocity_variance: float = 10.0
    timestamp: float | None = None


@dataclass
class _RotationChannel:
    raw: np.ndarray
    value: np.ndarray
    derivative: float
    timestamp: float


class TrackTemporalStabilizer:
    """Track-local One-Euro/Kalman state with SO(3)-aware rotation filtering."""

    def __init__(self, *, min_cutoff=.15, beta=.04, derivative_cutoff=1., process_noise=2., measurement_noise=4.,
                 rotation_reset_speed=900.):
        if (not np.isfinite([process_noise, measurement_noise]).all()
                or process_noise <= 0 or measurement_noise <= 0):
            raise ValueError("Kalman noise values must be positive and finite")
        if not np.isfinite(rotation_reset_speed) or rotation_reset_speed <= 0:
            raise ValueError("Rotation reset speed must be positive and finite")
        self.config = dict(min_cutoff=min_cutoff, beta=beta, derivative_cutoff=derivative_cutoff)
        self.process_noise, self.measurement_noise = float(process_noise), float(measurement_noise)
        self.rotation_reset_speed = float(rotation_reset_speed)
        self.channels: dict[tuple[int, str], _KalmanChannel] = {}
        self.rotations: dict[tuple[int, str], _RotationChannel] = {}

    def reset(self) -> None:
        self.channels.clear()
        self.rotations.clear()

    def retain(self, active_ids) -> None:
        active = set(active_ids)
        self.channels = {key: value for key, value in self.channels.items() if key[0] in active}
        self.rotations = {key: value for key, value in self.rotations.items() if key[0] in active}

    def forget_track(self, track_id: int) -> None:
        self.channels = {key: value for key, value in self.channels.items() if key[0] != track_id}
        self.rotations = {key: value for key, value in self.rotations.items() if key[0] != track_id}

    @staticmethod
    def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
        """Convert a proper rotation matrix to a normalized ``(w, x, y, z)`` quaternion."""
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape != (3, 3) or not np.isfinite(m).all():
            raise ValueError("Rotation must be a finite 3x3 matrix")
        if not np.allclose(m.T @ m, np.eye(3), atol=2e-3) or np.linalg.det(m) < 0.99:
            raise ValueError("Rotation matrix must be orthonormal and right-handed")
        trace = float(np.trace(m))
        if trace > 0:
            scale = math.sqrt(trace + 1.0) * 2.0
            q = np.array((.25 * scale, (m[2, 1] - m[1, 2]) / scale,
                          (m[0, 2] - m[2, 0]) / scale, (m[1, 0] - m[0, 1]) / scale))
        else:
            axis = int(np.argmax(np.diag(m)))
            if axis == 0:
                scale = math.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
                q = np.array(((m[2, 1] - m[1, 2]) / scale, .25 * scale,
                              (m[0, 1] + m[1, 0]) / scale, (m[0, 2] + m[2, 0]) / scale))
            elif axis == 1:
                scale = math.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
                q = np.array(((m[0, 2] - m[2, 0]) / scale, (m[0, 1] + m[1, 0]) / scale,
                              .25 * scale, (m[1, 2] + m[2, 1]) / scale))
            else:
                scale = math.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
                q = np.array(((m[1, 0] - m[0, 1]) / scale, (m[0, 2] + m[2, 0]) / scale,
                              (m[1, 2] + m[2, 1]) / scale, .25 * scale))
        return q / max(float(np.linalg.norm(q)), 1e-12)

    @staticmethod
    def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
        w, x, y, z = np.asarray(quaternion, dtype=np.float64)
        return np.array(((1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
                         (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
                         (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y))), np.float64)

    @staticmethod
    def _slerp(a: np.ndarray, b: np.ndarray, amount: float) -> np.ndarray:
        dot = float(np.dot(a, b))
        if dot < 0.0:
            b, dot = -b, -dot
        dot = float(np.clip(dot, -1.0, 1.0))
        if dot > .9995:
            result = a + amount * (b - a)
            return result / max(float(np.linalg.norm(result)), 1e-12)
        angle = math.acos(dot)
        sine = math.sin(angle)
        result = (math.sin((1.0 - amount) * angle) * a + math.sin(amount * angle) * b) / sine
        return result / max(float(np.linalg.norm(result)), 1e-12)

    def filter_rotation_matrix(self, track_id: int, channel: str, matrix, timestamp: float) -> np.ndarray:
        """Velocity-adaptive One-Euro filter on SO(3), interpolating with quaternion SLERP.

        Non-monotonic timestamps, gaps over 500 ms, and shape/lifecycle resets take the
        measurement immediately. Scene cuts are handled by ``reset`` at the tracker boundary.
        """
        if type(track_id) is not int or track_id < 0 or not channel or not np.isfinite(timestamp):
            raise ValueError("A valid track, channel, and finite timestamp are required")
        raw = self._matrix_to_quaternion(np.asarray(matrix, dtype=np.float64))
        key = (track_id, channel)
        state = self.rotations.get(key)
        if state is None or timestamp <= state.timestamp or timestamp - state.timestamp > .5:
            self.rotations[key] = _RotationChannel(raw.copy(), raw.copy(), 0.0, float(timestamp))
            return self._quaternion_to_matrix(raw)
        if np.dot(raw, state.raw) < 0:
            raw = -raw
        dt = float(timestamp - state.timestamp)
        angular_speed = math.degrees(2.0 * math.acos(float(np.clip(abs(np.dot(raw, state.raw)), 0.0, 1.0))) / dt)
        if angular_speed >= self.rotation_reset_speed:
            self.rotations[key] = _RotationChannel(raw.copy(), raw.copy(), angular_speed, float(timestamp))
            return self._quaternion_to_matrix(raw)
        derivative_alpha = float(OneEuroFilter._alpha(self.config["derivative_cutoff"], dt))
        derivative = derivative_alpha * angular_speed + (1.0 - derivative_alpha) * state.derivative
        cutoff = self.config["min_cutoff"] + self.config["beta"] * derivative
        amount = float(OneEuroFilter._alpha(cutoff, dt))
        filtered = self._slerp(state.value, raw, amount)
        self.rotations[key] = _RotationChannel(raw.copy(), filtered.copy(), derivative, float(timestamp))
        return self._quaternion_to_matrix(filtered)

    def filter_pose(self, track_id: int, channel: str, legacy_angles, timestamp: float) -> np.ndarray:
        """Filter a legacy ``(yaw, pitch-down, roll)`` pose without Euler wrap artifacts."""
        from roop_3d.core.pose import as_pose3d, euler_from_rotation, rotation_from_euler
        pose = as_pose3d(legacy_angles, frame="legacy")
        rotation = rotation_from_euler(pose.yaw, pose.pitch, pose.roll)
        filtered = self.filter_rotation_matrix(track_id, channel, rotation, timestamp)
        yaw, pitch, roll = euler_from_rotation(filtered)
        return np.asarray((yaw, -pitch, roll), dtype=np.float64)

    def filter(self, track_id: int, channel: str, value, timestamp: float) -> np.ndarray:
        if type(track_id) is not int or track_id < 0 or not channel:
            raise ValueError("A nonnegative track id and channel name are required")
        measurement = np.asarray(value, dtype=np.float64)
        if measurement.size == 0 or not np.isfinite(measurement).all() or not np.isfinite(timestamp):
            raise ValueError("Temporal measurements must be finite and non-empty")
        key = (track_id, channel)
        state = self.channels.get(key)
        if (state is None or state.position is None or state.position.shape != measurement.shape
                or state.timestamp is None or timestamp <= state.timestamp or timestamp - state.timestamp > .5):
            state = _KalmanChannel(OneEuroFilter(**self.config), position=measurement.copy(),
                                   velocity=np.zeros_like(measurement), timestamp=float(timestamp))
            self.channels[key] = state
            state.euro.update(measurement, timestamp)
            return measurement.copy()

        dt = float(timestamp - state.timestamp)
        predicted = state.position + state.velocity * dt
        p_pos = state.position_variance + dt * dt * state.velocity_variance + self.process_noise * dt
        smooth_measurement = state.euro.update(measurement, timestamp)
        gain = p_pos / (p_pos + self.measurement_noise)
        position = predicted + gain * (smooth_measurement - predicted)
        velocity = state.velocity + (gain / max(dt, 1e-6)) * (smooth_measurement - predicted)
        state.position, state.velocity = position, velocity
        state.position_variance = max(1e-6, (1 - gain) * p_pos)
        state.velocity_variance = max(1e-6, (1 - min(gain, .95)) * state.velocity_variance + self.process_noise * dt)
        state.timestamp = float(timestamp)
        return position.copy()


@dataclass
class _MaskState:
    value: np.ndarray
    frame: int | None


class TemporalMaskEMA:
    """Per-track temporal EMA for model alpha masks; onset is immediate, release is eased."""

    def __init__(self, *, alpha=.65, max_tracks=32, max_pixels=32 * 512 * 512):
        if not 0 < alpha <= 1 or max_tracks < 1 or max_pixels < 1:
            raise ValueError("Invalid temporal mask bounds")
        self.alpha, self.max_tracks, self.max_pixels = float(alpha), int(max_tracks), int(max_pixels)
        self._states: dict[tuple[int, str], _MaskState] = {}

    def clear(self):
        self._states.clear()

    def retain(self, active_ids):
        active = set(active_ids)
        self._states = {key: state for key, state in self._states.items() if key[0] in active}

    def update(self, track_id: int, mask: np.ndarray, *, frame: int | None = None, kind="composite") -> np.ndarray:
        value = np.asarray(mask, dtype=np.float32)
        if value.ndim != 2 or not value.size or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Mask must be a finite 2-D alpha map in 0..1")
        key = (int(track_id), str(kind))
        state = self._states.get(key)
        if state is None or state.value.shape != value.shape or (frame is not None and state.frame is not None and (frame <= state.frame or frame - state.frame > 2)):
            result = value.copy()
        else:
            blended = self.alpha * value + (1 - self.alpha) * state.value
            if kind in {"occluder", "manual_exclude"}:
                # Never delay foreground or an explicit exclusion; ease its release to avoid flicker.
                result = np.where(value >= state.value, value, blended)
            elif kind.startswith("manual"):
                result = value.copy()
            else:
                result = blended
        self._states[key] = _MaskState(result.astype(np.float32), frame)
        while len(self._states) > self.max_tracks * 4 or sum(s.value.size for s in self._states.values()) > self.max_pixels:
            self._states.pop(next(iter(self._states)))
        return result.astype(np.float32, copy=True)
