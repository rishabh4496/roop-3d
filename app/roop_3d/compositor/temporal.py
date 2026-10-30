"""Bounded per-track One-Euro filtering for affine, mask and color state."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math

import cv2
import numpy as np

from roop_3d.compositor.color_match import ColorStats
from roop_3d.core.errors import DomainError


def _alpha(cutoff: np.ndarray | float, dt: float) -> np.ndarray | float:
    tau = 1.0 / (2.0 * math.pi * np.maximum(cutoff, 1e-4))
    return 1.0 / (1.0 + tau / max(dt, 1e-3))


class OneEuroFilter:
    """Array-valued One-Euro filter with velocity-adaptive cutoff."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.025, derivative_cutoff: float = 1.0):
        if min(min_cutoff, beta, derivative_cutoff) < 0 or derivative_cutoff == 0:
            raise ValueError("Invalid One-Euro filter parameters")
        self.min_cutoff, self.beta, self.derivative_cutoff = min_cutoff, beta, derivative_cutoff
        self.value: np.ndarray | None = None
        self.derivative: np.ndarray | None = None

    def reset(self, value: np.ndarray | None = None) -> np.ndarray | None:
        self.value = None if value is None else np.asarray(value, dtype=np.float32).copy()
        self.derivative = None if value is None else np.zeros_like(self.value)
        return None if self.value is None else self.value.copy()

    def update(self, value: np.ndarray, dt: float = 1.0) -> np.ndarray:
        current = np.asarray(value, dtype=np.float32)
        if not np.isfinite(current).all():
            raise DomainError("TEMPORAL_VALUE_INVALID", "Temporal filter input must be finite", 422)
        if self.value is None or self.value.shape != current.shape:
            return self.reset(current)  # type: ignore[return-value]
        raw_derivative = (current - self.value) / max(dt, 1e-3)
        derivative_alpha = _alpha(self.derivative_cutoff, dt)
        derivative = derivative_alpha * raw_derivative + (1.0 - derivative_alpha) * self.derivative
        cutoff = self.min_cutoff + self.beta * np.abs(derivative)
        value_alpha = _alpha(cutoff, dt)
        filtered = value_alpha * current + (1.0 - value_alpha) * self.value
        self.value, self.derivative = filtered.astype(np.float32), derivative.astype(np.float32)
        return self.value.copy()


@dataclass
class _TrackState:
    affine: OneEuroFilter
    alpha: OneEuroFilter
    color: OneEuroFilter
    frame_index: int | None

    @property
    def nbytes(self) -> int:
        return sum(array.nbytes for filter_ in (self.affine, self.alpha, self.color)
                   for array in (filter_.value, filter_.derivative) if array is not None)


@dataclass(frozen=True)
class TemporalResult:
    inverse: np.ndarray
    alpha: np.ndarray
    color_stats: ColorStats


class TemporalSmoother:
    """Track-local EMA state with scene-cut/loss resets and a hard byte ceiling."""

    def __init__(self, *, max_tracks: int = 16, max_cache_bytes: int = 32 * 1024 * 1024,
                 max_mask_side: int = 512, min_cutoff: float = 1.0, beta: float = 0.035):
        if max_tracks <= 0 or max_cache_bytes <= 0 or max_mask_side <= 0:
            raise ValueError("Temporal cache limits must be positive")
        self.max_tracks = int(max_tracks)
        self.max_cache_bytes = int(max_cache_bytes)
        self.max_mask_side = min(int(max_mask_side), max(1, int(math.sqrt(max_cache_bytes / 8))))
        self.min_cutoff, self.beta = float(min_cutoff), float(beta)
        self._tracks: OrderedDict[int, _TrackState] = OrderedDict()
        self.last_frame_index: int | None = None

    @property
    def cache_bytes(self) -> int:
        return sum(state.nbytes for state in self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()
        self.last_frame_index = None

    def reset_track(self, track_id: int) -> None:
        self._tracks.pop(track_id, None)

    def begin_frame(self, frame_index: int | None, *, active_track_ids: set[int] | None = None,
                    scene_cut: bool = False) -> None:
        if scene_cut:
            self.clear()
        elif active_track_ids is not None:
            for track_id in tuple(self._tracks):
                if track_id not in active_track_ids:
                    self._tracks.pop(track_id, None)
        self.last_frame_index = frame_index

    def _bounded_alpha(self, alpha: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        value = np.asarray(alpha, dtype=np.float32)
        if value.ndim != 2 or not np.isfinite(value).all():
            raise DomainError("TEMPORAL_MASK_INVALID", "Temporal alpha must be a finite 2-D mask", 422)
        original = value.shape
        scale = min(1.0, self.max_mask_side / max(value.shape))
        if scale < 1:
            size = (max(1, int(round(value.shape[1] * scale))), max(1, int(round(value.shape[0] * scale))))
            value = cv2.resize(value, size, interpolation=cv2.INTER_AREA)
        return np.clip(value, 0.0, 1.0), original

    def update(self, track_id: int, inverse: np.ndarray, alpha: np.ndarray, color_stats: ColorStats,
               *, frame_index: int | None = None, scene_cut: bool = False) -> TemporalResult:
        if type(track_id) is not int or track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        matrix = np.asarray(inverse, dtype=np.float32)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise DomainError("TEMPORAL_AFFINE_INVALID", "Crop inverse must be a finite 2x3 matrix", 422)
        bounded, original_shape = self._bounded_alpha(alpha)
        color = color_stats.vector()
        if scene_cut:
            self.clear()
        frame = frame_index if frame_index is not None else self.last_frame_index
        state = self._tracks.get(track_id)
        dt = 1.0
        if state is not None and frame is not None and state.frame_index is not None:
            dt = frame - state.frame_index
            if dt <= 0 or dt > 2:
                self._tracks.pop(track_id, None)
                state = None
                dt = 1.0
        if state is None:
            state = _TrackState(OneEuroFilter(self.min_cutoff, self.beta),
                                OneEuroFilter(self.min_cutoff, self.beta),
                                OneEuroFilter(self.min_cutoff, self.beta), frame)
            smoothed_matrix = state.affine.update(matrix)
            smoothed_alpha = state.alpha.update(bounded)
            smoothed_color = state.color.update(color)
            self._tracks[track_id] = state
        else:
            if state.alpha.value is not None and state.alpha.value.shape != bounded.shape:
                state = _TrackState(OneEuroFilter(self.min_cutoff, self.beta),
                                    OneEuroFilter(self.min_cutoff, self.beta),
                                    OneEuroFilter(self.min_cutoff, self.beta), frame)
                self._tracks[track_id] = state
                smoothed_matrix = state.affine.update(matrix)
                smoothed_alpha = state.alpha.update(bounded)
                smoothed_color = state.color.update(color)
            else:
                smoothed_matrix = state.affine.update(matrix, dt)
                smoothed_alpha = state.alpha.update(bounded, dt)
                smoothed_color = state.color.update(color, dt)
                state.frame_index = frame
                self._tracks.move_to_end(track_id)
        while len(self._tracks) > self.max_tracks or self.cache_bytes > self.max_cache_bytes:
            evicted_id, _ = self._tracks.popitem(last=False)
            if evicted_id == track_id:
                raise DomainError("TEMPORAL_CACHE_LIMIT", "Temporal state exceeds the configured memory bound", 422)
        if smoothed_alpha.shape != original_shape:
            smoothed_alpha = cv2.resize(smoothed_alpha, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        return TemporalResult(smoothed_matrix.astype(np.float32), np.clip(smoothed_alpha, 0, 1).astype(np.float32),
                              ColorStats.from_vector(smoothed_color))
