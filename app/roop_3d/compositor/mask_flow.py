"""Dense-flow alpha-matte stabilisation (blueprint §3.6) and the compositor's single dense-flow source.

``FlowEstimator`` is the one place the compositor creates a CPU dense optical flow: OpenCV DIS (medium preset) when
the build has it, Farneback otherwise (``backend`` names which). ``occlusion_tracker`` and ``detail_temporal`` use it
for their own flow; ``FlowMaskStabilizer`` builds the §3.6 matte filter on top::

    B = flow(current -> previous)      F = flow(previous -> current)        (both on a <= ``flow_side`` px copy)
    q = p + B(p)                       err(p) = || B(p) + F(q) ||           (forward-backward consistency)
    conf = exp(-err^2 / sigma^2)       c = c_max * conf                     (c_max = 0.6)
    alpha_out = (1 - c) * alpha_t + c * warp(alpha_out_{t-1}, B)

The history is the *stabilised* previous matte (recursive), so noise falls roughly like ``(1-c)^2 / (1-c^2)`` while a
moving edge is carried by the flow instead of being smeared. ``c = 0`` wherever the flow is inconsistent (occlusion,
disocclusion, a hand entering) and for the whole frame after ``reset`` (scene cut or flash), so no pixel of the old
shot can leak into the new one. Limits: flow is estimated from the guide image, so a texture-less guide yields
unreliable flow and the consistency check then lowers ``c`` rather than trusting it; matte motion without matching
image motion is therefore not propagated.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

DIS_MEDIUM = getattr(cv2, "DISOPTICAL_FLOW_PRESET_MEDIUM", None)


class FlowEstimator:
    """Dense flow ``a -> b`` for 8-bit grayscale images: ``a(p) ~ b(p + flow(p))``. One instance per thread."""

    def __init__(self, preset: int | None = DIS_MEDIUM):
        if hasattr(cv2, "DISOpticalFlow_create") and preset is not None:
            self._dis = cv2.DISOpticalFlow_create(preset)
            self.backend = "dis"
        else:
            self._dis = None
            self.backend = "farneback"

    def calc(self, a: np.ndarray, b: np.ndarray, flow: np.ndarray | None = None) -> np.ndarray:
        if a.shape != b.shape or a.ndim != 2 or a.dtype != np.uint8 or b.dtype != np.uint8:
            raise DomainError("FLOW_INPUT_INVALID", "Dense flow needs two equally sized uint8 grayscale images", 422)
        if self._dis is not None:
            return self._dis.calc(a, b, flow)
        return cv2.calcOpticalFlowFarneback(a, b, flow, 0.5, 3, 15, 3, 5, 1.2, 0)


def _sample(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    return cv2.remap(image, grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


@dataclass(frozen=True)
class FlowMaskResult:
    alpha: np.ndarray                 # stabilised matte, float32 0..1, input shape
    confidence: float                 # mean forward-backward confidence over the frame (0 after a reset)
    reset_reason: str | None = None   # SCENE_CUT / FLASH / FIRST_FRAME / SHAPE_CHANGED / FRAME_GAP / None


@dataclass
class _State:
    gray: np.ndarray
    alpha: np.ndarray
    frame_index: int | None


class FlowMaskStabilizer:
    def __init__(self, *, c_max: float = 0.6, sigma_px: float = 0.75, flow_side: int = 256, max_tracks: int = 16,
                 max_gap: int = 2, estimator: FlowEstimator | None = None):
        if not 0 <= c_max < 1 or sigma_px <= 0 or flow_side < 32 or max_tracks < 1 or max_gap < 1:
            raise ValueError("Invalid flow mask stabiliser configuration")
        self.c_max, self.sigma_px, self.flow_side, self.max_tracks, self.max_gap = (
            float(c_max), float(sigma_px), int(flow_side), int(max_tracks), int(max_gap))
        self.estimator = estimator or FlowEstimator()
        self._states: OrderedDict[object, _State] = OrderedDict()
        self.last_confidence = 0.0

    @property
    def backend(self) -> str:
        return self.estimator.backend

    def reset(self, key: object | None = None) -> None:
        """Drop the history of one matte (``key``) or all of them (shot change / flash)."""
        if key is None:
            self._states.clear()
        else:
            self._states.pop(key, None)
        self.last_confidence = 0.0

    def retain(self, keys) -> None:
        keep = set(keys)
        for key in tuple(self._states):
            if key not in keep:
                del self._states[key]

    def _gray(self, guide: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        image = np.asarray(guide)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.shape != shape or image.dtype != np.uint8:
            raise DomainError("FLOW_INPUT_INVALID", "Guide must be a uint8 image matching the matte", 422)
        return image

    def update(self, key: object, alpha: np.ndarray, guide: np.ndarray, *, frame_index: int | None = None,
               scene_cut: bool = False, flash: bool = False) -> FlowMaskResult:
        current = np.asarray(alpha, np.float32)
        if current.ndim != 2 or not current.size or not np.isfinite(current).all():
            raise DomainError("FLOW_MASK_INVALID", "Matte must be a finite 2-D alpha map", 422)
        current = np.clip(current, 0.0, 1.0)
        gray = self._gray(guide, current.shape)
        if scene_cut or flash:
            self.reset()
        state = self._states.get(key)
        reason = "SCENE_CUT" if scene_cut else "FLASH" if flash else None
        if state is not None:
            gap = None if frame_index is None or state.frame_index is None else frame_index - state.frame_index
            if state.alpha.shape != current.shape:
                state, reason = None, "SHAPE_CHANGED"
            elif gap is not None and not 0 < gap <= self.max_gap:
                state, reason = None, "FRAME_GAP"
        if state is None:
            self._store(key, gray, current, frame_index)
            self.last_confidence = 0.0
            return FlowMaskResult(current.copy(), 0.0, reason or "FIRST_FRAME")

        height, width = current.shape
        scale = min(1.0, self.flow_side / max(height, width))
        size = (max(8, round(width * scale)), max(8, round(height * scale)))
        small_now = cv2.resize(gray, size, interpolation=cv2.INTER_AREA) if scale < 1 else gray
        small_before = cv2.resize(state.gray, size, interpolation=cv2.INTER_AREA) if scale < 1 else state.gray
        backward = self.estimator.calc(small_now, small_before)         # current -> previous
        forward = self.estimator.calc(small_before, small_now)          # previous -> current
        residual = backward + _sample(forward, backward)
        error = np.linalg.norm(residual, axis=-1)
        confidence = np.exp(-np.square(error / self.sigma_px)).astype(np.float32)
        if scale < 1:
            backward = cv2.resize(backward, (width, height), interpolation=cv2.INTER_LINEAR) / scale
            confidence = cv2.resize(confidence, (width, height), interpolation=cv2.INTER_LINEAR)
        weight = (self.c_max * confidence)
        warped = _sample(state.alpha, backward.astype(np.float32))
        fused = np.clip((1.0 - weight) * current + weight * warped, 0.0, 1.0).astype(np.float32)
        self._store(key, gray, fused, frame_index)
        self.last_confidence = float(confidence.mean())
        return FlowMaskResult(fused.copy(), self.last_confidence, None)

    def _store(self, key: object, gray: np.ndarray, alpha: np.ndarray, frame_index: int | None) -> None:
        self._states[key] = _State(gray.copy(), alpha.copy(), frame_index)
        self._states.move_to_end(key)
        while len(self._states) > self.max_tracks:
            self._states.popitem(last=False)
