"""Optical-flow-aware temporal stabilisation of per-face occluder mattes.

Design (mirrors the semantics of ``optical_flow_tracker.OpticalFlowTracker`` without its torch
dependency, because the compositor path is CPU/OpenCV):

* the previous matte is warped into the current frame with dense DIS optical flow computed
  *current -> previous* (backward sampling);
* history only influences pixels the current frame is **uncertain** about (probability near 0.5),
  weighted by photometric flow reliability. Confident new detections pass through immediately, so a
  hand entering the face is never delayed and never smeared (no "painted over the hand" lag);
* a scene cut (explicit flag, frame gap, or an appearance jump of the crop) discards all history;
* when the current analysis is unavailable the last matte is held for at most ``max_hold`` frames
  with decay and is reported as held, never silently kept.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.compositor.mask_flow import FlowEstimator
from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class OcclusionState:
    occluder: np.ndarray | None
    motion: tuple[float, float]          # target crop motion, crop pixels per frame (x, y); (0, 0) if unknown
    held: bool = False
    reset_reason: str | None = None
    history_influence: float = 0.0       # fraction of pixels whose value was changed by history


@dataclass
class _Track:
    gray: np.ndarray                     # flow-resolution grayscale of the previous crop
    occluder: np.ndarray | None
    frame_index: int | None
    hold: int = 0


class OcclusionTracker:
    def __init__(self, *, history: float = 0.7, max_hold: int = 4, hold_decay: float = 0.8, flow_side: int = 96,
                 cut_reliability: float = 0.18, cut_correlation: float = 0.6, max_tracks: int = 16):
        if not 0 <= history < 1 or not 0 < hold_decay <= 1 or max_hold < 0 or flow_side < 16 or max_tracks < 1:
            raise ValueError("Invalid occlusion tracker configuration")
        self.history, self.max_hold, self.hold_decay = history, max_hold, hold_decay
        self.flow_side, self.cut_reliability, self.cut_correlation, self.max_tracks = (
            flow_side, cut_reliability, cut_correlation, max_tracks)
        self._tracks: OrderedDict[int, _Track] = OrderedDict()
        self._flow = FlowEstimator()

    # -- lifecycle --------------------------------------------------------------------------------------
    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        for track_id in tuple(self._tracks):
            if track_id not in keep:
                self._tracks.pop(track_id, None)

    # -- helpers ----------------------------------------------------------------------------------------
    def _gray(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (self.flow_side, self.flow_side), interpolation=cv2.INTER_AREA)

    # -- update -----------------------------------------------------------------------------------------
    def update(self, track_id: int, occluder: np.ndarray | None, target_bgr: np.ndarray, *,
               frame_index: int | None = None, scene_cut: bool = False, release: float = 0.0) -> OcclusionState:
        """``release`` (0..1): fraction of the flow-warped previous matte that may linger where the current
        analysis dropped it. 0 = no lingering; onset (a new occluder) is never delayed by it."""
        if not 0.0 <= release < 1.0:
            raise ValueError("release must be within 0..1")
        if type(track_id) is not int or track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        image = np.asarray(target_bgr)
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise DomainError("OCCLUSION_TRACKER_INPUT_INVALID", "Target crop must be uint8 BGR", 422)
        current = None
        if occluder is not None:
            current = np.clip(np.asarray(occluder, dtype=np.float32), 0.0, 1.0)
            if current.shape != image.shape[:2] or not np.isfinite(current).all():
                raise DomainError("OCCLUSION_TRACKER_INPUT_INVALID", "Occluder must match the target crop", 422)
        gray = self._gray(image)
        state = self._tracks.get(track_id)
        reset_reason = None
        if scene_cut:
            # Only this track's history is dropped here; the compositor clears every track once per cut.
            self._tracks.pop(track_id, None)
            state, reset_reason = None, "SCENE_CUT"
        elif state is not None:
            gap = None if frame_index is None or state.frame_index is None else frame_index - state.frame_index
            if gap is not None and (gap <= 0 or gap > 2):
                state, reset_reason = None, "FRAME_GAP"
            elif state.occluder is not None and state.occluder.shape != image.shape[:2]:
                state, reset_reason = None, "SHAPE_CHANGED"
        if state is None:
            self._store(track_id, gray, current, frame_index)
            return OcclusionState(current, (0.0, 0.0), False, reset_reason, 0.0)

        # Dense flow current -> previous: pixel p of the current crop corresponds to p + flow(p) in the previous.
        flow = self._flow.calc(gray, state.gray, None)
        side, (height, width) = self.flow_side, image.shape[:2]
        scale_x, scale_y = width / side, height / side
        warped_gray = self._warp_small(state.gray, flow)
        photometric = np.abs(warped_gray.astype(np.float32) - gray.astype(np.float32)) / 255.0
        reliability_small = np.exp(-np.square(photometric / 0.10)).astype(np.float32)
        reliability_small = cv2.GaussianBlur(reliability_small, (5, 5), 1.0)
        # A shot change cannot be registered by any flow. Either most pixels stay photometrically inconsistent, or
        # (dense flow can "explain" unrelated smooth content) the flow-REGISTERED crops remain uncorrelated.
        # Measured: the same content shifted 4 px registers at correlation 0.96; unrelated content at 0.22-0.40.
        if float(reliability_small.mean()) < self.cut_reliability or self._correlation(warped_gray, gray) < self.cut_correlation:
            self._store(track_id, gray, current, frame_index)
            return OcclusionState(current, (0.0, 0.0), False, "APPEARANCE_CUT", 0.0)
        flow_full = cv2.resize(flow, (width, height), interpolation=cv2.INTER_LINEAR)
        flow_full[..., 0] *= scale_x
        flow_full[..., 1] *= scale_y
        reliability = cv2.resize(reliability_small, (width, height), interpolation=cv2.INTER_LINEAR)
        good = reliability_small > 0.6
        motion = (0.0, 0.0)
        if int(good.sum()) >= 16:
            motion = (float(-np.median(flow[..., 0][good]) * scale_x), float(-np.median(flow[..., 1][good]) * scale_y))

        previous = state.occluder
        warped = None
        if previous is not None:
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            warped = cv2.remap(previous, grid_x + flow_full[..., 0], grid_y + flow_full[..., 1], cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

        held, influence = False, 0.0
        if current is None:
            if warped is not None and state.hold < self.max_hold:
                fused = np.clip(warped * (self.hold_decay ** (state.hold + 1)) * np.maximum(reliability, 0.25), 0, 1)
                held = True
                self._store(track_id, gray, fused.astype(np.float32), frame_index, hold=state.hold + 1)
                return OcclusionState(fused.astype(np.float32), motion, True, None, 1.0)
            self._store(track_id, gray, None, frame_index)
            return OcclusionState(None, motion, False, "HOLD_EXPIRED" if previous is not None else None, 0.0)
        if warped is None:
            fused = current
        else:
            uncertainty = 1.0 - np.abs(2.0 * current - 1.0)
            weight = self.history * reliability * uncertainty
            fused = np.clip(current + (warped - current) * weight, 0.0, 1.0)
            if release > 0.0:
                fused = np.maximum(fused, warped * reliability * release)
            influence = float((np.abs(fused - current) > 0.02).mean())
        fused = fused.astype(np.float32)
        self._store(track_id, gray, fused, frame_index)
        return OcclusionState(fused, motion, held, None, influence)

    @staticmethod
    def _correlation(previous: np.ndarray, current: np.ndarray) -> float:
        a = previous.astype(np.float32) - float(previous.mean())
        b = current.astype(np.float32) - float(current.mean())
        return float((a * b).sum() / (np.sqrt(float((a * a).sum()) * float((b * b).sum())) + 1e-6))

    @staticmethod
    def _warp_small(gray: np.ndarray, flow: np.ndarray) -> np.ndarray:
        height, width = gray.shape
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        return cv2.remap(gray, grid_x + flow[..., 0], grid_y + flow[..., 1], cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)

    def _store(self, track_id: int, gray: np.ndarray, occluder: np.ndarray | None, frame_index: int | None,
               hold: int = 0) -> None:
        self._tracks[track_id] = _Track(gray, occluder, frame_index, hold)
        self._tracks.move_to_end(track_id)
        while len(self._tracks) > self.max_tracks:
            self._tracks.popitem(last=False)
