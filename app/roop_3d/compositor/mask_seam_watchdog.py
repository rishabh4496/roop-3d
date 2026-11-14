"""Temporal mask-seam watchdog: detects and repairs single-frame matte dropouts (crop/render agnostic)."""
from __future__ import annotations

import threading

import cv2
import numpy as np

from roop_3d.compositor.mask_flow import FlowEstimator
from roop_3d.core.errors import DomainError


class MaskSeamWatchdog:
    """Never compare occlusion coverage to a geometric-face-mask baseline."""
    def __init__(self, maximum=256):
        self.maximum = maximum
        self._baselines = {}
        self._edge_levels = {}
        self._temporal_masks = {}
        self._lock = threading.Lock()

    def observe_temporal(self, face_id: int, mask: np.ndarray, *, frame_index: int | None,
                         scene_cut: bool = False, context: dict | None = None) -> dict:
        report, _ = self.observe_and_recover(face_id, mask, frame_index=frame_index,
                                             scene_cut=scene_cut, context=context)
        return report

    @staticmethod
    def _gray_small(value, size=(128, 128)):
        guide = np.asarray(value)
        if guide.dtype != np.uint8 or guide.ndim not in (2, 3):
            raise ValueError("Mask recovery guides must be uint8 grayscale or BGR images")
        if guide.ndim == 3:
            if guide.shape[2] != 3:
                raise ValueError("Mask recovery guides must be grayscale or BGR")
            guide = cv2.cvtColor(guide, cv2.COLOR_BGR2GRAY)
        return cv2.resize(guide, size, interpolation=cv2.INTER_AREA)

    @staticmethod
    def _registered_correlation(first: np.ndarray, second: np.ndarray) -> float:
        a, b = first.astype(np.float32), second.astype(np.float32)
        a -= float(a.mean())
        b -= float(b.mean())
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float((a * b).sum() / denominator) if denominator > 1e-6 else (
            1.0 if np.array_equal(first, second) else 0.0)

    @staticmethod
    def _recover_triplet(mask: np.ndarray, context: dict):
        required = {"previous_mask", "next_mask", "previous_guide", "current_guide", "next_guide"}
        if not isinstance(context, dict) or not required.issubset(context):
            return None, None
        h, w = mask.shape
        previous_mask = np.asarray(context["previous_mask"], np.float32)
        next_mask = np.asarray(context["next_mask"], np.float32)
        if (previous_mask.ndim != 2 or next_mask.ndim != 2
                or min(previous_mask.shape, default=0) < 1 or min(next_mask.shape, default=0) < 1
                or not np.isfinite(previous_mask).all() or not np.isfinite(next_mask).all()
                or np.any((previous_mask < 0) | (previous_mask > 1))
                or np.any((next_mask < 0) | (next_mask > 1))):
            return None, None
        previous_mask = cv2.resize(previous_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        next_mask = cv2.resize(next_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        size = (128, 128)
        try:
            prev_guide, current_guide, next_guide = [
                MaskSeamWatchdog._gray_small(context[name], size)
                for name in ("previous_guide", "current_guide", "next_guide")]
        except (TypeError, ValueError, cv2.error):
            return None, None

        # For a current-frame output pixel, backward flow samples the previous
        # mask, while forward flow samples the next mask. Check each direction
        # against its inverse before allowing either temporal observation.
        try:
            flow = FlowEstimator()
            current_to_previous = flow.calc(current_guide, prev_guide, None)
            previous_to_current = flow.calc(prev_guide, current_guide, None)
            current_to_next = flow.calc(current_guide, next_guide, None)
            next_to_current = flow.calc(next_guide, current_guide, None)
        except (DomainError, cv2.error):
            return None, None
        side = size[0]
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        prev_x, prev_y = xx + current_to_previous[..., 0], yy + current_to_previous[..., 1]
        next_x, next_y = xx + current_to_next[..., 0], yy + current_to_next[..., 1]
        registered_previous = cv2.remap(prev_guide, prev_x, prev_y, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        registered_next = cv2.remap(next_guide, next_x, next_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        previous_correlation = MaskSeamWatchdog._registered_correlation(registered_previous, current_guide)
        next_correlation = MaskSeamWatchdog._registered_correlation(registered_next, current_guide)
        if min(previous_correlation, next_correlation) < .35:
            return None, {"appearance_correlation_previous": previous_correlation,
                          "appearance_correlation_next": next_correlation}
        prev_cycle = cv2.remap(previous_to_current, prev_x, prev_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        next_cycle = cv2.remap(next_to_current, next_x, next_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        prev_error = np.linalg.norm(current_to_previous + prev_cycle, axis=2)
        next_error = np.linalg.norm(current_to_next + next_cycle, axis=2)
        prev_mag = np.linalg.norm(current_to_previous, axis=2)
        next_mag = np.linalg.norm(current_to_next, axis=2)
        prev_valid = ((prev_x >= 0) & (prev_x <= side - 1) & (prev_y >= 0) & (prev_y <= side - 1))
        next_valid = ((next_x >= 0) & (next_x <= side - 1) & (next_y >= 0) & (next_y <= side - 1))
        prev_conf = np.exp(-np.square(prev_error / (.75 + .05 * prev_mag))) * prev_valid
        next_conf = np.exp(-np.square(next_error / (.75 + .05 * next_mag))) * next_valid

        previous_small = cv2.resize(previous_mask, size, interpolation=cv2.INTER_AREA)
        next_small = cv2.resize(next_mask, size, interpolation=cv2.INTER_AREA)
        warped_previous = cv2.remap(previous_small, prev_x, prev_y, cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        warped_next = cv2.remap(next_small, next_x, next_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        confidence = np.minimum(prev_conf, next_conf)
        supported = confidence > .15
        prev_binary, next_binary = (warped_previous >= .5) & supported, (warped_next >= .5) & supported
        union = int(np.logical_or(prev_binary, next_binary).sum())
        agreement = float(np.logical_and(prev_binary, next_binary).sum() / union) if union else 1.0
        confidence_mean = float(confidence.mean())
        area_delta = abs(float((warped_previous >= .5).mean()) - float((warped_next >= .5).mean()))
        if confidence_mean < .20 or agreement < .45 or area_delta > .20:
            return None, {"flow_confidence": confidence_mean, "context_iou": agreement,
                          "context_area_delta": area_delta}
        weights = prev_conf + next_conf
        repaired_small = np.divide(warped_previous * prev_conf + warped_next * next_conf,
                                   weights, out=np.zeros_like(weights), where=weights > 1e-4)
        repaired = cv2.resize(repaired_small, (w, h), interpolation=cv2.INTER_CUBIC).clip(0, 1)
        occluder = context.get("occluder_mask")
        if occluder is not None:
            occluder = np.asarray(occluder, np.float32)
            if occluder.shape != (h, w) or not np.isfinite(occluder).all() or np.any((occluder < 0) | (occluder > 1)):
                return None, None
            repaired *= (1.0 - occluder)
        return repaired.astype(np.float32), {"flow_confidence": confidence_mean,
                                              "context_iou": agreement,
                                              "context_area_delta": area_delta}

    def observe_and_recover(self, face_id: int, mask: np.ndarray, *, frame_index: int | None,
                            scene_cut: bool = False, context: dict | None = None) -> tuple[dict, np.ndarray]:
        """Measure combined-mask continuity independently for this worker thread and face.

        Recovery is attempted only for a single-frame anomaly and only when explicit
        previous/current/next guides and masks are supplied. With no buffered lookahead,
        this remains a report-only watchdog.
        """
        value = np.asarray(mask, np.float32)
        if (value.ndim != 2 or min(value.shape, default=0) < 1 or not np.isfinite(value).all()
                or np.any((value < 0) | (value > 1))):
            raise ValueError("Invalid temporal seam watchdog mask")
        small = cv2.resize(value, (128, 128), interpolation=cv2.INTER_AREA) if max(value.shape) > 128 else value
        binary = small >= .5
        gx = cv2.Sobel(small, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge_level = float(np.percentile(edge[edge > .01], 95)) if np.any(edge > .01) else 0.0
        key = (threading.get_ident(), face_id)
        with self._lock:
            previous = None if scene_cut else self._temporal_masks.get(key)
            if previous is not None and previous[0].shape == binary.shape:
                old_binary, old_area, old_edge, old_frame = previous
                consecutive = frame_index is None or old_frame is None or 0 < frame_index - old_frame <= 2
                if consecutive:
                    intersection = float(np.logical_and(old_binary, binary).sum())
                    union = float(np.logical_or(old_binary, binary).sum())
                    iou = intersection / union if union else 1.0
                    area_jump = abs(float(binary.mean()) - old_area)
                    boundary_jump = abs(edge_level - old_edge)
                    anomaly = bool(iou < .45 and area_jump > .12 and boundary_jump > .08)
                    reset_reason = None
                else:
                    iou, area_jump, boundary_jump, anomaly = 1.0, 0.0, 0.0, False
                    reset_reason = "FRAME_GAP"
            else:
                iou, area_jump, boundary_jump, anomaly = 1.0, 0.0, 0.0, False
                reset_reason = ("SCENE_CUT" if scene_cut else
                                "MASK_SHAPE_CHANGED" if previous is not None else "INITIAL")
        repaired = None
        recovery_info = None
        recovery = ("not_attempted_no_lookahead" if context is None else "rejected_context") if anomaly else None
        if anomaly and not scene_cut and context is not None:
            repaired, recovery_info = self._recover_triplet(value, context)
            if repaired is not None:
                recovery = "bidirectional_optical_flow"
        observed = repaired if repaired is not None else value
        observed_small = cv2.resize(observed, (128, 128), interpolation=cv2.INTER_AREA) if max(observed.shape) > 128 else observed
        observed_binary = observed_small >= .5
        observed_gx = cv2.Sobel(observed_small, cv2.CV_32F, 1, 0, ksize=3)
        observed_gy = cv2.Sobel(observed_small, cv2.CV_32F, 0, 1, ksize=3)
        observed_edge = np.sqrt(observed_gx * observed_gx + observed_gy * observed_gy)
        observed_level = float(np.percentile(observed_edge[observed_edge > .01], 95)) if np.any(observed_edge > .01) else 0.0
        with self._lock:
            if len(self._temporal_masks) >= self.maximum and key not in self._temporal_masks:
                self._temporal_masks.pop(next(iter(self._temporal_masks)))
            self._temporal_masks[key] = (observed_binary.copy(), float(observed_binary.mean()), observed_level, frame_index)
        report = {"iou": float(iou), "area_jump": float(area_jump), "boundary_gradient": edge_level,
                  "boundary_gradient_jump": float(boundary_jump), "anomaly": anomaly,
                  "reset_reason": reset_reason, "recovery": recovery}
        if recovery_info is not None:
            report["recovery_metrics"] = recovery_info
        return report, observed.astype(np.float32, copy=False)

    def observe(self, face_id: int, mask: np.ndarray, *, kind: str):
        if kind not in ("geometric", "occluder", "restoration", "composite"):
            raise ValueError("Mask baseline type must be explicit")
        mask = np.asarray(mask, np.float32)
        if mask.ndim != 2 or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
            raise ValueError("Invalid seam watchdog mask")
        key = (threading.get_ident(), face_id, kind)
        with self._lock:
            old = self._baselines.get(key)
            drift = float(np.abs(mask - old).mean()) if old is not None and old.shape == mask.shape else 0.0
            if key not in self._baselines and len(self._baselines) >= self.maximum:
                oldest = next(iter(self._baselines))
                self._baselines.pop(oldest)
                self._edge_levels.pop(oldest, None)
            self._baselines[key] = mask.copy()
        return drift

    def feather(self, face_id: int, mask: np.ndarray, *, kind: str = "composite",
                minimum_sigma: float = .35, maximum_sigma: float = 2.5) -> np.ndarray:
        """Reduce sharp boundary spikes using a baseline from the same mask class only."""
        if kind not in ("geometric", "occluder", "restoration", "composite"):
            raise ValueError("Mask baseline type must be explicit")
        value = np.asarray(mask, np.float32)
        if value.ndim != 2 or not np.isfinite(value).all() or np.any((value < 0) | (value > 1)):
            raise ValueError("Invalid seam watchdog mask")
        drift = self.observe(face_id, value, kind=kind)
        gx = cv2.Sobel(value, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(value, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        boundary = edge > .01
        level = float(np.percentile(edge[boundary], 95)) if np.any(boundary) else 0.0
        key = (threading.get_ident(), face_id, kind)
        with self._lock:
            previous = self._edge_levels.get(key)
            self._edge_levels[key] = level
        spike = max(0.0, level - previous * 1.15) if previous is not None else 0.0
        sigma = float(np.clip(minimum_sigma + drift * 8.0 + spike * 1.5,
                              minimum_sigma, maximum_sigma))
        if sigma <= .5:
            return np.clip(value, 0, 1).astype(np.float32)
        softened = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)
        return np.clip(softened, 0, 1).astype(np.float32)

    def clear(self):
        with self._lock:
            self._baselines.clear()
            self._edge_levels.clear()
            self._temporal_masks.clear()
