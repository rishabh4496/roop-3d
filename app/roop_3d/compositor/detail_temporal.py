"""Causal, confidence-gated stabilization of restoration-only high-frequency detail.

The compositor exchanges NumPy/OpenCV images with the restorer. NVIDIA OF 2.0 may
accelerate flow estimation when the custom CUDA bindings and runtime are available,
but flow vectors are downloaded and residual warping/fusion remain on CPU. DIS is the
portable fallback. State is bounded per track and never carries detail across a scene
cut, frame gap, shape change, or invalid/occluded pixel.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class _DetailState:
    guide: np.ndarray
    residual: np.ndarray
    frame_index: int | None

    @property
    def nbytes(self) -> int:
        return int(self.guide.nbytes + self.residual.nbytes)


class TemporalDetailStabilizer:
    """Flow-warp enhancement residuals, using CUDA OF when available and CPU DIS otherwise.

    The OpenCV hardware path accelerates flow estimation only. Residual warping and
    fusion still run on CPU because the compositor/restorer exchange NumPy images.
    """

    def __init__(self, *, temporal_alpha: float = .25, flow_side: int = 96, max_side: int = 512,
                 max_tracks: int = 12, max_cache_bytes: int = 32 * 1024 * 1024,
                 cut_correlation: float = .35, min_flow_confidence: float = .12,
                 use_cuda_flow: bool = True):
        if (not 0 <= temporal_alpha <= .5 or flow_side < 16 or max_side < flow_side or max_tracks < 1
                or max_cache_bytes < 1024 or not -1 <= cut_correlation <= 1
                or not 0 <= min_flow_confidence <= 1 or type(use_cuda_flow) is not bool):
            raise ValueError("Invalid temporal detail stabilizer configuration")
        self.temporal_alpha = float(temporal_alpha)
        self.flow_side, self.max_side = int(flow_side), int(max_side)
        self.max_tracks, self.max_cache_bytes = int(max_tracks), int(max_cache_bytes)
        self.cut_correlation, self.min_flow_confidence = float(cut_correlation), float(min_flow_confidence)
        self._tracks: OrderedDict[int, _DetailState] = OrderedDict()
        self._flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self._gpu_flow = None
        self._use_cuda_flow = use_cuda_flow
        self._gpu_flow_checked = not bool(use_cuda_flow)
        self._gpu_flow_error = None
        self._last_backend = "none_no_history"
        self._last_gpu_fallback_reason = None

    def capability(self) -> dict:
        return {
            "backend": self._last_backend,
            "backend_policy": "nvidia_of_2_0_if_available_else_cpu_dis" if self._use_cuda_flow else "cpu_dis_forced",
            "gpu_fallback_reason": self._last_gpu_fallback_reason,
            "tracks": self.track_count,
            "cache_bytes": self.cache_bytes,
            "cache_limit_bytes": self.max_cache_bytes,
            "previous_weight": self.temporal_alpha,
        }

    def _try_create_gpu_flow(self) -> None:
        """Lazily select the existing NVIDIA OF 2.0 bridge without making CUDA mandatory."""
        if self._gpu_flow_checked:
            return
        self._gpu_flow_checked = True
        try:
            import torch
            if not torch.cuda.is_available():
                self._gpu_flow_error = "CUDA_RUNTIME_UNAVAILABLE"
                return
            from roop_3d.optical_flow_tracker import NvidiaHardwareFlow
            self._gpu_flow = NvidiaHardwareFlow((self.flow_side, self.flow_side), device=0)
        except (ImportError, AttributeError, OSError, RuntimeError, cv2.error) as error:
            self._gpu_flow_error = f"{type(error).__name__}: {error}"

    def _flow_pair(self, previous: np.ndarray, current: np.ndarray):
        """Return previous->current and current->previous flow on host for CPU fusion."""
        self._try_create_gpu_flow()
        if self._gpu_flow is not None:
            try:
                import torch
                device = f"cuda:{self._gpu_flow.device}"
                previous_device = torch.from_numpy(np.ascontiguousarray(previous)).to(device)
                current_device = torch.from_numpy(np.ascontiguousarray(current)).to(device)
                forward_device, backward_device = self._gpu_flow.pair(previous_device, current_device)
                forward = forward_device[0].detach().float().cpu().numpy()
                backward = backward_device[0].detach().float().cpu().numpy()
                expected = (self.flow_side, self.flow_side, 2)
                if forward.shape != expected or backward.shape != expected:
                    raise ValueError("NVIDIA OF returned flow with an unexpected shape")
                return forward, backward, "opencv_nvidia_of_2_0_cpu_warp", None
            except (ImportError, AttributeError, OSError, RuntimeError, ValueError, cv2.error) as error:
                self._gpu_flow_error = f"{type(error).__name__}: {error}"
                try:
                    self._gpu_flow.close()
                except Exception:
                    pass
                self._gpu_flow = None

        forward = self._flow.calc(previous, current, None)
        backward = self._flow.calc(current, previous, None)
        return forward, backward, "opencv_dis_cpu", self._gpu_flow_error

    @property
    def cache_bytes(self) -> int:
        return sum(state.nbytes for state in self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()

    def release(self) -> None:
        self.clear()
        if self._gpu_flow is not None:
            self._gpu_flow.close()
            self._gpu_flow = None
        self._gpu_flow_checked = not self._use_cuda_flow
        self._gpu_flow_error = None
        self._last_backend = "none_no_history"
        self._last_gpu_fallback_reason = None

    def retain(self, track_ids) -> None:
        keep = set(track_ids)
        for track_id in tuple(self._tracks):
            if track_id not in keep:
                self._tracks.pop(track_id, None)

    @staticmethod
    def _gray(image: np.ndarray, side: int) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _correlation(a: np.ndarray, b: np.ndarray) -> float:
        af, bf = a.astype(np.float32), b.astype(np.float32)
        af -= float(af.mean())
        bf -= float(bf.mean())
        denom = float(np.linalg.norm(af) * np.linalg.norm(bf))
        return float((af * bf).sum() / denom) if denom > 1e-6 else (1.0 if np.array_equal(a, b) else 0.0)

    @staticmethod
    def _highpass(image: np.ndarray) -> np.ndarray:
        value = image.astype(np.float32)
        return value - cv2.GaussianBlur(value, (0, 0), 1.15, borderType=cv2.BORDER_REFLECT101)

    def _evict(self, keep_id: int) -> None:
        while len(self._tracks) > self.max_tracks or self.cache_bytes > self.max_cache_bytes:
            oldest = next(iter(self._tracks))
            self._tracks.pop(oldest)
            if oldest == keep_id:
                break

    def update(self, track_id: int, enhanced: np.ndarray, base: np.ndarray, guide: np.ndarray, *,
               valid_mask: np.ndarray | None = None, frame_index: int | None = None,
               scene_cut: bool = False) -> tuple[np.ndarray, dict]:
        """Stabilize the enhancer-only high-pass residual in a face-aligned crop.

        ``temporal_alpha`` is the maximum previous-frame weight, default 0.25. The current
        mask always gates history, so a new occluder or invalid face pixel is never delayed.
        This is causal (no t+1 lookahead), and therefore resets rather than fabricating a
        bidirectional interpolation when an isolated mask anomaly is observed.
        """
        if type(track_id) is not int or track_id < 0:
            raise ValueError("track_id must be a nonnegative integer")
        output = np.asarray(enhanced)
        source = np.asarray(base)
        target = np.asarray(guide)
        if (output.dtype != np.uint8 or output.ndim != 3 or output.shape[2] != 3
                or source.dtype != np.uint8 or source.shape != output.shape
                or target.dtype != np.uint8 or target.ndim != 3 or target.shape[2] != 3):
            raise ValueError("Enhanced, base, and guide must be valid uint8 BGR images")
        h, w = output.shape[:2]
        if valid_mask is None:
            valid = np.ones((h, w), np.float32)
        else:
            valid = np.asarray(valid_mask, np.float32)
            if valid.ndim != 2 or not np.isfinite(valid).all() or np.any((valid < 0) | (valid > 1)):
                raise ValueError("valid_mask must be a finite 2-D alpha map in 0..1")
            if valid.shape != (h, w):
                valid = cv2.resize(valid, (w, h), interpolation=cv2.INTER_LINEAR)

        scale = min(1.0, self.max_side / max(h, w))
        work_w, work_h = max(1, round(w * scale)), max(1, round(h * scale))
        if (work_w, work_h) == (w, h):
            enhanced_work, base_work = output, source
            valid_work = valid
        else:
            size = (work_w, work_h)
            enhanced_work = cv2.resize(output, size, interpolation=cv2.INTER_AREA)
            base_work = cv2.resize(source, size, interpolation=cv2.INTER_AREA)
            valid_work = cv2.resize(valid, size, interpolation=cv2.INTER_AREA)
        raw = self._highpass(enhanced_work) - self._highpass(base_work)
        guide_small = self._gray(target, self.flow_side)
        current = _DetailState(guide_small, raw.astype(np.float16), frame_index)
        state = self._tracks.get(track_id)
        reason = None
        if scene_cut:
            self._tracks.clear()
            state, reason = None, "SCENE_CUT"
        elif state is not None:
            gap = None if frame_index is None or state.frame_index is None else frame_index - state.frame_index
            if gap is not None and (gap <= 0 or gap > 2):
                self._tracks.pop(track_id, None)
                state, reason = None, "FRAME_GAP"
            elif state.residual.shape != current.residual.shape:
                self._tracks.pop(track_id, None)
                state, reason = None, "SHAPE_CHANGED"

        result = output.copy()
        confidence_mean, influenced_fraction = 0.0, 0.0
        backend, backend_fallback = "none_no_history", self._gpu_flow_error
        if state is not None:
            forward, backward, backend, backend_fallback = self._flow_pair(state.guide, guide_small)
            side = self.flow_side
            grid_x, grid_y = np.meshgrid(np.arange(side, dtype=np.float32), np.arange(side, dtype=np.float32))
            map_x, map_y = grid_x + backward[..., 0], grid_y + backward[..., 1]
            forward_at = cv2.remap(forward, map_x, map_y, cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            fb_error = np.linalg.norm(backward + forward_at, axis=2)
            magnitude = np.linalg.norm(backward, axis=2)
            in_bounds = ((map_x >= 0) & (map_x <= side - 1) & (map_y >= 0) & (map_y <= side - 1))
            fb_limit = 0.75 + .05 * magnitude
            fb_confidence = np.exp(-np.square(fb_error / np.maximum(fb_limit, .25))) * in_bounds
            warped_guide = cv2.remap(state.guide, map_x, map_y, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            photo_error = np.abs(warped_guide.astype(np.float32) - guide_small.astype(np.float32))
            photo_confidence = np.exp(-np.square(photo_error / 24.0))
            confidence_small = (fb_confidence * photo_confidence).astype(np.float32)
            confidence_mean = float(confidence_small.mean())
            registered_correlation = self._correlation(warped_guide, guide_small)
            if registered_correlation < self.cut_correlation:
                self._tracks.pop(track_id, None)
                state, reason = None, "APPEARANCE_CUT"
            elif confidence_mean < self.min_flow_confidence:
                self._tracks.pop(track_id, None)
                state, reason = None, "FLOW_UNRELIABLE"
            else:
                flow_x = cv2.resize(backward[..., 0], (work_w, work_h), interpolation=cv2.INTER_LINEAR) * (work_w / side)
                flow_y = cv2.resize(backward[..., 1], (work_w, work_h), interpolation=cv2.INTER_LINEAR) * (work_h / side)
                confidence = cv2.resize(confidence_small, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
                xx, yy = np.meshgrid(np.arange(work_w, dtype=np.float32), np.arange(work_h, dtype=np.float32))
                previous = cv2.remap(state.residual.astype(np.float32), xx + flow_x, yy + flow_y,
                                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                weight = (self.temporal_alpha * confidence * valid_work).astype(np.float32)
                old_residual = raw.astype(np.float32)
                fused = old_residual * (1.0 - weight[..., None]) + previous * weight[..., None]
                correction = fused - old_residual
                if (work_w, work_h) != (w, h):
                    correction = cv2.resize(correction, (w, h), interpolation=cv2.INTER_CUBIC)
                # Resampling the whole corrected image would blur even invalid/occluded pixels.
                correction *= valid[..., None]
                result = np.clip(output.astype(np.float32) + correction, 0, 255).round().astype(np.uint8)
                influenced_fraction = float((weight > .01).mean())
                current = _DetailState(guide_small, fused.astype(np.float16), frame_index)
                self._tracks.move_to_end(track_id)

        self._tracks[track_id] = current
        self._evict(track_id)
        report = {
            "backend": backend,
            "history_weight": self.temporal_alpha if reason is None else 0.0,
            "flow_confidence": confidence_mean,
            "influenced_fraction": influenced_fraction,
            "reset_reason": reason,
        }
        self._last_backend = backend
        self._last_gpu_fallback_reason = backend_fallback
        if backend_fallback:
            report["gpu_fallback_reason"] = backend_fallback
        return result, report
