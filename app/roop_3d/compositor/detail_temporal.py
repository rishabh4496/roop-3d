"""Causal, confidence-gated stabilization of restoration-only high-frequency detail.

The compositor/restorer boundary uses NumPy images. When NVIDIA OF 2.0 is available,
the small flow guides, flow vectors, temporal residual state, warping and fusion stay on
CUDA. Only the final correction crosses back to the host for OpenCV composition. DIS
is the portable CPU fallback. State is bounded per track and never carries detail
across a scene cut, frame gap, shape change, or invalid/occluded pixel.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.compositor.mask_flow import FlowEstimator


@dataclass
class _DetailState:
    guide: object
    residual: object
    frame_index: int | None

    @property
    def nbytes(self) -> int:
        return int(self.guide.nbytes + self.residual.nbytes)


class TemporalDetailStabilizer:
    """Flow-warp enhancement residuals, using CUDA OF when available and CPU DIS otherwise.

    With NVIDIA OF 2.0, internal state, flow consistency, residual warping and fusion
    remain on CUDA. The returned face crop is NumPy because that is the compositor API.
    ``temporal_alpha`` follows the conventional current-frame coefficient in
    ``alpha * current + (1-alpha) * warped_previous``. Confidence, visibility,
    and scene-cut gates still prevent stale or occluded history from leaking.
    """

    def __init__(self, *, temporal_alpha: float = .25, flow_side: int = 96, max_side: int = 512,
                 max_tracks: int = 12, max_cache_bytes: int = 32 * 1024 * 1024,
                 cut_correlation: float = .35, min_flow_confidence: float = .12,
                 use_cuda_flow: bool = True):
        if (not 0 <= temporal_alpha <= .95 or flow_side < 16 or max_side < 16 or max_tracks < 1
                or max_cache_bytes < 1024 or not -1 <= cut_correlation <= 1
                or not 0 <= min_flow_confidence <= 1 or type(use_cuda_flow) is not bool):
            raise ValueError("Invalid temporal detail stabilizer configuration")
        self.temporal_alpha = float(temporal_alpha)
        self.flow_side, self.max_side = int(flow_side), int(max_side)
        self.max_tracks, self.max_cache_bytes = int(max_tracks), int(max_cache_bytes)
        self.cut_correlation, self.min_flow_confidence = float(cut_correlation), float(min_flow_confidence)
        self._tracks: OrderedDict[int, _DetailState] = OrderedDict()
        self._last_flows: OrderedDict[int, object] = OrderedDict()
        self._flow = FlowEstimator()
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
            "temporal_alpha": self.temporal_alpha,
            "previous_weight": 1.0 - self.temporal_alpha,
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

    @staticmethod
    def _cuda_available() -> bool:
        """Avoid treating lightweight flow-test doubles as real CUDA tensor runtimes."""
        try:
            import torch
            return hasattr(torch, "Tensor") and torch.cuda.is_available()
        except (ImportError, AttributeError, RuntimeError):
            return False

    def _cuda_inputs(self, guide: np.ndarray, residual: np.ndarray):
        import torch
        device = f"cuda:{self._gpu_flow.device}"
        guide_device = torch.from_numpy(np.ascontiguousarray(guide)).to(device).contiguous()
        residual_device = torch.from_numpy(np.ascontiguousarray(residual.transpose(2, 0, 1)[None])).to(
            device=device, dtype=torch.float16).contiguous()
        return guide_device, residual_device

    def _fuse_cuda(self, state: _DetailState, current_guide, raw: np.ndarray, valid_work: np.ndarray,
                   valid_full: np.ndarray, work_size: tuple[int, int], output_size: tuple[int, int]):
        """GPU flow consistency, backward residual warp, and visibility-gated fusion."""
        import torch
        import torch.nn.functional as F

        device = current_guide.device
        forward, backward = self._gpu_flow.pair(state.guide, current_guide)
        backward = backward.float()
        forward = forward.float()
        side = self.flow_side
        grid_y, grid_x = torch.meshgrid(
            torch.arange(side, device=device, dtype=torch.float32),
            torch.arange(side, device=device, dtype=torch.float32), indexing="ij")
        map_x, map_y = grid_x[None] + backward[..., 0], grid_y[None] + backward[..., 1]
        sample_grid = torch.stack((map_x * (2.0 / max(side - 1, 1)) - 1.0,
                                   map_y * (2.0 / max(side - 1, 1)) - 1.0), -1)
        forward_at = F.grid_sample(forward.permute(0, 3, 1, 2), sample_grid,
                                   mode="bilinear", padding_mode="zeros", align_corners=True).permute(0, 2, 3, 1)
        fb_error = torch.linalg.vector_norm(backward + forward_at, dim=-1)
        magnitude = torch.linalg.vector_norm(backward, dim=-1)
        in_bounds = ((map_x >= 0) & (map_x <= side - 1) & (map_y >= 0) & (map_y <= side - 1))
        fb_limit = .75 + .05 * magnitude
        fb_confidence = torch.exp(-torch.square(fb_error / fb_limit.clamp_min(.25))) * in_bounds

        guide = current_guide.float()[None, None]
        warped_guide = F.grid_sample(state.guide.float()[None, None], sample_grid,
                                     mode="bilinear", padding_mode="zeros", align_corners=True)
        photo_confidence = torch.exp(-torch.square((warped_guide - guide).abs() / 24.0))
        confidence_small = fb_confidence[:, None] * photo_confidence
        confidence_mean = float(confidence_small.mean().item())
        af, bf = warped_guide.flatten(), guide.flatten()
        af, bf = af - af.mean(), bf - bf.mean()
        denom = torch.linalg.vector_norm(af) * torch.linalg.vector_norm(bf)
        denom_value = float(denom.item())
        correlation = (float(((af * bf).sum() / denom).item()) if denom_value > 1e-6
                       else (1.0 if torch.equal(warped_guide, guide) else 0.0))
        if correlation < self.cut_correlation:
            return None, confidence_mean, 0.0, None, "APPEARANCE_CUT", None
        if confidence_mean < self.min_flow_confidence:
            return None, confidence_mean, 0.0, None, "FLOW_UNRELIABLE", None

        work_w, work_h = work_size
        resized_flow = F.interpolate(backward.permute(0, 3, 1, 2), size=(work_h, work_w),
                                     mode="bilinear", align_corners=False)
        resized_flow[:, 0] *= work_w / side
        resized_flow[:, 1] *= work_h / side
        confidence = F.interpolate(confidence_small, size=(work_h, work_w), mode="bilinear", align_corners=False)
        y, x = torch.meshgrid(torch.arange(work_h, device=device, dtype=torch.float32),
                              torch.arange(work_w, device=device, dtype=torch.float32), indexing="ij")
        src_x = x[None] + resized_flow[:, 0]
        src_y = y[None] + resized_flow[:, 1]
        residual_h, residual_w = state.residual.shape[-2:]
        residual_grid = torch.stack((src_x * (2.0 / max(residual_w - 1, 1)) - 1.0,
                                     src_y * (2.0 / max(residual_h - 1, 1)) - 1.0), -1)
        previous = F.grid_sample(state.residual.float(), residual_grid, mode="bilinear",
                                 padding_mode="zeros", align_corners=True)
        current = torch.from_numpy(np.ascontiguousarray(raw.transpose(2, 0, 1)[None])).to(
            device=device, dtype=torch.float32)
        valid_device = torch.from_numpy(np.ascontiguousarray(valid_work[None, None])).to(device=device)
        weight = (1.0 - self.temporal_alpha) * confidence * valid_device
        fused = current * (1.0 - weight) + previous * weight
        correction = fused - current
        output_w, output_h = output_size
        if (work_w, work_h) != (output_w, output_h):
            correction = F.interpolate(correction, size=(output_h, output_w),
                                       mode="bicubic", align_corners=False)
        correction = correction * torch.from_numpy(np.ascontiguousarray(valid_full[None, None])).to(device=device)
        correction_host = correction[0].permute(1, 2, 0).contiguous().float().cpu().numpy()
        return fused.detach().to(dtype=torch.float16), confidence_mean, float((weight > .01).float().mean().item()), \
            float(weight.mean().item()), correction_host, None, -backward[0].detach()

    @property
    def cache_bytes(self) -> int:
        return sum(state.nbytes for state in self._tracks.values())

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def clear(self) -> None:
        self._tracks.clear()
        self._last_flows.clear()

    def flow_for(self, track_id: int):
        """Most recent reliable current-frame motion field; CUDA flow stays on CUDA until finishing consumes it."""
        return self._last_flows.get(track_id)

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
                self._last_flows.pop(track_id, None)

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

        ``temporal_alpha`` is the current-residual coefficient, default 0.25; the
        complementary 0.75 weight is assigned to flow-warped history where confidence
        and the current visibility mask allow it. Scene cuts reset state before fusion.
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
        self._try_create_gpu_flow()
        cuda_state = self._gpu_flow is not None and self._cuda_available()
        if cuda_state:
            try:
                guide_state, residual_state = self._cuda_inputs(guide_small, raw)
            except (ImportError, AttributeError, OSError, RuntimeError, ValueError, cv2.error) as error:
                self._gpu_flow_error = f"{type(error).__name__}: {error}"
                try:
                    self._gpu_flow.close()
                except Exception:
                    pass
                self._gpu_flow = None
                cuda_state = False
        if not cuda_state:
            guide_state, residual_state = guide_small, raw.astype(np.float16)
        current = _DetailState(guide_state, residual_state, frame_index)
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
        motion_flow = None
        confidence_mean, influenced_fraction, gpu_history_weight = 0.0, 0.0, 0.0
        backend, backend_fallback = "none_no_history", self._gpu_flow_error
        if state is not None:
            if cuda_state and hasattr(state.guide, "is_cuda") and state.guide.is_cuda:
                try:
                    fused_device, confidence_mean, influenced_fraction, gpu_history_weight, correction, reason, motion_flow = self._fuse_cuda(
                        state, guide_state, raw, valid_work, valid, (work_w, work_h), (w, h))
                    backend = "opencv_nvidia_of_2_0_cuda_fusion"
                    backend_fallback = None
                    if reason is None:
                        result = np.clip(output.astype(np.float32) + correction, 0, 255).round().astype(np.uint8)
                        current = _DetailState(guide_state, fused_device, frame_index)
                        self._tracks.move_to_end(track_id)
                    else:
                        self._tracks.pop(track_id, None)
                        state = None
                except (ImportError, AttributeError, OSError, RuntimeError, ValueError, cv2.error) as error:
                    self._gpu_flow_error = f"{type(error).__name__}: {error}"
                    motion_flow = None
                    try:
                        self._gpu_flow.close()
                    except Exception:
                        pass
                    self._gpu_flow = None
                    cuda_state = False
                    state = _DetailState(state.guide.detach().cpu().numpy(),
                                         state.residual[0].permute(1, 2, 0).detach().cpu().numpy(),
                                         state.frame_index)
                    current = _DetailState(guide_small, raw.astype(np.float16), frame_index)
                    backend = "opencv_dis_cpu"
                    backend_fallback = self._gpu_flow_error
                    reason = None
            if state is not None and backend != "opencv_nvidia_of_2_0_cuda_fusion":
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
                    motion_flow = -backward
                    flow_x = cv2.resize(backward[..., 0], (work_w, work_h), interpolation=cv2.INTER_LINEAR) * (work_w / side)
                    flow_y = cv2.resize(backward[..., 1], (work_w, work_h), interpolation=cv2.INTER_LINEAR) * (work_h / side)
                    confidence = cv2.resize(confidence_small, (work_w, work_h), interpolation=cv2.INTER_LINEAR)
                    xx, yy = np.meshgrid(np.arange(work_w, dtype=np.float32), np.arange(work_h, dtype=np.float32))
                    previous = cv2.remap(state.residual.astype(np.float32), xx + flow_x, yy + flow_y,
                                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    weight = ((1.0 - self.temporal_alpha) * confidence * valid_work).astype(np.float32)
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
        if motion_flow is None:
            self._last_flows.pop(track_id, None)
        else:
            self._last_flows[track_id] = motion_flow
            self._last_flows.move_to_end(track_id)
            while len(self._last_flows) > self.max_tracks:
                self._last_flows.popitem(last=False)
        report = {
            "backend": backend,
            "temporal_alpha": self.temporal_alpha,
            "history_weight": (gpu_history_weight if backend == "opencv_nvidia_of_2_0_cuda_fusion"
                               else (float(weight.mean()) if state is not None and reason is None and 'weight' in locals()
                                     else 0.0)),
            "flow_confidence": confidence_mean,
            "influenced_fraction": influenced_fraction,
            "reset_reason": reason,
        }
        self._last_backend = backend
        self._last_gpu_fallback_reason = backend_fallback
        if backend_fallback:
            report["gpu_fallback_reason"] = backend_fallback
        return result, report
