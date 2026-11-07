"""Dynamic CodeFormer & RestoreFormer++ with dynamic fidelity-weight binding."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.enhancers.ultramax import from_model_output, require_session, to_model_input

log = logging.getLogger(__name__)


def _blend(base: np.ndarray, restored: np.ndarray, weight: float) -> np.ndarray:
    if weight >= 1.0:
        return restored
    return np.clip(base.astype(np.float32) * (1 - weight) + restored.astype(np.float32) * weight, 0, 255).round().astype(np.uint8)


class DynamicCodeFormerRestorer:
    """CodeFormer with dynamic 2nd input `w` (fidelity_weight: 0.0 to 1.0)
    bound per-frame without session rebuild. RGB [-1, 1] input; output stays at the native 512 px.
    """

    def __init__(self, session: Any = None, allow_reference_filter: bool = False):
        self.session = session
        self.allow_reference_filter = allow_reference_filter
        self.resolution = 512
        self._gpu_runner = None
        self._output_names = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_names = tuple(node.name for node in session.get_outputs())

    def enhance(self, swap_crop: np.ndarray, fidelity_weight: float = 0.5, enhancer_blend: float = 1.0) -> np.ndarray:
        if not np.isfinite(enhancer_blend) or not 0 <= enhancer_blend <= 1:
            raise ValueError("enhancer_blend must be within 0..1")
        w_val = max(0.0, min(1.0, float(fidelity_weight)))
        require_session(self.session, "CodeFormer", self.allow_reference_filter)
        h, w = swap_crop.shape[:2]
        crop_512 = swap_crop if (h, w) == (512, 512) else cv2.resize(swap_crop, (512, 512), interpolation=cv2.INTER_CUBIC)

        if self.session is not None:
            chw = to_model_input(crop_512)
            fw_arr = np.array([w_val], dtype=np.float32)
            try:
                if self._gpu_runner is not None:
                    value = self._gpu_runner.run({"x": chw, "w": fw_arr})[self._output_names[0]].numpy()
                else:
                    value = self.session.run(None, {"x": chw, "w": fw_arr})[0]
            except Exception as error:
                log.critical("CodeFormer inference failed (input %s): %s", chw.shape, error, exc_info=True)
                raise DomainError("ENHANCER_INFERENCE_FAILED", f"CodeFormer inference failed: {error}", 503) from error
            if value.shape != (1, 3, 512, 512) or not np.isfinite(value).all():
                log.critical("CodeFormer returned an invalid tensor %s", getattr(value, "shape", None))
                raise DomainError("ENHANCER_OUTPUT_INVALID", "CodeFormer returned an unsupported output tensor", 503)
            restored = from_model_output(value)
        else:
            # Explicit offline reference filter (tests only; ``allow_reference_filter=True``).
            enhanced = cv2.detailEnhance(crop_512, sigma_s=5, sigma_r=0.2)
            restored = cv2.addWeighted(crop_512, 1.0 - w_val, enhanced, w_val, 0)
        return _blend(crop_512, restored, enhancer_blend)


class RestoreFormerPlusPlusRestorer:
    """RestoreFormer++ contextual facial detail restoration engine (RGB [-1, 1], native 512 px output)."""

    def __init__(self, session: Any = None, allow_reference_filter: bool = False):
        self.session = session
        self.allow_reference_filter = allow_reference_filter
        self.resolution = 512
        self._gpu_runner = None
        self._output_names = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_names = tuple(node.name for node in session.get_outputs())

    def enhance(self, swap_crop: np.ndarray, enhancer_blend: float = 1.0) -> np.ndarray:
        if not np.isfinite(enhancer_blend) or not 0 <= enhancer_blend <= 1:
            raise ValueError("enhancer_blend must be within 0..1")
        require_session(self.session, "RestoreFormer++", self.allow_reference_filter)
        h, w = swap_crop.shape[:2]
        crop_512 = swap_crop if (h, w) == (512, 512) else cv2.resize(swap_crop, (512, 512), interpolation=cv2.INTER_CUBIC)

        if self.session is not None:
            chw = to_model_input(crop_512)
            try:
                if self._gpu_runner is not None:
                    value = self._gpu_runner.run({"input": chw})[self._output_names[0]].numpy()
                else:
                    value = self.session.run(None, {"input": chw})[0]
            except Exception as error:
                log.critical("RestoreFormer++ inference failed (input %s): %s", chw.shape, error, exc_info=True)
                raise DomainError("ENHANCER_INFERENCE_FAILED", f"RestoreFormer++ inference failed: {error}", 503) from error
            if value.shape != (1, 3, 512, 512) or not np.isfinite(value).all():
                log.critical("RestoreFormer++ returned an invalid tensor %s", getattr(value, "shape", None))
                raise DomainError("ENHANCER_OUTPUT_INVALID", "RestoreFormer++ returned an unsupported output tensor", 503)
            restored = from_model_output(value)
        else:
            # Explicit offline reference filter (tests only; ``allow_reference_filter=True``).
            restored = cv2.edgePreservingFilter(crop_512, flags=1, sigma_s=50, sigma_r=0.4)
        return _blend(crop_512, restored, enhancer_blend)
