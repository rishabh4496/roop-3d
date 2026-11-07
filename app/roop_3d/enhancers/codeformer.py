"""Dynamic CodeFormer & RestoreFormer++ with Dynamic Fidelity Weight Binding."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


class DynamicCodeFormerRestorer:
    """CodeFormer with dynamic 2nd input `w` (fidelity_weight: 0.0 to 1.0)
    bound per-frame without session rebuild.
    """

    def __init__(self, session: Any = None):
        self.session = session
        self.resolution = 512
        self._gpu_runner = None
        self._output_names = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_names = tuple(node.name for node in session.get_outputs())

    def enhance(self, swap_crop: np.ndarray, fidelity_weight: float = 0.5) -> np.ndarray:
        h, w = swap_crop.shape[:2]
        w_val = max(0.0, min(1.0, float(fidelity_weight)))

        if (h, w) != (512, 512):
            crop_512 = cv2.resize(swap_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            crop_512 = swap_crop

        if self.session is not None:
            inp = crop_512.astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5
            chw = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            fw_arr = np.array([w_val], dtype=np.float32)
            if self._gpu_runner is not None:
                bound = self._gpu_runner.run({"x": chw, "w": fw_arr})
                value = bound[self._output_names[0]].numpy()
            else:
                value = self.session.run(None, {"x": chw, "w": fw_arr})[0]
            raw = value[0].transpose(1, 2, 0).astype(np.float32, copy=False)
            restored = np.clip((raw * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            # Weighted blend between input and smoothed high-frequency detail
            enhanced = cv2.detailEnhance(crop_512, sigma_s=5, sigma_r=0.2)
            restored = cv2.addWeighted(crop_512, 1.0 - w_val, enhanced, w_val, 0)

        if (h, w) != (512, 512):
            return cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)
        return restored


class RestoreFormerPlusPlusRestorer:
    """RestoreFormer++ contextual facial detail restoration engine."""

    def __init__(self, session: Any = None):
        self.session = session
        self.resolution = 512
        self._gpu_runner = None
        self._output_names = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_names = tuple(node.name for node in session.get_outputs())

    def enhance(self, swap_crop: np.ndarray) -> np.ndarray:
        h, w = swap_crop.shape[:2]

        if (h, w) != (512, 512):
            crop_512 = cv2.resize(swap_crop, (512, 512), interpolation=cv2.INTER_CUBIC)
        else:
            crop_512 = swap_crop

        if self.session is not None:
            inp = crop_512.astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5
            chw = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            if self._gpu_runner is not None:
                value = self._gpu_runner.run({"input": chw})[self._output_names[0]].numpy()
            else:
                value = self.session.run(None, {"input": chw})[0]
            raw = value[0].transpose(1, 2, 0).astype(np.float32, copy=False)
            restored = np.clip((raw * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            restored = cv2.edgePreservingFilter(crop_512, flags=1, sigma_s=50, sigma_r=0.4)

        if (h, w) != (512, 512):
            return cv2.resize(restored, (w, h), interpolation=cv2.INTER_AREA)
        return restored
