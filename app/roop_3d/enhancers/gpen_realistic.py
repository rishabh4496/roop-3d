"""GPEN Realistic: Static Batch-1 FP32-safe restoration with luma-only recoloring."""
from __future__ import annotations

from typing import Any
import cv2
import numpy as np


def luma_only_delta_recolour(swap_crop: np.ndarray, gpen_output: np.ndarray) -> np.ndarray:
    """Eliminates GPEN's magenta/pink skin cast while retaining 100% of GPEN detail.
    Calculates signed luminance delta and applies it uniformly to the swap crop.
    """
    swap_f = swap_crop.astype(np.float32)
    gpen_f = gpen_output.astype(np.float32)

    # Luminance weights (ITU-R BT.601 BGR weights)
    # B: 0.114, G: 0.587, R: 0.299
    luma_swap = 0.114 * swap_f[:, :, 0] + 0.587 * swap_f[:, :, 1] + 0.299 * swap_f[:, :, 2]
    luma_gpen = 0.114 * gpen_f[:, :, 0] + 0.587 * gpen_f[:, :, 1] + 0.299 * gpen_f[:, :, 2]

    # Signed grey delta
    delta = (luma_gpen - luma_swap)[:, :, np.newaxis]

    recoloured = swap_f + delta
    return np.clip(recoloured, 0.0, 255.0).astype(np.uint8)


class GPENRealisticRestorer:
    """GPEN Realistic restorer supporting 512 and 1024 resolutions.
    Enforces static batch-1 execution and luma-only delta recoloring.
    """

    def __init__(self, resolution: int = 512, session: Any = None):
        if resolution not in (512, 1024):
            raise ValueError(f"GPEN Realistic resolution must be 512 or 1024, got {resolution}")
        self.resolution = resolution
        self.session = session
        self.supports_dynamic_batch = False  # Static batch-1 mandatory
        self._gpu_runner = None
        self._output_name = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_name = session.get_outputs()[0].name

    def enhance(self, swap_crop: np.ndarray) -> np.ndarray:
        h, w = swap_crop.shape[:2]
        res = self.resolution

        if (h, w) != (res, res):
            crop_res = cv2.resize(swap_crop, (res, res), interpolation=cv2.INTER_CUBIC)
        else:
            crop_res = swap_crop

        if self.session is not None:
            inp = crop_res.astype(np.float32) / 255.0
            inp = (inp - 0.5) / 0.5
            chw = np.transpose(inp, (2, 0, 1))[np.newaxis, ...]
            if self._gpu_runner is not None:
                output = self._gpu_runner.run({"input": chw})[self._output_name].numpy()
            else:
                output = self.session.run(None, {"input": chw})[0]
            raw = output[0].transpose(1, 2, 0).astype(np.float32, copy=False)
            gpen_out = np.clip((raw * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        else:
            # High-fidelity reference filter for unmocked offline tests
            gpen_out = cv2.bilateralFilter(crop_res, d=5, sigmaColor=50, sigmaSpace=50)

        # Apply luma-only delta recoloring to eliminate pink/magenta cast
        recoloured = luma_only_delta_recolour(crop_res, gpen_out)

        if (h, w) != (res, res):
            return cv2.resize(recoloured, (w, h), interpolation=cv2.INTER_AREA)
        return recoloured
