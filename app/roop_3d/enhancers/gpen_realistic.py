"""GPEN Realistic: static batch-1 FP32-safe restoration with luma-only recolouring."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.enhancers.ultramax import from_model_output, require_session, to_model_input

log = logging.getLogger(__name__)


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

    Static batch-1 execution, RGB [-1, 1] input contract, luma-only delta recolouring. The output keeps the model's
    native resolution (512 / 1024); the caller warps it straight into the frame.
    """

    def __init__(self, resolution: int = 512, session: Any = None, allow_reference_filter: bool = False):
        if resolution not in (512, 1024):
            raise ValueError(f"GPEN Realistic resolution must be 512 or 1024, got {resolution}")
        self.resolution = resolution
        self.session = session
        self.allow_reference_filter = allow_reference_filter
        self.supports_dynamic_batch = False  # Static batch-1 mandatory
        self._gpu_runner = None
        self._output_name = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_name = session.get_outputs()[0].name

    def enhance(self, swap_crop: np.ndarray, enhancer_blend: float = 1.0) -> np.ndarray:
        if not np.isfinite(enhancer_blend) or not 0 <= enhancer_blend <= 1:
            raise ValueError("enhancer_blend must be within 0..1")
        require_session(self.session, "GPEN Realistic", self.allow_reference_filter)
        h, w = swap_crop.shape[:2]
        res = self.resolution
        crop_res = swap_crop if (h, w) == (res, res) else cv2.resize(swap_crop, (res, res), interpolation=cv2.INTER_CUBIC)

        if self.session is not None:
            chw = to_model_input(crop_res)
            try:
                if self._gpu_runner is not None:
                    output = self._gpu_runner.run({"input": chw})[self._output_name].numpy()
                else:
                    output = self.session.run(None, {"input": chw})[0]
            except Exception as error:
                log.critical("GPEN inference failed (input %s): %s", chw.shape, error, exc_info=True)
                raise DomainError("ENHANCER_INFERENCE_FAILED", f"GPEN inference failed: {error}", 503) from error
            if output.shape != (1, 3, res, res) or not np.isfinite(output).all():
                log.critical("GPEN returned an invalid tensor %s", getattr(output, "shape", None))
                raise DomainError("ENHANCER_OUTPUT_INVALID", "GPEN returned an unsupported output tensor", 503)
            gpen_out = from_model_output(output)
        else:
            # Explicit offline reference filter (tests only; ``allow_reference_filter=True``).
            gpen_out = cv2.bilateralFilter(crop_res, d=5, sigmaColor=50, sigmaSpace=50)

        # Luma-only delta recolouring removes the pink/magenta cast; blend weight applied last.
        recoloured = luma_only_delta_recolour(crop_res, gpen_out)
        if enhancer_blend < 1.0:
            recoloured = np.clip(crop_res.astype(np.float32) * (1 - enhancer_blend)
                                 + recoloured.astype(np.float32) * enhancer_blend, 0, 255).round().astype(np.uint8)
        return recoloured
