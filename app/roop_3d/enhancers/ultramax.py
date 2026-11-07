"""UltraMax Restorer: zero-copy CUDA IOBinding, explicit-only small-face bypass & 3-level Laplacian fusion."""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

log = logging.getLogger(__name__)


def laplacian_pyramid_fusion(
    swap_crop: np.ndarray,
    restorer_output: np.ndarray,
    levels: int = 3,
    high_freq_boost: float = 1.0,
    detail_levels: int = 1,
) -> np.ndarray:
    """Blend restorer detail with swap crop chrominance and low/mid frequencies via a Laplacian pyramid.

    The ``detail_levels`` finest bands come 100% from the restorer. When the restorer works at S x the swap's resolution
    (``image`` is the enlarged swap), every band finer than the swap's own Nyquist limit carries no swap information, so
    the caller passes ``detail_levels = log2(S)`` and none of the restorer's detail is halved by a 50/50 mix there.
    """
    swap_f = swap_crop.astype(np.float32)
    rest_f = restorer_output.astype(np.float32)

    # Build Gaussian pyramids
    g_swap = [swap_f]
    g_rest = [rest_f]
    for _ in range(levels - 1):
        g_swap.append(cv2.pyrDown(g_swap[-1]))
        g_rest.append(cv2.pyrDown(g_rest[-1]))

    # Build Laplacian pyramids
    l_swap = []
    l_rest = []
    for i in range(levels - 1):
        size = (g_swap[i].shape[1], g_swap[i].shape[0])
        l_swap.append(g_swap[i] - cv2.pyrUp(g_swap[i + 1], dstsize=size))
        l_rest.append(g_rest[i] - cv2.pyrUp(g_rest[i + 1], dstsize=size))

    # Blend: Base from swap crop, highest frequencies from restorer
    blended = g_swap[-1]
    for i in range(levels - 2, -1, -1):
        size = (g_swap[i].shape[1], g_swap[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        if i < detail_levels:
            # Bands finer than the swap can represent: the restorer's detail, undiluted
            blended += l_rest[i] * high_freq_boost
        else:
            # Mid levels: blend 50% swap, 50% restorer
            blended += 0.5 * l_swap[i] + 0.5 * l_rest[i]

    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def to_model_input(bgr_u8: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB NCHW float32 in [-1, 1] (CodeFormer / GPEN / GFPGAN / UltraMax training contract)."""
    rgb = bgr_u8[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])


def from_model_output(value: np.ndarray) -> np.ndarray:
    """RGB NCHW [-1, 1] model output -> BGR uint8 HWC."""
    rgb = np.clip((np.asarray(value, np.float32)[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255)
    return np.ascontiguousarray(rgb[:, :, ::-1]).round().astype(np.uint8)


def require_session(session: Any, name: str, allow_reference_filter: bool) -> None:
    """A restorer without a model must never quietly hand back a fake 'enhancement'."""
    if session is None and not allow_reference_filter:
        log.critical("%s has no ONNX/TensorRT session; refusing to return an unenhanced or fake-enhanced crop", name)
        raise DomainError("ENHANCER_SESSION_MISSING", f"{name} requires a loaded model session", 503)


class UltraMaxRestorer:
    """Decoupled zero-copy restorer running codeformer.fp16 weights with Laplacian pyramid fusion.

    The small-face bypass is OFF by default. It only exists as an explicit user setting
    (``native_bypass=True``) and is evaluated against the face's *destination* size in the target frame
    (``native_resolution``), never against the size of the swapper's working crop.
    """

    def __init__(
        self,
        native_bypass: bool = False,
        min_native_pixels: int = 320,
        session: Any = None,
        allow_reference_filter: bool = False,
    ):
        self.native_bypass = native_bypass
        self.min_native_pixels = min_native_pixels
        self.session = session
        self.allow_reference_filter = allow_reference_filter
        self._gpu_runner = None
        self._output_name = None
        if session is not None and hasattr(session, "io_binding"):
            from roop_3d.core.ort_session import GpuIoBinding
            self._gpu_runner = GpuIoBinding(session)
            self._output_name = session.get_outputs()[0].name

    def enhance(
        self,
        swap_crop: np.ndarray,
        native_resolution: int | None = None,
        fidelity_weight: float = 0.5,
        enhancer_blend: float = 1.0,
    ) -> np.ndarray:
        """Run UltraMax restoration. Returns the enhanced crop at ITS NATIVE 512 px resolution.

        The caller owns the warp into the frame (see ``roop_3d.hires_handoff``); shrinking the result back to the
        swap-crop size here would throw the detail away.
        """
        if not np.isfinite(fidelity_weight) or not 0 <= fidelity_weight <= 1:
            raise ValueError("UltraMax fidelity must be within 0..1")
        if not np.isfinite(enhancer_blend) or not 0 <= enhancer_blend <= 1:
            raise ValueError("enhancer_blend must be within 0..1")
        h, w = swap_crop.shape[:2]

        # 1. Explicit-only small-face bypass, judged on the destination face size when the caller supplies it.
        if self.native_bypass and native_resolution is not None and native_resolution <= self.min_native_pixels:
            return swap_crop

        require_session(self.session, "UltraMax", self.allow_reference_filter)

        # 2. Bicubic up to the enhancer's native 512x512 (never down-sample the swap crop first).
        crop_512 = swap_crop if (h, w) == (512, 512) else cv2.resize(swap_crop, (512, 512), interpolation=cv2.INTER_CUBIC)

        # 3. Model inference: RGB, [-1, 1], dynamic fidelity input `w`.
        if self.session is not None:
            chw = to_model_input(crop_512)
            fw = np.array([fidelity_weight], dtype=np.float32)
            try:
                if self._gpu_runner is not None:
                    outputs = self._gpu_runner.run({"x": chw, "w": fw})
                    raw = outputs[self._output_name].numpy()
                else:
                    raw = self.session.run(None, {"x": chw, "w": fw})[0]
            except Exception as error:
                log.critical("UltraMax inference failed (input %s, provider mismatch or shape error): %s",
                             chw.shape, error, exc_info=True)
                raise DomainError("ENHANCER_INFERENCE_FAILED", f"UltraMax inference failed: {error}", 503) from error
            if raw.shape != (1, 3, 512, 512) or not np.isfinite(raw).all():
                log.critical("UltraMax returned an invalid tensor %s", getattr(raw, "shape", None))
                raise DomainError("ENHANCER_OUTPUT_INVALID", "UltraMax returned an unsupported output tensor", 503)
            restored_512 = from_model_output(raw)
        else:
            # Explicit offline reference filter (tests only; ``allow_reference_filter=True``).
            restored_512 = cv2.detailEnhance(crop_512, sigma_s=10, sigma_r=0.15)

        # 4. 3-level Laplacian pyramid frequency fusion, then the dynamic blend weight.
        fused_512 = laplacian_pyramid_fusion(crop_512, restored_512, levels=3)
        if enhancer_blend < 1.0:
            fused_512 = np.clip(crop_512.astype(np.float32) * (1 - enhancer_blend)
                                + fused_512.astype(np.float32) * enhancer_blend, 0, 255).round().astype(np.uint8)
        return fused_512
