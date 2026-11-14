"""Offline UV-texture detailer: neural detail added to the CANONICAL UV atlas, never to a 2D frame or face crop.

Why this exists
---------------
The previous pipeline ran CodeFormer / GFPGAN / GPEN / UltraMax on aligned 2D face crops AFTER a 2D swap. That is
screen-space GAN restoration and it is removed. What is legitimate is to enrich an UNWRAPPED, pose-independent
albedo (and optionally tangent-space normal) atlas ONCE, offline, before it is rasterized with a perspective
K[R|t] camera. Detail added there is geometry-locked: it rotates with the head and cannot swim against it.

Contract
--------
* Input is a square UV map whose side is 1024 or 2048 (``ALLOWED_UV_SIZES``). Anything else (a frame, an
  aligned crop, a 512-px portrait) is rejected with ``UVDetailerInputError`` so this class cannot be misused
  as a 2D restorer.
* The atlas is processed in overlapping 512-px tiles with a feathered window; a validity mask keeps texels
  outside the UV islands untouched.
* Albedo: neural detail is fused with the base atlas through a Laplacian pyramid so low frequencies (identity
  colour, lighting-free albedo) always come from the measured atlas.
* Normal maps: a neural network is NEVER run on them (it would invent geometry); only a measured high-pass
  of the detailed albedo luminance is added to the tangent-space slope, then the vector is re-normalized.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

import cv2
import numpy as np

from roop_3d.core.errors import DomainError

log = logging.getLogger(__name__)

ALLOWED_UV_SIZES = (1024, 2048)
TILE = 512


class UVDetailerInputError(DomainError):
    def __init__(self, message: str):
        super().__init__("UV_DETAILER_INPUT_INVALID", message, 422)


def laplacian_pyramid_fusion(base: np.ndarray, detailed: np.ndarray, levels: int = 3, high_freq_boost: float = 1.0,
                             detail_levels: int = 1) -> np.ndarray:
    """Low/mid bands from ``base``, the ``detail_levels`` finest bands from ``detailed``; generic image operator."""
    base_f, det_f = base.astype(np.float32), detailed.astype(np.float32)
    g_base, g_det = [base_f], [det_f]
    for _ in range(levels - 1):
        g_base.append(cv2.pyrDown(g_base[-1]))
        g_det.append(cv2.pyrDown(g_det[-1]))
    l_base, l_det = [], []
    for i in range(levels - 1):
        size = (g_base[i].shape[1], g_base[i].shape[0])
        l_base.append(g_base[i] - cv2.pyrUp(g_base[i + 1], dstsize=size))
        l_det.append(g_det[i] - cv2.pyrUp(g_det[i + 1], dstsize=size))
    blended = g_base[-1]
    for i in range(levels - 2, -1, -1):
        size = (g_base[i].shape[1], g_base[i].shape[0])
        blended = cv2.pyrUp(blended, dstsize=size)
        blended += l_det[i] * high_freq_boost if i < detail_levels else 0.5 * l_base[i] + 0.5 * l_det[i]
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)


def to_model_input(bgr_u8: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> RGB NCHW float32 in [-1, 1]."""
    rgb = bgr_u8[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(((rgb - 0.5) / 0.5).transpose(2, 0, 1)[None])


def from_model_output(value: np.ndarray) -> np.ndarray:
    """RGB NCHW [-1, 1] model output -> BGR uint8 HWC."""
    rgb = np.clip((np.asarray(value, np.float32)[0].transpose(1, 2, 0) * 0.5 + 0.5) * 255.0, 0, 255)
    return np.ascontiguousarray(rgb[:, :, ::-1]).round().astype(np.uint8)


def _validate_uv(image: np.ndarray, name: str) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise UVDetailerInputError(f"{name} must be an HxWx3 uint8 UV map")
    h, w = image.shape[:2]
    if h != w or h not in ALLOWED_UV_SIZES:
        raise UVDetailerInputError(
            f"{name} is {w}x{h}; the UV detailer only accepts square {ALLOWED_UV_SIZES} canonical UV atlases. "
            "It is not a frame or face-crop restorer.")


class UVTextureDetailer:
    """Tile-wise neural detail for a canonical UV albedo atlas (offline texture builder)."""

    def __init__(self, session: Any = None, *, runner: Callable[[np.ndarray], np.ndarray] | None = None,
                 fidelity_weight: float = 0.7):
        if not np.isfinite(fidelity_weight) or not 0.0 <= fidelity_weight <= 1.0:
            raise ValueError("fidelity_weight must be within 0..1")
        if session is None and runner is None:
            raise DomainError("ENHANCER_SESSION_MISSING",
                              "UVTextureDetailer requires a loaded model session or an explicit runner", 503)
        self.session, self._runner, self.fidelity_weight = session, runner, float(fidelity_weight)
        self._input_names = tuple(node.name for node in session.get_inputs()) if session is not None else ()

    def _infer_tile(self, tile_bgr: np.ndarray) -> np.ndarray:
        chw = to_model_input(tile_bgr)
        if self._runner is not None:
            raw = np.asarray(self._runner(chw))
        else:
            feeds = {self._input_names[0]: chw}
            if len(self._input_names) > 1:
                feeds[self._input_names[1]] = np.array([self.fidelity_weight], dtype=np.float32)
            try:
                raw = self.session.run(None, feeds)[0]
            except Exception as error:
                raise DomainError("ENHANCER_INFERENCE_FAILED", f"UV detail inference failed: {error}", 503) from error
        if raw.shape != (1, 3, TILE, TILE) or not np.isfinite(raw).all():
            raise DomainError("ENHANCER_OUTPUT_INVALID", "UV detail model returned an unsupported tensor", 503)
        return from_model_output(raw)

    def detail_albedo(self, albedo_bgr: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
        """Return the detailed atlas at the same size. ``valid_mask`` (H, W, 0..1) marks texels inside UV islands."""
        _validate_uv(albedo_bgr, "albedo")
        size = albedo_bgr.shape[0]
        if valid_mask is not None and valid_mask.shape[:2] != (size, size):
            raise UVDetailerInputError("valid_mask must match the albedo atlas size")
        stride = TILE // 2
        hann = np.hanning(TILE + 2)[1:-1].astype(np.float32)
        window = np.outer(hann, hann)[..., None]
        accum = np.zeros((size, size, 3), np.float32)
        weight = np.zeros((size, size, 1), np.float32)
        for y in range(0, size - TILE + 1, stride):
            for x in range(0, size - TILE + 1, stride):
                if valid_mask is not None and float(valid_mask[y:y + TILE, x:x + TILE].max()) <= 0.0:
                    continue
                tile = albedo_bgr[y:y + TILE, x:x + TILE]
                accum[y:y + TILE, x:x + TILE] += self._infer_tile(tile).astype(np.float32) * window
                weight[y:y + TILE, x:x + TILE] += window
        detailed = np.where(weight > 1e-6, accum / np.maximum(weight, 1e-6), albedo_bgr.astype(np.float32))
        fused = laplacian_pyramid_fusion(albedo_bgr, np.clip(detailed, 0, 255).astype(np.uint8), levels=3)
        if valid_mask is not None:
            m = np.clip(valid_mask.astype(np.float32), 0.0, 1.0)
            m = m[..., None] if m.ndim == 2 else m
            fused = (fused.astype(np.float32) * m + albedo_bgr.astype(np.float32) * (1.0 - m)).round().astype(np.uint8)
        return fused

    def detail_normals(self, normal_map: np.ndarray, base_albedo_bgr: np.ndarray, detailed_albedo_bgr: np.ndarray,
                       strength: float = 0.25) -> np.ndarray:
        """Add measured albedo-luminance high-pass to the tangent-space slope of a normal map; no network runs here."""
        _validate_uv(normal_map, "normal_map")
        _validate_uv(base_albedo_bgr, "base_albedo")
        _validate_uv(detailed_albedo_bgr, "detailed_albedo")
        delta = (cv2.cvtColor(detailed_albedo_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                 - cv2.cvtColor(base_albedo_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)) / 255.0
        vec = normal_map[:, :, ::-1].astype(np.float32) / 127.5 - 1.0      # BGR u8 -> RGB xyz
        vec[..., 0] -= strength * cv2.Sobel(delta, cv2.CV_32F, 1, 0, ksize=3)
        vec[..., 1] -= strength * cv2.Sobel(delta, cv2.CV_32F, 0, 1, ksize=3)
        vec /= np.maximum(np.linalg.norm(vec, axis=2, keepdims=True), 1e-6)
        return np.ascontiguousarray(((vec + 1.0) * 127.5).clip(0, 255)[:, :, ::-1]).round().astype(np.uint8)


__all__ = ["ALLOWED_UV_SIZES", "UVDetailerInputError", "UVTextureDetailer", "laplacian_pyramid_fusion",
           "to_model_input", "from_model_output"]
