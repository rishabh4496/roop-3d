# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Clean-room intrinsic UV texture cleanup with a four-level detail pyramid."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.rendering.skin_calibration import linear_to_srgb, srgb_to_linear


@dataclass(frozen=True)
class TextureDecomposition:
    """Canonical albedo and extracted fine-scale relief in metric metres."""

    albedo_bgr: np.ndarray                  # HxWx3 uint8, sRGB BGR
    shading: np.ndarray                     # HxW relative low-frequency illumination
    specular_mask: np.ndarray               # HxW float32 confidence of removed highlights
    laplacian_bands: tuple[np.ndarray, ...]  # four signed linear-luminance bands
    displacement_bands_m: tuple[np.ndarray, ...]
    displacement_uv_m: np.ndarray           # HxW signed sum, metres


class IntrinsicTextureDecomposer:
    """Remove broad baked illumination while retaining observed skin detail.

    A robust log-luminance low pass estimates broad shading. The remaining
    reflectance is reconstructed from four Laplacian bands; fine bands are
    retained directly for pores and exported as small metric displacement
    bands. No synthetic pores, learned prior, or third-party checkpoint is used.
    """

    LEVELS = 4

    def __init__(
        self,
        *,
        shading_sigma_fraction: float = 0.045,
        highlight_percentile: float = 98.5,
        highlight_chroma_limit: float = 0.16,
        detail_gains: tuple[float, float, float, float] = (1.0, 1.0, 0.75, 0.5),
        displacement_scales_m: tuple[float, float, float, float] = (0.002, 0.001, 0.0005, 0.00025),
    ) -> None:
        if not 0 < shading_sigma_fraction <= 0.25:
            raise ValueError("shading_sigma_fraction must be within (0, .25]")
        if not 80 <= highlight_percentile < 100 or not 0 < highlight_chroma_limit < 1:
            raise ValueError("invalid highlight rejection parameters")
        if len(detail_gains) != self.LEVELS or any(not np.isfinite(v) or v < 0 for v in detail_gains):
            raise ValueError("detail_gains must contain four nonnegative finite values")
        if len(displacement_scales_m) != self.LEVELS or any(not np.isfinite(v) or v < 0 for v in displacement_scales_m):
            raise ValueError("displacement_scales_m must contain four nonnegative finite values")
        self.shading_sigma_fraction = float(shading_sigma_fraction)
        self.highlight_percentile = float(highlight_percentile)
        self.highlight_chroma_limit = float(highlight_chroma_limit)
        self.detail_gains = tuple(float(v) for v in detail_gains)
        self.displacement_scales_m = tuple(float(v) for v in displacement_scales_m)

    @staticmethod
    def _laplacian_pyramid(luminance: np.ndarray) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
        current = np.asarray(luminance, np.float32)
        bands: list[np.ndarray] = []
        for _ in range(IntrinsicTextureDecomposer.LEVELS):
            if min(current.shape) < 2:
                raise ValueError("UV texture must be at least 16x16 for a four-level pyramid")
            reduced = cv2.pyrDown(current)
            expanded = cv2.pyrUp(reduced, dstsize=(current.shape[1], current.shape[0]))
            bands.append(current - expanded)
            current = reduced
        return tuple(bands), current

    def decompose(self, texture_bgr: np.ndarray, valid_mask: np.ndarray | None = None) -> TextureDecomposition:
        texture = np.asarray(texture_bgr)
        if texture.ndim != 3 or texture.shape[2] != 3 or min(texture.shape[:2]) < 16:
            raise ValueError("texture_bgr must be an HxWx3 image of at least 16x16")
        if texture.dtype not in (np.uint8, np.float32, np.float64):
            raise ValueError("texture must be uint8 or floating point sRGB BGR")
        height, width = texture.shape[:2]
        valid = np.ones((height, width), bool) if valid_mask is None else np.asarray(valid_mask) > 0
        if valid.shape != (height, width):
            raise ValueError("valid_mask must match the texture dimensions")
        if int(valid.sum()) < 16:
            raise ValueError("valid_mask must contain at least 16 UV texels")

        linear = srgb_to_linear(texture)
        luminance = linear @ np.array([0.114, 0.587, 0.299], np.float32)
        log_luminance = np.log(np.maximum(luminance, 1e-4))
        sigma = max(1.0, min(height, width) * self.shading_sigma_fraction)
        log_shading = cv2.GaussianBlur(log_luminance, (0, 0), sigmaX=sigma, sigmaY=sigma)
        center = float(np.median(log_shading[valid]))
        shading = np.exp(log_shading - center).astype(np.float32)
        intrinsic_luma = luminance / np.maximum(shading, 1e-4)

        # White/specular flecks are low-chroma positive residuals, unlike skin
        # pigment. Replace only their luminance with a small robust neighborhood.
        maximum = linear.max(axis=2)
        chroma = (maximum - linear.min(axis=2)) / np.maximum(maximum, 1e-5)
        threshold = float(np.percentile(intrinsic_luma[valid], self.highlight_percentile))
        highlight = valid & (intrinsic_luma >= threshold) & (chroma <= self.highlight_chroma_limit)
        if highlight.any():
            local = cv2.medianBlur(intrinsic_luma.astype(np.float32), 5)
            intrinsic_luma[highlight] = local[highlight]

        # Preserve each texel's chromaticity while replacing its broad shading.
        ratio = intrinsic_luma / np.maximum(luminance, 1e-5)
        reflectance = np.clip(linear * ratio[..., None], 0.0, 1.0)
        reflectance_luma = reflectance @ np.array([0.114, 0.587, 0.299], np.float32)
        bands, coarse = self._laplacian_pyramid(reflectance_luma)

        # Reconstruct with all observed bands. Fine detail survives intact;
        # coarser bands that can contain residual illumination are attenuated.
        reconstructed_luma = coarse
        for band, gain in zip(reversed(bands), reversed(self.detail_gains)):
            up = cv2.pyrUp(reconstructed_luma, dstsize=(band.shape[1], band.shape[0]))
            reconstructed_luma = up + band * np.float32(gain)
        reconstructed_luma = np.maximum(reconstructed_luma, 1e-5)
        detail_ratio = reconstructed_luma / np.maximum(reflectance_luma, 1e-5)
        albedo_linear = np.clip(reflectance * detail_ratio[..., None], 0.0, 1.0)
        albedo_srgb = linear_to_srgb(albedo_linear)
        albedo_bgr = np.rint(np.clip(albedo_srgb, 0.0, 1.0) * 255).astype(np.uint8)
        albedo_bgr[~valid] = 0

        displacement_bands = tuple(
            (band * np.float32(scale)).astype(np.float32)
            for band, scale in zip(bands, self.displacement_scales_m)
        )
        displacement = np.zeros((height, width), np.float32)
        for item in displacement_bands:
            displacement += cv2.resize(item, (width, height), interpolation=cv2.INTER_LINEAR)
        displacement[~valid] = 0
        shading[~valid] = 0
        highlight_confidence = highlight.astype(np.float32)
        return TextureDecomposition(
            albedo_bgr=albedo_bgr,
            shading=shading,
            specular_mask=highlight_confidence,
            laplacian_bands=tuple(np.asarray(item, np.float32) for item in bands),
            displacement_bands_m=displacement_bands,
            displacement_uv_m=displacement,
        )

    def canonicalize(self, texture_bgr: np.ndarray, valid_mask: np.ndarray | None = None) -> TextureDecomposition:
        """Named seam for UV-atlas construction; equivalent to ``decompose``."""
        return self.decompose(texture_bgr, valid_mask)
