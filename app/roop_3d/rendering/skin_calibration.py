# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Scene-aware, skin-tone-preserving color calibration in linear light.

This is an original deterministic intrinsic-color adapter, not a learned
TRUST model. It makes the image/illumination ambiguity explicit: scene white
balance and exposure are estimated separately, then skin chromaticity is
measured in CIE Lab and a two-axis melanin/hemoglobin optical-density proxy.
The proxy is a color-coordinate system, not a spectrometer or racial classifier.
Images use OpenCV BGR ordering; albedo and illumination math use linear light.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


_LUMA_BGR = np.array([0.114, 0.587, 0.299], np.float32)
# Approximate optical-density directions in BGR order. They provide stable
# chromaticity coordinates, while the residual third color dimension is kept
# in the full linear-RGB albedo image.
_MELANIN_HEMOGLOBIN_BASIS = np.array([
    [0.78, 0.12],
    [0.40, 0.75],
    [0.17, 0.43],
], np.float32)
_MH_PINV = np.linalg.pinv(_MELANIN_HEMOGLOBIN_BASIS).astype(np.float32)


def _as_bgr_float(image: np.ndarray) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3 or min(value.shape[:2]) < 1:
        raise ValueError("image must be a non-empty BGR image with shape (H, W, 3)")
    if value.dtype == np.uint8:
        return value.astype(np.float32) / 255.0
    value = value.astype(np.float32)
    if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError("floating BGR images must be finite and within [0, 1]")
    return value


def srgb_to_linear(image: np.ndarray) -> np.ndarray:
    value = _as_bgr_float(image)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(image: np.ndarray) -> np.ndarray:
    value = np.maximum(np.asarray(image, np.float32), 0.0)
    return np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1 / 2.4) - 0.055).astype(np.float32)


def _blackbody_rgb(cct_kelvin: float) -> np.ndarray:
    """Approximate sRGB black-body chromaticity for 1000–40000 K."""
    t = float(np.clip(cct_kelvin, 1000.0, 40000.0)) / 100.0
    if t <= 66:
        red = 255.0
        green = 99.4708025861 * np.log(max(t, 1e-6)) - 161.1195681661
        blue = 0.0 if t <= 19 else 138.5177312231 * np.log(t - 10) - 305.0447927307
    else:
        red = 329.698727446 * (t - 60) ** -0.1332047592
        green = 288.1221695283 * (t - 60) ** -0.0755148492
        blue = 255.0
    return np.clip(np.array([red, green, blue], np.float32), 0, 255) / 255.0


def _rgb_to_cct(rgb: np.ndarray) -> float:
    """McCamy CCT estimate from linear RGB; clipped to practical indoor/outdoor range."""
    r, g, b = np.asarray(rgb, np.float64)
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    total = max(x + y + z, 1e-10)
    chroma_x, chroma_y = x / total, y / total
    n = (chroma_x - 0.3320) / (0.1858 - chroma_y + 1e-8)
    cct = 449.0 * n**3 + 3525.0 * n**2 + 6823.3 * n + 5520.33
    return float(np.clip(cct, 1800.0, 12000.0))


@dataclass(frozen=True)
class SceneIllumination:
    """Relative scene white balance and intensity, independent of skin reflectance."""

    cct_kelvin: float = 6500.0
    intensity: float = 1.0
    white_balance_bgr: tuple[float, float, float] = (1.0, 1.0, 1.0)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.cct_kelvin) or not 1000 <= self.cct_kelvin <= 40000:
            raise ValueError("cct_kelvin must be finite and within 1000..40000")
        if not np.isfinite(self.intensity) or self.intensity <= 0:
            raise ValueError("intensity must be finite and positive")
        gains = np.asarray(self.white_balance_bgr, np.float32)
        if gains.shape != (3,) or not np.isfinite(gains).all() or np.any(gains <= 0):
            raise ValueError("white_balance_bgr must contain three positive finite gains")
        if not np.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")

    @classmethod
    def from_temperature(cls, cct_kelvin: float, intensity: float = 1.0) -> "SceneIllumination":
        rgb_srgb = _blackbody_rgb(cct_kelvin)
        rgb_linear = np.where(rgb_srgb <= 0.04045, rgb_srgb / 12.92,
                              ((rgb_srgb + 0.055) / 1.055) ** 2.4)
        bgr = rgb_linear[::-1].astype(np.float32)
        luminance = max(float(bgr @ _LUMA_BGR), 1e-5)
        normalized = bgr / luminance
        return cls(float(cct_kelvin), float(intensity), tuple(float(v) for v in normalized), 1.0)

    @classmethod
    def from_scene(
        cls,
        scene_bgr: np.ndarray,
        *,
        neutral_mask: np.ndarray | None = None,
        assumed_neutral_luma: float = 0.18,
    ) -> "SceneIllumination":
        """Estimate a global illuminant from neutral scene pixels or a weak gray-world fallback.

        A supplied ``neutral_mask`` should identify a gray/white card or known
        neutral surface. Without one, only low-chroma bright pixels are trusted;
        if there are too few, a lower-confidence gray-world estimate is returned.
        """
        image = srgb_to_linear(scene_bgr)
        if not 0.01 <= assumed_neutral_luma <= 1:
            raise ValueError("assumed_neutral_luma must be within .01..1")
        luma = image @ _LUMA_BGR
        if neutral_mask is not None:
            mask = np.asarray(neutral_mask) > 0
            if mask.shape != image.shape[:2]:
                raise ValueError("neutral_mask must match the scene image")
            mask &= np.isfinite(luma) & (luma > 0.015)
            confidence = 0.95
        else:
            channel_max = image.max(axis=2)
            saturation = (channel_max - image.min(axis=2)) / np.maximum(channel_max, 1e-5)
            bright = luma >= np.quantile(luma, 0.78)
            mask = (saturation < 0.16) & bright & (luma > 0.02)
            confidence = 0.65
            if int(mask.sum()) < 16:
                lower, upper = np.quantile(luma, [0.2, 0.8])
                mask = (luma >= lower) & (luma <= upper) & (luma > 0.02)
                confidence = 0.2
        pixels = image[mask]
        if len(pixels) < 4:
            return cls(6500.0, 1.0, (1.0, 1.0, 1.0), 0.0)
        white = np.median(pixels, axis=0)
        median_luma = max(float(white @ _LUMA_BGR), 1e-5)
        wb = white / median_luma
        cct = _rgb_to_cct(white[::-1])
        intensity = float(np.clip(median_luma / assumed_neutral_luma, 0.25, 4.0))
        return cls(cct, intensity, tuple(float(v) for v in wb), confidence)

    @property
    def linear_bgr_gain(self) -> np.ndarray:
        return np.asarray(self.white_balance_bgr, np.float32) * np.float32(self.intensity)


@dataclass(frozen=True)
class SkinToneEstimate:
    albedo_linear_bgr: np.ndarray
    lab_median: np.ndarray
    melanin_median: float
    hemoglobin_median: float
    illumination: SceneIllumination


class UnbiasedSkinToneCalibrator:
    """Disentangle scene white balance/exposure from skin chromaticity.

    ``albedo_preservation_ratio`` is a direct identity/tone dial: zero adopts
    the target's intrinsic skin tone, one uses the source's intrinsic skin tone.
    The chosen chromaticity is then re-illuminated by the target scene estimate.
    This implementation does not infer race or impose a population skin prior.
    """

    def __init__(self, *, minimum_samples: int = 32) -> None:
        if minimum_samples < 4:
            raise ValueError("minimum_samples must be at least four")
        self.minimum_samples = int(minimum_samples)

    @staticmethod
    def _skin_mask(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        h, w = image.shape[:2]
        if mask is not None:
            selected = np.asarray(mask) > 0
            if selected.shape != (h, w):
                raise ValueError("skin mask must match its face crop")
        else:
            yy, xx = np.mgrid[:h, :w]
            selected = (((xx - (w - 1) / 2) / max(w * 0.42, 1)) ** 2
                        + ((yy - (h - 1) / 2) / max(h * 0.44, 1)) ** 2) <= 1
        linear = srgb_to_linear(image)
        luma = linear @ _LUMA_BGR
        candidates = luma[selected & np.isfinite(luma)]
        if candidates.size >= 16:
            low, high = np.quantile(candidates, [0.02, 0.995])
            selected &= (luma >= low) & (luma <= high)
        return selected

    def decompose(
        self,
        face_bgr: np.ndarray,
        *,
        skin_mask: np.ndarray | None = None,
        illumination: SceneIllumination | None = None,
        scene_bgr: np.ndarray | None = None,
        scene_neutral_mask: np.ndarray | None = None,
    ) -> SkinToneEstimate:
        image = _as_bgr_float(face_bgr)
        if illumination is not None and scene_bgr is not None:
            raise ValueError("provide illumination or scene_bgr, not both")
        light = illumination or (SceneIllumination.from_scene(scene_bgr, neutral_mask=scene_neutral_mask)
                                 if scene_bgr is not None else SceneIllumination())
        selected = self._skin_mask(face_bgr, skin_mask)
        if int(selected.sum()) < self.minimum_samples:
            raise ValueError("face crop contains too few usable skin pixels")
        linear = srgb_to_linear(face_bgr)
        albedo = np.clip(linear / light.linear_bgr_gain[None, None, :], 0.0, 1.0)
        encoded = np.clip(linear_to_srgb(albedo), 0, 1)
        encoded_u8 = np.rint(encoded * 255).astype(np.uint8)
        lab = cv2.cvtColor(encoded_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[..., 0] *= 100.0 / 255.0
        lab[..., 1:] -= 128.0
        lab_samples = lab[selected]
        optical_density = -np.log(np.maximum(albedo[selected], 1e-4))
        pigments = np.maximum(optical_density @ _MH_PINV.T, 0)
        return SkinToneEstimate(
            albedo_linear_bgr=albedo.astype(np.float32),
            lab_median=np.median(lab_samples, axis=0).astype(np.float32),
            melanin_median=float(np.median(pigments[:, 0])),
            hemoglobin_median=float(np.median(pigments[:, 1])),
            illumination=light,
        )

    def calibrate(
        self,
        source_face_bgr: np.ndarray,
        target_face_bgr: np.ndarray,
        *,
        source_skin_mask: np.ndarray | None = None,
        target_skin_mask: np.ndarray | None = None,
        source_illumination: SceneIllumination | None = None,
        target_illumination: SceneIllumination | None = None,
        source_scene_bgr: np.ndarray | None = None,
        target_scene_bgr: np.ndarray | None = None,
        albedo_preservation_ratio: float = 1.0,
    ) -> np.ndarray:
        """Return a target-sized BGR face with selected intrinsic tone under target light."""
        ratio = float(albedo_preservation_ratio)
        if not np.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError("albedo_preservation_ratio must be within [0, 1]")
        source = self.decompose(source_face_bgr, skin_mask=source_skin_mask,
                                illumination=source_illumination, scene_bgr=source_scene_bgr)
        target = self.decompose(target_face_bgr, skin_mask=target_skin_mask,
                                illumination=target_illumination, scene_bgr=target_scene_bgr)
        source_lab, target_lab = source.lab_median, target.lab_median
        mixed_lab = target_lab * (1.0 - ratio) + source_lab * ratio
        desired_srgb_bgr = cv2.cvtColor(mixed_lab.reshape(1, 1, 3).astype(np.float32), cv2.COLOR_LAB2BGR)[0, 0]
        desired_linear = srgb_to_linear(np.clip(desired_srgb_bgr.reshape(1, 1, 3), 0, 1))[0, 0]
        target_mask = self._skin_mask(target_face_bgr, target_skin_mask)
        target_median = np.median(target.albedo_linear_bgr[target_mask], axis=0)
        tone_scale = np.clip(desired_linear / np.maximum(target_median, 1e-4), 0.1, 8.0)
        intrinsic = np.clip(target.albedo_linear_bgr * tone_scale[None, None, :], 0, 1)
        target_light = target.illumination.linear_bgr_gain
        relit = linear_to_srgb(intrinsic * target_light[None, None, :])
        return np.rint(np.clip(relit, 0, 1) * 255).astype(np.uint8)
