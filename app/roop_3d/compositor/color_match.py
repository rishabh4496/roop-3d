"""Masked LAB statistics and histogram color transfer."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from roop_3d.core.errors import DomainError


@dataclass(frozen=True)
class ColorStats:
    source_mean: tuple[float, float, float]
    source_std: tuple[float, float, float]
    target_mean: tuple[float, float, float]
    target_std: tuple[float, float, float]

    def vector(self) -> np.ndarray:
        return np.asarray((*self.source_mean, *self.source_std, *self.target_mean, *self.target_std), np.float32)

    @classmethod
    def from_vector(cls, value: np.ndarray) -> "ColorStats":
        row = np.asarray(value, dtype=np.float32).reshape(-1)
        if row.size != 12 or not np.isfinite(row).all():
            raise DomainError("COLOR_STATS_INVALID", "Color statistics must contain 12 finite values", 422)
        return cls(tuple(row[0:3]), tuple(row[3:6]), tuple(row[6:9]), tuple(row[9:12]))


def _validate_pair(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = np.asarray(source), np.asarray(reference)
    weight = np.asarray(mask, dtype=np.float32)
    if (a.ndim != 3 or b.shape != a.shape or a.shape[2] != 3 or weight.shape != a.shape[:2]
            or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(weight).all()):
        raise DomainError("COLOR_MATCH_INPUT_INVALID", "Color inputs and mask have incompatible dimensions", 422)
    return a, b, np.clip(weight, 0.0, 1.0)


def _lab(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    rgb = np.clip(image.astype(np.float32), 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2LAB).astype(np.float32)


def estimate_lab_stats(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> ColorStats:
    a, b, weight = _validate_pair(source, reference, mask)
    selected = weight > 0.15
    if np.count_nonzero(selected) < 8:
        selected = weight > 0
    if np.count_nonzero(selected) < 2:
        raise DomainError("COLOR_MATCH_MASK_EMPTY", "At least two masked pixels are needed for color matching", 422)
    source_lab, target_lab = _lab(a), _lab(b)
    source_mean = source_lab[selected].mean(axis=0)
    source_std = np.maximum(source_lab[selected].std(axis=0), 1.0)
    target_mean = target_lab[selected].mean(axis=0)
    target_std = np.maximum(target_lab[selected].std(axis=0), 1.0)
    return ColorStats(tuple(source_mean), tuple(source_std), tuple(target_mean), tuple(target_std))


def apply_lab_stats(image: np.ndarray, stats: ColorStats, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    source = np.asarray(image)
    alpha = np.asarray(mask, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3 or alpha.shape != source.shape[:2] or not np.isfinite(alpha).all():
        raise DomainError("COLOR_MATCH_INPUT_INVALID", "Image and mask dimensions are invalid", 422)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("Color transfer strength must be within 0..1")
    if strength == 0:
        return source.copy()
    lab = _lab(source)
    src_mean = np.asarray(stats.source_mean, np.float32)
    src_std = np.maximum(np.asarray(stats.source_std, np.float32), 1.0)
    tgt_mean = np.asarray(stats.target_mean, np.float32)
    tgt_std = np.maximum(np.asarray(stats.target_std, np.float32), 1.0)
    matched = (lab - src_mean) * (tgt_std / src_std) + tgt_mean
    result_lab = lab.copy()
    active = alpha > 0
    result_lab[active] = lab[active] + (matched[active] - lab[active]) * strength
    output = cv2.cvtColor(np.clip(result_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    output[~active] = source[~active]
    return output


def lab_mean_std_match(source: np.ndarray, reference: np.ndarray, mask: np.ndarray, *, strength: float = 1.0) -> np.ndarray:
    return apply_lab_stats(source, estimate_lab_stats(source, reference, mask), mask, strength=strength)


def masked_histogram_match(source: np.ndarray, reference: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel masked LAB CDF matching; all pixels outside the mask are copied."""
    a, b, weight = _validate_pair(source, reference, mask)
    active = weight > 0
    if np.count_nonzero(active) < 2:
        return a.copy()
    source_lab, target_lab = _lab(a), _lab(b)
    mapped = source_lab.copy()
    for channel in range(3):
        x = source_lab[..., channel][active]
        y = target_lab[..., channel][active]
        values, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
        quantiles = np.cumsum(counts).astype(np.float64) / max(1, len(x))
        ref_values, ref_counts = np.unique(y, return_counts=True)
        ref_quantiles = np.cumsum(ref_counts).astype(np.float64) / max(1, len(y))
        mapped_values = np.interp(quantiles, ref_quantiles, ref_values)
        mapped[..., channel][active] = mapped_values[inverse]
    output = cv2.cvtColor(np.clip(mapped, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    output[~active] = a[~active]
    return output
