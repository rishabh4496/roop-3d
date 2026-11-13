"""Frequency-separated Laplacian-pyramid blend with high-frequency retention and a masked Lab Reinhard base (§3.7).

Both inputs are decomposed into ``levels`` Laplacian bands (``L^0`` finest) plus a Gaussian base. With the Gaussian
pyramid ``m^k`` of the alpha matte:

* **high bands** ``k < hf_bands`` (pores, hair, stubble: what a face swapper smears)::

      L_out^k = m^k * (retain_hf * L_tgt^k + (1 - retain_hf) * L_swap^k) + (1 - m^k) * L_tgt^k

  ``retain_hf = 1`` keeps the target's own fine detail everywhere (inside the mask the high bands ARE the target's),
  ``retain_hf = 0`` is the ordinary multi-band mix (inside the mask the swap's detail);
* **mid bands** ``hf_bands <= k < levels``: the ordinary multi-band mix ``m^k L_swap^k + (1 - m^k) L_tgt^k``;
* **base**: the swap's base is first moved to the target's colour statistics with a Reinhard transfer in CIE Lab,
  ``(x - mu_s) * sigma_t / sigma_s + mu_t``, where ``mu``/``sigma`` are the alpha-weighted mean/std over the matte
  (per channel, std ratio clamped to ``[1/4, 4]``), blended by ``color_strength``; then mixed with ``m^base``.

Identity (the swap's shape and shading, i.e. the mid/low bands) therefore comes from the swap while skin tone and
micro-texture can follow the target. Limits: Lab statistics are global over the matte, so a target with strong
lighting gradients across the face is matched on average, not locally; the blend works on encoded (display) values
and is rejected together with linear-light blending by ``CompositeOptions``.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrequencyBlendOptions:
    levels: int = 5
    hf_bands: int = 2
    retain_hf: float = 1.0
    color_strength: float = 1.0

    def __post_init__(self) -> None:
        if not 2 <= self.levels <= 8 or not 0 <= self.hf_bands <= self.levels:
            raise ValueError("Frequency blend needs 2..8 levels and 0..levels high-frequency bands")
        if not (np.isfinite(self.retain_hf) and 0 <= self.retain_hf <= 1
                and np.isfinite(self.color_strength) and 0 <= self.color_strength <= 1):
            raise ValueError("retain_hf and color_strength must be within 0..1")


def _gaussian_pyramid(image: np.ndarray, count: int) -> list[np.ndarray]:
    stack = [image]
    for _ in range(count):
        stack.append(cv2.pyrDown(stack[-1]))
    return stack


def _laplacian(stack: list[np.ndarray]) -> list[np.ndarray]:
    bands = [stack[i] - cv2.pyrUp(stack[i + 1], dstsize=(stack[i].shape[1], stack[i].shape[0]))
             for i in range(len(stack) - 1)]
    return bands


def lab_reinhard(source: np.ndarray, target: np.ndarray, weight: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Move ``source`` (float32 BGR 0..255) to ``target``'s Lab mean/std measured with ``weight`` (0..1) as the mask."""
    total = float(weight.sum())
    if total < 1e-6 or strength <= 0:
        return source
    lab_s = cv2.cvtColor(np.clip(source, 0, 255) / 255.0, cv2.COLOR_BGR2LAB)
    lab_t = cv2.cvtColor(np.clip(target, 0, 255) / 255.0, cv2.COLOR_BGR2LAB)
    w = (weight / total)[..., None]
    mean_s, mean_t = (w * lab_s).sum((0, 1)), (w * lab_t).sum((0, 1))
    std_s = np.sqrt((w * (lab_s - mean_s) ** 2).sum((0, 1)))
    std_t = np.sqrt((w * (lab_t - mean_t) ** 2).sum((0, 1)))
    ratio = np.clip(std_t / np.maximum(std_s, 1e-3), 0.25, 4.0)
    moved = (lab_s - mean_s) * ratio + mean_t
    moved = lab_s + strength * (moved - lab_s)
    return (cv2.cvtColor(moved.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0).astype(np.float32)


def frequency_blend(target: np.ndarray, source: np.ndarray, alpha: np.ndarray,
                    options: FrequencyBlendOptions = FrequencyBlendOptions()) -> np.ndarray:
    """Blend ``source`` into ``target`` (same-size uint8/float BGR) under ``alpha`` (0..1); returns float32 0..255.

    The result differs from ``target`` only where the pyramid footprint of ``alpha`` reaches (``2**levels`` px beyond
    its support); callers that need bit-identical pixels beyond that (``fuse_warped``) restore them.
    """
    tgt, src = np.asarray(target, np.float32), np.asarray(source, np.float32)
    matte = np.clip(np.asarray(alpha, np.float32), 0.0, 1.0)
    if tgt.ndim != 3 or tgt.shape[2] != 3 or tgt.shape != src.shape or matte.shape != tgt.shape[:2]:
        raise ValueError("target, source and alpha must be same-size BGR/matte arrays")
    levels = max(1, min(options.levels, int(np.floor(np.log2(max(2, min(tgt.shape[:2]))))) - 1))
    hf_bands = min(options.hf_bands, levels)
    gauss_t, gauss_s, gauss_m = (_gaussian_pyramid(tgt, levels), _gaussian_pyramid(src, levels),
                                 _gaussian_pyramid(matte, levels))
    base_weight = cv2.resize(matte, (gauss_t[-1].shape[1], gauss_t[-1].shape[0]), interpolation=cv2.INTER_AREA)
    base_s = lab_reinhard(gauss_s[-1], gauss_t[-1], base_weight, options.color_strength)
    lap_t, lap_s = _laplacian(gauss_t), _laplacian(gauss_s)
    mixed: list[np.ndarray] = []
    for k in range(levels):
        m = gauss_m[k][..., None]
        swap_band = lap_s[k]
        if k < hf_bands:
            swap_band = options.retain_hf * lap_t[k] + (1.0 - options.retain_hf) * lap_s[k]
        mixed.append(m * swap_band + (1.0 - m) * lap_t[k])
    m_base = gauss_m[-1][..., None]
    result = m_base * base_s + (1.0 - m_base) * gauss_t[-1]
    for k in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(mixed[k].shape[1], mixed[k].shape[0])) + mixed[k]
    return np.clip(result, 0.0, 255.0).astype(np.float32)
