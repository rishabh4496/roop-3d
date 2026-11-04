"""RGB primaries, gamut conversion matrices and YCbCr coefficients.

Matrices are derived from the chromaticity coordinates (ITU-T H.273 Table 2), not typed in, so the
derivation can be checked against the published matrices. Only D65-white primaries are supported:
DCI-P3 (code 11) has a different white point and would need chromatic adaptation, which is
deliberately not guessed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

D65 = (0.3127, 0.3290)

#: ((rx, ry), (gx, gy), (bx, by), white)
CHROMATICITIES: dict[int, tuple] = {
    1: ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), D65),      # BT.709 / sRGB
    5: ((0.640, 0.330), (0.290, 0.600), (0.150, 0.060), D65),      # BT.470 B/G (625-line)
    6: ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), D65),      # SMPTE 170M (525-line)
    7: ((0.630, 0.340), (0.310, 0.595), (0.155, 0.070), D65),      # SMPTE 240M
    9: ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), D65),      # BT.2020 / BT.2100
    12: ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), D65),     # Display P3 (SMPTE EG 432-1)
}

#: (Kr, Kb) luma weights per matrix coefficient code (ITU-T H.273 Table 3).
LUMA_WEIGHTS: dict[int, tuple[float, float]] = {
    1: (0.2126, 0.0722),
    4: (0.30, 0.11),
    5: (0.299, 0.114),
    6: (0.299, 0.114),
    7: (0.212, 0.087),
    9: (0.2627, 0.0593),
}

SUPPORTED_PRIMARIES = tuple(CHROMATICITIES)
SUPPORTED_MATRICES = tuple(LUMA_WEIGHTS)


def _xy_to_xyz(x: float, y: float) -> np.ndarray:
    return np.array([x / y, 1.0, (1.0 - x - y) / y])


@lru_cache(maxsize=None)
def rgb_to_xyz_matrix(primaries: int) -> np.ndarray:
    """3x3 matrix taking linear RGB with the given primaries to CIE XYZ (Y of white = 1)."""
    if primaries not in CHROMATICITIES:
        raise ValueError(f"Unsupported colour primaries code {primaries}; supported: {SUPPORTED_PRIMARIES}")
    red, green, blue, white = CHROMATICITIES[primaries]
    columns = np.stack([_xy_to_xyz(*red), _xy_to_xyz(*green), _xy_to_xyz(*blue)], axis=1)
    scale = np.linalg.solve(columns, _xy_to_xyz(*white))
    matrix = columns * scale
    matrix.setflags(write=False)
    return matrix


@lru_cache(maxsize=None)
def conversion_matrix(source: int, target: int) -> np.ndarray:
    """3x3 matrix taking linear RGB in ``source`` primaries to linear RGB in ``target`` primaries."""
    matrix = np.linalg.inv(rgb_to_xyz_matrix(target)) @ rgb_to_xyz_matrix(source)
    matrix.setflags(write=False)
    return matrix


def luma_coefficients(primaries: int) -> np.ndarray:
    """Linear-light luminance weights (the Y row of the RGB->XYZ matrix)."""
    return rgb_to_xyz_matrix(primaries)[1].copy()


def matrix_weights(matrix: int) -> tuple[float, float]:
    if matrix not in LUMA_WEIGHTS:
        raise ValueError(f"Unsupported YCbCr matrix code {matrix}; supported: {SUPPORTED_MATRICES}")
    return LUMA_WEIGHTS[matrix]


def quantisation(bits: int, full_range: bool) -> tuple[float, float, float, float]:
    """(luma offset, luma scale, chroma centre, chroma scale) in code values."""
    if not 8 <= bits <= 16:
        raise ValueError("Bit depth must be 8..16")
    shift = bits - 8
    if full_range:
        top = float((1 << bits) - 1)
        return 0.0, top, float(1 << (bits - 1)), top
    return float(16 << shift), float(219 << shift), float(128 << shift), float(224 << shift)


def rgb_to_ycbcr(rgb, matrix: int, full_range: bool, bits: int) -> np.ndarray:
    """Gamma-encoded RGB 0..1 (last axis 3) -> unrounded integer-scale Y'CbCr code values.

    Pure float reference implementation; used as the oracle that the production converters
    (swscale, libavfilter) are verified against.
    """
    kr, kb = matrix_weights(matrix)
    kg = 1.0 - kr - kb
    rgb = np.asarray(rgb, dtype=np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = kr * r + kg * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    y_off, y_scale, c_mid, c_scale = quantisation(bits, full_range)
    return np.stack([y_off + y * y_scale, c_mid + cb * c_scale, c_mid + cr * c_scale], axis=-1)


def ycbcr_to_rgb(ycbcr, matrix: int, full_range: bool, bits: int) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycbcr`; returns gamma-encoded RGB (not clipped)."""
    kr, kb = matrix_weights(matrix)
    kg = 1.0 - kr - kb
    values = np.asarray(ycbcr, dtype=np.float64)
    y_off, y_scale, c_mid, c_scale = quantisation(bits, full_range)
    y = (values[..., 0] - y_off) / y_scale
    cb = (values[..., 1] - c_mid) / c_scale
    cr = (values[..., 2] - c_mid) / c_scale
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.stack([r, g, b], axis=-1)
