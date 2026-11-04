"""Transfer functions (EOTF / OETF) with their published constants.

Every function is vectorised, finite for finite input, and defined on the closed interval it names;
inputs are clipped to that interval instead of returning NaN.

* PQ (SMPTE ST 2084 / ITU-R BT.2100): signal 0..1 <-> absolute display light 0..10000 cd/m^2.
* HLG (ARIB STD-B67 / ITU-R BT.2100): scene linear 0..1 <-> signal 0..1, plus the reference OOTF.
* BT.709 / BT.2020 OETF, IEC 61966-2-1 sRGB and pure power laws for SDR.
"""
from __future__ import annotations

import numpy as np

PQ_M1 = 2610.0 / 16384.0
PQ_M2 = 2523.0 / 4096.0 * 128.0
PQ_C1 = 3424.0 / 4096.0
PQ_C2 = 2413.0 / 4096.0 * 32.0
PQ_C3 = 2392.0 / 4096.0 * 32.0
PQ_PEAK_NITS = 10000.0

HLG_A = 0.17883277
HLG_B = 0.28466892  # 1 - 4a
HLG_C = 0.55991073  # 0.5 - a ln(4a)
#: BT.2100 HLG luminance weights are BT.2020 luma coefficients.
HLG_LUMA = np.array([0.2627, 0.6780, 0.0593])

#: Names accepted by :func:`to_linear` / :func:`from_linear` for SDR blending and conversion.
BLEND_TRANSFERS = ("srgb", "bt1886", "gamma22", "gamma28", "bt709_oetf", "linear")


def _f(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def pq_eotf(signal) -> np.ndarray:
    """PQ signal 0..1 -> absolute luminance in cd/m^2 (0..10000)."""
    p = np.power(np.clip(_f(signal), 0.0, 1.0), 1.0 / PQ_M2)
    numerator = np.maximum(p - PQ_C1, 0.0)
    denominator = PQ_C2 - PQ_C3 * p
    return PQ_PEAK_NITS * np.power(numerator / denominator, 1.0 / PQ_M1)


def pq_inverse_eotf(nits) -> np.ndarray:
    """Absolute luminance in cd/m^2 -> PQ signal 0..1."""
    y = np.power(np.clip(_f(nits) / PQ_PEAK_NITS, 0.0, 1.0), PQ_M1)
    return np.power((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2)


def hlg_oetf(scene) -> np.ndarray:
    """Normalised scene light 0..1 -> HLG signal 0..1."""
    e = np.clip(_f(scene), 0.0, 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        upper = HLG_A * np.log(np.maximum(12.0 * e - HLG_B, 1e-300)) + HLG_C
    return np.where(e <= 1.0 / 12.0, np.sqrt(3.0 * e), upper)


def hlg_inverse_oetf(signal) -> np.ndarray:
    """HLG signal 0..1 -> normalised scene light 0..1."""
    v = np.clip(_f(signal), 0.0, 1.0)
    return np.where(v <= 0.5, v * v / 3.0, (np.exp((v - HLG_C) / HLG_A) + HLG_B) / 12.0)


def hlg_system_gamma(peak_nits: float = 1000.0) -> float:
    """BT.2100 system gamma for a display of the given nominal peak (1.2 at 1000 cd/m^2)."""
    if peak_nits <= 0:
        raise ValueError("Display peak must be positive")
    return 1.2 + 0.42 * float(np.log10(peak_nits / 1000.0))


def hlg_ootf(scene_rgb, peak_nits: float = 1000.0, black_nits: float = 0.0) -> np.ndarray:
    """BT.2100 HLG reference OOTF: normalised scene RGB (last axis 3) -> display light in cd/m^2."""
    rgb = np.clip(_f(scene_rgb), 0.0, 1.0)
    luma = rgb @ HLG_LUMA
    gamma = hlg_system_gamma(peak_nits)
    gain = (peak_nits - black_nits) * np.power(luma, gamma - 1.0, where=luma > 0, out=np.zeros_like(luma))
    return gain[..., None] * rgb + black_nits


# BT.709 / BT.2020 OETF. The textbook constants 1.099 and 0.018 are rounded values of these exact ones;
# the exact pair makes the two segments meet, so the function is continuous and invertible.
BT709_ALPHA = 1.09929682680944
BT709_BETA = 0.018053968510807


def bt709_oetf(linear) -> np.ndarray:
    l = np.clip(_f(linear), 0.0, 1.0)
    return np.where(l < BT709_BETA, 4.5 * l, BT709_ALPHA * np.power(l, 0.45) - (BT709_ALPHA - 1.0))


def bt709_inverse_oetf(signal) -> np.ndarray:
    v = np.clip(_f(signal), 0.0, 1.0)
    return np.where(v < 4.5 * BT709_BETA, v / 4.5, np.power((v + (BT709_ALPHA - 1.0)) / BT709_ALPHA, 1.0 / 0.45))


def srgb_eotf(signal) -> np.ndarray:
    v = np.clip(_f(signal), 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, np.power((v + 0.055) / 1.055, 2.4))


def srgb_inverse_eotf(linear) -> np.ndarray:
    l = np.clip(_f(linear), 0.0, 1.0)
    return np.where(l <= 0.0031308, 12.92 * l, 1.055 * np.power(l, 1.0 / 2.4) - 0.055)


def power_eotf(signal, gamma: float) -> np.ndarray:
    return np.power(np.clip(_f(signal), 0.0, 1.0), gamma)


def power_inverse_eotf(linear, gamma: float) -> np.ndarray:
    return np.power(np.clip(_f(linear), 0.0, 1.0), 1.0 / gamma)


def to_linear(name: str, signal) -> np.ndarray:
    """Decode a display-referred SDR signal 0..1 to relative linear light 0..1."""
    if name == "srgb":
        return srgb_eotf(signal)
    if name == "bt1886":
        return power_eotf(signal, 2.4)
    if name == "gamma22":
        return power_eotf(signal, 2.2)
    if name == "gamma28":
        return power_eotf(signal, 2.8)
    if name == "bt709_oetf":
        return bt709_inverse_oetf(signal)
    if name == "linear":
        return np.clip(_f(signal), 0.0, 1.0)
    raise ValueError(f"Unknown blend transfer {name!r}; expected one of {BLEND_TRANSFERS}")


def from_linear(name: str, linear) -> np.ndarray:
    """Encode relative linear light 0..1 to the named SDR signal 0..1."""
    if name == "srgb":
        return srgb_inverse_eotf(linear)
    if name == "bt1886":
        return power_inverse_eotf(linear, 2.4)
    if name == "gamma22":
        return power_inverse_eotf(linear, 2.2)
    if name == "gamma28":
        return power_inverse_eotf(linear, 2.8)
    if name == "bt709_oetf":
        return bt709_oetf(linear)
    if name == "linear":
        return np.clip(_f(linear), 0.0, 1.0)
    raise ValueError(f"Unknown blend transfer {name!r}; expected one of {BLEND_TRANSFERS}")


def blend_transfer_for(transfer_code: int) -> str:
    """Blend transfer for a resolved SDR transfer code.

    sRGB-tagged material decodes with the piecewise sRGB curve. Video transfers (BT.709, BT.601,
    BT.2020) are graded on BT.1886 displays, so they decode with the pure 2.4 power law that a
    display applies; the scene-referred BT.709 OETF is available explicitly as ``bt709_oetf``.
    """
    return {13: "srgb", 4: "gamma22", 5: "gamma28", 8: "linear"}.get(int(transfer_code), "bt1886")


def lut_to_linear(name: str, bits: int = 8) -> np.ndarray:
    """Float32 lookup table, one entry per code value, for fast blending of integer images."""
    size = 1 << bits
    return to_linear(name, np.arange(size) / (size - 1)).astype(np.float32)
