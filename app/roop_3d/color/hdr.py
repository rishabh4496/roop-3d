"""Deterministic HDR (PQ / HLG, wide gamut) -> SDR (BT.709, BT.1886 display) conversion.

This is the only HDR processing path the swap pipeline offers, because the face models and the
compositor work on 8-bit display-referred RGB. It is an explicit, parameterised, documented operator,
never a silent reinterpretation of HDR code values as SDR:

1. Decode the signal to absolute display light: PQ via ST 2084, HLG via the BT.2100 OETF^-1 plus
   the reference OOTF for the declared peak.
2. Scale so HDR reference white (203 cd/m^2, ITU-R BT.2408) lands on SDR reference white
   (100 cd/m^2). Without this, diffuse white would be rendered 2x too bright.
3. Compress highlights with the ITU-R BT.2390 EETF (Hermite spline in the PQ domain), per channel.
4. Convert the primaries (BT.2020 / P3-D65 -> BT.709) in linear light, clipping out-of-gamut values.
5. Encode with the inverse BT.1886 power law (what an SDR display applies).

Per-channel highlight compression desaturates and can hue-shift very bright saturated highlights;
that is the documented trade-off for a fast, stateless, reproducible operator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from roop_3d.color import gamut, transfer
from roop_3d.color.codes import TRC_HLG, TRC_PQ

ALGORITHM = "bt2390_eetf_v1"
DEFAULT_SOURCE_PEAK_NITS = 1000.0


@dataclass(frozen=True)
class ToneMapParams:
    sdr_peak_nits: float = 100.0
    reference_white_nits: float = 203.0
    source_peak_nits: float = DEFAULT_SOURCE_PEAK_NITS
    algorithm: str = ALGORITHM

    def __post_init__(self) -> None:
        if not (0 < self.sdr_peak_nits <= self.reference_white_nits <= transfer.PQ_PEAK_NITS):
            raise ValueError("Require 0 < SDR peak <= reference white <= 10000 cd/m^2")
        if not self.sdr_peak_nits <= self.source_peak_nits <= transfer.PQ_PEAK_NITS:
            raise ValueError("Source peak must lie between the SDR peak and 10000 cd/m^2")

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def bt2390_eetf(signal, source_peak_nits: float, target_peak_nits: float, target_black_nits: float = 0.0):
    """ITU-R BT.2390-11 section 5.4.1 EETF: PQ signal -> PQ signal, compressing to the target peak.

    The input is clipped to the source mastering range first: the spline is only defined on it.
    When the target peak is not below the source peak there is nothing to compress.
    """
    e = np.clip(np.asarray(signal, dtype=np.float64), 0.0, 1.0)
    min_lw = float(transfer.pq_inverse_eotf(0.0))
    max_lw = float(transfer.pq_inverse_eotf(source_peak_nits))
    span = max_lw - min_lw
    max_lum = (float(transfer.pq_inverse_eotf(target_peak_nits)) - min_lw) / span
    min_lum = (float(transfer.pq_inverse_eotf(target_black_nits)) - min_lw) / span
    e1 = np.clip((e - min_lw) / span, 0.0, 1.0)
    if max_lum >= 1.0:
        return e1 * span + min_lw
    knee = 1.5 * max_lum - 0.5
    t = np.clip((e1 - knee) / (1.0 - knee), 0.0, 1.0)
    t2, t3 = t * t, t * t * t
    spline = ((2 * t3 - 3 * t2 + 1) * knee + (t3 - 2 * t2 + t) * (1.0 - knee) + (-2 * t3 + 3 * t2) * max_lum)
    e2 = np.where(e1 < knee, e1, spline)
    e3 = e2 + min_lum * np.power(1.0 - e2, 4.0)
    return e3 * span + min_lw


class HdrToSdr:
    """Stateless converter from 16-bit full-range R'G'B' (PQ or HLG coded) to 8-bit BT.709/BT.1886 RGB."""

    _TABLE_SIZE = 8192

    def __init__(self, transfer_code: int, primaries: int, params: ToneMapParams | None = None):
        if transfer_code not in (TRC_PQ, TRC_HLG):
            raise ValueError("HdrToSdr handles only PQ (16) and HLG (18) transfers")
        if primaries not in gamut.SUPPORTED_PRIMARIES:
            raise ValueError(f"Unsupported source primaries code {primaries}")
        self.transfer_code, self.primaries = transfer_code, primaries
        self.params = params or ToneMapParams()
        p = self.params
        self.scale = p.sdr_peak_nits / p.reference_white_nits
        self._effective_peak = max(p.source_peak_nits * self.scale, p.sdr_peak_nits)
        grid = np.linspace(0.0, 1.0, self._TABLE_SIZE)
        compressed = bt2390_eetf(grid, self._effective_peak, p.sdr_peak_nits)
        # Display light relative to the SDR peak: the value 1.0 is SDR reference white.
        self._grid = grid
        self._relative = (transfer.pq_eotf(compressed) / p.sdr_peak_nits).astype(np.float32)
        self._matrix = gamut.conversion_matrix(primaries, 1).astype(np.float32)
        codes = np.arange(65536, dtype=np.float64) / 65535.0
        if transfer_code == TRC_PQ:
            nits = transfer.pq_eotf(codes) * self.scale
            self._code_lut = np.interp(transfer.pq_inverse_eotf(nits), grid, self._relative).astype(np.float32)
        else:
            self._code_lut = transfer.hlg_inverse_oetf(codes).astype(np.float32)

    def describe(self) -> dict:
        return {"algorithm": self.params.algorithm, "parameters": asdict(self.params),
                "parameters_sha256": self.params.fingerprint(), "source_transfer": self.transfer_code,
                "source_primaries": self.primaries, "target_primaries": 1,
                "target_encoding": "inverse BT.1886 (gamma 2.4)", "gamut_mapping": "clip",
                "highlight_compression": "per-channel"}

    def to_relative_linear(self, rgb16: np.ndarray) -> np.ndarray:
        """16-bit code values -> linear BT.709 light relative to SDR reference white (unclipped above 1)."""
        data = np.asarray(rgb16)
        if data.ndim != 3 or data.shape[2] != 3 or data.dtype != np.uint16:
            raise ValueError("Expected uint16 R'G'B' with shape (height, width, 3)")
        if self.transfer_code == TRC_PQ:
            linear = self._code_lut[data]
        else:
            scene = self._code_lut[data]
            nits = transfer.hlg_ootf(scene, self.params.source_peak_nits).astype(np.float32) * np.float32(self.scale)
            signal = transfer.pq_inverse_eotf(nits).astype(np.float32)
            linear = np.interp(signal, self._grid, self._relative).astype(np.float32)
        return np.einsum("...c,dc->...d", linear, self._matrix, optimize=True)

    def convert(self, rgb16: np.ndarray) -> np.ndarray:
        """16-bit PQ/HLG R'G'B' -> uint8 RGB in BT.709 primaries with BT.1886 encoding."""
        linear = np.clip(self.to_relative_linear(rgb16), 0.0, 1.0)
        encoded = np.power(linear, np.float32(1.0 / 2.4))
        return np.rint(encoded * 255.0).astype(np.uint8)
