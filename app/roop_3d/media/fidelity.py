"""Does the finished file decode to what the encoder was given?

A wrong matrix, a limited/full range slip, or a lost bit depth shifts the *distribution* of decoded code
values even when every frame is present and every timestamp is right. A few frames sampled during encode
(first, 8th, 64th, 512th, 4096th and the last) are compared with the same frames decoded from the
finished file: signed per-channel bias, and the 1st / 99th percentile deltas that a range squeeze moves
immediately. Lossless profiles must match exactly. Random noise (which 4:2:0 subsampling destroys
pixel-for-pixel but not statistically) passes; a 16..235 squeeze or a 601/709 mix-up does not.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from roop_3d.core.errors import DomainError

KEEP = frozenset({0, 7, 63, 511, 4095})


class FidelitySampler:
    def __init__(self) -> None:
        self.samples: dict[int, np.ndarray] = {}
        self._tail: tuple[int, np.ndarray] | None = None

    def offer(self, index: int, image: np.ndarray) -> None:
        if index in KEEP:
            self.samples[index] = image.copy()
        self._tail = (index, image.copy()) if (self._tail is None or index >= self._tail[0]) else self._tail

    def finish(self) -> dict[int, np.ndarray]:
        samples = dict(self.samples)
        if self._tail is not None:
            samples[self._tail[0]] = self._tail[1]
        return samples


def _stats(expected: np.ndarray, observed: np.ndarray) -> dict:
    a = expected.reshape(-1, expected.shape[-1])[::3].astype(np.float32)
    b = observed.reshape(-1, observed.shape[-1])[::3].astype(np.float32)
    bias = (b.mean(axis=0) - a.mean(axis=0))
    tails = [np.percentile(b, q, axis=0) - np.percentile(a, q, axis=0) for q in (1, 99)]
    return {"bias": [round(float(v), 3) for v in bias],
            "p1_delta": [round(float(v), 3) for v in tails[0]], "p99_delta": [round(float(v), 3) for v in tails[1]],
            "mean_abs_error": round(float(np.abs(a - b).mean()), 3)}


def compare_frames(index: int, expected: np.ndarray, observed: np.ndarray, *, bias_tolerance: float,
                   tail_tolerance: float, scale: float = 1.0) -> dict:
    """Compare equal-shaped images; ``scale`` converts code values to 8-bit units (255 / max code)."""
    if expected.shape != observed.shape:
        raise DomainError("OUTPUT_FIDELITY", f"Decoded frame {index} has a different shape than the encoder input", 422)
    if bias_tolerance == 0 and tail_tolerance == 0:
        exact = bool(np.array_equal(expected, observed))
        stats = {"exact": exact, "mean_abs_error": 0.0 if exact else round(float(np.abs(
            expected.astype(np.float32) - observed.astype(np.float32)).mean() * scale), 3)}
        if not exact:
            raise DomainError("OUTPUT_FIDELITY", f"Lossless output differs from the encoder input at frame {index}", 422)
        return {"frame": index, **stats}
    stats = _stats(expected * scale, observed * scale)
    worst_bias = max(abs(v) for v in stats["bias"])
    worst_tail = max(abs(v) for v in stats["p1_delta"] + stats["p99_delta"])
    if worst_bias > bias_tolerance or worst_tail > tail_tolerance:
        raise DomainError(
            "OUTPUT_COLOR_DRIFT",
            f"Frame {index} decodes with colour/range drift (bias {worst_bias:.2f}, tail {worst_tail:.2f}; "
            f"limits {bias_tolerance}/{tail_tolerance} 8-bit code values)", 422)
    return {"frame": index, **stats}


def luma_plane(frame) -> np.ndarray:
    """The Y plane of a planar YCbCr ``av.VideoFrame`` as a 2-D array of its native integer type."""
    plane = frame.planes[0]
    dtype = np.uint8 if frame.format.components[0].bits <= 8 else np.uint16
    stride = plane.line_size // np.dtype(dtype).itemsize
    return np.ndarray((frame.height, stride), dtype, buffer=plane)[:, :frame.width].copy()


def compare_luma(index: int, source, produced, *, bias_tolerance: float, tail_tolerance: float) -> dict:
    """Native-path check: Y plane of a source frame against the same frame decoded from the output."""
    src, out = luma_plane(source), luma_plane(produced)
    src_scale = 255.0 / ((1 << source.format.components[0].bits) - 1)
    out_scale = 255.0 / ((1 << produced.format.components[0].bits) - 1)
    exact = bias_tolerance == 0 and tail_tolerance == 0
    if exact and src.dtype == out.dtype and np.array_equal(src, out):
        return {"frame": index, "exact": True, "plane": "Y", "mean_abs_error": 0.0}
    if exact:
        raise DomainError("OUTPUT_FIDELITY", f"Lossless native output differs from the source at frame {index}", 422)
    a = (src.astype(np.float32) * src_scale)[..., None]
    b = (out.astype(np.float32) * out_scale)[..., None]
    result = compare_frames(index, a, b, bias_tolerance=bias_tolerance, tail_tolerance=tail_tolerance)
    return {**result, "plane": "Y"}
