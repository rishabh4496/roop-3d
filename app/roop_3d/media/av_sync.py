"""Audio integrity and synchronisation evidence for a produced file.

Two independent checks per audio track:

* ``payload_digest`` -- SHA-256 over the track's compressed packet payloads. For a stream-copied track the
  source and output digests must be identical: the audio was not touched, not re-encoded, not re-timed.
* ``measure_lag`` -- the signal-level offset between the source and output audio, measured by decoding
  both onto the same absolute timeline (source trim origin subtracted) and cross-correlating. This is
  codec-independent, so it also bounds the drift of re-encoded tracks and of AAC encoder delay.

Video PTS are verified exactly elsewhere, so a bounded audio lag is a bounded audio/video offset.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from fractions import Fraction
from math import ceil
from pathlib import Path

import av
import numpy as np

from roop_3d.media.probe import OPTIONS

LAG_RATE = 8000
SEARCH_SECONDS = 0.5


@contextmanager
def _open_audio_source(path: Path):
    """Open a media file or an audio-only sidecar, always through a handle with remote protocols disabled."""
    path = Path(path)
    with path.open("rb") as handle, av.open(handle, mode="r", options=OPTIONS) as container:
        yield container


def payload_digest(path: Path, stream_index: int, *, start: Fraction | None = None, end: Fraction | None = None) -> tuple[str, int]:
    """(sha256, packet count) of the compressed payload of one track, optionally within a source window."""
    digest, count = hashlib.sha256(), 0
    with _open_audio_source(path) as container:
        for packet in container.demux(container.streams[stream_index]):
            if packet.pts is None or not packet.size:
                continue
            at = packet.pts * Fraction(packet.time_base)
            if (start is not None and at < start) or (end is not None and at >= end):
                continue
            digest.update(bytes(packet))
            count += 1
    return digest.hexdigest(), count


def decode_window(path: Path, stream_index: int, window_start: Fraction, seconds: float,
                  rate: int = LAG_RATE) -> np.ndarray:
    """Mono float32 samples for [window_start, window_start + seconds) placed by presentation time."""
    size = ceil(seconds * rate)
    buffer = np.zeros(size, np.float32)
    resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
    end = window_start + Fraction(seconds).limit_denominator(1000)
    position: int | None = None  # buffer index of the next resampled sample

    def place(frames, at_seconds: Fraction) -> None:
        nonlocal position
        for out in frames:
            if position is None:  # the first output sample sits at the first input frame's time
                position = round((at_seconds - window_start) * rate)
            data = out.to_ndarray().reshape(-1)
            lo, hi = max(position, 0), min(position + data.size, size)
            if hi > lo:
                buffer[lo:hi] = data[lo - position:hi - position]
            position += data.size

    with _open_audio_source(path) as container:
        for frame in container.decode(container.streams[stream_index]):
            at = (frame.pts * Fraction(frame.time_base)) if frame.pts is not None else Fraction(0)
            if at >= end:
                break
            place(resampler.resample(frame), at)
        place(resampler.resample(None), window_start)
    return buffer


@dataclass(frozen=True)
class LagMeasurement:
    measurable: bool
    lag_seconds: float | None
    correlation: float | None
    reason: str | None = None
    ambiguous: bool = False  # periodic content: several lags correlate equally well


def measure_lag(reference: np.ndarray, test: np.ndarray, rate: int = LAG_RATE, *,
                search_seconds: float = SEARCH_SECONDS, min_rms: float = 1e-4) -> LagMeasurement:
    """Positive lag = ``test`` is later than ``reference``. Not measurable on (near-)silence."""
    n = min(reference.size, test.size)
    x, y = reference[:n].astype(np.float64), test[:n].astype(np.float64)
    if n < rate // 10:
        return LagMeasurement(False, None, None, "window too short")
    if np.sqrt(np.mean(x * x)) < min_rms or np.sqrt(np.mean(y * y)) < min_rms:
        return LagMeasurement(False, None, None, "silent window: nothing to correlate")
    x, y = x - x.mean(), y - y.mean()
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(y, size) * np.conj(np.fft.rfft(x, size))
    corr = np.fft.irfft(spectrum, size)
    reach = int(search_seconds * rate)
    lags = np.concatenate([np.arange(0, reach + 1), np.arange(-reach, 0)])
    values = np.concatenate([corr[:reach + 1], corr[size - reach:]])
    norm = np.sqrt(np.dot(x, x) * np.dot(y, y))
    top = float(values.max())
    quality = top / norm if norm > 0 else 0.0
    if quality < 0.5:
        return LagMeasurement(False, None, quality, "audio differs too much to align (re-encoded or edited)")
    # A pure tone correlates equally at every multiple of its period. Take the near-maximal peak closest to
    # zero lag (the conservative reading) and say so when other equally good peaks exist further away.
    strong = np.flatnonzero(values >= 0.98 * top)
    best = int(strong[np.argmin(np.abs(lags[strong]))])
    ambiguous = bool(np.any(np.abs(lags[strong] - lags[best]) > 2))
    return LagMeasurement(True, float(lags[best]) / rate, quality, "periodic content: lag only known modulo its period" if ambiguous else None, ambiguous)


@dataclass(frozen=True)
class TrackSync:
    stream_index: int
    action: str
    codec: str
    lag: LagMeasurement
    tolerance_seconds: float
    payload_identical: bool | None
    ok: bool

    def as_dict(self) -> dict:
        return {"source_stream": self.stream_index, "action": self.action, "codec": self.codec,
                "measurable": self.lag.measurable,
                "lag_ms": None if self.lag.lag_seconds is None else round(self.lag.lag_seconds * 1000, 3),
                "correlation": None if self.lag.correlation is None else round(self.lag.correlation, 4),
                "reason": self.lag.reason, "ambiguous": self.lag.ambiguous, "tolerance_ms": round(self.tolerance_seconds * 1000, 3),
                "payload_identical": self.payload_identical, "ok": self.ok}


def check_track(source: Path, output: Path, plan, output_index: int, *, origin: Fraction, source_start: Fraction | None,
                source_end: Fraction | None, seconds: float, tolerance: float) -> TrackSync:
    """Compare one source audio track with the matching output track (identical payload if copied)."""
    identical = None
    if plan.action == "copy":
        want, _ = payload_digest(source, plan.source_index, start=source_start, end=source_end)
        got, _ = payload_digest(output, output_index)
        identical = want == got
    reference = decode_window(source, plan.source_index, origin, seconds)
    produced = decode_window(output, output_index, Fraction(0), seconds)
    lag = measure_lag(reference, produced)
    within = (not lag.measurable) or abs(lag.lag_seconds) <= tolerance
    ok = within and (identical is not False)
    return TrackSync(plan.source_index, plan.action, plan.codec, lag, tolerance, identical, ok)
