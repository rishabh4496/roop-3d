"""Exact rational mapping from VFR frame PTS to integer audio sample windows."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math

import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.audio_sync.audio_features import AudioFeatures, AudioSignal


@dataclass(frozen=True)
class AudioWindow:
    frame_time: Fraction
    center_sample: int
    start_sample: int
    end_sample: int
    samples: np.ndarray
    rms: float


def frame_sample_index(pts: int, time_base: Fraction, sample_rate: int = 16_000, *,
                       audio_start_time: Fraction = Fraction(0), offset_samples: int = 0) -> int:
    """Map each source PTS independently; integer offset is measured in audio samples."""
    if (type(pts) is not int or not isinstance(time_base, Fraction) or time_base <= 0
            or type(sample_rate) is not int or sample_rate <= 0 or type(offset_samples) is not int
            or not isinstance(audio_start_time, Fraction)):
        raise ValueError("PTS alignment requires integer ticks, rational bases and integer sample offsets")
    exact_sample = (pts * time_base - audio_start_time) * sample_rate + offset_samples
    return math.floor(exact_sample)


def window_for_frame(signal: AudioSignal, pts: int, time_base: Fraction, *,
                     window_samples: int = 3200, left_context_samples: int | None = None,
                     offset_samples: int = 0) -> AudioWindow:
    """Return a fixed-size, zero-padded window for one frame without cumulative drift."""
    if type(window_samples) is not int or window_samples <= 0:
        raise ValueError("window_samples must be a positive integer")
    left = window_samples // 2 if left_context_samples is None else left_context_samples
    if type(left) is not int or not 0 <= left <= window_samples:
        raise ValueError("left_context_samples must be within the requested window")
    frame_time = pts * time_base
    center = frame_sample_index(pts, time_base, signal.sample_rate,
                                audio_start_time=signal.start_time, offset_samples=offset_samples)
    start, end = center - left, center - left + window_samples
    output = np.zeros(window_samples, dtype=np.float32)
    source_start, source_end = max(0, start), min(signal.samples.size, end)
    if source_end > source_start:
        destination_start = source_start - start
        output[destination_start:destination_start + source_end - source_start] = signal.samples[source_start:source_end]
    rms = float(np.sqrt(np.mean(np.square(output, dtype=np.float64))))
    return AudioWindow(frame_time, center, start, end, output, rms)


def mel_window_for_audio_window(features: AudioFeatures, window: AudioWindow) -> np.ndarray:
    """Slice hop-anchored mel frames for a sample window, padding at clip boundaries."""
    hop = features.hop_length
    count = max(1, math.ceil(window.samples.size / hop))
    result = np.zeros((features.mel.shape[0], count), dtype=np.float32)
    first = window.start_sample // hop
    for index in range(count):
        source_index = first + index
        if 0 <= source_index < features.mel.shape[1]:
            result[:, index] = features.mel[:, source_index]
    return result


def window_for_timestamp(signal: AudioSignal, timestamp: Fraction, *,
                         window_samples: int = 3200, left_context_samples: int | None = None,
                         offset_samples: int = 0) -> AudioWindow:
    """Timestamp convenience wrapper that retains exact rational seconds."""
    if not isinstance(timestamp, Fraction):
        raise ValueError("timestamp must be a Fraction")
    return window_for_frame(signal, timestamp.numerator, Fraction(1, timestamp.denominator),
                            window_samples=window_samples, left_context_samples=left_context_samples,
                            offset_samples=offset_samples)
