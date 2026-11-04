"""Local audio decode, deterministic 16 kHz mono conversion and log-mel features."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import av
import numpy as np

from roop_3d.core.errors import DomainError


FEATURE_SAMPLE_RATE = 16_000
DEFAULT_N_FFT = 800
DEFAULT_HOP_LENGTH = 200
DEFAULT_MEL_BINS = 80


@dataclass(frozen=True)
class AudioSignal:
    """Resampled feature audio; source audio on disk is never overwritten."""

    samples: np.ndarray
    sample_rate: int = FEATURE_SAMPLE_RATE
    start_time: Fraction = Fraction(0)
    source_path: Path | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.samples, dtype=np.float32)
        if (data.ndim != 1 or data.size == 0 or self.sample_rate <= 0
                or not isinstance(self.start_time, Fraction) or not np.isfinite(data).all()):
            raise DomainError("AUDIO_FEATURE_INPUT_INVALID", "Audio must be finite mono samples with a positive rate", 422)
        object.__setattr__(self, "samples", np.ascontiguousarray(data))


@dataclass(frozen=True)
class AudioFeatures:
    signal: AudioSignal
    mel: np.ndarray
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP_LENGTH

    def __post_init__(self) -> None:
        mel = np.asarray(self.mel, dtype=np.float32)
        if (mel.ndim != 2 or mel.shape[0] <= 0 or not np.isfinite(mel).all()
                or self.n_fft <= 0 or self.hop_length <= 0):
            raise DomainError("AUDIO_FEATURE_OUTPUT_INVALID", "Log-mel tensor must be finite bins-by-frames", 422)
        object.__setattr__(self, "mel", np.ascontiguousarray(mel))


def _float_mono(samples: np.ndarray) -> np.ndarray:
    values = np.asarray(samples)
    if values.ndim == 2:
        # Public PCM arrays are sample-major: (samples, channels).
        if not 0 < values.shape[1] <= 32:
            raise DomainError("AUDIO_CHANNEL_LAYOUT_INVALID", "PCM must be sample-major with 1..32 channels", 422)
        if np.issubdtype(values.dtype, np.integer):
            scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
            values = values.astype(np.float32) / scale
        else:
            values = values.astype(np.float32)
        values = values.mean(axis=1)
    elif values.ndim == 1:
        if np.issubdtype(values.dtype, np.integer):
            scale = float(max(abs(np.iinfo(values.dtype).min), np.iinfo(values.dtype).max))
            values = values.astype(np.float32) / scale
        else:
            values = values.astype(np.float32)
    else:
        raise DomainError("AUDIO_CHANNEL_LAYOUT_INVALID", "PCM must be mono or sample-major multichannel", 422)
    if values.size == 0 or not np.isfinite(values).all():
        raise DomainError("AUDIO_FEATURE_INPUT_INVALID", "PCM samples must be nonempty and finite", 422)
    return np.ascontiguousarray(np.clip(values, -1.0, 1.0), dtype=np.float32)


def _resample_frames(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    frame = av.AudioFrame.from_ndarray(mono[None, :], format="fltp", layout="mono")
    frame.sample_rate = source_rate
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
    output = list(resampler.resample(frame))
    output.extend(resampler.resample(None))
    if not output:
        raise DomainError("AUDIO_RESAMPLE_FAILED", "Audio resampler returned no samples", 422)
    result = np.concatenate([item.to_ndarray().reshape(-1) for item in output]).astype(np.float32)
    if not np.isfinite(result).all():
        raise DomainError("AUDIO_RESAMPLE_FAILED", "Audio resampler returned non-finite samples", 422)
    return np.ascontiguousarray(np.clip(result, -1.0, 1.0))


def resample_mono(samples: np.ndarray, source_rate: int, target_rate: int = FEATURE_SAMPLE_RATE) -> np.ndarray:
    """Deterministically downmix and resample sample-major PCM with PyAV's fixed resampler."""
    if type(source_rate) is not int or type(target_rate) is not int or source_rate <= 0 or target_rate <= 0:
        raise ValueError("Sample rates must be positive integers")
    mono = _float_mono(samples)
    return mono.copy() if source_rate == target_rate else _resample_frames(mono, source_rate, target_rate)


def decode_audio(path: Path, *, stream_index: int | None = None,
                 target_rate: int = FEATURE_SAMPLE_RATE) -> AudioSignal:
    """Decode one local audio stream into mono features while preserving its exact start time."""
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise DomainError("AUDIO_SOURCE_INVALID", "Audio source must be an existing local regular file", 422)
    if type(target_rate) is not int or target_rate <= 0:
        raise ValueError("target_rate must be a positive integer")
    try:
        # Through a handle with every protocol disabled, like the rest of the media layer: a local file only.
        with source.open("rb") as handle, av.open(handle, mode="r", options={"protocol_whitelist": ""}) as container:
            streams = list(container.streams.audio)
            if not streams:
                raise DomainError("AUDIO_STREAM_MISSING", "Media contains no audio stream", 422)
            if stream_index is None:
                stream = streams[0]
            else:
                stream = next((item for item in streams if item.index == stream_index), None)
                if stream is None:
                    raise DomainError("AUDIO_STREAM_MISSING", "Requested audio stream index is unavailable", 422)
            declared = (Fraction(stream.start_time) * Fraction(stream.time_base)
                        if stream.start_time is not None and stream.time_base is not None else None)
            start_time = declared if declared is not None else Fraction(0)
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=target_rate)
            chunks: list[np.ndarray] = []
            first = True
            for decoded in container.decode(stream):
                if first and decoded.pts is not None:
                    # The first decoded sample's presentation time is authoritative: ``stream.start_time`` is
                    # frequently absent, and assuming 0 would misalign every video frame against the audio.
                    start_time = decoded.pts * Fraction(decoded.time_base)
                first = False
                for converted in resampler.resample(decoded):
                    chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
            for converted in resampler.resample(None):
                chunks.append(converted.to_ndarray().reshape(-1).astype(np.float32))
    except DomainError:
        raise
    except (OSError, av.FFmpegError, ValueError) as error:
        raise DomainError("AUDIO_DECODE_FAILED", "Local audio could not be decoded", 422) from error
    if not chunks:
        raise DomainError("AUDIO_STREAM_EMPTY", "Audio stream decoded no samples", 422)
    samples = np.concatenate(chunks)
    if not np.isfinite(samples).all():
        raise DomainError("AUDIO_DECODE_FAILED", "Decoded audio contains non-finite samples", 422)
    return AudioSignal(np.clip(samples, -1.0, 1.0), target_rate, start_time, source)


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    if not 0 <= fmin < fmax <= sample_rate / 2:
        raise ValueError("Mel frequency limits must fit below Nyquist")
    points = np.linspace(_hz_to_mel(np.array([fmin]))[0], _hz_to_mel(np.array([fmax]))[0], n_mels + 2)
    hz = 700.0 * (10.0 ** (points / 2595.0) - 1.0)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    filters = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for index in range(n_mels):
        left, center, right = hz[index:index + 3]
        filters[index] = np.maximum(0.0, np.minimum((frequencies - left) / max(center - left, 1e-12),
                                                    (right - frequencies) / max(right - center, 1e-12)))
    return filters


def extract_log_mel(signal: AudioSignal, *, n_fft: int = DEFAULT_N_FFT,
                    hop_length: int = DEFAULT_HOP_LENGTH, n_mels: int = DEFAULT_MEL_BINS,
                    fmin: float = 55.0, fmax: float = 7600.0) -> AudioFeatures:
    """Compute a deterministic centered log-mel spectrogram without changing source audio."""
    if n_fft < 16 or hop_length <= 0 or n_mels <= 0:
        raise ValueError("FFT, hop and mel-bin counts must be positive and bounded")
    frames = max(1, math.ceil(signal.samples.size / hop_length))
    left = n_fft // 2
    required = (frames - 1) * hop_length + n_fft
    padded = np.pad(signal.samples, (left, max(left, required - (signal.samples.size + left))))
    window = np.hanning(n_fft).astype(np.float32)
    filters = _mel_filterbank(signal.sample_rate, n_fft, n_mels, fmin, fmax)
    mel = np.empty((n_mels, frames), dtype=np.float32)
    chunk_size = 512
    for first in range(0, frames, chunk_size):
        last = min(frames, first + chunk_size)
        block = np.stack([padded[i * hop_length:i * hop_length + n_fft] for i in range(first, last)])
        spectrum = np.fft.rfft(block * window[None, :], n=n_fft, axis=1)
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
        mel[:, first:last] = np.log(np.maximum(filters @ power.T, 1e-10))
    return AudioFeatures(signal, mel, n_fft, hop_length)


def audio_features_from_pcm(samples: np.ndarray, source_rate: int, *, target_rate: int = FEATURE_SAMPLE_RATE,
                            start_time: Fraction = Fraction(0), **mel_options) -> AudioFeatures:
    """Convenience path for caller-supplied replacement audio samples."""
    signal = AudioSignal(resample_mono(samples, source_rate, target_rate), target_rate, start_time)
    return extract_log_mel(signal, **mel_options)
