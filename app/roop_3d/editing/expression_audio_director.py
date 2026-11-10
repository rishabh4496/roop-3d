# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""ExpressionAndAudioDirector: user emotion offsets in psi-space, and an audio lock for lip/jaw motion under occlusion.

Inspired by NED (Neural Emotion Director) and FaceFormer. What is and is not borrowed:

* NED manipulates emotion as a *direction in expression space* before rendering. So does this, with FIXED linear directions
  (smile, frown, surprise, eye openness) instead of NED's learned emotion translator. There is no learned model here.
* FaceFormer predicts face motion from speech with a trained transformer. No such weights are installed in this project, so
  this does NOT synthesise lip-sync. It maps measurable speech acoustics (loudness above the noise floor, and the balance of
  first- vs second-formant energy) to mouth opening / width, calibrated per actor against the actor's own *visible* mouth
  motion. It is meant for the few frames where a hand, microphone or mask hides the mouth and tracking would otherwise hold
  or lose it; it is not a replacement for visual tracking and it is opt-in.

psi-space here is the tracker's eight semantic channels (``tracking.video_tracker.EXPRESSION_CHANNELS``: mouth_open,
mouth_width, smile_a, smile_b, blink_a, blink_b, brow_a, brow_b), in the units of ``reenactment.ExpressionParams``. A FLAME
50-d psi needs a caller-supplied :class:`EmotionBasis`; FLAME weights are not installed, so none is shipped.

Order inside one frame: (1) audio lock replaces the lip channels and the jaw where the mouth is occluded, (2) emotion offsets
are added on top (so a surprise offset still opens the mouth during an audio-locked passage), (3) everything is clamped to the
rig's channel limits and the jaw is kept consistent with the final mouth opening.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.engine.audio_sync.audio_features import AudioSignal, decode_audio
from roop_3d.tracking.video_tracker import EXPRESSION_CHANNELS, EXPRESSION_FULL_SCALE, JAW_LEVER_M, JAW_MAX_RAD

EMOTIONS = ("smile", "frown", "surprise", "eye_openness")
# Channel limits of the rig (reenactment.ExpressionParams.clamped), in EXPRESSION_CHANNELS order.
_LOWER = np.array((0.0, -0.25, -0.12, -0.12, 0.0, 0.0, -0.12, -0.12))
_UPPER = np.array((0.55, 0.30, 0.22, 0.22, 1.0, 1.0, 0.16, 0.16))
SQUINT_MAX = 0.45                 # a full "narrow eyes" offset closes the lids this far (never a blink)
WIDE_EYE_BROW_CUE = 0.25          # the rig has no upper-lid raise: wide eyes are cued by this share of a brow raise


# =========================================================================== emotion basis
@dataclass(frozen=True)
class EmotionBasis:
    """Linear emotion directions over a psi channel layout, in fractions of each channel's full scale per +1 of the dial."""
    channels: tuple[str, ...]
    directions: dict[str, np.ndarray]              # smile / frown / surprise -> (D,)
    lower: np.ndarray
    upper: np.ndarray
    full_scale: np.ndarray
    blink_channels: tuple[int, ...] = ()           # indices whose value is eyelid CLOSURE (0 open .. 1 closed)
    brow_channels: tuple[int, ...] = ()
    lip_open: int | None = None
    lip_width: int | None = None

    @classmethod
    def default(cls) -> "EmotionBasis":
        index = {name: i for i, name in enumerate(EXPRESSION_CHANNELS)}

        def direction(**weights: float) -> np.ndarray:
            d = np.zeros(len(EXPRESSION_CHANNELS))
            for name, value in weights.items():
                d[index[name]] = value
            return d

        return cls(
            channels=tuple(EXPRESSION_CHANNELS),
            directions={
                "smile": direction(smile_a=1.0, smile_b=1.0, mouth_width=0.35),
                "frown": direction(smile_a=-0.5, smile_b=-0.5, brow_a=-0.7, brow_b=-0.7, mouth_width=-0.2),
                "surprise": direction(brow_a=1.0, brow_b=1.0, mouth_open=0.45),
            },
            lower=_LOWER.copy(), upper=_UPPER.copy(), full_scale=np.asarray(EXPRESSION_FULL_SCALE, np.float64),
            blink_channels=(index["blink_a"], index["blink_b"]), brow_channels=(index["brow_a"], index["brow_b"]),
            lip_open=index["mouth_open"], lip_width=index["mouth_width"])

    @property
    def size(self) -> int:
        return len(self.channels)


class EmotionOffsets(Contract):
    smile: float = Field(default=0.0, ge=-1.0, le=1.0)
    frown: float = Field(default=0.0, ge=-1.0, le=1.0)
    surprise: float = Field(default=0.0, ge=-1.0, le=1.0)
    eye_openness: float = Field(default=0.0, ge=-1.0, le=1.0, description="+ keeps the eyes open / wide, - narrows them")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in EMOTIONS], np.float64)


class DirectorSettings(Contract):
    ramp_s: float = Field(default=0.15, ge=0.0, le=5.0, description="Time constant for easing toward a changed offset (0 = jump)")
    audio_lock_enabled: bool = False
    lock_start: float = Field(default=0.45, ge=0.0, le=1.0, description="Mouth occlusion where the audio lock starts to engage")
    lock_full: float = Field(default=0.80, gt=0.0, le=1.0, description="Mouth occlusion where audio fully drives lip and jaw")
    blend_s: float = Field(default=0.10, ge=0.0, le=2.0)
    audio_offset_s: float = Field(default=0.0, ge=-0.5, le=0.5, description="Shift of the audio against the video clock")
    ipd_m: float = Field(default=0.063, gt=0.03, lt=0.1)
    calibration_min_samples: int = Field(default=24, ge=4, le=1000)
    calibration_forgetting: float = Field(default=0.985, gt=0.5, lt=1.0)
    calibration_visible_max: float = Field(default=0.10, ge=0.0, le=0.5, description="Mouth occlusion below which a frame trains the calibration")
    default_open_gain: float = Field(default=0.30, gt=0.0, le=0.55, description="mouth_open at full audio opening, before calibration")
    width_gain: float = Field(default=0.10, ge=0.0, le=0.3)


# =========================================================================== audio -> viseme proxies
@dataclass(frozen=True)
class AudioViseme:
    opening: float            # 0..1, loudness above the noise floor x vowel openness
    width: float              # -1 (rounded, /u o/) .. +1 (spread, /i e/)
    voiced: float             # 0..1 speech activity


@dataclass(frozen=True)
class AudioVisemeTrack:
    """Frame-rate speech-acoustics proxies for mouth motion, seekable by time. Heuristic - see the module docstring."""
    start_s: float
    hop_s: float
    opening: np.ndarray
    width: np.ndarray
    voiced: np.ndarray
    dynamic_range_db: float
    speech_detected: bool

    @classmethod
    def from_media(cls, path: Path | str, **kwargs) -> "AudioVisemeTrack":
        return cls.from_signal(decode_audio(Path(path), target_rate=16_000), **kwargs)

    @classmethod
    def from_signal(cls, signal: AudioSignal, *, hop_s: float = 0.01, window_s: float = 0.032,
                    min_dynamic_range_db: float = 12.0) -> "AudioVisemeTrack":
        rate = int(signal.sample_rate)
        hop, win = max(1, round(hop_s * rate)), max(16, round(window_s * rate))
        x = np.asarray(signal.samples, np.float64)
        padded = np.concatenate([np.zeros(win // 2), x, np.zeros(win)])
        count = max(1, 1 + (len(x) - 1) // hop)
        n_fft = 1 << int(math.ceil(math.log2(win)))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / rate)
        # Formant proxies: power-weighted spectral centroids of the F1 region (jaw height: low F1 = close vowel) and of the
        # F2 region (lip spreading: high F2 = /i e/, low F2 = rounded /u o/), plus the fricative band (s, f, sh: little jaw).
        f1_band = (freqs >= 200.0) & (freqs < 1000.0)
        f2_band = (freqs >= 900.0) & (freqs < 3000.0)
        hi_band = (freqs >= 3000.0) & (freqs < min(7000.0, rate / 2.0))
        taper = np.hanning(win)
        level = np.empty(count)
        feat = np.empty((count, 5))                    # F1 centroid (Hz), F2 centroid (Hz), F1 power, F2 power, fricative power
        view = np.lib.stride_tricks.sliding_window_view(padded, win)
        for start in range(0, count, 2048):
            idx = np.arange(start, min(count, start + 2048)) * hop
            frames = view[idx] * taper
            level[start:start + len(idx)] = 10.0 * np.log10((frames ** 2).mean(axis=1) + 1e-10)
            power = np.abs(np.fft.rfft(frames, n_fft, axis=1)) ** 2
            p1, p2, ph = power[:, f1_band].sum(axis=1), power[:, f2_band].sum(axis=1), power[:, hi_band].sum(axis=1)
            c1 = (power[:, f1_band] * freqs[f1_band]).sum(axis=1) / np.maximum(p1, 1e-18)
            c2 = (power[:, f2_band] * freqs[f2_band]).sum(axis=1) / np.maximum(p2, 1e-18)
            feat[start:start + len(idx)] = np.stack([c1, c2, p1, p2, ph], axis=1)
        floor, peak = float(np.percentile(level, 10)), float(np.percentile(level, 95))
        dynamic = peak - floor
        detected = dynamic >= min_dynamic_range_db
        if not detected:
            zeros = np.zeros(count)
            return cls(float(signal.start_time), hop / rate, zeros, zeros.copy(), zeros.copy(), dynamic, False)
        vad = _smooth(level, floor + 8.0, floor + 14.0)
        c1, c2 = feat[:, 0], feat[:, 1]
        total = feat[:, 2] + feat[:, 3] + feat[:, 4] + 1e-12
        loud = _smooth(level, floor + 8.0, peak - 2.0) ** 0.8
        vowel = 0.35 + 0.65 * np.clip((c1 - 300.0) / 400.0, 0.0, 1.0)          # F1 300 Hz (close) .. 700 Hz (open)
        opening = loud * vowel * (1.0 - 0.6 * feat[:, 4] / total)
        # Open vowels do not spread or round the lips; the F2 balance only moves the width as the mouth closes.
        width = vad * np.clip((c2 - 1450.0) / 700.0, -1.0, 1.0) * (1.0 - opening)
        attack, release = 1.0 - math.exp(-hop / rate / 0.03), 1.0 - math.exp(-hop / rate / 0.09)
        smooth_open = np.empty(count)
        state = 0.0
        for i, value in enumerate(opening):
            state += (value - state) * (attack if value > state else release)
            smooth_open[i] = state
        smooth_width = np.empty(count)
        state = 0.0
        for i, value in enumerate(width):
            state += (value - state) * attack
            smooth_width[i] = state
        return cls(float(signal.start_time), hop / rate, np.clip(smooth_open, 0.0, 1.0), np.clip(smooth_width, -1.0, 1.0),
                   vad, dynamic, True)

    def at(self, t: float) -> AudioViseme:
        position = (t - self.start_s) / self.hop_s
        if position < -0.5 or position > len(self.opening) - 0.5:
            return AudioViseme(0.0, 0.0, 0.0)
        low = int(np.clip(math.floor(position), 0, len(self.opening) - 1))
        high = min(low + 1, len(self.opening) - 1)
        w = float(np.clip(position - low, 0.0, 1.0))
        lerp = lambda a: float(a[low] * (1.0 - w) + a[high] * w)
        return AudioViseme(lerp(self.opening), lerp(self.width), lerp(self.voiced))


def _smooth(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((np.asarray(x, np.float64) - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class _OpenCalibrator:
    """Exponentially forgetting least squares of the actor's VISIBLE mouth_open against the audio opening proxy."""

    def __init__(self, settings: DirectorSettings) -> None:
        self.s = settings
        self.reset()

    def reset(self) -> None:
        self.n = 0.0
        self.sx = self.sy = self.sxx = self.sxy = 0.0
        self.samples = 0

    def add(self, audio_open: float, visual_open: float) -> None:
        k = self.s.calibration_forgetting
        self.n, self.sx, self.sy = k * self.n + 1.0, k * self.sx + audio_open, k * self.sy + visual_open
        self.sxx, self.sxy = k * self.sxx + audio_open ** 2, k * self.sxy + audio_open * visual_open
        self.samples += 1

    def fit(self) -> tuple[float, float, str]:
        """(gain, offset, source). Falls back to the default gain until enough varied samples exist."""
        if self.samples >= self.s.calibration_min_samples:
            mean_x, mean_y = self.sx / self.n, self.sy / self.n
            var_x = self.sxx / self.n - mean_x ** 2
            if var_x > 1e-3:
                gain = (self.sxy / self.n - mean_x * mean_y) / var_x
                if gain > 0.02:
                    return float(np.clip(gain, 0.02, 0.55)), float(np.clip(mean_y - gain * mean_x, 0.0, 0.08)), "calibrated"
        return self.s.default_open_gain, 0.0, "default"


# =========================================================================== director
@dataclass(frozen=True)
class DirectedExpression:
    psi: np.ndarray                                # (D,) directed channels
    jaw_open_rad: float
    offsets_applied: dict[str, float]              # eased dial values actually in effect
    audio_weight: float                            # 0 = fully visual, 1 = audio drives lip and jaw
    jaw_source: str                                # "visual" | "audio_blend" | "audio" | "neutral"
    lip_calibration: str                           # "n/a" | "default" | "calibrated"
    clamped: tuple[str, ...] = ()
    limits: tuple[str, ...] = ()
    baseline: str = "measured"

    def as_expression_params(self):
        """The rig's seven-channel ``ExpressionParams`` (smile = mean of the two sides)."""
        from roop_3d.reconstruction.reenactment import ExpressionParams
        v = dict(zip(EXPRESSION_CHANNELS, self.psi)) if len(self.psi) == len(EXPRESSION_CHANNELS) else None
        if v is None:
            raise DomainError("EXPRESSION_BASIS_UNSUPPORTED", "ExpressionParams needs the default eight-channel psi layout", 422)
        return ExpressionParams(mouth_open=v["mouth_open"], mouth_width=v["mouth_width"], smile=0.5 * (v["smile_a"] + v["smile_b"]),
                                blink_a=v["blink_a"], blink_b=v["blink_b"], brow_a=v["brow_a"], brow_b=v["brow_b"])


class ExpressionAndAudioDirector:
    def __init__(self, settings: DirectorSettings | None = None, *, basis: EmotionBasis | None = None,
                 audio: AudioVisemeTrack | None = None, offsets: EmotionOffsets | None = None) -> None:
        self.settings = settings or DirectorSettings()
        if self.settings.lock_start >= self.settings.lock_full:
            raise ValueError("lock_start must be below lock_full")
        self.basis = basis or EmotionBasis.default()
        self.audio = audio
        self._requested = (offsets or EmotionOffsets()).vector()
        self._calibrator = _OpenCalibrator(self.settings)
        self.reset()
        self._applied = self._requested.copy()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        """New shot: the eased state restarts at the requested offsets and the audio lock disengages."""
        self._last_t: float | None = None
        self._applied = self._requested.copy() if hasattr(self, "_requested") else np.zeros(len(EMOTIONS))
        self._weight = 0.0

    def set_offsets(self, offsets: EmotionOffsets) -> None:
        self._requested = offsets.vector()

    def set_audio(self, audio: AudioVisemeTrack | None) -> None:
        self.audio = audio

    # ------------------------------------------------------------------ helpers
    def _jaw_from_open(self, mouth_open: float) -> float:
        return float(np.clip(mouth_open * self.settings.ipd_m / JAW_LEVER_M, 0.0, JAW_MAX_RAD))

    def _ease(self, current, goal, dt: float | None, tau: float):
        if dt is None or tau <= 0.0:
            return np.asarray(goal, np.float64).copy() if np.ndim(goal) else float(goal)
        a = 1.0 - math.exp(-max(dt, 0.0) / tau)
        return current + (np.asarray(goal) - current) * a if np.ndim(goal) else float(current + (goal - current) * a)

    # ------------------------------------------------------------------ main
    def direct(self, timestamp: float, psi: np.ndarray | None, jaw_open_rad: float | None = None, *,
               mouth_occlusion: float | None = None, offsets: EmotionOffsets | None = None) -> DirectedExpression:
        """Direct one frame. ``psi`` None = nothing measured (offsets then apply to a neutral face and the result says so).

        ``mouth_occlusion`` is the share of the mouth region hidden by an occluder (0..1; see
        ``StrandAwareOcclusionCompositor`` -> ``OcclusionResult.occlusion_in``). ``None`` means unknown and is treated as visible.
        """
        s, basis = self.settings, self.basis
        if not np.isfinite(timestamp):
            raise DomainError("DIRECTOR_INPUT_INVALID", "timestamp must be finite", 422)
        dt = None if self._last_t is None or timestamp < self._last_t or timestamp - self._last_t > 1.0 else timestamp - self._last_t
        self._last_t = float(timestamp)
        limits: list[str] = []
        baseline = "measured"
        if psi is None:
            vector, baseline = np.zeros(basis.size), "neutral (no measurement)"
        else:
            vector = np.asarray(psi, np.float64).reshape(-1)
            if vector.shape != (basis.size,) or not np.isfinite(vector).all():
                raise DomainError("EXPRESSION_BASIS_UNSUPPORTED",
                                  f"psi has {vector.size} channels but the emotion basis has {basis.size}; "
                                  "pass an EmotionBasis for this layout", 422)
            vector = vector.copy()
        visual_open = float(vector[basis.lip_open]) if basis.lip_open is not None else 0.0
        visual_jaw = float(jaw_open_rad) if jaw_open_rad is not None and np.isfinite(jaw_open_rad) else self._jaw_from_open(visual_open)
        occlusion = 0.0 if mouth_occlusion is None else float(np.clip(mouth_occlusion, 0.0, 1.0))

        # (1) audio lock on lip + jaw
        jaw, jaw_source, calibration, target_weight = visual_jaw, "visual" if psi is not None else "neutral", "n/a", 0.0
        track = self.audio
        can_lock = (s.audio_lock_enabled and track is not None and basis.lip_open is not None and track.speech_detected)
        if s.audio_lock_enabled and not can_lock:
            reason = ("no audio supplied" if track is None else "no speech detected in the audio" if not track.speech_detected
                      else "this emotion basis has no lip channels")
            limits.append(f"audio lock requested but unavailable: {reason}")
        if can_lock:
            heard = track.at(timestamp + s.audio_offset_s)
            if occlusion <= s.calibration_visible_max and psi is not None and heard.voiced > 0.5:
                self._calibrator.add(heard.opening, visual_open)
            gain, offset, calibration = self._calibrator.fit()
            if calibration == "default":
                limits.append("lip amplitude uses the default gain: too little visible speech to calibrate on this actor")
            audio_open = float(np.clip(offset + gain * heard.opening, 0.0, _UPPER[0])) if heard.voiced > 0.05 else 0.0
            target_weight = float(_smooth(np.array([occlusion]), s.lock_start, s.lock_full)[0])
        self._weight = float(np.clip(self._ease(self._weight, target_weight, dt, s.blend_s), 0.0, 1.0))
        w = self._weight
        if can_lock and w > 1e-3:
            vector[basis.lip_open] = (1.0 - w) * vector[basis.lip_open] + w * audio_open
            if basis.lip_width is not None:
                vector[basis.lip_width] = (1.0 - w) * vector[basis.lip_width] + w * (s.width_gain * heard.width)
            jaw = (1.0 - w) * visual_jaw + w * self._jaw_from_open(audio_open)
            jaw_source = "audio" if w >= 0.99 else "audio_blend"
            limits.append("audio drives mouth opening/width from acoustics (loudness, vowel brightness); it is not phoneme-level lip-sync")

        # (2) emotion offsets, eased
        requested = (offsets.vector() if offsets is not None else self._requested)
        self._applied = np.asarray(self._ease(self._applied, requested, dt, s.ramp_s), np.float64)
        applied = dict(zip(EMOTIONS, (float(x) for x in self._applied)))
        before_open = float(vector[basis.lip_open]) if basis.lip_open is not None else 0.0
        delta = np.zeros(basis.size)
        for name in ("smile", "frown", "surprise"):
            if abs(applied[name]) > 0:
                delta += applied[name] * basis.directions[name] * basis.full_scale
        vector = vector + delta
        openness = float(np.clip(applied["eye_openness"] + 0.5 * max(applied["surprise"], 0.0), -1.0, 1.0))
        if openness > 0 and basis.blink_channels:
            for i in basis.blink_channels:
                vector[i] *= (1.0 - openness)
            if basis.brow_channels:
                for i in basis.brow_channels:
                    vector[i] += WIDE_EYE_BROW_CUE * openness * basis.full_scale[i]
            limits.append("eyes cannot open beyond the neutral lid position: eye_openness>0 suppresses blinks and adds a brow cue")
        elif openness < 0 and basis.blink_channels:
            for i in basis.blink_channels:
                vector[i] = max(vector[i], -openness * SQUINT_MAX)

        # (3) clamp, keep the jaw consistent with the final mouth opening
        clamped_vector = np.clip(vector, basis.lower, basis.upper)
        clamped = tuple(basis.channels[i] for i in np.nonzero(np.abs(clamped_vector - vector) > 1e-9)[0])
        if basis.lip_open is not None:
            # Only the mouth opening the OFFSETS added (or the clamp removed) moves the jaw here; the visual/audio jaw is kept.
            jaw = float(np.clip(jaw + (float(clamped_vector[basis.lip_open]) - before_open) * s.ipd_m / JAW_LEVER_M,
                                0.0, JAW_MAX_RAD))
        return DirectedExpression(clamped_vector, jaw, applied, w, jaw_source, calibration, clamped, tuple(dict.fromkeys(limits)), baseline)

    def direct_tracked(self, frame, *, mouth_occlusion: float | None = None,
                       offsets: EmotionOffsets | None = None) -> DirectedExpression:
        """Convenience for a ``tracking.TrackedHeadFrame`` (uses its timestamp, psi and jaw)."""
        return self.direct(frame.timestamp, frame.psi, frame.jaw_open_rad, mouth_occlusion=mouth_occlusion, offsets=offsets)
