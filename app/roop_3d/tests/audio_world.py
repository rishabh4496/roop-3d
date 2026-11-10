"""Synthetic speech with KNOWN structure, shared by the director's unit and integration tests.

Not a speech model: three sine-pair "vowels" whose F1/F2 and loudness are chosen so the expected ordering of the acoustic
proxies (open /a/ > rounded > spread /i/ in opening; /i/ widest) is known in advance, separated by silence over low room noise.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from roop_3d.engine.audio_sync.audio_features import AudioSignal

SR = 16_000


def speech(start=Fraction(0), seed=0, seconds=4.0, floor=0.002) -> AudioSignal:
    """/a/ 0.5-1.0 s, /i/ 1.5-2.0 s, an /o/-like vowel 2.5-3.0 s, silence between, low room noise throughout."""
    t = np.arange(int(SR * seconds)) / SR
    x = np.random.default_rng(seed).normal(0, floor, t.size)

    def vowel(t0, t1, f1, a1, f2, a2):
        m = (t >= t0) & (t < t1)
        env = np.sin(np.pi * (t[m] - t0) / (t1 - t0)) ** 0.5
        x[m] += env * (a1 * np.sin(2 * np.pi * f1 * t[m]) + a2 * np.sin(2 * np.pi * f2 * t[m]))

    vowel(0.5, 1.0, 700, 0.30, 1200, 0.08)
    vowel(1.5, 2.0, 300, 0.08, 2400, 0.30)
    vowel(2.5, 3.0, 450, 0.25, 800, 0.05)
    return AudioSignal(x.astype(np.float32), SR, start)
