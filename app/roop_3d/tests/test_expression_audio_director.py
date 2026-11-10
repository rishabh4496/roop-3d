"""ExpressionAndAudioDirector: emotion offsets in psi-space and the audio lock for lip/jaw motion under occlusion.

The audio side is tested on synthetic speech with KNOWN structure (a wide-open /a/, a spread /i/, a rounded /o/-like vowel,
silence, room noise). It asserts the ordering and gating the director relies on, not phoneme recognition, which this module
does not claim: loudness above the noise floor drives opening, the F1 centroid separates open from close vowels, the F2
centroid spreads or rounds the lips as the mouth closes.
"""
from __future__ import annotations

import sys
import wave
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from audio_world import SR, speech  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.editing import (
    AudioVisemeTrack, DirectorSettings, EMOTIONS, EmotionBasis, EmotionOffsets, ExpressionAndAudioDirector)
from roop_3d.engine.audio_sync.audio_features import AudioSignal
from roop_3d.tracking.video_tracker import EXPRESSION_CHANNELS, EXPRESSION_FULL_SCALE, JAW_LEVER_M, JAW_MAX_RAD

C = {name: i for i, name in enumerate(EXPRESSION_CHANNELS)}
FULL = dict(zip(EXPRESSION_CHANNELS, EXPRESSION_FULL_SCALE))
DT = 1 / 30
IPD = DirectorSettings().ipd_m


def jaw_of(mouth_open: float) -> float:
    return min(mouth_open * IPD / JAW_LEVER_M, JAW_MAX_RAD)


@pytest.fixture(scope="module")
def track():
    return AudioVisemeTrack.from_signal(speech())


def neutral() -> np.ndarray:
    return np.zeros(len(EXPRESSION_CHANNELS))


def instant(**kw) -> DirectorSettings:
    return DirectorSettings(ramp_s=0.0, blend_s=0.0, **kw)


# =========================================================================== emotion offsets
def test_zero_offsets_are_an_exact_identity_and_never_mutate_the_input():
    director = ExpressionAndAudioDirector(instant())
    psi = np.array([0.2, 0.05, 0.08, 0.06, 0.3, 0.0, 0.04, -0.02])
    before = psi.copy()
    out = director.direct(0.0, psi, 0.15)
    assert np.array_equal(out.psi, psi) and out.jaw_open_rad == pytest.approx(0.15) and out.jaw_source == "visual"
    assert np.array_equal(psi, before) and out.clamped == () and out.baseline == "measured"


def test_smile_dial_is_a_fraction_of_full_scale_on_the_smile_and_width_channels():
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(smile=0.5)).direct(0.0, neutral(), None)
    assert out.psi[C["smile_a"]] == pytest.approx(0.5 * FULL["smile_a"]) and out.psi[C["smile_b"]] == pytest.approx(0.5 * FULL["smile_b"])
    assert out.psi[C["mouth_width"]] == pytest.approx(0.5 * 0.35 * FULL["mouth_width"])
    assert out.psi[C["mouth_open"]] == 0 and out.psi[C["brow_a"]] == 0 and out.jaw_open_rad == 0     # nothing else moves
    assert out.offsets_applied["smile"] == 0.5 and out.clamped == ()


def test_frown_lowers_corners_and_brows_and_stays_inside_the_rig_limits():
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(frown=1.0)).direct(0.0, neutral(), None)
    assert out.psi[C["smile_a"]] < 0 and out.psi[C["smile_b"]] < 0 and out.psi[C["brow_a"]] < 0 and out.psi[C["mouth_width"]] < 0
    assert out.psi[C["brow_a"]] == pytest.approx(-0.7 * FULL["brow_a"])
    assert out.clamped == ()                                       # -0.7 of full scale is still inside [-0.12, 0.16]
    assert np.all(out.psi >= np.array([0.0, -0.25, -0.12, -0.12, 0.0, 0.0, -0.12, -0.12]) - 1e-12)


def test_surprise_raises_brows_opens_mouth_and_keeps_the_jaw_consistent():
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(surprise=1.0)).direct(0.0, neutral(), 0.0)
    assert out.psi[C["brow_a"]] > FULL["brow_a"] * 0.99 and out.psi[C["mouth_open"]] == pytest.approx(0.45 * FULL["mouth_open"])
    assert out.jaw_open_rad == pytest.approx(jaw_of(out.psi[C["mouth_open"]]), rel=1e-6)              # jaw follows the mouth


def test_offsets_add_to_the_measured_performance_and_clamp_with_a_report():
    psi = neutral()
    psi[C["smile_a"]] = psi[C["smile_b"]] = 0.20                      # already close to the 0.22 limit
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(smile=1.0)).direct(0.0, psi, None)
    assert out.psi[C["smile_a"]] == pytest.approx(0.22) and set(out.clamped) >= {"smile_a", "smile_b"}
    psi2 = neutral()
    psi2[C["mouth_open"]] = 0.52
    out2 = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(surprise=1.0)).direct(0.0, psi2, jaw_of(0.52))
    assert out2.psi[C["mouth_open"]] == pytest.approx(0.55) and "mouth_open" in out2.clamped
    assert out2.jaw_open_rad == pytest.approx(jaw_of(0.55), rel=1e-6)    # the jaw follows the CLAMPED opening, not the wish


def test_eye_openness_suppresses_blinks_or_squints_and_admits_what_the_rig_cannot_do():
    blinking = neutral()
    blinking[C["blink_a"]] = 0.8
    blinking[C["blink_b"]] = 0.6
    wide = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(eye_openness=1.0)).direct(0.0, blinking, None)
    assert wide.psi[C["blink_a"]] == 0 and wide.psi[C["blink_b"]] == 0
    assert wide.psi[C["brow_a"]] == pytest.approx(0.25 * FULL["brow_a"])                  # the only cue for "wider than neutral"
    assert any("cannot open beyond the neutral" in note for note in wide.limits)
    half = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(eye_openness=0.5)).direct(0.0, blinking, None)
    assert half.psi[C["blink_a"]] == pytest.approx(0.4) and half.psi[C["blink_b"]] == pytest.approx(0.3)
    squint = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(eye_openness=-1.0)).direct(0.0, neutral(), None)
    assert squint.psi[C["blink_a"]] == pytest.approx(0.45) and squint.psi[C["blink_b"]] == pytest.approx(0.45)
    stronger = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(eye_openness=-0.2)).direct(0.0, blinking, None)
    assert stronger.psi[C["blink_a"]] == pytest.approx(0.8)               # a real blink is never reduced by a squint dial


def test_directed_psi_converts_to_the_rigs_expression_params():
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(smile=0.4, surprise=0.5)).direct(0.0, neutral(), None)
    p = out.as_expression_params()
    assert p.smile == pytest.approx(0.5 * (out.psi[C["smile_a"]] + out.psi[C["smile_b"]])) and p.mouth_open == out.psi[C["mouth_open"]]
    assert p.brow_a == out.psi[C["brow_a"]] and p.clamped().mouth_open == pytest.approx(p.mouth_open)


def test_offsets_ease_over_ramp_s_and_seeking_backwards_jumps():
    director = ExpressionAndAudioDirector(DirectorSettings(ramp_s=0.15))
    director.direct(0.0, neutral(), None)
    director.set_offsets(EmotionOffsets(smile=1.0))
    values = [director.direct((i + 1) * DT, neutral(), None).offsets_applied["smile"] for i in range(90)]
    assert values[0] == pytest.approx(1 - np.exp(-DT / 0.15), abs=1e-9)          # first-order lag with tau = ramp_s
    assert np.all(np.diff(values) > 0) and values[9] == pytest.approx(1 - np.exp(-10 * DT / 0.15)) and values[9] < 0.9 and values[-1] > 0.9999
    director.set_offsets(EmotionOffsets())
    director.direct(0.0, neutral(), None)                                         # a seek back in time resets the easing
    assert director.direct(0.0, neutral(), None).offsets_applied["smile"] == 0.0
    assert ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(smile=1.0)).direct(0.0, neutral(), None).offsets_applied["smile"] == 1.0


def test_unmeasured_frames_start_from_neutral_and_say_so_and_bad_layouts_are_rejected():
    out = ExpressionAndAudioDirector(instant(), offsets=EmotionOffsets(smile=1.0)).direct(0.0, None, None)
    assert out.baseline.startswith("neutral") and out.jaw_source == "neutral" and out.psi[C["smile_a"]] > 0
    director = ExpressionAndAudioDirector(instant())
    with pytest.raises(DomainError, match="EmotionBasis"):
        director.direct(0.0, np.zeros(50), None)                                  # a FLAME psi needs its own basis
    with pytest.raises(DomainError):
        director.direct(0.0, np.full(8, np.nan), None)
    with pytest.raises(DomainError):
        director.direct(float("nan"), neutral(), None)
    with pytest.raises(ValueError):
        EmotionOffsets(smile=1.5)
    with pytest.raises(ValueError):
        ExpressionAndAudioDirector(DirectorSettings(lock_start=0.9, lock_full=0.5))


def test_a_custom_basis_extends_the_director_to_other_psi_layouts():
    basis = EmotionBasis(channels=("a", "b", "c"), directions={"smile": np.array([1.0, 0, 0]), "frown": np.array([-1.0, 0, 0]),
                                                                "surprise": np.array([0, 1.0, 0])},
                         lower=np.zeros(3), upper=np.ones(3), full_scale=np.ones(3))
    out = ExpressionAndAudioDirector(instant(), basis=basis, offsets=EmotionOffsets(smile=0.3, surprise=1.0)).direct(0.0, np.zeros(3), None)
    assert out.psi == pytest.approx([0.3, 1.0, 0.0]) and out.jaw_open_rad == 0.0
    with pytest.raises(DomainError):
        out.as_expression_params()                                                # only the default layout maps to ExpressionParams
    assert EMOTIONS == ("smile", "frown", "surprise", "eye_openness")


# =========================================================================== audio proxies
def test_viseme_track_orders_open_spread_and_rounded_vowels_and_goes_quiet_in_silence(track):
    assert track.speech_detected and track.dynamic_range_db > 25
    a, i, o = track.at(0.75), track.at(1.75), track.at(2.75)
    assert a.opening > 0.9 and i.opening < 0.5 < a.opening and o.opening < a.opening      # /a/ is the widest-open vowel
    assert i.width > 0.4 and a.width == pytest.approx(0.0, abs=0.1) and o.width < i.width # /i/ spreads, /a/ does not
    assert a.voiced > 0.9 and i.voiced > 0.9
    for silent in (0.2, 1.25, 2.25, 3.6):
        s = track.at(silent)
        assert s.opening < 0.08 and abs(s.width) < 0.05 and s.voiced < 0.1
    assert track.at(-1.0).voiced == 0 and track.at(99.0).voiced == 0 and track.at(-1.0).opening == 0       # outside the clip


def test_viseme_track_respects_the_signals_start_time_and_hop():
    late = AudioVisemeTrack.from_signal(speech(start=Fraction(2)))
    assert late.start_s == 2.0 and late.hop_s == pytest.approx(0.01)
    assert late.at(2.75).opening > 0.9 and late.at(0.75).voiced == 0           # the clip starts at t = 2 s of the media clock
    assert late.at(2.75).opening == pytest.approx(AudioVisemeTrack.from_signal(speech()).at(0.75).opening, abs=1e-6)


def test_noise_only_audio_is_not_speech_and_drives_nothing():
    noise = AudioSignal(np.random.default_rng(1).normal(0, 0.01, SR * 2).astype(np.float32), SR, Fraction(0))
    quiet = AudioVisemeTrack.from_signal(noise)
    assert not quiet.speech_detected and quiet.opening.max() == 0 and quiet.voiced.max() == 0


def test_viseme_track_decodes_from_a_media_file(tmp_path):
    signal = speech()
    path = tmp_path / "talk.wav"
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes((np.clip(signal.samples, -1, 1) * 32767).astype("<i2").tobytes())
    decoded = AudioVisemeTrack.from_media(path)
    assert decoded.speech_detected and decoded.at(0.75).opening > 0.9 and decoded.at(1.75).width > 0.4
    with pytest.raises(DomainError):
        AudioVisemeTrack.from_media(tmp_path / "missing.wav")


# =========================================================================== audio lock under occlusion
def run(director, seconds, *, occlusion, visual, jaw=None):
    out = []
    for i in range(int(seconds * 30)):
        t = i * DT
        psi = neutral()
        psi[C["mouth_open"]] = visual(t)
        out.append(director.direct(t, psi, jaw(t) if jaw else None, mouth_occlusion=occlusion(t)))
    return out


def test_lock_is_off_by_default_and_when_the_mouth_is_visible(track):
    held = lambda t: 0.12
    default = ExpressionAndAudioDirector(DirectorSettings(), audio=track)         # audio_lock_enabled defaults to False
    assert all(o.audio_weight == 0 and o.psi[C["mouth_open"]] == 0.12 for o in run(default, 2.0, occlusion=lambda t: 1.0, visual=held))
    armed = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), audio=track)
    clear = run(armed, 2.0, occlusion=lambda t: 0.0, visual=held)
    assert all(o.audio_weight == 0 and o.psi[C["mouth_open"]] == 0.12 and o.jaw_source == "visual" for o in clear)


def test_lock_is_unavailable_without_usable_audio_and_says_why(track):
    for audio, reason in ((None, "no audio"), (AudioVisemeTrack.from_signal(AudioSignal(
            np.random.default_rng(2).normal(0, 0.01, SR * 2).astype(np.float32), SR, Fraction(0))), "no speech")):
        out = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), audio=audio).direct(
            0.5, neutral(), None, mouth_occlusion=1.0)
        assert out.audio_weight == 0 and out.jaw_source != "audio" and any(reason in note for note in out.limits)
    no_lips = EmotionBasis(channels=("a",), directions={"smile": np.zeros(1), "frown": np.zeros(1), "surprise": np.zeros(1)},
                           lower=np.zeros(1), upper=np.ones(1), full_scale=np.ones(1))
    out = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), basis=no_lips, audio=track).direct(
        0.5, np.zeros(1), None, mouth_occlusion=1.0)
    assert out.audio_weight == 0 and any("no lip channels" in note for note in out.limits)


def test_heavy_occlusion_hands_lip_and_jaw_to_the_audio_and_follows_the_speech(track):
    director = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True, blend_s=0.1), audio=track)
    visual = lambda t: 0.3 * track.at(t).opening if t < 2.0 else 0.1            # tracked mouth until 2 s, then the hand: held at 0.1
    frames = run(director, 4.0, occlusion=lambda t: 0.0 if t < 2.0 else 0.95, visual=visual)
    # engagement is gradual (blend_s), complete within half a second, and complete only because the mouth is occluded
    first_occluded = next(i for i, o in enumerate(frames) if o.audio_weight > 0)
    assert frames[first_occluded - 1].audio_weight == 0 and frames[first_occluded].audio_weight < 0.5
    assert frames[int(2.6 * 30)].audio_weight > 0.99 and frames[int(2.6 * 30)].jaw_source == "audio"
    # while locked, the mouth follows the SPEECH (not the held 0.1) and the jaw follows the mouth
    locked = frames[int(2.6 * 30):]
    mouth = np.array([o.psi[C["mouth_open"]] for o in locked])
    heard = np.array([track.at(i * DT + 2.6).opening for i in range(len(locked))])
    assert np.corrcoef(mouth, heard)[0, 1] > 0.95 and mouth.max() > 0.15 and mouth.min() < 0.02
    assert all(o.jaw_open_rad == pytest.approx(jaw_of(o.psi[C["mouth_open"]]), abs=1e-6) for o in locked)
    # the spread /i/ at 1.75 s is the visible part; during the locked /o/-like vowel the width channel follows the audio too
    assert max(abs(o.psi[C["mouth_width"]]) for o in locked) > 0.005


def test_the_lock_calibrates_to_the_actors_visible_mouth_and_defaults_until_it_can():
    track_ = AudioVisemeTrack.from_signal(speech())
    for gain in (0.2, 0.45):
        director = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), audio=track_)
        # visible, talking actor whose mouth opens `gain` per unit of the audio proxy
        for i in range(int(2.2 * 30)):
            t = i * DT
            psi = neutral()
            psi[C["mouth_open"]] = gain * track_.at(t).opening
            out = director.direct(t, psi, None, mouth_occlusion=0.0)
        trained = out
        assert trained.lip_calibration == "calibrated"
        director.reset()                                           # seek back to the /a/ at 0.75 s; reset keeps the calibration
        occluded = director.direct(0.75, neutral(), None, mouth_occlusion=1.0)                # ... with the mouth now hidden
        assert occluded.audio_weight == 1.0 and occluded.lip_calibration == "calibrated"
        assert occluded.psi[C["mouth_open"]] == pytest.approx(gain * track_.at(0.75).opening, abs=0.06)
    fresh = ExpressionAndAudioDirector(DirectorSettings(audio_lock_enabled=True), audio=track_)
    out = fresh.direct(0.75, neutral(), None, mouth_occlusion=1.0)
    assert out.lip_calibration == "default" and any("default gain" in note for note in out.limits)


def test_offsets_still_apply_on_top_of_an_audio_locked_mouth(track):
    director = ExpressionAndAudioDirector(instant(audio_lock_enabled=True), audio=track, offsets=EmotionOffsets(surprise=1.0))
    out = director.direct(0.75, neutral(), None, mouth_occlusion=1.0)
    plain = ExpressionAndAudioDirector(instant(audio_lock_enabled=True), audio=track).direct(0.75, neutral(), None, mouth_occlusion=1.0)
    assert out.psi[C["mouth_open"]] == pytest.approx(min(plain.psi[C["mouth_open"]] + 0.45 * 0.55, 0.55))
    assert out.psi[C["brow_a"]] > plain.psi[C["brow_a"]] and out.jaw_open_rad >= plain.jaw_open_rad


def test_reset_and_audio_offset_and_swapping_audio(track):
    s = instant(audio_lock_enabled=True, audio_offset_s=0.25)
    director = ExpressionAndAudioDirector(s, audio=track)
    shifted = director.direct(0.5, neutral(), None, mouth_occlusion=1.0)          # hears 0.75 s: the /a/
    assert shifted.psi[C["mouth_open"]] > 0.2
    director.set_audio(None)
    assert director.direct(0.6, neutral(), None, mouth_occlusion=1.0).audio_weight == 0
    director.set_audio(track)
    director.reset()
    assert director.direct(0.5, neutral(), None, mouth_occlusion=0.0).audio_weight == 0
