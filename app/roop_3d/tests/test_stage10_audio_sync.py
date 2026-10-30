"""Stage 10 audio timing, silence gating and safe lip-sync integration tests."""
from __future__ import annotations

from fractions import Fraction
import hashlib
from pathlib import Path
import wave

import numpy as np
import pytest

from roop_3d.compositor.mouth import composite_mouth_region, lower_face_soft_mask
from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.compositor.temporal import TemporalSmoother
from roop_3d.engine.audio_sync.audio_alignment import (
    frame_sample_index,
    mel_window_for_audio_window,
    window_for_frame,
)
from roop_3d.engine.audio_sync.audio_features import (
    AudioSignal,
    audio_features_from_pcm,
    decode_audio,
    extract_log_mel,
    resample_mono,
)
from roop_3d.engine.audio_sync.lip_sync_adapter import (
    LipSyncAdapter,
    LipSyncFrameInput,
    LipSyncGate,
)
from roop_3d.engine.volumetric.session import ReferencePipeline


@pytest.mark.parametrize("fps", [Fraction(24000, 1001), Fraction(25, 1),
                                  Fraction(30000, 1001), Fraction(60, 1)])
def test_pts_to_sample_windows_are_exact_for_common_frame_rates(fps: Fraction):
    rate = 16_000
    time_base = 1 / fps
    samples = np.zeros(80_000, np.float32)
    frame_index = 50
    exact_center = Fraction(frame_index) * time_base * rate
    pulse_sample = exact_center.numerator // exact_center.denominator + 23
    samples[pulse_sample] = 0.9
    signal = AudioSignal(samples, rate)

    for frame in range(600):
        expected = Fraction(frame) * time_base * rate
        assert frame_sample_index(frame, time_base, rate) == expected.numerator // expected.denominator
    expected = Fraction(frame_index) * time_base * rate
    center = expected.numerator // expected.denominator + 23
    window = window_for_frame(signal, frame_index, time_base, window_samples=801,
                              left_context_samples=400, offset_samples=23)
    assert window.center_sample == center
    assert window.samples[400] == pytest.approx(0.9)


def test_vfr_trim_origin_integer_offset_and_no_cumulative_drift():
    origin = Fraction(201, 100)
    signal = AudioSignal(np.zeros(20_000, np.float32), 16_000, origin)
    pts_values = (2010, 2050, 2170, 2200, 2480)
    time_base = Fraction(1, 1000)
    for pts in pts_values:
        exact = (pts * time_base - origin) * 16_000 + 37
        assert frame_sample_index(pts, time_base, 16_000, audio_start_time=origin,
                                 offset_samples=37) == exact.numerator // exact.denominator
    window = window_for_frame(signal, 2170, time_base, window_samples=600,
                              left_context_samples=300, offset_samples=37)
    assert window.center_sample == 2597
    assert window.start_sample == 2297 and window.end_sample == 2897


def test_window_zero_pads_leading_and_trailing_silence_and_slices_mel():
    signal = AudioSignal(np.arange(10, dtype=np.float32) / 10, 16_000)
    leading = window_for_frame(signal, 0, Fraction(1, 16_000), window_samples=8,
                               left_context_samples=4)
    np.testing.assert_array_equal(leading.samples, np.array((0, 0, 0, 0, 0, .1, .2, .3), np.float32))
    trailing = window_for_frame(signal, 9, Fraction(1, 16_000), window_samples=8,
                                left_context_samples=4)
    np.testing.assert_array_equal(trailing.samples[5:], np.zeros(3, np.float32))
    features = extract_log_mel(signal, n_fft=16, hop_length=4, n_mels=4, fmin=0, fmax=7_600)
    mel = mel_window_for_audio_window(features, leading)
    assert mel.shape == (4, 2) and np.all(mel[:, 0] == 0)


def test_resampling_mel_extraction_and_local_decode_leave_original_audio_intact(tmp_path: Path):
    sample_rate = 48_000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    mono = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    stereo_i16 = np.column_stack((mono, mono))
    converted = resample_mono(stereo_i16, sample_rate)
    assert converted.shape == (16_000,)
    assert np.max(np.abs(converted)) == pytest.approx(.25, rel=.03)

    features = audio_features_from_pcm(stereo_i16, sample_rate)
    assert features.signal.sample_rate == 16_000
    assert features.mel.shape == (80, 80) and np.isfinite(features.mel).all()

    path = tmp_path / "tone.wav"
    pcm = np.clip(stereo_i16 * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(2)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    decoded = decode_audio(path)
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert decoded.sample_rate == 16_000 and decoded.samples.size == 16_000
    assert before == after and decoded.source_path == path


def test_silence_and_low_face_confidence_decay_lip_sync_gate_smoothly():
    gate = LipSyncGate()
    initial = gate.update(2, .12, .95, timestamp=Fraction(0))
    assert initial > 0
    values = [gate.update(2, 0.0, .95, timestamp=Fraction(index, 30)) for index in range(1, 10)]
    assert all(later < earlier for earlier, later in zip((initial, *values[:-1]), values))
    low_confidence = gate.update(2, .12, .1, timestamp=Fraction(10, 30))
    assert low_confidence < values[-1]
    gate.begin_frame(active_track_ids=set())
    assert gate.track_count == 0


def test_unavailable_wav2lip_is_explicit_and_returns_unmodified_crop():
    adapter = LipSyncAdapter()
    image = np.full((96, 96, 3), 137, np.uint8)
    result = adapter.process_frame(image, 4, LipSyncFrameInput(np.ones((80, 16), np.float32), .1, .9))
    np.testing.assert_array_equal(result.image, image)
    assert not result.applied and result.blend == 0 and not result.mouth_mask.any()
    capability = adapter.capability()
    assert capability["state"] == "UNSUPPORTED"
    assert capability["weights"] == "CANDIDATE"
    assert capability["three_d_visemes"] == "UNSUPPORTED"


def test_mouth_mask_and_fusion_never_modify_upper_face():
    mask = lower_face_soft_mask((128, 128, 3), feather_px=2)
    assert not mask[:52].any()
    assert mask[88, 64] > .5
    base = np.full((128, 128, 3), 80, np.uint8)
    synthesized = np.full_like(base, 220)
    result = composite_mouth_region(base, synthesized, mask)
    np.testing.assert_array_equal(result[mask == 0], base[mask == 0])
    np.testing.assert_array_equal(result[:52], base[:52])


class _FaceParser:
    def predict(self, image: np.ndarray) -> np.ndarray:
        scores = np.zeros((*image.shape[:2], 19), np.float32)
        scores[8:-8, 8:-8, 1] = 1
        return scores


def test_stage09_accepts_synthesized_crop_through_mouth_only_mask():
    frame = np.full((80, 80, 3), 30, np.uint8)
    base = np.full((64, 64, 3), 100, np.uint8)
    synthesized = np.full_like(base, 210)
    mouth = np.zeros((64, 64), np.float32)
    mouth[42:50, 28:36] = 1
    result = type("LipResult", (), {"image": synthesized, "mouth_mask": mouth,
                                    "blend": 1.0, "applied": True})()
    replacement = type("Replacement", (), {"track_id": 6, "image": base,
                                             "inverse": np.array(((1, 0, 8), (0, 1, 8)), np.float32)})()
    points = np.array(((18, 18), (61, 18), (61, 61), (18, 61), (40, 16)), np.float32)
    compositor = FrameCompositor(parser=_FaceParser(), temporal=TemporalSmoother())
    output = compositor.compose(frame, (replacement,), {6: points},
                                lip_sync_results={6: result},
                                options=CompositeOptions(color_strength=0, feather_px=0))
    assert np.all(output[50:58, 36:44] == 210)
    assert np.all(output[20:25, 32:38] == 100)


def test_reference_pipeline_keeps_audio_enabled_job_non_destructive_without_weights():
    frame = np.full((96, 96, 3), 25, np.uint8)
    crop = np.full((64, 64, 3), 160, np.uint8)
    replacement = type("Replacement", (), {"track_id": 9, "image": crop,
                                             "inverse": np.array(((1, 0, 16), (0, 1, 16)), np.float32)})()
    points = np.array(((25, 24), (70, 24), (70, 70), (25, 70), (47, 22)), np.float32)
    observation = type("Observation", (), {"track_id": 9,
                                            "observation": type("Face", (), {"landmarks106": points})()})()
    frame_result = type("FrameResult", (), {"observations": (observation,),
                                             "replacements": (replacement,), "scene_cut": False})()
    session = ReferencePipeline()
    session.compositor.parser = _FaceParser()
    session.frames_processed = 1
    audio_input = LipSyncFrameInput(np.ones((80, 16), np.float32), .1, .9, Fraction(0))
    try:
        without_audio = session.composite_frame(frame, frame_result,
                                                options=CompositeOptions(color_strength=0, feather_px=2))
        with_audio = session.composite_frame(frame, frame_result,
                                             options=CompositeOptions(color_strength=0, feather_px=2),
                                             lip_sync_enabled=True, lip_sync_inputs={9: audio_input})
        np.testing.assert_array_equal(with_audio, without_audio)
        assert session.capability()["lip_sync"]["state"] == "UNSUPPORTED"
    finally:
        session.release_session()
