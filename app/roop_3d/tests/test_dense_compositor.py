"""Compositor behaviour specific to dense 3D replacements: no second restorer, shading, coverage smoothing, presence."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.compositor.pipeline import CompositeOptions, FrameCompositor
from roop_3d.compositor.temporal import TemporalSmoother

SIDE = 192
YY, XX = np.mgrid[:340, :340].astype(np.float32)


class CountingRestorer:
    def __init__(self):
        self.calls = 0

    def restore_hires(self, crop, landmarks5=None, **_):
        self.calls += 1
        return cv2.resize(crop, (512, 512), interpolation=cv2.INTER_CUBIC)


def coverage_disc(cx=96.0, cy=100.0, radius=62.0):
    yy, xx = np.mgrid[:SIDE, :SIDE].astype(np.float32)
    return cv2.GaussianBlur((((xx - cx) ** 2 + (yy - cy) ** 2) < radius ** 2).astype(np.float32), (0, 0), 5)


def replacement(coverage=None, *, restored=True, method="dense_3d_correspondence_neural_ultramax", presence=1.0):
    inverse = np.array(((340 / SIDE * 0.62, 0, 70), (0, 340 / SIDE * 0.62, 60)), np.float32)
    points = np.array(((62, 82), (130, 82), (96, 118), (70, 146), (122, 146)), np.float32)
    return SimpleNamespace(track_id=1, image=np.full((SIDE, SIDE, 3), (110, 140, 190), np.uint8), inverse=inverse,
                           landmarks5=points, method=method, restored=restored, presence=presence,
                           coverage=coverage_disc() if coverage is None else coverage)


def compositor(restorer=None):
    comp = FrameCompositor(parser=SimpleNamespace(predict=lambda crop: None), temporal=TemporalSmoother())
    comp.restorer = restorer
    return comp


def frame_lit_from_the_right():
    light = 0.45 + 0.55 * (XX / 340)
    return np.clip(np.array((110, 140, 190), np.float32) * light[..., None], 0, 255).astype(np.uint8)


def compose(comp, rep, frame=None, options=None, index=0, observer=None):
    frame = frame_lit_from_the_right() if frame is None else frame
    options = options or CompositeOptions(restoration_blend=0.7, color_strength=0, feather_px=2)
    return comp.compose(frame, (rep,), {1: None}, frame_index=index, options=options, observer=observer)


def face_luma_sides(output):
    """Mean luma of the left and right cheek of the face disc centred at frame (175, 170), radius ~68 px."""
    luma = output.astype(np.float32) @ np.array((0.114, 0.587, 0.299), np.float32)
    return luma[150:190, 125:160].mean(), luma[150:190, 190:225].mean()


def test_dense_replacement_that_neural_face_restored_is_not_restored_again():
    restorer = CountingRestorer()
    compose(compositor(restorer), replacement(restored=True))
    assert restorer.calls == 0
    compose(compositor(restorer), replacement(restored=False))
    assert restorer.calls == 1                                  # every other replacement still gets the compositor's restorer


def test_a_restorer_is_still_required_when_restoration_is_requested():
    from roop_3d.core.errors import DomainError
    with pytest.raises(DomainError) as error:
        compose(compositor(None), replacement(restored=True))
    assert error.value.code == "RESTORER_UNAVAILABLE"


def test_dense_shading_gives_the_flat_synthesised_face_the_targets_lit_side():
    plain = compose(compositor(CountingRestorer()), replacement(), options=CompositeOptions(
        restoration_blend=0.7, color_strength=0, feather_px=2, dense_shading=0.0))
    shaded = compose(compositor(CountingRestorer()), replacement())
    target = frame_lit_from_the_right()
    ratio = lambda image: face_luma_sides(image)[0] / face_luma_sides(image)[1]
    assert ratio(plain) > 0.93                                  # flat crop composited as-is
    assert abs(ratio(shaded) - ratio(target)) < abs(ratio(plain) - ratio(target)) * 0.5


def test_shading_does_not_run_for_other_replacement_methods():
    off = compose(compositor(CountingRestorer()), replacement(method="identity_swap", restored=False, coverage=coverage_disc()),
                  options=CompositeOptions(restoration_blend=0.0, color_strength=0, feather_px=2, dense_shading=0.0))
    on = compose(compositor(CountingRestorer()), replacement(method="identity_swap", restored=False, coverage=coverage_disc()),
                 options=CompositeOptions(restoration_blend=0.0, color_strength=0, feather_px=2))
    np.testing.assert_array_equal(off, on)


def test_dense_coverage_is_temporally_smoothed_in_the_registered_crop():
    rng = np.random.default_rng(3)
    shifts = rng.normal(0, 3.0, (40, 2))
    comp = compositor(CountingRestorer())
    mattes, rawcentroids = [], []

    def observe(name, image, inverse, alpha):
        if name == "matte":
            mattes.append(np.asarray(image, np.float32) / 255.0)

    for index, (dx, dy) in enumerate(shifts):
        coverage = coverage_disc(96 + dx, 100 + dy)
        rawcentroids.append(((coverage * XX[:SIDE, :SIDE]).sum() / coverage.sum(), (coverage * YY[:SIDE, :SIDE]).sum() / coverage.sum()))
        compose(comp, replacement(coverage), index=index, observer=observe)
    centroid = lambda m: np.array(((m * XX[:SIDE, :SIDE]).sum() / m.sum(), (m * YY[:SIDE, :SIDE]).sum() / m.sum()))
    smoothed = np.array([centroid(m) for m in mattes])
    second = lambda s: np.sqrt(np.mean((np.asarray(s)[2:] - 2 * np.asarray(s)[1:-1] + np.asarray(s)[:-2]) ** 2))
    assert second(smoothed[5:]) < 0.8 * second(rawcentroids[5:])


def test_presence_scales_the_composited_alpha_of_a_held_face():
    frame = frame_lit_from_the_right()
    full = compose(compositor(CountingRestorer()), replacement(presence=1.0), frame=frame)
    half = compose(compositor(CountingRestorer()), replacement(presence=0.5), frame=frame)
    none = compose(compositor(CountingRestorer()), replacement(presence=0.0), frame=frame)
    np.testing.assert_array_equal(none, frame)
    inside = (slice(110, 170), slice(110, 160))
    d_full = np.abs(full[inside].astype(int) - frame[inside]).mean()
    d_half = np.abs(half[inside].astype(int) - frame[inside]).mean()
    assert d_half == pytest.approx(d_full * 0.5, rel=0.15)


def _laplacian_energy(image, alpha):
    gray = image.astype(np.float32) @ np.array((0.114, 0.587, 0.299), np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float((lap ** 2 * alpha).sum() / alpha.sum())


def test_sensor_optics_follow_the_targets_measured_softness_and_never_soften_a_face_below_it():
    """The optics stage blurs a face only while it is SHARPER than the target; a face softer than its target is left alone."""
    from roop_3d.compositor.sensor_physics import finish_target_sensor
    rng = np.random.default_rng(0)
    texture = cv2.GaussianBlur(rng.normal(128, 40, (256, 256, 3)).astype(np.float32), (0, 0), 1.0)
    sharp = np.clip(texture, 0, 255).astype(np.uint8)
    soft = cv2.GaussianBlur(sharp, (0, 0), 2.5)
    alpha = np.ones((256, 256), np.float32)
    # dense swap crop softer than its target (the measured case: ~0.2-0.3x of the target's Laplacian variance)
    out, _ = finish_target_sensor(soft, sharp, alpha, None, grain_strength=0.0, match_optics=True)
    assert _laplacian_energy(out, alpha) >= 0.97 * _laplacian_energy(soft, alpha)
    # a crop sharper than its target is brought toward the target's energy, no further
    softer_target = cv2.GaussianBlur(sharp, (0, 0), 1.2)
    out, _ = finish_target_sensor(sharp, softer_target, alpha, None, grain_strength=0.0, match_optics=True)
    target_energy = _laplacian_energy(softer_target, alpha)
    assert 0.6 * target_energy <= _laplacian_energy(out, alpha) <= 1.4 * target_energy


def test_full_diagnostics_measure_the_final_frame_of_each_composited_face(tmp_path):
    from roop_3d.engine.volumetric.angle_diagnostics import AngleDiagnostics
    diagnostics = AngleDiagnostics()
    diagnostics.configure("full", tmp_path / "angle.jsonl")
    comp = compositor(CountingRestorer())
    comp.angle_diagnostics = diagnostics
    rep = replacement()
    swap = {"crop_matrix": np.array(((1.0, 0, 0), (0, 1.0, 0))), "landmarks5": np.asarray(rep.landmarks5, np.float64),
            "coverage": rep.coverage, "coverage_scale": 1.0, "views": [{"index": 0, "yaw": 0, "pitch": 0, "raw": .1, "weight": 1.0}]}
    diagnostics.record_swap(0, 1, swap, pose=(12.0, 0.0, 0.0))
    compose(comp, rep, index=0)
    records = diagnostics.records()
    assert len(records) == 1 and "ring" in records[0] and records[0]["sharpness"]["yaw_bucket"] == 10
    summary = diagnostics.summary()
    assert summary["mask_edge"]["frames_measured"] == 1
    assert diagnostics.finalize()["rows"] == 1
    assert (tmp_path / "angle.jsonl").read_text().count("\n") == 1
    # the default level never touches the frame
    quiet = AngleDiagnostics()
    quiet.configure("summary")
    comp.angle_diagnostics = quiet
    quiet.record_swap(1, 1, swap, pose=(12.0, 0.0, 0.0))
    compose(comp, rep, index=1)
    assert quiet.records() == [] and "mask_edge" not in quiet.summary()
