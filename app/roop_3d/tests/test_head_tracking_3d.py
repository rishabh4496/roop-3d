"""Head/gaze tracker: filters, per-shot identity lock, gaze, lighting, shot boundaries and the tracker as a whole.

Everything is verified against a scripted synthetic performer with KNOWN pose, expression and timing (``tracking_world``); the
tracker sees only landmarks, timestamps and pixels. Where a property is a measured limit rather than a guarantee (pose accuracy
against a head that differs from the canonical one, light-direction accuracy under geometry mismatch) the bound sits clearly
outside what was measured and the test says what was measured.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tracking_world as T  # noqa: E402
import target_world as TW  # noqa: E402

from roop_3d.core.pose import rotation_from_euler  # noqa: E402
from roop_3d.flame_3d_engine import EyeGaze, FlameParameters  # noqa: E402
from roop_3d.temporal import OneEuroFilter  # noqa: E402
from roop_3d.tracking.filters import DEFAULT_SPECS, FilterSpec, SmoothedChannel, TrackFilterBank  # noqa: E402
from roop_3d.tracking.gaze import (  # noqa: E402
    DEFAULT_IPD_M, EYEBALL_TO_HALF_WIDTH, LEFT_EYE, RIGHT_EYE, SOURCE_ESTIMATOR, SOURCE_IRIS, iris_gaze_from_mesh478, vergence)
from roop_3d.tracking.identity_lock import (  # noqa: E402
    KIND_106, KIND_5, KIND_FLAME, ShotIdentityLock, landmark_proportions, pose_weight, weighted_median)
from roop_3d.tracking.lighting import MeshSHLightingEstimator, angle_between_deg  # noqa: E402
from roop_3d.tracking.video_tracker import (  # noqa: E402
    EXPRESSION_CHANNELS, FrameEvidence, ShotBoundaryDetector, TrackerSettings, VideoHeadTracker3D)

DT = 1 / 30
POSE_SPEC = FilterSpec(1.5, 0.03, 8.0, noise=0.5, accel=900.0)      # degrees, like the reenactment pose channel


@pytest.fixture(scope="module")
def actor():
    return T.performer()


# =========================================================================== filters
def _response(signal, spec=POSE_SPEC, frames=150):
    channel, euro = SmoothedChannel(spec), OneEuroFilter(spec.min_cutoff, spec.beta, spec.d_cutoff)
    kalman, plain = [], []
    for i in range(frames):
        t = i * DT
        kalman.append(float(channel.update([signal(t)], t)[0]))
        plain.append(float(euro.update([signal(t)], t)[0]))
    return np.array(kalman), np.array(plain)


@pytest.mark.parametrize("name,signal,final", [
    ("step", lambda t: 0.0 if t < 0.6 else 70.0, 70.0),
    ("turn_that_stops", lambda t: float(np.clip((t - 0.5) * 300.0, 0.0, 60.0)), 60.0),
])
def test_kalman_stage_never_overshoots_and_adds_no_lag(name, signal, final):
    """The defect this module exists to avoid: TrackTemporalStabilizer.filter overshoots a 70 degree step to 108 degrees."""
    kalman, plain = _response(signal)
    assert kalman.max() <= final + 1e-9, f"{name}: peak {kalman.max()}"
    assert abs(kalman[-1] - final) < 1e-6
    reach = lambda x: next(i for i, v in enumerate(x) if v >= 0.9 * final)
    assert reach(kalman) <= reach(plain)
    assert np.all(np.diff(kalman) >= -1e-9), "a monotone input must give a monotone output"


def test_velocity_memory_cannot_carry_the_output_past_a_target_that_stopped():
    """A smooth channel (low process noise, so a large velocity estimate) fed a long ramp that stops dead: an unconfined
    constant-velocity Kalman stage keeps extrapolating; this one cannot pass the One-Euro output."""
    spec = FilterSpec(2.0, 0.5, 8.0, noise=0.05, accel=2.0)
    signal = lambda t: float(np.clip((t - 0.3) * 8.0, 0.0, 10.0))
    kalman, plain = _response(signal, spec, frames=200)
    assert kalman.max() <= 10.0 + 1e-9 and abs(kalman[-1] - 10.0) < 1e-3
    assert np.all(kalman <= plain + 1e-9) or np.all(np.diff(kalman) >= -1e-9), "never ahead of, or past, the smoothed measurement"
    assert np.all(kalman <= np.maximum.accumulate(plain) + 1e-9)


@pytest.mark.parametrize("channel", ["translation", "expression", "gaze", "jaw"])
def test_rest_jitter_is_never_worse_than_one_euro(channel):
    spec, rng = DEFAULT_SPECS[channel], np.random.default_rng(1)
    smooth, euro = SmoothedChannel(spec), OneEuroFilter(spec.min_cutoff, spec.beta, spec.d_cutoff)
    a, b = [], []
    for i in range(600):
        z = 0.3 + rng.normal(0, spec.noise * 2, 1)
        a.append(smooth.update(z, i * DT)[0])
        b.append(euro.update(z, i * DT)[0])
    # Measured: the Kalman stage adds <=10 % rest smoothing on top of One-Euro with these parameters; it must not add noise.
    assert np.std(a[60:]) <= np.std(b[60:]) * 1.001
    assert np.std(a[60:]) < spec.noise * 2 * 0.7


def test_coast_bridges_missing_frames_better_than_holding_and_expires():
    channel, speed = SmoothedChannel(DEFAULT_SPECS["expression"]), 0.2
    truth = lambda t: speed * t
    for i in range(60):
        channel.update([truth(i * DT)], i * DT)
    held = truth(59 * DT)
    errors = []
    for k in range(60, 64):
        bridged = channel.coast(k * DT)
        errors.append((abs(bridged[0] - truth(k * DT)), abs(held - truth(k * DT))))
    assert all(bridge < hold for bridge, hold in errors)
    assert errors[-1][0] < 0.6 * errors[-1][1]
    assert channel.coast(80 * DT) is None, "coasting is bounded by max_coast_s"
    assert channel.update([truth(80 * DT)], 80 * DT)[0] == pytest.approx(truth(80 * DT)), "a gap over 0.5 s restarts at the measurement"


def test_coast_velocity_decays_instead_of_running_away():
    channel = SmoothedChannel(FilterSpec(2.0, 0.5, 8.0, noise=0.01, accel=5.0, max_coast_s=2.0, velocity_tau_s=0.1))
    for i in range(40):
        channel.update([i * DT * 3.0], i * DT)
    positions = [channel.coast((40 + k) * DT)[0] for k in range(30)]
    steps = np.diff(positions)
    assert steps[-1] < 0.1 * steps[0] and positions[-1] < positions[0] + 0.5


def test_channel_resets_on_gap_nonmonotonic_time_and_shape_change():
    channel = SmoothedChannel(DEFAULT_SPECS["expression"])
    channel.update([0.1, 0.1], 0.0)
    channel.update([0.1, 0.1], DT)
    assert np.allclose(channel.update([0.4, 0.4], 2.0), [0.4, 0.4]), "gap"
    assert np.allclose(channel.update([0.9, 0.9], 2.0), [0.9, 0.9]), "same timestamp"
    assert np.allclose(channel.update([0.2], 2.1), [0.2]), "shape change"
    with pytest.raises(ValueError):
        channel.update([float("nan")], 3.0)
    with pytest.raises(ValueError):
        FilterSpec(0.0, 1.0, 1.0, 0.1, 1.0)


def test_rotation_filter_is_so3_and_does_not_unwrap_through_zero():
    bank = TrackFilterBank()
    roll = lambda deg: rotation_from_euler(0.0, 0.0, deg)
    out = [bank.rotation(roll(179.0 if i % 2 else -179.0), i * DT) for i in range(30)]
    # Euler-angle averaging of +179 and -179 passes through 0; on SO(3) the filtered rotation stays near 180 degrees.
    assert all(abs(R[0, 0] + 1.0) < 0.05 for R in out[5:])
    assert all(np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0.999 for R in out)


def test_filter_bank_reset_and_rotation_coast():
    bank = TrackFilterBank()
    bank.update("jaw", [0.2], 0.0)
    bank.rotation(rotation_from_euler(10, 0, 0), 0.0)
    assert bank.coast_rotation(0.1) is not None and bank.coast_rotation(1.0) is None
    bank.reset()
    assert bank.value("jaw") is None and bank.coast("jaw", 0.1) is None and bank.coast_rotation(0.1) is None


# =========================================================================== identity lock
def _lock_stream(lock, centre, n, *, sigma=0.01, weight=0.9, seed=0, outliers=0.0):
    rng, snapshot = np.random.default_rng(seed), None
    for _ in range(n):
        vector = centre + rng.normal(0, sigma, centre.shape)
        if rng.random() < outliers:
            vector = centre + rng.normal(0, 0.5, centre.shape) + 0.5
        snapshot = lock.observe(KIND_106, vector, weight)
    return snapshot


def test_weighted_median_matches_numpy_and_respects_weights():
    rng = np.random.default_rng(3)
    values = rng.normal(size=(51, 4))
    assert np.allclose(weighted_median(values, np.ones(51)), np.median(values, axis=0))
    skewed = np.array([[0.0], [1.0], [100.0]])
    assert weighted_median(skewed, np.array([1.0, 1.0, 5.0]))[0] == 100.0
    assert weighted_median(skewed, np.array([1.0, 5.0, 1.0]))[0] == 1.0


def test_median_pooling_is_robust_to_gross_outliers_and_locks_then_freezes():
    centre = np.array((1.0, 2.3, 1.5, 0.46, 0.55))
    lock = ShotIdentityLock()
    states = []
    rng = np.random.default_rng(5)
    for i in range(120):
        vector = centre + rng.normal(0, 0.012, 5)
        if i % 5 == 4:                                   # 20 % gross outliers (mis-detections)
            vector = vector + rng.choice([-1, 1], 5) * 0.8
        states.append(lock.observe(KIND_106, vector, 0.9))
    final = states[-1]
    assert final.state == "locked" and final.locked_at_frame is not None
    assert np.max(np.abs(final.beta - centre)) < 0.01, "the mean would sit ~0.16 away; the median does not"
    assert final.rejected >= 3, "outliers met while collecting are refused, not pooled"
    assert final.frames == 30, "the pool stops growing at the lock"
    assert states[0].state == "collecting" and states[0].beta is None
    assert any(s.state == "provisional" for s in states)
    frozen = final.beta.copy()
    for _ in range(30):
        later = lock.observe(KIND_106, centre + 0.02, 0.9)
    assert np.array_equal(later.beta, frozen), "a locked shape is frozen"
    assert not later.break_detected


def test_lock_ignores_low_weight_other_kinds_and_non_finite():
    lock = ShotIdentityLock()
    centre = np.ones(5)
    lock.observe(KIND_106, centre, 0.9)
    before = lock.snapshot().rejected
    lock.observe(KIND_106, centre, 0.1)                    # not frontal / neutral enough
    lock.observe(KIND_5, np.ones(2), 0.9)                  # other descriptor kind
    lock.observe(KIND_106, np.full(5, np.nan), 0.9)
    lock.observe(KIND_106, np.ones(3), 0.9)                # wrong length
    assert lock.snapshot().rejected == before + 4 and lock.snapshot().frames == 1


def test_identity_break_after_a_run_of_far_frames_but_not_for_a_few():
    centre = np.array((1.0, 2.3, 1.5, 0.46, 0.55))
    lock = ShotIdentityLock(break_run=12)
    _lock_stream(lock, centre, 60)
    assert lock.snapshot().state == "locked"
    other = centre * np.array((1.3, 0.8, 1.2, 1.4, 0.7))
    for i in range(11):
        assert not lock.observe(KIND_106, other, 0.9).break_detected
    lock.observe(KIND_106, centre, 0.9)                    # one good frame resets the run
    for i in range(11):
        assert not lock.observe(KIND_106, other, 0.9).break_detected
    assert lock.observe(KIND_106, other, 0.9).break_detected


def test_lock_does_not_lock_a_drifting_descriptor():
    lock = ShotIdentityLock(lock_frames=20, stable_tolerance=0.1)
    centre = np.array((1.0, 2.3, 1.5, 0.46, 0.55))
    for i in range(80):
        snapshot = lock.observe(KIND_106, centre * (1 + 0.004 * i), 0.9)       # +0.4 % per frame
    assert snapshot.state != "locked"


def test_lock_reports_prior_agreement_and_resets():
    centre = np.array((1.0, 2.3, 1.5, 0.46, 0.55))
    lock = ShotIdentityLock()
    lock.set_prior(KIND_106, centre, np.full(5, 0.03))
    near = _lock_stream(lock, centre, 40)
    assert near.prior_distance is not None and near.prior_distance < 1.0
    lock.reset()
    far = _lock_stream(lock, centre * 1.3, 40)
    assert far.prior_distance > 3.0, "a different person than the calibrated actor"
    assert lock.snapshot().state != "collecting"
    lock.reset()
    assert lock.snapshot().state == "collecting" and lock.snapshot().frames == 0
    with pytest.raises(ValueError):
        ShotIdentityLock(min_frames=50, lock_frames=10)


def test_pose_weight_is_a_near_frontal_window():
    assert pose_weight(0, 0) == 1.0 and pose_weight(22, 0) == 0.0 and pose_weight(0, 16) == 0.0
    assert pose_weight(10, 5) > pose_weight(15, 5) > pose_weight(20, 5) > 0.0
    assert pose_weight(-12, 0) == pose_weight(12, 0)


def test_proportions_are_pose_stable_and_distinguish_actors(actor):
    """Neutral, noise-free: the nose-free ratios move by a few percent across the frontal window; another face differs more."""
    def descriptor(subject, yaw, pitch):
        obs, _ = TW.frame(subject, yaw=yaw, pitch=pitch, roll=0.0, t=0.0, noise_px=0.0)
        return landmark_proportions(obs["landmarks5"], obs["landmarks106"], yaw, pitch)
    kind, centre = descriptor(actor, 0, 0)
    assert kind == KIND_106 and centre.shape == (5,) and np.all(centre > 0)
    drift = max(np.max(np.abs(descriptor(actor, y, p)[1] / centre - 1)) for y in (-15, 0, 15) for p in (-8, 0, 8))
    assert drift < 0.08, drift
    stranger = T.performer(seed=9, width=0.30, jaw_drop=0.09, nose_depth=0.02)
    assert np.max(np.abs(descriptor(stranger, 0, 0)[1] / centre - 1)) > 0.25, "a very different face moves a ratio by far more than pose does"
    kind5, five = landmark_proportions(TW.frame(actor, t=0.0, noise_px=0.0)[0]["landmarks5"], None, 0, 0)
    assert kind5 == KIND_5 and five.shape == (2,)
    assert landmark_proportions(np.zeros((5, 2)), None, 0, 0) is None                      # degenerate
    assert landmark_proportions(TW.frame(actor, t=0.0)[0]["landmarks5"], None, 80, 0) is None   # too turned


# =========================================================================== gaze
def _mesh(left=(0.0, 0.0), right=(0.0, 0.0), width=0.1):
    """A 478-point mesh whose two eyes are `width` wide with the iris displaced by the given (yaw, pitch) in radians."""
    mesh = np.zeros((478, 3))
    radius = EYEBALL_TO_HALF_WIDTH * 0.5 * width
    for (outer, inner, iris), (yaw, pitch), x0 in ((LEFT_EYE, left, 0.65), (RIGHT_EYE, right, 0.35)):
        mesh[outer] = (x0 + width / 2, 0.4, 0)
        mesh[inner] = (x0 - width / 2, 0.4, 0)
        mesh[iris] = (x0 + radius * math.sin(yaw), 0.4 + radius * math.sin(pitch), 0)
    return mesh


def test_iris_gaze_recovers_known_angles_and_is_scale_free():
    """Checks the algebra of the (uncalibrated) iris estimator against its own forward model, not real eyes."""
    for width, aspect in ((0.1, 1.0), (0.04, 1.0), (0.1, 1.0)):
        gaze = iris_gaze_from_mesh478(_mesh(left=(math.radians(12), math.radians(-6)), right=(math.radians(-8), math.radians(4)), width=width), aspect=aspect)
        assert np.degrees(gaze.angles()) == pytest.approx([12, -6, -8, 4], abs=0.01)
    pixels = _mesh(left=(0.2, 0.1), right=(0.2, 0.1))[:, :2] * 800.0                          # frame pixels, (478, 2)
    assert np.allclose(iris_gaze_from_mesh478(pixels).angles(), [0.2, 0.1, 0.2, 0.1], atol=1e-6)
    assert iris_gaze_from_mesh478(np.zeros((478, 3))) is None, "eyes of zero width carry no angle"
    assert iris_gaze_from_mesh478(np.zeros((468, 3))) is None and iris_gaze_from_mesh478(np.full((478, 3), np.nan)) is None
    assert iris_gaze_from_mesh478(_mesh(), aspect=0.0) is None
    saturated = iris_gaze_from_mesh478(_mesh(left=(1.2, 0.0)))        # beyond the sphere: clamped, never NaN
    assert np.isfinite(saturated.angles()).all()


def test_vergence_intersects_the_rays_and_reports_parallel_gaze_as_infinite():
    distance = 0.5
    inward = math.atan2(DEFAULT_IPD_M / 2, distance)
    # subject's left eye is image-right (+x) and turns toward -x; the right eye turns toward +x
    near = vergence((-inward, 0.0, +inward, 0.0))
    assert near.angle_deg == pytest.approx(math.degrees(2 * inward)) and near.distance_m == pytest.approx(distance, rel=1e-6)
    assert vergence((0.0, 0.0, 0.0, 0.0)).distance_m is None
    assert vergence((0.1, 0.0, -0.1, 0.0)).distance_m is None and vergence((0.1, 0.0, -0.1, 0.0)).angle_deg < 0   # wall-eyed
    assert vergence((-0.0001, 0, 0.0001, 0)).distance_m is None, "below the noise floor"
    assert vergence((-inward, 0, inward, 0), ipd_m=0.07).distance_m == pytest.approx(distance * 0.07 / DEFAULT_IPD_M, rel=1e-6)


# =========================================================================== lighting
@pytest.mark.parametrize("yaw,pitch", [(0, 0), (20, 0), (-25, 8)])
@pytest.mark.parametrize("direction", [(0.6, -0.4, 0.69), (-0.7, 0.1, 0.7), (0.0, -0.7, 0.7)])
def test_lighting_recovers_the_key_light_on_a_different_head(actor, yaw, pitch, direction):
    """Rendered head != the template the estimator uses. Measured over 12 such cases: key-direction error 6-31 degrees."""
    coefficients = T.sh_environment(direction)
    image, five, _ = T.lit_frame(actor.subject, yaw, pitch, coefficients)
    tracker = VideoHeadTracker3D(TrackerSettings(lighting_stride=1), lighting=MeshSHLightingEstimator())
    frame = tracker.process(FrameEvidence(0.0, five, image_bgr=image))
    assert frame.status == "tracked" and frame.lighting is not None and frame.lighting.samples > 400
    truth = T.SH.extract_key_light_direction(coefficients)
    # (the camera here is the tracker's own pose estimate, which has its own few-degree error against this head)
    assert angle_between_deg(frame.lighting.key_direction, truth) < 40.0
    assert frame.lighting_coefficients.shape == (9, 3) and np.isfinite(frame.lighting_coefficients).all()
    # flat-shaded triangles of another head under a few degrees of pose error: measured residual 0.10-0.14 on a ~0.38 mean shade
    assert frame.lighting.residual < 0.2


def test_lighting_estimates_are_stable_and_follow_a_real_light_change(actor):
    left, right = T.sh_environment((-0.7, 0.1, 0.7)), T.sh_environment((0.7, 0.1, 0.7))
    tracker = VideoHeadTracker3D(TrackerSettings(lighting_stride=2), lighting=MeshSHLightingEstimator())
    keys = []
    for i in range(60):
        image, five, _ = T.lit_frame(actor.subject, 0, 0, left if i < 30 else right)
        frame = tracker.process(FrameEvidence(i * DT, five, image_bgr=image))
        keys.append(T.SH.extract_key_light_direction(frame.lighting_coefficients))
    assert keys[25][0] < -0.3 and keys[-1][0] > 0.3, "x of the key light flips with the real light"
    assert np.std([k[0] for k in keys[10:28]]) < 0.03, "and is quiet while the light does not change"
    sources = {tracker.process(FrameEvidence(60 * DT + i * DT, five, image_bgr=image)).lighting_source for i in range(4)}
    assert sources <= {"frame_fit", "held"}


def test_lighting_needs_enough_visible_skin():
    estimator = MeshSHLightingEstimator()
    from roop_3d.reconstruction.geometry_types import WeakPerspectiveCamera
    away = np.diag([-1.0, 1.0, -1.0])                                                # yaw 180: the back of the head
    camera = WeakPerspectiveCamera(rotation=tuple(map(tuple, away)), scale=2000.0, tx=300.0, ty=300.0)
    assert estimator.estimate(np.full((600, 600, 3), 128, np.uint8), camera) is None
    assert estimator.estimate(np.full((600, 600, 3), 128, np.uint8), WeakPerspectiveCamera(
        rotation=tuple(map(tuple, np.eye(3))), scale=2000.0, tx=300.0, ty=300.0)).samples > 400
    off_frame = WeakPerspectiveCamera(rotation=tuple(map(tuple, np.eye(3))), scale=2000.0, tx=-5000.0, ty=300.0)
    assert estimator.estimate(np.full((600, 600, 3), 128, np.uint8), off_frame) is None
    with pytest.raises(ValueError):
        estimator.estimate(np.zeros((10, 10), np.uint8), off_frame)


# =========================================================================== tracker: pose, smoothing, expression
def test_pose_tracks_a_moving_performer_on_a_head_unlike_the_template(actor):
    tracker = VideoHeadTracker3D()
    out = T.run(tracker, actor, T.moving_script())
    errors = T.pose_errors(out)
    # Measured on this head (8 % narrower than the template): mean 4.2 / 2.8 / 1.1 degrees, ~3.5 ms per frame.
    assert errors.mean(axis=0)[0] < 6.5 and errors.mean(axis=0)[1] < 4.5 and errors.mean(axis=0)[2] < 2.5
    truth = np.array([t["yaw"] for _, t in out[20:]])
    tracked = np.array([f.camera.yaw_deg for f, _ in out[20:]])
    assert np.corrcoef(truth, tracked)[0, 1] > 0.995
    assert all(f.status == "tracked" and 0.0 < f.confidence <= 1.0 for f, _ in out)
    assert tracker.summary()["shot_cuts"] == 0 and tracker.summary()["status"]["lost"] == 0, "no false cuts in a continuous take"
    camera = out[-1][0].camera
    assert np.allclose(camera.rotation @ camera.rotation.T, np.eye(3), atol=1e-9) and camera.scale > 0


def test_a_static_head_is_smoother_than_the_raw_solve(actor):
    from roop_3d.reconstruction.multiview_fit import estimate_pose_from_landmarks, solve_view_pose
    from roop_3d.tracking.video_tracker import HeadAnchors
    anchors, tracker = HeadAnchors.canonical(), VideoHeadTracker3D()
    raw, tracked = [], []
    for i in range(120):
        obs, _ = TW.frame(actor, yaw=10.0, pitch=0.0, roll=0.0, t=i * DT, noise_px=0.6, seed=i)
        fit = solve_view_pose(obs["landmarks5"], anchors.points, anchors.normals, estimate_pose_from_landmarks(obs["landmarks5"]), anchors.weights)
        raw.append(fit.yaw)
        tracked.append(tracker.process(T.evidence(obs)).camera.yaw_deg)
    assert np.std(tracked[30:]) < 0.6 * np.std(raw[30:])


def test_a_fast_head_turn_is_followed_without_overshoot_or_a_cut(actor):
    steps = [dict(yaw=0.0)] * 20 + [dict(yaw=min(60.0, 15.0 * k)) for k in range(1, 6)] + [dict(yaw=60.0)] * 25
    out = T.run(VideoHeadTracker3D(), actor, steps, noise=0.1)
    yaws = np.array([f.camera.yaw_deg for f, _ in out])
    settled = float(np.median(yaws[-15:]))        # ~63 on this head: the generic head biases the solve itself by ~3 degrees at 60
    assert 55.0 < settled < 66.0 and all(not f.shot_cut for f, _ in out)
    assert yaws.max() <= settled + 0.5, f"overshoot: peak {yaws.max():.2f} vs settled {settled:.2f}"


def test_expression_is_calibrated_per_track_and_measures_asymmetric_events(actor):
    tracker = VideoHeadTracker3D()
    neutral = T.run(tracker, actor, T.frontal_script(14))
    assert not neutral[0][0].expression_calibrated and neutral[0][0].expression is None, "nothing is invented before calibration"
    assert neutral[-1][0].expression_calibrated and set(neutral[-1][0].expression) == set(EXPRESSION_CHANNELS)
    assert max(abs(v) for v in neutral[-1][0].expression.values()) < 0.05
    wink = T.run(tracker, actor, [dict(blink_a=1.0, brow=0.0, mouth_open=0.0)] * 10, start=14)[-1][0]
    assert wink.expression["blink_a"] > 0.6 and wink.expression["blink_b"] < 0.2
    assert wink.expression["blink_a"] - wink.expression["blink_b"] > 0.4
    talk = T.run(tracker, actor, [dict(mouth_open=0.3)] * 12, start=24)[-1][0]
    assert 0.2 < talk.expression["mouth_open"] < 0.4
    expected_jaw = talk.expression["mouth_open"] * 0.063 / 0.085
    assert talk.jaw_source == "landmark_proxy" and talk.jaw_open_rad == pytest.approx(expected_jaw, abs=0.03)
    assert talk.expression_energy > wink.expression_energy * 0.3 and 0.0 <= talk.expression_energy <= 1.0


def test_mouth_corner_lift_is_asymmetric(actor):
    """Lift only one mouth corner of the 3-D performer; smile_a and smile_b must disagree in the right direction."""
    tracker, subject = VideoHeadTracker3D(), actor.subject
    def frame(i, lift):
        lm3d = TW.expression_3d(subject.true_lm, actor.iod_m)
        lm3d[48, 1] -= lift * actor.iod_m                                             # one corner only
        p2d = TW.SCALE * (lm3d - subject.true_lm.mean(axis=0))[:, :2] + np.array([200.0, 200.0])
        return FrameEvidence(i * DT, TW.five_points(p2d), TW.landmarks106(p2d))
    for i in range(14):
        tracker.process(frame(i, 0.0))
    smile = [tracker.process(frame(14 + i, 0.15)) for i in range(10)][-1]
    a, b = smile.expression["smile_a"], smile.expression["smile_b"]
    assert a - b > 0.06 and a > 0.08, "landmark 48 (the image-left corner of an unmirrored frame) is channel a"
    assert abs(b) < 0.05, "the other corner did not move"


def test_gaze_none_without_a_source_and_source_is_always_labelled(actor):
    tracker = VideoHeadTracker3D()
    assert T.run(tracker, actor, [dict()] * 3)[-1][0].gaze is None
    ev = lambda i, **kw: FrameEvidence(i * DT, TW.frame(actor, t=0.0, noise_px=0.0)[0]["landmarks5"], **kw)
    calibrated = tracker.process(ev(10, gaze=EyeGaze(-0.05, 0.0, 0.05, 0.0)))
    assert calibrated.gaze.source == SOURCE_ESTIMATOR and calibrated.gaze.vergence_deg == pytest.approx(math.degrees(0.1), abs=0.01)
    assert calibrated.gaze.convergence_distance_m == pytest.approx(DEFAULT_IPD_M / (2 * math.tan(0.05)), rel=1e-3)
    assert calibrated.gaze.head_vectors.shape == (2, 3) and np.allclose(np.linalg.norm(calibrated.gaze.camera_vectors, axis=1), 1)
    iris = VideoHeadTracker3D().process(ev(0, mesh478=_mesh(left=(-0.05, 0), right=(0.05, 0))))
    assert iris.gaze.source == SOURCE_IRIS, "the uncalibrated iris estimate is never presented as calibrated"
    assert iris.gaze.angles[0] == pytest.approx(-0.05, abs=1e-3)
    off = VideoHeadTracker3D(TrackerSettings(gaze_enabled=False)).process(ev(0, gaze=EyeGaze(0, 0, 0, 0)))
    assert off.gaze is None
    no_fallback = VideoHeadTracker3D(TrackerSettings(iris_gaze_fallback=False)).process(ev(0, mesh478=_mesh()))
    assert no_fallback.gaze is None


def test_gaze_is_smoothed_but_keeps_saccade_speed(actor):
    tracker, rng = VideoHeadTracker3D(), np.random.default_rng(2)
    five = TW.frame(actor, t=0.0, noise_px=0.0)[0]["landmarks5"]
    out = []
    for i in range(90):
        target = 0.0 if i < 45 else 0.25                                          # a 14 degree saccade
        noisy = target + rng.normal(0, 0.01)
        out.append(tracker.process(FrameEvidence(i * DT, five, gaze=EyeGaze(noisy, 0, noisy, 0))).gaze.angles[0])
    out = np.array(out)
    assert np.std(out[10:44]) < 0.01 * 0.8, "fixation noise is suppressed"
    assert out[49] > 0.9 * 0.25 * 0.8 and out[-1] == pytest.approx(0.25, abs=0.02), "within ~4 frames of the saccade"


def test_flame_evidence_supplies_beta_psi_and_jaw_but_not_the_camera(actor):
    rng = np.random.default_rng(4)
    beta, psi = rng.normal(0, 1, 100), rng.normal(0, 0.3, 50)
    turned = rotation_from_euler(40.0, 0.0, 0.0)                                  # a FLAME rotation we must NOT trust
    flame = lambda: FlameParameters(beta + rng.normal(0, 0.02, 100), psi, np.array((0.25, 0.0, 0.0)), turned)
    tracker, out = VideoHeadTracker3D(), []
    for i in range(50):
        obs, _ = TW.frame(actor, yaw=0.0, t=i * DT, noise_px=0.2, seed=i)
        out.append(tracker.process(T.evidence(obs, flame=flame())))
    last = out[-1]
    assert last.identity.kind == KIND_FLAME and last.identity.state == "locked"
    assert np.max(np.abs(last.identity.beta - beta)) < 0.05
    assert last.jaw_source == "flame_encoder" and last.jaw_open_rad == pytest.approx(0.25, abs=0.01)
    assert last.psi.shape == (50,) and np.max(np.abs(last.psi - psi)) < 0.1
    assert abs(last.camera.yaw_deg) < 8.0, "pose comes from the landmarks; the unverifiable FLAME axis convention is not used"
    assert last.expression_energy is not None


# =========================================================================== tracker: shots, gaps, bad input
def test_explicit_cut_resets_filters_identity_and_baseline_and_counts(actor):
    tracker = VideoHeadTracker3D()
    first = T.run(tracker, actor, T.frontal_script(60))
    assert first[-1][0].identity.state == "locked" and first[-1][0].shot_index == 0
    obs, _ = TW.frame(actor, yaw=-12.0, t=60 * DT, noise_px=0.0)
    cut = tracker.process(FrameEvidence(60 * DT, obs["landmarks5"], obs["landmarks106"], scene_cut=True))
    assert cut.shot_cut and cut.cut_reason == "explicit" and cut.shot_index == 1
    assert cut.identity.state == "collecting" and cut.identity.frames <= 1
    assert not cut.expression_calibrated, "a new shot recalibrates its neutral"
    assert cut.camera.yaw_deg == pytest.approx(-12.0, abs=6.0), "no smoothing carried over from the previous shot"
    nxt = tracker.process(T.evidence(TW.frame(actor, yaw=-12.0, t=61 * DT, noise_px=0.0)[0]))
    assert nxt.shot_index == 1 and not nxt.shot_cut
    assert tracker.summary()["shot_cuts"] == 1 and tracker.summary()["shots"] == 2


def test_first_frame_scene_cut_flag_is_not_a_cut(actor):
    obs, _ = TW.frame(actor, t=0.0)
    frame = VideoHeadTracker3D().process(FrameEvidence(0.0, obs["landmarks5"], scene_cut=True))
    assert not frame.shot_cut and frame.shot_index == 0


def test_image_cut_detection_and_no_false_cuts_on_a_steady_scene():
    detector = ShotBoundaryDetector(TrackerSettings())
    steady = [detector.observe(T.flat_image(100, seed=i)) for i in range(20)]
    assert not any(steady)
    assert detector.observe(T.flat_image(210, seed=99)), "a hard cut to a much brighter scene"
    assert not detector.observe(T.flat_image(212, seed=100))
    fade = ShotBoundaryDetector(TrackerSettings())
    fade.observe(T.flat_image(100))
    assert not any(fade.observe(T.flat_image(100 + 4 * k)) for k in range(1, 12)), "a slow fade is not a cut"


def test_image_cut_reaches_the_tracker(actor):
    tracker = VideoHeadTracker3D()
    out = []
    for i in range(12):
        obs, _ = TW.frame(actor, t=i * DT, noise_px=0.2, seed=i)
        out.append(tracker.process(T.evidence(obs, image_bgr=T.flat_image(90 if i < 6 else 220, seed=i))))
    assert [f.cut_reason for f in out if f.shot_cut] == ["image_discontinuity"] and out[6].shot_cut and out[6].shot_index == 1


def test_landmark_jump_is_a_cut_even_without_a_flag_or_image(actor):
    tracker, out = VideoHeadTracker3D(), []
    for i in range(20):
        obs, _ = TW.frame(actor, yaw=5.0, t=i * DT, noise_px=0.2, seed=i)
        out.append(tracker.process(T.evidence(obs)))
    obs, _ = TW.frame(actor, yaw=5.0, t=20 * DT, noise_px=0.2)
    shift = np.array((330.0, 0.0))                                                  # ~3 inter-ocular distances in one frame
    moved = {**obs, "landmarks5": obs["landmarks5"] + shift, "landmarks106": obs["landmarks106"] + shift}
    jump = tracker.process(T.evidence(moved))
    assert jump.shot_cut and jump.cut_reason == "landmark_discontinuity" and jump.shot_index == 1
    assert not any(f.shot_cut for f in out)


def test_an_impossible_pose_spike_is_rejected_not_followed_and_not_a_cut(actor):
    tracker = VideoHeadTracker3D()
    T.run(tracker, actor, [dict(yaw=0.0)] * 30, noise=0.1)
    spike = T.run(tracker, actor, [dict(yaw=75.0)], noise=0.1, start=30)[0][0]       # ~2200 deg/s in one frame
    assert spike.status == "coasting" and spike.reason == "POSE_JUMP_EXCEEDS_PHYSICAL_RATE" and not spike.shot_cut
    assert abs(spike.camera.yaw_deg) < 5.0, "the filters bridged the frame instead of following the spike"
    back = T.run(tracker, actor, [dict(yaw=0.0)], noise=0.1, start=31)[0][0]
    assert back.status == "tracked"


def test_missing_face_is_bridged_then_lost_then_recovered(actor):
    tracker = VideoHeadTracker3D()
    out = T.run(tracker, actor, [dict(yaw=10 + 0.5 * k) for k in range(30)], noise=0.1)
    last_good = out[-1][0]
    nothing = lambda i: FrameEvidence(i * DT, None)
    gap = [tracker.process(nothing(30 + k)) for k in range(14)]
    assert [f.status for f in gap[:7]] == ["coasting"] * 7 and gap[0].reason == "LANDMARKS_MISSING_OR_INVALID"
    assert gap[3].camera is not None and abs(gap[3].camera.yaw_deg - last_good.camera.yaw_deg) < 3.0
    assert gap[3].confidence == 0.0, "a bridged frame is never reported as a confident measurement"
    assert [f.status for f in gap[8:]] == ["lost"] * 6 and gap[-1].camera is None
    for k in range(40):                                                               # a long absence: filters restart, shot is kept
        tracker.process(nothing(44 + k))
    again = T.run(tracker, actor, [dict(yaw=25.0)], noise=0.1, start=84)[0][0]
    assert again.status == "tracked" and again.shot_index == 0 and not again.shot_cut
    assert again.camera.yaw_deg == pytest.approx(25.0, abs=6.0), "no smoothing against the stale pre-gap state"


def test_bad_input_never_raises_and_never_corrupts_state(actor):
    tracker = VideoHeadTracker3D()
    T.run(tracker, actor, T.frontal_script(20))
    obs, _ = TW.frame(actor, t=20 * DT)
    bad = [FrameEvidence(float("nan"), obs["landmarks5"]), FrameEvidence(19 * DT, obs["landmarks5"]),        # time
           FrameEvidence(21 * DT, np.full((5, 2), np.nan)), FrameEvidence(21 * DT, np.zeros((4, 2))),        # landmarks
           FrameEvidence(21 * DT, np.full((5, 2), 1e9)), FrameEvidence(21 * DT, np.zeros((5, 2)))]
    frames = [tracker.process(f) for f in bad]
    assert frames[0].reason == frames[1].reason == "TIMESTAMP_INVALID" and all(f.status in {"lost", "coasting"} for f in frames)
    assert all(f.confidence == 0.0 for f in frames)
    ok = tracker.process(FrameEvidence(22 * DT, obs["landmarks5"], obs["landmarks106"]))
    assert ok.status == "tracked", "state survived the garbage"
    weird106 = tracker.process(FrameEvidence(23 * DT, obs["landmarks5"], np.zeros((106, 2))))
    assert weird106.status == "tracked" and weird106.expression is not None and weird106.expression_held
    assert tracker.summary()["frames"] == 28


def test_frame_observer_failure_does_not_break_tracking(actor, caplog):
    seen = []
    def observer(frame):
        seen.append(frame.frame_index)
        raise RuntimeError("telemetry sink down")
    tracker = VideoHeadTracker3D(on_frame=observer)
    out = T.run(tracker, actor, [dict()] * 5)
    assert all(f.status == "tracked" for f, _ in out) and seen == [0, 1, 2, 3, 4]
    assert "observer failed" in caplog.text


def test_the_observer_sees_every_frame_exactly_once_including_lost_ones(actor):
    seen = []
    tracker = VideoHeadTracker3D(on_frame=seen.append)
    T.run(tracker, actor, [dict()] * 3)
    for k in range(20):
        tracker.process(FrameEvidence((3 + k) * DT, None))
    tracker.process(FrameEvidence(0.0, None))                                         # invalid timestamp
    assert len(seen) == 24 and [f.frame_index for f in seen] == list(range(24))


def test_an_undetected_change_of_person_inside_a_shot_becomes_an_identity_break_cut(actor):
    """Driven with FLAME-beta evidence (no pose dependence), so the wiring is tested on its own. For landmark proportions the
    break is only detectable when the new face still fits the head model: a face unlike the template biases the pose solve
    (here ~13 degrees of pitch), which pushes its frames out of the frontal window, so they are simply not used."""
    rng = np.random.default_rng(8)
    tracker = VideoHeadTracker3D(TrackerSettings(identity_break_run=8, lock_frames=20))
    person_a, person_b = rng.normal(0, 1, 100), rng.normal(0, 1, 100)

    def stream(beta, n, start):
        out = []
        for i in range(n):
            obs, _ = TW.frame(actor, t=(start + i) * DT, noise_px=0.15, seed=start + i, **T.frontal_script(1)[0])
            flame = FlameParameters(beta + rng.normal(0, 0.02, 100), np.zeros(50), np.zeros(3), np.eye(3))
            out.append(tracker.process(T.evidence(obs, flame=flame)))
        return out

    first = stream(person_a, 50, 0)
    assert first[-1].identity.state == "locked" and not any(f.shot_cut for f in first)
    after = stream(person_b, 40, 50)
    cuts = [f for f in after if f.shot_cut]
    assert len(cuts) == 1 and cuts[0].cut_reason == "identity_break" and cuts[0].shot_index == 1
    assert cuts[0].frame_index - 50 <= 8 + 2, "reported within the configured run (plus the one-frame latency)"
    assert after[-1].shot_index == 1 and after[-1].identity.state in {"provisional", "locked"}
    assert np.max(np.abs(after[-1].identity.beta - person_b)) < 0.1 and np.max(np.abs(first[-1].identity.beta - person_a)) < 0.1
    assert tracker.summary()["shot_cuts"] == 1


def test_the_identity_lock_is_per_shot(actor):
    tracker = VideoHeadTracker3D()
    one = T.run(tracker, actor, T.frontal_script(50), noise=0.15)
    other = T.performer(seed=11, width=0.20, jaw_drop=0.02)
    cut = TW.frame(other, t=50 * DT, noise_px=0.15)[0]
    tracker.process(FrameEvidence(50 * DT, cut["landmarks5"], cut["landmarks106"], scene_cut=True))
    two = T.run(tracker, other, T.frontal_script(50), noise=0.15, start=51)
    assert one[-1][0].identity.state == two[-1][0].identity.state == "locked"
    assert not np.allclose(one[-1][0].identity.beta, two[-1][0].identity.beta, rtol=0.03)
    assert two[-1][0].shot_index == 1


def test_per_frame_cost_is_small(actor):
    import time
    tracker = VideoHeadTracker3D()
    script = T.moving_script(60)
    t0 = time.perf_counter()
    T.run(tracker, actor, script)
    assert (time.perf_counter() - t0) / len(script) < 0.05, "measured ~3.5 ms per frame on the reference machine"


def test_settings_are_validated():
    for bad in ({"coast_s": 2.0}, {"lock_frames": 1}, {"ipd_m": 0.5}, {"max_abs_yaw_deg": 95}, {"extra": 1}):
        with pytest.raises(ValueError):
            TrackerSettings(**bad)
