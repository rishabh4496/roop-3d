"""Stage 4: temporal 3D reenactment engine, verified on a scripted synthetic performer with KNOWN pose, expression and timing.

Source identity = one synthetic person (Stage 3 asset built from rendered multi-view images); target = a different person
performing the scripts. The engine sees only landmarks, timestamps and focus measures; assertions compare with the script.
Acceptance criteria under test: no face-shape collapse, no eye/mouth misalignment, no frame-to-frame geometry flicker, no
uncontrolled identity drift, no fabricated geometry confidence, deterministic fallback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import identity_world as W  # noqa: E402
import target_world as TW  # noqa: E402

from roop_3d.core.errors import DomainError  # noqa: E402
from roop_3d.reconstruction.identity_asset import IdentityAssetSettings, build_identity_asset  # noqa: E402
from roop_3d.reconstruction.pose_blend import PoseBlendProjector, rasterize  # noqa: E402
from roop_3d.reconstruction.reenactment import (  # noqa: E402
    EYE_A,
    EYE_B,
    FAR_CAMERA_M,
    INNER_LIP,
    RENDER_MODES,
    ExpressionParams,
    IdentityRig,
    ReenactmentEngine,
    ReenactmentSettings,
    TargetObservation,
    measure_expression,
)
from roop_3d.reconstruction.multiview_fit import batch_rotations  # noqa: E402

SETTINGS = IdentityAssetSettings(uv_size=256)
NEUTRAL = [dict()] * 12                      # calibration warm-up (8 frames) + gate entry (3) + margin


@pytest.fixture(scope="module")
def source():
    return W.make_subject("re_src", W.LAYOUT_68, 201)


@pytest.fixture(scope="module")
def performer():
    return TW.Performer(W.make_subject("re_tgt", W.LAYOUT_68, 202, width=-0.08, nose_depth=0.004, jaw_drop=-0.004))


@pytest.fixture(scope="module")
def rigs(source):
    full = build_identity_asset(W.full_set(source, (0, -40, 40, -72, 72, 0, 0), pitches=[0, 0, 0, 0, 0, -22, 22]), SETTINGS)
    horizontal = build_identity_asset(W.full_set(source, (0, -40, 40, -72, 72)), SETTINGS)
    three = build_identity_asset(W.full_set(source, (0, -40, 40)), SETTINGS)
    front = build_identity_asset(W.make_faceset("re_front", [W.make_view(source, 0, 0.0)]), SETTINGS)
    return {name: IdentityRig(asset) for name, asset in
            (("full", full), ("horizontal", horizontal), ("three", three), ("front", front))}


def run(rig, performer, script, *, settings=None, noise=0.35, seed=0, calibrate=True):
    engine = ReenactmentEngine(rig, settings)
    steps = (NEUTRAL if calibrate else []) + list(script)
    sequence = TW.sequence(performer, steps, noise_px=noise, seed=seed)
    frames = [engine.process(TargetObservation(**obs), TW.CANVAS) for obs, _ in sequence]
    return frames, [truth for _, truth in sequence], engine


def tail(frames, truths, n):
    return list(zip(frames[-n:], truths[-n:]))


# =========================================================================== modes
def test_modes_and_validation():
    assert RENDER_MODES == ("standard", "pose_3d", "reenactment_3d")
    with pytest.raises(ValueError):
        ReenactmentSettings(enter_confidence=1.5)
    assert ExpressionParams(mouth_open=3.0, smile=9.0, blink_a=2.0).clamped().mouth_open <= 0.55
    assert np.array_equal(ExpressionParams.from_vector(ExpressionParams(mouth_open=.2, smile=.1).vector()).vector(),
                          ExpressionParams(mouth_open=.2, smile=.1).vector())


# =========================================================================== pose
def test_frontal_engages_only_after_calibration_and_aligns(rigs, performer):
    frames, truths, _ = run(rigs["horizontal"], performer, [dict()] * 8)
    assert all(not f.engaged and f.reason == "EXPRESSION_NOT_CALIBRATED" for f in frames[:7])
    last = frames[-1]
    assert last.engaged and last.weight == 1.0 and last.reason == "OK" and last.confidence > 0.9
    assert abs(last.pose.yaw) < 3.0 and last.alignment_error_iod < 0.05
    assert last.image.shape == TW.CANVAS + (3,) and last.coverage.shape == TW.CANVAS
    assert 0.0 <= float(last.coverage.min()) and float(last.coverage.max()) <= 1.0 and float((last.coverage > 0.9).mean()) > 0.15
    assert last.image[last.coverage <= 0].sum() == 0, "uncovered pixels must carry no colour"


@pytest.mark.parametrize("yaw,tolerance", [(45, 6.0), (70, 8.0)])
def test_large_yaw_follows_the_target_when_the_identity_was_observed_there(rigs, performer, yaw, tolerance):
    frames, truths, _ = run(rigs["horizontal"], performer, [dict(yaw=yaw)] * 8)
    last = frames[-1]
    assert last.engaged and last.reason == "OK"
    assert abs(last.pose.yaw - yaw) < tolerance
    assert last.alignment_error_iod < 0.15
    assert last.expression_held == (yaw > ReenactmentSettings().expression_max_yaw_deg)


def test_pose_beyond_the_observed_coverage_falls_back_instead_of_extrapolating(rigs, performer):
    # front-only identity: depth is only a prior, so a 45 deg target pose would extrapolate it
    front, _, _ = run(rigs["front"], performer, [dict(yaw=45)] * 8)[0:3]
    assert not front[-1].engaged and front[-1].reason == "POSE_OUTSIDE_OBSERVED_COVERAGE" and front[-1].image is None
    # three-view identity (yaw +-40): 45 is inside the observed range + margin, 70 is not
    inside, _, _ = run(rigs["three"], performer, [dict(yaw=45)] * 8)
    outside, _, _ = run(rigs["three"], performer, [dict(yaw=70)] * 8)
    assert inside[-1].engaged and not outside[-1].engaged and outside[-1].reason == "POSE_OUTSIDE_OBSERVED_COVERAGE"
    assert outside[-1].components["pose_coverage"] == 0.0 and outside[-1].confidence == 0.0
    # an identity with no pitch views refuses large pitch; one with pitch views follows it
    flat, _, _ = run(rigs["horizontal"], performer, [dict(pitch=28)] * 8)
    assert not flat[-1].engaged and flat[-1].reason == "POSE_OUTSIDE_OBSERVED_COVERAGE"


def test_profile_is_refused_not_forced(rigs, performer):
    frames, _, _ = run(rigs["full"], performer, [dict(yaw=88)] * 8)
    assert all(not f.engaged and f.image is None and f.coverage is None for f in frames[-8:])
    assert frames[-1].reason in ("LANDMARK_LAYOUT_IMPLAUSIBLE", "POSE_OUT_OF_SUPPORTED_RANGE", "POSE_OUTSIDE_OBSERVED_COVERAGE")


@pytest.mark.parametrize("pitch", [-28, 28])
def test_looking_up_and_down(rigs, performer, pitch):
    frames, truths, _ = run(rigs["full"], performer, [dict(pitch=pitch)] * 8)
    last = frames[-1]
    assert last.engaged and last.reason == "OK" and abs(last.pose.pitch - pitch) < 8.0 and last.alignment_error_iod < 0.1


def test_roll_is_transferred(rigs, performer):
    frames, _, _ = run(rigs["full"], performer, [dict(roll=25)] * 8)
    assert frames[-1].engaged and abs(frames[-1].pose.roll - 25) < 4.0


def test_fast_head_turn_is_tracked_without_discontinuities(rigs, performer):
    yaws = [0, 8, 20, 36, 52, 60, 62, 60, 50, 30, 12, 0, -10, -10]
    frames, truths, _ = run(rigs["full"], performer, [dict(yaw=y) for y in yaws])
    smooth = np.array([f.smoothed_pose_deg[0] for f in frames[len(NEUTRAL):]])
    assert all(f.engaged for f in frames[len(NEUTRAL):])
    assert np.max(np.abs(np.diff(smooth))) < 20.0                                # 600 deg/s at 30 fps: no jumps
    assert np.max(np.abs(smooth - np.array(yaws, float))) < 16.0                 # bounded lag at 400 deg/s
    assert abs(smooth[-1] - yaws[-1]) < 3.0                                      # settles on the target pose


def test_impossible_pose_jump_is_a_measurement_error_and_does_not_corrupt_state(rigs, performer):
    script = [dict(yaw=20)] * 6 + [dict(yaw=75)] + [dict(yaw=20)] * 6
    frames, truths, _ = run(rigs["full"], performer, script)
    spike = frames[len(NEUTRAL) + 6]
    assert not spike.engaged and spike.reason == "POSE_JUMP_EXCEEDS_PHYSICAL_RATE" and spike.image is None
    before, after = frames[len(NEUTRAL) + 5], frames[len(NEUTRAL) + 7]
    assert abs(after.smoothed_pose_deg[0] - before.smoothed_pose_deg[0]) < 4.0, "the rejected sample leaked into the filters"
    assert frames[-1].engaged and abs(frames[-1].pose.yaw - 20) < 4.0


# =========================================================================== expression
def _series(frames, name):
    return np.array([getattr(f.params, name) for f in frames if f.params is not None])


def test_mouth_opening_is_estimated_and_rendered(rigs, performer):
    openings = [0, .1, .25, .4, .4, .25, .1, 0, 0, 0]
    frames, truths, _ = run(rigs["horizontal"], performer, [dict(mouth_open=o) for o in openings])
    measured = _series(frames[len(NEUTRAL):], "mouth_open")
    truth = np.array(openings)
    assert float(np.corrcoef(measured, truth)[0, 1]) > 0.95
    assert abs(measured[4] - 0.4) < 0.1
    peak, closed = frames[len(NEUTRAL) + 4], frames[-1]
    iod = np.linalg.norm(closed.projected[list(EYE_A)].mean(0) - closed.projected[list(EYE_B)].mean(0))
    drop = (peak.projected[57, 1] - closed.projected[57, 1]) / iod
    assert drop > 0.2 and abs(drop - peak.params.mouth_open) < 0.07, "the source's lower lip must move down by the measured opening"
    # the source atlas has no mouth interior: the compositor must be told to keep the target there
    centre = peak.projected[list(INNER_LIP)].mean(0).round().astype(int)
    assert peak.coverage[centre[1], centre[0]] < 0.2 and closed.coverage[centre[1], centre[0]] > 0.8


def test_talking_follows_rapid_mouth_motion(rigs, performer):
    t = np.arange(40) / TW.FPS
    openings = 0.18 * (1 + np.sin(2 * np.pi * 3.0 * t)) / 1.0
    frames, truths, _ = run(rigs["horizontal"], performer, [dict(mouth_open=float(o)) for o in openings])
    measured = _series(frames[len(NEUTRAL):], "mouth_open")
    assert float(np.corrcoef(measured[4:], openings[4:])[0, 1]) > 0.9
    assert measured.max() > 0.2 and measured.min() < 0.08
    assert all(f.engaged for f in frames[len(NEUTRAL):])


def test_smile_moves_the_source_mouth_corners_and_cheeks(rigs, performer):
    frames, truths, _ = run(rigs["horizontal"], performer, [dict(smile=s) for s in (0, .05, .1, .14, .14, .14, .1, .05, 0, 0)])
    measured = _series(frames[len(NEUTRAL):], "smile")
    assert 0.08 < measured[4] < 0.2 and abs(measured[-1]) < 0.03
    peak, rest = frames[len(NEUTRAL) + 4], frames[-1]
    iod = np.linalg.norm(rest.projected[list(EYE_A)].mean(0) - rest.projected[list(EYE_B)].mean(0))
    rise = ((rest.projected[[48, 54], 1] - peak.projected[[48, 54], 1]).mean()) / iod
    assert 0.06 < rise < 0.25, "mouth corners rise by about the target's smile"
    assert (rest.projected[[48, 54], 0] != peak.projected[[48, 54], 0]).any()


def test_blinks_close_the_source_eyes_and_release_them_to_the_target(rigs, performer):
    blinks = [0, .5, 1, 1, .5, 0, 0, 0]
    frames, truths, _ = run(rigs["horizontal"], performer, [dict(blink_a=b, blink_b=b) for b in blinks])
    a = _series(frames[len(NEUTRAL):], "blink_a")
    assert a.max() > 0.8 and a[-1] < 0.15
    closed, open_ = frames[len(NEUTRAL) + 2], frames[-1]
    centre = closed.projected[list(EYE_A)].mean(0).round().astype(int)
    assert closed.coverage[centre[1], centre[0]] < 0.3 and open_.coverage[centre[1], centre[0]] > 0.8
    assert closed.alignment_error_iod < 0.08 and all(f.engaged for f in frames[len(NEUTRAL):])
    # one eye only (wink): the other eye keeps its texture
    wink, _, _ = run(rigs["horizontal"], performer, [dict(blink_a=b) for b in (0, .6, 1, 1, .6, 0)])
    w = wink[len(NEUTRAL) + 3]
    other = w.projected[list(EYE_B)].mean(0).round().astype(int)
    assert w.params.blink_a > 0.7 and w.params.blink_b < 0.2 and w.coverage[other[1], other[0]] > 0.8


def test_eyebrow_motion_is_transferred(rigs, performer):
    frames, _, _ = run(rigs["horizontal"], performer, [dict(brow=b) for b in (0, .06, .12, .12, .12, .06, 0)])
    raised = frames[len(NEUTRAL) + 3]
    assert raised.params.brow_a > 0.05 and raised.params.brow_b > 0.05


def test_extreme_expression_is_clamped_and_never_collapses_the_face(rigs, performer):
    extreme = dict(mouth_open=1.6, smile=0.6, brow=0.5, mouth_width=0.9, blink_a=1.0, blink_b=1.0)
    frames, _, engine = run(rigs["horizontal"], performer, [dict()] * 2 + [extreme] * 6 + [dict()] * 6)
    rig = engine.rig
    for f in frames[len(NEUTRAL):]:
        if f.params is None:
            continue
        p = f.params
        assert 0 <= p.mouth_open <= 0.55 and -0.12 <= p.smile <= 0.22 and 0 <= p.blink_a <= 1 and -0.12 <= p.brow_a <= 0.16
        kept = np.einsum("ij,ij->i", rig._face_normals(f.vertices), rig.rest_normals_face)
        assert (kept[rig.fold_faces] > 0.15).all(), "a rigidly moving triangle flipped over (fold)"
        # the guard may damp an out-of-range expression to avoid a fold, but it always REPORTS that, and never to nothing
        assert 0.4 <= f.deformation_strength <= 1.0
        assert np.max(np.linalg.norm(f.vertices - rig.V0, axis=1)) <= 0.0451, "displacement beyond the cap"
        edge = lambda v: np.linalg.norm(v[rig.faces[:, 0]] - v[rig.faces[:, 1]], axis=1)
        ratio = edge(f.vertices) / np.maximum(edge(rig.V0), 1e-9)
        # only triangles of the mouth / eye / brow features (released to the target) may stretch strongly; nothing collapses
        strong = ratio > 3.0
        mid = (rig.T[rig.faces[:, 0]] + rig.T[rig.faces[:, 1]]) / 2.0
        in_features = ((np.abs(mid[:, 0]) < 0.06) & (mid[:, 1] > 0.03) & (mid[:, 1] < 0.08)) | (np.abs(mid[:, 1] + 0.016) < 0.016)             | (mid[:, 1] < -0.02)
        assert not np.any(strong & ~in_features), "a strongly stretched triangle outside the eyes / mouth / brows"
        assert float(ratio.max()) < 8.0 and float(ratio.min()) > 0.2
        iod = np.linalg.norm(f.vertices[list(EYE_A)].mean(0) - f.vertices[list(EYE_B)].mean(0))
        assert abs(iod / rig.iod - 1) < 0.05, "the face width changed: shape collapse"
    assert frames[-1].engaged, "must recover to the neutral face after the extreme"
    assert frames[-1].deformation_strength == 1.0, "a neutral face is never damped"
    ordinary, _, _ = run(rigs["horizontal"], performer, [dict(mouth_open=0.3, smile=0.1, brow=0.08, blink_a=1.0)] * 6)
    assert all(f.deformation_strength == 1.0 for f in ordinary[len(NEUTRAL):]), "in-range expressions are transferred in full"


# =========================================================================== temporal stability
def test_no_geometry_flicker_on_noisy_landmarks(rigs, performer):
    noisy, truths, _ = run(rigs["horizontal"], performer, [dict(yaw=25, mouth_open=0.12)] * 60, noise=0.9)
    frames = [f for f in noisy[len(NEUTRAL) + 6:] if f.engaged]
    raw = np.array([t["p2d"] for t in truths[len(NEUTRAL) + 6:]])                 # noisy detector landmarks
    out = np.array([f.projected[[36, 45, 30, 48, 54]] for f in frames])           # rendered source eye / nose / mouth corners
    jitter_in = np.mean(np.std(np.diff(raw[:, [36, 45, 30, 48, 54]], axis=0), axis=0))
    jitter_out = np.mean(np.std(np.diff(out, axis=0), axis=0))
    assert jitter_out < 0.5 * jitter_in, f"rendered geometry jitters ({jitter_out:.3f}px) as much as the input ({jitter_in:.3f}px)"
    assert np.std(np.array([f.params.mouth_open for f in frames])) < 0.02


def test_gate_does_not_flicker_when_confidence_hovers_around_the_thresholds(rigs, performer):
    """Confidence is driven across the enter (0.6) / exit (0.4) thresholds every other frame through the focus measure.
    The focus reference adapts slowly, so a 'good' value is ~full reference and a 'bad' one a fifth of it."""
    GOOD, BAD = 100.0, 20.0

    def drive(prefix, pattern, frames=30):
        engine = ReenactmentEngine(rigs["horizontal"])
        for obs, _ in TW.sequence(performer, NEUTRAL + [dict(yaw=10)] * 4, noise_px=0.2):
            engine.process(TargetObservation(**{**obs, "sharpness": GOOD}), TW.CANVAS)
        out, t = [], 2.0
        for i, sharp in enumerate(list(prefix) + [pattern[i % len(pattern)] for i in range(frames)]):
            obs, _ = TW.frame(performer, yaw=10, t=t, noise_px=0.2, seed=i)
            t += 1 / TW.FPS
            out.append(engine.process(TargetObservation(**{**obs, "sharpness": sharp}), TW.CANVAS))
        return out[len(prefix):]
    # engaged; confidence alternates between good and BELOW the exit threshold, but never two bad frames in a row
    engaged = drive([], [GOOD, BAD])
    assert all(f.engaged for f in engaged), "a single bad frame must not drop the engine"
    assert sum(f.confidence < 0.4 for f in engaged) >= 12, "premise: confidence really crossed the exit threshold repeatedly"
    # withdrawn; confidence alternates good / bad: never `enter_frames` good frames in a row -> never re-engages
    idle = drive([BAD] * 6, [GOOD, BAD])                     # 6 bad frames: withdrawn AND the cross-fade has fully faded out
    assert not any(f.engaged for f in idle) and sum(f.confidence >= 0.6 for f in idle) >= 12
    # sustained bad focus withdraws it, sustained good focus re-admits it
    assert not drive([], [BAD], frames=6)[-1].engaged
    back = drive([BAD] * 4, [GOOD], frames=10)
    assert back[-1].engaged and not back[0].engaged


def test_engine_ramps_in_and_out_instead_of_switching(rigs, performer):
    settings = ReenactmentSettings()
    frames, _, _ = run(rigs["horizontal"], performer, [dict()] * 12)
    weights = [f.weight for f in frames if f.engaged]
    assert weights == sorted(weights) and weights[0] < 1.0 and weights[-1] == 1.0
    assert np.max(np.diff(weights)) <= 1.0 / settings.ramp_frames + 1e-9


# =========================================================================== fallback conditions
def test_missing_or_corrupt_landmarks_fall_back_with_a_reason(rigs, performer):
    engine = ReenactmentEngine(rigs["horizontal"])
    for i, (obs, _) in enumerate(TW.sequence(performer, NEUTRAL + [dict()] * 6)):
        engine.process(TargetObservation(**obs), TW.CANVAS)
    base, _ = TW.frame(performer, t=99 / TW.FPS)
    t = [100 / TW.FPS]

    def feed(**overrides):
        t[0] += 1 / TW.FPS
        return engine.process(TargetObservation(**{**base, "timestamp": t[0], **overrides}), TW.CANVAS)

    assert engine.engaged
    assert feed(landmarks106=None).reason == "LANDMARKS_MISSING_OR_INVALID"
    assert feed(landmarks5=None).reason == "LANDMARKS_MISSING_OR_INVALID"
    bad = base["landmarks106"].copy()
    bad[10] = np.nan
    assert feed(landmarks106=bad).reason == "LANDMARKS_MISSING_OR_INVALID"
    assert feed(landmarks106=base["landmarks106"] + 5000).reason == "LANDMARKS_MISSING_OR_INVALID"
    assert feed(landmarks106=np.random.default_rng(0).uniform(0, 400, (106, 2))).reason == "LANDMARK_LAYOUT_IMPLAUSIBLE"
    assert feed(landmarks106=base["landmarks106"][:50]).reason == "LANDMARKS_MISSING_OR_INVALID"
    assert feed(timestamp=0.0).reason == "TIMESTAMP_INVALID"
    # each of those withdrew the engine at once, and re-entry needs `enter_frames` good frames (no instant flip-flop)
    assert not engine.engaged
    recovered = [feed() for _ in range(6)]
    assert [f.engaged for f in recovered][:2] == [False, False] and recovered[-1].engaged
    result = feed(landmarks106=None)
    assert result.image is None and result.coverage is None and result.confidence == 0.0 and result.weight == 0.0


def test_scene_cut_discards_all_temporal_state(rigs, performer):
    engine = ReenactmentEngine(rigs["horizontal"])
    first = TW.sequence(performer, NEUTRAL + [dict(yaw=40)] * 8, noise_px=0.2)
    for obs, _ in first:
        out = engine.process(TargetObservation(**obs), TW.CANVAS)
    assert out.engaged and out.smoothed_pose_deg[0] > 30
    cut_obs, _ = TW.frame(performer, yaw=0.0, t=10.0, scene_cut=True)
    after = engine.process(TargetObservation(**cut_obs), TW.CANVAS)
    assert not after.engaged and after.reason == "EXPRESSION_NOT_CALIBRATED"
    assert abs(after.smoothed_pose_deg[0]) < 3.0, "the previous shot's pose dragged into the new one"
    assert engine.frame_index == 1 and engine.ramp == 0.0
    # same timestamps as before the cut are legal after it (state was reset)
    again = TW.frame(performer, t=0.0)[0]
    assert engine.process(TargetObservation(**{**again, "timestamp": 10.1}), TW.CANVAS).reason == "EXPRESSION_NOT_CALIBRATED"


def test_motion_blur_lowers_confidence_and_withdraws_the_engine(rigs, performer):
    script = [dict(yaw=10)] * 6
    engine = ReenactmentEngine(rigs["horizontal"])
    steps = NEUTRAL + script
    sequence = TW.sequence(performer, steps, noise_px=0.3)
    for obs, _ in sequence:
        out = engine.process(TargetObservation(**{**obs, "sharpness": 100.0}), TW.CANVAS)
    assert out.engaged and out.components["sharpness"] == 1.0
    t0 = 20.0
    blurred = []
    for i in range(6):
        obs, _ = TW.frame(performer, yaw=10, t=t0 + i / TW.FPS, noise_px=0.3, seed=i)
        blurred.append(engine.process(TargetObservation(**{**obs, "sharpness": 6.0}), TW.CANVAS))
    assert blurred[-1].components["sharpness"] < 0.2 and not blurred[-1].engaged
    assert blurred[-1].reason.startswith("LOW_GEOMETRY_CONFIDENCE:sharpness") or blurred[-1].reason == "CONFIDENCE_BELOW_THRESHOLD"
    # an unmeasured focus value is reported as unmeasured, never as good
    obs, _ = TW.frame(performer, yaw=10, t=t0 + 1.0)
    assert engine.process(TargetObservation(**{**obs, "sharpness": None}), TW.CANVAS).components["sharpness"] is None


def test_confidence_is_the_minimum_of_measured_components_only(rigs, performer):
    frames, _, _ = run(rigs["horizontal"], performer, [dict(yaw=70, sharpness=None)] * 6)
    f = frames[-1]
    measured = [v for v in f.components.values() if v is not None]
    assert f.confidence == pytest.approx(min(measured)) and 0.0 <= f.confidence <= 1.0
    assert f.components["alignment"] < 1.0 and f.confidence < 0.9      # a hard pose is NOT reported as a confident one
    assert f.components["sharpness"] is None                           # not measured -> not counted


# =========================================================================== identity retention
def test_source_identity_is_retained_and_does_not_drift(rigs, performer, source):
    rig = rigs["horizontal"]
    asset_hash = {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in rig.asset.array_fields.items()}
    rng = np.random.default_rng(3)
    script = [dict(yaw=float(rng.uniform(-30, 30)), pitch=float(rng.uniform(-3, 3)), mouth_open=float(rng.uniform(0, .35)),
                   smile=float(rng.uniform(-.05, .15)), blink_a=float(rng.choice([0, 0, 0, 1])), brow=float(rng.uniform(-.05, .1)))
              for _ in range(120)] + [dict()] * 40
    frames, _, engine = run(rig, performer, script, noise=0.5)
    assert {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in rig.asset.array_fields.items()} == asset_hash
    neutral = [f for f in frames[-12:] if f.engaged]
    assert neutral, "the engine must recover after 120 frames of random performance"
    # back at neutral, the geometry returns to the SOURCE rest shape (nothing accumulated)
    assert max(float(np.abs(f.vertices - rig.V0).max()) for f in neutral[-4:]) < 0.0035
    # the rendered proportions are the source's, not the target's
    last = neutral[-1]
    # geometry: the rendered rest face IS the source's landmark shape, not the target's (after removing size/translation)
    err = lambda est, ref: float(np.linalg.norm(W.sim_align(est, ref) - ref, axis=1).mean() * 1000)
    target_vs_source = err(performer.subject.true_lm, source.true_lm)
    rendered_vs_source = err(last.vertices[:68], source.true_lm)
    assert target_vs_source > 3.0, "test premise: the two people must differ"
    assert rendered_vs_source < 0.3 * target_vs_source
    assert err(last.vertices[:68], performer.subject.true_lm) > 0.8 * target_vs_source
    # the neutral render is the SOURCE person's own appearance: compare with an independent render of the true source head
    cam = last.camera
    truth = W.render(source, 0, 0, 0, cam=(cam.rotation_matrix(), cam.scale, np.array([cam.tx, cam.ty])), shape=TW.CANVAS)
    inside = last.coverage > 0.9
    assert inside.sum() > 3000
    assert float(np.abs(last.image[inside].astype(float) - truth[inside].astype(float)).mean()) < 25.0
    footprint = np.array(np.where((truth.astype(int) - 95 > 0).any(axis=2), True, False))
    assert rig.build_key == engine.rig.build_key


def test_engine_is_deterministic(rigs, performer):
    script = [dict(yaw=15 * np.sin(i / 4), mouth_open=0.2 * abs(np.sin(i / 3)), blink_a=float(i % 17 == 0)) for i in range(40)]
    a, _, _ = run(rigs["horizontal"], performer, script, seed=11)
    b, _, _ = run(rigs["horizontal"], performer, script, seed=11)
    digest = lambda frames: hashlib.sha256(b"".join(
        (f.image.tobytes() + f.coverage.tobytes() if f.image is not None else f.reason.encode()) + f"{f.confidence:.9f}".encode()
        for f in frames)).hexdigest()
    assert digest(a) == digest(b)


# =========================================================================== correspondence / renderer
def test_eye_block_order_of_the_detector_does_not_matter(rigs):
    tgt = W.make_subject("re_tgt2", W.LAYOUT_68, 202, width=-0.08, nose_depth=0.004, jaw_drop=-0.004)
    results = []
    for swap in (False, True):
        perf = TW.Performer(tgt, swap_eye_blocks=swap)
        frames, _, _ = run(rigs["horizontal"], perf, [dict(blink_a=1.0)] * 2 + [dict(blink_a=1.0, mouth_open=.3)] * 3)
        results.append(frames[-1].params)
    assert results[0].blink_a == pytest.approx(results[1].blink_a, abs=0.05) and results[0].blink_b == pytest.approx(results[1].blink_b, abs=0.05)
    assert results[0].mouth_open == pytest.approx(results[1].mouth_open, abs=0.03) and results[0].blink_a > 0.6


def test_expression_measure_is_invariant_to_roll_and_yaw_foreshortening(performer):
    base = TW.frame(performer, mouth_open=0.3, t=0.0, noise_px=0.0)[0]
    reference = measure_expression(base["landmarks5"], base["landmarks106"], 0.0)
    for roll in (-25, 30):
        obs = TW.frame(performer, roll=roll, mouth_open=0.3, t=0.0, noise_px=0.0)[0]
        m = measure_expression(obs["landmarks5"], obs["landmarks106"], 0.0)
        assert m.mouth_open == pytest.approx(reference.mouth_open, abs=0.02) and m.smile == pytest.approx(reference.smile, abs=0.02)
    obs = TW.frame(performer, yaw=40, mouth_open=0.3, t=0.0, noise_px=0.0)[0]
    yawed = measure_expression(obs["landmarks5"], obs["landmarks106"], 40.0)
    assert yawed.mouth_open == pytest.approx(reference.mouth_open, abs=0.04)
    with pytest.raises(DomainError):
        measure_expression(np.zeros((5, 2)), base["landmarks106"], 0.0)


def test_fast_renderer_matches_the_reference_rasterizer(rigs, performer):
    frames, _, engine = run(rigs["horizontal"], performer, [dict(yaw=30, mouth_open=0.1)] * 4)
    f = frames[-1]
    rot = batch_rotations(np.array([list(f.smoothed_pose_deg)]))[0]
    camera_vertices = engine._camera_vertices(f.vertices, rot)
    rig = engine.rig
    reference, _, _ = rasterize(rig.texture, rig.texture_coverage, f.projected, camera_vertices, rig.uv_pixels,
                                rig.faces[:, [0, 2, 1]], TW.CANVAS, perspective=True)
    assert float(np.abs(f.image.astype(int) - reference.astype(int)).mean()) < 1.0
    assert FAR_CAMERA_M > 10


# =========================================================================== 3D Pose mode (identity-specific projection)
def test_pose_projector_with_identity_geometry_fits_the_source_better_than_the_generic_proxy(source, rigs):
    rig = rigs["horizontal"]
    img = W.render(source, 35, 4, 0)
    landmarks = W.project_landmarks(source, 35, 4, 0, noise_px=0.2)
    five = W.five_from_dense(landmarks)
    crop = img[100:500, 40:520]
    observed = five - np.array([40, 100])
    generic = PoseBlendProjector().render(crop, observed, 1.0)
    identity = PoseBlendProjector.from_identity_asset(rig.asset).render(crop, observed, 1.0)
    assert identity.reprojection_error < generic.reprojection_error
    assert identity.coverage.shape == crop.shape[:2] and float(identity.coverage.max()) > 0.9
    assert PoseBlendProjector().geometry == "ESTIMATED_CANONICAL_PROXY_68"                 # default behaviour is unchanged
    with pytest.raises(DomainError):
        PoseBlendProjector(vertices=np.zeros((4, 3)))                                      # geometry must come as a complete set


def test_large_anatomy_mismatch_is_aligned_but_never_reported_as_confident(rigs, source):
    """Five landmarks cannot separate head pitch from a different mouth-to-eye proportion. With a target whose lower face
    differs by ~22 mm the engine must keep the features aligned and stay safe, and must not claim more confidence than
    the pose-range gate supports."""
    strong = TW.Performer(W.make_subject("re_tgt_strong", W.LAYOUT_68, 203, width=-0.10, nose_depth=0.004, jaw_drop=-0.012))
    frames, _, _ = run(rigs["full"], strong, [dict(yaw=20, mouth_open=0.2)] * 8)
    last = frames[-1]
    assert last.pose is not None and last.alignment_error_iod < 0.1, "features must still land on the target's"
    if last.engaged:
        assert last.confidence == pytest.approx(min(v for v in last.components.values() if v is not None))
    mild = TW.Performer(W.make_subject("re_tgt_mild", W.LAYOUT_68, 204, width=-0.08, nose_depth=0.004, jaw_drop=-0.004))
    reference, _, _ = run(rigs["full"], mild, [dict(yaw=20, mouth_open=0.2)] * 8)
    assert last.components["pose_coverage"] <= reference[-1].components["pose_coverage"] + 1e-9
