"""Scene adaptation (the 16-frame / 3-second calibration pass), the video runner and pipeline tap, and live telemetry on
``/ws/v1/pipeline/status``.

Adaptation is judged on frames it never saw, against a head that differs from the canonical one. The WebSocket tests drive the
real app (``create_app``) and also prove the non-regression that matters: tracking frames never touch ``/ws/events`` or
``/ws/telemetry`` (no extra messages, no sequence gaps, no evicted job events).
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import tracking_world as T  # noqa: E402
import target_world as TW  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

from roop_3d.server.app import create_app  # noqa: E402
from roop_3d.server.ws import EVENT_ADAPTER, EventBroker, StreamGapEvent, TrackingTelemetryEvent  # noqa: E402
from roop_3d.tracking.lighting import MeshSHLightingEstimator  # noqa: E402
from roop_3d.tracking.runner import (  # noqa: E402
    AnalyzedFaceExtractor, HeadTrackingTap, SourceFrame, VideoFrameServiceSource, calibrate_video, track_video)
from roop_3d.tracking.scene_adaptation import SceneAdaptation, SceneTrackerAdaptation, sample_frame_indices  # noqa: E402
from roop_3d.tracking.telemetry import EVENT_TYPE, TrackingTelemetryPublisher, telemetry_payload  # noqa: E402
from roop_3d.tracking.video_tracker import FrameEvidence, HeadAnchors, VideoHeadTracker3D  # noqa: E402

DT = 1 / 30


@pytest.fixture(scope="module")
def actor():
    return T.performer()


@pytest.fixture(scope="module")
def take(actor):
    """150 frames of a moving performance by a head that is NOT the canonical template, with the truth alongside."""
    return TW.sequence(actor, T.moving_script(150), noise_px=0.35, seed=3)


def spaced(take, count=16):
    picks = np.linspace(5, len(take) - 5, count).astype(int)
    return picks, [T.evidence(take[i][0]) for i in picks]


# =========================================================================== adaptation
def test_sample_indices_cover_the_video_evenly():
    picks = sample_frame_indices(1600, 16)
    assert len(picks) == 16 and picks[0] < 100 and picks[-1] > 1500 and picks == sorted(set(picks))
    assert np.all(np.diff(picks) == 100)
    assert sample_frame_indices(5, 16) == [0, 1, 2, 3, 4] and sample_frame_indices(0) == [] and sample_frame_indices(10, 0) == []


def test_adaptation_improves_unseen_frames_on_a_head_unlike_the_template(actor, take):
    picks, samples = spaced(take)
    adaptation = SceneTrackerAdaptation().calibrate(samples)
    report = adaptation.report
    assert report.anchors_applied and report.frames_used == 16 and report.frames_requested == 16
    assert report.elapsed_s < 3.0 and not report.budget_exhausted, f"measured ~0.3 s; took {report.elapsed_s:.2f}s"
    before, after = report.holdout_residual_mm
    assert after < 0.9 * before, f"held-out landmark residual {before:.3f} -> {after:.3f} mm (measured 1.05 -> 0.82)"
    assert report.yaw_span_deg > 40 and report.depth_observed > 0.5 and 0.2 < report.anchor_shift_mm < 8.0

    unseen = [i for i in range(len(take)) if i not in set(picks)]
    def errors(tracker):
        out = []
        for i in range(len(take)):
            frame = tracker.process(T.evidence(take[i][0]))
            if i in unseen and i >= 20 and frame.status == "tracked":
                truth = take[i][1]
                out.append([abs(frame.camera.yaw_deg - truth["yaw"]), abs(frame.camera.pitch_deg - truth["pitch"]),
                            abs(frame.camera.roll_deg - truth["roll"])])
        return np.mean(out, axis=0)
    generic, adapted = errors(VideoHeadTracker3D()), errors(VideoHeadTracker3D(adaptation=adaptation))
    # Measured yaw/pitch/roll: 4.07/2.99/1.18 -> 2.55/2.80/1.26 degrees on frames the pass never saw: a clear yaw gain, no channel
    # materially worse. (The remaining error is the limit of 5 landmarks plus expression, not of the solver.)
    assert np.all(adapted <= generic * 1.15) and adapted.mean() < 0.85 * generic.mean(), (generic, adapted)


def test_adaptation_leaves_a_head_that_already_matches_alone():
    anchors, rng = HeadAnchors.canonical(), np.random.default_rng(6)
    centre, samples = anchors.points.mean(axis=0), []
    from roop_3d.reconstruction.multiview_fit import batch_rotations
    for i in range(16):
        rotation = batch_rotations(rng.uniform([-35, -12, -6], [35, 12, 6], (1, 3)))[0]
        five = 2200.0 * ((anchors.points - centre) @ rotation.T)[:, :2] + np.array((300.0, 300.0)) + rng.normal(0, 0.2, (5, 2))
        samples.append(FrameEvidence(i * DT, five))
    result = SceneTrackerAdaptation().calibrate(samples)
    assert not result.report.anchors_applied and result.anchors is None
    assert any("generic head is kept" in w for w in result.report.warnings)


def test_too_few_or_unusable_frames_adapt_nothing_and_say_why(take):
    garbage = [FrameEvidence(0.0, None), FrameEvidence(0.1, np.full((5, 2), np.nan)), FrameEvidence(0.2, np.zeros((5, 2))),
               FrameEvidence(0.3, np.zeros((3, 2)))]
    result = SceneTrackerAdaptation().calibrate(garbage + [T.evidence(take[0][0])])
    assert result.anchors is None and result.expression_baseline is None and result.identity is None and result.lighting is None
    assert result.report.frames_requested == 5 and result.report.frames_used < 6
    assert result.report.warnings and "at least 6" in result.report.warnings[0]
    with pytest.raises(ValueError):
        SceneTrackerAdaptation(frames=1)
    with pytest.raises(ValueError):
        SceneTrackerAdaptation(budget_s=0)


def test_the_time_budget_is_enforced_and_reported(take):
    ticks = iter(range(10_000))
    adaptation = SceneTrackerAdaptation(budget_s=3.0, clock=lambda: float(next(ticks)))       # every call costs "1 second"
    result = adaptation.calibrate(spaced(take)[1])
    assert result.report.budget_exhausted and result.report.elapsed_s >= 3.0
    assert result.report.rounds < SceneTrackerAdaptation().rounds
    assert any("time budget" in w for w in result.report.warnings)
    assert isinstance(result, SceneAdaptation), "the pass still returns what it managed"


def test_extraction_and_adaptation_share_one_budget(actor, take):
    clock = [0.0]
    source = ListSource([SourceFrame(i, i * DT, np.zeros((8, 8, 3), np.uint8), take[i][0]) for i in range(len(take))],
                        on_read=lambda: clock.__setitem__(0, clock[0] + 1.0))                    # each decode costs one second
    adaptation = SceneTrackerAdaptation(budget_s=3.0, clock=lambda: clock[0])
    result = calibrate_video(source, lambda frame: T.evidence(frame.raw), adaptation)
    assert len(source.reads) == 3, "decoding stops when the budget is spent"
    assert result.report.frames_requested == 16 and result.report.frames_used == 3 and result.anchors is None
    assert result.report.budget_exhausted


class ListSource:
    def __init__(self, frames, on_read=None, fail=()):
        self.frames, self.total_frames, self.on_read, self.fail, self.reads = frames, len(frames), on_read, set(fail), []

    def read(self, index):
        self.reads.append(index)
        if self.on_read:
            self.on_read()
        if index in self.fail:
            raise OSError("corrupt packet")
        return self.frames[index]


def test_calibrate_video_samples_the_expected_frames_and_counts_decode_failures(actor, take, caplog):
    frames = [SourceFrame(i, i * DT, np.zeros((8, 8, 3), np.uint8), take[i][0]) for i in range(len(take))]
    expected = SceneTrackerAdaptation().frame_indices(len(frames))
    source = ListSource(frames, fail={expected[3]})
    result = calibrate_video(source, lambda frame: T.evidence(frame.raw))
    assert source.reads == expected and len(expected) == 16
    assert result.report.frames_used == 15 and result.report.frames_requested == 16
    assert any("failed to decode" in w for w in result.report.warnings) and "could not be read" in caplog.text
    assert result.report.anchors_applied


def test_neutral_baseline_removes_the_warm_up(actor, take):
    _, samples = spaced(take)
    adaptation = SceneTrackerAdaptation().calibrate(samples)
    assert adaptation.expression_baseline is not None and adaptation.report.expression_samples >= 12
    cold = VideoHeadTracker3D().process(T.evidence(TW.frame(actor, t=0.0, noise_px=0.2)[0]))
    warm = VideoHeadTracker3D(adaptation=adaptation).process(T.evidence(TW.frame(actor, t=0.0, noise_px=0.2)[0]))
    assert cold.expression is None and not cold.expression_calibrated, "an unknown actor needs a warm-up"
    assert warm.expression_calibrated and warm.expression is not None, "a calibrated one does not"
    assert max(abs(v) for v in warm.expression.values()) < 0.15, "and reads near-neutral at rest (baseline from 16 posed frames)"


def test_lighting_environment_is_pooled_and_judged_static_or_not(actor):
    left, right = T.sh_environment((-0.7, 0.1, 0.7)), T.sh_environment((0.7, 0.1, 0.7))
    estimator = MeshSHLightingEstimator()
    def samples(environment):
        out = []
        for i in range(16):
            image, five, lm106 = T.lit_frame(actor.subject, -20 + 40 * i / 15, 0, environment(i))
            out.append(FrameEvidence(i * DT, five, lm106, image_bgr=image))
        return out
    steady = SceneTrackerAdaptation(lighting=estimator).calibrate(samples(lambda i: left))
    assert steady.lighting_static and steady.lighting.shape == (9, 3) and steady.report.lighting_samples == 16
    assert steady.report.lighting_spread_deg < 10.0
    key = T.SH.extract_key_light_direction(steady.lighting)
    assert key[0] < -0.4, "the pooled environment points at the real (left) light"
    flickering = SceneTrackerAdaptation(lighting=estimator).calibrate(samples(lambda i: left if i % 2 else right))
    assert not flickering.lighting_static and flickering.report.lighting_spread_deg > 25.0

    tracker = VideoHeadTracker3D(adaptation=steady)                                 # no estimator, no image: the prior answers
    obs, _ = TW.frame(actor, t=0.0)
    frame = tracker.process(FrameEvidence(0.0, obs["landmarks5"]))
    assert frame.lighting_source == "adaptation_prior" and np.allclose(frame.lighting_coefficients, steady.lighting)
    assert VideoHeadTracker3D(adaptation=flickering).process(FrameEvidence(0.0, obs["landmarks5"])).lighting_coefficients is None


def test_identity_prior_agrees_with_the_same_actor_and_flags_another(actor):
    script = T.frontal_script(60)
    seq = TW.sequence(actor, script, noise_px=0.2, seed=1)
    adaptation = SceneTrackerAdaptation().calibrate([T.evidence(seq[i][0]) for i in np.linspace(0, 59, 16).astype(int)])
    assert adaptation.identity is not None and adaptation.identity[0] == "landmark_proportions_106"
    tracker = VideoHeadTracker3D(adaptation=adaptation)
    last = [tracker.process(T.evidence(o)) for o, _ in seq][-1]
    assert last.identity.state == "locked" and last.identity.prior_distance < 2.0 and last.components["identity"] == 1.0
    mismatch = VideoHeadTracker3D(adaptation=adaptation)
    mismatch.lock.set_prior(adaptation.identity[0], adaptation.identity[1] * 1.4, adaptation.identity[2])
    last = [mismatch.process(T.evidence(o)) for o, _ in seq][-1]
    assert last.identity.prior_distance > 3.0 and last.components["identity"] == 0.5


def test_adaptation_survives_a_json_round_trip(actor):
    environment, samples = T.sh_environment((-0.7, 0.1, 0.7)), []
    for i in range(16):
        image, five, lm106 = T.lit_frame(actor.subject, -10 + 20 * i / 15, 0, environment)
        samples.append(FrameEvidence(i * DT, five, lm106, image_bgr=image))
    original = SceneTrackerAdaptation(lighting=MeshSHLightingEstimator()).calibrate(samples)
    assert original.identity is not None and original.lighting is not None
    restored = SceneAdaptation.from_dict(json.loads(json.dumps(original.to_dict())))
    assert np.allclose(restored.anchors.points, original.anchors.points) and restored.anchors.personalised
    assert np.allclose(restored.expression_baseline, original.expression_baseline)
    assert restored.asymmetry_baseline == pytest.approx(original.asymmetry_baseline)
    assert restored.identity[0] == original.identity[0] and np.allclose(restored.identity[1], original.identity[1])
    assert restored.report == original.report
    obs, _ = TW.frame(actor, yaw=15.0, t=0.0, noise_px=0.0)
    a, b = (VideoHeadTracker3D(adaptation=x).process(T.evidence(obs)) for x in (original, restored))
    assert a.camera.yaw_deg == pytest.approx(b.camera.yaw_deg, abs=1e-6)


# =========================================================================== runner and tap
def test_track_video_visits_frames_reports_cuts_and_finishes_the_publisher(actor):
    sent, steps = [], T.frontal_script(40)
    frames = []
    for i, step in enumerate(steps):
        obs, _ = TW.frame(actor, t=i * DT, noise_px=0.2, seed=i, **step)
        frames.append(SourceFrame(i, i * DT, T.flat_image(90 if i < 20 else 220, seed=i), obs))
    tracker = VideoHeadTracker3D()
    publisher = TrackingTelemetryPublisher(sent.append, job_id="j", max_hz=1000)
    tracker.on_frame = publisher
    summary = track_video(ListSource(frames), lambda f: T.evidence(f.raw, image_bgr=f.image), tracker, publisher=publisher)
    assert summary["visited"] == 40 and summary["shot_cuts"] == 1 and summary["frames"] == 40
    assert sent[-1]["state"] == "ended" and sum(1 for p in sent if p["shot_cut"]) == 1 and sent[-1]["shot_cuts"] == 1
    stopped = track_video(ListSource(frames), lambda f: T.evidence(f.raw), VideoHeadTracker3D(), start=5, step=3, stop=30,
                          should_stop=iter([False, False, True]).__next__)
    assert stopped["visited"] == 2
    missing = track_video(ListSource(frames[:3]), lambda f: None, VideoHeadTracker3D())
    assert missing["status"]["lost"] == 3, "a frame with no face is a lost frame, not an error"
    with pytest.raises(ValueError):
        track_video(ListSource(frames), lambda f: None, VideoHeadTracker3D(), step=0)


class FakeFace:
    def __init__(self, track_id, five, lm106, box=(0, 0, 100, 100), mesh=None):
        self.track_id, self.box, self.kps5, self.kps106 = track_id, np.array(box, float), five, lm106
        self.observation = type("Obs", (), dict(landmarks5=five, landmarks106=lm106, landmarks478=mesh))()


def test_pipeline_tap_tracks_each_face_publishes_one_track_and_isolates_failures(actor, caplog):
    published = []
    tap = HeadTrackingTap(publisher_factory=lambda track_id: TrackingTelemetryPublisher(published.append, track_id=track_id, max_hz=1000),
                          max_tracks=2)
    for i in range(6):
        obs, _ = TW.frame(actor, t=i * DT, noise_px=0.2, seed=i)
        other, _ = TW.frame(actor, yaw=20.0, t=i * DT, noise_px=0.2, seed=i)
        faces = [FakeFace(7, obs["landmarks5"], obs["landmarks106"]), FakeFace(3, other["landmarks5"], other["landmarks106"]),
                 FakeFace(9, other["landmarks5"], other["landmarks106"])]
        results = tap.observe(None, faces, i * DT, False)
    assert set(results) == {7, 3} and len(tap.trackers) == 2, "max_tracks bounds the work; the third face is ignored"
    assert {p["track_id"] for p in published} == {7}, "telemetry is one track's, never an interleaving of two people"
    assert results[3].camera.yaw_deg > results[7].camera.yaw_deg + 8
    # failures stay inside the tap
    broken = FakeFace(7, "not landmarks", None)
    assert tap.observe(None, [broken], 99 * DT, False) == {} and "Head tracking failed" in caplog.text
    tap.release()
    assert published[-1]["state"] == "ended" and not tap.trackers
    only = HeadTrackingTap(track_ids={3})
    assert set(only.observe(None, faces, 0.0, False)) == {3}
    cut = HeadTrackingTap()
    cut.observe(None, faces[:1], 0.0, False)
    assert cut.observe(None, faces[:1], DT, True)[7].shot_cut


def test_extractor_picks_the_largest_face_and_crops_exactly(actor):
    obs, _ = TW.frame(actor, t=0.0, noise_px=0.0)
    big = FakeFace(0, obs["landmarks5"] + 100, obs["landmarks106"] + 100, box=(60, 60, 360, 400))
    small = FakeFace(1, obs["landmarks5"], obs["landmarks106"], box=(0, 0, 50, 60))
    calls = []
    service = type("S", (), dict(analyzed_faces=lambda self, path, raw, threshold: calls.append((path, raw, threshold)) or [small, big]))()
    frame = SourceFrame(4, 0.5, np.zeros((500, 500, 3), np.uint8), "raw-handle")
    whole = AnalyzedFaceExtractor(service, "clip.mp4")(frame)
    assert np.allclose(whole.landmarks5, big.kps5) and whole.image_bgr.shape == (500, 500, 3) and calls[0] == ("clip.mp4", "raw-handle", 0.5)
    assert whole.frame_index == 4 and whole.timestamp == 0.5
    cropped = AnalyzedFaceExtractor(service, "clip.mp4", crop_margin=0.2)(frame)
    left, top = max(0, int(60 - 0.2 * 300)), max(0, int(60 - 0.2 * 340))
    assert cropped.image_bgr.shape[:2] == (min(500, int(400 + 0.2 * 340)) - top, min(500, int(360 + 0.2 * 300)) - left)
    assert np.allclose(cropped.landmarks5, big.kps5 - np.array((left, top))), "landmarks move with the crop"
    nobody = type("S", (), dict(analyzed_faces=lambda self, *a: []))()
    assert AnalyzedFaceExtractor(nobody, "clip.mp4")(frame) is None
    pick = AnalyzedFaceExtractor(service, "clip.mp4", pick=lambda faces: faces[0])(frame)
    assert np.allclose(pick.landmarks5, small.kps5)


def test_frame_service_source_adapts_the_media_stack():
    result = type("R", (), dict(frame_index=12, timestamp_sec=0.4, image=np.zeros((4, 4, 3), np.uint8)))()
    service = type("S", (), dict(info=lambda self, path: type("I", (), dict(total_frames=90, fps=30.0))(),
                                 read=lambda self, path, frame_index: result))()
    source = VideoFrameServiceSource(service, "clip.mp4")
    frame = source.read(12)
    assert source.total_frames == 90 and source.fps == 30.0 and frame.index == 12 and frame.timestamp == 0.4 and frame.raw is result


# =========================================================================== telemetry payload and throttling
def _tracked_frames(actor, n=30, **extra):
    tracker = VideoHeadTracker3D()
    return [f for f, _ in T.run(tracker, actor, T.moving_script(n), **extra)]


def test_payload_is_a_valid_event_with_nulls_for_what_was_not_measured(actor):
    frames = _tracked_frames(actor)
    payload = telemetry_payload(frames[-1], job_id="job-7", track_id=2, state="tracking", shot_cuts=0, adapted=True)
    event = TrackingTelemetryEvent.model_validate(payload)
    assert payload["type"] == EVENT_TYPE == "tracking.telemetry" and event.job_id == "job-7" and event.track_id == 2
    assert event.yaw_deg == pytest.approx(frames[-1].camera.yaw_deg, abs=0.006) and event.adapted
    assert event.gaze_convergence_deg is None and event.gaze_source is None and event.lighting_key is None, "never a placeholder"
    assert 0.0 <= event.expression_energy <= 1.0 and event.identity_state in {"collecting", "provisional", "locked"}
    json.dumps(payload)                                                              # plain JSON: no numpy leaked out
    lost = VideoHeadTracker3D().process(FrameEvidence(0.0, None))
    empty = TrackingTelemetryEvent.model_validate(telemetry_payload(lost, job_id=None, track_id=0, state="tracking", shot_cuts=0, adapted=False))
    assert empty.status == "lost" and empty.yaw_deg is None and empty.confidence == 0.0
    bad_time = VideoHeadTracker3D().process(FrameEvidence(float("nan"), None))
    assert TrackingTelemetryEvent.model_validate(telemetry_payload(bad_time, job_id=None, track_id=0, state="tracking", shot_cuts=0, adapted=False)).timestamp_s is None


def test_publisher_throttles_steady_frames_but_never_the_ones_that_matter(actor):
    now, sent = [0.0], []
    publisher = TrackingTelemetryPublisher(sent.append, max_hz=10, clock=lambda: now[0])
    tracker = VideoHeadTracker3D(on_frame=publisher)
    for i in range(90):                                                              # 3 s of video at 30 fps
        now[0] = i * DT
        obs, _ = TW.frame(actor, t=i * DT, noise_px=0.2, seed=i, **T.frontal_script(1)[0])
        tracker.process(T.evidence(obs, scene_cut=(i == 45)))
    steady = [p for p in sent if not p["shot_cut"]]
    assert publisher.seen == 90 and 25 <= len(sent) <= 40, f"~10 Hz for 3 s plus the cut and the first frame, got {len(sent)}"
    assert sent[0]["frame_index"] == 0, "the first frame is always sent"
    cut = [p for p in sent if p["shot_cut"]]
    assert len(cut) == 1 and cut[0]["frame_index"] == 45 and cut[0]["shot_cuts"] == 1 and cut[0]["cut_reason"] == "explicit"
    assert sent[-1]["shot_cuts"] == 1 and sent[-1]["shot_index"] == 1, "counters are cumulative, so later frames carry the cut"
    assert steady and all(b["frame_index"] > a["frame_index"] for a, b in zip(sent, sent[1:]))
    publisher.finish()
    assert sent[-1]["state"] == "ended" and sent[-1]["frame_index"] == sent[-2]["frame_index"] and not sent[-1]["shot_cut"]


def test_publisher_sends_status_and_lock_changes_immediately(actor):
    now, sent = [0.0], []
    publisher = TrackingTelemetryPublisher(sent.append, max_hz=1, clock=lambda: now[0])             # 1 Hz: only urgent frames pass
    tracker = VideoHeadTracker3D(on_frame=publisher)
    for i in range(80):
        now[0] = i * 0.001
        obs, _ = TW.frame(actor, t=i * DT, noise_px=0.2, seed=i, **T.frontal_script(1)[0])
        tracker.process(T.evidence(obs) if i < 40 or i >= 45 else FrameEvidence(i * DT, None))
    kinds = [(p["status"], p["identity_state"]) for p in sent]
    assert ("coasting", "provisional") in kinds or any(p["status"] == "coasting" for p in sent), "a status change is urgent"
    assert any(p["status"] == "tracked" and p["frame_index"] >= 45 for p in sent), "and so is the recovery"
    states = [p["identity_state"] for p in sent]
    assert "locked" in states or "provisional" in states
    assert publisher.emitted == len(sent) and publisher.emitted < 20


def test_publisher_validation_and_idle_finish():
    with pytest.raises(ValueError):
        TrackingTelemetryPublisher(lambda p: None, max_hz=0)
    sent = []
    TrackingTelemetryPublisher(sent.append).finish()
    assert sent == [], "finishing a run that never produced a frame says nothing"


# =========================================================================== WebSocket
@pytest.fixture()
def served(tmp_path):
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    return app, TestClient(app, base_url="http://127.0.0.1:8030")


STATUS, TELEMETRY, EVENTS = ("ws://127.0.0.1:8030/ws/v1/pipeline/status", "ws://127.0.0.1:8030/ws/telemetry",
                             "ws://127.0.0.1:8030/ws/events")


def test_tracking_frames_stream_on_pipeline_status_and_nowhere_else(served, actor):
    app, client = served
    ticks = iter(range(100_000))                       # a deterministic clock: Windows' monotonic clock ticks every ~15 ms
    publisher = app.state.pipeline.tracking_publisher(job_id="job-1", max_hz=10_000, clock=lambda: float(next(ticks)))
    tracker = VideoHeadTracker3D(on_frame=publisher)
    with client.websocket_connect(STATUS) as status, client.websocket_connect(TELEMETRY) as telemetry, \
            client.websocket_connect(EVENTS) as events:
        ready = [stream.receive_json() for stream in (status, telemetry, events)]
        assert all(m["type"] == "stream.ready" for m in ready)
        assert status.receive_json()["type"] == "pipeline.status"                          # the opening snapshot; nothing tracked yet
        out = T.run(tracker, actor, T.moving_script(20) , noise=0.2)
        out += T.run(tracker, actor, [dict(yaw=3.0)], noise=0.2, start=20)
        # a cut mid-stream
        obs, _ = TW.frame(actor, t=21 * DT, noise_px=0.2)
        tracker.process(FrameEvidence(21 * DT, obs["landmarks5"], obs["landmarks106"], scene_cut=True))
        publisher.finish()
        received = [status.receive_json() for _ in range(23)]
        assert all(m["type"] == "tracking.telemetry" for m in received)
        first, cut, ended = received[0], received[-2], received[-1]
        assert first["job_id"] == "job-1" and first["state"] == "tracking" and first["frame_index"] == 0
        assert [m["frame_index"] for m in received[:-1]] == list(range(22))
        assert received[20]["yaw_deg"] == pytest.approx(out[20][0].camera.yaw_deg, abs=0.006)
        assert cut["shot_cut"] and cut["shot_cuts"] == 1 and cut["shot_index"] == 1 and cut["cut_reason"] == "explicit"
        assert ended["state"] == "ended" and ended["shot_cuts"] == 1
        for key in ("yaw_deg", "pitch_deg", "roll_deg", "gaze_convergence_deg", "expression_energy", "shot_cuts", "confidence"):
            assert key in first
        # Non-regression: the job/telemetry streams saw none of it and their sequence did not move.
        app.state.pipeline.set_engine("COMPILING_TRT", label="x FP32", model="x", precision="FP32")
        for stream, label in ((telemetry, "telemetry"), (events, "events")):
            frame = stream.receive_json()
            assert frame["type"] == "pipeline.status" and frame["engine_state"] == "COMPILING_TRT", label
            assert frame["sequence"] == ready[0]["sequence"] + 1, f"{label}: tracking frames must not consume sequence numbers"
        assert status.receive_json()["type"] == "pipeline.status"


def test_a_client_that_connects_mid_run_is_current_at_once_and_after_the_run_sees_it_ended(served, actor):
    app, client = served
    ticks = iter(range(100_000))
    publisher = app.state.pipeline.tracking_publisher(job_id="job-2", max_hz=10_000, clock=lambda: float(next(ticks)))
    tracker = VideoHeadTracker3D(on_frame=publisher)
    out = T.run(tracker, actor, T.moving_script(12), noise=0.2)
    with client.websocket_connect(STATUS) as late:
        assert late.receive_json()["type"] == "stream.ready" and late.receive_json()["type"] == "pipeline.status"
        latest = late.receive_json()
        assert latest["type"] == "tracking.telemetry" and latest["frame_index"] == 11 and latest["state"] == "tracking"
        assert latest["yaw_deg"] == pytest.approx(out[-1][0].camera.yaw_deg, abs=0.006)
    publisher.finish()
    with client.websocket_connect(STATUS) as after:
        after.receive_json(), after.receive_json()
        assert after.receive_json()["state"] == "ended"


def test_status_stream_without_any_tracking_is_unchanged(served):
    app, client = served
    with client.websocket_connect(STATUS) as status:
        assert [status.receive_json()["type"] for _ in range(2)] == ["stream.ready", "pipeline.status"]
        app.state.pipeline.set_engine("COMPILING_TRT", label="x", model="x", precision="FP32")
        assert status.receive_json()["engine_state"] == "COMPILING_TRT"


def test_an_invalid_frame_is_refused_and_not_remembered(served, actor):
    app, _ = served
    good = telemetry_payload(_tracked_frames(actor, 5)[-1], job_id=None, track_id=0, state="tracking", shot_cuts=0, adapted=False)
    app.state.pipeline.tracking_frame(good)
    for broken in ({**good, "status": "invisible"}, {**good, "confidence": 3.0}, {**good, "unexpected": 1}, {**good, "frame_index": -1}):
        with pytest.raises(ValidationError):
            app.state.pipeline.tracking_frame(broken)
    assert app.state.pipeline.tracking_snapshot().frame_index == good["frame_index"]
    assert app.state.pipeline.tracking_snapshot().status == good["status"]


def test_event_schema_documents_the_tracking_event():
    schema = json.dumps(EVENT_ADAPTER.json_schema())
    assert "tracking.telemetry" in schema and "gaze_convergence_deg" in schema and "shot_cuts" in schema


def test_a_full_tracking_queue_drops_frames_without_gaps_or_evicting_job_events():
    async def scenario():
        broker = EventBroker(capacity=8)
        payload = {"type": "tracking.telemetry", "frame_index": 0, "status": "tracked", "reason": "OK"}
        with broker.subscribe(tracking=True) as tracking, broker.subscribe() as plain:
            for i in range(200):
                broker.publish({**payload, "frame_index": i})
            await asyncio.sleep(0.05)
            kinds = [tracking.get_nowait() for _ in range(tracking.qsize())]
            assert len(kinds) == 8 and all(isinstance(k, TrackingTelemetryEvent) for k in kinds), "bounded, newest dropped"
            assert [k.frame_index for k in kinds] == list(range(8)) and not any(isinstance(k, StreamGapEvent) for k in kinds)
            assert plain.qsize() == 0, "a client that did not ask for tracking sees nothing"
            assert broker.sequence == 0, "and the global sequence did not move"
            broker.publish({"type": "queue.event", "action": "x"})
            await asyncio.sleep(0.05)
            assert plain.qsize() == 1 and tracking.qsize() == 1 and broker.sequence == 1
            assert broker.subscriber_count == 2
        assert broker.subscriber_count == 0 and not broker._tracking
    asyncio.run(scenario())
