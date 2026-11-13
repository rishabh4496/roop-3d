"""The track thread's target fits are only ever used when they are exactly what the swap stage would have computed."""
import threading

import numpy as np
import pytest

from roop_3d.reconstruction.fit_ahead import FitAhead, fit_ahead_enabled
from roop_3d.temporal import CoupledTrackSmoother


class RecordingBackbone:
    """``fit`` returns the landmarks it was given, so a test can tell exactly which trajectory made a fit."""

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on
        self.lock = threading.Lock()

    def fit(self, image, landmarks5):
        with self.lock:
            self.calls += 1
            number = self.calls
        if self.fail_on is not None and number == self.fail_on:
            raise ValueError("fit exploded")
        return np.array(landmarks5, np.float64)


def face(frame, track=1, wobble=1.0):
    base = np.array([[30., 40.], [70., 40.], [50., 60.], [35., 80.], [65., 80.]])
    return base + np.array([frame * 1.7, np.sin(frame * wobble) * 3.0])


def image_for(frame):
    return np.full((8, 8, 3), frame % 255, np.uint8)


class SwapSide:
    """What ``NeuralFaceEngine.process`` does: advance its own smoother, then ask FitAhead for the matching fit."""

    def __init__(self, ahead):
        self.ahead, self.smoother, self.served = ahead, CoupledTrackSmoother(), []

    def process(self, frame, image, track=1, landmarks=None):
        landmarks = face(frame) if landmarks is None else landmarks
        step = self.smoother.advance(track, frame, landmarks, timestamp=frame / 30.0)
        fit = self.ahead.take(track, frame, image, step.landmarks, frame / 30.0)
        if fit is not None:
            assert np.array_equal(fit, step.landmarks), "a precomputed fit must come from the swap stage's own landmarks"
        self.served.append(fit is not None)
        return fit


def run(ahead, swap, frames, *, lookahead, skip=(), track=1):
    """The track side runs ``lookahead`` frames ahead of the swap side; ``skip`` frames are never processed by the swap side."""
    ahead.begin(0)
    for frame in range(frames + lookahead):
        if frame < frames:
            ahead.submit(image_for(frame), [(track, face(frame))], timestamp=frame / 30.0, scene_cut=False, active_ids={track})
        done = frame - lookahead
        if 0 <= done < frames and done not in skip:
            swap.process(done, image_for(done), track)


def test_every_fit_is_reused_when_the_swap_stage_follows_the_same_frames():
    ahead = FitAhead(RecordingBackbone())
    swap = SwapSide(ahead)
    run(ahead, swap, 40, lookahead=3)
    assert all(swap.served)
    stats = ahead.stats()
    assert stats["hits"] == 40 and stats["misses"] == 0 and stats["resyncs"] == 0
    assert ahead.backbone.calls == 40                       # each frame fitted once, on the track side only


def test_a_gap_in_the_swap_stage_never_yields_a_wrong_fit_and_the_lookahead_recovers():
    ahead = FitAhead(RecordingBackbone())
    swap = SwapSide(ahead)
    run(ahead, swap, 60, lookahead=3, skip={10, 11, 12, 30})
    served = swap.served                                     # frames 0..9, 13..29, 31..59 in order
    assert len(served) == 60 - 4
    stats = ahead.stats()
    assert stats["resyncs"] >= 2 and stats["misses"] >= 2     # the gaps were noticed
    assert sum(served) >= len(served) - 16                    # only a few frames per gap are lost, not the rest of the render
    assert all(served[-10:])                                  # in step again at the end


def test_a_changed_image_is_never_served_a_stale_fit():
    ahead = FitAhead(RecordingBackbone())
    ahead.begin(0)
    ahead.submit(image_for(0), [(1, face(0))], timestamp=0.0, scene_cut=False, active_ids={1})
    swap = SwapSide(ahead)
    step = swap.smoother.advance(1, 0, face(0), timestamp=0.0)
    assert ahead.take(1, 0, image_for(7), step.landmarks, 0.0) is None


def test_inactive_and_cleared_instances_do_nothing():
    backbone = RecordingBackbone()
    ahead = FitAhead(backbone)
    ahead.submit(image_for(0), [(1, face(0))], timestamp=0.0, scene_cut=False, active_ids={1})
    assert backbone.calls == 0 and ahead.take(1, 0, image_for(0), face(0), 0.0) is None
    ahead.begin(0)
    ahead.clear()
    ahead.submit(image_for(0), [(1, face(0))], timestamp=0.0, scene_cut=False, active_ids={1})
    assert backbone.calls == 0


def test_a_failing_fit_is_skipped_and_the_swap_stage_fits_inline():
    ahead = FitAhead(RecordingBackbone(fail_on=3))
    swap = SwapSide(ahead)
    run(ahead, swap, 6, lookahead=2)
    assert swap.served == [True, True, False, True, True, True]
    assert ahead.stats()["errors"] == 1 and ahead.stats()["resyncs"] == 0     # the trajectory was still right


def test_a_second_track_and_a_scene_cut_follow_the_same_rules():
    ahead = FitAhead(RecordingBackbone())
    ahead.begin(0)
    real = {1: CoupledTrackSmoother(), 2: CoupledTrackSmoother()}
    for frame in range(12):
        cut = frame == 6
        ahead.submit(image_for(frame), [(1, face(frame)), (2, face(frame, wobble=.3) + 5)], timestamp=frame / 30.0,
                     scene_cut=cut, active_ids={1, 2})
    for frame in range(12):
        for track, offset in ((1, 0), (2, 5)):
            landmarks = face(frame) + offset if track == 1 else face(frame, wobble=.3) + 5
            step = real[track].advance(track, frame, landmarks, timestamp=frame / 30.0, scene_cut=frame == 6)
            fit = ahead.take(track, frame, image_for(frame), step.landmarks, frame / 30.0)
            assert fit is not None and np.array_equal(fit, step.landmarks)


def test_threads_track_ahead_while_the_swap_thread_consumes_in_order():
    ahead = FitAhead(RecordingBackbone())
    ahead.begin(0)
    swap = SwapSide(ahead)
    produced = threading.Semaphore(0)
    slots = threading.Semaphore(4)                            # a bounded queue between the two threads

    def track_thread():
        for frame in range(80):
            slots.acquire()
            ahead.submit(image_for(frame), [(1, face(frame))], timestamp=frame / 30.0, scene_cut=False, active_ids={1})
            produced.release()

    thread = threading.Thread(target=track_thread)
    thread.start()
    for frame in range(80):
        produced.acquire()
        swap.process(frame, image_for(frame))
        slots.release()
    thread.join(10)
    assert all(swap.served) and ahead.stats()["hits"] == 80


def test_the_switch(monkeypatch):
    monkeypatch.delenv("ROOP3D_FIT_AHEAD", raising=False)
    assert fit_ahead_enabled()
    for value in ("0", "off", "False"):
        monkeypatch.setenv("ROOP3D_FIT_AHEAD", value)
        assert not fit_ahead_enabled()
