"""Target-face fit computed one stage ahead of the swap stage.

``NeuralFaceEngine.process`` starts every dense-3D face with ``backbone.fit(image, landmarks5)``: four neural graphs plus
two rasterisations, about 40 % of the swap thread's time.  In the asynchronous pipeline the *track* thread is nearly
idle, so it can fit frame N+1 while the swap thread is still compositing frame N.

The one thing in the way is that ``fit`` is called with the *smoothed* landmarks, and the smoother
(``CoupledTrackSmoother``) is swap-thread state whose history depends on which frames the swap stage actually processed.
So the track thread keeps a private **shadow** smoother, fed the landmarks of every face that is likely to be swapped,
and fits with the landmarks the shadow predicts.  The swap stage then *validates* the prediction: the precomputed fit is
used only when the landmarks it would have used itself are bit-equal to the predicted ones (and it is the same frame
image).  Otherwise it fits inline exactly as before, so the output can never differ from the sequential path; a wrong
prediction only costs the wasted work.  A miss also re-synchronises the shadow with the real smoother (by replaying the
recorded landmarks of the frames the fitter is already ahead), so a gap in the swap stage's history costs a few frames
of lookahead, not the rest of the render.

``ROOP3D_FIT_AHEAD=0`` switches it off.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import threading

import numpy as np

from roop_3d.temporal import CoupledTrackSmoother, _CoupledState

log = logging.getLogger(__name__)

MAX_ENTRIES = 48          # precomputed fits held at once (a few frames x faces; the pipeline queue bounds the lookahead)
LOG_EVERY = 300           # takes between [FIT-AHEAD] summary lines


def fit_ahead_enabled() -> bool:
    return os.environ.get("ROOP3D_FIT_AHEAD", "1").strip().lower() not in ("0", "false", "off", "no")


@dataclass
class _Entry:
    image: np.ndarray
    landmarks: np.ndarray
    fit: object


class FitAhead:
    def __init__(self, backbone):
        self.backbone = backbone
        self._lock = threading.Lock()
        self.begin(0)
        self.active = False                       # nothing is precomputed or validated until a render calls begin()

    # -- lifecycle ------------------------------------------------------------------------------------------
    def begin(self, base_frame: int) -> None:
        """Start of a render: ``base_frame`` is the swap stage's ``frames_processed`` before its first frame."""
        with self._lock:
            self.base, self.count = int(base_frame), 0
            self.shadow = CoupledTrackSmoother()
            self.entries: dict[tuple[int, int], _Entry] = {}
            self.lineage: dict[tuple[int, int], np.ndarray] = {}      # (track, frame) -> landmarks the shadow smoothed to
            self.inputs: deque = deque(maxlen=512)                    # (frame, track, raw landmarks, timestamp, scene_cut)
            self.hits = self.misses = self.resyncs = self.errors = 0
            self.active = True

    def clear(self) -> None:
        self.begin(self.base)
        self.active = False

    # -- track thread ---------------------------------------------------------------------------------------
    def submit(self, image: np.ndarray, faces, *, timestamp: float, scene_cut: bool, active_ids) -> None:
        """Fit every ``(track_id, landmarks5)`` of the next frame.  Frames must be submitted in order, one call each."""
        if not self.active:
            return
        with self._lock:
            frame = self.base + self.count
            self.count += 1
            self.shadow.retain(active_ids)            # the swap stage retains before it advances: same order here
        for track_id, landmarks5 in faces:
            raw = np.array(landmarks5, np.float64)
            try:
                with self._lock:
                    step = self.shadow.advance(track_id, frame, raw, timestamp=timestamp, scene_cut=scene_cut)
                    smoothed = np.array(step.landmarks, np.float64)
                    self.inputs.append((frame, track_id, raw, timestamp, bool(scene_cut)))
                    self.lineage[(track_id, frame)] = smoothed
                fit = self.backbone.fit(image, smoothed)
            except Exception as error:                # the swap stage will fit inline and raise its own, real error
                self.errors += 1
                log.debug("[FIT-AHEAD] frame %s track %s skipped: %s: %s", frame, track_id, type(error).__name__, error)
                continue
            with self._lock:
                self.entries[(track_id, frame)] = _Entry(image, smoothed, fit)
                self._trim(frame)

    def _trim(self, newest: int) -> None:
        while len(self.entries) > MAX_ENTRIES:
            self.entries.pop(next(iter(self.entries)))
        floor = newest - 4 * MAX_ENTRIES
        if len(self.lineage) > 8 * MAX_ENTRIES:
            self.lineage = {key: value for key, value in self.lineage.items() if key[1] >= floor}

    # -- swap thread ----------------------------------------------------------------------------------------
    def take(self, track_id: int, frame: int, image: np.ndarray, landmarks5, timestamp):
        """The precomputed fit for exactly these inputs, else ``None`` (the caller fits inline)."""
        if not self.active:
            return None
        landmarks = np.asarray(landmarks5, np.float64)
        with self._lock:
            entry = self.entries.pop((track_id, frame), None)
            for key in [key for key in self.entries if key[1] < frame]:       # frames the swap stage has moved past
                del self.entries[key]
            if (entry is not None and entry.landmarks.shape == landmarks.shape and np.array_equal(entry.landmarks, landmarks)
                    and (entry.image is image or (entry.image.shape == image.shape and np.array_equal(entry.image, image)))):
                self.hits += 1
                self._report()
                return entry.fit
            self.misses += 1
            predicted = self.lineage.get((track_id, frame))
            if predicted is None or predicted.shape != landmarks.shape or not np.array_equal(predicted, landmarks):
                self._resync(track_id, frame, landmarks, timestamp)
            self._report()
            return None

    def _resync(self, track_id: int, frame: int, landmarks: np.ndarray, timestamp) -> None:
        """Adopt the swap stage's smoothed state for this track and replay the frames the fitter is already past."""
        self.resyncs += 1
        self.shadow.states[track_id] = _CoupledState(int(frame), timestamp, {CoupledTrackSmoother.LANDMARKS: landmarks.copy()}, {})
        self.lineage[(track_id, frame)] = landmarks.copy()
        for later, track, raw, stamp, cut in list(self.inputs):
            if track != track_id or later <= frame:
                continue
            step = self.shadow.advance(track, later, raw, timestamp=stamp, scene_cut=cut)
            self.lineage[(track, later)] = np.array(step.landmarks, np.float64)
            self.entries.pop((track, later), None)          # fitted from the old trajectory: not what the swap stage will ask for

    def _report(self) -> None:
        total = self.hits + self.misses
        if total and total % LOG_EVERY == 0:
            log.info("[FIT-AHEAD] %d of %d target fits were precomputed on the track thread (%d re-syncs, %d errors)",
                     self.hits, total, self.resyncs, self.errors)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "resyncs": self.resyncs, "errors": self.errors,
                    "held": len(self.entries)}
