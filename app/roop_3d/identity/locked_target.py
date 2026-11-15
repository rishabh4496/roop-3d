"""Bind the frame-by-frame reconstruction loop to the locked canonical target IDs.

The inspection gave every person one canonical ID (``identity.consolidation``) and stored a pose-aware signature for it in
the identity report. The render replays a tracker whose own IDs are *not* the canonical ones (they are whatever the replay
minted, and they can split where the global pass merged). :class:`LockedTargetBinder` re-labels the tracker's faces with the
canonical IDs on every frame, in this order:

1. **Pre-pass table** (optional): a full-timeline pre-pass records which canonical ID each box got on each frame; the replay
   looks the frame and box up again (IoU >= 0.5), so a face is bound by the *global* decision, not by a local guess.
2. **Sticky binding**: a tracker ID already bound to a canonical ID keeps it while the face still resembles that person or
   still sits where that person last was.
3. **Embedding fallback**: faces nothing above explained are matched to the canonical IDs not yet used on this frame by
   Hungarian assignment on ArcFace cosine against the pose-aware signature, with the same turn/continuity relief and ambiguity
   margin the tracker uses.

A face that no canonical ID claims keeps a tracker ID shifted by :data:`UNBOUND_BASE`, which no lock can select: it is never
reconstructed under someone else's identity. Nothing here decides *whether* to render (that stays with the identity guard).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Iterable, Mapping

import numpy as np

from roop_3d.engine.detection.face_tracker import TrackedFace, optimal_assignment
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.identity.consolidation import TrackletRecorder, consolidate
from roop_3d.media.detector import iou

UNBOUND_BASE = 1_000_000
PREPASS_IOU = 0.5


class LockedTargetBinder:
    def __init__(self, signatures: Mapping[int, np.ndarray], policy: TrackPolicy | None = None):
        self.policy = policy or TrackPolicy()
        self.signatures: dict[int, np.ndarray] = {}
        for canonical, rows in signatures.items():
            matrix = np.asarray(rows, np.float32).reshape(-1, 512)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            if len(matrix) and (norms > 0).all():
                self.signatures[int(canonical)] = matrix / norms
        self.raw_to_canonical: dict[int, int] = {}
        self.last_seen: dict[int, tuple[int, np.ndarray]] = {}
        self.timeline: dict[int, list[tuple[np.ndarray, int]]] = {}
        self.recording = False
        self._recorder: TrackletRecorder | None = None
        self.prepass_summary: dict[str, int] = {}
        self.stats: Counter[str] = Counter()

    @classmethod
    def from_report(cls, report: dict, policy: TrackPolicy | None = None) -> "LockedTargetBinder":
        """Signatures (pose-aware centroid + anchors) of every canonical track in an identity report."""
        signatures: dict[int, np.ndarray] = {}
        for track in report["tracks"].values():
            signature = track["signature"]
            vectors = ([signature["centroid"]] if signature["centroid"] else []) + list(signature["anchors"])
            if vectors:
                signatures[int(track["persistent_id"])] = np.asarray(vectors, np.float32)
        return cls(signatures, policy)

    # ---------------------------------------------------------------------------------------------- lifecycle
    def start_prepass(self) -> None:
        """Begin a full-timeline scan: every tracker face of the following frames is recorded for the global pass."""
        self.recording = True
        self.timeline.clear()
        self._recorder = TrackletRecorder()
        self.reset_state()

    def finish_prepass(self) -> None:
        """End the scan: consolidate the recorded tracklets globally (``identity.consolidation``), match each consolidated
        identity to a locked signature, and keep the per-frame result so the replay binds by the global decision."""
        self.recording = False
        self.reset_state()
        recorder, self._recorder = self._recorder, None
        self.timeline = self._global_timeline(recorder) if recorder is not None else {}

    def _global_timeline(self, recorder: TrackletRecorder) -> dict[int, list[tuple[np.ndarray, int]]]:
        tracklets = recorder.tracklets()
        result = consolidate(tracklets, self.policy)
        locked = sorted(self.signatures)
        scores = np.full((len(result.identities), len(locked)), -1.0)
        for row, identity in enumerate(result.identities):
            vectors = np.stack([vector for vector in [identity.centroid, *identity.anchors] if vector is not None])
            for column, canonical in enumerate(locked):
                value = float((vectors @ self.signatures[canonical].T).max())
                if value >= self.policy.merge_cosine:
                    scores[row, column] = value
        mapping = {result.identities[row].canonical_id: locked[column] for row, column in optimal_assignment(scores)}
        self.prepass_summary = {"tracker_ids": len(result.canonical_of), "identities": len(result.identities),
                                "matched_to_lock": len(mapping)}
        timeline: dict[int, list[tuple[np.ndarray, int]]] = {}
        for tracklet in tracklets:
            canonical = result.canonical(tracklet.track_id)
            target = mapping.get(canonical) if canonical is not None else None
            if target is None:
                continue
            for frame, box in zip(tracklet.frames.tolist(), tracklet.boxes):
                timeline.setdefault(int(frame), []).append((box.astype(np.float32), target))
        return timeline

    def reset_state(self) -> None:
        self.raw_to_canonical.clear()
        self.last_seen.clear()

    # ---------------------------------------------------------------------------------------------- evidence
    def similarity(self, canonical: int, embedding: np.ndarray) -> float:
        rows = self.signatures.get(canonical)
        return -1.0 if rows is None else float((rows @ np.asarray(embedding, np.float32)).max())

    def _near_last_seen(self, canonical: int, box: np.ndarray, frame: int) -> bool:
        seen = self.last_seen.get(canonical)
        if seen is None or frame - seen[0] > self.policy.continuity_gap_frames or frame < seen[0]:
            return False
        return iou(np.asarray(box, np.float32), seen[1]) >= 0.3

    def _steep(self, face: TrackedFace) -> bool:
        return abs(float(face.observation.pose[0])) >= self.policy.turn_yaw

    def _sticky(self, canonical: int, face: TrackedFace, frame: int) -> bool:
        similarity = self.similarity(canonical, face.observation.embedding)
        if similarity >= self.policy.weak_cosine:
            return True
        return self._near_last_seen(canonical, face.observation.box, frame) and (similarity >= 0.0 or self._steep(face))

    # ---------------------------------------------------------------------------------------------- binding
    def bind(self, faces: tuple[TrackedFace, ...], frame: int) -> tuple[TrackedFace, ...]:
        if not self.signatures or not faces:
            return faces
        frame = int(frame)
        if self.recording:
            self._recorder.observe(frame, faces)
        chosen: dict[int, int] = {}
        table = None if self.recording else self.timeline.get(frame)
        if table:
            for index, face in enumerate(faces):
                best = max(((iou(face.observation.box, box), canonical) for box, canonical in table if canonical not in chosen.values()),
                           default=None, key=lambda item: item[0])
                if best is not None and best[0] >= PREPASS_IOU:
                    chosen[index] = best[1]
                    self.stats["prepass"] += 1
        used = set(chosen.values())
        for index, face in enumerate(faces):
            canonical = self.raw_to_canonical.get(face.track_id)
            if index in chosen or canonical is None or canonical in used:
                continue
            if self._sticky(canonical, face, frame):
                chosen[index] = canonical
                used.add(canonical)
                self.stats["sticky"] += 1
        remaining = [index for index in range(len(faces)) if index not in chosen]
        free = [canonical for canonical in sorted(self.signatures) if canonical not in used]
        if remaining and free:
            for index, canonical in self._fallback(faces, remaining, free, frame).items():
                chosen[index] = canonical
                self.stats["embedding"] += 1
        output = []
        for index, face in enumerate(faces):
            canonical = chosen.get(index)
            if canonical is None:
                self.stats["unbound"] += 1
                output.append(replace(face, track_id=UNBOUND_BASE + face.track_id))
                continue
            self.raw_to_canonical[face.track_id] = canonical
            self.last_seen[canonical] = (frame, np.asarray(face.observation.box, np.float32).copy())
            output.append(replace(face, track_id=canonical))
        return tuple(output)

    def _fallback(self, faces, remaining: list[int], free: list[int], frame: int) -> dict[int, int]:
        policy = self.policy
        similarity = np.array([[self.similarity(canonical, faces[index].observation.embedding) for canonical in free]
                               for index in remaining])
        scores = np.full(similarity.shape, -1.0)
        for row, index in enumerate(remaining):
            face = faces[index]
            for column, canonical in enumerate(free):
                value = float(similarity[row, column])
                continuous = self._near_last_seen(canonical, face.observation.box, frame)
                floor = policy.match_cosine
                if continuous:
                    floor = policy.continuity_cosine - (policy.turn_relief if self._steep(face) else 0.0)
                if value >= floor:
                    scores[row, column] = value + (0.1 if continuous else 0.0)
        result: dict[int, int] = {}
        for row, column in optimal_assignment(scores):
            rivals = [float(similarity[row, other]) for other in range(len(free)) if other != column]
            own = float(similarity[row, column])
            continuous = self._near_last_seen(free[column], faces[remaining[row]].observation.box, frame)
            if continuous or not rivals or own - max(rivals) >= policy.ambiguity_margin:
                result[remaining[row]] = free[column]
        return result

    def canonical_ids(self, raw_ids: Iterable[int]) -> frozenset[int]:
        """Canonical IDs of the tracker IDs that are still alive (for per-track resource retention)."""
        return frozenset(self.raw_to_canonical[raw] for raw in raw_ids if raw in self.raw_to_canonical)
