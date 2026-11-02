"""Render-time identity guard and run report.

The render replays the same tracker the inspection used, so its track IDs normally equal
the locked ones. The guard does not trust that: for every face the tracker wants swapped it
re-checks the face's ArcFace embedding against the signature recorded for that *locked*
track during inspection. A face that does not look like its locked target, or that looks
more like a different (unlocked or differently-assigned) person, is **withheld** — the
original pixels stay — and the frame range is reported. Nothing is ever re-routed to
another person.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

NO_SIGNATURE = "NO_SIGNATURE"
SIGNATURE_MISMATCH = "SIGNATURE_MISMATCH"
IDENTITY_SWITCH_SUSPECTED = "IDENTITY_SWITCH_SUSPECTED"
MAX_RANGES = 256


@dataclass(frozen=True)
class GuardVerdict:
    allowed: bool
    reason: str | None = None
    similarity: float = 1.0


class IdentityGuard:
    def __init__(self, signatures: Mapping[int, np.ndarray], groups: Mapping[int, str],
                 mapped: frozenset[int], *, min_cosine: float = 0.25):
        if not 0 < min_cosine < 1:
            raise ValueError("min_cosine must lie in (0, 1)")
        self.min_cosine, self.mapped, self.groups = min_cosine, frozenset(mapped), dict(groups)
        owners, rows = [], []
        for identity, matrix in signatures.items():
            for row in np.asarray(matrix, np.float32).reshape(-1, 512):
                owners.append(identity)
                rows.append(row)
        self._owners = np.array(owners, dtype=np.int64)
        self._rows = np.stack(rows) if rows else np.zeros((0, 512), np.float32)

    @classmethod
    def from_report(cls, report: dict, label_to_id: Mapping[str, int], mapped_groups: Mapping[str, str],
                    *, min_cosine: float = 0.25) -> "IdentityGuard":
        """``mapped_groups``: track label -> FaceSet id for locked tracks. Others are their own group."""
        signatures: dict[int, np.ndarray] = {}
        groups: dict[int, str] = {}
        for label, track in report["tracks"].items():
            identity = int(track["persistent_id"])
            signature = track["signature"]
            vectors = ([signature["centroid"]] if signature["centroid"] else []) + list(signature["anchors"])
            if vectors:
                signatures[identity] = np.asarray(vectors, np.float32)
            groups[identity] = mapped_groups.get(label, f"unlocked:{identity}")
        return cls(signatures, groups, frozenset(label_to_id[label] for label in mapped_groups), min_cosine=min_cosine)

    def check(self, tracked: Any) -> GuardVerdict:
        identity = tracked.track_id
        if identity not in self.mapped:
            return GuardVerdict(False, "NOT_LOCKED", 0.0)
        own_rows = self._rows[self._owners == identity]
        if not len(own_rows):
            return GuardVerdict(False, NO_SIGNATURE, 0.0)
        embedding = np.asarray(tracked.observation.embedding, np.float32)
        own = float((own_rows @ embedding).max())
        if own < self.min_cosine:
            return GuardVerdict(False, SIGNATURE_MISMATCH, own)
        similarity = self._rows @ embedding
        own_group = self.groups.get(identity)
        for other in np.unique(self._owners):
            if other == identity or self.groups.get(int(other)) == own_group:
                continue                      # the same person split over two labels is not a rival
            rival = float(similarity[self._owners == other].max())
            if rival >= self.min_cosine and rival > own:
                return GuardVerdict(False, IDENTITY_SWITCH_SUSPECTED, own)
        return GuardVerdict(True, None, own)


class _Ranges:
    def __init__(self):
        self.ranges: list[list[int]] = []
        self.count = 0

    def add(self, frame: int) -> None:
        self.count += 1
        if self.ranges and frame == self.ranges[-1][1] + 1:
            self.ranges[-1][1] = frame
        elif len(self.ranges) >= MAX_RANGES:
            self.ranges[-1][1] = frame
        else:
            self.ranges.append([frame, frame])


class IdentityRunReport:
    """Accumulates what the render did for each locked track, for the job manifest."""

    def __init__(self, lock: Any, label_to_id: Mapping[str, int]):
        self.lock = lock
        self.label_to_id = dict(label_to_id)
        self.assignment = {item.track_id: item for item in lock.assignments}
        self.swapped = {label: _Ranges() for label in self.label_to_id}
        self.seen = {label: _Ranges() for label in self.label_to_id}
        self.withheld: dict[str, dict[str, _Ranges]] = {label: {} for label in self.label_to_id}
        self._by_id = {identity: label for label, identity in self.label_to_id.items()}
        self.untargeted_faces = 0
        self.frames = 0

    def observe(self, frame_index: int, result: Any) -> None:
        self.frames += 1
        swapped = {item.track_id for item in result.replacements}
        withheld = {item.track_id: item for item in getattr(result, "withheld", ())}
        for tracked in result.observations:
            label = self._by_id.get(tracked.track_id)
            if label is None:
                self.untargeted_faces += 1
                continue
            self.seen[label].add(frame_index)
            if tracked.track_id in swapped:
                self.swapped[label].add(frame_index)
            elif tracked.track_id in withheld:
                self.withheld[label].setdefault(withheld[tracked.track_id].reason, _Ranges()).add(frame_index)

    def never_verified(self) -> list[str]:
        """Locked tracks that were seen but never once cleared for replacement."""
        return [label for label in self.label_to_id if self.seen[label].count and not self.swapped[label].count]

    def never_seen(self) -> list[str]:
        return [label for label in self.label_to_id if not self.seen[label].count]

    def manifest(self) -> dict:
        tracks = []
        for label, identity in self.label_to_id.items():
            seen = self.seen[label]
            gaps = [[a[1] + 1, b[0] - 1] for a, b in zip(seen.ranges, seen.ranges[1:])]
            reasons = {reason: {"frames": ranges.count, "ranges": ranges.ranges}
                       for reason, ranges in sorted(self.withheld[label].items())}
            item = self.assignment[label]
            tracks.append({
                "track_id": label, "persistent_id": identity,
                "faceset_id": item.faceset_id, "faceset_sha256": item.faceset_sha256,
                "frames_seen": seen.count, "frames_swapped": self.swapped[label].count,
                "frames_withheld": sum(value["frames"] for value in reasons.values()),
                "withheld": reasons, "not_detected_ranges": gaps[:MAX_RANGES],
                "first_seen_frame": seen.ranges[0][0] if seen.ranges else None,
                "last_seen_frame": seen.ranges[-1][1] if seen.ranges else None})
        clean = all(not entry["frames_withheld"] and not entry["not_detected_ranges"] for entry in tracks)
        return {"lock_id": self.lock.lock_id, "lock_digest": self.lock.digest, "lock_revision": self.lock.revision,
                "target_sha256": self.lock.target_sha256, "inspection_id": self.lock.inspection_id,
                "tracking_policy_fingerprint": self.lock.tracking_policy_fingerprint,
                "frames_processed": self.frames, "untargeted_face_observations": self.untargeted_faces,
                "all_locked_frames_verified": clean, "tracks": tracks}
