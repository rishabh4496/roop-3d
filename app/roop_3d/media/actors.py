"""Stage 23: multi-person target clustering.

A face inspection yields *tracks* (one per continuous appearance). A person who leaves and re-enters the frame owns
several tracks. This module groups tracks into *actors* ("Person #1" ... "Person #N"), ranks them by how much of the
clip they occupy and how large they are, and turns an ``{actor -> FaceSet}`` choice into the per-track
``face_mappings`` a job already understands. A job may therefore swap several actors at once, each with its own
FaceSet (and so its own 3D skull lock and atlas); nothing about the renderer changes.

Grouping evidence, strongest first:

1. ``persistent_id``: the tracker already decided these tracks are the same identity (identity-lock inspections).
2. Optional identity embeddings supplied by the caller (cosine similarity >= ``merge_threshold``), merged only when
   the two groups are never on screen at the same time, because one person cannot appear twice in a frame.
3. Otherwise every track is its own actor. We never guess that two tracks are one person from position alone.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math

from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.media.schema import FaceInspection, FaceTrack, TrackToFaceSetMap


class Actor(Contract):
    actor_id: str = Field(pattern=r"^actor_[0-9]{4}$")
    label: str
    rank: int = Field(ge=1)
    track_ids: tuple[str, ...] = Field(min_length=1)
    persistent_ids: tuple[int, ...] = ()
    grouping: str  # persistent_id | embedding | single_track
    screen_time_seconds: float = Field(ge=0)
    first_frame: int = Field(ge=0)
    last_frame: int = Field(ge=0)
    first_timestamp_seconds: float = Field(ge=0)
    last_timestamp_seconds: float = Field(ge=0)
    mean_box_area: float = Field(ge=0)
    prominence: float = Field(ge=0)
    thumbnail_base64: str
    thumbnail_mime: str = "image/jpeg"


class ActorReport(Contract):
    inspection_id: str
    actor_count: int
    actors: tuple[Actor, ...]
    warnings: tuple[str, ...] = ()


class ActorRouting(Contract):
    actor_id: str
    label: str
    faceset_id: str
    track_ids: tuple[str, ...]


class ActorMappingResult(Contract):
    face_mappings: tuple[TrackToFaceSetMap, ...]
    routes: tuple[ActorRouting, ...]
    unassigned_actor_ids: tuple[str, ...]


def _area(track: FaceTrack) -> float:
    box = track.bounding_box
    return (box.x2 - box.x1) * (box.y2 - box.y1)


def _union_seconds(spans: Sequence[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(spans):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _overlaps(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> bool:
    return any(a0 < b1 and b0 < a1 for a0, a1 in left for b0, b1 in right)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm > 0 else 0.0


def _merge_by_embedding(
    groups: dict[str, list[str]],
    grouping: dict[str, str],
    tracks: Mapping[str, FaceTrack],
    embeddings: Mapping[str, Sequence[float]],
    merge_threshold: float,
) -> None:
    """Greedily merge the most similar pair of groups that are never on screen together (in place)."""
    usable = {tid: vec for tid, vec in embeddings.items() if tid in tracks and len(vec) > 0}

    def spans(key: str) -> list[tuple[float, float]]:
        return [(tracks[t].first_timestamp_seconds, tracks[t].last_timestamp_seconds) for t in groups[key]]

    def centroid(key: str) -> list[float] | None:
        vectors = [usable[t] for t in groups[key] if t in usable]
        return [sum(col) / len(vectors) for col in zip(*vectors)] if vectors else None

    while True:
        best_score, best_pair = merge_threshold, None
        keys = list(groups)
        for i, a in enumerate(keys):
            ca = centroid(a)
            if ca is None:
                continue
            for b in keys[i + 1:]:
                cb = centroid(b)
                if cb is None or _overlaps(spans(a), spans(b)):
                    continue
                score = _cosine(ca, cb)
                if score >= best_score:
                    best_score, best_pair = score, (a, b)
        if best_pair is None:
            return
        a, b = best_pair
        groups[a].extend(groups.pop(b))
        grouping[a] = "embedding"
        grouping.pop(b, None)


def cluster_actors(
    inspection: FaceInspection,
    embeddings: Mapping[str, Sequence[float]] | None = None,
    *,
    merge_threshold: float = 0.55,
) -> ActorReport:
    """Group an inspection's tracks into actors ranked by prominence (largest first)."""
    if not 0 < merge_threshold <= 1:
        raise DomainError("ACTOR_THRESHOLD_INVALID", "merge_threshold must be in (0, 1]", 422)
    tracks = {track.track_id: track for track in inspection.tracks}
    groups: dict[str, list[str]] = {}
    grouping: dict[str, str] = {}
    for track in inspection.tracks:
        if track.persistent_id is not None:
            key, kind = f"pid_{track.persistent_id}", "persistent_id"
        else:
            key, kind = f"solo_{track.track_id}", "single_track"
        grouping[key] = kind
        groups.setdefault(key, []).append(track.track_id)

    warnings: list[str] = []
    if embeddings:
        dims = {len(vec) for tid, vec in embeddings.items() if tid in tracks and len(vec) > 0}
        if len(dims) > 1:
            warnings.append("Embeddings have mixed dimensions; embedding merge skipped.")
        else:
            _merge_by_embedding(groups, grouping, tracks, embeddings, merge_threshold)

    max_area = max((_area(track) for track in inspection.tracks), default=1.0) or 1.0
    drafts = []
    for key, track_ids in groups.items():
        members = sorted((tracks[t] for t in track_ids), key=lambda item: (item.first_frame, item.track_id))
        # Screen time counts each second once even when an actor's tracks touch or overlap in time.
        seconds = _union_seconds([(t.first_timestamp_seconds, t.last_timestamp_seconds) for t in members])
        weights = [max(t.sampled_detections, 1) for t in members]
        mean_area = sum(_area(t) * w for t, w in zip(members, weights)) / sum(weights)
        prominence = seconds * math.sqrt(mean_area / max_area)
        lead = max(members, key=lambda item: (_area(item) * item.sampled_detections, item.track_id))
        drafts.append((prominence, seconds, members, key, mean_area, lead))
    # Rank: prominence, then raw screen time, then earliest appearance, so the order is total and repeatable.
    drafts.sort(key=lambda d: (-d[0], -d[1], d[2][0].first_frame, d[2][0].track_id))

    actors = []
    for rank, (prominence, seconds, members, key, mean_area, lead) in enumerate(drafts, start=1):
        actors.append(Actor(
            actor_id=f"actor_{rank:04d}", label=f"Person #{rank}", rank=rank,
            track_ids=tuple(t.track_id for t in members),
            persistent_ids=tuple(sorted({t.persistent_id for t in members if t.persistent_id is not None})),
            grouping=grouping[key],
            screen_time_seconds=round(seconds, 3),
            first_frame=min(t.first_frame for t in members), last_frame=max(t.last_frame for t in members),
            first_timestamp_seconds=min(t.first_timestamp_seconds for t in members),
            last_timestamp_seconds=max(t.last_timestamp_seconds for t in members),
            mean_box_area=round(mean_area, 2), prominence=round(prominence, 4),
            thumbnail_base64=lead.thumbnail_base64, thumbnail_mime=lead.thumbnail_mime))
    return ActorReport(inspection_id=inspection.inspection_id, actor_count=len(actors), actors=tuple(actors),
                       warnings=tuple(warnings))


def actor_face_mappings(report: ActorReport, assignments: Mapping[str, str]) -> ActorMappingResult:
    """Expand ``{actor_id: faceset_id}`` into one ``TrackToFaceSetMap`` per member track.

    Actors without an assignment are left out (their faces stay untouched in the render).
    """
    known = {actor.actor_id: actor for actor in report.actors}
    unknown = sorted(set(assignments) - set(known))
    if unknown:
        raise DomainError("ACTOR_NOT_FOUND", f"Unknown actor id(s): {', '.join(unknown)}", 422)
    mappings: list[TrackToFaceSetMap] = []
    routes: list[ActorRouting] = []
    for actor in report.actors:  # report order keeps the mapping list deterministic (rank order)
        faceset = assignments.get(actor.actor_id)
        if not faceset:
            continue
        mappings.extend(TrackToFaceSetMap(track_id=track_id, faceset_id=faceset) for track_id in actor.track_ids)
        routes.append(ActorRouting(actor_id=actor.actor_id, label=actor.label, faceset_id=faceset, track_ids=actor.track_ids))
    return ActorMappingResult(
        face_mappings=tuple(mappings), routes=tuple(routes),
        unassigned_actor_ids=tuple(a.actor_id for a in report.actors if not assignments.get(a.actor_id)))
