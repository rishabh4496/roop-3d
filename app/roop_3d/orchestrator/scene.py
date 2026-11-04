"""Scene segments and traits from media-inspection output (the signals it really produces)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from roop_3d.orchestrator.schema import SceneRules


@dataclass(frozen=True)
class SceneSegment:
    """One shot (``scene_id``) of the target. Every track in a shot shares one Pipeline plan because the
    tracks share the same frames, the same Engine residency and the same runner invocation."""
    scene_id: int | None
    tracks: tuple[str, ...]
    first_frame: int | None
    last_frame: int | None
    traits: tuple[str, ...]          # frontal | large_yaw | pose_unassessed
    scoring_categories: tuple[str, ...]  # Stage 11 SceneCategory values the plan is scored on
    evidence: tuple[str, ...]        # human-readable provenance of each trait, labelled as estimates

    def as_dict(self) -> dict[str, Any]:
        return {"scene_id": self.scene_id, "tracks": list(self.tracks), "first_frame": self.first_frame,
                "last_frame": self.last_frame, "traits": list(self.traits),
                "scoring_categories": list(self.scoring_categories), "evidence": list(self.evidence),
                "low_confidence": "withhold", "occlusion": "not_assessed"}


def whole_target_segment() -> SceneSegment:
    return SceneSegment(None, (), None, None, ("pose_unassessed",), (),
                        ("no inspection supplied; plan scored on the overall composite",))


def _trait_for(track: Any, rules: SceneRules) -> tuple[str, str]:
    pose = getattr(track, "pose_range", None)
    yaw = getattr(pose, "yaw_degrees", None) if pose is not None else None
    method = getattr(pose, "method", "unknown")
    if yaw is None or method != "five_point_geometric_estimate":
        return "pose_unassessed", f"{track.track_id}: yaw not estimated (method {method})"
    extent = max(abs(float(yaw[0])), abs(float(yaw[1])))
    label = f"{track.track_id}: |yaw| extent {extent:.1f} deg from a 5-point geometric estimate (candidate estimate)"
    return ("large_yaw" if extent >= rules.large_yaw_degrees else "frontal"), label


def segments_from_tracks(tracks: Iterable[Any], rules: SceneRules) -> tuple[SceneSegment, ...]:
    """Group inspection tracks by ``scene_id``; traits are the union across the shot's tracks."""
    shots: dict[int, list[Any]] = {}
    for track in tracks:
        shots.setdefault(int(track.scene_id), []).append(track)
    mapping = {item.trait: item.scene_categories for item in rules.trait_map}
    segments: list[SceneSegment] = []
    for scene_id in sorted(shots):
        members = sorted(shots[scene_id], key=lambda t: t.track_id)
        labelled = [_trait_for(t, rules) for t in members]
        traits = tuple(sorted({trait for trait, _ in labelled}))
        categories = tuple(sorted({c for trait in traits for c in mapping.get(trait, ())}))
        segments.append(SceneSegment(scene_id, tuple(t.track_id for t in members),
                                     min(int(t.first_frame) for t in members), max(int(t.last_frame) for t in members),
                                     traits, categories, tuple(text for _, text in labelled)))
    return tuple(segments)
