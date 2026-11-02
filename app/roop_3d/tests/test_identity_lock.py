"""Identity lock: report, hash-bound mapping, staleness, render guard, run report and job binding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import av
import cv2
from fractions import Fraction
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.isolation import safe_path
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.face_tracker import FaceObservation, TrackedFace
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.engine.scheduler import RuntimeScheduler
from roop_3d.engine.vram_governor import VRAMGovernor
from roop_3d.engine.volumetric.session import FrameResult, ReferencePipeline, Withheld
from roop_3d.identity import lock as lock_module
from roop_3d.identity.guard import (IDENTITY_SWITCH_SUSPECTED, NO_SIGNATURE, SIGNATURE_MISMATCH, IdentityGuard,
                                    IdentityRunReport)
from roop_3d.identity.lock import (AssignmentRequest, LockCreate, LockEdit, LockEnvironment, LockStore,
                                   settings_fingerprint)
from roop_3d.identity.report import IdentityReportCollector, public_tracks, read_report, write_report
from roop_3d.jobs.worker import JobWorker
from roop_3d.media.schema import BoundingBox, FaceInspection, FaceTrack, PoseRange
from roop_3d.server.app import create_app

ARCHIVE = b"fixture faceset archive"


def basis(index: int) -> np.ndarray:
    vector = np.zeros(512, np.float32)
    vector[index] = 1
    return vector


def observation(vector: np.ndarray, left: float) -> FaceObservation:
    box = np.array((left, 10, left + 40, 50), np.float32)
    return FaceObservation(box, np.zeros((5, 2), np.float32), np.zeros((106, 2), np.float32), vector, (0, 0, 0))


def tracked(identity: int, vector: np.ndarray, left: float = 20, *, allowed: bool = True, state: str = tp.TRACKED,
            reason: str | None = None, confidence: float = 1.0, scene: int = 0) -> TrackedFace:
    return TrackedFace(identity, observation(vector, left), 1, scene, state, confidence, allowed, reason)


def textured(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (80, 160, 3), dtype=np.uint8)


class StubCatalog:
    def __init__(self):
        self.detector_record = {"model_id": "det", "sha256": "d" * 64}
        self.recognizer_record = {"model_id": "rec", "sha256": "e" * 64}

    def detector(self):
        return self.detector_record

    def recognizer(self):
        return self.recognizer_record


def sample_report(inspection_id: str, target_sha: str) -> dict:
    collector = IdentityReportCollector(TrackPolicy(), detection_threshold=0.5)
    image = textured()
    for frame in range(10):
        faces = [tracked(0, basis(0), 10)]
        if frame not in (4, 5):                               # track 1 is missing for frames 4-5
            faces.append(tracked(1, basis(1), 90, allowed=frame != 2, state=tp.TRACKED if frame != 2 else tp.UNCERTAIN,
                                 reason=None if frame != 2 else tp.AMBIGUOUS, confidence=0.9 if frame != 2 else 0.2))
        collector.observe(frame, frame / 25, image, faces)
    return collector.finalize(inspection_id=inspection_id, target_sha256=target_sha, events=[], events_dropped=0,
                              scene_cut_frames=[7], decoded_frames=10, truncated=False)


class World:
    """A tmp project root holding a target file, an identity inspection + report, and two FaceSets."""

    def __init__(self, root: Path, *, create_app_state=None):
        self.root = root
        self.media_rel = "roop_3d/.runtime/media/target.mp4"
        media = root / self.media_rel
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(b"original target video bytes")
        self.target_sha = hashlib.sha256(media.read_bytes()).hexdigest()
        self.inspection_id = "b" * 32
        self.report = sample_report(self.inspection_id, self.target_sha)
        self.report_sha = write_report(root, self.report)
        self.write_inspection()
        for name in ("alice", "bob"):
            atomic_json(root, f"roop_3d/.runtime/facesets/{name}.json",
                        {"faceset_id": name, "name": name.title(), "item_count": 1, "schema_version": 1})
            archive = root / f"roop_3d/outputs/facesets/{name}.fsz"
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_bytes(ARCHIVE + name.encode())
        self.catalog, self.settings = StubCatalog(), Settings()
        self.store = LockStore(LockEnvironment(root, lambda path: safe_path(root, path), lambda: self.settings, self.catalog))

    def write_inspection(self, **overrides) -> None:
        def track(number: int) -> FaceTrack:
            return FaceTrack(track_id=f"track_{number:04d}", persistent_id=number, first_frame=0, last_frame=9,
                             first_timestamp_seconds=0, last_timestamp_seconds=0.36,
                             bounding_box=BoundingBox(x1=1, y1=1, x2=30, y2=30), pose_range=PoseRange(),
                             thumbnail_base64="", sampled_detections=10)
        values = dict(inspection_id=self.inspection_id, target_sha256=self.target_sha, detector_model_id="det",
                      detector_sha256="d" * 64, recognizer_model_id="rec", recognizer_sha256="e" * 64,
                      tracks=(track(0), track(1)), decoded_frames=10, sampled_frames=10, truncated=False,
                      tracking_method="identity_lock_v1", tracking_policy_fingerprint=TrackPolicy().fingerprint(),
                      detection_threshold=0.5, identity_report_sha256=self.report_sha)
        values.update(overrides)
        atomic_json(self.root, f"roop_3d/.runtime/inspections/{self.inspection_id}.json",
                    FaceInspection(**values).model_dump(mode="json"))

    def create(self, mapping=(("track_0000", "alice"),), excluded=()):
        return self.store.create(LockCreate(
            target_path=self.media_rel, inspection_id=self.inspection_id,
            assignments=tuple(AssignmentRequest(track_id=t, faceset_id=f) for t, f in mapping), excluded_tracks=tuple(excluded)))

    def codes(self, lock) -> set[str]:
        return {item["code"] for item in self.store.validate(lock, fresh=True)["reasons"]}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


# --------------------------------------------------------------------------- report
def test_report_records_ranges_confidence_missing_frames_and_thumbnails():
    report = sample_report("b" * 32, "c" * 64)
    zero, one = report["tracks"]["track_0000"], report["tracks"]["track_0001"]
    assert zero["segments"] == [[0, 9, 0.0, 0.36]] and zero["missing_ranges"] == [] and zero["visible_frames"] == 10
    assert one["segments"][0][:2] == [0, 3] and one["segments"][1][:2] == [6, 9]
    assert one["missing_ranges"] == [[4, 5]]
    assert one["visible_frames"] == 8 and one["swappable_frames"] == 7 and one["withheld_frames"] == 1
    assert one["withheld_reasons"] == {tp.AMBIGUOUS: 1}
    assert one["confidence_min"] == 0.2 and 0.7 < one["confidence_mean"] < 0.9
    assert report["scene_cut_frames"] == [7] and report["tracking_policy_fingerprint"] == TrackPolicy().fingerprint()
    assert 1 <= len(zero["thumbnails"]) <= 3 and zero["thumbnails"][0]["jpeg_base64"]


def test_signature_comes_only_from_frames_cleared_for_swapping():
    collector = IdentityReportCollector(TrackPolicy(), detection_threshold=0.5)
    image = textured()
    collector.observe(0, 0.0, image, [tracked(0, basis(3), allowed=False, state=tp.UNCERTAIN, reason=tp.APPEARANCE_LOW)])
    collector.observe(1, 0.04, image, [tracked(0, basis(4))])
    report = collector.finalize(inspection_id="b" * 32, target_sha256="c" * 64, events=[], events_dropped=0,
                                scene_cut_frames=[], decoded_frames=2, truncated=False)
    signature = report["tracks"]["track_0000"]["signature"]
    assert signature["evidence_frames"] == 1 and len(signature["anchors"]) == 1
    assert signature["anchors"][0][4] == 1.0 and signature["anchors"][0][3] == 0.0


def test_report_is_sha_bound_and_public_view_has_no_biometric_vectors(world):
    def keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield key
                yield from keys(item)
        elif isinstance(value, list):
            for item in value:
                yield from keys(item)
    assert not {"signature", "anchors", "centroid"} & set(keys(public_tracks(world.report)))
    assert '0.0, 0.0, 0.0, 0.0' not in json.dumps(public_tracks(world.report))        # no raw embedding vectors
    assert read_report(world.root, world.inspection_id, expected_sha256=world.report_sha)["tracks"]
    path = world.root / f"roop_3d/.runtime/identity/{world.inspection_id}.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DomainError) as error:
        read_report(world.root, world.inspection_id, expected_sha256=world.report_sha)
    assert error.value.code == "IDENTITY_REPORT_CHANGED"


# --------------------------------------------------------------------------- lock
def test_lock_binds_every_input_and_its_digest_is_reproducible(world):
    first = world.create()
    assert first.target_sha256 == world.target_sha and first.inspection_id == world.inspection_id
    assert first.identity_report_sha256 == world.report_sha
    assert first.tracking_policy_fingerprint == TrackPolicy().fingerprint() and first.detection_threshold == 0.5
    assert (first.detector_sha256, first.recognizer_sha256) == ("d" * 64, "e" * 64)
    assert first.settings_fingerprint == settings_fingerprint(world.settings)
    assignment = first.assignments[0]
    assert assignment.faceset_sha256 == hashlib.sha256(ARCHIVE + b"alice").hexdigest()
    second = world.create()
    assert second.lock_id != first.lock_id and second.digest == first.digest          # same inputs -> same digest
    assert world.store.validate(first) == {"valid": True, "reasons": []}
    assert world.store.get(first.lock_id) == first
    assert [item.lock_id for item in world.store.list(world.media_rel)] == sorted(
        [first.lock_id, second.lock_id], key=lambda lock_id: (world.store.get(lock_id).created_at, lock_id))
    assert world.create(mapping=(("track_0000", "bob"),)).digest != first.digest


def test_one_to_one_and_multi_track_assignments(world):
    both = world.create(mapping=(("track_0000", "alice"), ("track_0001", "bob")))
    assert [(m.track_id, m.faceset_id) for m in both.mappings()] == [("track_0000", "alice"), ("track_0001", "bob")]
    many_to_one = world.create(mapping=(("track_0000", "alice"), ("track_0001", "alice")))     # one person, split tracks
    assert {item.faceset_id for item in many_to_one.assignments} == {"alice"}
    with pytest.raises(DomainError) as error:
        world.create(mapping=(("track_0000", "alice"), ("track_0000", "bob")))
    assert error.value.code == "IDENTITY_TRACK_DUPLICATE"


@pytest.mark.parametrize("mapping,excluded,code", [
    ((("track_0009", "alice"),), (), "TRACK_NOT_FOUND"),
    ((("track_0000", "alice"),), ("track_0009",), "TRACK_NOT_FOUND"),
    ((("track_0000", "alice"),), ("track_0000",), "IDENTITY_TRACK_CONFLICT"),
    ((("track_0000", "nobody"),), (), "FACESET_NOT_FOUND"),
])
def test_lock_creation_rejects_unknown_or_conflicting_selection(world, mapping, excluded, code):
    with pytest.raises(DomainError) as error:
        world.create(mapping=mapping, excluded=excluded)
    assert error.value.code == code


def test_lock_requires_an_identity_inspection_of_this_exact_video(world):
    world.write_inspection(tracking_method="iou_arcface_hungarian", tracking_policy_fingerprint=None,
                           detection_threshold=None, identity_report_sha256=None)
    with pytest.raises(DomainError) as error:
        world.create()
    assert error.value.code == "IDENTITY_INSPECTION_REQUIRED"
    world.write_inspection(target_sha256="9" * 64)
    with pytest.raises(DomainError) as error:
        world.create()
    assert error.value.code == "INSPECTION_TARGET_MISMATCH"


def test_changed_video_after_inspection_invalidates_the_lock(world):
    lock = world.create()
    (world.root / world.media_rel).write_bytes(b"re-encoded video")
    assert world.codes(lock) == {"TARGET_CHANGED"}
    with pytest.raises(DomainError) as error:
        world.store.require_valid(lock.lock_id)
    assert error.value.code == "IDENTITY_LOCK_STALE" and error.value.status == 409
    (world.root / world.media_rel).unlink()
    assert "TARGET_MISSING" in world.codes(lock)


def test_changed_source_faceset_invalidates_until_explicitly_reapproved(world):
    lock = world.create()
    (world.root / "roop_3d/outputs/facesets/alice.fsz").write_bytes(b"alice with a new face view")
    assert world.codes(lock) == {"FACESET_CHANGED"}
    with pytest.raises(DomainError):
        world.store.require_valid(lock.lock_id)
    fixed = world.store.edit(lock.lock_id, LockEdit(revision=0, action="assign", track_id="track_0000", faceset_id="alice"))
    assert world.store.validate(fixed)["valid"] and fixed.digest != lock.digest and fixed.revision == 1
    assert fixed.assignments[0].faceset_sha256 == hashlib.sha256(b"alice with a new face view").hexdigest()
    (world.root / "roop_3d/outputs/facesets/alice.fsz").unlink()
    assert world.codes(fixed) == {"FACESET_MISSING"}


def test_changed_inspection_report_models_settings_and_policy_each_invalidate(world, monkeypatch):
    lock = world.create()
    path = world.root / f"roop_3d/.runtime/identity/{world.inspection_id}.json"
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    assert "IDENTITY_REPORT_STALE" in world.codes(lock)
    path.write_bytes(original)
    assert world.store.validate(lock, fresh=True)["valid"]

    world.catalog.recognizer_record = {"model_id": "rec", "sha256": "f" * 64}
    assert world.codes(lock) == {"MODEL_CHANGED"}
    world.catalog.recognizer_record = {"model_id": "rec", "sha256": "e" * 64}

    world.settings = Settings(temporal_beta=0.5)
    assert world.codes(lock) == {"SETTINGS_CHANGED"}
    world.settings = Settings(port=8031)                       # unrelated setting must not invalidate
    assert world.store.validate(lock, fresh=True)["valid"]
    world.settings = Settings()

    monkeypatch.setattr(lock_module, "TrackPolicy", lambda: TrackPolicy(match_cosine=0.4))
    assert world.codes(lock) == {"POLICY_CHANGED"}
    monkeypatch.undo()

    world.write_inspection(truncated=True)                     # inspection file edited in place
    assert "INSPECTION_CHANGED" in world.codes(lock)


def test_tampered_lock_file_is_detected(world):
    lock = world.create()
    path = world.root / f"roop_3d/.runtime/identity_locks/{lock.lock_id}.json"
    data = json.loads(path.read_text("utf-8"))
    data["assignments"][0]["faceset_id"] = "bob"
    path.write_text(json.dumps(data), "utf-8")
    assert "LOCK_TAMPERED" in world.codes(world.store.get(lock.lock_id))


def test_manual_correction_actions_are_revisioned_and_leave_no_stale_digest(world):
    lock = world.create(mapping=(("track_0000", "alice"), ("track_0001", "bob")))
    swapped = world.store.edit(lock.lock_id, LockEdit(revision=0, action="swap", track_id="track_0000", other_track_id="track_0001"))
    assert {item.track_id: item.faceset_id for item in swapped.assignments} == {"track_0000": "bob", "track_0001": "alice"}
    assert world.store.validate(swapped)["valid"] and swapped.revision == 1
    with pytest.raises(DomainError) as error:                   # stale editor
        world.store.edit(lock.lock_id, LockEdit(revision=0, action="unassign", track_id="track_0000"))
    assert error.value.code == "IDENTITY_LOCK_CONFLICT"
    excluded = world.store.edit(lock.lock_id, LockEdit(revision=1, action="exclude", track_id="track_0001"))
    assert excluded.excluded_tracks == ("track_0001",) and [item.track_id for item in excluded.assignments] == ["track_0000"]
    included = world.store.edit(lock.lock_id, LockEdit(revision=2, action="include", track_id="track_0001"))
    assert included.excluded_tracks == ()
    reassigned = world.store.edit(lock.lock_id, LockEdit(revision=3, action="assign", track_id="track_0001", faceset_id="alice"))
    assert {item.track_id: item.faceset_id for item in reassigned.assignments} == {"track_0000": "bob", "track_0001": "alice"}
    removed = world.store.edit(lock.lock_id, LockEdit(revision=4, action="unassign", track_id="track_0000"))
    assert [item.track_id for item in removed.assignments] == ["track_0001"]
    with pytest.raises(DomainError) as error:
        world.store.edit(lock.lock_id, LockEdit(revision=5, action="unassign", track_id="track_0000"))
    assert error.value.code == "IDENTITY_TRACK_NOT_ASSIGNED"
    with pytest.raises(DomainError) as error:
        world.store.edit(lock.lock_id, LockEdit(revision=5, action="assign", track_id="track_0000"))
    assert error.value.code == "FACESET_REQUIRED"


def test_a_stale_video_blocks_editing_instead_of_being_repaired(world):
    lock = world.create()
    (world.root / world.media_rel).write_bytes(b"other video")
    with pytest.raises(DomainError) as error:
        world.store.edit(lock.lock_id, LockEdit(revision=0, action="exclude", track_id="track_0000"))
    assert error.value.code == "IDENTITY_LOCK_STALE"


def test_delete_and_unknown_lock(world):
    lock = world.create()
    world.store.delete(lock.lock_id)
    for action in (lambda: world.store.get(lock.lock_id), lambda: world.store.delete(lock.lock_id),
                   lambda: world.store.get("not-a-lock")):
        with pytest.raises(DomainError) as error:
            action()
        assert error.value.code == "IDENTITY_LOCK_NOT_FOUND"


def segment(track_id: str, faceset_id: str | None, role: str = "swap", start: int = 0, end: int | None = None):
    return SimpleNamespace(track_id=track_id, faceset_id=faceset_id, role=role, start_frame=start, end_frame=end)


def test_segments_may_narrow_a_lock_but_never_widen_it(world):
    lock = world.create(mapping=(("track_0000", "alice"),))
    base = dict(identity_lock_id=lock.lock_id, inspection_id=world.inspection_id, target_path=world.media_rel,
                face_mappings=lock.mappings())
    narrowing = (segment("track_0000", "alice", start=10, end=20), segment("track_0001", None, role="reference"),
                 segment("track_0001", None, role="exclude"))
    assert world.store.verify_job(SimpleNamespace(**base, face_segments=narrowing)).lock_id == lock.lock_id


def test_job_must_match_its_lock_exactly(world):
    lock = world.create(mapping=(("track_0000", "alice"),))
    request = SimpleNamespace(identity_lock_id=lock.lock_id, inspection_id=world.inspection_id, target_path=world.media_rel,
                              face_mappings=lock.mappings(), face_segments=None)
    assert world.store.verify_job(request).lock_id == lock.lock_id
    for change, code in (
            (dict(face_mappings=(SimpleNamespace(track_id="track_0000", faceset_id="bob"),)), "IDENTITY_LOCK_MISMATCH"),
            (dict(face_mappings=lock.mappings() + (SimpleNamespace(track_id="track_0001", faceset_id="bob"),)), "IDENTITY_LOCK_MISMATCH"),
            (dict(inspection_id="c" * 32), "IDENTITY_LOCK_MISMATCH"),
            (dict(target_path="roop_3d/.runtime/media/other.mp4"), "IDENTITY_LOCK_MISMATCH"),
            (dict(face_segments=(segment("track_0000", "bob"),)), "IDENTITY_LOCK_SEGMENT_UNAUTHORIZED"),
            (dict(face_segments=(segment("track_0001", "alice"),)), "IDENTITY_LOCK_SEGMENT_UNAUTHORIZED")):
        with pytest.raises(DomainError) as error:
            world.store.verify_job(SimpleNamespace(**{**vars(request), **change}))
        assert error.value.code == code


# --------------------------------------------------------------------------- guard + run report
def guard_for(world, mapping):
    labels = {"track_0000": 0, "track_0001": 1}
    return IdentityGuard.from_report(world.report, {label: labels[label] for label in mapping}, mapping)


def test_guard_allows_the_locked_person_and_withholds_everyone_else(world):
    guard = guard_for(world, {"track_0000": "alice"})
    assert guard.check(tracked(0, basis(0))).allowed
    verdict = guard.check(tracked(0, basis(7)))                                  # a stranger carrying the locked ID
    assert (verdict.allowed, verdict.reason) == (False, SIGNATURE_MISMATCH)
    switched = guard.check(tracked(0, (basis(0) * 0.5 + basis(1) * 0.9) / np.linalg.norm(basis(0) * 0.5 + basis(1) * 0.9)))
    assert (switched.allowed, switched.reason) == (False, IDENTITY_SWITCH_SUSPECTED)    # looks more like track_0001
    assert guard.check(tracked(1, basis(1))).reason == "NOT_LOCKED"                      # unselected face is never allowed


def test_guard_does_not_treat_one_person_split_over_two_tracks_as_a_rival(world):
    guard = guard_for(world, {"track_0000": "alice", "track_0001": "alice"})
    blended = (basis(0) + basis(1)) / np.sqrt(2)
    assert guard.check(tracked(0, blended)).allowed and guard.check(tracked(1, blended)).allowed


def test_guard_without_signature_never_allows(world):
    report = json.loads(json.dumps(world.report))
    report["tracks"]["track_0000"]["signature"] = {"evidence_frames": 0, "centroid": [], "anchors": []}
    guard = IdentityGuard.from_report(report, {"track_0000": 0}, {"track_0000": "alice"})
    assert guard.check(tracked(0, basis(0))).reason == NO_SIGNATURE


def test_run_report_lists_swapped_withheld_missing_and_untargeted_frames(world):
    lock = world.create(mapping=(("track_0000", "alice"),))
    report = IdentityRunReport(lock, {"track_0000": 0})
    stranger = tracked(5, basis(5))
    for frame in range(12):
        if frame in (4, 5):
            result = SimpleNamespace(observations=(stranger,), replacements=(), withheld=())            # target absent
        else:
            face = tracked(0, basis(0), allowed=frame not in (8, 9))
            swapped = (SimpleNamespace(track_id=0),) if frame not in (8, 9) else ()
            held = (Withheld(0, tp.AMBIGUOUS, tp.UNCERTAIN, 0.2),) if frame in (8, 9) else ()
            result = SimpleNamespace(observations=(face, stranger), replacements=swapped, withheld=held)
        report.observe(frame, result)
    manifest = report.manifest()
    track = manifest["tracks"][0]
    assert track["frames_seen"] == 10 and track["frames_swapped"] == 8 and track["frames_withheld"] == 2
    assert track["withheld"][tp.AMBIGUOUS] == {"frames": 2, "ranges": [[8, 9]]}
    assert track["not_detected_ranges"] == [[4, 5]] and track["first_seen_frame"] == 0 and track["last_seen_frame"] == 11
    assert track["faceset_sha256"] == lock.assignments[0].faceset_sha256
    assert manifest["untargeted_face_observations"] == 12 and manifest["all_locked_frames_verified"] is False
    assert manifest["lock_digest"] == lock.digest and report.never_verified() == [] and report.never_seen() == []


# --------------------------------------------------------------------------- session gating
class StubTracker:
    scene_cut = False
    active_ids = frozenset({0, 1, 2})

    def __init__(self, faces):
        self.faces = faces

    def process(self, image, **kwargs):
        return tuple(self.faces)

    def release(self):
        pass


def gated_session(faces, *, selected=frozenset({0, 1}), guard=None):
    session = ReferencePipeline()
    session.prepared, session.tracker, session.selected_ids = True, StubTracker(faces), selected
    session.portrait.retain_tracks = lambda ids: None
    session.swapper._hysteresis = {}
    calls = []

    def swap(image, landmarks5, *, track_id, **kwargs):
        calls.append(track_id)
        return SimpleNamespace(image=np.zeros((128, 128, 3), np.uint8), alignment=SimpleNamespace(inverse=np.eye(2, 3)),
                               reference_index=0)
    session.swapper.infer_frame = swap
    session.identity_guard = guard
    return session, calls


def test_session_never_swaps_unselected_uncertain_or_unverified_faces():
    faces = [tracked(0, basis(0)), tracked(1, basis(1), allowed=False, state=tp.UNCERTAIN, reason=tp.AMBIGUOUS, confidence=0.3),
             tracked(2, basis(2))]                                           # 2 is detected but not selected
    session, calls = gated_session(faces)
    result = session.infer_frame(np.zeros((80, 160, 3), np.uint8))
    assert calls == [0] and [item.track_id for item in result.replacements] == [0]
    assert result.withheld == (Withheld(1, tp.AMBIGUOUS, tp.UNCERTAIN, 0.3),)
    assert {item.track_id for item in result.observations} == {0, 1, 2}


def test_session_applies_the_identity_guard_before_any_inference(world):
    guard = guard_for(world, {"track_0000": "alice"})
    session, calls = gated_session([tracked(0, basis(7))], selected=frozenset({0}), guard=guard)      # wrong person on locked ID
    result = session.infer_frame(np.zeros((80, 160, 3), np.uint8))
    assert calls == [] and result.replacements == () and result.withheld[0].reason == SIGNATURE_MISMATCH
    good, calls = gated_session([tracked(0, basis(0))], selected=frozenset({0}), guard=guard)
    assert good.infer_frame(np.zeros((80, 160, 3), np.uint8)).replacements and calls == [0]


def test_release_session_drops_the_guard(world):
    session, _ = gated_session([], guard=guard_for(world, {"track_0000": "alice"}))
    session.release_session()
    assert session.identity_guard is None


# --------------------------------------------------------------------------- jobs + API
def make_video(path: Path, count: int = 6) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 25)
        stream.codec_context.max_b_frames = 0
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((64, 96, 3), index * 20, np.uint8), format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return path


class JobSession:
    """Plays a scripted identity timeline: frame 2 is ambiguous (withheld), the rest are swapped."""

    def __init__(self, config):
        self.config, self.compositor, self.index, self.guard, self.tracking = config, SimpleNamespace(restorer=None), 0, None, None

    def prepare_session(self, source, *, track_ids, sources_by_track, retarget, **tracking):
        self.tracking = tracking

    def set_identity_guard(self, guard):
        self.guard = guard

    def infer_frame(self, image):
        frame, self.index = self.index, self.index + 1
        held = frame == 2
        face = tracked(0, basis(0), allowed=not held, state=tp.UNCERTAIN if held else tp.TRACKED,
                       reason=tp.AMBIGUOUS if held else None)
        other = tracked(1, basis(1))                                          # an unselected bystander
        return SimpleNamespace(observations=(face, other), scene_cut=False,
                               replacements=() if held else (SimpleNamespace(track_id=0),),
                               withheld=(Withheld(0, tp.AMBIGUOUS, tp.UNCERTAIN, 0.2),) if held else ())

    def composite_frame(self, image, result, **kwargs):
        return image

    def capability(self):
        return {"identity_swapper": {"inference_state": "VERIFIED_LOCAL"}, "portrait": {"state": "UNSUPPORTED"},
                "lip_sync": {"state": "UNSUPPORTED"}}

    def release_session(self):
        pass


def job_world(tmp_path):
    from roop_3d.faceset.schemas import FaceSetDocument, FaceView, PoseBinSummary, QualityMetrics

    def document(faceset_id: str) -> FaceSetDocument:
        embedding = tuple([1.0] + [0.0] * 511)
        view = FaceView(index=0, image_bytes=b"test-crop", embedding=embedding,
                        landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)),
                        bbox=(0.0, 0.0, 32.0, 32.0), yaw=0.0, pitch=0.0, roll=0.0, expression={},
                        quality=QualityMetrics(), pose_bin="center_center")
        summary = PoseBinSummary(yaw_bin="center", pitch_bin="center", members=(0,), support=1,
                                 representative_index=0, embedding=embedding)
        return FaceSetDocument(faceset_id=faceset_id, name=faceset_id, source_sha256="a" * 64, archive_format="v2_standard",
                               canonical_embedding=embedding, medoid_embedding=embedding, views=(view,),
                               pose_bins={"center_center": summary})

    class Reader:
        def read(self, path, *, faceset_id):
            assert path.is_file()
            return document(faceset_id)

    settings = Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider", audio_policy="drop")
    app = create_app(settings, repo_root=tmp_path)
    world = World.__new__(World)
    world.root, world.media_rel, world.inspection_id = tmp_path, "roop_3d/.runtime/media/target.mp4", "b" * 32
    media = make_video(tmp_path / world.media_rel)
    world.target_sha = hashlib.sha256(media.read_bytes()).hexdigest()
    world.report = sample_report(world.inspection_id, world.target_sha)
    world.report_sha = write_report(tmp_path, world.report)
    world.write_inspection()
    for name in ("alice", "bob"):
        atomic_json(tmp_path, f"roop_3d/.runtime/facesets/{name}.json",
                    {"faceset_id": name, "name": name.title(), "item_count": 1, "schema_version": 1})
        archive = tmp_path / f"roop_3d/outputs/facesets/{name}.fsz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(ARCHIVE + name.encode())
    runner = MasterPipelineRunner(tmp_path, app.state.models, app.state.media, lambda: app.state.settings,
                                  faceset_reader=Reader(), pipeline_factory=lambda catalog, config: JobSession(config))
    app.state.worker = JobWorker(tmp_path, app.state.jobs, lambda: app.state.settings, app.state.models, app.state.media,
                                 pipeline_runner=runner, scheduler=RuntimeScheduler(VRAMGovernor()), poll_interval=0.01)
    return app, world


def finish(client, job_id, timeout=20.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/jobs/{job_id}").json()
        if record["state"] in {"succeeded", "failed", "cancelled"}:
            return record
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_api_contact_sheet_lock_lifecycle_and_locked_render(tmp_path):
    app, world = job_world(tmp_path)
    app.state.worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            # The saved inspection's model ids must match the app's catalog for the lock to validate.
            for catalog in {id(app.state.models): app.state.models, id(app.state.media.detector.catalog): app.state.media.detector.catalog}.values():
                catalog.detector = lambda: {"model_id": "det", "sha256": "d" * 64}
                catalog.recognizer = lambda: {"model_id": "rec", "sha256": "e" * 64}
            sheet = client.get(f"/api/v1/identity/inspections/{world.inspection_id}", params={"path": world.media_rel})
            assert sheet.status_code == 200, sheet.text
            body = sheet.json()
            assert [track["track_id"] for track in body["tracks"]] == ["track_0000", "track_0001"]
            assert '"anchors"' not in sheet.text and '"centroid"' not in sheet.text and '"signature"' not in sheet.text
            assert body["tracks"][1]["missing_ranges"] == [[4, 5]] and body["tracks"][1]["thumbnails"]
            assert client.get(f"/api/v1/identity/inspections/{world.inspection_id}", params={"path": "roop_3d/.runtime/media/x.mp4"}).status_code in (404, 422, 409)

            created = client.post("/api/v1/identity/locks", json={
                "target_path": world.media_rel, "inspection_id": world.inspection_id,
                "assignments": [{"track_id": "track_0000", "faceset_id": "alice"}]})
            assert created.status_code == 201, created.text
            lock = created.json()
            assert lock["status"] == {"valid": True, "reasons": []} and lock["assignments"][0]["faceset_sha256"]
            listed = client.get("/api/v1/identity/locks", params={"target_path": world.media_rel}).json()
            assert [item["lock_id"] for item in listed] == [lock["lock_id"]]
            assert client.get(f"/api/v1/identity/locks/{lock['lock_id']}", params={"verify": True}).json()["status"]["valid"]

            payload = {"target_path": world.media_rel, "inspection_id": world.inspection_id, "identity_lock_id": lock["lock_id"],
                       "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}]}
            wrong = {**payload, "face_mappings": [{"track_id": "track_0000", "faceset_id": "bob"}]}
            refused = client.post("/api/v1/jobs", json=wrong, headers={"Idempotency-Key": "lock-wrong"})
            assert refused.status_code == 409 and refused.json()["error"]["code"] == "IDENTITY_LOCK_MISMATCH"

            created_job = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "lock-ok"})
            assert created_job.status_code == 201, created_job.text
            record = finish(client, created_job.json()["job_id"])
            assert record["state"] == "succeeded", record.get("failure")
            manifest = json.loads((tmp_path / record["output"]["manifest_path"]).read_text("utf-8"))
            identity = manifest["stage12"]["identity_lock"]
            assert identity["lock_id"] == lock["lock_id"] and identity["lock_digest"] == lock["digest"]
            track = identity["tracks"][0]
            assert track["frames_swapped"] == 5 and track["withheld"][tp.AMBIGUOUS]["ranges"] == [[2, 2]]
            assert identity["untargeted_face_observations"] == 6 and not identity["all_locked_frames_verified"]

            # Editing the FaceSet after locking makes the lock stale; the next job is refused, not silently re-run.
            (tmp_path / "roop_3d/outputs/facesets/alice.fsz").write_bytes(b"alice changed")
            stale = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "lock-stale"})
            assert stale.status_code == 409 and stale.json()["error"]["code"] == "IDENTITY_LOCK_STALE"
            assert client.get(f"/api/v1/identity/locks/{lock['lock_id']}", params={"verify": True}).json()["status"]["reasons"][0]["code"] == "FACESET_CHANGED"
            fixed = client.post(f"/api/v1/identity/locks/{lock['lock_id']}/edit", json={
                "revision": 0, "action": "assign", "track_id": "track_0000", "faceset_id": "alice"})
            assert fixed.status_code == 200 and fixed.json()["status"]["valid"]

            # Changing the video makes it unrecoverable.
            (tmp_path / world.media_rel).write_bytes(b"a different video")
            assert client.get(f"/api/v1/identity/locks/{lock['lock_id']}", params={"verify": True}).json()["status"]["reasons"][0]["code"] == "TARGET_CHANGED"
            assert client.delete(f"/api/v1/identity/locks/{lock['lock_id']}").status_code == 204
            assert client.get(f"/api/v1/identity/locks/{lock['lock_id']}").status_code == 404
    finally:
        app.state.worker.stop()
