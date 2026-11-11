"""Stage 23: multi-person actors, paint/erase mask ranges, GOP batch planning and the 10-state lifecycle."""
from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.engine.edit_plan import FrameEditPlan
from roop_3d.jobs.schema import JobRequest, MaskKeyframe
from roop_3d.media.actors import actor_face_mappings, cluster_actors
from roop_3d.media.schema import BoundingBox, FaceInspection, FaceTrack, PoseRange
from roop_3d.queue.batch_matrix import BatchMatrixEngine, plan_segments, segment_stamp
from roop_3d.queue.lifecycle import LIFECYCLE_ORDER, LifecycleState, lifecycle_of
from roop_3d.queue_manager import CanonicalJobState, QueueManager


def _track(index: int, first: float, last: float, *, box=(0, 0, 100, 100), persistent=None, detections=10) -> FaceTrack:
    return FaceTrack(
        track_id=f"track_{index:04d}", persistent_id=persistent, first_frame=int(first * 25), last_frame=int(last * 25),
        first_timestamp_seconds=first, last_timestamp_seconds=last,
        bounding_box=BoundingBox(x1=box[0], y1=box[1], x2=box[2], y2=box[3]),
        pose_range=PoseRange(), thumbnail_base64=f"thumb{index}", sampled_detections=detections)


def _inspection(*tracks: FaceTrack) -> FaceInspection:
    return FaceInspection(
        inspection_id="a" * 32, target_sha256="b" * 64, detector_model_id="scrfd", detector_sha256="c" * 64,
        tracks=tracks, decoded_frames=1000, sampled_frames=100, truncated=False)


# -- multi-person actors ------------------------------------------------------------------------------------------

def test_actors_rank_by_prominence_and_merge_reappearing_identity():
    # Person A leaves and returns (two tracks, same persistent id); B is on screen briefly but large; C is tiny.
    report = cluster_actors(_inspection(
        _track(1, 0, 4, persistent=7), _track(2, 10, 16, persistent=7),
        _track(3, 2, 5, box=(0, 0, 200, 200), persistent=9),
        _track(4, 0, 9, box=(0, 0, 20, 20))))
    assert [a.label for a in report.actors] == ["Person #1", "Person #2", "Person #3"]
    first = report.actors[0]
    assert first.track_ids == ("track_0001", "track_0002") and first.persistent_ids == (7,)
    assert first.screen_time_seconds == pytest.approx(10.0)  # 4 s + 6 s, each second counted once
    assert first.grouping == "persistent_id"
    assert report.actors[2].grouping == "single_track" and report.actors[2].track_ids == ("track_0004",)
    assert [a.rank for a in report.actors] == [1, 2, 3]


def test_actor_screen_time_counts_overlapping_tracks_once():
    report = cluster_actors(_inspection(_track(1, 0, 6, persistent=1), _track(2, 4, 10, persistent=1)))
    assert report.actors[0].screen_time_seconds == pytest.approx(10.0)


def test_embedding_merge_never_joins_people_on_screen_together():
    near = [1.0, 0.0, 0.0]
    inspection = _inspection(_track(1, 0, 4), _track(2, 6, 9), _track(3, 1, 7))
    embeddings = {"track_0001": near, "track_0002": near, "track_0003": near}
    report = cluster_actors(inspection, embeddings)
    # track 1 and 2 never overlap -> same person; track 3 overlaps both in time -> must stay separate.
    merged = [a for a in report.actors if len(a.track_ids) == 2]
    assert len(merged) == 1 and set(merged[0].track_ids) == {"track_0001", "track_0002"}
    assert report.actor_count == 2 and merged[0].grouping == "embedding"


def test_embedding_merge_respects_threshold_and_dimension_mismatch():
    inspection = _inspection(_track(1, 0, 2), _track(2, 5, 7))
    apart = cluster_actors(inspection, {"track_0001": [1.0, 0.0], "track_0002": [0.0, 1.0]})
    assert apart.actor_count == 2
    mixed = cluster_actors(inspection, {"track_0001": [1.0, 0.0], "track_0002": [1.0, 0.0, 0.0]})
    assert mixed.actor_count == 2 and mixed.warnings
    with pytest.raises(DomainError):
        cluster_actors(inspection, merge_threshold=0)


def test_actor_mappings_expand_to_every_track_with_distinct_facesets():
    report = cluster_actors(_inspection(_track(1, 0, 4, persistent=7), _track(2, 10, 16, persistent=7), _track(3, 2, 5, persistent=9)))
    result = actor_face_mappings(report, {"actor_0001": "alice", "actor_0002": "bob"})
    assert [(m.track_id, m.faceset_id) for m in result.face_mappings] == [
        ("track_0001", "alice"), ("track_0002", "alice"), ("track_0003", "bob")]
    assert [r.label for r in result.routes] == ["Person #1", "Person #2"] and result.unassigned_actor_ids == ()
    partial = actor_face_mappings(report, {"actor_0002": "bob"})
    assert partial.unassigned_actor_ids == ("actor_0001",) and len(partial.face_mappings) == 1
    with pytest.raises(DomainError):
        actor_face_mappings(report, {"actor_0099": "x"})
    # The expansion is directly usable in a multi-FaceSet job request.
    job = JobRequest(target_path="t.mp4", inspection_id="a" * 32, face_mappings=result.face_mappings)
    assert {m.faceset_id for m in job.face_mappings} == {"alice", "bob"}


# -- paint / erase mask ranges --------------------------------------------------------------------------------------

def _png(include=0, exclude=0, size=8) -> str:
    """RGBA brush layer as the studio brush writes it: R = keep/include, G = exclude, A = opacity."""
    image = np.zeros((size, size, 4), np.uint8)
    image[..., 2], image[..., 1], image[..., 3] = include, exclude, 255  # OpenCV is BGRA: R is index 2
    ok, data = cv2.imencode(".png", image)
    assert ok
    return base64.b64encode(data.tobytes()).decode()


def _plan(**request) -> FrameEditPlan:
    return FrameEditPlan(SimpleNamespace(root=Path(".")), SimpleNamespace(face_segments=None, mask_base64=None, mask_keyframes=None, **request), {})


def _replacement():
    return SimpleNamespace(track_id=1, inverse=np.eye(3, dtype=np.float32)[:2], image=np.zeros((8, 8, 3), np.uint8))


def test_rgba_brush_mask_is_active_even_without_a_grayscale_mask():
    plan = FrameEditPlan(SimpleNamespace(root=Path(".")), SimpleNamespace(face_segments=None, mask_base64=_png(exclude=255), mask_keyframes=None), {})
    assert plan.mask is None and plan.mask_channels is not None
    assert plan.has_masks  # the old gate checked ``plan.mask is not None`` and silently dropped these strokes
    out = plan.occluders(np.zeros((8, 8, 3), np.uint8), SimpleNamespace(replacements=[_replacement()]), 0)
    include, exclude = out[1]
    assert float(exclude.mean()) == pytest.approx(1.0) and float(include.mean()) == 0.0


def test_mask_keyframe_range_overrides_whole_clip_mask_only_inside_its_range():
    ranged = [MaskKeyframe(start_frame=10, end_frame=20, mask_base64=_png(include=255))]
    plan = FrameEditPlan(SimpleNamespace(root=Path(".")),
                         SimpleNamespace(face_segments=None, mask_base64=_png(exclude=255), mask_keyframes=ranged), {})
    result, image = SimpleNamespace(replacements=[_replacement()]), np.zeros((8, 8, 3), np.uint8)
    inside = plan.occluders(image, result, 15)[1]
    outside = plan.occluders(image, result, 21)[1]
    assert float(inside[0].mean()) == pytest.approx(1.0) and float(inside[1].mean()) == 0.0
    assert float(outside[1].mean()) == pytest.approx(1.0)
    only_ranges = FrameEditPlan(SimpleNamespace(root=Path(".")), SimpleNamespace(face_segments=None, mask_base64=None, mask_keyframes=ranged), {})
    assert only_ranges.has_masks and only_ranges.occluders(image, result, 5) is None


def test_mask_keyframes_validate_ranges_and_png():
    kw = dict(target_path="t.mp4", inspection_id="a" * 32, face_mappings=[{"track_id": "track_0001", "faceset_id": "a"}])
    ok = [MaskKeyframe(start_frame=0, end_frame=9, mask_base64=_png()), MaskKeyframe(start_frame=10, end_frame=None, mask_base64=_png())]
    assert JobRequest(mask_keyframes=ok, **kw).identity_payload()["mask_keyframes"]
    with pytest.raises(ValueError):
        JobRequest(mask_keyframes=[MaskKeyframe(start_frame=0, end_frame=10, mask_base64=_png()),
                                   MaskKeyframe(start_frame=10, end_frame=12, mask_base64=_png())], **kw)
    with pytest.raises(ValueError):
        MaskKeyframe(start_frame=5, end_frame=4, mask_base64=_png())
    assert "mask_keyframes" not in JobRequest(**kw).identity_payload()  # earlier fingerprints stay stable
    with pytest.raises(DomainError):
        FrameEditPlan(SimpleNamespace(root=Path(".")), SimpleNamespace(face_segments=None, mask_base64=None, mask_keyframes=[
            MaskKeyframe(start_frame=0, end_frame=1, mask_base64="bm90LWEtcG5nLWF0LWFsbC1ub3BlLW5vcGU=")]), {})


# -- batch matrix: GOP planning -------------------------------------------------------------------------------------

def test_queue_package_does_not_shadow_stdlib_queue_and_legacy_alias_works():
    import queue
    import roop_3d.batch_matrix as legacy
    import roop_3d.queue.batch_matrix as current
    assert hasattr(queue, "Queue") and legacy is current


def test_plan_segments_splits_each_target_once_and_never_aborts_the_batch(tmp_path: Path):
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(name.encode() * 10)
    manifests = BatchMatrixEngine.expand_batch({
        "strategy": "per_file_matrix", "sources": ["s1", "s2"],
        "targets": ["a.mp4", "b.mp4", "missing.mp4"], "timestamp": "T"})
    calls: list[tuple[str, str]] = []

    def splitter(video, out, max_segment_duration, timestamp):
        calls.append((Path(video).name, timestamp))
        if Path(video).name == "b.mp4":
            raise RuntimeError("ffmpeg exploded")
        return [SimpleNamespace(model_dump=lambda i=i: {"slice_index": i, "chunk_path": f"{timestamp}_{i}", "start_seconds": i * 5.0})
                for i in (1, 2)]

    plans = plan_segments(manifests, lambda p: tmp_path / p, tmp_path / "seg", max_segment_duration=5.0, splitter=splitter)
    assert len(plans) == len(manifests) == 6
    assert sorted(name for name, _ in calls) == ["a.mp4", "b.mp4"]  # 2 splits, not 4: the matrix repeats each target
    stamps = dict(calls)
    assert stamps["a.mp4"] != stamps["b.mp4"] and all(len(stamp) == 12 for stamp in stamps.values())
    by_target = {}
    for plan in plans:
        by_target.setdefault(plan.target_path, []).append(plan)
    assert all(p.slices and len(p.slices) == 2 and p.warning is None for p in by_target["a.mp4"])
    assert all(p.slices is None and "GOP split failed" in p.warning for p in by_target["b.mp4"])
    assert all(p.slices is None and "not found" in p.warning for p in by_target["missing.mp4"])
    assert segment_stamp(tmp_path / "a.mp4") == stamps["a.mp4"]  # deterministic per file


def test_plan_segments_treats_a_single_gop_span_as_unsplit(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    manifest = BatchMatrixEngine.expand_batch({"strategy": "one_to_many", "source": "s", "targets": ["a.mp4"]})
    plan = plan_segments(manifest, lambda p: tmp_path / p, tmp_path,
                         splitter=lambda *a, **k: [SimpleNamespace(model_dump=lambda: {"slice_index": 1})])[0]
    assert plan.slices is None and plan.warning is None


# -- 10-state lifecycle ----------------------------------------------------------------------------------------------

def test_lifecycle_has_ten_states_and_every_canonical_state_projects_onto_one():
    assert [s.value for s in LIFECYCLE_ORDER] == [
        "queued", "preparing", "compiling_trt", "tracking_3d", "rendering", "paused",
        "muxing_audio", "completed", "failed", "cancelled"]
    reached = {lifecycle_of(state.value) for state in CanonicalJobState}
    assert reached == set(LifecycleState) - {LifecycleState.compiling_trt}
    for stage in ("compiling_trt", "TensorRT engine build", "trt_build 12/40"):
        for state in ("PRESCANNING", "PREPASSING", "RENDERING"):
            assert lifecycle_of(state, stage) is LifecycleState.compiling_trt
    assert lifecycle_of("QUEUED", "compiling_trt") is LifecycleState.queued  # stage text never overrides a non-build phase
    with pytest.raises(ValueError):
        lifecycle_of("EXPLODING")


def test_queue_jobs_expose_lifecycle_state_and_survive_restart(tmp_path: Path):
    queue = QueueManager(tmp_path)
    a, _ = queue.submit(request={"target_path": "a.mp4"}, name="a")
    b, _ = queue.submit(request={"target_path": "b.mp4"}, name="b")
    queue.transition(a.job_id, CanonicalJobState.PRESCANNING)
    queue.transition(b.job_id, CanonicalJobState.PAUSED)
    assert queue.get(a.job_id).lifecycle_state == "tracking_3d"
    queue.update_progress(a.job_id, {"stage": "trt_build engine", "frame": 0})
    assert queue.get(a.job_id).lifecycle_state == "compiling_trt"
    assert queue.get(a.job_id).model_dump(mode="json")["lifecycle_state"] == "compiling_trt"
    # "Machine restart": a fresh manager over the same disk root.
    revived = QueueManager(tmp_path)
    revived.recover_on_startup()
    assert revived.get(b.job_id).lifecycle_state == "paused"
    assert {job.job_id for job in revived.list_jobs()} == {a.job_id, b.job_id}
    duplicate = revived.duplicate(b.job_id)
    revived.reorder([duplicate.job_id, a.job_id, b.job_id])
    assert [job.job_id for job in revived.list_jobs()][0] == duplicate.job_id


# -- HTTP surface ----------------------------------------------------------------------------------------------------

def test_http_actor_routes_lifecycle_field_and_batch_split_warning(tmp_path: Path):
    from fastapi.testclient import TestClient

    from roop_3d.core.settings import Settings
    from roop_3d.core.storage import atomic_json
    from roop_3d.server.app import create_app

    inspection = _inspection(_track(1, 0, 4, persistent=7), _track(2, 10, 16, persistent=7), _track(3, 2, 5, persistent=9))
    atomic_json(tmp_path, f"roop_3d/.runtime/inspections/{inspection.inspection_id}.json", inspection.model_dump(mode="json"))
    with TestClient(create_app(Settings(), repo_root=tmp_path), base_url="http://127.0.0.1:8030") as client:
        listed = client.get(f"/api/v1/actors/{inspection.inspection_id}")
        assert listed.status_code == 200 and [a["label"] for a in listed.json()["actors"]] == ["Person #1", "Person #2"]
        mapped = client.post(f"/api/v1/actors/{inspection.inspection_id}/mappings",
                             json={"assignments": {"actor_0001": "alice", "actor_0002": "bob"}})
        assert mapped.status_code == 200 and len(mapped.json()["face_mappings"]) == 3
        assert client.post(f"/api/v1/actors/{inspection.inspection_id}/mappings", json={"assignments": {"actor_0009": "x"}}).status_code == 422
        assert client.post(f"/api/v1/actors/{inspection.inspection_id}/mappings", json={"assignments": {"actor_0001": "bad id!"}}).status_code == 422
        assert client.get("/api/v1/actors/" + "0" * 32).status_code == 404
        assert client.get("/api/v1/actors/not-hex").status_code == 404

        assert client.get("/api/v1/queue-lifecycle").json()[:3] == ["queued", "preparing", "compiling_trt"]
        batch = client.post("/api/v1/batch/submit", json={
            "strategy": "one_to_many", "source": "alice", "targets": ["ghost.mp4"], "auto_split": True})
        assert batch.status_code == 201
        job = batch.json()[0]
        assert job["lifecycle_state"] == "queued"
        assert "queued without" in (job["progress"]["advisory"] or "")  # a target that cannot be split is reported, not hidden
        assert client.get("/api/v1/queue").json()[0]["lifecycle_state"] == "queued"


def test_reorder_rejects_unknown_and_duplicate_ids_and_a_duplicate_job_starts_from_scratch(tmp_path: Path):
    queue = QueueManager(tmp_path)
    slice_done = {"slice_index": 1, "chunk_path": "c1.mp4", "status": "COMPLETED", "rendered_chunk_path": "r1.mp4", "retry_count": 2,
                  "error": "old", "start_seconds": 0.0}
    a, _ = queue.submit(request={"target_path": "a.mp4"}, name="a", slices=[slice_done])
    b, _ = queue.submit(request={"target_path": "b.mp4"}, name="b")
    before = [job.job_id for job in queue.list_jobs()]
    for bad, code in (([a.job_id, "nope"], "QUEUE_JOB_NOT_FOUND"), ([a.job_id, a.job_id], "QUEUE_ORDER_DUPLICATE")):
        with pytest.raises(DomainError) as caught:
            queue.reorder(bad)
        assert caught.value.code == code
    assert [job.job_id for job in queue.list_jobs()] == before  # a rejected request changes nothing
    assert [job.job_id for job in queue.reorder([b.job_id, a.job_id])] == [b.job_id, a.job_id]

    copy = queue.duplicate(a.job_id)
    assert copy.slices[0].status == "PENDING" and copy.slices[0].rendered_chunk_path is None
    assert copy.slices[0].retry_count == 0 and copy.slices[0].error is None
    assert queue.get(a.job_id).slices[0].status == "COMPLETED"  # the original is untouched
