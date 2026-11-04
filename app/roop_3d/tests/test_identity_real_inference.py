"""Identity lock on REAL local detector + ArcFace inference with exact ground truth.

Two real people (read-only legacy FaceSets used as photographs) are composited into a
scripted scene: they walk toward each other and cross, one leaves for longer than the
track memory and re-enters, a hand covers a face, and the shot cuts to a new layout.
Ground truth is the pasted position of each person. The product guarantee tested:

    whenever a face is cleared for swapping, its track ID belongs to that person.

Skipped (with a reason) when the local SCRFD/ArcFace models or the reference FaceSets
are not installed; it never fakes inference.
"""
from __future__ import annotations

from contextlib import closing
from fractions import Fraction
import hashlib
import json

import av
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.project_paths import legacy_root
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.detection import track_policy as tp
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.media.video_reader import VideoReader
from roop_3d.models.catalog import ModelCatalog
from roop_3d.server.app import create_app
from roop_3d.tests.model_governance_support import require_reference_models

WIDTH, HEIGHT, SIZE = 960, 540, 220


@pytest.fixture(scope="module")
def people():
    try:
        catalog = ModelCatalog()
        require_reference_models(catalog)
        reader = FaceSetReader()
        documents = {name: reader.read(legacy_root() / f"app/facesets/{name}.fsz") for name in ("akansha", "anushree")}
    except (DomainError, OSError, KeyError, ValueError) as error:
        pytest.skip(f"local models or reference FaceSets unavailable: {error}")
    images = {}
    for name, document in documents.items():
        view = min(document.views, key=lambda item: abs(item.yaw))
        images[name] = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return images["akansha"], images["anushree"], documents


def scene(seed: int = 0, tone: int = 90) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = cv2.GaussianBlur(rng.integers(0, 60, (HEIGHT, WIDTH, 3), dtype=np.uint8), (0, 0), 6)
    return cv2.add(np.full((HEIGHT, WIDTH, 3), tone, np.uint8), noise)


def paste(image: np.ndarray, face: np.ndarray, x: int, y: int) -> tuple[float, float]:
    resized = cv2.resize(face, (SIZE, SIZE))
    x1, y1, x2, y2 = max(0, x), max(0, y), min(WIDTH, x + SIZE), min(HEIGHT, y + SIZE)
    if x2 > x1 and y2 > y1:
        image[y1:y2, x1:x2] = resized[y1 - y:y2 - y, x1 - x:x2 - x]
    return x + SIZE / 2, y + SIZE / 2


def hand(image: np.ndarray, x: int, y: int) -> None:
    cv2.ellipse(image, (x + SIZE // 2, y + SIZE // 2), (int(SIZE * .55), int(SIZE * .6)), 0, 0, 360, (120, 150, 200), -1)


def build_script(a: np.ndarray, b: np.ndarray):
    """-> list of (phase, frame, {'A': center|None, 'B': center|None})."""
    frames = []
    def add(phase, draw):
        image = scene()
        centers = draw(image)
        frames.append((phase, image, centers))
    for i in range(4):
        add("apart", lambda im, i=i: {"A": paste(im, a, 80 + 2 * i, 160), "B": paste(im, b, 640 - 2 * i, 160)})
    for i in range(12):
        def cross(im, i=i):
            bc = paste(im, b, 620 - i * 30, 160)
            return {"B": bc, "A": paste(im, a, 100 + i * 30, 170)}
        add("cross", cross)
    for _ in range(18):                                              # A absent longer than max_missed (15)
        add("absent", lambda im: {"A": None, "B": paste(im, b, 120, 160)})
    for _ in range(6):
        add("back", lambda im: {"B": paste(im, b, 120, 160), "A": paste(im, a, 600, 160)})
    for _ in range(4):                                               # hand fully across A's face
        def covered(im):
            result = {"B": paste(im, b, 120, 160), "A": paste(im, a, 600, 160)}
            hand(im, 600, 160)
            result["A"] = None
            return result
        add("hand", covered)
    for _ in range(6):
        add("after_hand", lambda im: {"B": paste(im, b, 120, 160), "A": paste(im, a, 600, 160)})
    for _ in range(8):                                               # hard cut, new background, positions swapped
        image = scene(seed=9, tone=170)
        frames.append(("cut", image, {"A": paste(image, a, 150, 120), "B": paste(image, b, 560, 220)}))
    return frames


def truth_for(output, centers):
    """Match each tracked face to the nearest ground-truth person (None when nothing is nearby)."""
    labelled = {}
    for tracked in output:
        box = tracked.observation.box
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        nearest = min(((name, np.hypot(cx - c[0], cy - c[1])) for name, c in centers.items() if c is not None),
                      key=lambda item: item[1], default=(None, 1e9))
        labelled[tracked.track_id] = nearest[0] if nearest[1] < SIZE / 2 else None
    return labelled


def test_real_embeddings_separate_people_at_the_policy_thresholds(people):
    """The thresholds are heuristics; this pins what was MEASURED for the two reference people."""
    _, _, documents = people
    policy = TrackPolicy()
    first, second = (np.asarray(view.embedding, np.float32) for view in (documents["akansha"].views[0], documents["anushree"].views[0]))
    within = [float(np.dot(np.asarray(x.embedding, np.float32), np.asarray(y.embedding, np.float32)))
              for document in documents.values() for i, x in enumerate(document.views) for y in document.views[i + 1:]]
    between = [float(np.dot(np.asarray(x.embedding, np.float32), np.asarray(y.embedding, np.float32)))
               for x in documents["akansha"].views for y in documents["anushree"].views]
    assert min(within) >= policy.continuity_cosine          # same person, even across large pose changes
    assert max(between) < policy.reacquire_cosine          # different people are never reacquired as each other
    assert float(first @ second) < policy.match_cosine


def test_real_inference_never_switches_identity_and_withholds_instead(people):
    a, b, _ = people
    script = build_script(a, b)
    tracker = FaceTracker(policy=TrackPolicy())
    id_of: dict[str, int] = {}
    phases: dict[str, list] = {}
    try:
        for phase, image, centers in script:
            output = tracker.process(image)
            labelled = truth_for(output, centers)
            for tracked in output:
                person = labelled[tracked.track_id]
                if person is None:
                    assert not tracked.swap_allowed or tracked.track_id not in id_of.values()   # never swap a phantom
                    continue
                if tracked.swap_allowed:
                    id_of.setdefault(person, tracked.track_id)
                    assert tracked.track_id == id_of[person], f"{phase}: {person} swapped on track {tracked.track_id}"
                phases.setdefault(phase, []).append((person, tracked))
        # Both people were established, and each kept exactly one identity for the whole clip.
        assert set(id_of) == {"A", "B"} and id_of["A"] != id_of["B"]
        assert len(tracker.track_states()) == 2, tracker.track_states()
        # Returning from absence, from the hand dropout and across the cut costs a probation, never a wrong swap.
        states = [(phase, t.state, t.swap_allowed) for phase, items in phases.items() for person, t in items if person == "A"]
        assert any(state == tp.REACQUIRED and not allowed for phase, state, allowed in states if phase == "back")
        assert any(state == tp.REACQUIRED and not allowed for phase, state, allowed in states if phase == "after_hand")
        assert any(state == tp.REACQUIRED and not allowed for phase, state, allowed in states if phase == "cut")
        assert [allowed for phase, _, allowed in states if phase == "cut"][-1] is True        # and it resumes afterwards
        assert tracker.scene_id >= 1 and any(event.reason == tp.SCENE_CUT for event in tracker.events)
    finally:
        tracker.release()


def encode(path, script) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 25)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"crf": "14"}
        for index, (_, image, _) in enumerate(script):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)


def test_real_video_inspection_is_reproducible_and_feeds_a_valid_lock(tmp_path, people):
    a, b, documents = people
    script = build_script(a, b)
    relative = "roop_3d/.runtime/media/people.mp4"
    encode(tmp_path / relative, script)
    for name in ("akansha", "anushree"):
        atomic_json(tmp_path, f"roop_3d/.runtime/facesets/{name}.json",
                    {"faceset_id": name, "name": name, "item_count": 1, "schema_version": 1})
        archive = tmp_path / f"roop_3d/outputs/facesets/{name}.fsz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(name.encode())
    app = create_app(Settings(), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        body = {"path": relative, "max_samples": 128, "identity_lock": True, "time_budget_seconds": 600}
        first = client.post("/api/v1/media/inspect-faces", json=body)
        assert first.status_code == 200, first.text
        inspection = first.json()
        assert inspection["tracking_method"] == "identity_lock_v1" and inspection["identity_report_sha256"]
        assert inspection["decoded_frames"] == len(script) and not inspection["truncated"]
        assert len(inspection["tracks"]) == 2 and inspection["scene_cut_frames"]

        # Reproducibility: inspecting the same bytes again yields the same identities and visibility.
        second = client.post("/api/v1/media/inspect-faces", json=body).json()
        sheet_a = client.get(f"/api/v1/identity/inspections/{inspection['inspection_id']}", params={"path": relative}).json()
        sheet_b = client.get(f"/api/v1/identity/inspections/{second['inspection_id']}", params={"path": relative}).json()
        keep = ("track_id", "persistent_id", "segments", "missing_ranges", "visible_frames", "swappable_frames", "state_frames")
        assert [{key: t[key] for key in keep} for t in sheet_a["tracks"]] == [{key: t[key] for key in keep} for t in sheet_b["tracks"]]
        assert sheet_a["tracking_policy_fingerprint"] == TrackPolicy().fingerprint()

        # The render replays the same tracker: identities and visibility equal what the inspection reported.
        replay = FaceTracker(policy=TrackPolicy())
        seen: dict[int, int] = {}
        try:
            with closing(VideoReader(tmp_path / relative).frames()) as frames:
                for frame in frames:
                    for tracked in replay.process(frame.image):
                        seen[tracked.track_id] = seen.get(tracked.track_id, 0) + 1
        finally:
            replay.release()
        assert seen == {track["persistent_id"]: track["visible_frames"] for track in sheet_a["tracks"]}

        # Contact sheet: thumbnails, ranges, confidence; the absent person has a reported gap, cut tracks were reacquired.
        by_person = {}
        for track in sheet_a["tracks"]:
            assert track["thumbnails"] and 0 < track["confidence_mean"] <= 1 and track["segments"]
            report = json.loads((tmp_path / f"roop_3d/.runtime/identity/{inspection['inspection_id']}.json").read_text("utf-8"))
            centroid = np.asarray(report["tracks"][track["track_id"]]["signature"]["centroid"], np.float32)
            scores = {name: float(centroid @ np.asarray(documents[name].canonical_embedding, np.float32)) for name in documents}
            by_person[max(scores, key=scores.get)] = track
        assert set(by_person) == {"akansha", "anushree"}
        assert by_person["akansha"]["missing_ranges"], "the absent person's gap must be reported"
        assert by_person["akansha"]["state_frames"].get(tp.REACQUIRED, 0) >= 3
        assert by_person["akansha"]["withheld_frames"] > 0 and by_person["akansha"]["swappable_frames"] > 0

        # Explicit selection -> hash-bound lock that survives validation, then dies with the video.
        target = by_person["anushree"]["track_id"]
        created = client.post("/api/v1/identity/locks", json={
            "target_path": relative, "inspection_id": inspection["inspection_id"],
            "assignments": [{"track_id": target, "faceset_id": "akansha"}]})
        assert created.status_code == 201, created.text
        lock = created.json()
        assert lock["status"]["valid"] and lock["target_sha256"] == hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest()
        assert [item["track_id"] for item in lock["assignments"]] == [target]       # the other person is NOT assigned
        (tmp_path / relative).write_bytes((tmp_path / relative).read_bytes() + b"\0")
        stale = client.get(f"/api/v1/identity/locks/{lock['lock_id']}", params={"verify": True}).json()["status"]
        assert not stale["valid"] and stale["reasons"][0]["code"] == "TARGET_CHANGED"
