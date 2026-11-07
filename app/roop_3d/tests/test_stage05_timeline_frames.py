"""Stage 5: the scrubbing frame service and its /media/info, /media/frame, /media/filmstrip endpoints.

Every frame of the fixture video encodes its own index as sixteen black/white blocks, so "which frame did I get?"
is answered from the pixels, not from metadata. 10 200 frames also cover the indices the older bounded
`MediaService.frame` refuses (>= 10 000).
"""
from __future__ import annotations

import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.media.frame_service import VideoFrameService
from roop_3d.server.app import create_app

BITS, BLOCK, WIDTH, HEIGHT, FPS, FRAMES = 16, 16, 256, 64, 25, 10_200


def frame_image(index: int) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    for bit in range(BITS):
        if index >> bit & 1:
            image[:, bit * BLOCK:(bit + 1) * BLOCK] = 255
    return image


def index_of(image: np.ndarray) -> int:
    gray = image.mean(axis=2)
    return sum(1 << bit for bit in range(BITS) if gray[HEIGHT // 2, bit * BLOCK + BLOCK // 2] > 127)


def write_video(path: Path, frames: int = FRAMES, gop: int = 12) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.codec_context.gop_size = gop
        stream.codec_context.options = {"qscale": "2"}
        for index in range(frames):
            for packet in stream.encode(av.VideoFrame.from_ndarray(frame_image(index), format="bgr24")):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


@pytest.fixture(scope="module")
def root(tmp_path_factory) -> Path:
    base = tmp_path_factory.mktemp("stage05")
    write_video(base / "roop_3d/.runtime/media/counter.mp4")
    write_video(base / "roop_3d/.runtime/media/second.mp4", frames=60)
    write_video(base / "roop_3d/.runtime/media/third.mp4", frames=60)
    return base


def service_for(root: Path, **kwargs) -> VideoFrameService:
    return VideoFrameService(lambda value: root / value, **kwargs)


COUNTER = "roop_3d/.runtime/media/counter.mp4"


def test_info_reports_size_rate_and_length(root):
    info = service_for(root).info(COUNTER)
    assert (info.width, info.height, info.fps, info.total_frames) == (WIDTH, HEIGHT, FPS, FRAMES)
    assert info.duration_sec == pytest.approx(FRAMES / FPS, abs=0.1)
    assert info.rotation_degrees == 0


@pytest.mark.parametrize("order", [
    [0, 1, 2, 3, 4, 5],                      # sequential: continue decoding
    [500, 200, 9999, 10000, 10199, 7, 10001],   # random, backward, across the 10 000 boundary
    [100, 101, 130, 131, 90, 91],            # forward steps, then back over a keyframe
])
def test_every_requested_frame_is_the_frame_returned(root, order):
    service = service_for(root)
    for index in order:
        result = service.read(COUNTER, frame_index=index)
        assert index_of(result.image) == index, f"asked for {index}"
        assert result.frame_index == index
        assert result.timestamp_sec == pytest.approx(index / FPS, abs=1e-3)
    service.close()


def test_one_container_serves_sequential_and_random_access(root):
    service = service_for(root)
    for index in range(40):
        service.read(COUNTER, frame_index=index)
    assert service.stats["containers_opened"] == 1
    assert service.stats["seeks"] == 1 and service.stats["forward_decodes"] == 39    # no reopen, no re-seek while playing forward
    for index in (8000, 3, 9000, 12, 10100):
        service.read(COUNTER, frame_index=index)
    assert service.stats["containers_opened"] == 1
    # A far jump decodes at most one GOP (12 frames) plus the decoder's frame-thread lookahead, not 8000 frames.
    assert service.stats["frames_decoded"] < 40 + 5 * 40
    service.close()


def test_cached_frames_are_not_decoded_again(root):
    service = service_for(root)
    first = service.read(COUNTER, frame_index=4321)
    decoded = service.stats["frames_decoded"]
    again = service.read(COUNTER, frame_index=4321)
    assert (first.cache, again.cache) == ("miss", "hit")
    assert service.stats["frames_decoded"] == decoded
    assert index_of(again.image) == 4321
    service.close()


def test_a_timestamp_selects_the_frame_that_starts_there(root):
    service = service_for(root)
    result = service.read(COUNTER, timestamp=40 / FPS)
    assert index_of(result.image) == 40
    service.close()


def test_selectors_are_validated_and_the_end_is_reported(root):
    service = service_for(root)
    for kwargs in ({}, {"frame_index": 1, "timestamp": 0.1}, {"frame_index": -1}, {"timestamp": float("nan")}, {"timestamp": 90000}):
        with pytest.raises(DomainError) as raised:
            service.read(COUNTER, **kwargs)
        assert raised.value.status == 422
    with pytest.raises(DomainError) as raised:
        service.read(COUNTER, frame_index=FRAMES)
    assert (raised.value.code, raised.value.status) == ("FRAME_OUT_OF_RANGE", 416)
    service.close()


def test_open_containers_are_bounded_and_least_recently_used_goes_first(root):
    service = service_for(root, max_open=2)
    for name in ("counter", "second", "third"):
        service.read(f"roop_3d/.runtime/media/{name}.mp4", frame_index=1)
    assert service.stats["containers_opened"] == 3 and len(service._entries) == 2
    names = {Path(key[0]).stem for key in service._entries}
    assert names == {"second", "third"}
    service.close()
    assert not service._entries


def test_filmstrip_is_one_pass_evenly_spaced_and_cached(root):
    service = service_for(root)
    strip = service.filmstrip(COUNTER, 16)
    assert strip["count"] == 16 and len(strip["frames"]) == 16 and strip["thumb_height"] == 90
    indices = [item["frame_index"] for item in strip["frames"]]
    assert indices == sorted(indices) and len(set(indices)) == 16
    spacing = FRAMES / 16
    for slot, index in enumerate(indices):
        wanted = (slot + 0.5) * spacing
        assert abs(index - wanted) <= 40, (slot, index, wanted)     # thumbnails accept a frame near the target
    first = cv2.imdecode(np.frombuffer(__import__("base64").b64decode(strip["frames"][0]["data_url"].split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)
    assert first.shape == (90, strip["thumb_width"], 3)
    assert service.filmstrip(COUNTER, 16) is strip
    with pytest.raises(DomainError):
        service.filmstrip(COUNTER, 65)
    service.close()


class FakeDetector:
    """Two faces returned right-to-left; the service must order them left to right."""

    def __init__(self):
        self.calls = 0

    def detect_scored(self, image, threshold):
        self.calls += 1
        right = (np.array([150.0, 10.0, 210.0, 60.0]), np.array([[165, 28], [195, 28], [180, 38], [168, 50], [192, 50]], np.float32), 0.93)
        left = (np.array([20.0, 8.0, 80.0, 62.0]), np.array([[35, 26], [65, 26], [50, 38], [38, 50], [62, 50]], np.float32), 0.88)
        return [right, left], {}


def test_faces_are_left_to_right_in_native_pixels_with_pose_and_confidence_and_are_cached(root):
    detector = FakeDetector()
    service = service_for(root, detector=detector)
    result = service.read(COUNTER, frame_index=77)
    faces = service.faces(COUNTER, result, 0.5)
    assert [face["face_index"] for face in faces] == [0, 1]
    assert faces[0]["bbox"] == [20.0, 8.0, 80.0, 62.0] and faces[1]["bbox"][0] == 150.0
    assert faces[0]["bbox_norm"] == [round(20 / WIDTH, 5), round(8 / HEIGHT, 5), round(80 / WIDTH, 5), round(62 / HEIGHT, 5)]
    assert len(faces[0]["kps_5"]) == 5 and faces[0]["kps_5"][2] == [50.0, 38.0]
    assert [face["confidence"] for face in faces] == [0.88, 0.93]
    assert set(faces[0]["pose"]) == {"yaw", "pitch", "roll"} and set(faces[0]["pose3d"]) == {"yaw", "pitch", "roll"}
    assert faces[0]["pose3d"]["pitch"] == pytest.approx(-faces[0]["pose"]["pitch"], abs=0.02)   # legacy + = down, standard + = up
    assert faces[0]["track_id"] is None
    service.faces(COUNTER, result, 0.5)
    assert detector.calls == 1
    service.faces(COUNTER, result, 0.7)
    assert detector.calls == 2
    service.close()


# ---------------------------------------------------------------- HTTP
@pytest.fixture
def client(root):
    app = create_app(Settings(), repo_root=root)
    app.state.media.frames.detector = FakeDetector()
    with TestClient(app, base_url="http://127.0.0.1:8030") as test_client:
        yield test_client


def test_info_endpoint(client):
    body = client.get("/api/v1/media/info", params={"path": COUNTER}).json()
    assert body["width"] == WIDTH and body["height"] == HEIGHT and body["fps"] == FPS
    assert body["total_frames"] == FRAMES and body["duration_sec"] == pytest.approx(FRAMES / FPS, abs=0.1)


def test_frame_endpoint_returns_a_jpeg_with_its_metadata_in_a_header(client):
    response = client.get("/api/v1/media/frame", params={"path": COUNTER, "frame_index": 10_150})
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    meta = json.loads(response.headers["x-frame-meta"])
    assert meta["frame_index"] == 10_150 and meta["width"] == WIDTH and meta["height"] == HEIGHT and "faces" not in meta
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert image.shape == (HEIGHT, WIDTH, 3) and index_of(image) == 10_150
    again = json.loads(client.get("/api/v1/media/frame", params={"path": COUNTER, "frame_index": 10_150}).headers["x-frame-meta"])
    assert again["cache"] == "hit"


def test_frame_endpoint_returns_faces_only_when_asked_and_scales_the_image_not_the_boxes(client):
    response = client.get("/api/v1/media/frame", params={"path": COUNTER, "frame_index": 300, "detect_faces": True, "max_size": 128})
    meta = json.loads(response.headers["x-frame-meta"])
    assert (meta["image_width"], meta["image_height"]) == (128, 32)           # the JPEG is smaller ...
    assert (meta["width"], meta["height"]) == (WIDTH, HEIGHT)                 # ... the face boxes stay in native pixels
    assert [face["bbox"][0] for face in meta["faces"]] == [20.0, 150.0]
    assert "embedding_norm" in meta["unsupported"] and "track_id" not in meta["unsupported"]   # Stage 12: track_id is provided (appearance cluster)


def test_frame_endpoint_json_format(client):
    body = client.get("/api/v1/media/frame", params={"path": COUNTER, "frame_index": 12, "detect_faces": True, "format": "json"}).json()
    assert body["image"].startswith("data:image/jpeg;base64,") and len(body["faces"]) == 2 and body["frame_index"] == 12


@pytest.mark.parametrize("params,status", [
    ({"path": COUNTER}, 422),                                              # no selector
    ({"path": COUNTER, "frame_index": 3, "timestamp": 1.0}, 422),          # two selectors
    ({"path": COUNTER, "frame_index": FRAMES}, 416),                       # past the end
    ({"path": "../outside.mp4", "frame_index": 0}, 422),                   # outside the runtime
    ({"path": COUNTER, "frame_index": 0, "format": "png"}, 422),
    ({"path": COUNTER, "frame_index": 0, "max_size": 10}, 422),
])
def test_frame_endpoint_rejects_bad_requests(client, params, status):
    assert client.get("/api/v1/media/frame", params=params).status_code == status


def test_filmstrip_endpoint(client):
    body = client.get("/api/v1/media/filmstrip", params={"path": COUNTER, "count": 16}).json()
    assert len(body["frames"]) == 16 and body["total_frames"] == FRAMES and body["frames"][0]["data_url"].startswith("data:image/jpeg")
    assert client.get("/api/v1/media/filmstrip", params={"path": COUNTER, "count": 0}).status_code == 422
    assert client.get("/api/v1/media/filmstrip", params={"path": COUNTER, "count": 65}).status_code == 422
