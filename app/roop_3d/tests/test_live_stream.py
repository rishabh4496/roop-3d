"""Defect #3: binary live transport -- header, GPU/CPU JPEG, latest-wins hub, lazy cache entries, WebSocket, telemetry."""
from __future__ import annotations

import asyncio
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.async_pipeline.live import (FLAG_DROPPED, HEADER_SIZE, JpegEncoder, LiveFrameHub, SOURCE_PREVIEW,
                                         pack_header, unpack_header)
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app
from roop_3d.server.live_preview import CacheEntry, PreviewCache

try:
    import torch
    CUDA = torch.cuda.is_available()
except ImportError:  # pragma: no cover
    CUDA = False


# ------------------------------------------------------------------------------------------------ header
def test_header_is_exactly_sixteen_bytes_and_round_trips():
    packed = pack_header(123456, 12.3456, (1.25, 80.5, 33.0), flags=0x10 | 0x40, source=SOURCE_PREVIEW)
    assert len(packed) == HEADER_SIZE == 16
    header = unpack_header(packed + b"jpeg")
    assert header["frame_index"] == 123456 and header["timestamp_s"] == pytest.approx(12.346, abs=1e-3)
    assert header["stage_ms"] == pytest.approx((1.2, 80.5, 33.0), abs=0.1)
    assert header["jpeg"] and header["last"] and not header["dropped"] and header["source"] == SOURCE_PREVIEW


def test_header_saturates_instead_of_overflowing_and_rejects_foreign_versions():
    header = unpack_header(pack_header(2**33, 99999999.0, (1e9, -5.0, 0.0)))
    assert header["frame_index"] == (2**33) & 0xFFFFFFFF and header["stage_ms"][0] == 6553.5 and header["stage_ms"][1] == 0.0
    with pytest.raises(ValueError):
        unpack_header(b"\x00" * 16)               # version nibble 0
    with pytest.raises(ValueError):
        unpack_header(b"\x01\x02")


# ------------------------------------------------------------------------------------------------ jpeg
def _scene():
    image = np.zeros((90, 160, 3), np.uint8)
    image[:, :, 2] = 220          # BGR: strong red
    cv2.circle(image, (80, 45), 25, (255, 255, 255), -1)
    return image


@pytest.mark.parametrize("backend", ["opencv", pytest.param("nvjpeg", marks=pytest.mark.skipif(not CUDA, reason="CUDA required"))])
def test_jpeg_backends_decode_to_the_same_picture_with_bgr_order_preserved(backend):
    try:
        encoder = JpegEncoder(quality=92, backend=backend)
    except RuntimeError:
        pytest.skip(f"{backend} unavailable")
    source = _scene()
    decoded = cv2.imdecode(np.frombuffer(encoder.encode(source), np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == source.shape
    assert np.abs(decoded.astype(int) - source.astype(int)).mean() < 3.0
    assert decoded[5, 5, 2] > 200 and decoded[5, 5, 0] < 40, "red must stay red (BGR order)"


def test_jpeg_downscales_to_max_edge_and_auto_picks_a_working_backend():
    encoder = JpegEncoder(max_edge=64)
    data = encoder.encode(np.full((90, 160, 3), 120, np.uint8))
    assert cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (36, 64)
    assert encoder.backend in ("nvjpeg", "turbojpeg", "opencv")
    with pytest.raises(ValueError):
        encoder.encode(np.zeros((4, 4), np.uint8))


@pytest.mark.parametrize("backend", ["nvjpeg", "turbojpeg"])
def test_optional_encoder_probe_sees_its_selected_backend(monkeypatch, backend):
    def initialize(self):
        assert self.backend == backend

    monkeypatch.setattr(JpegEncoder, f"_init_{backend}", initialize)
    encoder = JpegEncoder(backend=backend)
    assert encoder.backend == backend


# ------------------------------------------------------------------------------------------------ hub
class _SlowEncoder:
    backend = "slow"
    fallback_reason = None

    def __init__(self, delay):
        self.delay = delay

    def encode(self, frame):
        time.sleep(self.delay)
        return b"\xff\xd8" + bytes([int(frame[0, 0, 0])]) + b"\xff\xd9"


def test_hub_never_blocks_the_producer_and_keeps_the_newest_frame():
    async def scenario():
        hub = LiveFrameHub(_SlowEncoder(0.05), max_fps=0)
        subscription = hub.subscribe(asyncio.get_running_loop(), depth=2)
        slowest = 0.0
        for number in range(40):
            began = time.perf_counter()
            hub.publish(np.full((8, 8, 3), number, np.uint8), frame_index=number, timestamp_s=number / 30)
            slowest = max(slowest, time.perf_counter() - began)
        await asyncio.sleep(0.4)
        received = []
        while subscription._messages:
            received.append(unpack_header(await subscription.next()))
        hub.stop()
        return hub, slowest, received

    hub, slowest, received = asyncio.run(scenario())
    assert slowest < 0.02, f"publish blocked for {slowest * 1000:.1f} ms behind a 50 ms encoder"
    assert hub.dropped > 20 and hub.encoded < 20
    assert received and received[-1]["frame_index"] == 39, "the newest frame must always be the one that arrives"
    assert any(header["dropped"] for header in received)


def test_hub_spends_nothing_when_nobody_is_watching_and_honours_the_rate_limit():
    encoder = _SlowEncoder(0)
    hub = LiveFrameHub(encoder, max_fps=10)
    frame = np.zeros((8, 8, 3), np.uint8)
    assert hub.publish(frame, frame_index=0, timestamp_s=0) is False and hub._thread is None

    async def scenario():
        hub.subscribe(asyncio.get_running_loop())
        sent = sum(hub.publish(frame, frame_index=i, timestamp_s=0) for i in range(20))
        await asyncio.sleep(0.1)
        hub.stop()
        return sent

    assert asyncio.run(scenario()) <= 2, "10 fps cap: a burst of 20 frames yields at most the first (and maybe one more)"


def test_lazy_encoder_initialization_does_not_block_frame_publisher(monkeypatch):
    initializing, finish_initialization = threading.Event(), threading.Event()

    def delayed_encoder():
        initializing.set()
        assert finish_initialization.wait(2)
        return _SlowEncoder(0)

    monkeypatch.setattr("roop_3d.async_pipeline.live.JpegEncoder", delayed_encoder)

    async def scenario():
        hub = LiveFrameHub(None, max_fps=0)
        subscriber = hub.subscribe(asyncio.get_running_loop())
        began = time.perf_counter()
        assert hub.publish(np.zeros((8, 8, 3), np.uint8), frame_index=0, timestamp_s=0)
        elapsed = time.perf_counter() - began
        assert initializing.wait(1), "the encoder should initialize on the hub worker"
        finish_initialization.set()
        message = await asyncio.wait_for(subscriber.next(), 2)
        hub.stop()
        return elapsed, message

    elapsed, message = asyncio.run(scenario())
    assert elapsed < 0.05, f"first publish blocked for {elapsed * 1000:.1f} ms on encoder initialization"
    assert unpack_header(message)["frame_index"] == 0


def test_encoder_initialization_failure_is_reported_without_escaping_publish(monkeypatch):
    def broken_encoder():
        raise RuntimeError("encoder unavailable")

    monkeypatch.setattr("roop_3d.async_pipeline.live.JpegEncoder", broken_encoder)

    async def scenario():
        hub = LiveFrameHub(None, max_fps=0)
        hub.subscribe(asyncio.get_running_loop())
        accepted = hub.publish(np.zeros((8, 8, 3), np.uint8), frame_index=0, timestamp_s=0)
        deadline = time.monotonic() + 2
        while hub.encoder_error is None and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        error = hub.encoder_error
        rejected = hub.publish(np.zeros((8, 8, 3), np.uint8), frame_index=1, timestamp_s=0)
        hub.stop()
        return accepted, error, rejected

    accepted, error, rejected = asyncio.run(scenario())
    assert accepted
    assert error == "RuntimeError: encoder unavailable"
    assert not rejected


def test_targeted_frame_reaches_only_its_subscriber():
    async def scenario():
        hub = LiveFrameHub(_SlowEncoder(0), max_fps=0)
        loop = asyncio.get_running_loop()
        one, two = hub.subscribe(loop), hub.subscribe(loop)
        hub.publish(np.full((8, 8, 3), 7, np.uint8), frame_index=5, timestamp_s=0, target=one, force=True,
                    source=SOURCE_PREVIEW)
        message = await asyncio.wait_for(one.next(), 2)
        await asyncio.sleep(0.1)
        hub.stop()
        return unpack_header(message), len(two._messages)

    header, other = asyncio.run(scenario())
    assert header["frame_index"] == 5 and header["source"] == SOURCE_PREVIEW and other == 0


def test_late_subscriber_is_shown_the_current_picture_at_once():
    async def scenario():
        hub = LiveFrameHub(_SlowEncoder(0), max_fps=0)
        loop = asyncio.get_running_loop()
        first = hub.subscribe(loop)
        hub.publish(np.zeros((8, 8, 3), np.uint8), frame_index=9, timestamp_s=0)
        await asyncio.wait_for(first.next(), 2)
        late = hub.subscribe(loop)
        message = await asyncio.wait_for(late.next(), 2)
        hub.stop()
        return unpack_header(message)

    assert asyncio.run(scenario())["frame_index"] == 9


# ------------------------------------------------------------------------------------------------ cache entry
def test_raw_entry_encodes_lazily_and_only_once():
    raw = np.random.default_rng(1).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    entry = CacheEntry.from_raw(raw, {})
    assert entry._image is None, "storing a raw frame must not run an encoder"
    assert entry.nbytes == raw.nbytes
    first = entry.image
    assert entry.image is first and first[:8] == b"\x89PNG\r\n\x1a\n"
    assert np.array_equal(cv2.imdecode(np.frombuffer(first, np.uint8), cv2.IMREAD_COLOR), raw)
    assert entry.bgr() is raw


def test_cache_accounts_raw_entries_by_the_bytes_they_were_charged():
    cache = PreviewCache(max_entries=10, max_bytes=3 * 64 * 64 * 3)
    frames = [CacheEntry.from_raw(np.zeros((64, 64, 3), np.uint8), {}) for _ in range(5)]
    for number, entry in enumerate(frames):
        cache.put(("m", number, "h"), entry)
        entry.image                                  # encoding later must not change what the cache thinks it holds
    assert len(cache) == 3 and cache.size_bytes == 3 * 64 * 64 * 3


# ------------------------------------------------------------------------------------------------ server
@pytest.fixture()
def client(tmp_path):
    app = create_app(Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider"), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as test_client:
        yield test_client


def test_websocket_delivers_binary_jpeg_frames_with_the_sixteen_byte_header(client):
    hub = client.app.state.live_hub
    hub.encoder = JpegEncoder(quality=85, backend="opencv")
    with client.websocket_connect("ws://127.0.0.1:8030/ws/v1/preview/stream") as socket:
        ready = socket.receive_json()
        assert ready["type"] == "ready" and ready["header_bytes"] == 16 and ready["protocol"] == 1
        image = _scene()
        for number in range(3):
            hub.publish(image, frame_index=40 + number, timestamp_s=1.5, stage_ms=(12.0, 80.0, 40.0))
            time.sleep(0.05)
        message = socket.receive_bytes()
        header = unpack_header(message)
        assert header["frame_index"] >= 40 and header["stage_ms"] == pytest.approx((12.0, 80.0, 40.0), abs=0.1)
        assert message[HEADER_SIZE:HEADER_SIZE + 2] == b"\xff\xd8", "payload is a bare JPEG, not base64 or JSON"
        decoded = cv2.imdecode(np.frombuffer(message[HEADER_SIZE:], np.uint8), cv2.IMREAD_COLOR)
        assert decoded.shape == image.shape
        socket.send_text('{"op":"ping","t":7}')
        while True:
            reply = socket.receive()
            if reply.get("text"):
                assert '"pong"' in reply["text"] and '"t":7' in reply["text"]
                break


def test_websocket_rejects_malformed_and_unknown_messages_without_dropping(client):
    with client.websocket_connect("ws://127.0.0.1:8030/ws/v1/preview/stream") as socket:
        socket.receive_json()
        for bad in ("not json", '{"op":"nope"}', '{"op":"render","config":"x","frame_index":-1}'):
            socket.send_text(bad)
            assert socket.receive_json()["type"] == "error"
        socket.send_text('{"op":"render","id":4,"config":"deadbeefdeadbeef","frame_index":3}')
        reply = socket.receive_json()
        assert reply["type"] == "error" and reply["code"] == "PREVIEW_CONFIG_UNKNOWN" and reply["id"] == 4


def test_websocket_enforces_the_local_origin_policy(client):
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("ws://127.0.0.1:8030/ws/v1/preview/stream", headers={"origin": "http://evil.example"}):
            pass


def test_pipeline_metrics_endpoint_reports_hardware_queues_and_encoder(client):
    body = client.get("/api/v1/system/pipeline").json()
    assert set(body) >= {"pipelines", "hardware", "live_stream", "fence"}
    assert body["hardware"]["backend"] in ("nvml", "nvidia-smi", "unavailable")
    assert set(body["fence"]) == {"shared_holders", "exclusive", "writers_waiting"}
    assert "published" in body["live_stream"] and "mean_encode_ms" in body["live_stream"]
    # the alias route the UI uses
    assert client.get("/api/system/pipeline").status_code == 200


def test_application_shutdown_joins_live_encoder_thread(tmp_path):
    app = create_app(Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider"), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        hub = client.app.state.live_hub
        hub.encoder = _SlowEncoder(0.01)
        assert hub.publish(np.zeros((8, 8, 3), np.uint8), frame_index=0, timestamp_s=0, last=True)
        encoder_thread = hub._thread
        assert encoder_thread is not None
    assert not encoder_thread.is_alive(), "FastAPI shutdown must join its owned live encoder thread"
