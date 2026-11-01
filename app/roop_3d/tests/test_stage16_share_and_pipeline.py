"""Single-origin hardening: share-mode token, unversioned API aliases, hardware profile, pipeline frames."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

from roop_3d.core.hardware import Hardware, tensor_core_precisions
from roop_3d.core.settings import Settings
from roop_3d.server.__main__ import SHARE_BIND_HOST, parse_arguments
from roop_3d.server.app import create_app
from roop_3d.server.security import COOKIE_NAME, ShareMode, is_loopback_peer
from roop_3d.server.ws import EVENT_ADAPTER, EventBroker, PipelineStatusEvent
from roop_3d.server.pipeline_status import PipelineTracker

LOOPBACK = "http://127.0.0.1:8030"
STAGES = {"STAGE_1_SPARSE_PRESCAN", "STAGE_2_ACTIVE_PREPASS", "STAGE_3_MAIN_SWAP"}


@pytest.fixture
def plain(tmp_path: Path):
    return create_app(Settings(), repo_root=tmp_path)


@pytest.fixture
def shared(tmp_path: Path):
    return create_app(Settings(), repo_root=tmp_path, share=ShareMode("t" * 43))


def test_loopback_rejects_non_local_origins_with_403_and_accepts_localhost(plain) -> None:
    with TestClient(plain, base_url=LOOPBACK) as client:
        for origin in ("https://evil.example", "null", "http://127.0.0.1:9999", "http://192.168.1.5:8030"):
            response = client.get("/api/v1/health", headers={"Origin": origin})
            assert response.status_code == 403 and response.json()["error"]["code"] == "ORIGIN_DENIED", origin
        assert client.get("/api/v1/health", headers={"Origin": "http://localhost:8030"}).status_code == 200
        assert client.get("/api/v1/health", headers={"Origin": LOOPBACK}).status_code == 200
        assert client.get("/api/v1/health").status_code == 200


def test_unversioned_aliases_and_meta(plain) -> None:
    with TestClient(plain, base_url=LOOPBACK) as client:
        versioned = client.get("/api/v1/system/hardware").json()
        assert client.get("/api/system/hardware").status_code == 200
        assert set(client.get("/api/system/hardware").json()) == set(versioned)
        assert client.get("/api/meta").json() == client.get("/api/v1/meta").json()
        assert client.get("/api/meta").json()["mock"] is False
        assert client.get("/api/meta").json()["share_mode"] is False
        assert client.get("/api/v1/health").status_code == 200
        assert client.get("/api/no-such-route").status_code == 404


def test_hardware_profile_fields_are_present_and_typed(plain) -> None:
    with TestClient(plain, base_url=LOOPBACK) as client:
        for path in ("/api/system/hardware", "/api/system/telemetry"):
            body = client.get(path).json()
            for field in ("gpu_name", "compute_capability", "vram_total_gb", "vram_available_gb", "cuda_version",
                          "tensorrt_version", "onnxruntime_providers", "tensor_core_precisions", "tensor_core_state",
                          "nvdec_available", "nvenc_available"):
                assert field in body, (path, field)
            assert isinstance(body["onnxruntime_providers"], list)
        telemetry = client.get("/api/system/telemetry").json()
        assert telemetry["pipeline"]["type"] == "pipeline.status"
        assert set(telemetry["pipeline"]["stages"]) == STAGES


@pytest.mark.parametrize("capability,expected", [
    (None, []), ("garbage", []), ("6.1", []), ("7.0", ["fp16"]), ("7.5", ["fp16", "int8"]),
    ("8.6", ["fp16", "int8", "bf16", "tf32"]), ("8.9", ["fp16", "int8", "bf16", "tf32", "fp8"])])
def test_tensor_core_precisions_follow_compute_capability(capability, expected) -> None:
    assert tensor_core_precisions(capability) == expected


def test_hardware_summary_is_derived_from_observed_gpu() -> None:
    from roop_3d.core.hardware import GPU, _summarize
    result = Hardware(timestamp="t", os={}, logical_cpu_count=1,
                      gpus=[GPU(index=0, name="Test GPU", total_vram_mib=6144, free_vram_mib=4096, driver_version="1",
                                compute_capability="8.6")],
                      cuda_runtime_version="12.4", advertised_providers=["CUDAExecutionProvider"],
                      ffmpeg_probe_state="VERIFIED_LOCAL", ffmpeg_nvenc_encoders=["h264_nvenc"])
    _summarize(result)
    assert (result.gpu_name, result.vram_total_gb, result.vram_available_gb) == ("Test GPU", 6.0, 4.0)
    assert result.cuda_version == "12.4" and result.onnxruntime_providers == ["CUDAExecutionProvider"]
    assert result.nvenc_available is True and result.nvdec_available is False
    assert result.tensor_core_state == "INFERRED_FROM_COMPUTE_CAPABILITY"


# -- share mode ---------------------------------------------------------------------------------

def test_share_mode_requires_a_token_from_remote_peers(shared) -> None:
    token = shared.state.share.token
    with TestClient(shared, base_url="http://192.168.1.20:8030", client=("192.168.1.99", 51000)) as client:
        denied = client.get("/api/v1/health")
        assert denied.status_code == 401 and denied.json()["error"]["code"] == "TOKEN_REQUIRED"
        assert client.get("/api/v1/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/api/v1/health", headers={"Authorization": f"Bearer {token}"}).status_code == 200
        assert client.get(f"/api/v1/health?token={token}").status_code == 200
        assert client.get("/api/meta", headers={"Authorization": f"Bearer {token}"}).json()["share_mode"] is True
        assert token not in client.get("/api/meta", headers={"Authorization": f"Bearer {token}"}).text


def test_share_mode_cookie_handshake_and_csrf_defense(shared) -> None:
    token = shared.state.share.token
    with TestClient(shared, base_url="http://192.168.1.20:8030", client=("192.168.1.99", 51000),
                    follow_redirects=False) as client:
        landing = client.get(f"/?token={token}&x=1")
        assert landing.status_code == 303 and landing.headers["location"] == "/?x=1"
        cookie = landing.headers["set-cookie"]
        assert f"{COOKIE_NAME}={token}" in cookie and "HttpOnly" in cookie and "SameSite=strict" in cookie
        client.cookies.set(COOKIE_NAME, token)
        assert client.get("/api/v1/health").status_code == 200
        forged = client.get("/api/v1/health", headers={"Origin": "https://evil.example"})
        assert forged.status_code == 403
        same = client.get("/api/v1/health", headers={"Origin": "http://192.168.1.20:8030"})
        assert same.status_code == 200


def test_share_mode_websocket_needs_the_token(shared) -> None:
    token = shared.state.share.token
    with TestClient(shared, base_url="http://192.168.1.20:8030", client=("192.168.1.99", 51000)) as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("/ws/v1/pipeline/status"):
                pass
        assert error.value.code == 1008
        with client.websocket_connect(f"/ws/v1/pipeline/status?token={token}") as socket:
            assert socket.receive_json()["type"] == "stream.ready"
            assert socket.receive_json()["type"] == "pipeline.status"


def test_share_mode_keeps_strict_loopback_rules_for_local_peers(shared) -> None:
    with TestClient(shared, base_url=LOOPBACK, client=("127.0.0.1", 51000)) as client:
        assert client.get("/api/v1/health").status_code == 200  # no token needed locally
        assert client.get("/api/v1/health", headers={"Origin": "https://evil.example"}).status_code == 403


def test_loopback_helpers_and_cli() -> None:
    assert is_loopback_peer(("127.0.0.1", 1)) and is_loopback_peer(("::1", 1)) and is_loopback_peer(("::ffff:127.0.0.1", 1))
    assert not is_loopback_peer(("192.168.1.2", 1)) and not is_loopback_peer(("testclient", 1)) and not is_loopback_peer(None)
    assert SHARE_BIND_HOST == "0.0.0.0"
    assert parse_arguments([]).server_share is False
    assert parse_arguments(["--server_share"]).server_share is True
    assert parse_arguments(["--server-share"]).server_share is True
    assert len({ShareMode.create().token for _ in range(20)}) == 20


# -- pipeline frames ------------------------------------------------------------------------------

def _tracker() -> tuple[EventBroker, PipelineTracker]:
    broker = EventBroker()
    tracker = PipelineTracker(broker.publish, lambda: broker.sequence)
    broker.add_observer(tracker.observe)
    return broker, tracker


def _state(job: str, state: str, previous: str | None = None) -> dict:
    return {"type": "job.state", "job_id": job, "previous_state": previous, "state": state, "revision": 1}


def test_idle_frame_has_no_invented_measurements() -> None:
    _, tracker = _tracker()
    frame = tracker.snapshot()
    assert frame.stage is None and set(frame.stages.values()) == {"PENDING"}
    assert (frame.active_ranges_count, frame.verified_target_frames, frame.skipped_frames_pct, frame.eta_seconds) == (None,) * 4
    assert frame.engine_state == "IDLE" and frame.fps == 0


def test_render_progress_drives_stage_three_and_eta() -> None:
    broker, tracker = _tracker()
    broker.publish(_state("a" * 8, "preparing", "queued"))
    assert tracker.snapshot().stages == {"STAGE_1_SPARSE_PRESCAN": "SKIPPED", "STAGE_2_ACTIVE_PREPASS": "SKIPPED",
                                         "STAGE_3_MAIN_SWAP": "PENDING"}
    broker.publish(_state("a" * 8, "running", "preparing"))
    broker.publish({"type": "job.progress", "job_id": "a" * 8, "revision": 2, "frame": 25, "total_frames": 100,
                    "fps": 5.0, "stage": "render", "stage_latency_ms": 10})
    frame = tracker.snapshot()
    assert frame.stage == "STAGE_3_MAIN_SWAP" and frame.stages["STAGE_3_MAIN_SWAP"] == "RUNNING"
    assert frame.stage_progress_pct == 25.0 and frame.fps == 5.0 and frame.eta_seconds == 15.0
    broker.publish(_state("a" * 8, "succeeded", "running"))
    done = tracker.snapshot()
    assert done.stages["STAGE_3_MAIN_SWAP"] == "DONE" and done.stage is None
    assert done.stage_progress_pct == 100 and done.eta_seconds == 0


def test_failed_and_cancelled_jobs_and_foreign_job_events() -> None:
    broker, tracker = _tracker()
    broker.publish(_state("j1", "preparing"))
    broker.publish(_state("j1", "running"))
    broker.publish({"type": "job.progress", "job_id": "other", "revision": 1, "frame": 1, "total_frames": 2,
                    "fps": 1.0, "stage": "render", "stage_latency_ms": 1})
    assert tracker.snapshot().stage_progress_pct == 0.0  # a different job does not move this frame
    broker.publish(_state("j1", "failed", "running"))
    assert tracker.snapshot().stages["STAGE_3_MAIN_SWAP"] == "FAILED"
    broker.publish(_state("j2", "preparing"))
    broker.publish(_state("j2", "cancelled", "preparing"))
    assert set(tracker.snapshot().stages.values()) == {"PENDING"}


def test_engine_state_transitions_publish_frames_with_since() -> None:
    broker, tracker = _tracker()
    seen: list[PipelineStatusEvent] = []
    broker.add_observer(lambda event: seen.append(event) if isinstance(event, PipelineStatusEvent) else None)
    tracker.engine_observer("COMPILING_TRT", {"name": "inswapper"}, "fp16")
    first = tracker.snapshot()
    assert (first.engine_state, first.engine_model, first.engine_precision) == ("COMPILING_TRT", "inswapper", "fp16")
    assert first.engine_label == "inswapper fp16" and first.engine_since is not None
    tracker.engine_observer("RUNNING_CANARY", {"name": "inswapper"}, "fp16")
    assert tracker.snapshot().engine_state == "RUNNING_CANARY"
    tracker.engine_observer("IDLE", {"name": "inswapper"}, "fp16")
    idle = tracker.snapshot()
    assert idle.engine_state == "IDLE" and idle.engine_since is None and idle.engine_label is None
    assert [event.engine_state for event in seen] == ["COMPILING_TRT", "RUNNING_CANARY", "IDLE"]
    assert EVENT_ADAPTER.validate_python(seen[0].model_dump(mode="json")).type == "pipeline.status"


def test_pipeline_streams_over_both_websocket_routes(plain) -> None:
    with TestClient(plain, base_url=LOOPBACK) as client:
        with client.websocket_connect("ws://127.0.0.1:8030/ws/v1/pipeline/status") as status, \
                client.websocket_connect("ws://127.0.0.1:8030/ws/telemetry") as telemetry:
            assert status.receive_json()["type"] == "stream.ready"
            assert telemetry.receive_json()["type"] == "stream.ready"
            assert status.receive_json()["type"] == "pipeline.status"

            def until(socket, wanted, limit: int = 12) -> dict:
                # A 2-second hardware heartbeat may interleave; skip frames that are not the one awaited.
                for _ in range(limit):
                    frame = socket.receive_json()
                    if wanted(frame):
                        return frame
                raise AssertionError("expected frame never arrived")

            plain.state.events.publish(_state("w" * 8, "preparing", "queued"))
            skipped = until(status, lambda f: f["job_id"] == "w" * 8 if f["type"] == "pipeline.status" else pytest.fail(f["type"]))
            assert skipped["stages"]["STAGE_1_SPARSE_PRESCAN"] == "SKIPPED"
            # /ws/telemetry keeps its job-event contract and also carries the pipeline frames.
            assert until(telemetry, lambda f: f["type"] == "job.state")["state"] == "preparing"
            assert until(telemetry, lambda f: f["type"] == "pipeline.status" and f["job_id"] == "w" * 8)
            plain.state.events.publish(_state("w" * 8, "failed", "preparing"))
            assert until(status, lambda f: f["stages"]["STAGE_3_MAIN_SWAP"] == "FAILED")["stage"] is None
