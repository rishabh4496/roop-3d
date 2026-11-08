from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import logging
import socket
from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import SecretStr, ValidationError
import pytest
from starlette.websockets import WebSocketDisconnect

from roop_3d.core.redaction import redact
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.jobs.schema import JobProgress, JobState
from roop_3d.server.__main__ import JsonFormatter
from roop_3d.server.app import create_app
from roop_3d.server.ws import EventBroker, EVENT_ADAPTER


@pytest.fixture
def application(tmp_path: Path):
    target = "roop_3d/.runtime/media/target.mp4"
    path = tmp_path / target
    path.parent.mkdir(parents=True)
    # Queue-contract fixture: no decoder or detector behavior is replaced/mocked.
    path.write_bytes(b"contract fixture target bytes")
    atomic_json(tmp_path, "roop_3d/.runtime/facesets/alice.json",
                {"faceset_id": "alice", "name": "Alice", "item_count": 1, "schema_version": 1})
    inspection_id = "1" * 32
    atomic_json(tmp_path, f"roop_3d/.runtime/inspections/{inspection_id}.json", {
        "inspection_id": inspection_id, "target_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "detector_model_id": "contract_fixture", "detector_sha256": "0" * 64,
        "decoded_frames": 1, "sampled_frames": 1, "truncated": False,
        "tracks": [{"track_id": "track_0000", "first_frame": 0, "last_frame": 0,
                    "first_timestamp_seconds": 0, "last_timestamp_seconds": 0,
                    "bounding_box": {"x1": 1, "y1": 1, "x2": 10, "y2": 10},
                    "pose_range": {}, "thumbnail_base64": "", "sampled_detections": 1}],
    })
    app = create_app(Settings(), repo_root=tmp_path)
    payload = {"target_path": target, "inspection_id": inspection_id,
               "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}]}
    return app, payload


def test_all_rest_contracts_are_in_openapi(application) -> None:
    app, _ = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        paths = client.get("/openapi.json").json()["paths"]
        expected = {"/health": "get", "/system/hardware": "get", "/system/capabilities": "get",
                    "/config": "put", "/media/probe": "post", "/media/inspect-faces": "post",
                    "/media/preview-frame": "get", "/models": "get", "/models/{model_id}/download": "post",
                    "/facesets": "get", "/jobs": "post", "/jobs/{job_id}": "get",
                    "/jobs/{job_id}/cancel": "post", "/jobs/{job_id}/pause": "post", "/jobs/{job_id}/resume": "post"}
        assert all(method in paths["/api/v1" + path] for path, method in expected.items())
        capabilities = client.get("/api/v1/system/capabilities")
        assert capabilities.status_code == 200
        data = capabilities.json()
        assert data["rendering_worker_available"] is False
        # The catalogue includes candidate and remote-only records; these are not installed model files.
        assert data["models"]
        assert data["legal_parameter_bounds"]["properties"]["max_concurrent_jobs"]["maximum"] == 8
        assert data["event_schema"]["discriminator"]["propertyName"] == "type"
        assert client.get("/api/v1/facesets").json()[0]["faceset_id"] == "alice"
        assert client.post("/api/v1/models/absent/download").status_code == 404


def test_config_put_backup_reload_secret_rejection_and_csrf(application, caplog) -> None:
    app, _ = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        first = client.get("/api/v1/config").json()
        second = {**first, "vram_limit_mb": 4096}
        assert client.put("/api/v1/config", json=first).status_code == 200
        assert client.put("/api/v1/config", json=second).status_code == 200
        assert client.get("/api/v1/config").json() == second
        saved = app.state.root / "roop_3d/.runtime/config.json"
        assert json.loads(saved.read_text()) == second
        assert json.loads(saved.with_suffix(".json.bak").read_text()) == first
        with caplog.at_level(logging.INFO):
            bad = client.put("/api/v1/config", json={**second, "api_key": "VERY_SECRET_VALUE"})
        assert bad.status_code == 422
        assert "VERY_SECRET_VALUE" not in bad.text + caplog.text
        assert client.put("/api/v1/config", json=first, headers={"Origin": "https://evil.example"}).status_code == 403
        assert json.loads(saved.read_text()) == second
        assert client.put("/api/v1/config", content='{"vram_limit_mb":NaN}', headers={"Content-Type": "application/json"}).status_code == 422
        assert client.put("/api/v1/config", json={**second, "port": 8040}).status_code == 409
    restored = create_app(repo_root=app.state.root)
    assert restored.state.settings.vram_limit_mb == 4096


def test_job_rest_idempotency_controls_and_validation(application) -> None:
    app, payload = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        assert client.post("/api/v1/jobs", json=payload).status_code == 422
        created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "job-one"})
        assert created.status_code == 201
        job_id = created.json()["job_id"]
        assert client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "job-one"}).json() == created.json()
        assert client.get("/api/v1/jobs").json() == [created.json()]
        assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
        assert client.post(f"/api/v1/jobs/{job_id}/resume").status_code == 409
        assert client.post(f"/api/v1/jobs/{job_id}/pause").json()["state"] == "paused"
        assert client.post(f"/api/v1/jobs/{job_id}/resume").json()["state"] == "queued"
        assert client.post(f"/api/v1/jobs/{job_id}/cancel").json()["state"] == "cancelled"
        assert client.post(f"/api/v1/jobs/{job_id}/pause").status_code == 409
        changed = {**payload, "face_mappings": [{"track_id": "track_0001", "faceset_id": "alice"}]}
        assert client.post("/api/v1/jobs", json=changed, headers={"Idempotency-Key": "job-one"}).json()["error"]["code"] == "IDEMPOTENCY_CONFLICT"
        assert client.post("/api/v1/jobs", json=changed, headers={"Idempotency-Key": "job-two"}).json()["error"]["code"] == "TRACK_NOT_FOUND"
        unknown_set = {**payload, "face_mappings": [{"track_id": "track_0000", "faceset_id": "unknown"}]}
        assert client.post("/api/v1/jobs", json=unknown_set, headers={"Idempotency-Key": "job-three"}).json()["error"]["code"] == "FACESET_NOT_FOUND"
        assert client.post("/api/v1/jobs", json={**payload, "processing": "remote"}, headers={"Idempotency-Key": "remote"}).status_code == 422
        consent = {**payload, "processing": "remote", "remote_consent": True}
        assert client.post("/api/v1/jobs", json=consent, headers={"Idempotency-Key": "remote"}).json()["error"]["code"] == "REMOTE_PROCESSING_DISABLED"
        (app.state.root / payload["target_path"]).write_bytes(b"changed")
        assert client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "new-job"}).json()["error"]["code"] == "INSPECTION_TARGET_MISMATCH"
        # Exact replay is still available after source changes and after restart.
        assert client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "job-one"}).status_code == 200
    with TestClient(create_app(Settings(), repo_root=app.state.root), base_url="http://127.0.0.1:8030") as client:
        assert client.get(f"/api/v1/jobs/{job_id}").json()["state"] == "cancelled"


def receive_type(socket, event_type: str) -> dict:
    for _ in range(32):
        value = socket.receive_json()
        if value["type"] == event_type:
            return value
    raise AssertionError(f"No {event_type} event received")


def test_websocket_fanout_state_progress_and_live_telemetry(application) -> None:
    app, payload = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        with client.websocket_connect("ws://127.0.0.1:8030/api/v1/ws/events") as left, client.websocket_connect("ws://127.0.0.1:8030/api/v1/ws/events") as right:
            assert left.receive_json()["type"] == right.receive_json()["type"] == "stream.ready"
            created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "ws"}).json()
            a, b = receive_type(left, "job.state"), receive_type(right, "job.state")
            assert a == b and a["state"] == "queued"
            job_id = created["job_id"]
            app.state.jobs.transition(job_id, JobState.preparing)
            app.state.jobs.transition(job_id, JobState.running)
            app.state.jobs.progress(job_id, JobProgress(frame=1, total_frames=8, fps=12, stage="fixture", stage_latency_ms=4))
            value = receive_type(left, "job.progress")
            assert value["frame"] == 1 and value["fps"] == 12 and value["stage_latency_ms"] == 4
            EVENT_ADAPTER.validate_python(value)
            telemetry = receive_type(left, "system.telemetry")
            assert telemetry["hardware"]["system_ram_total_mib"] > 0
        assert app.state.events.subscriber_count == 0


@pytest.mark.parametrize("headers", [{"Origin": "https://evil.example"}, {"Origin": "null"},
                                      {"Host": "evil.example"}, {"Host": "127.0.0.1:9999"}])
def test_websocket_origin_and_host_checks(application, headers) -> None:
    app, _ = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        with pytest.raises(WebSocketDisconnect) as error:
            with client.websocket_connect("ws://127.0.0.1:8030/api/v1/ws/events", headers=headers):
                pass
        assert error.value.code == 1008


def test_slow_subscriber_gets_explicit_gap_and_events_reject_secrets() -> None:
    async def scenario():
        broker = EventBroker(capacity=1)
        with broker.subscribe() as queue:
            payload = {"type": "job.state", "job_id": "test", "previous_state": None, "state": "queued", "revision": 0}
            broker.publish(payload)
            broker.publish(payload)
            await asyncio.sleep(0)
            assert (await queue.get()).type == "stream.gap"
            with pytest.raises(ValidationError):
                broker.publish({**payload, "api_key": "SECRET"})
    asyncio.run(scenario())


def test_structured_secret_redaction() -> None:
    data = {"authorization": "Bearer TOKEN", "nested": {"api_key": "TOKEN"}, "normal": "visible",
            "value": SecretStr("TOKEN")}
    assert "TOKEN" not in json.dumps(redact(data))
    assert redact(data)["normal"] == "visible"
    record = logging.LogRecord("test", logging.INFO, "file", 1, json.dumps({"password": "TOKEN"}), (), None)
    assert "TOKEN" not in JsonFormatter().format(record)


def test_duplicate_origin_header_cannot_override_csrf_check(application) -> None:
    app, _ = application
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/config", headers=[("Origin", "https://evil.example"),
                                                       ("Origin", "http://127.0.0.1:8030")])
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "AMBIGUOUS_HEADERS"


def test_native_uvicorn_websocket_transport_and_origin_denial(application) -> None:
    import httpx
    import uvicorn
    from websockets.asyncio.client import connect
    from websockets.exceptions import InvalidStatus
    from roop_3d.server.__main__ import ReadyServer

    _, payload = application

    async def scenario():
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            listener.setblocking(False)
            port = listener.getsockname()[1]
            app = create_app(Settings(port=port), repo_root=application[0].state.root)
            server = ReadyServer(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None,
                                               access_log=False, ws="websockets-sansio"))
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                for _ in range(100):
                    if server.started:
                        break
                    await asyncio.sleep(0.01)
                assert server.started
                address = f"ws://127.0.0.1:{port}/api/v1/ws/events"
                async with connect(address, origin=f"http://127.0.0.1:{port}", proxy=None) as connection:
                    ready = json.loads(await asyncio.wait_for(connection.recv(), 5))
                    assert ready["type"] == "stream.ready"
                    async with httpx.AsyncClient() as client:
                        response = await client.post(f"http://127.0.0.1:{port}/api/v1/jobs", json=payload,
                                                     headers={"Idempotency-Key": "native-ws"})
                        assert response.status_code == 201
                    for _ in range(8):
                        event = json.loads(await asyncio.wait_for(connection.recv(), 5))
                        if event["type"] == "job.state":
                            assert event["job_id"] == response.json()["job_id"]
                            break
                    else:
                        raise AssertionError("No native WebSocket job event")
                with pytest.raises(InvalidStatus) as error:
                    async with connect(address, origin="https://evil.example", proxy=None):
                        pass
                assert error.value.response.status_code == 403
            finally:
                server.should_exit = True
                await asyncio.wait_for(serving, 10)
    asyncio.run(scenario())
