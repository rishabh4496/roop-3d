"""Endpoints behind the face-bank UI: angle crops, unlock, rescan, and the engine status / switch-failure read-back."""
from __future__ import annotations

import numpy as np
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Settings
from roop_3d.engine import active_models
from roop_3d.identity.target_lock import FaceRef, TargetLockStore
from roop_3d.server import routes, target_lock as target_routes
from roop_3d.server.pipeline_status import PipelineTracker


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def person(seed: int) -> np.ndarray:
    return unit(np.random.default_rng(seed).normal(size=512))


def crop(value: int) -> np.ndarray:
    return np.full((256, 256, 3), value, np.uint8)


@pytest.fixture()
def client():
    active_models.reset()
    app = FastAPI()

    @app.exception_handler(DomainError)
    async def domain_error(request: Request, error: DomainError):
        return JSONResponse({"code": error.code, "message": error.message}, status_code=error.status)

    app.include_router(routes.router)
    app.include_router(target_routes.router)
    app.state.settings = Settings()
    app.state.pipeline = PipelineTracker(lambda payload: None)
    app.state.target_locks = TargetLockStore()
    yield TestClient(app)
    active_models.reset()


def make_lock(client, seed=1, yaw=0.0, pitch=0.0):
    store: TargetLockStore = client.app.state.target_locks
    face = FaceRef(person(seed), yaw, pitch, 0.0, [0, 0, 100, 100], quality=250.0, crop=crop(120))
    return store, store.create("clip.mp4", 3, face)


# -- angle crops -------------------------------------------------------------------------------------------------------

def test_angle_crop_serves_the_stored_jpeg_for_a_captured_cell_only(client):
    store, lock = make_lock(client)
    ok = client.get(f"/api/v1/target/{lock.lock_id}/angle/2/1/crop.jpg")          # yaw 0, pitch 0 -> frontal / level
    assert ok.status_code == 200 and ok.headers["content-type"] == "image/jpeg" and ok.content[:2] == b"\xff\xd8"
    assert client.get(f"/api/v1/target/{lock.lock_id}/angle/0/0/crop.jpg").json()["code"] == "ANGLE_NOT_CAPTURED"
    assert client.get("/api/v1/target/" + "0" * 32 + "/angle/2/1/crop.jpg").status_code == 404
    assert client.get(f"/api/v1/target/{lock.lock_id}/angle/5/1/crop.jpg").status_code == 422     # outside the 5 x 3 grid


# -- unlock ------------------------------------------------------------------------------------------------------------

def test_unlock_removes_the_lock_but_keeps_the_face_slot_and_its_bank(client):
    store, lock = make_lock(client)
    registry = store.registry("clip.mp4")
    assert registry.clusters[lock.cluster_id].locked is True
    response = client.delete(f"/api/v1/target/{lock.lock_id}")
    assert response.status_code == 200 and response.json() == {"lock_id": lock.lock_id, "face_id": lock.cluster_id - 1, "unlocked": True}
    assert store.get(lock.lock_id) is None and registry.clusters[lock.cluster_id].locked is False
    assert len(registry.clusters[lock.cluster_id].bank) == 1                      # the reference survives for re-locking
    assert client.delete(f"/api/v1/target/{lock.lock_id}").status_code == 404
    # locking the same person again attaches a new lock to the same immutable slot
    again = store.create("clip.mp4", 9, FaceRef(person(1), 0, 0, 0, [0, 0, 100, 100], quality=250.0, crop=crop(120)))
    assert again.cluster_id == lock.cluster_id and registry.clusters[lock.cluster_id].locked is True


def test_unlocking_one_of_two_faces_leaves_the_other_locked(client):
    store, first = make_lock(client, seed=1)
    second = store.create("clip.mp4", 3, FaceRef(person(2), 0, 0, 0, [200, 0, 300, 100], quality=250.0, crop=crop(60)))
    assert second.cluster_id != first.cluster_id
    client.delete(f"/api/v1/target/{first.lock_id}")
    registry = store.registry("clip.mp4")
    assert not registry.clusters[first.cluster_id].locked and registry.clusters[second.cluster_id].locked


# -- rescan ------------------------------------------------------------------------------------------------------------

def test_rescan_forgets_auto_references_keeps_anchor_and_user_angles_and_refuses_a_running_scan(client, monkeypatch):
    store, lock = make_lock(client)
    registry = store.registry("clip.mp4")
    cluster = registry.clusters[lock.cluster_id]
    auto = FaceRef(person(1), 40.0, 0.0, 0.0, [0, 0, 100, 100], quality=300.0, crop=crop(30))
    user = FaceRef(person(1), -40.0, 20.0, 0.0, [0, 0, 100, 100], quality=300.0, crop=crop(40))
    cluster.consider(auto.entry(5, "auto"))
    cluster.consider(user.entry(6, "user"), force=True)
    assert len(cluster.bank) == 3

    started: list[str] = []
    monkeypatch.setattr(target_routes.threading, "Thread",
                        lambda target, args, name, daemon: type("T", (), {"start": lambda self: started.append(name)})())
    response = client.post(f"/api/v1/target/{lock.lock_id}/rescan", json={"sample_count": 8})
    assert response.status_code == 200 and response.json()["scan"]["state"] == "running"
    assert sorted(e.source for e in cluster.bank) == ["anchor", "user"] and started == [f"target-rescan-{lock.lock_id[:6]}"]

    again = client.post(f"/api/v1/target/{lock.lock_id}/rescan", json={})
    assert again.status_code == 409 and again.json()["code"] == "TARGET_SCAN_RUNNING"
    assert client.post("/api/v1/target/" + "1" * 32 + "/rescan", json={}).status_code == 404


def test_a_scan_stops_when_its_target_is_unlocked(client, monkeypatch):
    store, lock = make_lock(client)

    class Frames:
        reads = 0

        def info(self, path):
            return type("Info", (), {"total_frames": 50})()

        def read(self, path, frame_index):
            Frames.reads += 1
            store.remove(lock.lock_id)                                           # the user unlocks mid-scan
            return type("Frame", (), {"image": crop(0), "frame_index": frame_index})()

        def analyzed_faces(self, *args):
            return []

    request = type("R", (), {"app": type("A", (), {"state": type("S", (), {
        "target_locks": store, "media": type("M", (), {"frames": Frames()})()})()})()})()
    target_routes.scan_keyframes(request, lock, 10, 0.5)
    assert Frames.reads == 1 and lock.scan["state"] == "cancelled"


# -- engine status -----------------------------------------------------------------------------------------------------

def test_pipeline_status_reports_loaded_models_precision_and_nothing_when_idle(client):
    idle = client.get("/api/v1/pipeline/status").json()
    assert idle["active"] == {"detector": None, "swapper": None, "enhancer": None}
    assert idle["precision"]["label"] is None and idle["failures"] == [] and idle["pipeline"]["engine_state"] == "IDLE"
    assert idle["selected"]["swapper_model"] == "inswapper_128"

    active_models.activate("detector", "scrfd", "models/scrfd_2.5g_kps.onnx", "CUDAExecutionProvider", "auto", "a" * 64)
    active_models.activate("swapper", "inswapper_128", "models/inswapper_128.onnx", "TensorrtExecutionProvider", "fp16", "b" * 64)
    status = client.get("/api/v1/pipeline/status").json()
    assert status["active"]["swapper"]["filename"] == "inswapper_128.onnx"
    assert status["precision"]["label"] == "TensorRT FP16 Mixed" and status["precision"]["stage"] == "swapper"
    assert status["precision"]["stages"] == {"swapper": "TensorRT FP16 Mixed", "detector": "CUDA (auto precision)"}


@pytest.mark.parametrize("provider, policy, label", [
    ("TensorrtExecutionProvider", "fp16", "TensorRT FP16 Mixed"), ("TensorrtExecutionProvider", "auto", "TensorRT (auto precision)"),
    ("TensorrtExecutionProvider", "fp32", "TensorRT FP32"), ("CUDAExecutionProvider", "fp16", "CUDA FP16"),
    ("CPUExecutionProvider", "fp16", "CPU FP32"), (None, "fp16", None)])
def test_precision_label_never_claims_more_than_the_session_was_built_with(provider, policy, label):
    assert routes.precision_label(provider, policy) == label


def test_a_failed_model_switch_is_reported_with_the_exact_reason_and_cleared_by_a_working_session(client):
    active_models.activate("swapper", "inswapper_128", "models/inswapper_128.onnx", "CUDAExecutionProvider", "auto", "c" * 64)
    with pytest.raises(DomainError):
        with active_models.switch_guard("swapper", "simswap"):
            raise DomainError("MODEL_NOT_INSTALLED", "Selected swapper has no installed local ONNX weights", 404)
    status = client.get("/api/v1/pipeline/status").json()
    [failure] = status["failures"]
    assert (failure["stage"], failure["model_id"], failure["code"]) == ("swapper", "simswap", "MODEL_NOT_INSTALLED")
    assert failure["reason"] == "Selected swapper has no installed local ONNX weights"
    assert status["active"]["swapper"]["model_id"] == "inswapper_128"             # the old model is shown as such, not as the new one

    active_models.activate("swapper", "inswapper_128", "models/inswapper_128.onnx", "CUDAExecutionProvider", "auto", "c" * 64)
    assert client.get("/api/v1/pipeline/status").json()["failures"] == []
    first = active_models.report_failure("detector", "yoloface", ValueError("boom"))
    second = active_models.report_failure("detector", "retinaface", RuntimeError("worse"))
    assert second.id > first.id and active_models.failures()[0]["code"] == "RuntimeError"
    active_models.clear_failure("detector")
    assert active_models.failures() == []
