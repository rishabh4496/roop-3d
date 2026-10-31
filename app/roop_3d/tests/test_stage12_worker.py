from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import av
import numpy as np
from fastapi.testclient import TestClient

from roop_3d.core.hardware import GPU, Hardware
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.engine.scheduler import RuntimeScheduler
from roop_3d.engine.vram_governor import VRAMGovernor
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, PoseBinSummary, QualityMetrics
from roop_3d.jobs.schema import JobState
from roop_3d.jobs.worker import JobWorker
from roop_3d.media.schema import BoundingBox, FaceInspection, FaceTrack, PoseRange
from roop_3d.server.app import create_app


def _video(path: Path, count: int = 8) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 96, 64, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 25)
        stream.codec_context.max_b_frames = 0
        for index in range(count):
            image = np.full((64, 96, 3), index * 20, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 25)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return path


def _faceset(faceset_id: str) -> FaceSetDocument:
    embedding = tuple([1.0] + [0.0] * 511)
    view = FaceView(index=0, image_bytes=b"test-crop", embedding=embedding,
        landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)),
        bbox=(0.0, 0.0, 32.0, 32.0), yaw=0.0, pitch=0.0, roll=0.0,
        expression={}, quality=QualityMetrics(), pose_bin="center_center")
    summary = PoseBinSummary(yaw_bin="center", pitch_bin="center", members=(0,), support=1,
                             representative_index=0, embedding=embedding)
    return FaceSetDocument(faceset_id=faceset_id, name=faceset_id, source_sha256="a" * 64,
        archive_format="v2_standard", canonical_embedding=embedding, medoid_embedding=embedding,
        views=(view,), pose_bins={"center_center": summary})


class _Reader:
    def read(self, path: Path, *, faceset_id: str):
        assert path.is_file()
        return _faceset(faceset_id)


class _Session:
    def __init__(self, config, *, delay: float = 0.0, oom_on_cuda: bool = False,
                 cpu_started: Event | None = None, allow_cpu: Event | None = None):
        self.config = config
        self.delay = delay
        self.oom_on_cuda = oom_on_cuda
        self.cpu_started, self.allow_cpu = cpu_started, allow_cpu
        self.compositor = SimpleNamespace(restorer=None)
        self.released = False
        self.prepared_source_ids = None

    def prepare_session(self, source, *, track_ids, sources_by_track, retarget):
        self.prepared_source_ids = set(sources_by_track)
        assert set(track_ids) == self.prepared_source_ids
        assert retarget == "auto"

    def infer_frame(self, image):
        if self.config.provider == "CUDAExecutionProvider" and self.oom_on_cuda:
            raise RuntimeError("CUDA out of memory in fixture allocation")
        if self.config.provider == "CPUExecutionProvider" and self.cpu_started is not None:
            self.cpu_started.set()
            if self.allow_cpu is not None:
                self.allow_cpu.wait(5)
        if self.delay:
            time.sleep(self.delay)
        return SimpleNamespace(replacements=(SimpleNamespace(track_id=0),))

    def composite_frame(self, image, result, **kwargs):
        return 255 - image

    def capability(self):
        return {"identity_swapper": {"inference_state": "VERIFIED_LOCAL"},
                "portrait": {"state": "UNSUPPORTED"}, "lip_sync": {"state": "UNSUPPORTED"}}

    def release_session(self):
        self.released = True


def _fixture(tmp_path: Path, *, count: int = 8, settings: Settings | None = None,
             session_factory=None, hardware_probe=None):
    settings = settings or Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider",
                                    audio_policy="drop")
    app = create_app(settings, repo_root=tmp_path)
    target_relative = "roop_3d/.runtime/media/target.mp4"
    target = _video(tmp_path / target_relative, count)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    inspection_id = "a" * 32
    inspection = FaceInspection(inspection_id=inspection_id, target_sha256=digest,
        detector_model_id="fixture", detector_sha256="0" * 64,
        tracks=(FaceTrack(track_id="track_0000", persistent_id=0, first_frame=0,
            last_frame=count - 1, first_timestamp_seconds=0, last_timestamp_seconds=(count - 1) / 25,
            bounding_box=BoundingBox(x1=1, y1=1, x2=30, y2=30), pose_range=PoseRange(),
            thumbnail_base64="", sampled_detections=max(1, count)),),
        decoded_frames=count, sampled_frames=count, truncated=False,
        tracking_method="iou_arcface_hungarian")
    atomic_json(tmp_path, f"roop_3d/.runtime/inspections/{inspection_id}.json", inspection.model_dump(mode="json"))
    atomic_json(tmp_path, "roop_3d/.runtime/facesets/alice.json",
                {"faceset_id": "alice", "name": "Alice", "item_count": 1, "schema_version": 1})
    archive = tmp_path / "roop_3d/outputs/facesets/alice.fsz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    archive.write_bytes(b"read-only fixture archive")

    factory = session_factory or (lambda catalog, config: _Session(config))
    runner = MasterPipelineRunner(tmp_path, app.state.models, app.state.media,
        lambda: app.state.settings, faceset_reader=_Reader(), pipeline_factory=factory)
    scheduler = RuntimeScheduler(VRAMGovernor(hardware_probe)) if hardware_probe else RuntimeScheduler(VRAMGovernor())
    worker = JobWorker(tmp_path, app.state.jobs, lambda: app.state.settings, app.state.models,
                       app.state.media, pipeline_runner=runner, scheduler=scheduler, poll_interval=0.01)
    app.state.worker = worker
    payload = {"target_path": target_relative, "inspection_id": inspection_id,
               "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}]}
    return app, worker, payload


def _wait_state(client: TestClient, job_id: str, state: str, timeout: float = 15.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = client.get(f"/api/v1/jobs/{job_id}").json()
        if record["state"] == state:
            return record
        if record["state"] in {"failed", "cancelled"}:
            raise AssertionError(f"Job entered {record['state']}: {record.get('failure')}")
        time.sleep(0.01)
    raise AssertionError(f"Job did not reach {state}")


def test_post_job_runs_pipeline_publishes_manifest_and_streams_progress(tmp_path: Path):
    app, worker, payload = _fixture(tmp_path)
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            with client.websocket_connect("ws://127.0.0.1:8030/api/v1/ws/events") as socket:
                assert socket.receive_json()["type"] == "stream.ready"
                created = client.post("/api/v1/jobs", json=payload,
                                      headers={"Idempotency-Key": "stage12-e2e"})
                assert created.status_code == 201
                job_id = created.json()["job_id"]
                events = []
                while not any(event.get("type") == "job.state" and event.get("state") == "succeeded"
                              for event in events):
                    events.append(socket.receive_json())
                result = client.get(f"/api/v1/jobs/{job_id}").json()
                assert result["state"] == "succeeded"
                assert result["output"]["frames"] == 8
                assert any(event["type"] == "job.progress" for event in events)
                output = tmp_path / result["output"]["output_path"]
                manifest_path = tmp_path / result["output"]["manifest_path"]
                manifest = json.loads(manifest_path.read_text("utf-8"))
                assert output.is_file() and hashlib.sha256(output.read_bytes()).hexdigest() == manifest["output_sha256"]
                assert manifest["stage12"]["job_id"] == job_id
                assert manifest["verified_frames"] == 8
                deadline = time.monotonic() + 3
                while (tmp_path / f"roop_3d/.runtime/jobs/{job_id}").exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert not (tmp_path / f"roop_3d/.runtime/jobs/{job_id}").exists()
                # Starlette's test client sends a disconnect then immediately
                # cancels its server task on context exit. Let the receive loop
                # consume the explicit disconnect so teardown remains graceful.
                socket.close()
                time.sleep(0.02)
    finally:
        worker.stop()


def test_worker_pause_checkpoint_resume_continues_to_success(tmp_path: Path):
    app, worker, payload = _fixture(tmp_path, count=24,
        session_factory=lambda catalog, config: _Session(config, delay=0.025))
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            created = client.post("/api/v1/jobs", json=payload,
                                  headers={"Idempotency-Key": "stage12-pause"}).json()
            job_id = created["job_id"]
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                state = client.get(f"/api/v1/jobs/{job_id}").json()
                if state["state"] == "running" and state["progress"]["frame"] >= 2:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Worker did not begin rendering")
            paused = client.post(f"/api/v1/jobs/{job_id}/pause").json()
            assert paused["state"] == "pausing"
            paused = _wait_state(client, job_id, "paused")
            checkpoint_path = tmp_path / f"roop_3d/.runtime/jobs/{job_id}/checkpoint.json"
            checkpoint = json.loads(checkpoint_path.read_text("utf-8"))
            assert checkpoint["completed_frame"] >= 0
            assert paused["progress"]["frame"] > 0
            resumed = client.post(f"/api/v1/jobs/{job_id}/resume").json()
            assert resumed["state"] == "queued"
            succeeded = _wait_state(client, job_id, "succeeded")
            assert succeeded["output"]["frames"] == 24
            deadline = time.monotonic() + 3
            while checkpoint_path.parent.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert not checkpoint_path.parent.exists()
    finally:
        worker.stop()


def test_simulated_cuda_oom_retries_on_cpu_without_partial_manifest(tmp_path: Path):
    settings = Settings(hardware_profile="12gb", preferred_provider="CUDAExecutionProvider",
                        audio_policy="drop")
    cpu_started, allow_cpu = Event(), Event()
    observed_providers: list[str] = []

    def factory(catalog, config):
        observed_providers.append(config.provider)
        return _Session(config, oom_on_cuda=True, cpu_started=cpu_started, allow_cpu=allow_cpu)

    snapshot = Hardware(timestamp=datetime.now(timezone.utc).isoformat(), os={"system": "Windows"},
        logical_cpu_count=16, gpus=[GPU(index=0, name="simulated", total_vram_mib=12288,
            free_vram_mib=12000, effective_vram_budget_mib=11200, driver_version="fixture")],
        gpu_probe_state="VERIFIED_LOCAL", advertised_providers=["CUDAExecutionProvider"],
        provider_probe_state="VERIFIED_LOCAL")
    app, worker, payload = _fixture(tmp_path, settings=settings, session_factory=factory,
                                    hardware_probe=lambda current: snapshot)
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            created = client.post("/api/v1/jobs", json=payload,
                                  headers={"Idempotency-Key": "stage12-oom"}).json()
            job_id = created["job_id"]
            assert cpu_started.wait(10)
            output = tmp_path / f"roop_3d/.runtime/outputs/roop3d_{job_id}.mp4"
            manifest = Path(str(output) + ".manifest.json")
            assert observed_providers[:2] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
            assert not output.exists() and not manifest.exists()
            allow_cpu.set()
            succeeded = _wait_state(client, job_id, "succeeded")
            assert succeeded["output"]["frames"] == 8
            content = json.loads(manifest.read_text("utf-8"))
            assert content["verified_frames"] == 8
            assert content["output_sha256"] == succeeded["output"]["output_sha256"]
    finally:
        allow_cpu.set()
        worker.stop()
