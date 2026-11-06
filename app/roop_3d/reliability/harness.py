"""Test and soak harness: synthetic long videos, a stub inference stage and a ready-to-run job rig.

What is real and what is stubbed - stated plainly because it decides what a soak result proves:

* **Real**: video decode (PyAV), per-frame PNG checkpoint with atomic fsync'd writes, checkpoint JSON/BIN/NPZ rewrites,
  job state machine and file locks, worker threads, scheduler/governor admission, mux/encode, manifest + atomic
  publish, cleanup, WebSocket-less progress reporting, crash recovery.
* **Stubbed**: hardware snapshots are injected; dependency identities use small local hash-verified fixture files; the neural inference step (detector / swapper / restorer). ``StubSession`` returns a deterministic
  frame so thousands of frames can run in minutes. Model-session memory behaviour is therefore measured separately
  (``scenario_model_switch`` loads and evicts real ONNX Runtime sessions).

Never imported by the server; lives in the package so ``python -m roop_3d.reliability.soak`` can use it.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from typing import Any, Callable

import av
import numpy as np

from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.engine.scheduler import RuntimeScheduler
from roop_3d.engine.vram_governor import VRAMGovernor
from roop_3d.faceset.schemas import FaceSetDocument, FaceView, PoseBinSummary, QualityMetrics
from roop_3d.jobs.schema import JobRecord, JobRequest, JobState
from roop_3d.jobs.worker import JobWorker
from roop_3d.media.schema import BoundingBox, FaceInspection, FaceTrack, PoseRange

TERMINAL = {JobState.succeeded, JobState.failed, JobState.cancelled}


def make_video(path: Path, frames: int, *, size: tuple[int, int] = (96, 64), fps: int = 25, noise: bool = True,
               seed: int = 7) -> Path:
    """H.264 test clip with a moving gradient plus noise so PNG checkpoints are a realistic size, not 200 bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    rng = np.random.default_rng(seed)
    base = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None].repeat(height, 0).repeat(3, 2)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.time_base = stream.codec_context.time_base = Fraction(1, fps)
        stream.codec_context.max_b_frames = 0
        stream.codec_context.options = {"preset": "ultrafast", "crf": "30"}
        for index in range(frames):
            image = np.roll(base, index % width, axis=1)
            if noise:
                image = image ^ rng.integers(0, 24, image.shape, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    return path


def faceset_document(faceset_id: str) -> FaceSetDocument:
    embedding = tuple([1.0] + [0.0] * 511)
    view = FaceView(index=0, image_bytes=b"stub-crop", embedding=embedding,
                    landmarks=((10.0, 10.0), (20.0, 10.0), (15.0, 15.0), (12.0, 20.0), (18.0, 20.0)),
                    bbox=(0.0, 0.0, 32.0, 32.0), yaw=0.0, pitch=0.0, roll=0.0, expression={},
                    quality=QualityMetrics(), pose_bin="center_center")
    summary = PoseBinSummary(yaw_bin="center", pitch_bin="center", members=(0,), support=1,
                             representative_index=0, embedding=embedding)
    return FaceSetDocument(faceset_id=faceset_id, name=faceset_id, source_sha256="a" * 64, archive_format="v2_standard",
                           canonical_embedding=embedding, medoid_embedding=embedding, views=(view,),
                           pose_bins={"center_center": summary})


class StubReader:
    def read(self, path: Path, *, faceset_id: str):
        assert path.is_file()
        return faceset_document(faceset_id)


class StubSession:
    """Deterministic stand-in for ``ReferencePipeline``. ``on_frame(index)`` lets a scenario inject faults."""

    frames_seen = 0
    live_instances = 0
    _lock = threading.Lock()

    def __init__(self, config, *, tracks: int = 1, delay: float = 0.0, on_frame: Callable[[int], None] | None = None):
        self.config, self.tracks, self.delay, self.on_frame = config, tracks, delay, on_frame
        self.compositor = SimpleNamespace(restorer=None)
        self.released = False
        self._index = 0
        with StubSession._lock:
            StubSession.live_instances += 1

    def prepare_session(self, source, *, track_ids, sources_by_track, retarget, **kwargs):
        assert set(track_ids) == set(sources_by_track)

    def infer_frame(self, image, **kwargs):
        index, self._index = self._index, self._index + 1
        StubSession.frames_seen += 1
        if self.on_frame is not None:
            self.on_frame(index)
        if self.delay:
            time.sleep(self.delay)
        return SimpleNamespace(replacements=tuple(SimpleNamespace(track_id=i) for i in range(self.tracks)))

    def composite_frame(self, image, result, **kwargs):
        return 255 - image

    def capability(self):
        return {"identity_swapper": {"inference_state": "VERIFIED_LOCAL"},
                "portrait": {"state": "UNSUPPORTED"}, "lip_sync": {"state": "UNSUPPORTED"}}

    def release_session(self):
        if not self.released:
            self.released = True
            with StubSession._lock:
                StubSession.live_instances -= 1


class Rig:
    """One app root with a worker, a target video and N tracks mapped to N FaceSets."""

    def __init__(self, root: Path, app, worker: JobWorker, payload: dict, target: Path, frames: int):
        self.root, self.app, self.worker, self.payload, self.target, self.frames = root, app, worker, payload, target, frames
        self.store = app.state.jobs
        self._keys = 0

    def submit(self, key: str | None = None, **overrides) -> JobRecord:
        self._keys += 1
        request = JobRequest.model_validate({**self.payload, **overrides})
        record, _ = self.store.create(request, key or f"rig-{time.time_ns()}-{self._keys}")
        self.worker.wake()
        return record

    def job(self, job_id: str) -> JobRecord:
        return self.store.get(job_id)

    def wait(self, job_id: str, states: set[JobState] | JobState | None = None, timeout: float = 60.0) -> JobRecord:
        wanted = {states} if isinstance(states, JobState) else (states or TERMINAL)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.store.get(job_id)
            if record.state in wanted:
                return record
            self.worker.wake()
            time.sleep(0.01)
        raise TimeoutError(f"job {job_id[:8]} still {self.store.get(job_id).state.value}, wanted {sorted(s.value for s in wanted)}")

    def wait_frames(self, job_id: str, frames: int, timeout: float = 60.0) -> JobRecord:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.store.get(job_id)
            if record.progress.frame >= frames or record.state in TERMINAL:
                return record
            time.sleep(0.005)
        raise TimeoutError("job did not reach the requested frame")

    def workdir(self, job_id: str) -> Path:
        return self.root / "roop_3d/.runtime/jobs" / job_id

    def output_files(self) -> list[Path]:
        out = self.root / self.app.state.settings.output_root
        return sorted(p for p in out.glob("roop3d_*") if not p.name.endswith((".lock",))) if out.is_dir() else []


def fixture_catalog(root: Path):
    """Small, hash-verified identities for stub inference; no installed weights are read.

    These files are dependency identity fixtures, not executable neural models. They
    exercise resume/drift verification through the real catalog and local provider.
    Real ONNX session soak scenarios construct their own production catalog.
    """
    from roop_3d.models.catalog import ModelCatalog
    from roop_3d.models.local_provider import LocalWeightProvider
    from roop_3d.models.registry import Manifest, ModelRecord, ModelRegistry
    directory = root / "roop_3d/.runtime/harness_models"
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for role, name in (("detector", "scrfd_10g_bnkps.onnx"), ("recognizer", "w600k_r50.onnx"),
                       ("identity_swapper", "inswapper_128.onnx")):
        data = ("Roop 3D deterministic dependency fixture: " + role).encode()
        path = directory / name
        if not path.exists():
            path.write_bytes(data)
        records.append(ModelRecord(model_id="reliability_fixture_" + role, name=name, role=role,
            role_evidence="Stub inference dependency identity; no neural execution", status="verified_local",
            filenames=(name,), expected_sha256=hashlib.sha256(data).hexdigest(), byte_size=len(data),
            locator={"scope": "fixture", "path": name},
            input_output_contract={"outputs": [{"name": str(i)} for i in range(9)] if role == "detector" else []},
            evidence="Deterministic local test bytes", license_id="CC0-1.0", license_notice="Generated test fixture",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/", commercial_use="allow"))
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Reliability stub dependency fixtures", models=tuple(records)))
    catalog = ModelCatalog(root, registry=registry)
    catalog.provider = LocalWeightProvider(registry, root=root, scopes={"fixture": directory})
    return catalog


def build_rig(root: Path, *, frames: int = 8, tracks: int = 1, session_factory=None, size=(96, 64), fps: int = 25,
              settings: Settings | None = None, checkpoint_every: int | None = None, poll_interval: float = 0.01,
              hardware_probe=None, video: Path | None = None, delay: float = 0.0, faceset_ids: list[str] | None = None,
              on_frame: Callable[[int], None] | None = None, start: bool = True) -> Rig:
    """Build (or re-open, if ``root`` already holds a rig) an app + worker. ``start=False`` leaves the worker stopped."""
    from roop_3d.server.app import create_app
    settings = settings or Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider", audio_policy="drop")
    app = create_app(settings, repo_root=root)
    app.state.models = fixture_catalog(root)
    target = video or (root / "roop_3d/.runtime/media/target.mp4")
    target_relative = target.relative_to(root).as_posix()
    if not target.exists():
        make_video(target, frames, size=size, fps=fps)
    digest = hashlib.sha256(target.read_bytes()).hexdigest() if target.stat().st_size < 512 * 1024 * 1024 else None
    if digest is None:
        h = hashlib.sha256()
        with target.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                h.update(block)
        digest = h.hexdigest()
    inspection_id = "a" * 32
    ids = list(faceset_ids) if faceset_ids else [f"face{index}" for index in range(tracks)]
    assert len(ids) == tracks
    fps_ = Fraction(fps)
    track_models = tuple(FaceTrack(
        track_id=f"track_{index:04d}", persistent_id=index, first_frame=0, last_frame=max(0, frames - 1),
        first_timestamp_seconds=0, last_timestamp_seconds=max(0, frames - 1) / float(fps_),
        bounding_box=BoundingBox(x1=1, y1=1, x2=30, y2=30), pose_range=PoseRange(), thumbnail_base64="",
        sampled_detections=max(1, frames)) for index in range(tracks))
    inspection = FaceInspection(inspection_id=inspection_id, target_sha256=digest, detector_model_id="fixture",
                                detector_sha256="0" * 64, tracks=track_models, decoded_frames=frames,
                                sampled_frames=frames, truncated=False, tracking_method="iou_arcface_hungarian")
    atomic_json(root, f"roop_3d/.runtime/inspections/{inspection_id}.json", inspection.model_dump(mode="json"))
    for faceset_id in ids:
        atomic_json(root, f"roop_3d/.runtime/facesets/{faceset_id}.json",
                    {"faceset_id": faceset_id, "name": faceset_id, "item_count": 1, "schema_version": 1})
        archive = root / f"roop_3d/outputs/facesets/{faceset_id}.fsz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_bytes(b"read-only fixture archive")
    factory = session_factory or (lambda catalog, config: StubSession(config, tracks=tracks, delay=delay, on_frame=on_frame))
    from roop_3d.core.hardware import Hardware
    fixture_probe = hardware_probe or (lambda current: Hardware(timestamp="STUB_HARDWARE", os={}, logical_cpu_count=1))
    runner_kwargs = {"checkpoint_every": checkpoint_every} if checkpoint_every else {}
    runner_kwargs["hardware_probe"] = fixture_probe
    runner = MasterPipelineRunner(root, app.state.models, app.state.media, lambda: app.state.settings,
                                  faceset_reader=StubReader(), pipeline_factory=factory, **runner_kwargs)
    scheduler = RuntimeScheduler(VRAMGovernor(hardware_probe)) if hardware_probe else RuntimeScheduler(VRAMGovernor())
    worker = JobWorker(root, app.state.jobs, lambda: app.state.settings, app.state.models, app.state.media,
                       pipeline_runner=runner, scheduler=scheduler, poll_interval=poll_interval)
    app.state.worker = worker
    payload = {"target_path": target_relative, "inspection_id": inspection_id,
               "face_mappings": [{"track_id": f"track_{i:04d}", "faceset_id": ids[i]} for i in range(tracks)]}
    if start:
        worker.start()
    return Rig(root, app, worker, payload, target, frames)


def decoded_digest(path: Path) -> str:
    """SHA-256 over every decoded frame: compares two encodes by *content*, independent of muxer timestamps."""
    from roop_3d.media.video_reader import VideoReader
    digest = hashlib.sha256()
    count = 0
    for frame in VideoReader(path, hardware="software").frames():
        digest.update(frame.image.tobytes())
        count += 1
    return f"{count}:{digest.hexdigest()}"


def dump(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, default=str)
