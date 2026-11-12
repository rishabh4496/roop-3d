"""The runner configures the angle diagnostics from the job, publishes the JSONL atomically and attests it in the manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core.settings import Settings
from roop_3d.engine.pipeline_runner import MasterPipelineRunner
from roop_3d.engine.volumetric import angle_diagnostics as ad
from roop_3d.tests.test_stage12_worker import _Session, _fixture

FIVE = np.array(((10, 10), (20, 10), (15, 15), (12, 20), (18, 20)), np.float64)


class DiagSession(_Session):
    """Contract stub that behaves like the dense path towards the recorder: one swap record and one composite per frame."""
    fail_after: int | None = None

    def __init__(self, config):
        super().__init__(config)
        self.angle_diagnostics = ad.AngleDiagnostics()
        self.swapped = self.composited = 0

    def infer_frame(self, image, **options):
        if self.fail_after is not None and self.swapped >= self.fail_after:
            raise RuntimeError("fixture failure mid-render")
        coverage = np.zeros((32, 32), np.float32)
        coverage[8:24, 8:24] = 1
        self.angle_diagnostics.record_swap(
            self.swapped, 0, {"crop_matrix": np.array(((1.0, 0, 0), (0, 1.0, 0))), "landmarks5": FIVE + (self.swapped, 0),
                              "coverage": coverage, "coverage_scale": 1.0,
                              "views": [{"index": 0, "yaw": 0, "pitch": 0, "raw": .1, "weight": 1.0}]}, pose=(25.0, 0.0, 0.0))
        self.swapped += 1
        return super().infer_frame(image)

    def composite_frame(self, image, result, **kwargs):
        if self.angle_diagnostics.full:
            self.angle_diagnostics.record_composite(self.composited, 0, {"ring": {"ring_px": 5, "dark_px": 0, "in_luma_step": 1.0},
                                                                         "sharpness_ratio": 0.9})
        self.composited += 1
        return super().composite_frame(image, result, **kwargs)


class InvocationRestorer:
    def __init__(self, *, called=False, digest=None):
        self.inference_verified = called
        self.runner = SimpleNamespace(model=SimpleNamespace(record={"sha256": digest}) if digest else None)

    def restore_hires(self, image, landmarks5=None):
        self.inference_verified = True
        return image

    def capability(self):
        return {"architecture": "ultramax",
                "inference": "VERIFIED_THIS_SESSION" if self.inference_verified else "NOT_RUN",
                "weights": "VERIFIED_LOCAL" if self.runner.model is not None else "NOT_LOADED"}


class UltraMaxInvocationSession(DiagSession):
    def __init__(self, config):
        super().__init__(config)
        self.neural_face = SimpleNamespace(enhancer=InvocationRestorer(digest="a" * 64))

    def infer_frame(self, image, **options):
        self.neural_face.enhancer.restore_hires(image)
        return super().infer_frame(image, **options)

    def capability(self):
        capability = super().capability()
        capability["dense_face"] = {"enhancer": self.neural_face.enhancer.capability()}
        compositor = getattr(self.compositor, "restorer", None)
        capability["full_frame_compositor"] = {
            "restorer": compositor.capability() if compositor is not None else {
                "architecture": "ultramax", "inference": "NOT_RUN", "weights": "NOT_LOADED"}}
        return capability


class IdleUltraMax:
    def __init__(self, *args):
        self.inference_verified = False
        self.runner = SimpleNamespace(model=None)

    def capability(self):
        return {"architecture": "ultramax", "inference": "NOT_RUN", "weights": "NOT_LOADED"}


class FailingSession(DiagSession):
    fail_after = 4


def run_job(tmp_path, *, angle_diagnostics=None, settings=None, factory=DiagSession, expect="succeeded",
            writer=None, workstation_parameters=None):
    app, worker, payload = _fixture(tmp_path, settings=settings, session_factory=lambda catalog, config: factory(config))
    if angle_diagnostics is not None:
        payload = {**payload, "angle_diagnostics": angle_diagnostics}
    if workstation_parameters is not None:
        payload = {**payload, "workstation_parameters": workstation_parameters}
    if writer is not None:
        worker.runner.writer = writer
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "angle-diag"})
            assert created.status_code == 201, created.text
            job_id = created.json()["job_id"]
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                record = client.get(f"/api/v1/jobs/{job_id}").json()
                if record["state"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(0.02)
            assert record["state"] == expect, record
    finally:
        worker.stop()
    return job_id, record, tmp_path / "roop_3d/.runtime/outputs"



def angle_files(outputs: Path) -> list[str]:
    return sorted(p.name for p in outputs.glob("*angle*")) if outputs.exists() else []


def stage12(tmp_path, record) -> dict:
    return json.loads((tmp_path / record["output"]["manifest_path"]).read_text("utf-8"))["stage12"]

def test_manifest_reports_the_dense_ultramax_invocation_not_the_idle_compositor_adapter(tmp_path, monkeypatch):
    from roop_3d.compositor import restoration
    monkeypatch.setattr(restoration, "UltraMaxPipelineRestorer", IdleUltraMax)
    _, record, _ = run_job(
        tmp_path,
        factory=UltraMaxInvocationSession,
        workstation_parameters={"restorer_model": "ultramax", "restoration_blend": 0.7},
    )
    stage = stage12(tmp_path, record)
    actual = stage["restoration"]
    assert actual["architecture"] == "ultramax"
    assert actual["inference"] == "VERIFIED_THIS_SESSION" and actual["weights"] == "VERIFIED_LOCAL"
    assert actual["route"] == "dense_face.enhancer" and actual["invoked"] is True
    assert actual["routes"]["full_frame_compositor.restorer"] == {
        "architecture": "ultramax", "inference": "NOT_RUN", "weights": "NOT_LOADED"}
    assert stage["model_hashes"]["restorer"] == "a" * 64


def test_default_job_is_summary_level_writes_no_jsonl_and_says_so_in_the_manifest(tmp_path):
    job_id, record, outputs = run_job(tmp_path)
    section = stage12(tmp_path, record)["angle_diagnostics"]
    assert section["level"] == "summary" and section["level_source"] == "job" and section["frames"] == 8
    assert "jsonl" not in section and "mask_edge" not in section
    assert section["landmarks5_second_difference_over_iod"]["n"] == 6
    assert angle_files(outputs) == []


def test_full_job_field_writes_the_documented_rows_and_attests_path_hash_and_rows(tmp_path):
    job_id, record, outputs = run_job(tmp_path, angle_diagnostics="full")
    section = stage12(tmp_path, record)["angle_diagnostics"]
    name = f"roop3d_{job_id}.angle.jsonl"
    assert section["level"] == "full" and section["level_source"] == "job" and section["frames"] == 8
    assert section["jsonl"]["path"] == name                       # relative to the output root
    path = outputs / name
    assert angle_files(outputs) == [name]                         # no .partial left behind
    assert section["jsonl"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines()]
    assert section["jsonl"]["rows"] == len(rows) == 8
    assert [row["frame"] for row in rows] == list(range(8))
    assert set(rows[-1]) >= {"frame", "track", "pose", "crop", "coverage", "jitter", "views", "ring", "sharpness"}
    assert set(rows[-1]["crop"]) == {"s", "theta_deg", "tx", "ty"} and rows[-1]["sharpness"]["yaw_bucket"] == 20
    assert section["mask_edge"]["frames_measured"] == 8 and section["sharpness_ratio_by_abs_yaw"]["20-30"]["n"] == 8


def test_job_field_is_the_only_level_selector_even_when_application_debug_is_enabled(tmp_path):
    settings = Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider", audio_policy="drop",
                        diagnostics_level="debug")
    _, record, outputs = run_job(tmp_path, settings=settings)
    section = stage12(tmp_path, record)["angle_diagnostics"]
    assert (section["level"], section["level_source"]) == ("summary", "job")
    assert "jsonl" not in section and angle_files(outputs) == []


def test_a_failed_job_leaves_no_diagnostics_artifact(tmp_path):
    job_id, record, outputs = run_job(tmp_path, angle_diagnostics="full", factory=FailingSession, expect="failed")
    assert angle_files(outputs) == []


def test_a_failure_after_full_diagnostics_publish_removes_the_artifact(tmp_path):
    def fail_writer(*args, **kwargs):
        raise RuntimeError("fixture mux failure")

    _, record, outputs = run_job(tmp_path, angle_diagnostics="full", expect="failed", writer=fail_writer)
    assert angle_files(outputs) == []


def test_discarding_a_published_output_removes_its_diagnostics_jsonl(tmp_path):
    from roop_3d.jobs.schema import JobOutput
    runner = object.__new__(MasterPipelineRunner)
    runner.root = tmp_path
    outputs = tmp_path / "roop_3d/.runtime/outputs"
    outputs.mkdir(parents=True)
    job = "c" * 32
    for name in (f"roop3d_{job}.mp4", f"roop3d_{job}.mp4.manifest.json", f"roop3d_{job}.angle.jsonl", "roop3d_other.angle.jsonl"):
        (outputs / name).write_bytes(b"x")
    runner.discard_output(JobOutput(output_path=f"roop_3d/.runtime/outputs/roop3d_{job}.mp4",
                                    manifest_path=f"roop_3d/.runtime/outputs/roop3d_{job}.mp4.manifest.json",
                                    output_sha256="0" * 64, frames=1, audio_policy="drop"))
    assert sorted(p.name for p in outputs.iterdir()) == ["roop3d_other.angle.jsonl"]


def test_job_field_defaults_and_participates_in_request_identity():
    from roop_3d.jobs.schema import JobRequest
    payload = {"target_path": "roop_3d/.runtime/media/target.mp4", "inspection_id": "a" * 32,
               "face_mappings": [{"track_id": "track_0000", "faceset_id": "alice"}]}
    default = JobRequest.model_validate(payload)
    full = JobRequest.model_validate({**payload, "angle_diagnostics": "full"})
    assert default.angle_diagnostics == "summary"
    assert default.model_dump()["angle_diagnostics"] == "summary"
    assert "angle_diagnostics" not in default.identity_payload()
    assert full.identity_payload()["angle_diagnostics"] == "full"


def test_invalid_value_is_rejected_with_422_and_full_is_accepted(tmp_path):
    app, worker, payload = _fixture(tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.post("/api/v1/jobs", json={**payload, "angle_diagnostics": "verbose"},
                               headers={"Idempotency-Key": "angle-invalid"})
        assert response.status_code == 422
        nested = client.post("/api/v1/jobs", json={**payload, "workstation_parameters": {"angle_diagnostics": "full"}},
                             headers={"Idempotency-Key": "angle-nested"})
        assert nested.status_code == 422
        ok = client.post("/api/v1/jobs", json={**payload, "angle_diagnostics": "full"},
                         headers={"Idempotency-Key": "angle-valid"})
        assert ok.status_code == 201
