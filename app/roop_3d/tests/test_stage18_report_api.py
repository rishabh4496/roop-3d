"""Stage 18: the diagnostic report - contents, redaction, local-only behaviour and the audit/repair endpoints."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import time

from fastapi.testclient import TestClient
import pytest

from roop_3d.jobs.schema import JobState
from roop_3d.reliability.harness import build_rig, make_video
from roop_3d.reliability.report import REPORT_SCHEMA

SECRET_VIDEO = "very_secret_family_holiday.mp4"
SECRET_PERSON = "sentinel-person-jane"


@pytest.fixture
def rig(tmp_path):
    video = make_video(tmp_path / "roop_3d/.runtime/media" / SECRET_VIDEO, 30)
    rig = build_rig(tmp_path, frames=30, tracks=2, video=video, faceset_ids=[SECRET_PERSON, "sentinel-person-bob"])
    yield rig
    rig.worker.stop()


def _client(rig):
    # The worker releases a job's work directory just after acknowledging success; the lifespan's startup
    # reconciliation assumes nothing is running, so let the worker finish first (as a real restart would).
    deadline = time.monotonic() + 15
    while (rig.worker._futures or any(p.is_dir() for p in (rig.root / "roop_3d/.runtime/jobs").glob("*")))             and time.monotonic() < deadline:
        time.sleep(0.02)
    return TestClient(rig.app, base_url="http://127.0.0.1:8030")


def test_report_has_every_required_section_with_real_values(rig):
    done = rig.wait(rig.submit("r1").job_id)
    assert done.state == JobState.succeeded
    with _client(rig) as client:
        report = client.get("/api/v1/reliability/report").json()
    assert report["schema"] == REPORT_SCHEMA
    assert report["privacy"] == {"local_only": True, "transmitted": False, "contains_media": False, "contains_biometrics": False}
    assert report["application"]["version"] == "0.1.0"
    assert report["hardware"]["cpu"]["logical"] and report["hardware"]["ram_mib"]["total"]
    assert "libraries" in report["drivers"] and report["drivers"]["libraries"]["onnxruntime"]
    assert report["models"]["count"] >= 0 and "render_dependencies" in report["models"]
    assert report["execution_providers"]["configured"]["preferred"] and "advertised" in report["execution_providers"]
    assert {"decode", "encode"} <= set(report["pipeline_stage_timings_ms"]), report["pipeline_stage_timings_ms"]
    assert report["memory"]["process"]["rss_mib"] > 10 and report["memory"]["peaks"]["rss_mib"] > 10
    assert report["memory"]["job_peaks"], "peak RAM must be recorded per job"
    assert "gpu" in report["memory"] and "growth" in report["memory"]
    assert report["jobs"]["by_state"]["succeeded"] == 1 and report["jobs"]["recent"][0]["frame"] == 30
    assert report["disk"]["volume_free_mib"] > 0 and "job_workdirs_mib" in report["disk"]
    assert "errors_and_warnings" in report and "resources" in report and report["resources"]["ledger"]
    assert report["redaction"]["applied"]


def test_report_contains_no_media_names_people_paths_or_biometrics(rig):
    rig.wait(rig.submit("r2").job_id)
    with _client(rig) as client:
        text = client.get("/api/v1/reliability/report", params={"series": True}).text
    home = str(Path.home())
    for forbidden in (SECRET_VIDEO, "very_secret", SECRET_PERSON, "sentinel-person", home, home.replace("\\", "\\\\"),
                      str(rig.root), str(rig.root).replace("\\", "\\\\"), '"embedding"', '"thumbnail_base64"', '"landmarks"'):
        assert forbidden not in text, f"report leaked {forbidden!r}"
    document = json.loads(text)
    ids = [item["job"] for item in document["jobs"]["recent"]]
    assert all(len(job) == 8 for job in ids)  # truncated random ids only


def test_failures_are_listed_by_code_without_the_exception_text(tmp_path):
    def boom(index):
        if index == 3:
            raise KeyError(rf"C:\Users\someone\{SECRET_VIDEO}")
    rig = build_rig(tmp_path, frames=20, tracks=1, on_frame=boom)
    try:
        with _client(rig) as client:  # the lifespan attaches the warning buffer, as it does for the real server
            failed = rig.wait(rig.submit("f").job_id)
            assert failed.state == JobState.failed
            text = client.get("/api/v1/reliability/report").text
        document = json.loads(text)
        assert document["jobs"]["failure_codes"] == {"WORKER_FAILED": 1}
        assert SECRET_VIDEO not in text and "someone" not in text
        assert any("unclassified runtime error (KeyError)" in w["message"] for w in document["errors_and_warnings"]["recent"])
    finally:
        rig.worker.stop()


def test_building_the_report_makes_no_network_connection(rig, monkeypatch):
    attempts: list[object] = []
    real_connect, real_create = socket.socket.connect, socket.create_connection

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):  # the event loop's own loopback socketpair is not egress
            attempts.append(address)
            raise AssertionError("diagnostics attempted a network connection")
        return real_connect(self, address)

    def guarded_create(address, *args, **kwargs):
        attempts.append(address)
        raise AssertionError("diagnostics attempted a network connection")

    with _client(rig) as client:
        monkeypatch.setattr(socket.socket, "connect", guarded_connect)
        monkeypatch.setattr(socket, "create_connection", guarded_create)
        try:
            for url in ("/api/v1/reliability/report", "/api/v1/reliability/audit", "/api/v1/reliability/resources"):
                assert client.get(url).status_code == 200
        finally:
            monkeypatch.undo()
    assert attempts == []


def test_saved_reports_stay_inside_the_runtime_and_are_pruned(rig):
    with _client(rig) as client:
        paths = []
        for _ in range(23):
            paths.append(client.get("/api/v1/reliability/report", params={"save": True}).json()["saved_to"])
            time.sleep(1.01)  # filenames carry second resolution
    files = sorted((rig.root / "roop_3d/.runtime/diagnostics").glob("diagnostics-*.json"))
    assert len(files) == 20
    assert all(p.startswith("roop_3d/.runtime/diagnostics/") for p in paths)
    assert json.loads(files[-1].read_text("utf-8"))["privacy"]["transmitted"] is False


def test_startup_reconciliation_runs_in_the_lifespan_and_is_reported(rig):
    rig.worker.stop()
    orphan = rig.root / "roop_3d/outputs" / ("f" * 32 + ".video.mp4")
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"left by a crash")
    part = rig.root / "roop_3d/.runtime/jobs/state.json.0123456789abcdef0123456789abcdef.part"
    part.write_bytes(b"{")
    with _client(rig) as client:  # entering the context runs the application lifespan
        report = client.get("/api/v1/reliability/report").json()
    assert report["startup_reconciliation"]["repaired"] >= 2 and report["startup_reconciliation"]["ok"]
    assert not orphan.exists() and not part.exists()


def test_audit_and_repair_endpoints(rig):
    rig.worker.stop()
    with _client(rig) as client:
        stray = rig.root / "roop_3d/.runtime/jobs" / ("9" * 32) / "frames"   # appears after startup, e.g. a crashed run
        stray.mkdir(parents=True)
        (stray / "00000000.png").write_bytes(b"x")
        old = time.time() - 7200
        import os
        os.utime(stray.parent, (old, old))
        found = client.get("/api/v1/reliability/audit").json()
        assert any(f["code"] == "ORPHAN_JOB_WORKDIR" for f in found["findings"])
        repaired = client.post("/api/v1/reliability/repair", json={}).json()
        assert repaired["repaired"] >= 1 and repaired["ok"]
        assert client.get("/api/v1/reliability/audit").json()["findings"] == []
        resources = client.get("/api/v1/reliability/resources").json()
        assert "growth" in resources and "ledger" in resources
        assert client.post("/api/v1/reliability/repair", json={"unknown": 1}).status_code == 422
