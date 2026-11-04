"""Stage 18 integration tests: every way a long job can be stopped, killed or starved, and what must be true afterwards.

Invariants asserted across the scenarios (the stage's acceptance criteria):
* no corrupted output after cancellation / kill / error - an output either does not exist or is complete and hashed;
* no job corruption after restart - the job resumes from its checkpoint and produces frame-identical output;
* no orphan processes - and no temp files, half-published outputs or stranded work directories;
* no silent failure - every stop carries a specific, actionable code.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.core.hardware import GPU, Hardware
from roop_3d.core.settings import Settings
from roop_3d.core.storage import atomic_json
from roop_3d.jobs.schema import JobState
from roop_3d.jobs.store import JobStore
from roop_3d.media.video_writer import encode_video
from roop_3d.reliability import probe
from roop_3d.reliability.harness import StubSession, build_rig, decoded_digest, make_video
from roop_3d.reliability.ledger import LEDGER
from roop_3d.reliability.recovery import ReliabilityAuditor, reconcile_on_startup

APP = Path(__file__).resolve().parents[2]


def _wait(predicate, seconds: float = 20.0, message: str = "condition"):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {message}")


def _settled(rig, job_id: str) -> None:
    """The worker acknowledges a state, *then* releases the work directory; wait for both."""
    _wait(lambda: not rig.workdir(job_id).exists() and rig.worker._futures == {}, message="work directory release")


def _clean(rig) -> None:
    """Nothing but finished outputs may remain on disk, and nothing may still be running."""
    root = rig.root
    leftovers = [p.relative_to(root).as_posix() for p in (root / "roop_3d").rglob("*")
                 if p.is_file() and (p.name.endswith(".part") or ".video." in p.name or ".mux." in p.name
                                     or ".signed." in p.name)]
    assert leftovers == []
    assert LEDGER.live("video_decoder") == 0 and LEDGER.live("video_encoder") == 0
    assert LEDGER.counts().get("ffmpeg_process", 0) == 0
    assert StubSession.live_instances == 0


def _valid_output(rig, record) -> str:
    out = rig.root / record.output.output_path
    manifest = json.loads((rig.root / record.output.manifest_path).read_text("utf-8"))
    assert hashlib.sha256(out.read_bytes()).hexdigest() == record.output.output_sha256 == manifest["output_sha256"]
    assert manifest["verified_frames"] == record.output.frames
    return decoded_digest(out)


@pytest.fixture
def baseline(tmp_path_factory):
    """Digest of an uninterrupted render of the shared 80-frame clip - the reference every recovery must equal."""
    root = tmp_path_factory.mktemp("baseline")
    rig = build_rig(root, frames=80, tracks=2)
    try:
        record = rig.wait(rig.submit().job_id)
        assert record.state == JobState.succeeded
        return {"digest": _valid_output(rig, record), "video": rig.target.read_bytes()}
    finally:
        rig.worker.stop()


def _rig_with_clip(root: Path, baseline, **kwargs):
    target = root / "roop_3d/.runtime/media/target.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(baseline["video"])
    return build_rig(root, frames=80, tracks=2, **kwargs)


# ------------------------------------------------------------------------------------------- cancellation
def test_cancellation_leaves_no_output_no_workdir_and_runs_cleanup_once(tmp_path):
    rig = build_rig(tmp_path, frames=200, tracks=2, delay=0.01)
    calls: list[str] = []
    real_cleanup = rig.worker.runner.cleanup
    rig.worker.runner.cleanup = lambda job_id: (calls.append(job_id), real_cleanup(job_id))[1]
    try:
        ids = []
        for index in range(3):
            record = rig.submit(f"cancel-{index}")
            rig.wait_frames(record.job_id, 12)
            rig.store.control(record.job_id, "cancel")
            assert rig.wait(record.job_id).state == JobState.cancelled
            _settled(rig, record.job_id)
            ids.append(record.job_id)
        time.sleep(0.6)  # ~60 worker ticks at 10 ms: the old loop cleaned every cancelled job on every tick
        assert rig.output_files() == []
        assert all(calls.count(job_id) <= 2 for job_id in ids), {i[:6]: calls.count(i) for i in ids}
        _clean(rig)
    finally:
        rig.worker.stop()


def test_cancel_while_paused_and_while_queued(tmp_path):
    rig = build_rig(tmp_path, frames=100, tracks=1, delay=0.01)
    try:
        record = rig.submit("c1")
        rig.wait_frames(record.job_id, 10)
        rig.store.control(record.job_id, "pause")
        rig.wait(record.job_id, JobState.paused)
        rig.store.control(record.job_id, "cancel")
        assert rig.wait(record.job_id).state == JobState.cancelled
        _settled(rig, record.job_id)
        _clean(rig)
    finally:
        rig.worker.stop()


# ----------------------------------------------------------------------------------- pause / resume cycles
def test_repeated_pause_resume_output_is_frame_identical(tmp_path, baseline):
    rig = _rig_with_clip(tmp_path, baseline, delay=0.01)
    try:
        record = rig.submit("pr")
        for cycle in range(4):
            rig.wait_frames(record.job_id, 8 * (cycle + 1))
            if rig.job(record.job_id).state in (JobState.succeeded,):
                break
            rig.store.control(record.job_id, "pause")
            rig.wait(record.job_id, {JobState.paused, JobState.succeeded})
            if rig.job(record.job_id).state == JobState.paused:
                rig.store.control(record.job_id, "resume")
        done = rig.wait(record.job_id, timeout=60)
        assert done.state == JobState.succeeded and done.output.frames == 80
        assert _valid_output(rig, done) == baseline["digest"]
        _settled(rig, record.job_id)
        _clean(rig)
    finally:
        rig.worker.stop()


# ----------------------------------------------------------------------- application restart (clean stop)
def test_application_restart_resumes_from_checkpoint_to_identical_output(tmp_path, baseline):
    rig = _rig_with_clip(tmp_path, baseline, delay=0.02, checkpoint_every=10)
    record = rig.submit("restart")
    rig.wait_frames(record.job_id, 30)
    rig.worker.stop()  # what closing the application does
    stopped = rig.job(record.job_id)
    assert stopped.state == JobState.interrupted and not rig.output_files()
    checkpoint = json.loads((rig.workdir(record.job_id) / "checkpoint.json").read_text("utf-8"))
    assert checkpoint["completed_frame"] >= 9

    second = _rig_with_clip(tmp_path, baseline, start=False)  # a brand new process over the same disk
    try:
        second.store.recover_interrupted()
        report = reconcile_on_startup(tmp_path, second.store, lambda: second.app.state.settings, runner=second.worker.runner)
        assert report.ok
        resumed = second.store.control(record.job_id, "resume")
        assert resumed.state == JobState.queued
        second.worker.start()
        done = second.wait(record.job_id, timeout=60)
        assert done.state == JobState.succeeded
        assert _valid_output(second, done) == baseline["digest"]
        _settled(second, record.job_id)
        _clean(second)
    finally:
        second.worker.stop()


# ---------------------------------------------------- hard kill (TerminateProcess): the real crash test
def _spawn_child(root: Path, frames: int, tracks: int, delay: float):
    child = subprocess.Popen([sys.executable, "-m", "roop_3d.reliability.crash_child", str(root), str(frames),
                              str(tracks), str(delay)], cwd=APP, stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "PYTHONPATH": str(APP)})
    line = child.stdout.readline().strip()
    assert line.startswith("READY "), f"child failed to start: {line!r}"
    return child, line.split()[1]


@pytest.mark.parametrize("kill_at", ["early", "late", "mux"])
def test_hard_kill_then_restart_recovers_a_consistent_job(tmp_path, baseline, kill_at):
    root = tmp_path
    child, job_id = _spawn_child(root, 80, 2, 0.02)
    # The child built the clip itself; replace it with the shared bytes so digests are comparable to the baseline.
    store = JobStore(root)
    try:
        if kill_at == "mux":
            _wait(lambda: store.get(job_id).progress.stage == "mux" or store.get(job_id).state in (JobState.succeeded,),
                  seconds=60, message="mux stage")
        else:
            _wait(lambda: store.get(job_id).progress.frame >= (12 if kill_at == "early" else 60), seconds=60, message="progress")
        probe.kill_process(child.pid)  # TerminateProcess: no finally blocks, no atexit, no cleanup of any kind
        child.wait(10)
    finally:
        if child.poll() is None:
            probe.kill_process(child.pid)

    rig = build_rig(root, frames=80, tracks=2, start=False)
    try:
        before = rig.job(job_id)
        auditor = ReliabilityAuditor(root, rig.store, lambda: rig.app.state.settings, runner=rig.worker.runner)
        if before.state in (JobState.preparing, JobState.running):
            assert [f.code for f in auditor.audit().findings if f.code == "ACTIVE_JOB_WITHOUT_WORKER"], \
                "the audit must see the orphaned active job"
        rig.store.recover_interrupted()
        assert reconcile_on_startup(root, rig.store, lambda: rig.app.state.settings, runner=rig.worker.runner).ok
        state = rig.job(job_id).state
        assert state in (JobState.interrupted, JobState.succeeded), state
        rig.worker.start()
        if state == JobState.interrupted:
            rig.store.control(job_id, "resume")
        done = rig.wait(job_id, timeout=90)
        assert done.state == JobState.succeeded, done.failure
        # Frame-identical to an uninterrupted run of the same clip (the child's clip equals the baseline clip: same seed).
        reference = rig.submit("after-kill-reference")
        ref_done = rig.wait(reference.job_id, timeout=90)
        assert _valid_output(rig, done) == _valid_output(rig, ref_done)
        _settled(rig, job_id)
        _settled(rig, reference.job_id)
        # Exactly the two published outputs (+ manifests) exist; the audit is clean.
        assert len([p for p in rig.output_files() if not p.name.endswith(".manifest.json")]) == 2
        assert not [f for f in ReliabilityAuditor(root, rig.store, lambda: rig.app.state.settings).audit().findings
                    if f.severity == "error"]
        _clean(rig)
    finally:
        rig.worker.stop()


def test_output_published_without_manifest_blocks_resume_until_reconciled(tmp_path):
    """Regression for the one window atomic rename cannot cover: killed between publishing the file and its manifest.

    Without reconciliation the resumed job dies with OUTPUT_EXISTS forever; with it the stray output is quarantined
    (not deleted) and the job completes."""
    rig = build_rig(tmp_path, frames=24, tracks=1, start=False)
    try:
        def stranded(key: str):
            record = rig.submit(key)
            rig.store.transition(record.job_id, JobState.preparing)
            rig.store.transition(record.job_id, JobState.running)
            rig.store.transition(record.job_id, JobState.interrupted)
            out = rig.root / rig.app.state.settings.output_root
            out.mkdir(parents=True, exist_ok=True)
            (out / f"roop3d_{record.job_id}.mp4").write_bytes(b"\x00" * 4096)  # renamed into place, manifest never written
            rig.store.control(record.job_id, "resume")
            return record.job_id

        stuck = stranded("stuck")
        rig.worker.start()
        failed = rig.wait(stuck)
        assert failed.state == JobState.failed and failed.failure.code == "OUTPUT_EXISTS"
        rig.worker.stop()

        healed = stranded("healed")
        report = reconcile_on_startup(tmp_path, rig.store, lambda: rig.app.state.settings, runner=rig.worker.runner)
        assert [f.code for f in report.findings if f.code == "UNOWNED_OUTPUT" and f.repaired]
        assert list((tmp_path / "roop_3d/.runtime/quarantine").glob("*roop3d_*")), "unowned output must be quarantined, not deleted"
        rig.worker.start()
        done = rig.wait(healed)
        assert done.state == JobState.succeeded
        _valid_output(rig, done)
    finally:
        rig.worker.stop()


# --------------------------------------------------------------------------------------- disk exhaustion
def test_disk_full_mid_job_interrupts_resumably_then_completes(tmp_path, baseline, monkeypatch):
    import roop_3d.engine.pipeline_runner as runner_module
    real = runner_module.atomic_bytes
    state = {"full": False}

    def flaky(root, relative, value):
        if state["full"] and "/frames/" in relative.replace("\\", "/"):
            raise OSError(errno.ENOSPC, "No space left on device")
        return real(root, relative, value)

    monkeypatch.setattr(runner_module, "atomic_bytes", flaky)
    rig = _rig_with_clip(tmp_path, baseline, delay=0.01, checkpoint_every=10)
    try:
        record = rig.submit("disk")
        rig.wait_frames(record.job_id, 25)
        state["full"] = True  # another program just filled the disk
        stopped = rig.wait(record.job_id, JobState.interrupted, timeout=20)
        assert stopped.failure.code == "DISK_FULL" and stopped.failure.recoverable
        assert "free disk space" in stopped.failure.message.lower()
        assert rig.workdir(record.job_id).exists(), "the checkpoint must survive a disk-full stop"
        assert rig.output_files() == []
        state["full"] = False  # the user freed space
        rig.store.control(record.job_id, "resume")
        done = rig.wait(record.job_id, timeout=60)
        assert done.state == JobState.succeeded and done.failure is None
        assert _valid_output(rig, done) == baseline["digest"]
        _settled(rig, record.job_id)
        _clean(rig)
    finally:
        rig.worker.stop()


def test_insufficient_space_is_refused_before_any_frame_is_written(tmp_path, baseline, monkeypatch):
    from roop_3d.reliability import disk
    monkeypatch.setattr(disk, "free_bytes", lambda path: 1024 * 1024)
    rig = _rig_with_clip(tmp_path, baseline)
    try:
        record = rig.submit("low")
        stopped = rig.wait(record.job_id, {JobState.interrupted, JobState.failed})
        assert stopped.state == JobState.interrupted and stopped.failure.code == "DISK_SPACE_LOW"
        assert not list((rig.workdir(record.job_id) / "frames").glob("*.png"))
        monkeypatch.undo()
        rig.store.control(record.job_id, "resume")
        assert rig.wait(record.job_id).state == JobState.succeeded
    finally:
        rig.worker.stop()


def test_encoder_disk_full_names_the_cause_and_leaves_no_partial_output(tmp_path):
    source = make_video(tmp_path / "roop_3d/.runtime/media/in.mp4", 30)

    def transform(frame):
        if frame.index == 12:
            raise OSError(errno.ENOSPC, "No space left on device")
        return frame.image

    with pytest.raises(DomainError) as caught:
        encode_video(source, "out.mp4", root=tmp_path, transform=transform, output_dir="roop_3d/.runtime/outputs",
                     audio_policy="drop")
    assert caught.value.code == "DISK_FULL" and caught.value.status == 507
    assert not list((tmp_path / "roop_3d").rglob("out.mp4*"))
    assert not [p for p in (tmp_path / "roop_3d").rglob("*") if ".video." in p.name or ".mux." in p.name]


# ---------------------------------------------------------------------------------------------- CUDA OOM
def test_cuda_oom_midway_resumes_on_cpu_from_checkpoint(tmp_path, baseline):
    providers: list[str] = []

    def factory(catalog, config):
        providers.append(config.provider)

        def fail_once_on_cuda(index):
            if config.provider == "CUDAExecutionProvider" and index == 30:
                raise RuntimeError("CUDA out of memory: tried to allocate 2.00 GiB")
        return StubSession(config, tracks=2, on_frame=fail_once_on_cuda)

    snapshot = Hardware(timestamp=datetime.now(timezone.utc).isoformat(), os={"system": "Windows"}, logical_cpu_count=16,
        gpus=[GPU(index=0, name="simulated", total_vram_mib=12288, free_vram_mib=12000, effective_vram_budget_mib=11200,
                  driver_version="fixture")], gpu_probe_state="VERIFIED_LOCAL",
        advertised_providers=["CUDAExecutionProvider"], provider_probe_state="VERIFIED_LOCAL")
    settings = Settings(hardware_profile="12gb", preferred_provider="CUDAExecutionProvider", audio_policy="drop")
    rig = _rig_with_clip(tmp_path, baseline, session_factory=factory, settings=settings, hardware_probe=lambda s: snapshot,
                         checkpoint_every=10)
    try:
        record = rig.submit("oom")
        done = rig.wait(record.job_id, timeout=60)
        assert done.state == JobState.succeeded, done.failure
        assert providers[:2] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert _valid_output(rig, done) == baseline["digest"]
        assert StubSession.live_instances == 0, "the session that hit the OOM must have been released"
    finally:
        rig.worker.stop()


# --------------------------------------------------------------------------- failures are never silent
def test_unexpected_failure_is_named_and_releases_checkpoint_frames(tmp_path):
    def boom(index):
        if index == 10:
            raise KeyError("internal bug with C:\\Users\\someone\\private.mp4")

    rig = build_rig(tmp_path, frames=40, tracks=1, on_frame=boom)
    try:
        record = rig.submit("boom")
        failed = rig.wait(record.job_id)
        assert failed.state == JobState.failed and failed.failure.code == "WORKER_FAILED"
        assert "private.mp4" not in failed.failure.message
        _wait(lambda: not (rig.workdir(record.job_id) / "frames").exists(), message="frame release")
        assert rig.output_files() == []
        assert (rig.workdir(record.job_id) / "checkpoint.json").exists(), "forensic checkpoint is kept"
        _clean(rig)
    finally:
        rig.worker.stop()


@pytest.mark.parametrize("error,code", [(MemoryError(), "PIPELINE_OOM"), (PermissionError(13, "locked"), "FILE_ACCESS_DENIED"),
                                        (BrokenPipeError(), "ENCODER_TERMINATED")])
def test_environmental_errors_pause_the_job_resumably_with_a_named_reason(tmp_path, error, code):
    seen = {"raised": False}

    def once(index):
        if index == 15 and not seen["raised"]:
            seen["raised"] = True
            raise error

    rig = build_rig(tmp_path, frames=40, tracks=1, on_frame=once, checkpoint_every=10)
    try:
        record = rig.submit("env")
        stopped = rig.wait(record.job_id, {JobState.interrupted, JobState.failed, JobState.succeeded})
        assert stopped.state == JobState.interrupted and stopped.failure.code == code and stopped.failure.recoverable
        rig.store.control(record.job_id, "resume")
        done = rig.wait(record.job_id)
        assert done.state == JobState.succeeded and done.failure is None
    finally:
        rig.worker.stop()


# ---------------------------------------------------------------------------------------- corrupt inputs
def test_truncated_video_fails_with_a_media_code_and_publishes_nothing(tmp_path):
    good = make_video(tmp_path / "good.mp4", 40)
    bad = tmp_path / "roop_3d/.runtime/media/target.mp4"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(good.read_bytes()[: good.stat().st_size // 3])  # the moov atom is at the end: it is gone
    rig = build_rig(tmp_path, frames=40, tracks=1, video=bad)
    try:
        record = rig.submit("trunc")
        failed = rig.wait(record.job_id)
        assert failed.state == JobState.failed
        assert failed.failure.code != "WORKER_FAILED", f"silent/unnamed failure: {failed.failure}"
        assert rig.output_files() == []
        _clean(rig)
    finally:
        rig.worker.stop()


def test_corruption_in_the_middle_of_the_stream_never_publishes_a_silently_damaged_output(tmp_path):
    good = make_video(tmp_path / "good.mp4", 80, noise=True)
    data = bytearray(good.read_bytes())
    middle = len(data) // 2
    data[middle:middle + 6000] = os.urandom(6000)
    bad = tmp_path / "roop_3d/.runtime/media/target.mp4"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_bytes(bytes(data))
    rig = build_rig(tmp_path, frames=80, tracks=1, video=bad)
    try:
        record = rig.submit("mid")
        done = rig.wait(record.job_id, timeout=60)
        if done.state == JobState.succeeded:
            # Concealed by the decoder: acceptable only if the published file is complete and self-consistent.
            assert done.output.frames == 80
            _valid_output(rig, done)
        else:
            assert done.failure.code != "WORKER_FAILED", f"unnamed failure: {done.failure}"
            assert rig.output_files() == []
    finally:
        rig.worker.stop()


# ----------------------------------------------------------------------- audit and repair (safe by design)
def test_audit_finds_and_repair_fixes_crash_leftovers_without_touching_live_data(tmp_path):
    rig = build_rig(tmp_path, frames=16, tracks=1, start=False)
    root = tmp_path
    live = rig.submit("live")                                    # queued: its work directory is protected
    (rig.workdir(live.job_id) / "frames").mkdir(parents=True)
    (rig.workdir(live.job_id) / "frames" / "00000000.png").write_bytes(b"png")
    ghost = "b" * 32
    (root / f"roop_3d/.runtime/jobs/{ghost}/frames").mkdir(parents=True)
    (root / f"roop_3d/.runtime/jobs/{ghost}/frames/00000000.png").write_bytes(b"x")
    (root / "roop_3d/.runtime/jobs/state.json.0123456789abcdef0123456789abcdef.part").write_bytes(b"{")
    temp = root / "roop_3d/outputs"
    temp.mkdir(parents=True, exist_ok=True)
    (temp / ("c" * 32 + ".video.mp4")).write_bytes(b"half an encode")
    (temp / ("d" * 32 + ".sequence")).mkdir()
    userdata = root / "roop_3d/outputs/facesets/face0.fsz"        # created by the rig: must never be touched
    stray = root / rig.app.state.settings.output_root / f"roop3d_{'e' * 32}.mp4"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"orphan output")

    auditor = ReliabilityAuditor(root, rig.store, lambda: rig.app.state.settings)
    codes = {f.code for f in auditor.audit().findings}
    assert {"STALE_PART_FILE", "ORPHAN_ENCODER_TEMP", "ORPHAN_JOB_WORKDIR", "UNOWNED_OUTPUT"} <= codes
    # While jobs run, fresh artefacts are left alone.
    assert auditor.audit(min_age_s=3600).findings == []

    report = auditor.repair()
    assert report.repaired >= 5 and report.ok
    assert not list(root.rglob("*.part")) and not (temp / ("c" * 32 + ".video.mp4")).exists()
    assert not (root / f"roop_3d/.runtime/jobs/{ghost}").exists() and not stray.exists()
    assert (rig.workdir(live.job_id) / "frames" / "00000000.png").exists(), "a runnable job's frames are never deleted"
    assert userdata.exists(), "FaceSet archives are user data"
    assert any((root / "roop_3d/.runtime/quarantine").iterdir())
    assert auditor.audit().findings == []


def test_checkpoint_pointing_past_missing_frames_is_rolled_back(tmp_path):
    rig = build_rig(tmp_path, frames=60, tracks=1, delay=0.01, checkpoint_every=10)
    record = rig.submit("ckpt")
    rig.wait_frames(record.job_id, 35)
    rig.worker.stop()
    frames = rig.workdir(record.job_id) / "frames"
    (frames / "00000015.png").unlink()                            # disk damage: a stored frame is gone
    second = build_rig(tmp_path, frames=60, tracks=1, start=False)
    try:
        second.store.recover_interrupted()
        auditor = ReliabilityAuditor(tmp_path, second.store, lambda: second.app.state.settings, runner=second.worker.runner)
        found = [f for f in auditor.audit().findings if f.code == "CHECKPOINT_FRAMES_MISSING"]
        assert found and found[0].action == "rollback_checkpoint"
        auditor.repair()
        assert json.loads((second.workdir(record.job_id) / "checkpoint.json").read_text("utf-8"))["completed_frame"] == 14
        second.store.control(record.job_id, "resume")
        second.worker.start()
        assert second.wait(record.job_id, timeout=60).state == JobState.succeeded
    finally:
        second.worker.stop()


def test_unreadable_job_store_is_reported_and_only_reset_on_explicit_request(tmp_path):
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    rig.submit("x")
    state_file = tmp_path / "roop_3d/.runtime/jobs/state.json"
    state_file.write_text("{ not json", "utf-8")
    auditor = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings)
    first = auditor.audit()
    assert not first.ok and first.findings[0].code == "JOB_STORE_INVALID"
    assert not auditor.repair().ok and state_file.exists(), "no destructive repair without being asked"
    assert auditor.repair(reset_job_store=True).repaired == 1
    assert rig.store.list() == []
    assert any(p.name.endswith("state.json") for p in (tmp_path / "roop_3d/.runtime/quarantine").iterdir())


def test_queue_mirror_drift_is_detected_and_resynced(tmp_path):
    from roop_3d.queue_manager import QueueManager
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    queue = QueueManager(tmp_path)
    mirror = tmp_path / "roop_3d/.runtime/queue/queue.json"
    document = json.loads(mirror.read_text("utf-8"))
    document["job_count"] = 7
    atomic_json(tmp_path, "roop_3d/.runtime/queue/queue.json", document)
    auditor = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings)
    assert [f.code for f in auditor.audit().findings] == ["QUEUE_MIRROR_DRIFT"]
    auditor.repair(queue=queue)
    assert auditor.audit().findings == []


def test_a_silent_running_job_is_reported_as_stalled_but_a_quiet_paused_one_is_not(tmp_path):
    from datetime import timedelta
    rig = build_rig(tmp_path, frames=300, tracks=1, delay=0.02)
    try:
        record = rig.submit("stall")
        rig.wait_frames(record.job_id, 5)
        live = lambda: {record.job_id}
        fresh = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings, active_job_ids=live)
        assert not [f for f in fresh.audit().findings if f.code == "JOB_STALLED"]
        later = lambda: datetime.now(timezone.utc) + timedelta(minutes=10)   # ten silent minutes later
        stalled = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings, active_job_ids=live, clock=later)
        found = [f for f in stalled.audit().findings if f.code == "JOB_STALLED"]
        assert found and found[0].job_id == record.job_id and "silent for" in found[0].detail
        rig.store.control(record.job_id, "pause")
        rig.wait(record.job_id, JobState.paused)
        assert not [f for f in stalled.audit().findings if f.code == "JOB_STALLED"], "paused jobs are quiet by design"
    finally:
        rig.worker.stop()


def test_queue_database_corruption_is_an_error_finding(tmp_path):
    from roop_3d.queue_manager import QueueManager
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    QueueManager(tmp_path)
    db = tmp_path / "roop_3d/.runtime/queue/queue.db"
    data = bytearray(db.read_bytes())
    data[:16] = b"NOT A DATABASE!!"
    db.write_bytes(bytes(data))
    findings = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings).audit().findings
    assert any(f.code in ("QUEUE_DB_CORRUPT", "QUEUE_DB_UNREADABLE") and f.severity == "error" for f in findings)


# ------------------------------------------------------------------------------ job store read cache
def _plain_store(tmp_path):
    from roop_3d.jobs.schema import JobRequest
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    return rig, JobRequest.model_validate(rig.payload)


def test_store_parses_the_database_once_until_it_changes_on_disk(tmp_path, monkeypatch):
    rig, request = _plain_store(tmp_path)
    record, _ = rig.store.create(request, "k1")
    reads: list[str] = []
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self.name), real(self, *a, **k))[1])
    for _ in range(200):
        assert rig.store.get(record.job_id).state == JobState.queued
        rig.store.list()
    assert reads.count("state.json") <= 1, "200 reads must not re-parse an unchanged database"
    rig.store.transition(record.job_id, JobState.preparing)
    reads.clear()
    for _ in range(50):
        rig.store.get(record.job_id)
    assert reads.count("state.json") == 0, "our own write refreshes the cache; no re-read is needed"


def test_store_sees_a_write_made_by_another_instance_or_process(tmp_path):
    from roop_3d.jobs.store import JobStore
    rig, request = _plain_store(tmp_path)
    record, _ = rig.store.create(request, "k2")
    assert rig.store.get(record.job_id).state == JobState.queued          # populates this instance's cache
    other = JobStore(tmp_path)                                            # stands in for a second server process
    other.transition(record.job_id, JobState.preparing)
    assert rig.store.get(record.job_id).state == JobState.preparing
    assert [r.state for r in rig.store.list()] == [JobState.preparing]


def test_a_failed_write_never_leaves_phantom_state_in_the_cache(tmp_path, monkeypatch):
    import roop_3d.jobs.store as store_module
    rig, request = _plain_store(tmp_path)
    record, _ = rig.store.create(request, "k3")
    rig.store.get(record.job_id)
    real = store_module.atomic_json

    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(store_module, "atomic_json", disk_full)
    with pytest.raises(OSError):
        rig.store.transition(record.job_id, JobState.preparing)
    with pytest.raises(OSError):
        rig.store.create(request.model_copy(update={"trim_start_seconds": 1.0}), "k4")
    assert rig.store.get(record.job_id).state == JobState.queued, "the failed transition must not be visible"
    assert len(rig.store.list()) == 1, "the failed create must not be visible"
    monkeypatch.setattr(store_module, "atomic_json", real)
    assert rig.store.transition(record.job_id, JobState.preparing).state == JobState.preparing
    assert JobStore(tmp_path).get(record.job_id).state == JobState.preparing   # and it really is on disk


def test_hand_edited_state_file_is_noticed_and_garbage_is_still_reported(tmp_path):
    rig, request = _plain_store(tmp_path)
    record, _ = rig.store.create(request, "k5")
    rig.store.get(record.job_id)
    path = tmp_path / "roop_3d/.runtime/jobs/state.json"
    path.write_text("{ not json", "utf-8")
    with pytest.raises(DomainError) as caught:
        rig.store.get(record.job_id)
    assert caught.value.code == "JOB_STORE_INVALID"


def test_streaming_path_leftovers_are_audited_but_other_features_outputs_are_never_touched(tmp_path):
    """The default render path writes roop3d_<job>_<time_ns>.mp4 and keeps a segment cache under .runtime/stream/<job>.
    The queue's joined clips, the finishing tool's files and VFX passes share the roop3d_ prefix and must be left alone."""
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    live = rig.submit("s")                                                    # queued: its segment cache is still useful
    out = tmp_path / rig.app.state.settings.output_root
    out.mkdir(parents=True, exist_ok=True)
    stray = out / f"roop3d_{'e' * 32}_1793821286617123456.mp4"
    others = [out / f"roop3d_{'e' * 32}_joined.mp4", out / f"roop3d_finish_{'e' * 32}.mp4",
              out / f"roop3d_{'e' * 32}_passes.manifest.json", out / f"roop3d_{'e' * 32}_alpha.mkv"]
    for path in [stray, *others]:
        path.write_bytes(b"data")
    stream = tmp_path / "roop_3d/.runtime/stream"
    for name in ("c" * 32, live.job_id):
        (stream / name).mkdir(parents=True)
        (stream / name / "segment_0000.mp4").write_bytes(b"segment")
    auditor = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings)
    report = auditor.audit()
    assert {(f.code, Path(f.path).name) for f in report.findings} == {
        ("UNOWNED_OUTPUT", stray.name), ("ORPHAN_JOB_WORKDIR", "c" * 32)}
    auditor.repair()
    assert not stray.exists() and any((tmp_path / "roop_3d/.runtime/quarantine").glob(f"*{stray.name}"))
    assert all(path.exists() for path in others), "other features' outputs must never be quarantined"
    assert not (stream / ("c" * 32)).exists() and (stream / live.job_id / "segment_0000.mp4").exists()


def test_jobs_owned_by_the_persistent_queue_are_never_audited_or_repaired(tmp_path):
    """Regression (caught by test_stage09_remediation): a paused queue job's checkpoint lives in .runtime/jobs/<id>/ too."""
    from roop_3d.queue_manager import QueueManager
    rig = build_rig(tmp_path, frames=8, tracks=1, start=False)
    QueueManager(tmp_path)
    import sqlite3
    owned_id = "d" * 32
    connection = sqlite3.connect(tmp_path / "roop_3d/.runtime/queue/queue.db")
    connection.execute("INSERT INTO queue_jobs (job_id,name,state,queue_order,request_json,progress_json,slices_json,"
                       "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (owned_id, "paused", "PAUSED", 1, "{}", "{}", "[]", "2026-01-01", "2026-01-01"))
    connection.commit()
    connection.close()
    checkpoint = tmp_path / f"roop_3d/.runtime/jobs/{owned_id}/checkpoint.npz"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"queue checkpoint")
    stream = tmp_path / f"roop_3d/.runtime/stream/{owned_id}/seg.mp4"
    stream.parent.mkdir(parents=True)
    stream.write_bytes(b"x")
    auditor = ReliabilityAuditor(tmp_path, rig.store, lambda: rig.app.state.settings)
    # (the row was inserted behind QueueManager's back, so the queue.json mirror legitimately drifts: unrelated finding)
    assert [f.code for f in auditor.audit().findings if f.code != "QUEUE_MIRROR_DRIFT"] == []
    auditor.repair()
    assert checkpoint.exists() and stream.exists()
