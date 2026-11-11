"""Stage 18 unit tests: growth analysis, lifecycle ledger, disk handling, failure classification, orphan guard, redaction."""
from __future__ import annotations

import errno
import gc
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import time

import pytest

from roop_3d.core.errors import DomainError
from roop_3d.reliability import disk, probe
from roop_3d.reliability.classify import classify, classify_domain
from roop_3d.reliability.growth import BoundedProfile, DEFAULT_PROFILE, MetricBound, analyse, theil_sen
from roop_3d.reliability.ledger import ResourceLedger
from roop_3d.reliability.processes import ProcessRegistry, terminate_tree
from roop_3d.reliability.report import Redactor

PROFILE = BoundedProfile("t", {"m": MetricBound(allowance=10.0)}, warmup_fraction=0.25, min_samples=8)


def verdict(values):
    return analyse(PROFILE, {"m": list(enumerate(values))}).verdicts["m"]


# ------------------------------------------------------------------------------------------- growth analysis
def test_theil_sen_ignores_outliers():
    xs = list(range(40))
    ys = [2.0 * x for x in xs]
    ys[10] = 10_000.0  # a single spike must not move the median slope
    assert abs(theil_sen(xs, ys) - 2.0) < 1e-9


def test_linear_leak_is_flagged():
    assert verdict([100 + 1.5 * i + random.Random(i).uniform(-2, 2) for i in range(80)]).status == "growing"


def test_warmup_then_plateau_is_bounded():
    values = [100 + 200 * (1 - math.exp(-i / 6)) for i in range(120)]  # allocator ramp, then flat
    assert verdict(values).status == "ok"


def test_gc_sawtooth_is_bounded():
    values = [300 + (i % 10) * 8 for i in range(120)]  # climbs 72 MiB then collapses: net zero
    assert verdict(values).status == "ok"


def test_single_spike_is_not_a_leak():
    values = [200.0] * 100
    values[70] = 900.0
    assert verdict(values).status == "ok"


def test_slow_leak_below_allowance_is_tolerated_but_above_is_not():
    assert verdict([100 + 0.05 * i for i in range(100)]).status == "ok"        # +5 over the window < 10
    assert verdict([100 + 0.30 * i for i in range(100)]).status == "growing"  # +30 over the window


def test_ceiling_always_fails_and_short_series_is_insufficient():
    profile = BoundedProfile("c", {"m": MetricBound(allowance=100.0, ceiling=500.0)})
    assert analyse(profile, {"m": list(enumerate([100.0] * 20 + [600.0]))}).verdicts["m"].status == "ceiling"
    assert analyse(profile, {"m": list(enumerate([1.0, 2.0, 3.0]))}).verdicts["m"].status == "insufficient"


def test_live_counts_default_to_strict_no_growth():
    series = {"live.thing": [(i, float(i // 5)) for i in range(60)]}
    assert not analyse(DEFAULT_PROFILE, series).ok


# ------------------------------------------------------------------------------------------------- ledger
def test_ledger_counts_hold_and_high_water():
    ledger = ResourceLedger()
    with ledger.hold("decoder"):
        with ledger.hold("decoder"):
            assert ledger.live("decoder") == 2
    snap = ledger.snapshot()["decoder"]
    assert (snap["live"], snap["high_water"], snap["opened"], snap["closed"], snap["abandoned"]) == (0, 2, 2, 2, 0)


def test_ledger_detects_abandoned_token_and_settles_the_books():
    ledger = ResourceLedger()
    ledger.open("encoder", "never closed")
    gc.collect()
    assert ledger.live("encoder") == 0 and ledger.abandoned() == 1


def test_ledger_double_close_and_failing_gauge_are_safe():
    ledger = ResourceLedger()
    token = ledger.open("x")
    token.close()
    token.close()
    ledger.gauge("boom", lambda: 1 / 0)
    assert ledger.counts()["x"] == 0 and ledger.counts()["boom"] == 0
    assert ledger.open_older_than(3600) == []


# ---------------------------------------------------------------------------------------------------- disk
@pytest.mark.parametrize("error", [
    OSError(errno.ENOSPC, "No space left on device"),
    PermissionError(0, "x"),
])
def test_is_disk_full_errno(error):
    assert disk.is_disk_full(error) == (error.errno == errno.ENOSPC)


def test_is_disk_full_windows_codes_and_chained_cause():
    windows = OSError(0, "There is not enough space on the disk")
    windows.winerror = 112
    assert disk.is_disk_full(windows)
    try:
        try:
            raise OSError(errno.ENOSPC, "x")
        except OSError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert disk.is_disk_full(outer)
    assert not disk.is_disk_full(ValueError("nope"))


def test_disk_estimate_discounts_resumed_frames_and_require_refuses(tmp_path, monkeypatch):
    full = disk.estimate(1920, 1080, 108_000, 4_000_000_000)
    resumed = disk.estimate(1920, 1080, 108_000, 4_000_000_000, remaining_frames=8_000)
    assert full.frame_store_bytes > 10 * resumed.frame_store_bytes
    assert full.frame_store_bytes == int(108_000 * 1920 * 1080 * 3 * disk.PNG_RATIO)
    monkeypatch.setattr(disk, "free_bytes", lambda path: 1024)
    with pytest.raises(DomainError) as caught:
        disk.require(tmp_path, full)
    assert caught.value.code == "DISK_SPACE_LOW" and caught.value.status == 507
    monkeypatch.setattr(disk, "free_bytes", lambda path: full.required_bytes + 1)
    disk.require(tmp_path, full)


def test_tree_bytes_skips_links_and_missing(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "f").write_bytes(b"x" * 100)
    assert disk.tree_bytes(tmp_path) == 100 and disk.tree_bytes(tmp_path / "nope") == 0


# ----------------------------------------------------------------------------------------- classification
def test_classification_table():
    assert classify(OSError(errno.ENOSPC, "x")).code == "DISK_FULL" and classify(OSError(errno.ENOSPC, "x")).recoverable
    assert classify(MemoryError()).code == "HOST_MEMORY_EXHAUSTED"
    assert classify(PermissionError(13, "locked")).code == "FILE_ACCESS_DENIED"
    assert classify(BrokenPipeError()).code == "ENCODER_TERMINATED"
    other = classify(KeyError("k"))
    assert other.code == "WORKER_FAILED" and not other.recoverable and "KeyError" in other.warning


def test_real_corrupt_model_is_classified_as_model_corrupt():
    import onnxruntime as ort
    with pytest.raises(Exception) as caught:
        ort.InferenceSession(b"this is not an onnx model", providers=["CPUExecutionProvider"])
    result = classify(caught.value)
    assert result.code == "MODEL_CORRUPT" and not result.recoverable


def test_real_corrupt_media_is_classified_as_media_corrupt(tmp_path):
    import av
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"\x00\x00\x00\x18ftypmp42" + os.urandom(2048))
    with pytest.raises(Exception) as caught:
        av.open(str(bad))
    assert classify(caught.value).code == "MEDIA_CORRUPT"


def test_classification_never_leaks_exception_text():
    secret = r"C:\Users\someone\Videos\private_clip.mp4"
    result = classify(RuntimeError(f"failed on {secret}"))
    assert secret not in result.message and secret not in result.warning


def test_domain_disk_errors_are_resumable_others_are_not():
    assert classify_domain(DomainError("DISK_SPACE_LOW", "low", 507)).recoverable
    assert classify_domain(DomainError("MEDIA_EMPTY", "x", 422)) is None


# ------------------------------------------------------------------------------------------ process probes
def test_probe_reports_this_process():
    memory = probe.process_memory()
    assert memory.rss_mib and memory.rss_mib > 10 and memory.handles and memory.os_threads
    assert probe.process_alive(os.getpid()) and not probe.process_alive(2 ** 22 + 12345)
    assert probe.process_create_time(os.getpid()) is not None


def test_child_and_descendant_enumeration():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        deadline = time.monotonic() + 5
        while child.pid not in probe.child_pids() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert child.pid in probe.child_pids()
    finally:
        terminate_tree(child.pid)
        child.wait(5)
    assert child.pid not in probe.child_pids()


# ------------------------------------------------------------------------------------ orphan process guard
def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_dead(pid: int, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if not probe.process_alive(pid):
            return True
        time.sleep(0.05)
    return False


def test_registry_sweep_kills_orphan_whose_owner_is_gone(tmp_path):
    registry = ProcessRegistry(tmp_path)
    child = registry.track(_sleeper(), "ffmpeg_decode")
    record = next((tmp_path / registry.relative).glob("*.json"))
    import json
    data = json.loads(record.read_text("utf-8"))
    data["owner_pid"] = 2 ** 22 + 777  # the server that spawned it no longer exists
    data["owner_create_time"] = None
    record.write_text(json.dumps(data), "utf-8")
    try:
        found = ProcessRegistry(tmp_path).sweep_stale()
        assert [item["action"] for item in found] == ["killed_orphan"]
        child.wait(5)
        assert not probe.process_alive(child.pid) and not list((tmp_path / registry.relative).glob("*.json"))
    finally:
        terminate_tree(child.pid)


def test_registry_sweep_leaves_children_of_a_live_owner(tmp_path):
    registry = ProcessRegistry(tmp_path)
    child = registry.track(_sleeper(), "ffmpeg_encode")
    try:
        found = ProcessRegistry(tmp_path).sweep_stale()
        assert found[0]["action"] == "owner_alive_left_running"
        assert child.poll() is None
    finally:
        terminate_tree(child.pid)


def test_registry_never_kills_a_recycled_pid(tmp_path):
    """The pid in the record now belongs to an unrelated process (creation time differs): it must survive."""
    import json
    registry = ProcessRegistry(tmp_path)
    innocent = _sleeper()
    try:
        (tmp_path / registry.relative).mkdir(parents=True)
        (tmp_path / registry.relative / "1-a.json").write_text(json.dumps({
            "pid": innocent.pid, "create_time": 1, "purpose": "ffmpeg_decode", "owner_pid": 2 ** 22 + 5,
            "owner_create_time": None}), "utf-8")
        found = ProcessRegistry(tmp_path).sweep_stale()
        assert found[0]["action"] == "already_exited"
        assert innocent.poll() is None
    finally:
        terminate_tree(innocent.pid)


def test_registry_live_count_is_exact(tmp_path):
    registry = ProcessRegistry(tmp_path)
    child = registry.track(_sleeper(), "x")
    assert registry.live_count() == 1
    terminate_tree(child.pid)
    child.wait(5)
    assert registry.live_count() == 0 and not list((tmp_path / registry.relative).glob("*.json"))


@pytest.mark.skipif(os.name != "nt", reason="Job Objects are the Windows mechanism; POSIX uses PR_SET_PDEATHSIG")
def test_job_object_reaps_grandchildren_when_the_parent_is_terminated(tmp_path):
    """The real unexpected-termination proof: TerminateProcess on the server must not strand its FFmpeg children."""
    script = textwrap.dedent("""
        import subprocess, sys, time
        from roop_3d.reliability.processes import bind_children_to_process_lifetime
        state = bind_children_to_process_lifetime()
        grandchild = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
        print(state["bound"], grandchild.pid, flush=True)
        time.sleep(120)
    """)
    parent = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": os.getcwd()})
    bound, grandchild = parent.stdout.readline().split()
    assert bound == "True", "job object could not be created on this machine"
    grandchild_pid = int(grandchild)
    assert probe.process_alive(grandchild_pid)
    probe.kill_process(parent.pid)  # TerminateProcess: no handlers, no atexit, no finally
    parent.wait(10)
    assert _wait_dead(grandchild_pid), "grandchild survived the death of its parent (orphan process)"


# ------------------------------------------------------------------------------------------- redaction
def test_redactor_removes_media_names_paths_embeddings_and_names(tmp_path):
    redactor = Redactor(tmp_path, ("Jane Doe Likeness",))
    home = str(Path.home())
    document = {
        "target": "holiday_with_alice.mp4", "path": f"{home}\\Videos\\secret clip.mov",
        "root": str(tmp_path / "roop_3d" / "x"), "faceset_id": "alice-smith",
        "embedding": [0.1] * 512, "landmarks": [[1.0, 2.0]] * 5, "thumbnail_base64": "A" * 400,
        "scores": [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99], "note": "Jane Doe Likeness approved",
        "api_token": "tok_123456", "nested": {"image_bytes": b"\x89PNG", "frame_count": 99},
        "message": f"failed reading D:\\media\\clips\\holiday_with_alice.mp4 for {home}"}
    out = redactor.apply(document)
    text = repr(out)
    for forbidden in ("holiday_with_alice", "secret clip", "alice-smith", "Jane Doe", "tok_123456", home,
                      "D:\\media", "0.1, 0.1", "AAAAAAAA"):
        assert forbidden not in text, forbidden
    assert out["nested"]["frame_count"] == 99
    assert out["redaction"]["applied"] and out["redaction"]["vectors_removed"] >= 1
    assert ".mp4" in text  # the extension survives for triage


# ----------------------------------------------------------------------------- storage lock handle leak
def test_per_path_locks_do_not_leak_one_os_handle_per_job(tmp_path):
    """Regression: `exclusive()` cached an RLock per lock path forever, and on Windows/CPython 3.10 every RLock owns a
    kernel semaphore - one leaked handle per rendered job (found by the Stage 18 successive-jobs soak)."""
    import shutil
    import uuid
    from roop_3d.core import storage
    root = Path(__file__).resolve().parents[2]
    try:
        rel = tmp_path.relative_to(root).as_posix()
        cleanup = None
    except ValueError:
        rel = f".runtime/pytest_locks_{uuid.uuid4().hex}"
        cleanup = root / rel
    gc.collect()
    before = probe.process_memory().handles
    try:
        for index in range(300):
            with storage.exclusive(root, f"{rel}/.out_{index}.lock"):
                pass
    finally:
        if cleanup and cleanup.exists():
            shutil.rmtree(cleanup, ignore_errors=True)
    gc.collect()
    assert probe.process_memory().handles - before <= 6
    assert len(storage._threads) == 0


def test_weakly_held_path_locks_still_exclude_across_threads(tmp_path):
    import shutil
    import threading
    import uuid
    from roop_3d.core import storage
    root = Path(__file__).resolve().parents[2]
    try:
        rel = tmp_path.relative_to(root).as_posix() + "/.shared.lock"
        cleanup = None
    except ValueError:
        folder = f".runtime/pytest_locks_{uuid.uuid4().hex}"
        rel = f"{folder}/.shared.lock"
        cleanup = root / folder
    counter = {"value": 0, "inside": 0, "overlap": 0}

    def work():
        for _ in range(120):
            with storage.exclusive(root, rel):
                counter["inside"] += 1
                if counter["inside"] > 1:
                    counter["overlap"] += 1
                value = counter["value"]
                time.sleep(0)            # invite a thread switch inside the critical section
                counter["value"] = value + 1
                counter["inside"] -= 1

    try:
        threads = [threading.Thread(target=work) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
    finally:
        if cleanup and cleanup.exists():
            shutil.rmtree(cleanup, ignore_errors=True)
    assert counter["value"] == 720 and counter["overlap"] == 0


def test_a_phase_burst_that_is_released_by_the_end_is_not_growth():
    """Found by the long-video soak: the mux phase runs more decoder threads than the render phase, then releases them."""
    values = [51.0] * 40 + [51, 44, 52, 41, 52, 45, 52, 40, 52, 61, 58, 44, 65, 58, 68, 49, 67, 68, 59, 56, 57, 49, 54, 45, 69,
                            59, 60, 63, 49, 56, 68, 58, 73, 58, 35]
    thread_profile = BoundedProfile("t", {"m": MetricBound(allowance=4.0)}, warmup_fraction=0.25, min_samples=8)
    assert analyse(thread_profile, {"m": list(enumerate(values))}).verdicts["m"].status == "ok"
    leak = [51.0 + 0.5 * i + (i % 3) for i in range(75)]            # the same shape of noise on a metric that never comes back
    assert analyse(thread_profile, {"m": list(enumerate(leak))}).verdicts["m"].status == "growing"
