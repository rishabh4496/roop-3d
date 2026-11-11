"""Pinokio terminal HUD, 14-section runtime report and TensorRT build heartbeat.

The HUD is presentation over measured values, so these tests pin the contracts that matter: the launcher's output
trip-wires are never printed, an unmeasured value says so, a short build adds nothing to the socket, and the live line is
never wider than the terminal.
"""
from __future__ import annotations

import io
import logging
import os
from pathlib import Path
import re
import threading
import time

import pytest
from fastapi.testclient import TestClient

from roop_3d.server.app import create_app
from roop_3d.server.pipeline_status import PipelineTracker
from roop_3d.server.ws import EVENT_ADAPTER, TrtCompilationProgressEvent
from roop_3d.telemetry import pinokio_terminal as terminal
from roop_3d.telemetry import runtime_report as report_module
from roop_3d.telemetry.pinokio_terminal import (CLEAR_LINE, JobHud, PinokioTerminalHUD, ResourceReading, ResourceSampler,
                                                defuse, enforce_terminal_environment, render_frame_line, rolling_fps,
                                                truncate_visible, visible_len)
from roop_3d.telemetry.runtime_report import SECTION_TITLES, build_runtime_report, render_report
from roop_3d.telemetry.trt_heartbeat import TrtCompilationHeartbeat, format_shape, heartbeat_line, precision_label

ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
STATUS = "ws://127.0.0.1:8030/ws/v1/pipeline/status"


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def hud(stream: io.StringIO | None = None, *, in_place: bool = True, width: int = 140, clock: Clock | None = None, **kwargs):
    sampler = ResourceSampler(gpu=lambda: None)
    sampler.sample_once = lambda: ResourceReading()  # type: ignore[method-assign]
    return PinokioTerminalHUD(stream or io.StringIO(), color=False, in_place=in_place, clock=clock or Clock(), sampler=sampler,
                              width=lambda: width, min_redraw=0.0, **kwargs).arm()


# ---------------------------------------------------------------------------------------------- environment lock
def test_environment_lock_sets_unbuffered_utf8_and_reports_only_what_changed():
    env: dict[str, str] = {"PYTHONUTF8": "1"}
    changed = enforce_terminal_environment(env, streams=False)
    assert env == {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    assert set(changed) == {"PYTHONUNBUFFERED", "PYTHONIOENCODING"}
    assert enforce_terminal_environment(env, streams=False) == {}


def test_child_environment_overrides_a_hostile_parent_environment():
    child = terminal.child_environment({"PYTHONUNBUFFERED": "0", "PYTHONIOENCODING": "cp1252", "PATH": "x"})
    assert child["PYTHONUNBUFFERED"] == "1" and child["PYTHONIOENCODING"] == "utf-8" and child["PATH"] == "x"


def test_launcher_scripts_set_the_same_contract_as_the_server():
    root = Path(__file__).resolve().parents[3]
    for script in ("start.js", "install.js"):
        text = (root / script).read_text("utf-8")
        for name, value in terminal.TERMINAL_ENVIRONMENT.items():
            assert re.search(rf'{name}:\s*"{value}"', text), f"{script} must set {name}={value}"


# ----------------------------------------------------------------------------------------------- launcher tripwires
def test_defuse_removes_every_token_start_js_treats_as_a_failure():
    hostile = "error: PORT_IN_USE INVALID_SETTINGS ROOP3D_GUARD_ERROR ModuleNotFoundError"
    assert not re.search(r"(PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)", defuse(hostile))


def test_hud_lines_are_defused_before_they_reach_the_terminal():
    out = io.StringIO()
    hud(out).line("error: something PORT_IN_USE")
    assert "error:" not in out.getvalue() and "PORT_IN_USE" not in out.getvalue()


def test_report_text_cannot_trip_the_launcher_and_only_names_loopback_urls():
    text = build_runtime_report(None, None)["text"]
    assert not re.search(r"(PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR|ModuleNotFoundError|error:)", text)
    # Pinokio captures the first numeric-host URL; the report may only name the machine's own loopback address.
    assert set(re.findall(r"http://([0-9.]+):", text)) <= {"127.0.0.1"}


# ---------------------------------------------------------------------------------------------------- the report
def test_report_has_the_fourteen_sections_in_order_with_status_and_rows():
    report = build_runtime_report(None, None)
    assert [s["key"] for s in report["sections"]] == [key for key, _ in SECTION_TITLES]
    assert [s["number"] for s in report["sections"]] == list(range(1, 15))
    assert {s["status"] for s in report["sections"]} <= {"OK", "INFO", "WARN", "UNAVAILABLE", "UNKNOWN"}
    assert all(s["rows"] for s in report["sections"])
    assert "UNKNOWN" not in {s["status"] for s in report["sections"]}, [s for s in report["sections"] if s["status"] == "UNKNOWN"]
    assert sum(report["counts"].values()) == 14


def test_a_failing_collector_degrades_to_unknown_and_leaves_the_rest_intact(monkeypatch):
    def broken(context, section):
        raise RuntimeError("secret C:/Users/someone/path")
    monkeypatch.setattr(report_module, "_COLLECTORS", (report_module._COLLECTORS[0], broken, *report_module._COLLECTORS[2:]))
    report = build_runtime_report(None, None)
    bad = report["sections"][1]
    assert bad["status"] == "UNKNOWN" and bad["rows"] == [{"label": "collector", "value": "failed: RuntimeError"}]
    assert "secret" not in report["text"]  # only the exception type is reported, never its message
    assert report["sections"][0]["status"] != "UNKNOWN" and report["sections"][2]["rows"]
    assert report["overall"] == "ATTENTION"


def test_report_masks_the_home_directory():
    home = str(Path.home()).replace("\\", "/")
    assert report_module.tilde(Path.home() / "x" / "y") == "~/x/y"
    assert home not in build_runtime_report(None, None)["text"]


def test_environment_section_is_only_ok_when_the_terminal_contract_holds(monkeypatch):
    for name in terminal.TERMINAL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    assert build_runtime_report(None, None)["sections"][0]["status"] == "WARN"
    enforce_terminal_environment(streams=False)
    for name, value in terminal.TERMINAL_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    assert "terminal contract" not in " ".join(build_runtime_report(None, None)["sections"][0]["notes"]).lower()


def test_ansi_and_plain_renderings_have_identical_visible_geometry():
    report = build_runtime_report(None, None)
    plain, colored = render_report(report, width=100, ansi=False), render_report(report, width=100, ansi=True)
    assert "\x1b" not in plain and "\x1b" in colored
    assert ANSI.sub("", colored) == plain
    assert {len(line) for line in plain.splitlines()} == {100}, "every box row must be exactly the requested width"


def test_the_gpu_section_reads_hardware_and_never_invents_an_architecture():
    assert report_module._architecture("8.9") == "Ada Lovelace" and report_module._architecture("12.0") == "Blackwell"
    assert report_module._architecture(None) is None and report_module._architecture("1.0") is None


def test_runtime_report_endpoint_returns_json_and_copyable_text(tmp_path):
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/v1/system/runtime-report")
    assert response.status_code == 200, response.text
    data = response.json()
    assert [s["number"] for s in data["sections"]] == list(range(1, 15))
    assert "\x1b" not in data["text"] and data["text"].count("┌") == 14
    # With a real app state the governance section reads the live catalogue instead of reporting none attached.
    governance = next(s for s in data["sections"] if s["key"] == "governance")
    assert "No model catalogue is attached to this process." not in governance["notes"]


# ------------------------------------------------------------------------------------------------- the live line
def test_frame_line_shows_every_field_and_marks_unmeasured_stages():
    job = JobHud("3fa9c2e1" + "0" * 24, "FULL_CRANIOFACIAL_LOCK", 1200, 0.0, budget_mib=8192.0)
    job.current_fps = 23.4
    line = render_frame_line(job, frame=540, rolling=21.9, reading=ResourceReading(62.0, 3174.4, 12282.0, "process"), peak_mib=3993.6,
                             timings={"detection": 4.1, "pose_tps": 6.2, "inswapper": 9.8, "restorer": 15.1, "masking": 8.2},
                             width=400, color=False)
    for expected in ("3fa9c2e1", "FULL_CRANIOFACIAL_LOCK", "45.0%", "540/1200", "23.4/21.9 fps", "ETA 00:30",
                     "VRAM 3.1/3.9/8.0GB", "CPU 62%", "Det 4.1", "3D-LBS 6.2", "3DSwap+SH 9.8", "Enhancer 15.1", "Comp 8.2"):
        assert expected in line, expected
    # The compositor times Inpaint+Cage and LapBlend together; they are not split, so each says so.
    assert "Inpaint+Cage --" in line and "LapBlend --" in line


def test_unmeasured_resources_print_dashes_not_zeros():
    job = JobHud("abcd1234" + "0" * 24, "INNER_FACE", None, 0.0)
    line = render_frame_line(job, frame=3, rolling=None, reading=ResourceReading(), peak_mib=None, timings=None, width=400, color=False)
    assert "--/-- fps" in line
    assert "CPU --%" in line and "VRAM --/--/--GB" in line and "ETA --:--" in line and "3/?" in line


def test_device_wide_vram_is_labelled_as_such():
    job = JobHud("abcd1234" + "0" * 24, "X", 10, 0.0)
    line = render_frame_line(job, frame=1, rolling=1.0, reading=ResourceReading(1.0, 1024.0, 4096.0, "device"), peak_mib=1024.0,
                             timings=None, width=300, color=False)
    assert "VRAM(dev) 1.0/1.0/--GB" in line


@pytest.mark.parametrize("width", [40, 60, 80, 100, 140])
def test_the_live_line_never_exceeds_the_terminal_width_and_keeps_the_essentials(width):
    job = JobHud("3fa9c2e1" + "0" * 24, "FULL_CRANIOFACIAL_LOCK", 1200, 0.0, budget_mib=8192.0)
    line = render_frame_line(job, frame=540, rolling=21.9, reading=ResourceReading(62.0, 3174.4, 12282.0, "process"), peak_mib=3993.6,
                             timings={"detection": 4.1}, width=width, color=True)
    assert visible_len(line) <= width
    assert "3fa9c2e1" in ANSI.sub("", line)


def test_truncate_never_splits_an_escape_sequence():
    text = "\x1b[32m" + "x" * 50 + "\x1b[0m" + "y" * 50
    cut = truncate_visible(text, 20)
    assert visible_len(cut) == 20 and cut.endswith("\x1b[0m")
    assert truncate_visible("short", 20) == "short" and truncate_visible("anything", 0) == ""


def test_rolling_fps_uses_the_window_and_refuses_to_guess():
    from collections import deque
    assert rolling_fps(deque(), 5) is None and rolling_fps(deque([(0.0, 0)]), 5) is None
    samples = deque([(0.0, 0), (1.0, 10), (10.0, 100), (11.0, 130)])
    assert rolling_fps(samples, 5) == pytest.approx(30.0)   # only the last second is inside the 5 s window... measured from t=10
    assert rolling_fps(deque([(0.0, 5), (1.0, 5)]), 5) == 0.0


def test_in_place_bar_rewrites_one_line_and_permanent_lines_scroll_above_it():
    out, clock = io.StringIO(), Clock()
    h = hud(out, clock=clock)
    h.job_started("a" * 32, swap_mode="full_craniofacial", total_frames=100, report=False)
    for frame in (10, 20, 30):
        clock.now += 1.0
        h.frame("a" * 32, frame=frame, total=100, stage="render", stage_timings_ms={"detection": 4.0})
    h.milestone("GOP", "segment complete", segment=1)
    clock.now += 1.0
    h.frame("a" * 32, frame=40, total=100, stage="render")
    text = out.getvalue()
    assert text.count(CLEAR_LINE) >= 4 and "FULL_CRANIOFACIAL_LOCK" in text and "30.0%" in text
    assert text.index("[GOP]") < text.rindex("40.0%"), "the bar is redrawn after the milestone line"
    assert "\n" in text.split("[GOP]")[1].split("%")[0], "the milestone is a permanent, newline-terminated line"


def test_non_tty_mode_prints_plain_lines_at_a_low_rate_without_cursor_codes():
    out, clock = io.StringIO(), Clock()
    h = hud(out, in_place=False, clock=clock, plain_interval=5.0)
    h.job_started("b" * 32, swap_mode="inner_face", total_frames=100, report=False)
    for step in range(1, 13):
        clock.now += 1.0
        h.frame("b" * 32, frame=step * 5, total=100, stage="render")
    text = out.getvalue()
    assert "\r" not in text and "\x1b[K" not in text
    assert len([line for line in text.splitlines() if "INNER_FACE" in line and "▶" in line]) <= 3


def test_an_unarmed_hud_writes_nothing_and_never_raises():
    out = io.StringIO()
    h = PinokioTerminalHUD(out)
    h.job_started("c" * 32, swap_mode="x", report=True)
    h.frame("c" * 32, frame=1, total=2, stage="render")
    assert h.milestone("A", "b") is None
    h.trt_beat("x"); h.trt_done("y"); h.job_finished("c" * 32, "failed"); h.boot(None); h.print_report("x")
    assert out.getvalue() == ""


def test_roop3d_hud_off_keeps_the_hud_silent(monkeypatch):
    monkeypatch.setenv("ROOP3D_HUD", "0")
    assert PinokioTerminalHUD(io.StringIO()).arm().armed is False


def test_a_broken_terminal_never_fails_a_render():
    class Broken(io.StringIO):
        def write(self, text):
            raise OSError("terminal gone")
    h = hud(Broken())
    h.job_started("d" * 32, swap_mode="x", total_frames=2, report=False)
    h.frame("d" * 32, frame=1, total=2, stage="render")
    h.milestone("A", "b")
    h.job_finished("d" * 32, "failed")   # none of the above raised


def test_log_records_clear_the_live_line_before_printing_and_redraw_it_after():
    out = io.StringIO()
    h = hud(out)
    h.job_started("e" * 32, swap_mode="x", total_frames=10, report=False)
    h.frame("e" * 32, frame=5, total=10, stage="render")
    handler = terminal.HudLogHandler(out, h)
    handler.setFormatter(logging.Formatter("%(message)s"))
    before = len(out.getvalue())
    handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, "log record", None, None))
    tail = out.getvalue()[before:]
    assert tail.startswith(CLEAR_LINE) and "log record\n" in tail
    assert "50.0%" in tail.split("log record\n")[1], "the bar is redrawn below the log record"


def test_sampler_peak_is_the_maximum_seen_and_reset_starts_a_new_window():
    readings = iter([ResourceReading_gpu(100), ResourceReading_gpu(300), ResourceReading_gpu(200)])
    sampler = ResourceSampler(gpu=lambda: next(readings), cpu=_FakeCpu())
    for _ in range(3):
        sampler.sample_once()
    assert sampler.peak_vram_mib == 300 and sampler.latest().vram_used_mib == 200 and sampler.latest().vram_scope == "device"
    sampler.reset_peak()
    assert sampler.peak_vram_mib == 200


class ResourceReading_gpu:
    available, per_process, process_mib, total_mib = True, False, None, 1000.0

    def __init__(self, used):
        self.used_mib = float(used)


class _FakeCpu:
    def read(self):
        return 12.5


def test_a_failing_gpu_probe_is_an_unmeasured_value_not_an_error():
    def explode():
        raise OSError("nvidia-smi missing")
    reading = ResourceSampler(gpu=explode, cpu=_FakeCpu()).sample_once()
    assert reading.vram_used_mib is None and reading.vram_scope is None and reading.cpu_pct == 12.5


# ----------------------------------------------------------------------------------------------- TRT heartbeat
def test_heartbeat_line_matches_the_requested_format():
    line = heartbeat_line("inswapper_128.onnx", "FP16", "4x3x512x512", 14.04, 3.21, "process")
    assert line == "[TRT-BUILDER] ⚙ Compiling inswapper_128.onnx (FP16+FP32 Mix | opt=4x3x512x512) ... Elapsed: 14.0s | VRAM: 3.2 GB"
    assert heartbeat_line("m", None, None, 0.0, None, None).endswith("Elapsed: 0.0s | VRAM: n/a")
    assert "(dev)" in heartbeat_line("m", "FP32", None, 1.0, 2.0, "device") and "opt=n/a" in heartbeat_line("m", "FP32", None, 1.0, 2.0, "device")


def test_precision_label_and_shape_formatting():
    assert precision_label("fp16") == "FP16+FP32 Mix" and precision_label("FP32") == "FP32" and precision_label(None) == "precision n/a"
    assert format_shape((4, 3, 512, 512)) == "4x3x512x512" and format_shape({"in": (1, 3, 128, 128)}) == "1x3x128x128"
    assert format_shape(None) is None and format_shape("1x3") == "1x3" and format_shape(("a",)) is None


class _BlockingStdin:
    """A sidecar stdin that stays open until `release` is set, then reports end-of-file."""

    def __init__(self, release):
        self.release = release

    def read(self, size):
        self.release.wait()
        return ""


def _thread(target):
    return threading.Thread(target=target, daemon=True)


class _Terminal:
    """Stands in for the sidecar process handle; records the order of events against the heartbeat's other sinks."""

    def __init__(self, log):
        self.log = log

    def stop(self):
        self.log.append("terminal.stop")


def _heartbeat(interval=0.01, vram=lambda: (1.5, "process"), *, settle=0.0, spawn=None):
    log, beats, done, frames = [], [], [], []
    beat = TrtCompilationHeartbeat(beat=lambda line: (beats.append(line), log.append("beat")), done=lambda line: (done.append(line), log.append("done")),
                                   publish=lambda frame: (frames.append(frame), log.append(f"publish:{frame['status']}")), interval=interval, vram=vram,
                                   settle=settle, spawn_terminal=spawn)
    beat.log = log
    return beat, beats, done, frames


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition not reached")


def test_a_build_publishes_its_start_frame_first_then_streams_while_the_process_can_run_and_closes_once():
    beat, beats, done, frames = _heartbeat()
    beat.begin("inswapper_128.onnx", "FP16", (4, 3, 512, 512))
    assert frames[0]["status"] == "compiling" and frames[0]["elapsed_seconds"] == 0.0 and frames[0]["vram_gb"] is None
    assert frames[0]["started_at"] > 0, "the UI counts its own clock from this frame, so it must be there before any native call"
    _wait_for(lambda: len(frames) >= 4)
    assert beat.active and beat.snapshot()["model"] == "inswapper_128.onnx"
    beat.end("compile finished; canary running")
    beat.end("again")  # idempotent
    compiling = [f for f in frames if f["status"] == "compiling"]
    assert len(compiling) >= 4 and frames[-1]["status"] == "finished" and frames[-1]["reason"].startswith("compile finished")
    assert [f["elapsed_seconds"] for f in compiling] == sorted(f["elapsed_seconds"] for f in compiling)
    assert {f["opt_shape"] for f in frames} == {"4x3x512x512"} and {f["precision_label"] for f in frames} == {"FP16+FP32 Mix"}
    assert any(f["vram_gb"] == 1.5 and f["vram_scope"] == "process" for f in compiling[1:]), "later frames carry the measured VRAM"
    assert beats[0] == heartbeat_line("inswapper_128.onnx", "FP16", "4x3x512x512", 0.0, 1.5, "process"), "no sidecar: the in-process line starts at t=0"
    assert len(done) == 1 and not beat.active and beat.snapshot() is None
    for frame in frames:
        assert EVENT_ADAPTER.validate_python(frame).type == "trt_compilation_progress"


def test_the_start_frame_is_followed_by_a_settle_pause_that_precedes_the_terminal_sidecar_and_the_build():
    started = []
    def spawn(info):
        started.append(info["model"])
        beat.log.append("spawn")
        return _Terminal(beat.log)
    beat, beats, done, frames = _heartbeat(interval=30.0, settle=0.2, spawn=spawn)
    t0 = time.monotonic()
    beat.begin("a.onnx", "FP32")
    assert time.monotonic() - t0 >= 0.2, "begin sleeps (releasing the GIL) so the event loop can deliver the start frame"
    assert beat.log[:2] == ["publish:compiling", "spawn"], "the frame goes out before the sidecar starts and before the build can take the GIL"
    beat.end()
    assert started == ["a.onnx"]


def test_with_a_sidecar_the_terminal_line_comes_from_it_and_it_is_stopped_before_the_closing_line():
    beat, beats, done, frames = _heartbeat(spawn=lambda info: _Terminal(beat.log))
    beat.begin("a.onnx", "FP16")
    _wait_for(lambda: len(frames) >= 3)
    beat.end("ended")
    assert beats == [], "the in-process thread must not print a second, competing line"
    assert beat.log.index("terminal.stop") < beat.log.index("done"), "the sidecar erases its line first, so the closing line starts clean"
    assert [f["status"] for f in frames][-1] == "finished"


def test_a_sidecar_that_cannot_start_falls_back_to_the_in_process_line():
    def refuse(info):
        raise OSError("cannot spawn")
    beat, beats, done, frames = _heartbeat(spawn=refuse)
    beat.begin("a.onnx", "FP16")
    _wait_for(lambda: len(beats) >= 2)
    beat.end()
    assert "Compiling a.onnx" in beats[0] and frames[0]["status"] == "compiling"
    none_beat, none_beats, *_ = _heartbeat(spawn=lambda info: None)
    none_beat.begin("b.onnx", "FP16")
    _wait_for(lambda: len(none_beats) >= 1)
    none_beat.end()


def test_a_build_that_ends_at_once_costs_the_socket_exactly_a_start_and_a_finish_frame():
    beat, beats, done, frames = _heartbeat(interval=30.0)
    beat.begin("tiny.onnx", "FP32")
    beat.end("ended")
    assert [f["status"] for f in frames] == ["compiling", "finished"] and len(done) == 1


def test_a_late_client_gets_the_current_build_with_its_elapsed_time_brought_up_to_date():
    clock = Clock()
    log, frames = [], []
    beat = TrtCompilationHeartbeat(beat=log.append, done=log.append, publish=frames.append, interval=30.0, clock=clock, settle=0.0)
    beat.begin("a.onnx", "FP16")
    clock.now += 12.34
    assert beat.snapshot()["elapsed_seconds"] == 12.3 and beat.snapshot()["status"] == "compiling"
    beat.end()


def test_a_second_model_restarts_the_clock_and_the_same_model_does_not():
    beat, beats, done, frames = _heartbeat()
    beat.begin("a.onnx", "FP16")
    _wait_for(lambda: len(frames) >= 2)
    started = beat.snapshot()["started_at"]
    beat.begin("a.onnx", "FP16", (1, 3, 128, 128))
    assert beat.snapshot()["started_at"] == started
    beat.begin("b.onnx", "FP32")
    assert len(done) == 1 and "a.onnx" in done[0]
    _wait_for(lambda: any(f["model"] == "b.onnx" and f["status"] == "compiling" for f in frames))
    beat.end()


def test_a_failing_sink_or_vram_probe_never_breaks_the_build_thread():
    def boom(*_):
        raise RuntimeError("sink down")
    beat = TrtCompilationHeartbeat(beat=boom, done=boom, publish=boom, interval=0.01, vram=lambda: 1 / 0, settle=0.0, spawn_terminal=boom)
    beat.begin("x.onnx", "FP16")
    time.sleep(0.05)
    beat.end()
    assert not beat.active


def test_the_sidecar_prints_nothing_during_its_grace_period_then_one_rising_line_per_interval(monkeypatch):
    from roop_3d.telemetry import trt_heartbeat_sidecar as sidecar
    out, release = io.StringIO(), threading.Event()
    stdin = _BlockingStdin(release)
    clock = Clock()
    worker = _thread(lambda: sidecar.run("m.onnx", "FP16", "1x3x8x8", 1234, clock.now, 0.5, 0.01, stdin=stdin, out=out, clock=clock))
    monkeypatch.setattr(sidecar, "_vram", lambda pid: (2.5, "device"))
    worker.start()
    time.sleep(0.1)
    assert out.getvalue() == "", "inside the grace window (a cache hit finishes here) nothing may be printed"
    clock.now += 1.0
    _wait_for(lambda: out.getvalue().count("\n") >= 2)
    clock.now += 1.0
    release.set()
    worker.join(3)
    lines = [line for line in out.getvalue().splitlines() if line]
    assert lines and all(line.startswith("[TRT-BUILDER] ⚙ Compiling m.onnx (FP16+FP32 Mix | opt=1x3x8x8) ... Elapsed: ") for line in lines)
    assert "VRAM: 2.5 GB (dev)" in lines[0] and not worker.is_alive()


def test_on_a_tty_the_sidecar_rewrites_one_line_and_erases_it_on_exit(monkeypatch):
    from roop_3d.telemetry import trt_heartbeat_sidecar as sidecar

    class Tty(io.StringIO):
        def isatty(self):
            return True
    out, release = Tty(), threading.Event()
    clock = Clock()
    monkeypatch.setattr(sidecar, "_vram", lambda pid: (None, None))
    worker = _thread(lambda: sidecar.run("m.onnx", None, None, 1, clock.now - 5, 0.0, 0.01, stdin=_BlockingStdin(release), out=out, clock=clock))
    worker.start()
    _wait_for(lambda: "Compiling" in out.getvalue())
    release.set()
    worker.join(3)
    text = out.getvalue()
    assert text.startswith("\r\x1b[K[TRT-BUILDER]") and "\n" not in text and text.endswith("\r\x1b[K")
    assert "precision n/a" in text and "opt=n/a" in text and "VRAM: n/a" in text


def test_the_real_sidecar_process_prints_while_its_parent_is_busy_and_exits_when_stdin_closes():
    import subprocess
    import sys as system
    from roop_3d.audit.project_paths import application_root
    process = subprocess.Popen([system.executable, "-B", "-m", "roop_3d.telemetry.trt_heartbeat_sidecar", "--model", "real.onnx", "--parent", str(os.getpid()),
                                "--precision", "FP16", "--shape", "1x3x8x8", "--grace", "0", "--interval", "0.1"],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=str(application_root()), env=terminal.child_environment())
    try:
        first = process.stdout.readline().decode("utf-8")
        second = process.stdout.readline().decode("utf-8")
    finally:
        process.stdin.close()
    rest = process.stdout.read().decode("utf-8")
    assert process.wait(timeout=10) == 0
    assert "[TRT-BUILDER] ⚙ Compiling real.onnx (FP16+FP32 Mix | opt=1x3x8x8) ... Elapsed: " in first
    assert float(re.search(r"Elapsed: ([0-9.]+)s", second).group(1)) > float(re.search(r"Elapsed: ([0-9.]+)s", first).group(1))
    assert "\x1b[K" not in first + second, "piped output (not a TTY) is plain lines, never cursor codes"
    del rest


def test_the_hud_only_hands_out_a_sidecar_when_it_is_armed_on_the_real_terminal():
    info = {"model": "m.onnx", "wall": time.time(), "precision": "FP16", "opt_shape": "1x3x8x8"}
    assert PinokioTerminalHUD(io.StringIO()).spawn_trt_sidecar(info) is None                      # unarmed
    assert hud(io.StringIO()).spawn_trt_sidecar(info) is None                                     # custom stream: the child could not reach it
    off = PinokioTerminalHUD().arm()
    import os
    os.environ["ROOP3D_TRT_SIDECAR"] = "0"
    try:
        assert off.spawn_trt_sidecar(info) is None                                                # explicitly disabled
    finally:
        del os.environ["ROOP3D_TRT_SIDECAR"]


def test_tracker_starts_the_heartbeat_after_the_status_frame_and_stops_it_on_idle():
    frames: list[dict] = []
    tracker = PipelineTracker(frames.append)
    tracker._heartbeat.interval, tracker._heartbeat.settle = 0.01, 0.0
    tracker.set_engine("COMPILING_TRT", label="m FP16", model="m.onnx", precision="FP16", opt_shape=(2, 3, 64, 64))
    assert frames[0]["type"] == "pipeline.status" and frames[0]["engine_state"] == "COMPILING_TRT"
    assert frames[1]["type"] == "trt_compilation_progress" and frames[1]["elapsed_seconds"] == 0.0, "the start frame follows the status frame at once"
    _wait_for(lambda: sum(f["type"] == "trt_compilation_progress" for f in frames) >= 3)
    tracker.set_engine("RUNNING_CANARY", label="m FP16", model="m.onnx", precision="FP16")
    assert not tracker._heartbeat.active
    tracker.set_engine("IDLE")
    trt = [f for f in frames if f["type"] == "trt_compilation_progress"]
    assert trt[0]["opt_shape"] == "2x3x64x64" and trt[-1]["status"] == "finished"
    assert len([f for f in trt if f["status"] == "finished"]) == 1


def test_compilation_event_carries_the_optimisation_shape_to_the_heartbeat():
    frames: list[dict] = []
    tracker = PipelineTracker(frames.append)
    tracker._heartbeat.interval, tracker._heartbeat.settle = 0.01, 0.0
    tracker.compilation_event({"engine_state": "COMPILING_TRT", "engine_label": "Compiling TensorRT Engine...", "engine_model": "d.onnx",
                               "engine_precision": "FP16", "engine_since": time.time(), "engine_batch_profile": [1, 2],
                               "engine_opt_shape": {"input": (2, 3, 640, 640)}})
    assert tracker.trt_snapshot()["opt_shape"] == "2x3x640x640"
    tracker.set_engine("IDLE")


def test_trt_progress_reaches_pipeline_status_clients_and_late_joiners_are_current(tmp_path):
    app = create_app(repo_root=tmp_path, worker_enabled=False)
    app.state.pipeline._heartbeat.interval, app.state.pipeline._heartbeat.settle = 0.02, 0.0
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        with client.websocket_connect(STATUS) as status:
            assert [status.receive_json()["type"] for _ in range(2)] == ["stream.ready", "pipeline.status"]
            app.state.pipeline.set_engine("COMPILING_TRT", label="x FP16", model="x.onnx", precision="FP16", opt_shape=(4, 3, 512, 512))
            frame = status.receive_json()
            assert frame["type"] == "pipeline.status" and frame["engine_state"] == "COMPILING_TRT"
            frame = status.receive_json()
            assert frame["type"] == "trt_compilation_progress" and frame["status"] == "compiling" and frame["elapsed_seconds"] == 0.0
            assert frame["model"] == "x.onnx" and frame["opt_shape"] == "4x3x512x512" and frame["precision_label"] == "FP16+FP32 Mix"
            assert frame["started_at"] > 0 and "percent" not in frame
            with client.websocket_connect(STATUS) as late:
                kinds = []
                while "trt_compilation_progress" not in kinds:
                    kinds.append(late.receive_json()["type"])
                assert kinds[:2] == ["stream.ready", "pipeline.status"], "a late client is current at once"
            app.state.pipeline.set_engine("IDLE")
            seen = []
            while not (seen and seen[-1].get("status") == "finished"):
                message = status.receive_json()
                if message["type"] == "trt_compilation_progress":
                    seen.append(message)
            assert seen[-1]["reason"] == "ended"


def test_the_event_model_rejects_a_negative_clock_and_unknown_scopes():
    with pytest.raises(Exception):
        TrtCompilationProgressEvent(model="m", elapsed_seconds=-1)
    with pytest.raises(Exception):
        TrtCompilationProgressEvent(model="m", vram_scope="gpu")


# --------------------------------------------------------------------------------- job milestones, end to end
def test_a_real_job_emits_start_gop_checkpoint_remux_and_finish_milestones(tmp_path, monkeypatch):
    from roop_3d.tests.test_stage12_worker import _fixture, _wait_state
    out = io.StringIO()
    private = hud(out, in_place=False, plain_interval=0.0)
    reports: list[tuple[str, dict | None]] = []
    private.print_report = lambda trigger, *, job=None, wait=False: reports.append((trigger, job))  # type: ignore[method-assign]
    monkeypatch.setattr("roop_3d.jobs.worker.HUD", private)
    monkeypatch.setattr("roop_3d.engine.pipeline_runner.HUD", private)
    app, worker, payload = _fixture(tmp_path, count=8)
    worker.runner.checkpoint_every = 4
    worker.start()
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            created = client.post("/api/v1/jobs", json=payload, headers={"Idempotency-Key": "hud-e2e"})
            assert created.status_code == 201
            job_id = created.json()["job_id"]
            _wait_state(client, job_id, "succeeded")
            deadline = time.monotonic() + 5
            while "[JOB] succeeded" not in out.getvalue() and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        worker.stop()
    text = out.getvalue()
    short = job_id[:8]
    assert f"[JOB] started job={short} mode=FULL_CRANIOFACIAL_LOCK" in text
    assert reports and reports[0][0] == "job start" and reports[0][1]["job_id"] == job_id
    assert re.search(rf"\[GOP\] segment complete job={short} segment=1 frames=0-3 resume_at=4", text)
    assert re.search(rf"\[GOP\] segment complete job={short} segment=2 frames=4-7 resume_at=8", text)
    assert re.search(rf"\[CHECKPOINT\] committed job={short} completed_frame=3 resume_frame=4", text)
    assert f"[MUX] encode and audio remux started job={short}" in text
    assert re.search(rf"\[REMUX\] output written and verified job={short} frames=8 audio=drop - output has no audio sha256=[0-9a-f]{{16}}", text)
    assert re.search(rf"\[JOB\] succeeded job={short} frames=", text)
    assert "FULL_CRANIOFACIAL_LOCK" in text and "8/8" in text
    assert text.index("[JOB] started") < text.index("[GOP]") < text.index("[REMUX]") < text.index("[JOB] succeeded")
