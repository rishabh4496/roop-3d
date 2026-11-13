"""W2 ops G13/G14: FrameStats math, dashboard selection, JSON logs, CLI diagnostics (exactly 14 sections), queue + render guards."""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from roop_3d import cli
from roop_3d.telemetry import json_log, tui
from roop_3d.telemetry.runtime_report import SECTION_TITLES


# ------------------------------------------------------------------------------------------------------ FrameStats
def test_percentiles_fps_and_eta_match_numpy_over_the_rolling_window():
    stats = tui.FrameStats(window=5, total_frames=100)
    for ms in (1000, 10, 20, 30, 40, 50):                  # the 1000 ms outlier rolls out of the 5-frame window
        stats.record(ms)
    window = np.array([10, 20, 30, 40, 50], float)
    got = stats.percentiles()
    assert got == {k: pytest.approx(v) for k, v in zip(("p50", "p95", "p99"), np.percentile(window, [50, 95, 99]))}
    assert got["p50"] == 30.0 and got["p95"] == pytest.approx(48.0)
    assert stats.fps() == pytest.approx(1000 / 30.0)
    assert stats.done == 6 and stats.eta_seconds() == pytest.approx((100 - 6) / (1000 / 30.0))


def test_unmeasured_values_are_none_never_zero_and_bad_input_is_rejected():
    stats = tui.FrameStats()
    assert stats.percentiles() == {"p50": None, "p95": None, "p99": None} and stats.fps() is None and stats.eta_seconds() is None
    stats.record(10)
    assert stats.eta_seconds() is None                      # total unknown
    with pytest.raises(ValueError):
        stats.record(float("nan"))
    with pytest.raises(ValueError):
        stats.record(-1)
    done = tui.FrameStats(total_frames=2)
    done.record(5), done.record(5), done.record(5)
    assert done.eta_seconds() == 0.0
    assert tui.format_eta(None) == "--" and tui.format_eta(75) == "1:15" and tui.format_eta(3725) == "1:02:05"


def test_stage_shares_group_pipeline_timings_into_the_five_bars():
    rows = tui.stage_shares({"decode": 10, "detection": 10, "pose_tps": 10, "inswapper": 40, "restorer": 20, "masking": 5, "encode": 5})
    assert [r[0] for r in rows] == ["Decode", "3D Track/Detect", "3D Mesh/Swap", "Restore/Blend", "Encode"]
    assert [r[1] for r in rows] == [10, 20, 40, 25, 5] and sum(r[2] for r in rows) == pytest.approx(1.0)
    assert tui.stage_shares({"decode": 4})[2][1:] == (None, None)       # unmeasured != 0


# ------------------------------------------------------------------------------------------------ probe + selection
class _Gpu:
    name, total_vram_mib, free_vram_mib = "RTX", 6144, 5000


class _Hardware:
    gpus = [_Gpu()]
    system_ram_total_mib, system_ram_available_mib = 16000, 9000
    tensorrt_mixed_precision = {"active_provider": "TensorrtExecutionProvider"}
    advertised_providers = ["CUDAExecutionProvider"]


def test_system_probe_reports_hardware_and_null_with_reason_for_what_is_not_measurable():
    snap = tui.SystemProbe(lambda: _Hardware()).snapshot()
    assert snap.provider == "TensorrtExecutionProvider" and snap.vram_total_mib == 6144 and snap.ram_available_mib == 9000
    assert snap.vram_allocated_mib is None and snap.vram_reserved_mib is None
    assert "vram_allocated_reserved" in snap.unavailable
    failing = tui.SystemProbe(lambda: (_ for _ in ()).throw(RuntimeError("no nvidia-smi"))).snapshot()
    assert failing.vram_total_mib is None and failing.ram_available_mib is None and "RuntimeError" in failing.unavailable["vram"]


def test_plain_line_prints_dashes_not_zeros_for_unmeasured_values():
    stats = tui.FrameStats(total_frames=10)
    line = tui.plain_line(stats, {}, tui.SystemSnapshot(), "decode")
    assert "fps=--" in line and "p50/p95/p99=--/--/--ms" in line and "vram(alloc/res/total)=--" in line and "\x1b" not in line
    stats.record(20)
    assert "fps=50.0" in tui.plain_line(stats, {"decode": 3.0}, tui.SystemSnapshot(provider="CPUExecutionProvider"), "render")


def test_dashboard_selection_is_real(monkeypatch):
    plain = lambda **k: tui.make_dashboard(**k).kind                                   # noqa: E731
    assert plain(is_tty=True, rich_importable=False) == "plain"                         # no Rich
    assert plain(is_tty=False, rich_importable=True) == "plain"                         # not a TTY
    assert plain(is_tty=True, rich_importable=True, headless=True) == "plain"           # --headless
    if tui.rich_available():
        assert plain(is_tty=True, rich_importable=True) == "rich"
    else:
        monkeypatch.setattr(tui, "RichDashboard", lambda stream: type("R", (), {"kind": "rich"})())
        assert plain(is_tty=True, rich_importable=True) == "rich"                       # selection only; Rich itself is absent here


def test_plain_dashboard_throttles_and_writes_to_its_stream():
    clock = iter([0.0, 0.5, 2.5])
    stream = io.StringIO()
    dash = tui.PlainDashboard(stream, every_seconds=2.0, clock=lambda: next(clock))
    stats = tui.FrameStats()
    for _ in range(3):
        dash.update(stats, {}, tui.SystemSnapshot(), "x")
    assert len(stream.getvalue().splitlines()) == 2


@pytest.mark.skipif(not tui.rich_available(), reason="rich is not installed in this environment (import rich fails)")
def test_rich_dashboard_renders_to_a_terminal_stream():
    stream = io.StringIO()
    stats = tui.FrameStats(total_frames=4)
    stats.record(20)
    with tui.RichDashboard(stream) as dash:
        dash.update(stats, {"decode": 1.0, "inswapper": 9.0}, tui.SystemSnapshot(provider="CUDAExecutionProvider"), "render", force=True)
    assert "Decode" in stream.getvalue() and "CUDAExecutionProvider" in stream.getvalue()


# ----------------------------------------------------------------------------------------------------------- json log
def test_json_logs_are_one_versioned_object_per_line_without_ansi():
    stream = io.StringIO()
    log = json_log.JsonLogger(stream, "info")
    log.emit("render.progress", frame=3, note="\x1b[31mred\x1b[0m text", nan=float("nan"), nested={"\x1b[1mk": ["\x1b[2mv"]})
    log.emit("hidden", "debug")
    lines = stream.getvalue().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema"] == json_log.SCHEMA == "roop3d.cli.log/1" and record["event"] == "render.progress" and record["level"] == "info"
    assert record["note"] == "red text" and record["nan"] is None and record["nested"] == {"k": ["v"]}
    assert "\x1b" not in stream.getvalue() and stream.getvalue().isascii()


# ------------------------------------------------------------------------------------------------------------- CLI
def test_diagnostics_emits_exactly_the_14_section_report_plus_model_and_asset_status():
    stream = io.StringIO()
    code = cli.main(["--diagnostics", "--json-logs", "--no-model-hash"], stream)
    record = json.loads(stream.getvalue().splitlines()[0])
    assert record["schema"] == "roop3d.cli.log/1" and record["event"] == "diagnostics"
    sections = record["runtime_report"]["sections"]
    assert len(sections) == 14 == len(SECTION_TITLES) and [s["key"] for s in sections] == [k for k, _ in SECTION_TITLES]
    assert record["model_sha256"]["state"] == "SKIPPED" and "assets" in record["asset_integrity"]
    assert code in (0, 1) and (code == 0) == record["ok"]


def test_diagnostics_text_mode_lists_all_sections_and_integrity_lines(capsys):
    stream = io.StringIO()
    cli.main(["diagnostics", "--no-model-hash"], stream)
    text = stream.getvalue()
    assert text.count(" · ") >= 14 and "MODEL SHA-256 VERIFICATION" in text and "3D ASSET INTEGRITY" in text and "\x1b" not in text


def test_flags_given_before_or_after_the_subcommand_are_both_honoured():
    before = cli.build_parser().parse_args(["--json-logs", "--headless", "diagnostics"])
    after = cli.build_parser().parse_args(["diagnostics", "--json-logs", "--headless", "--log-level", "debug"])
    assert before.json_logs and before.headless and after.json_logs and after.headless and after.log_level == "debug"


def test_queue_list_and_resume_refusal_use_the_persisted_queue(tmp_path):
    from roop_3d.queue_manager import QueueManager
    manager = QueueManager(tmp_path)
    job, _ = manager.submit({"target_path": "x.mp4"}, name="demo")
    stream = io.StringIO()
    assert cli.main(["queue", "list", "--json-logs", "--root", str(tmp_path)], stream) == 0
    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert events[0]["event"] == "queue.job" and events[0]["job_id"] == job.job_id and events[0]["lifecycle_state"] == "queued"
    assert events[-1] == {**events[-1], "event": "queue.summary", "count": 1}
    refused = io.StringIO()
    assert cli.main(["queue", "resume", job.job_id, "--json-logs", "--root", str(tmp_path)], refused) == 1     # only PAUSED jobs resume
    assert json.loads(refused.getvalue())["code"] == "ILLEGAL_JOB_TRANSITION"


def test_render_with_a_missing_input_exits_2_with_a_machine_readable_error(tmp_path):
    stream = io.StringIO()
    code = cli.main(["render", str(tmp_path / "nope.mp4"), "--source", "x.fsz", "-o", str(tmp_path / "o.mp4"), "--json-logs"], stream)
    record = json.loads(stream.getvalue())
    assert code == 2 and record["event"] == "render.error" and record["code"] == "INPUT_MISSING"
