"""W2 ops G17: the Stage-17 remux keeps ALL audio + subtitle streams, metadata and chapters; NVENC/CPU selection (no GPU needed)."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from roop_3d.pipeline import video_job_runner as runner
from roop_3d.segment_dispatcher import find_ffmpeg, find_ffprobe

FFMPEG = shutil.which("ffmpeg")
needs_ffmpeg = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg is not on PATH (which ffmpeg -> None)")

META = """;FFMETADATA1
title=W2 Container Title
[CHAPTER]
TIMEBASE=1/1000
START=0
END=400
title=Intro
[CHAPTER]
TIMEBASE=1/1000
START=400
END=1000
title=Main
"""


def probe(path: Path) -> dict:
    run = subprocess.run([find_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", "-show_chapters", str(path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def build_container(tmp_path: Path, *, subtitles: bool = True) -> Path:
    (tmp_path / "a.srt").write_text("1\n00:00:00,000 --> 00:00:00,500\nhello\n", "utf-8")
    (tmp_path / "b.srt").write_text("1\n00:00:00,200 --> 00:00:00,800\nbonjour\n", "utf-8")
    (tmp_path / "meta.txt").write_text(META, "utf-8")
    command = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=25", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
               "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000", "-i", str(tmp_path / "a.srt"), "-i", str(tmp_path / "b.srt"),
               "-i", str(tmp_path / "meta.txt"), "-map", "0:v", "-map", "1:a", "-map", "2:a"]
    if subtitles:
        command += ["-map", "3:s", "-map", "4:s", "-c:s", "srt", "-metadata:s:s:0", "language=eng", "-metadata:s:s:1", "language=fra"]
    command += ["-map_metadata", "5", "-map_chapters", "5", "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=deu",
                "-frames:v", "25", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-t", "1", str(tmp_path / "src.mkv")]
    done = subprocess.run(command, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    return tmp_path / "src.mkv"


@needs_ffmpeg
def test_all_audio_subtitle_streams_metadata_and_chapters_survive(tmp_path):
    source = build_container(tmp_path)
    info = runner.probe_video(source)
    assert info.audio_streams == 2 and info.subtitle_streams == 2
    result = runner.stream_video(source, tmp_path / "out.mp4", lambda frame, index: frame, prefer_nvenc=False)
    document = probe(result.output)
    kinds = [s["codec_type"] for s in document["streams"]]
    assert kinds.count("video") == 1 and kinds.count("audio") == 2 and kinds.count("subtitle") == 2
    assert [s["tags"].get("language") for s in document["streams"] if s["codec_type"] == "audio"] == ["eng", "deu"]
    assert [s["tags"].get("language") for s in document["streams"] if s["codec_type"] == "subtitle"] == ["eng", "fra"]
    assert [c["tags"]["title"] for c in document["chapters"]] == ["Intro", "Main"]
    assert document["format"]["tags"]["title"] == "W2 Container Title"
    # srt cannot be stream-copied into MP4: the documented per-container fallback converted it, audio was stream-copied
    assert result.remux.subtitles == "mov_text" and result.remux.audio == "copy"
    assert result.remux.audio_streams == 2 and result.remux.subtitle_streams == 2
    assert runner.verify_output(result.output)["subtitle_streams"] == 2


@needs_ffmpeg
def test_audio_policy_drop_keeps_subtitles_and_chapters_but_no_audio(tmp_path):
    source = build_container(tmp_path)
    result = runner.stream_video(source, tmp_path / "drop.mp4", lambda f, i: f, prefer_nvenc=False, audio_policy="drop")
    document = probe(result.output)
    kinds = [s["codec_type"] for s in document["streams"]]
    assert "audio" not in kinds and kinds.count("subtitle") == 2 and len(document["chapters"]) == 2
    assert result.remux.audio == "drop" and result.remux.audio_streams == 0


@needs_ffmpeg
def test_source_without_subtitles_reports_none_and_still_copies_chapters(tmp_path):
    source = build_container(tmp_path, subtitles=False)
    result = runner.stream_video(source, tmp_path / "out.mp4", lambda f, i: f, prefer_nvenc=False)
    assert result.remux.subtitles == "none" and result.remux.audio_streams == 2
    assert len(probe(result.output)["chapters"]) == 2


@needs_ffmpeg
def test_gop_aligned_slices_are_stream_copied_not_reencoded(tmp_path):
    from roop_3d.segment_dispatcher import compute_gop_aligned_cuts
    keyframes = [0.0, 2.0, 4.0, 6.0, 8.0]
    cuts = compute_gop_aligned_cuts(keyframes, 10.0, max_segment_duration=4.0)
    assert all(start in keyframes or start == 0.0 for start, _ in cuts) and cuts[0][0] == 0.0 and cuts[-1][1] == 10.0
    assert all(b > a for a, b in cuts) and all(cuts[i][1] == cuts[i + 1][0] for i in range(len(cuts) - 1))



def _writer(monkeypatch, tmp_path: Path, *, nvenc_ok: bool, prefer: bool):
    from fractions import Fraction
    from roop_3d.media import nv_writer
    monkeypatch.setattr(nv_writer, "probe_nvenc", lambda codec="h264", ffmpeg=None: (nvenc_ok, "" if nvenc_ok else "Cannot load nvcuda.dll"))
    monkeypatch.setattr(runner.ColourManagedWriter, "_start", lambda self, encoder: setattr(self, "encoder", encoder))
    return runner.ColourManagedWriter(tmp_path / "o.mp4", 64, 64, Fraction(25), prefer_hardware=prefer, ffmpeg="ffmpeg")


def test_nvenc_is_selected_when_the_probe_succeeds_and_command_uses_vbr_cq(monkeypatch, tmp_path):
    writer = _writer(monkeypatch, tmp_path, nvenc_ok=True, prefer=True)
    assert writer.encoder == "h264_nvenc" and writer.fallback_reason is None
    command = writer._command(writer.encoder, None)
    assert command[command.index("-c:v") + 1] == "h264_nvenc" and "-cq" in command and "-crf" not in command


def test_cpu_fallback_when_nvenc_probe_fails_or_hardware_not_requested(monkeypatch, tmp_path):
    failed = _writer(monkeypatch, tmp_path, nvenc_ok=False, prefer=True)
    assert failed.encoder == "libx264" and "NVENC unavailable" in failed.fallback_reason
    command = failed._command(failed.encoder, None)
    assert "-crf" in command and "-cq" not in command
    cpu = _writer(monkeypatch, tmp_path, nvenc_ok=True, prefer=False)
    assert cpu.encoder == "libx264" and cpu.fallback_reason == "hardware encoding not requested"


def test_hevc_selection_uses_hevc_nvenc_and_libx265_fallback(monkeypatch, tmp_path):
    from roop_3d.media import nv_writer
    monkeypatch.setattr(nv_writer.NVHardwareVideoWriter, "_start", lambda self, encoder: setattr(self, "encoder", encoder))
    monkeypatch.setattr(nv_writer, "probe_nvenc", lambda codec="h264", ffmpeg=None: (codec == "hevc", ""))
    hw = nv_writer.NVHardwareVideoWriter(tmp_path / "a.mp4", 64, 64, 25.0, codec="hevc", ffmpeg="ffmpeg")
    assert hw.encoder == "hevc_nvenc" and "hvc1" in hw._command("hevc_nvenc", None)
    monkeypatch.setattr(nv_writer, "probe_nvenc", lambda codec="h264", ffmpeg=None: (False, "no"))
    assert nv_writer.NVHardwareVideoWriter(tmp_path / "b.mp4", 64, 64, 25.0, codec="hevc", ffmpeg="ffmpeg").encoder == "libx265"
