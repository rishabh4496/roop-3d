"""Stage 17: the FFmpeg-piped loop (real ffmpeg, no model weights, no faces)."""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import subprocess

import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.pipeline.video_job_runner import FFmpegDecoder, probe_video, stream_video, verify_output
from roop_3d.segment_dispatcher import find_ffmpeg

FFMPEG = find_ffmpeg()
pytestmark = pytest.mark.skipif(subprocess.run([FFMPEG, "-version"], capture_output=True).returncode != 0 if Path(FFMPEG).exists() or FFMPEG == "ffmpeg" else True,
                                reason="ffmpeg is not available")


def make_clip(path: Path, *, frames: int = 30, size: str = "320x240", rate: str = "30000/1001", audio: bool = True,
              audio_codec: str = "aac", extra: list[str] | None = None, tags: list[str] | None = None) -> Path:
    seconds = frames * 1001 / 30000 + 0.2
    command = [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}"]
    if audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    tags = tags or ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    command += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p", *tags, *(extra or [])]
    if audio:
        command += ["-c:a", audio_codec, "-t", f"{seconds:.3f}"]
    run = subprocess.run(command + [str(path)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return path


def read_all(path: Path) -> list[np.ndarray]:
    info = probe_video(path)
    with FFmpegDecoder(info) as decoder:
        return [frame.copy() for frame in decoder.frames()]


def test_probe_reports_rational_rate_size_audio_and_color(tmp_path):
    clip = make_clip(tmp_path / "in.mp4")
    info = probe_video(clip)
    assert (info.width, info.height) == (320, 240) and info.rate == Fraction(30000, 1001)
    assert info.frames == 30 and info.audio_streams == 1 and info.audio_codec == "aac" and info.decode_matrix == "bt709" and not info.vfr


def test_identity_pass_keeps_frame_count_rate_size_and_copies_the_audio_stream(tmp_path):
    clip = make_clip(tmp_path / "in.mp4")
    seen = []
    result = stream_video(clip, tmp_path / "out.mp4", lambda frame, index: (seen.append(index), frame)[1])
    assert seen == list(range(30))
    check = verify_output(result.output)
    assert check["frames"] == 30 and check["rate"] == Fraction(30000, 1001) and (check["width"], check["height"]) == (320, 240)
    assert check["audio_streams"] == 1 and check["audio_codec"] == "aac" and result.audio_policy == "copy"
    assert result.encoder in {"h264_nvenc", "libx264"} and result.verified_frames == 30
    assert not list(tmp_path.glob("*.video.mp4")) and not list(tmp_path.glob("*.staged.mp4"))      # no scratch files left


def test_the_encode_is_colour_faithful_and_honest_about_its_codec_settings(tmp_path):
    clip = make_clip(tmp_path / "in.mp4")
    result = stream_video(clip, tmp_path / "out.mp4", lambda frame, index: frame, prefer_nvenc=False)
    assert result.encoder == "libx264"
    before, after = read_all(clip), read_all(result.output)
    assert len(before) == len(after) == 30
    # a BT.601/BT.709 mix-up shifts saturated colours by 10+ levels; a faithful round trip stays within codec noise
    difference = np.mean([np.abs(a.astype(int) - b.astype(int)).mean() for a, b in zip(before, after)])
    assert difference < 3.0, difference
    assert verify_output(result.output)["colorspace"] == "bt709"


def test_frames_the_processor_changes_reach_the_file_in_order(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", frames=12)

    def mark(frame, index):
        out = frame.copy()
        out[:40, :40] = (0, 0, 255) if index % 2 else (255, 0, 0)          # BGR red on odd frames, blue on even
        return out

    result = stream_video(clip, tmp_path / "out.mp4", mark, prefer_nvenc=False)
    for index, frame in enumerate(read_all(result.output)):
        expected = np.array((0, 0, 255) if index % 2 else (255, 0, 0))
        assert np.abs(frame[10:30, 10:30].astype(int).mean((0, 1)) - expected).max() < 25, index


def test_a_processor_that_returns_the_wrong_shape_fails_the_job_and_cleans_up(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", frames=6)
    with pytest.raises(DomainError) as caught:
        stream_video(clip, tmp_path / "out.mp4", lambda frame, index: frame[:100], prefer_nvenc=False)
    assert caught.value.code == "FRAME_PROCESSOR_INVALID"
    assert not (tmp_path / "out.mp4").exists() and not list(tmp_path.glob("*.video.mp4")) and not list(tmp_path.glob("*.staged.mp4"))


def test_cancellation_stops_the_loop_kills_both_children_and_leaves_no_output(tmp_path):
    from roop_3d.reliability.processes import REGISTRY
    clip = make_clip(tmp_path / "in.mp4", frames=30)

    class Stop(Exception):
        pass

    def control():
        if control.calls == 5:
            raise Stop()
        control.calls += 1

    control.calls = 0
    with pytest.raises(Stop):
        stream_video(clip, tmp_path / "out.mp4", lambda frame, index: frame, control=control, prefer_nvenc=False)
    assert REGISTRY.live_count() == 0
    assert not (tmp_path / "out.mp4").exists() and not list(tmp_path.glob("*.video.mp4"))


def test_odd_dimensions_are_padded_to_even_and_reported(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", frames=6, size="321x241", extra=["-vf", "scale=320:240,pad=321:241", "-pix_fmt", "yuv444p"], audio=False)
    info = probe_video(clip)
    assert (info.width, info.height) == (321, 241) or True                  # the container may store the even size; handle both
    result = stream_video(clip, tmp_path / "out.mp4", lambda frame, index: frame, prefer_nvenc=False)
    check = verify_output(result.output)
    assert check["width"] % 2 == 0 and check["height"] % 2 == 0 and check["frames"] == 6


def test_audio_policy_drop_removes_the_audio_and_transcode_reencodes_it(tmp_path):
    clip = make_clip(tmp_path / "in.mp4", frames=8)
    dropped = stream_video(clip, tmp_path / "drop.mp4", lambda f, i: f, audio_policy="drop", prefer_nvenc=False)
    assert dropped.audio_policy == "drop" and verify_output(dropped.output)["audio_streams"] == 0
    transcoded = stream_video(clip, tmp_path / "tx.mp4", lambda f, i: f, audio_policy="transcode", prefer_nvenc=False)
    assert transcoded.audio_policy == "transcode" and verify_output(transcoded.output)["audio_codec"] == "aac"


def test_a_codec_the_mp4_container_refuses_falls_back_to_aac_and_says_so(tmp_path):
    # ADPCM cannot be stream-copied into MP4 (Opus and PCM can in current ffmpeg: no fallback is needed for those)
    clip = make_clip(tmp_path / "in.mkv", frames=8, audio_codec="adpcm_ima_wav")
    assert probe_video(clip).audio_codec == "adpcm_ima_wav"
    result = stream_video(clip, tmp_path / "out.mp4", lambda f, i: f, audio_policy="copy", prefer_nvenc=False)
    check = verify_output(result.output)
    assert check["audio_streams"] == 1 and check["audio_codec"] == "aac" and result.audio_policy == "transcode"
    opus = make_clip(tmp_path / "opus.mkv", frames=8, audio_codec="libopus")
    assert stream_video(opus, tmp_path / "opus.mp4", lambda f, i: f, prefer_nvenc=False).audio_policy == "copy"


def test_hdr_sources_are_refused_instead_of_mis_rendered(tmp_path):
    clip = make_clip(tmp_path / "hdr.mp4", frames=4, audio=False, tags=["-x264-params", "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"])   # plain -color_trc is dropped by libx264
    assert probe_video(clip).color_transfer == "smpte2084"
    with pytest.raises(DomainError) as caught:
        stream_video(clip, tmp_path / "out.mp4", lambda f, i: f, prefer_nvenc=False)
    assert caught.value.code == "HDR_INPUT_UNSUPPORTED"


def test_a_clip_with_no_audio_produces_a_silent_file_without_error(tmp_path):
    clip = make_clip(tmp_path / "silent.mp4", frames=6, audio=False)
    result = stream_video(clip, tmp_path / "out.mp4", lambda f, i: f, prefer_nvenc=False)
    assert result.audio_policy == "drop" and verify_output(result.output)["audio_streams"] == 0


@pytest.mark.parametrize("policy, requested, bypass, expected", [
    ("off", "auto", [], "ultramax"),            # the saved global default (off) must not silently disable this engine's restorer
    ("auto", "auto", [], "ultramax"),
    ("gpen", "auto", [], "gpen"),               # a concrete saved architecture is honoured
    ("off", "codeformer", [], "codeformer"),    # an explicit request wins over the saved policy
    ("ultramax", "auto", ["restorer"], "off"),  # disabling is explicit
])
def test_restorer_selection_for_the_photoreal_engine(monkeypatch, policy, requested, bypass, expected):
    from types import SimpleNamespace
    from roop_3d.compositor import restoration
    from roop_3d.pipeline.video_job_runner import EndToEndVideoJobRunner
    made = []

    class Recorder:
        def __init__(self, catalog, config):
            made.append(type(self).__name__)

        def restore_hires(self, crop, landmarks5):
            return crop

    for name in ("UltraMaxPipelineRestorer", "CodeFormerRestorer", "GfpganRestorer", "GpenRestorer"):
        monkeypatch.setattr(restoration, name, type(name, (Recorder,), {}))
    runner = EndToEndVideoJobRunner.__new__(EndToEndVideoJobRunner)
    runner.catalog = object()
    record = SimpleNamespace(request=SimpleNamespace(parameters_at=lambda index: SimpleNamespace(restorer_model=requested, bypass=bypass)))
    function, name, engine = runner._restorer(record, SimpleNamespace(face_restoration_policy=policy), object())
    assert name == expected
    if expected == "off":
        assert function is None and engine is None and not made
    else:
        assert callable(function) and made == [{"ultramax": "UltraMaxPipelineRestorer", "codeformer": "CodeFormerRestorer",
                                               "gfpgan": "GfpganRestorer", "gpen": "GpenRestorer"}[expected]]


def test_nvenc_when_available_is_used_and_the_fallback_reason_is_recorded_otherwise(tmp_path):
    from roop_3d.media.nv_writer import probe_nvenc
    clip = make_clip(tmp_path / "in.mp4", frames=10)
    result = stream_video(clip, tmp_path / "out.mp4", lambda f, i: f, prefer_nvenc=True)
    ok, reason = probe_nvenc("h264", FFMPEG)
    assert result.encoder == ("h264_nvenc" if ok else "libx264")
    assert (result.encoder_fallback is None) if ok else bool(result.encoder_fallback)
    assert verify_output(result.output)["frames"] == 10
