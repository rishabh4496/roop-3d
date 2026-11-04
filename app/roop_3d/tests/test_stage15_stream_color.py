"""The raw NVDEC/NVENC stream backend must not change colour (software decode/encode; needs the ffmpeg CLI it shells out to)."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from roop_3d.media.probe import probe
from roop_3d.media.video_reader import VideoReader
from roop_3d.tests.stage15_media import expected_bgr8, make_color_video, read_stripes

hardware_streamer = pytest.importorskip("roop_3d.hardware_streamer")
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="the stream backend shells out to the ffmpeg CLI")

HD = [("709", dict(matrix=1), 1280, 720), ("709_full", dict(matrix=1, full=True), 1280, 720),
      ("601", dict(matrix=5, primaries=5, trc=6), 720, 576), ("2020", dict(matrix=9, primaries=9, trc=14), 1280, 720)]


def roundtrip(tmp_path, name, kw, w, h, color):
    source_path = make_color_video(tmp_path / f"roop_3d/{name}.mp4", frames=3, width=w, height=h, vcodec="libx264",
                                   lossless=True, **kw)
    source = hardware_streamer.NvdecFrameSource(source_path, strict=False, hardware=False)
    frames = [frame for frame, _ in source.frames()]
    destination = tmp_path / f"roop_3d/out_{name}.mp4"
    writer = hardware_streamer.NVHardwareVideoWriter(destination, source.shape, source.rate, hardware=False, strict=False,
                                                     color=source.color if color else None)
    for frame in frames:
        writer.write(frame)
    writer.close()
    image = next(iter(VideoReader(destination, hardware="software").frames())).image
    return source_path, destination, np.abs(read_stripes(image, w) - expected_bgr8()).max()


def tags(path):
    v = probe(path).videos[0]
    return (v.colorspace, v.color_primaries, v.color_trc, v.color_range)


@pytest.mark.parametrize("name,kw,w,h", HD)
def test_stream_backend_preserves_colour_and_carries_all_four_tags(tmp_path, name, kw, w, h):
    source, destination, error = roundtrip(tmp_path, name, kw, w, h, color=True)
    assert error <= 8, error
    assert tags(destination) == tags(source)


def test_the_untagged_default_really_did_shift_hd_colour(tmp_path):
    # Teeth: without the colour argument the encoder uses FFmpeg's untagged BT.601 default, which measured ~41/255 off
    # on tagged BT.709 HD and left every colour tag unspecified. This is the defect the colour argument removes.
    source, destination, error = roundtrip(tmp_path, "legacy", dict(matrix=1), 1280, 720, color=False)
    assert error > 25, error
    assert tags(destination)[:2] != tags(source)[:2] and tags(destination)[0] != 1


def test_untagged_and_high_bit_depth_material_is_routed_to_the_reference_path(tmp_path):
    untagged = make_color_video(tmp_path / "roop_3d/u.mp4", frames=3, width=1280, height=720, tagged=False, vcodec="libx264")
    with pytest.raises(ValueError, match="Untagged"):
        hardware_streamer.NvdecFrameSource(untagged, strict=False, hardware=False)
    deep = make_color_video(tmp_path / "roop_3d/d.mkv", frames=3, bits=10, matrix=9, primaries=9, trc=14)
    with pytest.raises(ValueError, match="10-bit"):
        hardware_streamer.NvdecFrameSource(deep, strict=False, hardware=False)
    pq = make_color_video(tmp_path / "roop_3d/p.mkv", frames=3, bits=10, matrix=9, primaries=9, trc=16)
    with pytest.raises(ValueError, match="HDR"):
        hardware_streamer.NvdecFrameSource(pq, strict=False, hardware=False)


def test_color_contract_helper_reports_why():
    from roop_3d.color.description import resolve_color
    ok = resolve_color(primaries=1, transfer=1, matrix=1, range_=1, width=1920, height=1080, pix_fmt="yuv420p")
    assert hardware_streamer.stream_color_supported(ok) is None
    assert "Untagged" in hardware_streamer.stream_color_supported(
        resolve_color(primaries=1, transfer=1, matrix=2, range_=1, width=1920, height=1080, pix_fmt="yuv420p"))
    assert "422" in hardware_streamer.stream_color_supported(
        resolve_color(primaries=1, transfer=1, matrix=1, range_=1, width=1920, height=1080, pix_fmt="yuv422p"))
