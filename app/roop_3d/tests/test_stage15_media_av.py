"""Stage 15 audio, subtitles, timing, rotation and image sequences, all through real codecs and containers."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json

import av
import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.media import av_sync
from roop_3d.media.probe import probe
from roop_3d.media.sequence import SequenceReader, SequenceWriter, discover_sequence, to_bgr8
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.video_writer import encode_video
from roop_3d.tests.stage15_media import (
    expected_bgr8, make_color_video, read_stripes, stripe_rgb, write_click_audio)


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "roop_3d").mkdir(exist_ok=True)
    return tmp_path


def encode(root, source, name, **kwargs):
    kwargs.setdefault("hardware", "software")
    return encode_video(source, name, root=root, **kwargs)


def out_path(root, manifest):
    return root / manifest["output_path"]


def click_seconds(path, stream_index):
    """First sample above 0.3 full scale, as a presentation time. Independent of ``av_sync``."""
    with av.open(str(path)) as container:
        stream = container.streams[stream_index]
        for frame in container.decode(stream):
            data = frame.to_ndarray()
            channel = data[0] if frame.format.is_planar else data.reshape(-1, len(frame.layout.channels))[:, 0]
            scale = 32768.0 if channel.dtype == np.int16 else 1.0
            hits = np.flatnonzero(np.abs(channel / scale) > 0.3)
            if hits.size:
                return float(frame.pts * frame.time_base + Fraction(int(hits[0]), frame.sample_rate))
    return None


def audio_indices(path):
    with av.open(str(path)) as container:
        return [s.index for s in container.streams.audio]


def packet_hash(path, index):
    digest = hashlib.sha256()
    with av.open(str(path)) as container:
        for packet in container.demux(container.streams[index]):
            if packet.size and packet.pts is not None:
                digest.update(bytes(packet))
    return digest.hexdigest()


def frame_times(path):
    with av.open(str(path)) as container:
        return [f.pts * Fraction(f.time_base) for f in container.decode(video=0)]


# --- audio: preserved, synchronised, never changed by default -------------------------------------------
def test_default_audio_is_stream_copied_bit_identically_with_zero_lag(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=2, width=96, height=64)
    manifest = encode(root, source, "o.mov", profile="mov_workflow", transform=lambda f: f.image)
    out = out_path(root, manifest)
    sync = manifest["stage15"]["audio_sync"]
    assert [item["action"] for item in sync] == ["copy", "copy"] and all(item["payload_identical"] for item in sync)
    assert all(item["ok"] and item["measurable"] and abs(item["lag_ms"]) < 0.5 for item in sync)
    sources, produced = audio_indices(source), audio_indices(out)
    assert len(produced) == 2
    for s, p in zip(sources, produced):
        assert packet_hash(source, s) == packet_hash(out, p)
    # Independent check of the click positions (tracks click at 0.5 s and 0.75 s): sample-exact for a copy.
    for index, expected in zip(produced, (0.5, 0.75)):
        assert abs(click_seconds(out, index) - expected) <= 1 / 48000 + 1e-9


def test_transcoded_audio_stays_in_sync_within_one_aac_frame(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=1, width=96, height=64)
    manifest = encode(root, source, "o.mp4", audio_policy="aac", transform=lambda f: f.image)
    out = out_path(root, manifest)
    item = manifest["stage15"]["audio_sync"][0]
    assert item["action"] == "transcode" and item["measurable"] and item["ok"]
    assert abs(click_seconds(out, audio_indices(out)[0]) - 0.5) < 0.025
    assert abs(item["lag_ms"]) <= 0.5      # measured: the MP4 edit list compensates the AAC encoder delay


def test_trim_rebases_audio_and_video_by_the_same_amount(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=1, width=96, height=64)
    manifest = encode(root, source, "t.mp4", start=Fraction(1, 5), end=Fraction(3, 2), transform=lambda f: f.image)
    out = out_path(root, manifest)
    times = frame_times(out)
    # Frames are 40 ms apart: the last one before the 1.5 s cut is at 1.48 s, i.e. 1.28 s after the 0.2 s origin.
    assert times[0] == 0 and times[-1] == Fraction(32, 25)
    assert abs(click_seconds(out, audio_indices(out)[0]) - (0.5 - 0.2)) < 0.025
    assert manifest["stage15"]["audio_sync"][0]["ok"]


def test_the_sync_measurement_has_teeth(root, tmp_path):
    """A 100 ms offset must be reported as a 100 ms lag and fail the check."""
    import av as _av

    def clip(name, click):
        path = root / "roop_3d" / name
        with _av.open(str(path), "w") as out:
            video = out.add_stream("libx264", rate=25)
            video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
            stream = out.add_stream("pcm_s16le", rate=48000)
            stream.layout = "mono"
            for i in range(50):
                frame = _av.VideoFrame(64, 48, "yuv420p")
                frame.pts, frame.time_base = i, Fraction(1, 25)
                for packet in video.encode(frame):
                    out.mux(packet)
            for packet in video.encode(None):
                out.mux(packet)
            write_click_audio(out, stream, 2.0, click_at=click)
        return path
    reference, late = clip("a.mkv", 0.5), clip("b.mkv", 0.6)

    class Plan:
        source_index, action, codec = 1, "transcode", "pcm_s16le"
    check = av_sync.check_track(reference, late, Plan, 1, origin=Fraction(0), source_start=None, source_end=None,
                                seconds=2.0, tolerance=0.025)
    assert check.lag.measurable and abs(check.lag.lag_seconds - 0.1) < 0.001 and not check.ok
    same = av_sync.check_track(reference, reference, Plan, 1, origin=Fraction(0), source_start=None, source_end=None,
                               seconds=2.0, tolerance=0.025)
    assert same.ok and same.lag.lag_seconds == 0


def test_lag_measurement_is_honest_about_silence_and_periodic_content():
    rate = av_sync.LAG_RATE
    silent = np.zeros(rate, np.float32)
    assert not av_sync.measure_lag(silent, silent).measurable
    tone = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
    shifted = np.roll(tone, 3)
    measured = av_sync.measure_lag(tone, shifted)
    assert measured.measurable and measured.ambiguous and abs(measured.lag_seconds) < 0.01
    noise = np.random.default_rng(1).standard_normal(rate).astype(np.float32)
    assert av_sync.measure_lag(noise, np.roll(noise, 40)).lag_seconds == pytest.approx(40 / rate)


# --- timing -------------------------------------------------------------------------------------------------
VFR_PTS = (0, 40, 120, 160, 280, 320, 360, 440)


@pytest.mark.parametrize("profile,suffix", [("standard_mp4", ".mp4"), ("lossless_intermediate", ".mkv"),
                                            ("mov_workflow", ".mov"), ("archival_hevc10", ".mkv")])
def test_vfr_source_timestamps_survive_every_path_exactly(root, profile, suffix):
    source = make_color_video(root / "roop_3d/in.mkv", frames=8, pts=VFR_PTS, time_base=Fraction(1, 1000),
                              width=96, height=64)
    transform = (lambda f: f.image) if profile == "standard_mp4" else None  # others exercise the native path
    manifest = encode(root, source, "v" + suffix, profile=profile, transform=transform)
    got = frame_times(out_path(root, manifest))
    assert got == [Fraction(p, 1000) for p in VFR_PTS]
    assert manifest["verified_frames"] == 8


def test_vfr_flag_and_ntsc_rational_rate_are_reported_exactly(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=8, pts=VFR_PTS, time_base=Fraction(1, 1000))
    assert probe(source).videos[0].vfr is True
    ntsc = make_color_video(root / "roop_3d/ntsc.mkv", frames=10, rate=Fraction(30000, 1001))
    info = probe(ntsc).videos[0]
    assert info.average_rate == Fraction(30000, 1001) and info.vfr is False


def test_explicit_cfr_records_every_drop_and_duplicate(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=8, pts=VFR_PTS, time_base=Fraction(1, 1000))
    manifest = encode(root, source, "c.mp4", timing="explicit_cfr", rate=Fraction(25), transform=lambda f: f.image)
    times = frame_times(out_path(root, manifest))
    assert all(b - a == Fraction(1, 25) for a, b in zip(times, times[1:]))
    assert {e["action"] for e in manifest["frame_map"]} >= {"keep", "duplicate"}


def test_duplicated_cfr_frames_on_the_native_path_do_not_corrupt_timestamps(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=8, pts=VFR_PTS, time_base=Fraction(1, 1000), width=96, height=64)
    manifest = encode(root, source, "c.mkv", profile="lossless_intermediate", timing="explicit_cfr", rate=Fraction(25))
    times = frame_times(out_path(root, manifest))
    assert all(b - a == Fraction(1, 25) for a, b in zip(times, times[1:]))
    assert manifest["stage15"]["color"]["path"] == "native_planes" and manifest["verified_frames"] == len(times)


# --- rotation ----------------------------------------------------------------------------------------------
def test_rotation_is_applied_to_pixels_on_the_rgb_path_and_kept_as_metadata_on_the_native_path(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=3, rotation=90, width=96, height=64)
    assert probe(source).videos[0].rotation == 90
    upright = encode(root, source, "u.mp4", transform=lambda f: f.image)
    info = probe(out_path(root, upright)).videos[0]
    assert (info.width, info.height, info.rotation) == (64, 96, 0)
    assert upright["orientation_policy"] == "upright_pixels"
    native = encode(root, source, "n.mov", profile="mov_workflow")
    info = probe(out_path(root, native)).videos[0]
    assert (info.width, info.height, info.rotation) == (96, 64, 90)
    assert native["orientation_policy"] == "rotation_metadata_preserved"
    with pytest.raises(DomainError) as error:
        encode(root, source, "n.mkv", profile="lossless_intermediate")
    assert error.value.code == "NATIVE_ROTATION_UNSUPPORTED"


# --- subtitles -----------------------------------------------------------------------------------------------
def subtitle_packets(path):
    with av.open(str(path)) as container:
        streams = list(container.streams.subtitles)
        return [(bytes(p), p.pts) for p in container.demux(streams) if p.size] if streams else []


def test_subtitles_are_stream_copied_when_the_container_can_hold_them(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=1, subtitle=True, width=96, height=64)
    assert [s.codec for s in probe(source).subtitles] == ["ssa"]
    manifest = encode(root, source, "s.mkv", profile="archival_hevc10")
    out = out_path(root, manifest)
    info = probe(out)
    assert [(s.codec, s.language) for s in info.subtitles] == [("ssa", "eng")]
    assert [b for b, _ in subtitle_packets(out)] == [b for b, _ in subtitle_packets(source)]
    assert manifest["stage15"]["subtitles"][0]["action"] == "copy"


def test_subtitle_trim_rebases_like_audio_and_video(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, subtitle=True, width=96, height=64)
    manifest = encode(root, source, "s.mkv", profile="archival_hevc10", start=Fraction(1, 20))
    pts = [p for _, p in subtitle_packets(out_path(root, manifest))]
    assert pts == [80 - 50]  # the cue at 0 ms precedes the trim; the one at 80 ms moves to 30 ms


def test_subtitles_that_cannot_be_preserved_are_reported_never_silently_lost(root):
    source = make_color_video(root / "roop_3d/in.mkv", frames=10, subtitle=True, width=96, height=64)
    manifest = encode(root, source, "s.mp4", transform=lambda f: f.image)
    plan = manifest["stage15"]["subtitles"][0]
    assert plan["action"] == "drop" and "cannot hold ssa" in plan["reason"]
    assert probe(out_path(root, manifest)).subtitles == ()
    with pytest.raises(DomainError) as error:
        encode(root, source, "r.mp4", transform=lambda f: f.image, subtitles="require")
    assert error.value.code == "SUBTITLE_NOT_PRESERVABLE"
    dropped = encode(root, source, "d.mkv", profile="archival_hevc10", subtitles="drop")
    assert probe(out_path(root, dropped)).subtitles == ()


# --- image sequences -------------------------------------------------------------------------------------------
def write_sequence(directory, count=5, fmt="png", stem="shot", start=1001):
    directory.mkdir(parents=True, exist_ok=True)
    writer = SequenceWriter(directory, fmt, stem=stem, start_number=start)
    frames = []
    for i in range(count):
        image = np.rint(np.roll(stripe_rgb(64, 32)[..., ::-1], i * 8, axis=1) * 255).astype(np.uint8)
        writer.write(i, image, pts=Fraction(i, 24), duration=Fraction(1, 24))
        frames.append(image)
    return frames


@pytest.mark.parametrize("fmt", ["png", "tiff", "dpx"])
def test_image_sequence_input_is_read_with_exact_rational_timestamps(root, fmt):
    frames = write_sequence(root / "roop_3d/seq", fmt=fmt)
    rate = Fraction(24000, 1001)
    info = discover_sequence(root / "roop_3d/seq", rate)
    assert (info.count, info.width, info.height, info.first_number) == (5, 64, 32, 1001)
    decoded = list(SequenceReader(info).frames())
    assert [f.timestamp.seconds for f in decoded] == [Fraction(i * 1001, 24000) for i in range(5)]
    assert all(np.array_equal(f.image, want) for f, want in zip(decoded, frames))  # 8-bit formats are exact


def test_sequence_to_video_keeps_ntsc_timing_and_colour(root):
    write_sequence(root / "roop_3d/seq", count=6)
    manifest = encode(root, root / "roop_3d/seq", "from_seq.mp4", sequence_rate=Fraction(24000, 1001),
                      transform=lambda f: f.image)
    out = out_path(root, manifest)
    assert frame_times(out) == [Fraction(i * 1001, 24000) for i in range(6)]
    image = next(iter(VideoReader(out, hardware="software").frames())).image
    assert np.abs(read_stripes(image) - expected_bgr8()).max() <= 20
    color = manifest["stage15"]["color"]
    assert color["source"]["assumed_fields"] and "RGB" in " ".join(color["notes"])


def test_sequence_errors_are_specific(root):
    write_sequence(root / "roop_3d/seq", count=4)
    with pytest.raises(DomainError) as error:
        encode(root, root / "roop_3d/seq", "x.mp4")
    assert error.value.code == "SEQUENCE_RATE_REQUIRED"
    (root / "roop_3d/seq/shot_00001002.png").unlink()
    with pytest.raises(DomainError) as error:
        discover_sequence(root / "roop_3d/seq", Fraction(24))
    assert error.value.code == "SEQUENCE_GAP"
    write_sequence(root / "roop_3d/two", count=2, stem="a")
    write_sequence(root / "roop_3d/two", count=2, stem="b")
    with pytest.raises(DomainError) as error:
        discover_sequence(root / "roop_3d/two", Fraction(24))
    assert error.value.code == "SEQUENCE_AMBIGUOUS"


def test_sixteen_bit_stills_reduce_to_eight_bits_with_rounding(root):
    directory = root / "roop_3d/deep"
    directory.mkdir(parents=True)
    wide = np.zeros((8, 8, 3), np.uint16)
    wide[:, :4] = 65535
    wide[:, 4:] = 128 * 257
    assert cv2.imwrite(str(directory / "f_0001.png"), wide)
    info = discover_sequence(directory, Fraction(24))
    assert info.source_bit_depth == 16
    from roop_3d.media.sequence import _decode_image
    image = to_bgr8(_decode_image(info.files[0]), ".png")
    assert image[0, 0, 0] == 255 and image[0, 7, 0] == 128


@pytest.mark.parametrize("fmt", ["png", "tiff", "dpx"])
def test_image_sequence_export_is_exact_hashed_and_carries_timing_and_audio(root, fmt):
    source = make_color_video(root / "roop_3d/in.mkv", frames=12, pts=tuple(range(0, 480, 40)), time_base=Fraction(1, 1000),
                              audio=1, width=96, height=64)
    marks = []

    def capture(frame):
        marks.append(frame.image.copy())
        return frame.image
    manifest = encode(root, source, "seq_out", profile="image_sequence", transform=capture, sequence_format=fmt)
    directory = out_path(root, manifest)
    document = json.loads((directory / "sequence.json").read_text())
    assert document["frame_count"] == 12 == len(list(directory.glob(f"frame_*.*"))) and document["format"] == fmt
    assert document["frames"][3]["pts_seconds"] == "3/25" and document["frames"][0]["pts_seconds"] == "0/1"
    for record, want in zip(document["frames"], marks):
        path = directory / record["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    assert document["audio"] == ["audio.mka"]
    sidecar_index = audio_indices(directory / "audio.mka")[0]
    assert packet_hash(directory / "audio.mka", sidecar_index) == packet_hash(source, audio_indices(source)[0])
    assert manifest["stage15"]["audio_sync"][0]["payload_identical"] in (True, None)
    assert manifest["stage15"]["fidelity"] and all(item["exact"] for item in manifest["stage15"]["fidelity"])
    assert manifest["output_sha256"] == hashlib.sha256((directory / "sequence.json").read_bytes()).hexdigest()


def test_a_failed_or_cancelled_sequence_export_leaves_nothing_behind(root):
    from threading import Event
    source = make_color_video(root / "roop_3d/in.mkv", frames=10, width=96, height=64)
    cancel = Event()

    def stop(frame):
        cancel.set()
        return frame.image
    with pytest.raises(DomainError):
        encode(root, source, "seq_out", profile="image_sequence", transform=stop, cancel=cancel)
    assert not (root / "roop_3d/outputs/seq_out").exists()
    assert not [p for p in (root / "roop_3d/outputs").glob("*.sequence")]
