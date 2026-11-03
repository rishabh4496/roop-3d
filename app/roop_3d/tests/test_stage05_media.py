"""Stage 05 acceptance uses generated real containers and real codecs only."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from threading import Event

import av
import cv2
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.media.audio import plan_audio
from roop_3d.media.frame_map import FrameMap
from roop_3d.media.probe import probe
from roop_3d.media.timestamps import Timestamp, TimestampEngine, rescale
from roop_3d.media.video_reader import DecodedFrame, VideoReader
from roop_3d.media.video_writer import encode_video


def make_video(path: Path, *, vfr: bool = False, rotation: int = 0, audio: int = 0,
               audio_codec: str = "pcm_s16le", offset: int = 0, rate: Fraction = Fraction(25),
               size: tuple[int, int] = (96, 64), video_codec: str = "libx264", sar: Fraction | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        video = output.add_stream(video_codec, rate=rate)
        video.width, video.height, video.pix_fmt = size[0], size[1], "yuv420p"
        base = Fraction(1, 1000) if rate == 25 else Fraction(1, rate.numerator)
        video.codec_context.time_base = base
        video.codec_context.max_b_frames = 0
        video.codec_context.gop_size = 2
        video.codec_context.colorspace = video.codec_context.color_primaries = video.codec_context.color_trc = 1
        video.codec_context.color_range = 1
        if sar is not None:
            video.codec_context.sample_aspect_ratio = sar
        if rotation:
            video.set_display_rotation(rotation)
        streams = []
        for index in range(audio):
            stream = output.add_stream(audio_codec, rate=48000)
            stream.layout = "mono" if index == 0 else "stereo"
            stream.metadata["language"] = ("eng", "hin")[index % 2]
            streams.append(stream)
        pts_list = (0, 40, 120, 160, 280, 320, 360, 440) if vfr else tuple(range(0, 320, 40))
        duration = 40
        if rate != 25:
            pts_list = tuple(i * rate.denominator for i in range(90))
            duration = rate.denominator
        for index, pts in enumerate(pts_list):
            image = np.full((size[1], size[0], 3), index * 25 % 256, np.uint8)
            image[:, :32, 2] = 240
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base, frame.duration = pts + offset, base, duration
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode(None):
            output.mux(packet)
        for index, stream in enumerate(streams):
            count = int((pts_list[-1] + duration) * base * 48000)
            signal = (np.sin(np.arange(count) * (index + 1) / 15) * 5000).astype(np.int16)
            channels = 1 if index == 0 else 2
            values = np.repeat(signal, channels).reshape(1, -1)
            frame = av.AudioFrame.from_ndarray(values, format="s16", layout=stream.layout.name)
            frame.pts, frame.time_base, frame.sample_rate = offset * 48, Fraction(1, 48000), 48000
            for packet in stream.encode(frame):
                output.mux(packet)
            for packet in stream.encode(None):
                output.mux(packet)
    return path


def frame_times(path: Path) -> list[Fraction]:
    with av.open(str(path)) as source:
        return [f.pts * Fraction(f.time_base) for f in source.decode(video=0)]


def audio_payloads(path: Path) -> list[str]:
    with av.open(str(path)) as source:
        hashes = {s.index: hashlib.sha256() for s in source.streams.audio}
        if not hashes:
            return []
        for packet in source.demux(source.streams.audio):
            if packet.size:
                hashes[packet.stream.index].update(bytes(packet))
        return [h.hexdigest() for h in hashes.values()]


@pytest.mark.parametrize("vfr,rotation,audio", [(False, 0, 0), (True, 0, 1), (False, 90, 0), (True, 0, 2)])
def test_probe_exact_metadata_decode_and_random_access(tmp_path, vfr, rotation, audio):
    source = make_video(tmp_path / "roop_3d/input.mov", vfr=vfr, rotation=rotation, audio=audio)
    info = probe(source)
    video = info.videos[0]
    assert isinstance(video.time_base, Fraction)
    assert video.nominal_rate is not None and video.average_rate is not None
    assert video.codec == "h264" and video.pixel_format == "yuv420p"
    assert video.color_space_name == "BT.709" and video.color_range == 1
    assert video.rotation == rotation and video.vfr is vfr and video.sample_complete
    assert len(info.audio) == audio
    if audio:
        assert info.audio[0].layout == "mono" and info.audio[0].sample_rate == 48000
    if audio == 2:
        assert info.audio[1].layout == "stereo" and info.audio[1].language == "hin"
    reader = VideoReader(source, hardware="software")
    frames = list(reader.frames())
    assert [f.timestamp.seconds for f in frames] == frame_times(source)
    assert frames[0].image.shape == ((96, 64, 3) if rotation else (64, 96, 3))
    if rotation:
        assert frames[0].image[-20:, :, 2].mean() > frames[0].image[:20, :, 2].mean() + 100
    at = frames[5].timestamp
    assert reader.at(at).timestamp == at
    thumbnails = list(reader.thumbnails([at]))
    assert thumbnails[0][0] == at
    assert cv2.imdecode(np.frombuffer(thumbnails[0][1], np.uint8), cv2.IMREAD_COLOR) is not None
    keys = list(reader.frames(keyframes_only=True))
    assert all(f.key_frame for f in keys) and 1 < len(keys) < len(frames)


def test_exact_tick_engine_long_timeline_and_errors():
    base = Fraction(1, 90000)
    for index in (0, 1, 107892, 100000000):
        assert rescale(index, Fraction(1001, 30000), base) == index * 3003
    assert rescale(-1, Fraction(1, 2), Fraction(1)) == -1
    assert rescale(1, Fraction(1, 2), Fraction(1)) == 1
    engine = TimestampEngine()
    assert engine.accept(9999999999999, base).pts == 9999999999999
    with pytest.raises(DomainError, match="increase"):
        engine.accept(1, base)
    with pytest.raises(DomainError, match="no source PTS"):
        TimestampEngine().accept(None, base)


def test_frame_map_logs_every_drop_and_duplicate():
    frames = [DecodedFrame(i, Timestamp(pts, Fraction(1, 1000)), 40, True, np.zeros((2, 2, 3), np.uint8))
              for i, pts in enumerate((0, 10, 20, 120))]
    mapping = FrameMap(Fraction(25))
    result = list(mapping.select(frames))
    assert [frame.index for frame, _ in result] == [0, 2, 2, 3]
    assert [stamp.seconds for _, stamp in result] == [Fraction(i, 25) for i in range(4)]
    assert [(e.source_index, e.action) for e in mapping.entries] == [(0, "keep"), (1, "drop"), (2, "keep"), (2, "duplicate"), (3, "keep")]


@pytest.mark.parametrize("vfr,rotation,audio,offset", [(False, 0, 0, 0), (True, 0, 1, 0), (True, 90, 2, 0), (True, 0, 2, 2000)])
def test_real_encode_preserves_every_pts_audio_track_and_manifest(tmp_path, vfr, rotation, audio, offset):
    source = make_video(tmp_path / "roop_3d/input.mov", vfr=vfr, rotation=rotation, audio=audio, offset=offset)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = encode_video(source, "result.mov", root=tmp_path, hardware="software")
    target = tmp_path / "roop_3d/outputs/result.mov"
    assert manifest["encoded_frames"] == manifest["verified_frames"] == 8
    assert manifest["encoder"] == "libx264" and manifest["input_sha256"] == digest
    assert manifest["provenance"]["synthetic_media"] is True
    assert json.loads(target.with_suffix(".mov.manifest.json").read_text()) == manifest
    assert audio_payloads(source) == audio_payloads(target)
    times = frame_times(source)
    assert all(abs(a - b) <= probe(source).videos[0].time_base for a, b in zip(times, frame_times(target)))
    assert len(frame_times(target)) == len(times)
    info = probe(target)
    assert info.videos[0].colorspace == 1 and info.videos[0].color_range == 1
    assert info.videos[0].rotation == 0
    assert len(info.audio) == audio
    assert source.read_bytes() and hashlib.sha256(source.read_bytes()).hexdigest() == digest
    assert not list(target.parent.glob("*.video.*")) and not list(target.parent.glob("*.mux.*"))
    with pytest.raises(DomainError, match="new files"):
        encode_video(source, "result.mov", root=tmp_path, hardware="software")


def test_auto_hardware_uses_verified_backend_or_real_software_fallback(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov")
    reader = VideoReader(source)
    frames = list(reader.frames())
    assert len(frames) == 8
    assert (reader.backend, reader.hardware_state) in (("nvdec", "VERIFIED_LOCAL"), ("software", "UNSUPPORTED"))
    manifest = encode_video(source, "auto.mp4", root=tmp_path)
    assert manifest["encoder"] in ("h264_nvenc", "libx264")
    assert len(frame_times(tmp_path / "roop_3d/outputs/auto.mp4")) == 8


@pytest.mark.parametrize("policy,suffix", [("copy", ".mp4"), ("aac", ".mkv"), ("pcm", ".mov")])
def test_incompatible_audio_and_explicit_transcode_keep_all_tracks(tmp_path, policy, suffix):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=2)
    manifest = encode_video(source, "audio" + suffix, root=tmp_path, hardware="software", audio_policy=policy)
    target = tmp_path / ("roop_3d/outputs/audio" + suffix)
    info = probe(target)
    assert len(info.audio) == 2
    assert [s.language for s in info.audio] == ["eng", "hin"]
    assert [s.channels for s in info.audio] == [1, 2]
    expected = "pcm_s16le" if policy == "pcm" else "aac"
    assert all(s.codec == expected for s in info.audio)
    assert all(p["action"] == "transcode" for p in manifest["audio_tracks"])
    with av.open(str(target)) as container:
        for track in info.audio:
            with av.open(str(target)) as reopened:
                frames = list(reopened.decode(reopened.streams[track.index]))
                assert sum(f.samples for f in frames) >= 14000


def test_explicit_cfr_records_real_frame_mapping(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", vfr=True, audio=1)
    manifest = encode_video(source, "cfr.mov", root=tmp_path, hardware="software", timing="explicit_cfr", rate=Fraction(25))
    target = tmp_path / "roop_3d/outputs/cfr.mov"
    assert frame_times(target) == [Fraction(i, 25) for i in range(12)]
    assert sum(e["action"] == "duplicate" for e in manifest["frame_map"]) == 4
    assert audio_payloads(source) == audio_payloads(target)


def test_trim_reencodes_unaligned_audio_and_preserves_sync(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=2)
    manifest = encode_video(source, "trim.mov", root=tmp_path, hardware="software", audio_policy="pcm",
                            start=Fraction(1, 25), end=Fraction(6, 25))
    target = tmp_path / "roop_3d/outputs/trim.mov"
    assert frame_times(target) == [Fraction(i, 25) for i in range(5)]
    assert all(p["action"] == "transcode" for p in manifest["audio_tracks"])
    for track in probe(target).audio:
        with av.open(str(target)) as reopened:
            frames = list(reopened.decode(reopened.streams[track.index]))
            assert frames[0].pts * frames[0].time_base == 0
            assert sum(f.samples for f in frames) == 9600


def test_cancellation_and_transform_failure_clean_temporary_outputs(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=2)
    event = Event()
    def cancel_during_frame(frame):
        event.set()
        return frame.image
    with pytest.raises(DomainError) as error:
        encode_video(source, "cancel.mov", root=tmp_path, hardware="software", cancel=event, transform=cancel_during_frame)
    assert error.value.code == "MEDIA_CANCELLED"
    output = tmp_path / "roop_3d/outputs"
    assert not [p for p in output.iterdir() if p.suffix != ".lock"]
    event.clear()
    assert encode_video(source, "cancel.mov", root=tmp_path, hardware="software")["verified_frames"] == 8
    def fail(frame):
        raise RuntimeError("fixture transform failure")
    with pytest.raises(RuntimeError, match="fixture transform"):
        encode_video(source, "failed.mov", root=tmp_path, hardware="software", transform=fail)
    assert not (output / "failed.mov").exists()
    assert not list(output.glob("*.video.*")) and not list(output.glob("*.mux.*"))


def test_budgets_and_audio_drop_fail_closed(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=1)
    with pytest.raises(DomainError) as error:
        list(VideoReader(source, hardware="software", max_frame_bytes=100).frames())
    assert error.value.code == "FRAME_MEMORY_BUDGET"
    with pytest.raises(DomainError, match="every audio"):
        plan_audio(source, ".mov", "drop")
    with pytest.raises(ValueError):
        encode_video(source, "../outside.mp4", root=tmp_path)


def test_limited_probe_does_not_claim_cfr(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov")
    video = probe(source, sample_frames=3).videos[0]
    assert video.vfr is None and not video.sample_complete


@pytest.mark.parametrize("timing", ["preserve_source_timing", "explicit_cfr"])
def test_fractional_ntsc_rate_retains_exact_ticks(tmp_path, timing):
    source = make_video(tmp_path / "roop_3d/ntsc.mp4", rate=Fraction(30000, 1001))
    info = probe(source)
    assert info.videos[0].average_rate == Fraction(30000, 1001)
    manifest = encode_video(source, "ntsc.mp4", root=tmp_path, hardware="software", timing=timing,
                            rate=Fraction(30000, 1001) if timing == "explicit_cfr" else None)
    assert manifest["verified_frames"] == 90
    assert frame_times(source) == frame_times(tmp_path / "roop_3d/outputs/ntsc.mp4")


def test_hevc_software_output_is_really_decodable(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", vfr=True)
    manifest = encode_video(source, "hevc.mov", root=tmp_path, hardware="software", codec="hevc")
    target = tmp_path / "roop_3d/outputs/hevc.mov"
    assert manifest["encoder"] == "libx265" and probe(target).videos[0].codec == "hevc"
    assert frame_times(source) == frame_times(target)


def test_packet_aligned_audio_trim_uses_copy(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=1)
    # MOV's PCM packets have sample-aligned boundaries which we obtain from real packets.
    with av.open(str(source)) as container:
        packets = [p for p in container.demux(audio=0) if p.size and p.pts is not None]
        start = packets[1].pts * Fraction(packets[1].time_base)
        end = (packets[-2].pts + packets[-2].duration) * Fraction(packets[-2].time_base)
    assert start < end
    plans = plan_audio(source, ".mov", start=start, end=end)
    assert plans[0].action == "copy"
    manifest = encode_video(source, "aligned.mov", root=tmp_path, hardware="software", start=start, end=end)
    assert manifest["audio_tracks"][0]["action"] == "copy"
    target = tmp_path / "roop_3d/outputs/aligned.mov"
    with av.open(str(target)) as container:
        frames = list(container.decode(audio=0))
        assert frames[0].pts * frames[0].time_base == 0
        assert sum(f.samples for f in frames) == (end - start) * 48000


def test_reader_cancellation_releases_source_handle(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov")
    event = Event()
    iterator = VideoReader(source, hardware="software", cancel=event).frames()
    assert next(iterator).index == 0
    event.set()
    with pytest.raises(DomainError) as error:
        next(iterator)
    assert error.value.code == "MEDIA_CANCELLED"
    # Windows rename fails if an active decoder retains the input handle.
    source.rename(source.with_name("closed.mov"))


def test_external_output_root_rejected_before_writes(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov")
    from roop_3d.audit.isolation import AuditError
    from roop_3d.audit.project_paths import application_root
    # TEMP may legitimately live in this app's runtime. Use a root that is explicitly
    # outside the application root while remaining inside the checkout's hard boundary.
    with pytest.raises(AuditError, match="outside"):
        encode_video(source, "forbidden.mov", root=application_root().parent, hardware="software")


def test_media_service_preview_applies_rotation(tmp_path):
    from roop_3d.media.service import MediaService
    from roop_3d.models.catalog import ModelCatalog
    source = make_video(tmp_path / "roop_3d/.runtime/media/rotate.mov", rotation=90)
    service = MediaService(tmp_path, ModelCatalog())
    path = source.relative_to(tmp_path).as_posix()
    assert service.probe(path).rotation_degrees == 90
    assert service.frame(path, None, 0).shape == (96, 64, 3)
    assert service.frame(path, 0.12, None).shape == (96, 64, 3)


def test_aac_copy_retains_priming_timing_and_encoded_payloads(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mp4", audio=2, audio_codec="aac")
    manifest = encode_video(source, "copy.mp4", root=tmp_path, hardware="software")
    target = tmp_path / "roop_3d/outputs/copy.mp4"
    assert all(p["action"] == "copy" for p in manifest["audio_tracks"])
    assert audio_payloads(source) == audio_payloads(target)
    assert [(a.start_time, a.duration, a.time_base) for a in probe(source).audio] == [
        (a.start_time, a.duration, a.time_base) for a in probe(target).audio]
    for track_index in (0, 1):
        with av.open(str(source)) as original, av.open(str(target)) as copied:
            original_frames = list(original.decode(audio=track_index))
            copied_frames = list(copied.decode(audio=track_index))
            assert sum(f.samples for f in original_frames) == sum(f.samples for f in copied_frames)


def test_unaligned_trim_copy_policy_explicitly_falls_back_to_aac(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", audio=2)
    start, end = Fraction(1, 25), Fraction(6, 25)
    assert all(p.action == "transcode" for p in plan_audio(source, ".mov", start=start, end=end))
    manifest = encode_video(source, "fallback.mov", root=tmp_path, hardware="software", start=start, end=end)
    assert all(p["codec"] == "aac" for p in manifest["audio_tracks"])
    target = tmp_path / "roop_3d/outputs/fallback.mov"
    assert frame_times(target) == [Fraction(i, 25) for i in range(5)]
    for audio in probe(target).audio:
        assert audio.start_time == 0 and audio.duration * audio.time_base == end - start


def test_cfr_high_source_rate_logs_dropped_indices(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", rate=Fraction(50))
    manifest = encode_video(source, "drop.mov", root=tmp_path, hardware="software", timing="explicit_cfr", rate=Fraction(25))
    drops = [e["source_index"] for e in manifest["frame_map"] if e["action"] == "drop"]
    assert drops == list(range(1, 90, 2))
    assert frame_times(tmp_path / "roop_3d/outputs/drop.mov") == [Fraction(i, 25) for i in range(45)]


def test_auto_hardware_standard_dimensions_with_rotation_and_audio(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov", size=(320, 180), rotation=90, vfr=True, audio=2)
    manifest = encode_video(source, "hardware.mov", root=tmp_path)
    target = tmp_path / "roop_3d/outputs/hardware.mov"
    assert (manifest["width"], manifest["height"]) == (180, 320)
    assert frame_times(source) == frame_times(target)
    assert audio_payloads(source) == audio_payloads(target)
    assert manifest["nvenc_state"] in ("VERIFIED_LOCAL", "UNSUPPORTED")
    assert manifest["encoder"] == ("h264_nvenc" if manifest["nvenc_state"] == "VERIFIED_LOCAL" else "libx264")
    software = list(VideoReader(source, hardware="software").frames())
    automatic = list(VideoReader(source).frames())
    assert [(f.timestamp, f.duration, f.key_frame) for f in automatic] == [(f.timestamp, f.duration, f.key_frame) for f in software]
    assert max(np.abs(a.image.astype(float) - b.image.astype(float)).mean() for a, b in zip(software, automatic)) < 3
    assert [f.timestamp for f in VideoReader(source).frames(keyframes_only=True)] == [f.timestamp for f in software if f.key_frame]


def test_unsupported_nvdec_codec_really_falls_back_to_software(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mkv", video_codec="ffv1")
    reader = VideoReader(source)
    frames = list(reader.frames())
    assert len(frames) == 8 and reader.backend == "software"
    assert reader.hardware_state == "UNSUPPORTED" and reader.fallback_reason
    assert [frame.timestamp.seconds for frame in frames] == frame_times(source)


def test_manifest_publication_failure_rolls_back_video_and_all_temps(tmp_path):
    source = make_video(tmp_path / "roop_3d/input.mov")
    manifest_path = tmp_path / "roop_3d/outputs/rollback.mov.manifest.json"
    def interfere(frame):
        if frame.index == 7:
            manifest_path.mkdir()
        return frame.image
    with pytest.raises(DomainError) as error:
        encode_video(source, "rollback.mov", root=tmp_path, hardware="software", transform=interfere)
    assert error.value.code == "MEDIA_ENCODE_FAILED"
    assert not (manifest_path.parent / "rollback.mov").exists()
    assert not list(manifest_path.parent.glob("*.video.*"))
    assert not list(manifest_path.parent.glob("*.mux.*"))
    assert not list(manifest_path.parent.glob("*.part"))
    manifest_path.rmdir()
    assert encode_video(source, "rollback.mov", root=tmp_path, hardware="software")["verified_frames"] == 8


def test_input_errors_and_invalid_transform_have_domain_envelopes(tmp_path):
    with pytest.raises(DomainError) as error:
        probe(tmp_path / "absent.mp4")
    assert error.value.code == "MEDIA_NOT_FOUND" and error.value.status == 404
    source = make_video(tmp_path / "roop_3d/input.mov")
    with pytest.raises(DomainError) as error:
        encode_video(source, "bad.mov", root=tmp_path, hardware="software", transform=lambda _: "invalid")
    assert error.value.code == "ENCODE_FRAME_INVALID"
    assert not (tmp_path / "roop_3d/outputs/bad.mov").exists()
    with pytest.raises(ValueError):
        Timestamp(1, 0.1)
    with pytest.raises(ValueError):
        rescale(0.1, Fraction(1), Fraction(1))
