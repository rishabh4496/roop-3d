"""Stage 15: input containers, still sequences (JPEG/EXR), validation, capabilities, API, OCIO, audio-sync extension."""
from __future__ import annotations

from fractions import Fraction
import shutil
import subprocess

import av
import numpy as np
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.engine.audio_sync import choose_track, decode_audio, decode_selected, frame_sample_index, window_for_frame
from roop_3d.media.capabilities import media_capabilities, selftest_profile
from roop_3d.media.export_profiles import PROFILES
from roop_3d.media.probe import probe
from roop_3d.media.sequence import SequenceReader, discover_sequence
from roop_3d.media.validation import validate_media
from roop_3d.media.video_reader import VideoReader
from roop_3d.media.video_writer import encode_video
from roop_3d.tests.stage15_media import expected_bgr8, make_color_video, read_stripes, stripe_rgb


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "roop_3d").mkdir(exist_ok=True)
    return tmp_path


def encode(root, source, name, **kwargs):
    kwargs.setdefault("hardware", "software")
    return encode_video(source, name, root=root, **kwargs)


# --- input containers ------------------------------------------------------------------------------------------
CONTAINERS = [(".mp4", "libx264", "aac", "mov,mp4,m4a,3gp,3g2,mj2"), (".mov", "libx264", "pcm_s16le", "mov,mp4,m4a,3gp,3g2,mj2"),
              (".mkv", "libx264", "pcm_s16le", "matroska,webm"), (".webm", "libvpx-vp9", "libopus", "matroska,webm")]


@pytest.mark.parametrize("suffix,vcodec,acodec,container", CONTAINERS)
def test_input_containers(root, suffix, vcodec, acodec, container):
    source = make_color_video(root / f"roop_3d/in{suffix}", frames=50, audio=1, vcodec=vcodec, acodec=acodec,
                              width=96, height=64, lossless=False)
    info = probe(source)
    assert info.container == container and info.videos[0].codec in ("h264", "vp9") and len(info.audio) == 1
    report = validate_media(source, deep=True)
    assert report["ok"] and report["deep"]["frames_decoded"] == 50 and report["deep"]["audio_errors"] == 0
    manifest = encode(root, source, "out.mp4", transform=lambda f: f.image)
    assert manifest["verified_frames"] == 50
    assert manifest["stage15"]["audio_sync"][0]["ok"]
    image = next(iter(VideoReader(root / manifest["output_path"], hardware="software").frames())).image
    got = np.array([image[32, x] for x in range(6, 96, 12)]).astype(int)
    assert np.abs(got - expected_bgr8()).mean() <= 6


def test_webm_tagged_pq_takes_the_hdr_path(root):
    # This build's libvpx has no high-bit-depth encoder (its pixel formats stop at 8 bits), so a 10-bit VP9
    # fixture cannot be made here. An 8-bit PQ-tagged WebM exercises the same tag-driven HDR handling.
    assert "yuv420p10le" not in [f.name for f in av.codec.Codec("libvpx-vp9", "w").video_formats]
    source = make_color_video(root / "roop_3d/hdr.webm", matrix=9, primaries=9, trc=16, bits=8, frames=3,
                              vcodec="libvpx-vp9", width=64, height=32)
    info = probe(source).videos[0]
    assert info.color.dynamic_range == "HDR_PQ" and info.codec == "vp9"
    manifest = encode(root, source, "sdr.mp4", transform=lambda f: f.image)
    assert manifest["stage15"]["color"]["claims"]["hdr_output"] == "NOT_PRODUCED_TONE_MAPPED_TO_SDR"
    assert probe(root / manifest["output_path"]).videos[0].color.dynamic_range == "SDR"


# --- still sequences: JPEG and EXR ---------------------------------------------------------------------------------
def test_jpeg_and_exr_sequences_are_read_with_the_documented_assumptions(root):
    directory = root / "roop_3d/stills"
    directory.mkdir()
    base = np.rint(stripe_rgb(64, 32) * 255).astype(np.uint8)
    for ext, codec, pix in ((".jpg", "mjpeg", "yuvj420p"), ):
        for i in range(3):
            frame = av.VideoFrame.from_ndarray(base, format="rgb24").reformat(format=pix)
            with av.open(str(directory / f"j_{i:04d}{ext}"), "w") as out:
                stream = out.add_stream(codec)
                stream.width, stream.height, stream.pix_fmt = 64, 32, pix
                stream.options = {"qscale:v": "1"}
                for p in stream.encode(frame):
                    out.mux(p)
                for p in stream.encode(None):
                    out.mux(p)
    info = discover_sequence(directory, Fraction(24))
    frames = list(SequenceReader(info).frames())
    assert len(frames) == 3 and frames[0].image.shape == (32, 64, 3)
    assert np.abs(read_stripes(frames[0].image) - expected_bgr8()).max() <= 8      # lossy but colour-correct

    exr = root / "roop_3d/linear"
    exr.mkdir()
    linear = np.zeros((32, 64, 3), np.float32)           # scene-linear 0.5 grey, 0.18 and 1.0
    linear[:, :32], linear[:, 32:] = 0.5, 0.18
    for i in range(2):
        planar = np.stack([linear[..., 1], linear[..., 2], linear[..., 0]], axis=-1)   # G, B, R planes
        frame = av.VideoFrame.from_ndarray(planar, format="gbrpf32le")
        with av.open(str(exr / f"x_{i:04d}.exr"), "w") as out:
            stream = out.add_stream("exr")
            stream.width, stream.height, stream.pix_fmt = 64, 32, "gbrpf32le"
            for p in stream.encode(frame):
                out.mux(p)
            for p in stream.encode(None):
                out.mux(p)
    einfo = discover_sequence(exr, Fraction(24))
    decoded = next(iter(SequenceReader(einfo).frames())).image
    from roop_3d.color import transfer
    assert abs(int(decoded[16, 8, 0]) - round(float(transfer.srgb_inverse_eotf(0.5)) * 255)) <= 1     # 0.5 linear -> sRGB 188
    assert abs(int(decoded[16, 48, 0]) - round(float(transfer.srgb_inverse_eotf(0.18)) * 255)) <= 1    # 0.18 linear -> sRGB 118
    assert SequenceReader(einfo).source_color.transfer == 8


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="creating a mov_text track needs the ffmpeg command-line tool")
def test_mov_text_subtitles_are_copied_into_mp4(root, tmp_path):
    ass = make_color_video(root / "roop_3d/ass.mkv", frames=25, subtitle=True, width=64, height=32)
    source = root / "roop_3d/text.mp4"
    done = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", str(ass), "-c:v", "copy", "-c:s", "mov_text", str(source)],
                          capture_output=True)
    if done.returncode:
        pytest.skip("this ffmpeg cannot convert ASS to mov_text")
    assert [s.codec for s in probe(source).subtitles] == ["mov_text"]
    manifest = encode(root, source, "out.mp4", transform=lambda f: f.image)
    assert manifest["stage15"]["subtitles"][0]["action"] == "copy"
    assert [s.codec for s in probe(root / manifest["output_path"]).subtitles] == ["mov_text"]


# --- input validation -----------------------------------------------------------------------------------------------
def codes_of(report):
    return {issue["code"] for issue in report["issues"]}


def test_validate_media_reports_what_it_finds_without_refusing_unusual_files(root):
    hdr = make_color_video(root / "roop_3d/hdr.mkv", matrix=9, primaries=9, trc=16, bits=10, frames=4, rotation=90)
    report = validate_media(hdr)
    assert report["ok"] and {"HDR_INPUT", "HDR_METADATA_ABSENT", "HIGH_BIT_DEPTH", "ROTATION_METADATA"} <= codes_of(report)
    untagged = make_color_video(root / "roop_3d/u.mkv", tagged=False, frames=3)
    report = validate_media(untagged)
    assert "COLOR_ASSUMED" in codes_of(report) and report["video"]["color"]["assumed_fields"]
    vfr = make_color_video(root / "roop_3d/v.mkv", frames=8, pts=(0, 40, 120, 160, 280, 320, 360, 440), time_base=Fraction(1, 1000))
    assert "VARIABLE_FRAME_RATE" in codes_of(validate_media(vfr))
    multi = make_color_video(root / "roop_3d/m.mkv", frames=25, audio=2, subtitle=True)
    report = validate_media(multi, deep=True)
    assert "MULTIPLE_AUDIO" in codes_of(report) and [s["codec"] for s in report["subtitles"]] == ["ssa"]
    assert report["deep"]["pts_order_violations"] == 0


def test_validate_media_flags_a_truncated_file(root):
    good = make_color_video(root / "roop_3d/ok.mp4", frames=50, vcodec="libx264", lossless=False, width=96, height=64)
    broken = root / "roop_3d/cut.mp4"
    data = good.read_bytes()
    broken.write_bytes(data[: len(data) // 3])
    with pytest.raises(DomainError):
        report = validate_media(broken, deep=True)
        assert not report["ok"] or report["deep"]["frames_decoded"] < 50
        raise DomainError("TRUNCATED", "detected", 422)


# --- capabilities, profile self-test, API -----------------------------------------------------------------------------
def test_capabilities_never_claim_what_the_pipeline_cannot_do():
    caps = media_capabilities()
    assert caps["processing_bit_depth"] == 8
    assert caps["color"]["bit_depth"]["face_processing"]["state"] == "UNSUPPORTED"
    assert caps["color"]["hdr"]["rendered_output"]["state"] == "UNSUPPORTED"
    assert caps["color"]["hdr"]["unmodified_passthrough"]["state"] == "VERIFIED_LOCAL"
    assert caps["audio"]["lip_sync_correction"]["state"] == "UNSUPPORTED" and caps["audio"]["lip_sync_correction"]["default"] == "off"
    assert caps["inputs"]["subtitles"]["transcode"]["state"] == "UNSUPPORTED"
    assert set(caps["export_profiles"]) == set(PROFILES) and caps["encoders"]["libx264"] is True
    assert caps["color"]["ocio"]["default_path"] is False


@pytest.mark.parametrize("profile_id", sorted(PROFILES))
def test_every_export_profile_self_verifies_on_this_machine(root, profile_id):
    result = selftest_profile(PROFILES[profile_id], root / "roop_3d/selftest")
    assert result["state"] == "VERIFIED_LOCAL", result
    assert "rendered_frames" in result["checks"]
    assert ("hdr_passthrough" in result["checks"]) == PROFILES[profile_id].hdr_passthrough


def test_media_api_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from roop_3d.core.settings import Settings
    from roop_3d.server.app import create_app
    app = create_app(Settings(port=8030), repo_root=tmp_path, worker_enabled=False)
    headers = {"Origin": "http://127.0.0.1:8030"}
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        caps = client.get("/api/v1/media/capabilities", headers=headers)
        assert caps.status_code == 200 and caps.json()["processing_bit_depth"] == 8
        listing = client.get("/api/v1/media/export-profiles", headers=headers).json()
        assert [p["id"] for p in listing["profiles"]][0] == "standard_mp4" and listing["default"] == "standard_mp4"
        assert all(p["verification"]["state"] == "UNKNOWN" for p in listing["profiles"])      # not verified until asked
        verified = client.get("/api/v1/media/export-profiles?verify=true", headers=headers).json()
        assert all(p["verification"]["state"] == "VERIFIED_LOCAL" for p in verified["profiles"])
        media = tmp_path / "roop_3d/.runtime/media"
        media.mkdir(parents=True)
        make_color_video(media / "clip.mkv", matrix=9, primaries=9, trc=16, bits=10, frames=3)
        result = client.post("/api/v1/media/validate", json={"path": "roop_3d/.runtime/media/clip.mkv", "deep": True}, headers=headers)
        assert result.status_code == 200 and result.json()["video"]["color"]["dynamic_range"] == "HDR_PQ"
        probed = client.post("/api/v1/media/probe", json={"path": "roop_3d/.runtime/media/clip.mkv"}, headers=headers).json()
        assert probed["dynamic_range"] == "HDR_PQ" and probed["color_transfer"] == "smpte2084" and probed["bit_depth"] == 10
        assert probed["color_primaries"] == "bt2020" and probed["color_matrix"] == "bt2020nc" and probed["color_assumed_fields"] == []
        denied = client.post("/api/v1/media/validate", json={"path": "../outside.mkv"}, headers=headers)
        assert denied.status_code >= 400


def test_job_export_options_validate_and_keep_old_fingerprints():
    from roop_3d.jobs.schema import JobRequest
    from roop_3d.media.finishing import ExportOptions
    base = dict(target_path="roop_3d/.runtime/media/x.mp4", inspection_id="0" * 32,
                face_mappings=[{"track_id": "track_0001", "faceset_id": "a"}])
    before = JobRequest(**base, export=ExportOptions(passes=("alpha",), baked_video=True)).identity_payload()
    assert before["export"].keys() == {"baked_video", "passes", "finishing"}
    chosen = JobRequest(**base, export=ExportOptions(profile="archival_hevc10", hdr_policy="reject")).identity_payload()
    assert chosen["export"]["profile"] == "archival_hevc10" and chosen["export"]["hdr_policy"] == "reject"
    with pytest.raises(ValueError):
        ExportOptions(profile="mystery")
    with pytest.raises(ValueError):
        ExportOptions(baked_video=False, passes=("alpha",), profile="standard_mp4")
    with pytest.raises(ValueError):
        ExportOptions(hdr_policy="passthrough")   # the 8-bit render pipeline can never pass HDR through


# --- OCIO: optional, explicit, built-in configs only ------------------------------------------------------------------------
def test_ocio_adapter_is_optional_and_refuses_config_files():
    from roop_3d.color import ocio
    assert ocio.capability()["default_path"] is False
    with pytest.raises(DomainError) as error:
        ocio.OcioTransform("a", "b", config_uri="C:/configs/studio.ocio")
    assert error.value.code == "OCIO_CONFIG_DENIED"
    if not ocio.is_available():
        assert ocio.capability()["state"] == "UNSUPPORTED"
        with pytest.raises(DomainError) as unavailable:
            ocio.OcioTransform("a", "b").apply_float(np.zeros((1, 3)))
        assert unavailable.value.code == "OCIO_UNAVAILABLE"


def test_native_colour_maths_agrees_with_opencolorio():
    pytest.importorskip("PyOpenColorIO")
    from roop_3d.color import ocio
    result = ocio.cross_validate()
    limits = {"srgb_eotf": 1e-4, "bt1886_gamma24": 1e-4, "gamma22": 1e-4, "bt2020_to_bt709_matrix": 1e-5,
              "bt709_to_bt2020_matrix": 1e-5, "pq_inverse_eotf": 1e-5, "hlg_oetf_with_inverse_ootf": 1e-5,
              "bt709_oetf_rounded_constants": 5e-4}
    assert set(result) == set(limits)
    for name, limit in limits.items():
        assert result[name] <= limit, (name, result[name])
    transform = ocio.OcioTransform("sRGB Encoded Rec.709 (sRGB)", "Linear Rec.709 (sRGB)")
    assert transform.fingerprint()["config"] == "ocio://studio-config-latest"
    white = transform.apply_bgr8(np.full((2, 2, 3), 255, np.uint8))
    assert white.shape == (2, 2, 3) and white.dtype == np.uint8 and int(white.min()) == 255


# --- audio-sync architecture extension ---------------------------------------------------------------------------------------
def test_audio_start_time_comes_from_the_first_decoded_sample_not_from_container_metadata(root):
    source = root / "roop_3d/late.mkv"
    with av.open(str(source), "w") as out:
        video = out.add_stream("libx264", rate=25)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        for i in range(50):
            frame = av.VideoFrame(64, 48, "yuv420p")
            frame.pts, frame.time_base = i, Fraction(1, 25)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        tone = (np.sin(np.arange(48000) / 5) * 8000).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(tone, format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 48000, 12000, Fraction(1, 48000)   # audio begins at 0.25 s
        for packet in audio.encode(frame):
            out.mux(packet)
        for packet in audio.encode(None):
            out.mux(packet)
    signal = decode_audio(source)
    assert signal.start_time == Fraction(1, 4)
    # A video frame at t = 0.25 s sits exactly on the first audio sample; one at 0.5 s is 0.25 s (4000 samples @16 kHz) in.
    assert frame_sample_index(250, Fraction(1, 1000), 16_000, audio_start_time=signal.start_time) == 0
    assert frame_sample_index(500, Fraction(1, 1000), 16_000, audio_start_time=signal.start_time) == 4000


def test_track_selection_is_explicit_when_a_file_has_several_audio_tracks(root):
    multi = make_color_video(root / "roop_3d/m.mkv", frames=50, audio=2)
    tracks = probe(multi).audio
    with pytest.raises(DomainError) as error:
        choose_track(tracks)
    assert error.value.code == "AUDIO_TRACK_AMBIGUOUS"
    assert choose_track(tracks, language="hin").stream_index == tracks[1].index
    assert choose_track(tracks, stream_index=tracks[0].index).reason == "selected by stream index"
    with pytest.raises(DomainError):
        choose_track(tracks, language="deu")
    with pytest.raises(DomainError):
        choose_track(tracks, stream_index=tracks[0].index, language="eng")
    choice, signal = decode_selected(multi, language="eng")
    assert choice.stream_index == tracks[0].index and signal.samples.size > 0
    single = make_color_video(root / "roop_3d/s.mkv", frames=50, audio=1)
    assert decode_selected(single)[0].reason == "the only audio track"


def test_video_frame_to_audio_alignment_survives_vfr_and_trim_end_to_end(root):
    """Click at 0.5 s. After a 0.2 s trim and a sample-exact PCM transcode the click sits at 0.3 s in the export."""
    pts = tuple(int(i * 40) for i in range(50))
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=1, pts=pts, time_base=Fraction(1, 1000), width=96, height=64)
    manifest = encode(root, source, "t.mov", profile="mov_workflow", start=Fraction(1, 5), audio_policy="pcm",
                      transform=lambda f: f.image)
    out = root / manifest["output_path"]
    assert manifest["stage15"]["audio_sync"][0]["action"] == "transcode"
    signal = decode_audio(out)
    with av.open(str(out)) as container:
        times = [f.pts * Fraction(f.time_base) for f in container.decode(video=0)]
    # Frame at 0.32 s: the click (0.30 s) is 20 ms = 320 samples at 16 kHz before the window centre (index 400).
    at = Fraction(8, 25)
    assert at in times
    window = window_for_frame(signal, at.numerator, Fraction(1, at.denominator), window_samples=801, left_context_samples=400)
    peak = int(np.argmax(np.abs(window.samples)))
    expected = 400 - 320
    assert expected - 2 <= peak <= expected + 18, (peak, expected)    # the click is 16 samples wide at 16 kHz
    # And the frame at exactly 0.28 s sees it 320 samples *after* the centre.
    early = window_for_frame(signal, 7, Fraction(1, 25), window_samples=801, left_context_samples=400)
    assert 400 + 320 - 2 <= int(np.argmax(np.abs(early.samples))) <= 400 + 320 + 18


def test_audio_is_identical_whatever_the_video_transform_does(root):
    """Lip-sync (or any pixel edit) can only change pixels: the audio path takes no pixels."""
    source = make_color_video(root / "roop_3d/in.mkv", frames=50, audio=2, width=96, height=64)
    plain = encode(root, source, "a.mov", profile="mov_workflow", transform=lambda f: f.image)
    heavy = encode(root, source, "b.mov", profile="mov_workflow", transform=lambda f: 255 - f.image)

    def payloads(manifest):
        out = root / manifest["output_path"]
        digests = []
        with av.open(str(out)) as container:
            import hashlib
            for stream in container.streams.audio:
                h = hashlib.sha256()
                with av.open(str(out)) as again:
                    for packet in again.demux(again.streams[stream.index]):
                        if packet.size:
                            h.update(bytes(packet))
                digests.append(h.hexdigest())
        return digests
    assert payloads(plain) == payloads(heavy) and len(payloads(plain)) == 2
    assert [s["lag_ms"] for s in plain["stage15"]["audio_sync"]] == [s["lag_ms"] for s in heavy["stage15"]["audio_sync"]]


def test_lip_sync_is_off_by_default_unsupported_and_cannot_touch_audio():
    from roop_3d.core.settings import Settings
    from roop_3d.engine.audio_sync import LipSyncAdapter
    assert Settings().lip_sync_mode == "off"
    capability = LipSyncAdapter().capability()
    assert capability["state"] == "UNSUPPORTED" and capability["default"] == "off"
    assert capability["audio_modification"] == "never"
