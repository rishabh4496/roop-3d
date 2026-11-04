"""What the media layer can really do on this machine, and the evidence for it.

Static facts are things the code implements and the Stage 15 tests verify; dynamic facts (encoder
availability, profile self-tests) are measured on the running installation. Every entry carries a state
using the project vocabulary: ``VERIFIED_LOCAL`` (exercised by a test or a self-test), ``UNSUPPORTED``
(not offered, with the reason) or ``UNKNOWN``. Nothing is ``VERIFIED_LOCAL`` unless something checked it.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import shutil
import uuid

import av
import numpy as np

from roop_3d.color import gamut
from roop_3d.core.errors import DomainError
from roop_3d.media.export_profiles import PROFILES, ExportProfile

ENCODERS = ("libx264", "libx265", "h264_nvenc", "hevc_nvenc", "prores_ks", "ffv1", "png", "tiff", "dpx", "libvpx-vp9")
_SELFTEST_CACHE: dict[tuple, dict] = {}


def available_encoders() -> dict[str, bool]:
    result = {}
    for name in ENCODERS:
        try:
            av.codec.Codec(name, "w")
            result[name] = True
        except av.FFmpegError:
            result[name] = False
    return result


def media_capabilities() -> dict:
    encoders = available_encoders()
    return {
        "schema_version": 1,
        "inputs": {
            "containers": {name: {"state": "VERIFIED_LOCAL", "evidence": "tests/test_stage15_media_extras.py::test_input_containers"}
                           for name in ("mp4", "mov", "mkv", "webm")}
                          | {"webm_10_bit_vp9": {"state": "UNKNOWN",
                                                 "reason": "the decode path is shared with other 10-bit codecs, but this build's libvpx "
                                                           "has no high-bit-depth encoder, so no fixture could be made to verify it"}},
            "image_sequences": {name: {"state": "VERIFIED_LOCAL", "bit_depth": depth}
                                for name, depth in (("png", "8/16 -> 8"), ("tiff", "8/16 -> 8"), ("dpx", "8/16 -> 8"))}
                               | {"jpeg": {"state": "VERIFIED_LOCAL", "bit_depth": "8", "note": "lossy"},
                                  "exr": {"state": "VERIFIED_LOCAL", "bit_depth": "float -> 8",
                                          "note": "treated as scene-linear and sRGB-encoded; stated in the manifest"}},
            "timing": {"cfr": "VERIFIED_LOCAL", "vfr": "VERIFIED_LOCAL", "rational_pts": "VERIFIED_LOCAL",
                       "image_sequence_clock": "requires an explicit rational frame rate"},
            "rotation_metadata": "VERIFIED_LOCAL",
            "multiple_audio_tracks": "VERIFIED_LOCAL",
            "subtitles": {"text_and_bitmap_probe": "VERIFIED_LOCAL",
                          "copy": {"verified": {"mkv": ["ssa/ass"]},
                                   "implemented_not_fixtured": {"mkv": ["subrip", "webvtt", "dvd_subtitle", "hdmv_pgs_subtitle", "dvb_subtitle"],
                                                                "mp4/mov": ["mov_text (verified only where the ffmpeg CLI is present)"]}},
                          "transcode": {"state": "UNSUPPORTED",
                                        "reason": "this PyAV build cannot open the text subtitle encoders; unsupported tracks are "
                                                  "reported as dropped (policy 'require' makes that an error)"}},
        },
        "color": {
            "ycbcr_matrices": {"bt709": "VERIFIED_LOCAL", "bt470bg/bt601": "VERIFIED_LOCAL", "smpte170m": "VERIFIED_LOCAL",
                               "bt2020nc": "VERIFIED_LOCAL", "rgb (identity)": "VERIFIED_LOCAL"},
            "primaries": {code: "supported" for code in gamut.SUPPORTED_PRIMARIES},
            "transfers": ["bt709", "smpte170m", "bt2020-10", "iec61966-2-1", "gamma22", "gamma28", "linear", "smpte2084", "arib-std-b67"],
            "range": {"limited": "VERIFIED_LOCAL", "full": "VERIFIED_LOCAL",
                      "policy": "source range is preserved and re-tagged; never converted implicitly"},
            "untagged_media": {"state": "VERIFIED_LOCAL",
                               "policy": "matrix/primaries/transfer/range resolved by documented convention and reported as assumptions; "
                                         "untagged material is never treated as HDR"},
            "hdr": {
                "input": {"state": "VERIFIED_LOCAL", "formats": ["PQ (SMPTE ST 2084)", "HLG (ARIB STD-B67)"],
                          "static_metadata": "MaxCLL / MaxFALL / mastering display parsed and reported"},
                "rendered_output": {"state": "UNSUPPORTED",
                                    "reason": "face swapping runs on 8-bit display-referred RGB; PQ/HLG input is tone-mapped to SDR "
                                              "BT.709 (BT.2390 EETF) or refused, and the manifest says so"},
                "unmodified_passthrough": {"state": "VERIFIED_LOCAL", "applies_to": "media with no transform",
                                           "profiles": [p.id for p in PROFILES.values() if p.hdr_passthrough]},
            },
            "bit_depth": {
                "decode": {"state": "VERIFIED_LOCAL", "verified": [8, 10]},
                "face_processing": {"state": "UNSUPPORTED", "bits": 8,
                                    "reason": "detection, swapping, restoration and compositing models and buffers are 8-bit"},
                "encode": {"state": "VERIFIED_LOCAL", "verified": [8, 10],
                           "note": "a 10-bit container reduces encode-stage quantisation; it does not add precision to rendered frames"},
                "unmodified_native_path": {"state": "VERIFIED_LOCAL", "note": "planes are re-encoded without an RGB round trip"},
            },
            "linear_light_compositing": {"state": "VERIFIED_LOCAL", "option": "WorkstationParameters.linear_light",
                                         "transfers": ["srgb", "bt1886", "gamma22", "gamma28", "bt709_oetf", "linear"],
                                         "scope": "alpha blending in feather, multi-band and Poisson modes; colour matching, "
                                                  "restoration and relighting stay in the encoded domain"},
            "ocio": ocio_capability(),
        },
        "audio": {
            "default": "every audio track is stream-copied when the container allows, else re-encoded; never dropped or altered by default",
            "integrity": "VERIFIED_LOCAL (payload SHA-256 for copied tracks)",
            "sync": "VERIFIED_LOCAL (signal-level lag measured by cross-correlation; sample-exact for copies)",
            "lip_sync_correction": {"state": "UNSUPPORTED", "default": "off",
                                    "reason": "no verified local weights / tensor contract are registered; the feature edits pixels "
                                              "only and cannot change audio even when enabled"},
        },
        "nvidia_stream_backend": {
            "state": "VERIFIED_LOCAL", "evidence": "tests/test_stage15_stream_color.py (software decode/encode; GPU decode was not exercisable here)",
            "supports": "tagged 8-bit 4:2:0 SDR (BT.709 / BT.601 / BT.2020), matrix, range and primaries/transfer tags carried exactly",
            "routed_to_reference_pipeline": ["PQ/HLG HDR", ">8-bit or non-4:2:0", "untagged matrix or range"],
            "defect_fixed": "the encoder used FFmpeg's untagged BT.601 default and dropped every colour tag (~41/255 off on tagged BT.709 HD)"},
        "encoders": encoders,
        "export_profiles": [profile.id for profile in PROFILES.values()],
        "processing_bit_depth": 8,
    }


def ocio_capability() -> dict:
    """OpenColorIO is an optional, explicitly selected adapter; it is never a dependency of the default path."""
    from roop_3d.color import ocio
    return ocio.capability()


# --- profile self-test ------------------------------------------------------------------------------------------
def _clip(path: Path, *, bits: int, pq: bool) -> Path:
    from roop_3d.color.gamut import rgb_to_ycbcr
    width, height, frames = 64, 48, 4
    pix = "yuv420p10le" if bits > 8 else "yuv420p"
    dtype = np.uint16 if bits > 8 else np.uint8
    matrix, primaries, trc = (9, 9, 16) if pq else (1, 1, 1)
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx265" if bits > 8 else "libx264", rate=Fraction(25))
        video.width, video.height, video.pix_fmt = width, height, pix
        ctx = video.codec_context
        ctx.time_base, ctx.max_b_frames = Fraction(1, 1000), 0
        ctx.colorspace, ctx.color_primaries, ctx.color_trc, ctx.color_range = matrix, primaries, trc, 1
        video.options = {"crf": "0"} if bits == 8 else {"x265-params": "lossless=1:log-level=none"}
        audio = out.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        rgb = np.zeros((height, width, 3))
        for i, value in enumerate(((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 0), (0, 0, 1), (.5, .5, .5), (1, 1, 0), (.2, .4, .8))):
            rgb[:, i * 8:(i + 1) * 8] = value
        ycc = np.clip(np.rint(rgb_to_ycbcr(rgb, matrix, False, bits)), 0, (1 << bits) - 1)
        for index in range(frames):
            frame = av.VideoFrame(width, height, pix)
            for plane, data in zip(frame.planes, (ycc[..., 0], ycc[::2, ::2, 1], ycc[::2, ::2, 2])):
                view = np.ndarray((data.shape[0], plane.line_size // np.dtype(dtype).itemsize), dtype, buffer=plane)
                view[:, :data.shape[1]] = data.astype(dtype)
            frame.pts, frame.time_base = index * 40, Fraction(1, 1000)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode(None):
            out.mux(packet)
        samples = (np.sin(np.arange(24000) / 7) * 6000).astype(np.int16).reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 48000, 0, Fraction(1, 48000)
        for packet in audio.encode(frame):
            out.mux(packet)
        for packet in audio.encode(None):
            out.mux(packet)
    return path


def selftest_profile(profile: ExportProfile, scratch: Path) -> dict:
    """Encode and fully validate a tiny clip with ``profile`` on this machine. ``scratch`` must be a writable runtime root."""
    from roop_3d.media.video_writer import encode_video
    key = (profile.id, av.__version__)
    if key in _SELFTEST_CACHE:
        return _SELFTEST_CACHE[key]
    root = scratch / uuid.uuid4().hex
    (root / "roop_3d").mkdir(parents=True)
    checks: dict[str, dict] = {}
    suffix = "" if profile.is_sequence else profile.container

    def run(label: str, source: Path, **kwargs) -> None:
        name = f"{label}{suffix}"
        try:
            manifest = encode_video(source, name, root=root, hardware="software", profile=profile.id, **kwargs)
            checks[label] = {"state": "VERIFIED_LOCAL", "frames": manifest["verified_frames"],
                             "pixel_format": manifest["pixel_format"], "fidelity_samples": len(manifest["stage15"]["fidelity"])}
        except DomainError as error:
            checks[label] = {"state": "UNSUPPORTED", "code": error.code, "reason": error.message}
        except (av.FFmpegError, OSError, ValueError) as error:
            checks[label] = {"state": "UNSUPPORTED", "code": type(error).__name__, "reason": str(error)[:160]}
    try:
        sdr = _clip(root / "roop_3d/sdr.mkv", bits=8, pq=False)
        run("rendered_frames", sdr, transform=lambda frame: frame.image)
        if profile.native_passthrough:
            run("unmodified_media", sdr)
        if profile.hdr_passthrough:
            pq = _clip(root / "roop_3d/pq.mkv", bits=10, pq=True)
            run("hdr_passthrough", pq, hdr_policy="passthrough")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    state = "VERIFIED_LOCAL" if checks and all(c["state"] == "VERIFIED_LOCAL" for c in checks.values()) else "UNSUPPORTED"
    result = {"state": state, "checks": checks, "encoder": profile.encoder}
    _SELFTEST_CACHE[key] = result
    return result
