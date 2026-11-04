"""Re-measure the Stage 15 media / colour claims on this machine and write ``MEDIA_COLOR_STAGE15_VALIDATION.json``.

Run from ``app/``:  ``roop_3d/.venv/Scripts/python.exe -m roop_3d.audit.stage15_media_validation``
Optional: set ``STAGE15_OCIO_PATH`` to a directory holding an isolated PyOpenColorIO install to include the
OpenColorIO cross-validation (the package is deliberately not a dependency of the project environment).
Encodes only throw-away clips under ``roop_3d/.runtime/stage15_validation``, which is removed afterwards.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import time

import av
import numpy as np

from roop_3d.audit.project_paths import application_root

#: Colour error of the PRE-Stage-15 writer, measured with the same oracle at the start of Stage 15 (per-stripe maxima;
#: the 8 stripes are black, white, red, green, blue, grey, yellow, muted blue). Kept as a record, not re-measurable.
BEFORE_STAGE15 = {
    "method": "tagged source -> encode_video (default settings) -> decode; max |error| per stripe in 8-bit code values",
    "bt709_limited": [0, 0, 0, 2, 4, 7, 6, 4], "bt709_full": [0, 0, 0, 1, 7, 4, 6, 1], "bt601_limited": [2, 2, 6, 3, 3, 5, 6, 9],
    "bt2020_sdr_10bit": [0, 2, 6, 33, 17, 14, 19, 7],
    "note": "BT.2020 was encoded with the BT.601 matrix but tagged BT.2020 (green off by 33/255); PQ/HLG were written as 8-bit "
            "yuv420p tagged PQ/HLG with SDR processing; untagged HD decoded with BT.601 (up to 23/255).",
    "stream_backend_709_hd": {"decode_error": 2, "after_encode_error": 41, "output_tags": "all unspecified"}}


def main() -> int:
    from roop_3d.color import transfer
    from roop_3d.color.hdr import HdrToSdr, ToneMapParams
    from roop_3d.media.capabilities import available_encoders, selftest_profile
    from roop_3d.media.export_profiles import PROFILES
    from roop_3d.media.probe import probe
    from roop_3d.media.video_reader import VideoReader
    from roop_3d.media.video_writer import encode_video
    from roop_3d.tests.stage15_media import expected_bgr8, make_color_video, read_stripes, stripe_rgb

    root = application_root()
    scratch = root / "roop_3d/.runtime/stage15_validation"
    shutil.rmtree(scratch, ignore_errors=True)
    (scratch / "roop_3d").mkdir(parents=True)
    report: dict = {"schema_version": 1, "generated_unix": int(time.time()), "python": sys.version.split()[0],
                    "pyav": av.__version__, "libraries": {k: ".".join(map(str, v)) for k, v in av.library_versions.items()},
                    "encoders": available_encoders(), "before_stage15": BEFORE_STAGE15}
    try:
        decode, roundtrip = {}, {}
        cases = [("bt709_limited", dict(matrix=1), 8), ("bt709_full", dict(matrix=1, full=True), 8),
                 ("bt601_limited", dict(matrix=5, primaries=5, trc=6), 8), ("bt2020_sdr_10bit", dict(matrix=9, primaries=9, trc=14), 10),
                 ("bt2020_sdr_10bit_full", dict(matrix=9, primaries=9, trc=14, full=True), 10)]
        for name, kw, bits in cases:
            src = make_color_video(scratch / f"roop_3d/{name}.mkv", frames=2, bits=bits, **kw)
            img = next(iter(VideoReader(src, hardware="software").frames())).image
            decode[name] = np.abs(read_stripes(img) - expected_bgr8()).max(axis=1).tolist()
            manifest = encode_video(src, f"{name}.mp4", root=scratch, hardware="software")
            out = scratch / manifest["output_path"]
            img = next(iter(VideoReader(out, hardware="software").frames())).image
            roundtrip[name] = np.abs(read_stripes(img) - expected_bgr8()).max(axis=1).tolist()
        report["decode_vs_oracle_max_error_per_stripe"] = decode
        report["encode_roundtrip_max_error_per_stripe"] = roundtrip

        hdr = {}
        for peak in (1000.0, 4000.0):
            converter = HdrToSdr(16, 9, ToneMapParams(source_peak_nits=peak))
            nits = np.array([0.5, 5, 18, 36.5, 100, 203, 400, 1000, 4000])
            gray = np.repeat(np.rint(transfer.pq_inverse_eotf(nits) * 65535).astype(np.uint16)[:, None, None], 3, axis=2)
            hdr[f"pq_peak_{int(peak)}"] = {"cd_m2": nits.tolist(), "sdr_linear": np.round(converter.to_relative_linear(gray)[:, 0, 0], 4).tolist(),
                                          "sdr_8bit_code": converter.convert(gray)[:, 0, 0].tolist(), "parameters": converter.describe()["parameters"]}
        report["hdr_to_sdr_tone_map"] = hdr
        report["published_constants"] = {
            "pq_203_nit": float(transfer.pq_inverse_eotf(203.0)), "pq_1000_nit": float(transfer.pq_inverse_eotf(1000.0)),
            "hlg_oetf_1_12": float(transfer.hlg_oetf(1 / 12)), "hlg_reference_white_nits": float(
                transfer.hlg_ootf(np.full(3, transfer.hlg_inverse_oetf(0.75)), 1000.0)[0])}

        report["export_profiles"] = {pid: selftest_profile(profile, scratch / "roop_3d/selftest") for pid, profile in PROFILES.items()}

        sync = {}
        src = make_color_video(scratch / "roop_3d/sync.mkv", frames=50, audio=2, width=96, height=64)
        for label, kwargs in (("stream_copy_mov", dict(profile="mov_workflow")), ("aac_transcode", dict(audio_policy="aac")),
                              ("pcm_transcode", dict(profile="mov_workflow", audio_policy="pcm")),
                              ("aac_transcode_trimmed_0.2s", dict(audio_policy="aac", start=__import__("fractions").Fraction(1, 5)))):
            suffix = ".mov" if kwargs.get("profile") == "mov_workflow" else ".mp4"
            manifest = encode_video(src, label + suffix, root=scratch, hardware="software", transform=lambda f: f.image, **kwargs)
            sync[label] = manifest["stage15"]["audio_sync"]
        report["audio_sync_measurements"] = sync

        ocio_path = os.environ.get("STAGE15_OCIO_PATH")
        if ocio_path:
            sys.path.insert(0, ocio_path)
        from roop_3d.color import ocio
        report["opencolorio"] = ({"version": ocio.capability()["version"], "max_abs_difference": ocio.cross_validate(),
                                  "configs": "built-in ocio:// only"} if ocio.is_available()
                                 else {"state": "not installed in this environment (optional)"})
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    target = root / "roop_3d/audit/MEDIA_COLOR_STAGE15_VALIDATION.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
