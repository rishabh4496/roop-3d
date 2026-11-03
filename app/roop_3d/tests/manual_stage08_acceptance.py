"""Prepare owned, local-media browser fixtures or verify its real rendered output.

Run from app/: python -B roop_3d/tests/manual_stage08_acceptance.py --source ... --target-faceset ...
Then run the production Playwright campaign documented in STAGE08_OFFLINE_WORKFLOW.md.
No downloads, external services, legacy modules, or mock inference are used.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.audit.isolation import safe_path, sha256_file
from roop_3d.core.storage import atomic_json, owned
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.media.video_reader import VideoReader
from roop_3d.tests.test_stage15_e2e_golden import create_golden_video


BASE = "roop_3d/.runtime/stage08_acceptance"


def verify(root: Path) -> dict:
    evidence = json.loads(owned(root, BASE + "/browser/evidence.json").read_text("utf-8"))
    fixture = json.loads(owned(root, BASE + "/fixture.json").read_text("utf-8"))
    source = safe_path(root, fixture["target_path"])
    output = owned(root, evidence["output"]["output_path"])
    manifest = json.loads(owned(root, evidence["output"]["manifest_path"]).read_text("utf-8"))
    assert manifest["output_sha256"] == sha256_file(output)[1]
    assert manifest["stage12"]["identity_transfer"] == "VERIFIED_LOCAL"
    assert manifest["provenance"]["synthetic_media"] is True
    assert manifest["timing_policy"] == "preserve_source_timing"
    assert manifest["stage12"]["restoration"]["inference"] == "VERIFIED_LOCAL"
    assert len(manifest["model_hashes"]["restorer"]) == 64
    before = list(VideoReader(source, hardware="software").frames())
    after = list(VideoReader(output, hardware="software").frames())
    assert len(before) == len(after) == fixture["frames"]
    assert [f.timestamp.seconds for f in before] == [f.timestamp.seconds for f in after]
    background, face = [], []
    for left, right in zip(before, after):
        diff = np.abs(left.image.astype(float) - right.image.astype(float))
        background.append(float(diff[:32, :32].mean()))
        face.append(float(diff[160:350, 160:350].mean()))
    assert max(background) < 3, background
    assert max(face) > 5, face
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        assert video.duration * Fraction(video.time_base) == Fraction(fixture["frames"], 25)
        assert len(container.streams.audio) == 1
        assert next(container.decode(audio=0), None) is not None
    result = {"state": "PASS", "frames": len(after), "rational_pts_equal": True,
              "audio_tracks": 1, "background_mean_lsb_max": max(background), "face_mean_lsb_max": max(face),
              "actual_identity_inference": manifest["stage12"]["identity_transfer"],
              "restoration": manifest["stage12"]["restoration"],
              "identity_assets": manifest["stage12"].get("identity_assets", {}),
              "output_sha256": manifest["output_sha256"]}
    atomic_json(root, BASE + "/output_verification.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source")
    parser.add_argument("--target-faceset")
    parser.add_argument("--verify-output", action="store_true")
    args = parser.parse_args()
    root = application_root()
    if args.verify_output:
        print(json.dumps(verify(root), indent=2))
        return
    if not args.source or not args.target_faceset:
        parser.error("--source and --target-faceset must reference local owned .fsz files")
    source = safe_path(root, args.source)
    target = safe_path(root, args.target_faceset)
    # Validate both inputs through the real FaceSet reader and actual local recognition dependencies.
    reader = FaceSetReader(cache_root=owned(root, BASE + "/cache"))
    reader.read(source)
    document = reader.read(target)
    view = min(document.views, key=lambda item: abs(item.yaw) + abs(item.pitch))
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert image is not None
    relative = BASE + "/target.mp4"
    create_golden_video(owned(root, relative), image, count=12)
    fixture = {"source_path": source.relative_to(root).as_posix(), "target_path": relative,
               "frames": 12, "source_sha256": sha256_file(source)[1], "target_sha256": sha256_file(target)[1]}
    atomic_json(root, BASE + "/fixture.json", fixture)
    print(json.dumps(fixture, indent=2))


if __name__ == "__main__":
    main()
