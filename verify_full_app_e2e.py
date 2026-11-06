"""Full-app end-to-end verification (Stage 6): backend boot, 3D reconstruction, frame + faces, all three render modes
with XSeg + GPEN / UltraMax on the GPU, and a muxed video render. Real models, real clip, no mocks.

What it checks (every check prints ok/FAIL; exit code 0 only when all pass):

1. Backend boot      - ``create_app`` starts and ``GET /api/v1/health`` answers; the GPU provider (CUDA / TensorRT) is present.
2. 3D reconstruction - a real multi-view FaceSet -> ``reconstruct_faceset``: 96-bin sphere coverage, 2.5D + full 3D meshes.
3. Frame + faces     - a frame from the real target clip, detected face boxes by the production tracker.
4. Render passes     - ``standard_2d`` (render_mode standard), ``estimated_2_5d`` (pose_3d) and ``full_3d_reconstructed``
                       (reenactment_3d), each with the robust XSeg + occluder + BiSeNet matte and each restorer
                       (GPEN Realistic, UltraMax). The composite must differ from the input, keep the frame, and the XSeg
                       network must actually have run (compositor capability), never silently skipped.
5. Video render      - ``encode_video`` over a short span through the full per-frame pipeline: h264_nvenc when the GPU
                       encoder is usable (libx264 fallback is reported, not hidden), original audio muxed, output non-empty.

Run:  app/roop_3d/.venv/Scripts/python.exe verify_full_app_e2e.py [clip.mp4] [faceset.fsz]
Outputs go to app/roop_3d/.runtime/e2e_verify/.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import sys
import time
import traceback

import av
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.compositor.pipeline import CompositeOptions
from roop_3d.compositor.restoration import GpenRestorer, UltraMaxPipelineRestorer
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.media.video_writer import encode_video
from roop_3d.models.catalog import ModelCatalog
from roop_3d.reconstruction.faceset_3d import TOTAL_BINS, reconstruct_faceset
from roop_3d.reconstruction.identity_asset import IdentityAssetStore

DEFAULT_CLIP = APP / "roop_3d/.runtime/media/03de3642d8d64b22bee7ffe543fa2966.mp4"
DEFAULT_FACESET = legacy_root() / "app/facesets/ashna.fsz"
OUT = APP / "roop_3d/.runtime/e2e_verify"
START_FRAME, WARMUP, VIDEO_SECONDS = 600, 24, 2.0
VIDEO_FRAMES = 40   # lower bound of frames that must pass through the pipeline (2 s at 30 fps, minus GOP slack)

# prompt name -> engine render_mode (the saved parameter value)
MODES = {"standard_2d": "standard", "estimated_2_5d": "pose_3d", "full_3d_reconstructed": "reenactment_3d"}
RESTORERS = {"gpen_realistic": GpenRestorer, "ultramax": UltraMaxPipelineRestorer}
failures: list[str] = []


def check(condition: bool, message: str) -> bool:
    print(("  ok   " if condition else "  FAIL ") + message)
    if not condition:
        failures.append(message)
    return bool(condition)


def read_frames(path: Path, start: int, count: int):
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, image = capture.read()
        if not ok:
            break
        frames.append(image)
    capture.release()
    return frames, fps


def boot_backend() -> None:
    print("== 1. unified backend boot")
    import tempfile
    from fastapi.testclient import TestClient
    from roop_3d.server.app import create_app
    scratch = Path(tempfile.mkdtemp(prefix="e2e_boot_"))
    (scratch / "roop_3d/.runtime").mkdir(parents=True)
    app = create_app(repo_root=scratch, worker_enabled=False)
    with TestClient(app, base_url=f"http://127.0.0.1:{app.state.settings.port}") as client:
        response = client.get("/api/v1/health")
        check(response.status_code == 200, f"GET /api/v1/health -> {response.status_code} {str(response.json())[:90]}")
    import onnxruntime as ort
    providers = ort.get_available_providers()
    check("CUDAExecutionProvider" in providers or "TensorrtExecutionProvider" in providers, f"GPU execution provider available: {providers}")


def reconstruction(document) -> None:
    print("== 2. 3D faceset reconstruction")
    payload = reconstruct_faceset(document)
    check(payload["sphere_coverage_total"] == TOTAL_BINS == 96, f"sphere coverage total = {payload['sphere_coverage_total']} (96 bins)")
    check(payload["sphere_coverage_count"] >= 1, f"{payload['sphere_coverage_count']}/96 bins observed from {payload['observed_views']} view(s)")
    for key in ("mesh_2_5d", "mesh_full_3d"):
        mesh = payload[key]
        check(len(mesh["vertices"]) > 0 and len(mesh["indices"]) > 0 and len(mesh["uvs"]) == len(mesh["vertices"]),
              f"{key}: {len(mesh['vertices'])} vertices, {len(mesh['indices']) } index entries, UVs match")


def render_pass(document, frames, fps, mode_name, render_mode, restorer_name, restorer_class):
    label = f"{mode_name} + XSeg + {restorer_name}"
    session = ReferencePipeline()
    try:
        session.prepare_session(document, retarget="off", render_mode=render_mode,
                                identity_store=IdentityAssetStore(OUT / "assets"))
        session.compositor.restorer = restorer_class(session.catalog, session.config)
        options = CompositeOptions(occlusion_mode="robust", restoration_blend=0.7, feather_px=5.0)
        last = None
        for index, image in enumerate(frames):
            session.set_frame_timestamp(index / fps)
            result = session.infer_frame(image)
            if index == len(frames) - 1:
                last = (image, result, session.composite_frame(image, result, options=options))
            else:
                session.composite_frame(image, result, options=options)   # keeps temporal state identical to a render
        image, result, output = last
        check(bool(result.replacements), f"{label}: {len(result.replacements)} face(s) swapped")
        check(output.shape == image.shape and output.dtype == np.uint8, f"{label}: output keeps the frame ({output.shape[1]}x{output.shape[0]})")
        changed = float(np.abs(output.astype(np.int16) - image.astype(np.int16)).mean())
        check(changed > 0.05, f"{label}: composite differs from the input (mean abs diff {changed:.2f})")
        for replacement in result.replacements:
            note = f"{replacement.method} / mode {replacement.render_mode}" + (f" (fallback: {replacement.fallback_reason})" if replacement.fallback_reason else "")
            print(f"       track {replacement.track_id}: {note}")
        capability = session.compositor.capability()
        text = str(capability).lower()
        check("xseg" in text, f"{label}: XSeg network is part of the compositor capability")
        restorer_cap = session.compositor.restorer.capability()
        check(restorer_cap.get("inference") in ("VERIFIED_THIS_SESSION", "VERIFIED_LOCAL"), f"{label}: restorer inference ran ({restorer_cap})")
        path = OUT / f"{mode_name}__{restorer_name}.png"
        cv2.imwrite(str(path), output)
        check(path.is_file() and path.stat().st_size > 1000, f"{label}: wrote {path.name} ({path.stat().st_size // 1024} KiB)")
        return output
    finally:
        session.release_session()


def cut_segment(clip: Path, start_seconds: float, seconds: float, out: Path) -> Path:
    """Stream-copy (no re-encode) ``seconds`` of ``clip`` from the keyframe at/before ``start_seconds``."""
    with av.open(str(clip)) as source, av.open(str(out), "w") as target:
        mapping = {s.index: target.add_stream_from_template(s) for s in source.streams if s.type in ("video", "audio")}
        source.seek(int(start_seconds * av.time_base))
        begin = None
        for packet in source.demux():
            if packet.dts is None or packet.stream.index not in mapping:
                continue
            seconds_at = float(packet.dts * packet.stream.time_base)
            if packet.stream.type == "video":
                if begin is None:
                    begin = seconds_at
                if seconds_at - begin >= seconds:
                    break
            elif begin is None or seconds_at < begin:
                continue
            packet.stream = mapping[packet.stream.index]
            target.mux(packet)
    return out


def video_render(document, clip: Path, fps: float) -> None:
    print("== 5. video render (swap -> robust matte -> UltraMax -> encode -> audio mux)")
    session = ReferencePipeline()
    session.prepare_session(document, retarget="off", render_mode="standard")
    session.compositor.restorer = UltraMaxPipelineRestorer(session.catalog, session.config)
    options = CompositeOptions(occlusion_mode="robust", restoration_blend=0.7)
    count = {"n": 0}

    def transform(frame):
        session.set_frame_timestamp(float(frame.timestamp.seconds))
        result = session.infer_frame(frame.image)
        count["n"] += 1
        return session.composite_frame(frame.image, result, options=options)

    name = f"e2e_{int(time.time())}.mp4"
    segment = cut_segment(clip, START_FRAME / fps, VIDEO_SECONDS, OUT / f"segment_{int(time.time())}.mp4")
    try:
        manifest = encode_video(segment, name, root=application_root(), hardware="auto",
                                audio_policy="copy", transform=transform, output_dir="roop_3d/.runtime/e2e_verify")
    finally:
        session.release_session()
    output = APP / "roop_3d/.runtime/e2e_verify" / name
    check(output.is_file() and output.stat().st_size > 10_000, f"output exists and is non-empty: {output.name} ({output.stat().st_size // 1024} KiB)")
    check(count["n"] >= VIDEO_FRAMES - 2, f"{count['n']} frames went through the full pipeline")
    print(f"       encoder = {manifest.get('encoder')}  nvenc_state = {manifest.get('nvenc_state')}  decoder = {manifest.get('decoder')}")
    check(manifest.get("encoder") in ("h264_nvenc", "hevc_nvenc", "libx264", "libx265"), f"encoder is one of the planned codecs: {manifest.get('encoder')}")
    with av.open(str(segment)) as source, av.open(str(output)) as result:
        check(len(result.streams.video) == 1 and result.streams.video[0].frames > 0, f"output video stream has {result.streams.video[0].frames} frames")
        if source.streams.audio:
            check(len(result.streams.audio) == 1, "original audio stream muxed into the output")
        else:
            print("       (source clip has no audio stream)")


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video_only = "--video-only" in sys.argv   # skip step 4 while iterating on the encode step
    clip = Path(args[0]) if args else DEFAULT_CLIP
    faceset = Path(args[1]) if len(args) > 1 else DEFAULT_FACESET
    for path in (clip, faceset):
        if not path.is_file():
            print(f"missing input: {path}")
            return 2
    OUT.mkdir(parents=True, exist_ok=True)
    began = time.perf_counter()
    boot_backend()
    document = FaceSetReader().read(faceset)
    reconstruction(document)

    print("== 3. frame + face boxes from the target clip")
    frames, fps = read_frames(clip, START_FRAME, WARMUP)
    check(len(frames) == WARMUP, f"decoded {len(frames)} frames from {clip.name} at {fps:.2f} fps")
    probe = ReferencePipeline()
    probe.prepare_session(document, retarget="off")
    probe.set_frame_timestamp(0.0)
    found = probe.infer_frame(frames[0])
    probe.release_session()
    check(bool(found.observations), f"{len(found.observations)} face(s) detected on the extracted frame")
    height, width = frames[0].shape[:2]
    for item in found.observations:
        x1, y1, x2, y2 = (float(v) for v in item.observation.box[:4])
        check(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, f"face box ({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}) lies inside {width}x{height}")

    print("== 4. render passes: 3 modes x 2 restorers, robust XSeg matte, GPU")
    for mode_name, render_mode in ({} if video_only else MODES).items():
        for restorer_name, restorer_class in RESTORERS.items():
            render_pass(document, frames, fps, mode_name, render_mode, restorer_name, restorer_class)

    video_render(document, clip, fps)
    print(f"\n{len(failures)} failure(s) in {time.perf_counter() - began:.0f} s")
    for message in failures:
        print("  FAILED:", message)
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)
