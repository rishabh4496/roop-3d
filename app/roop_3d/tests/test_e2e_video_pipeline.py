"""Stage 17 end-to-end: generated .fsz + 30-frame .mp4 -> import/upload routes -> real job -> download -> ffprobe / SSIM / identity.

Nothing in the render is mocked: SCRFD detection + 106 landmarks, ArcFace, ``inswapper_128`` swap, the BiSeNet / XSeg / occluder
parsers, the UltraMax restorer, the Laplacian composite, the FFmpeg decode and encode pipes and the audio remux all run for real
(CPU provider, so the result does not depend on a TensorRT engine cache; it takes a few minutes).

The inputs are the repo's consented FaceSet fixtures, as in ``test_stage15_e2e_golden``. The source ``.fsz`` is GENERATED here
(re-exported from a consented FaceSet under a new id) and sent through ``POST /api/v1/facesets/import``.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import subprocess
import threading
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.audit.project_paths import legacy_root
from roop_3d.core.settings import Settings
from roop_3d.faceset.faceset_reader import FaceSetReader, export_v2
from roop_3d.pipeline.video_job_runner import ENGINE_ID, probe_video, verify_output
from roop_3d.segment_dispatcher import find_ffprobe
from roop_3d.server.app import create_app
from roop_3d.telemetry.pinokio_terminal import PinokioTerminalHUD
from roop_3d.tests.model_governance_support import require_model_named, require_reference_models
from roop_3d.tests.test_stage15_e2e_golden import create_golden_video

FRAMES, FPS = 30, 25
STATUS_WS = "ws://127.0.0.1:8030/ws/v1/pipeline/status"


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean structural similarity of two BGR images (Wang et al. 2004; 11x11 Gaussian window, sigma 1.5, on luma)."""
    x = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda v: cv2.GaussianBlur(v, (11, 11), 1.5)                                  # noqa: E731
    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    return float((((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2))).mean())


def cosine(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def frames_of(path: Path) -> list[np.ndarray]:
    from roop_3d.pipeline.video_job_runner import FFmpegDecoder
    with FFmpegDecoder(probe_video(path)) as decoder:
        return [frame.copy() for frame in decoder.frames()]


def ffprobe_json(path: Path) -> dict:
    run = subprocess.run([find_ffprobe(), "-v", "error", "-print_format", "json", "-count_packets", "-show_streams", "-show_format", str(path)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def _save_qc_sheet(before: Path, after: Path, directory: Path) -> None:
    """Keep what the run produced for a human to look at (pytest's own temp dir is wiped by the next session): the result MP4 and a
    sheet with the original on top, the render below, at three moments. Lives in the gitignored runtime folder."""
    directory.mkdir(parents=True, exist_ok=True)
    originals, results = frames_of(before), frames_of(after)
    indices = [3, FRAMES // 2, FRAMES - 3]
    cv2.imwrite(str(directory / "before_after.png"), np.vstack([np.hstack([originals[i] for i in indices]), np.hstack([results[i] for i in indices])]))
    (directory / "result.mp4").write_bytes(after.read_bytes())


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory):
    """Run the whole pipeline once; every test below inspects a different aspect of the same real render."""
    tmp_path = tmp_path_factory.mktemp("e2e_photoreal")
    catalog = require_reference_models()
    require_model_named(catalog, "codeformer.fp16.onnx")
    source_fsz = legacy_root() / "app/facesets/akansha.fsz"
    target_fsz = legacy_root() / "app/facesets/anushree.fsz"
    assert source_fsz.is_file() and target_fsz.is_file(), "consented FaceSet fixtures are missing"
    reader = FaceSetReader()
    source_doc, target_doc = reader.read(source_fsz), reader.read(target_fsz)

    # -- GENERATE the test .fsz: a new id, the consented source's most useful views (frontal first, then spread in yaw)
    views = sorted((v for v in source_doc.views if not v.is_synthetic), key=lambda v: abs(v.yaw))[:6]
    views = tuple(v.model_copy(update={"index": i}) for i, v in enumerate(views))
    generated = source_doc.model_copy(update={"faceset_id": "e2e_generated", "name": "E2E generated", "views": views,
                                              "pose_bins": reader._build_pose_bins(list(views))})
    generated_path = tmp_path / "gen" / "roop_3d/outputs/facesets/e2e_generated.fsz"
    export_v2(generated, generated_path, app_root=tmp_path / "gen")
    assert 0 < generated_path.stat().st_size < 2 * 1024 * 1024, "the generated .fsz must fit the upload body cap"

    # -- GENERATE the 30-frame target clip (consented target face on a flat background, 440 Hz audio)
    target_view = min(target_doc.views, key=lambda v: abs(v.yaw))
    face = cv2.imdecode(np.frombuffer(target_view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    clip = create_golden_video(tmp_path / "clip" / "target.mp4", face, count=FRAMES, fps=FPS)
    assert clip.stat().st_size < 2 * 1024 * 1024

    out = io.StringIO()
    hud = PinokioTerminalHUD(out, color=False, in_place=False, plain_interval=0.0, min_redraw=0.0).arm()
    settings = Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider", audio_policy="copy")
    (tmp_path / "app").mkdir()
    app = create_app(settings, repo_root=tmp_path / "app", worker_enabled=True)
    result: dict = {"hud": out, "source_doc": source_doc, "target_doc": target_doc, "clip": clip, "root": tmp_path / "app", "ws": []}
    import roop_3d.jobs.worker as worker_module
    import roop_3d.pipeline.video_job_runner as runner_module
    import roop_3d.engine.pipeline_runner as base_runner_module
    saved = [(m, m.HUD) for m in (worker_module, runner_module, base_runner_module)]
    for module, _ in saved:
        module.HUD = hud
    try:
        with TestClient(app, base_url="http://127.0.0.1:8030") as client:
            result["client"] = client
            # -- import + upload through the NEW one-call routes
            imported = client.post("/api/v1/facesets/import?filename=e2e_generated.fsz", content=generated_path.read_bytes(),
                                   headers={"Content-Type": "application/octet-stream"})
            assert imported.status_code == 201, imported.text
            result["imported"] = imported.json()
            uploaded = client.post("/api/v1/media/upload?filename=target.mp4", content=clip.read_bytes(),
                                   headers={"Content-Type": "application/octet-stream"})
            assert uploaded.status_code == 201, uploaded.text
            result["uploaded"] = uploaded.json()
            inspection = client.post("/api/v1/media/inspect-faces", json={"path": uploaded.json()["path"]})
            assert inspection.status_code == 200, inspection.text
            tracks = inspection.json()["tracks"]
            assert tracks, "the detector must find the target face"

            collected: list[dict] = []
            with client.websocket_connect(STATUS_WS) as status:
                def listen() -> None:
                    try:
                        while True:
                            collected.append(status.receive_json())
                    except Exception:
                        return
                reader_thread = threading.Thread(target=listen, daemon=True)
                reader_thread.start()
                created = client.post("/api/v1/jobs", headers={"Idempotency-Key": "stage17-e2e"}, json={
                    "engine_id": ENGINE_ID, "target_path": uploaded.json()["path"], "inspection_id": inspection.json()["inspection_id"],
                    "face_mappings": [{"track_id": tracks[0]["track_id"], "faceset_id": imported.json()["faceset_id"]}]})
                assert created.status_code == 201, created.text
                job_id = created.json()["job_id"]
                deadline, record = time.monotonic() + 1500, {}
                while time.monotonic() < deadline:
                    record = client.get(f"/api/v1/jobs/{job_id}").json()
                    if record["state"] in {"succeeded", "failed", "cancelled", "interrupted"}:
                        break
                    time.sleep(0.25)
                assert record["state"] == "succeeded", record.get("failure") or record
                time.sleep(0.3)
                result["ws"] = list(collected)
            result["job"] = record
            result["job_id"] = job_id
            download = client.get(f"/api/v1/jobs/{job_id}/download")
            assert download.status_code == 200, download.text[:200]
            result["download"] = download
            result["ranged"] = client.get(f"/api/v1/jobs/{job_id}/download", headers={"Range": "bytes=0-99"})
            path = tmp_path / "downloaded.mp4"
            path.write_bytes(download.content)
            result["output"] = path
            result["manifest"] = json.loads((tmp_path / "app" / record["output"]["manifest_path"]).read_text("utf-8"))
            _save_qc_sheet(clip, path, Path(__file__).resolve().parents[1] / ".runtime/e2e/photoreal")
    finally:
        for module, original in saved:
            module.HUD = original
    return result


def test_the_import_and_upload_routes_stored_real_assets(pipeline_run):
    assert pipeline_run["imported"]["kind"] == "faceset" and len(pipeline_run["imported"]["sha256"]) == 64
    assert pipeline_run["uploaded"]["kind"] == "video"
    assert hashlib.sha256(pipeline_run["clip"].read_bytes()).hexdigest() == pipeline_run["uploaded"]["sha256"]


def test_ffprobe_verifies_the_output_frames_rate_size_and_audio(pipeline_run):
    path, document = pipeline_run["output"], ffprobe_json(pipeline_run["output"])
    assert path.is_file() and path.stat().st_size > 10_000
    video = next(s for s in document["streams"] if s["codec_type"] == "video")
    audio = [s for s in document["streams"] if s["codec_type"] == "audio"]
    source = ffprobe_json(pipeline_run["clip"])
    source_video = next(s for s in source["streams"] if s["codec_type"] == "video")
    assert int(video["nb_read_packets"]) == FRAMES == int(source_video["nb_read_packets"])
    assert Fraction(video["avg_frame_rate"]) == Fraction(FPS) == Fraction(source_video["avg_frame_rate"])
    assert (video["width"], video["height"]) == (512, 512) and video["codec_name"] == "h264" and video["pix_fmt"] == "yuv420p"
    assert len(audio) == 1 and audio[0]["codec_name"] == "aac"
    assert abs(float(document["format"]["duration"]) - float(source["format"]["duration"])) < 0.15
    assert verify_output(path)["frames"] == FRAMES


def test_the_audio_was_stream_copied_not_re_encoded(pipeline_run):
    source = next(s for s in ffprobe_json(pipeline_run["clip"])["streams"] if s["codec_type"] == "audio")
    out = next(s for s in ffprobe_json(pipeline_run["output"])["streams"] if s["codec_type"] == "audio")
    assert pipeline_run["manifest"]["audio_policy"] == "copy"
    assert (out["codec_name"], out["sample_rate"], out["channels"]) == (source["codec_name"], source["sample_rate"], source["channels"])
    # a stream copy leaves the compressed packets byte-identical
    def packet_hash(path: Path) -> str:
        run = subprocess.run([find_ffprobe().replace("ffprobe", "ffmpeg"), "-v", "error", "-i", str(path), "-map", "0:a:0", "-c", "copy",
                              "-f", "md5", "-"], capture_output=True, text=True)
        return run.stdout.strip()
    assert packet_hash(pipeline_run["output"]) == packet_hash(pipeline_run["clip"])


def test_download_route_serves_the_verified_file_with_range_support(pipeline_run):
    job, download, manifest = pipeline_run["job"], pipeline_run["download"], pipeline_run["manifest"]
    assert download.headers["content-type"] == "video/mp4"
    assert hashlib.sha256(download.content).hexdigest() == job["output"]["output_sha256"] == manifest["output_sha256"]
    assert pipeline_run["ranged"].status_code == 206 and len(pipeline_run["ranged"].content) == 100
    assert job["output"]["output_path"].endswith(f"{pipeline_run['job_id']}/result.mp4")


def test_ssim_background_is_preserved_and_the_face_region_really_changed(pipeline_run):
    before, after = frames_of(pipeline_run["clip"]), frames_of(pipeline_run["output"])
    assert len(before) == len(after) == FRAMES
    background, face = [], []
    for a, b in zip(before, after):
        background.append(ssim(a[:64, :64], b[:64, :64]))
        face.append(ssim(a[160:350, 160:350], b[160:350, 160:350]))
    assert min(background) > 0.9, f"background corner drifted: min SSIM {min(background):.3f}"
    assert np.mean(face) < 0.8, f"the face region did not change enough: mean SSIM {np.mean(face):.3f}"
    pipeline_run["ssim"] = {"background_min": min(background), "face_mean": float(np.mean(face))}


def test_identity_moved_from_the_target_person_to_the_source_person(pipeline_run):
    """ArcFace (the pipeline's own recognizer): the output face matches the source FaceSet, no longer the target."""
    from roop_3d.engine.detection.face_tracker import FaceTracker
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.models.catalog import ModelCatalog
    tracker = FaceTracker(ModelCatalog(), config=RuntimeConfig(provider="CPUExecutionProvider"), threshold=0.4)
    source, target = pipeline_run["source_doc"].canonical_embedding, pipeline_run["target_doc"].canonical_embedding
    before, after = frames_of(pipeline_run["clip"]), frames_of(pipeline_run["output"])
    gains = []
    for index in (3, 15, 27):
        faces_in, faces_out = tracker.process(before[index]), tracker.process(after[index])
        assert faces_in and faces_out, f"no face found in frame {index}"
        in_embedding, out_embedding = faces_in[0].observation.embedding, faces_out[0].observation.embedding
        to_source, to_target = cosine(out_embedding, source), cosine(out_embedding, target)
        gains.append((cosine(in_embedding, source), to_source, cosine(in_embedding, target), to_target))
        assert to_source > to_target, (index, to_source, to_target)
        assert to_source > cosine(in_embedding, source) + 0.10, (index, to_source, cosine(in_embedding, source))
    pipeline_run["identity"] = gains


def test_the_face_is_not_plasticised_fine_detail_survives(pipeline_run):
    """The 'no CGI look' guard: skin micro-contrast in the swapped face stays within the original photo's range."""
    before, after = frames_of(pipeline_run["clip"]), frames_of(pipeline_run["output"])
    ratios = []
    for a, b in zip(before[::5], after[::5]):
        la = cv2.Laplacian(cv2.cvtColor(a[200:330, 200:330], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        lb = cv2.Laplacian(cv2.cvtColor(b[200:330, 200:330], cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        ratios.append(lb / max(la, 1e-6))
    assert 0.3 < float(np.mean(ratios)) < 4.0, ratios
    pipeline_run["detail_ratio"] = float(np.mean(ratios))


def test_manifest_is_honest_about_what_ran(pipeline_run):
    manifest = pipeline_run["manifest"]
    assert manifest["engine_id"] == ENGINE_ID and manifest["verified_frames"] == FRAMES
    assert manifest["faces"]["swapped"] == FRAMES and manifest["faces"]["skipped"] == 0
    assert manifest["encoder"] in {"h264_nvenc", "libx264"} and manifest["color"]["matrix"] == "bt709"
    assert manifest["restorer"] in {"ultramax", "gpen", "codeformer", "gfpgan"} and manifest["identity_swapper"].get("sha256")
    assert manifest["limits"]["resumable"] is False and manifest["metadata_stripped"] is False
    assert manifest["per_frame_mean_ms"].get("inswapper", 0) > 0 and manifest["per_frame_mean_ms"].get("restorer", 0) > 0
    assert manifest["swap_scope"] == "full_craniofacial"


def test_the_terminal_hud_saw_the_stream_and_the_remux(pipeline_run):
    text = pipeline_run["hud"].getvalue()
    short = pipeline_run["job_id"][:8]
    assert f"[JOB] started job={short}" in text and f"[PIPE] ffmpeg stream started job={short}" in text
    assert f"[REMUX] output written and verified job={short} frames={FRAMES}" in text and f"[JOB] succeeded job={short}" in text
    assert f"{FRAMES}/{FRAMES}" in text                                         # the frame bar reached the end


def test_the_pipeline_status_socket_streamed_live_progress(pipeline_run):
    frames = [f for f in pipeline_run["ws"] if f.get("type") == "pipeline.status" and f.get("job_id") == pipeline_run["job_id"]]
    assert len(frames) >= 5, f"only {len(frames)} live status frames"
    percentages = [f["stage_progress_pct"] for f in frames if f.get("stage_progress_pct") is not None]
    assert percentages == sorted(percentages) and max(percentages) >= 90
    assert any((f.get("fps") or 0) > 0 for f in frames)
    progress = [f for f in pipeline_run["ws"] if f.get("type") == "job.progress" and f.get("job_id") == pipeline_run["job_id"]]
    assert any(f.get("stage_timings_ms", {}).get("inswapper") for f in progress) or not progress


def test_no_colour_fringes_or_specks_were_introduced(pipeline_run):
    """Regression guard for defects only a LOOK found: a green fringe along the hair (fill taken from the far backdrop), blue specks at
    the restorer's crop border and from Laplacian overshoot. Inside the photo (away from the green canvas) the original has no strongly
    saturated blue or green pixel, so the render may add at most a handful."""
    before, after = frames_of(pipeline_run["clip"]), frames_of(pipeline_run["output"])

    def saturated(frame):
        inner = frame[70:440, 70:440].astype(int)
        blue = (inner[..., 0] > inner[..., 1] + 50) & (inner[..., 0] > inner[..., 2] + 50)
        green = (inner[..., 1] > inner[..., 0] + 60) & (inner[..., 1] > inner[..., 2] + 60)
        return int(blue.sum()), int(green.sum())

    worst = max(max(0, saturated(b)[0] - saturated(a)[0]) + max(0, saturated(b)[1] - saturated(a)[1]) for a, b in zip(before, after))
    assert worst <= 25, f"{worst} new saturated pixels in one frame"
    pipeline_run["new_saturated_px_worst"] = worst
