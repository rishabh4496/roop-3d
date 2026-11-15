#!/usr/bin/env python
"""Stage 5 - end-to-end batch verification of the true-3D head-swap pipeline on real videos.

Run from anywhere (the script puts ``app/`` on ``sys.path`` and switches into it):

    app/roop_3d/.venv/Scripts/python.exe -B scripts/verify_e2e_test1.py [--only s1.mp4 ...] [--faceset harjot]
        [--videos DIR] [--output DIR] [--provider CUDAExecutionProvider] [--profile 12gb]

Per video it runs the PRODUCTION stack, nothing mocked or replayed:

  A. identity scan   - the identity tracker runs over the whole timeline, tracklets are consolidated globally
                       (``identity.consolidation``) into canonical identities; the most-observed one is the locked target.
  B. locked pre-pass - ``ReferencePipeline.scan_locked_targets`` binds the canonical target ID by the global decision.
  C. render          - ``track_frame`` -> ``infer_frame`` (dense68 PnP, MakeHuman volumetric head, SH lighting) -> semantic
                       masking (BiSeNet + XSeg + occluder, hair excluded) -> Laplacian edge blend, streamed through FFmpeg
                       (``stream_video``) at the source resolution / exact frame rate with the source audio remuxed.
  D. integrity       - ffprobe checks of the output file (stream, size, frame count, fps, audio) + full decode.
  E. coverage        - per-frame bookkeeping (target present / swapped / skipped / withheld, pose class) AND an independent
                       pixel audit (output vs source inside the target box) so an unswapped frame cannot hide.

Gate: exit code 0 only when every assertion of every video holds; 1 on any failed assertion; 2 on an aborted run.
The gate certifies mechanics (files, counts, coverage, ID stability) - it does NOT certify visual realism; look at the videos.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"
sys.path.insert(0, str(APP))
os.chdir(APP)

import cv2                                   # noqa: E402
import numpy as np                           # noqa: E402

from roop_3d.audit.project_paths import application_root                       # noqa: E402
from roop_3d.core.errors import DomainError                                    # noqa: E402
from roop_3d.core.pose import as_pose3d                                        # noqa: E402
from roop_3d.core.settings import Settings                                     # noqa: E402
from roop_3d.core.storage import atomic_json                                   # noqa: E402
from roop_3d.engine.detection.track_policy import TrackPolicy                  # noqa: E402
from roop_3d.engine.runtime import RuntimeConfig                               # noqa: E402
from roop_3d.engine.volumetric.session import ReferencePipeline                # noqa: E402
from roop_3d.identity.consolidation import TrackletRecorder, consolidate       # noqa: E402
from roop_3d.identity.locked_target import UNBOUND_BASE, LockedTargetBinder    # noqa: E402
from roop_3d.models.catalog import ModelCatalog                                # noqa: E402
from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore, SwapCoreOptions  # noqa: E402
from roop_3d.pipeline.session_adapter import SessionFrameProcessor            # noqa: E402
from roop_3d.pipeline.video_job_runner import probe_video, stream_video, verify_output  # noqa: E402
from roop_3d.segment_dispatcher import find_ffmpeg, find_ffprobe               # noqa: E402
from roop_3d.tools.stage3_scan import natural                                  # noqa: E402
from roop_3d.tools.verify_stage4 import load_source                            # noqa: E402

DEFAULT_VIDEOS = Path(r"G:\pinokio\roop-keep\single")
DEFAULT_OUTPUT = Path(r"G:\pinokio\api\roop-3d\app\roop_3d\outputs\test 1")
DEFAULT_FACESETS = Path(r"G:\pinokio\api\roop-3d\app\facesets")
RECEIPT = "roop_3d/.runtime/stage5_verification.json"
ROW_DIR = APP / "roop_3d/.runtime/stage5_rows"
POSE_CLASSES = (("frontal", 0.0, 25.0), ("angled", 25.0, 60.0), ("profile", 60.0, 181.0))
PIXEL_SIDE = 640                 # long side the pixel audit downsizes to
PIXEL_DELTA = 15                 # gray levels a pixel must differ by to count as "changed"
PIXEL_MIN_CHANGED = 0.02         # a target box with fewer changed pixels than this is an unswapped frame
AUDIO_TOLERANCE_S = 0.15


def say(*parts) -> None:
    print(*parts, flush=True)


# ------------------------------------------------------------------------------------------------------ data
@dataclass
class FrameRecord:
    index: int
    target_present: bool = False
    box: list[float] | None = None
    yaw: float | None = None
    pose_class: str | None = None
    swapped: bool = False
    geometry_mode: str | None = None
    skip_codes: list[str] = field(default_factory=list)
    skip_messages: list[str] = field(default_factory=list)
    withheld: list[str] = field(default_factory=list)
    other_canonical_ids: list[int] = field(default_factory=list)
    unbound_faces: int = 0
    total_ms: float = 0.0


def classify(yaw: float | None) -> str | None:
    if yaw is None:
        return None
    for name, low, high in POSE_CLASSES:
        if low <= abs(yaw) < high:
            return name
    return "profile"


def face_yaw(observation) -> float | None:
    pose6 = getattr(observation, "pose6dof", None)
    if pose6 is not None and np.isfinite(getattr(pose6, "yaw_deg", np.nan)):
        return float(pose6.yaw_deg)
    try:
        return float(as_pose3d(observation.pose).yaw)
    except Exception:
        return None


# ------------------------------------------------------------------------------------------------------ passes
def decode_frames(video: Path):
    from roop_3d.media.video_reader import VideoReader
    return VideoReader(video, hardware="auto", hdr_policy="tonemap_sdr").frames()


def identity_scan(session: ReferencePipeline, video: Path, on_progress) -> tuple[object, int]:
    """Pass A: the identity tracker over every frame, then global consolidation. Returns (consolidation, frames_scanned)."""
    recorder = TrackletRecorder()
    frames = decode_frames(video)
    count = 0
    tracker = session.tracker
    try:
        for index, decoded in enumerate(frames):
            if decoded.image is not None:
                session.set_frame_timestamp(float(decoded.timestamp.seconds))
                tracker.frame_number = index
                recorder.observe(index, tracker.process(decoded.image))
            count = index + 1
            on_progress(count)
    finally:
        frames.close()
        tracker.reset_tracking()
        tracker.frame_number = None
    return consolidate(recorder.tracklets(), TrackPolicy()), count


def build_binder(consolidation) -> tuple[LockedTargetBinder, int, dict]:
    signatures = {}
    for identity in consolidation.identities:
        vectors = ([identity.centroid] if identity.centroid is not None else []) + list(identity.anchors)
        if vectors:
            signatures[identity.canonical_id] = np.stack(vectors)
    if not signatures:
        raise DomainError("TARGET_NOT_FOUND", "No identity with an embedding signature was found in the whole timeline", 422)
    target = max((i for i in consolidation.identities if i.canonical_id in signatures),
                 key=lambda i: (i.trusted, i.observations))
    summary = {"identities": [{"canonical_id": i.canonical_id, "raw_tracker_ids": list(i.raw_ids), "observations": i.observations,
                               "trusted": i.trusted, "frames": [i.first_frame, i.last_frame]} for i in consolidation.identities],
               "raw_tracks": len(consolidation.canonical_of), "suppressed": {str(k): v for k, v in consolidation.suppressed.items()},
               "merges": sum(1 for d in consolidation.decisions if d.merged), "target_canonical_id": target.canonical_id}
    return LockedTargetBinder(signatures, TrackPolicy()), target.canonical_id, summary


# --------------------------------------------------------------------------------------------------- integrity
def ffprobe_json(path: Path, *extra: str) -> dict:
    run = subprocess.run([find_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", *extra, str(path)],
                         capture_output=True, text=True, timeout=300)
    return json.loads(run.stdout or "{}")


def full_decode_errors(path: Path) -> str:
    run = subprocess.run([find_ffmpeg(), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=1800)
    return (run.stderr or "").strip()[-300:] if run.returncode != 0 or run.stderr.strip() else ""


def audio_facts(path: Path) -> dict:
    document = ffprobe_json(path)
    streams = [s for s in document.get("streams", []) if s.get("codec_type") == "audio"]
    facts = []
    for stream in streams:
        try:
            duration = float(stream.get("duration") or document.get("format", {}).get("duration") or 0)
        except (TypeError, ValueError):
            duration = 0.0
        facts.append({"codec": stream.get("codec_name"), "channels": stream.get("channels"), "sample_rate": stream.get("sample_rate"),
                      "duration": duration})
    return {"streams": facts}


def check(name: str, passed: bool, detail: str) -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def integrity_checks(source: Path, output: Path, src_info, expected_frames: int) -> tuple[list[dict], dict]:
    checks: list[dict] = []
    facts: dict = {}
    exists = output.is_file()
    size = output.stat().st_size if exists else 0
    checks.append(check("file_exists_non_zero", exists and size > 0, f"{size:,} bytes"))
    facts["bytes"] = size
    if not (exists and size > 0):
        return checks, facts
    try:
        out = verify_output(output)
        src = verify_output(source)
    except DomainError as error:
        checks.append(check("valid_video_stream", False, f"{error.code}: {error.message}"))
        return checks, facts
    checks.append(check("valid_video_stream", True, f"{out['codec']} {out['width']}x{out['height']} {out['pix_fmt']}"))
    facts.update({"codec": out["codec"], "pix_fmt": out["pix_fmt"], "colorspace": out["colorspace"]})
    decode_error = full_decode_errors(output)
    checks.append(check("full_decode_clean", not decode_error, decode_error or "ffmpeg decoded every frame without error"))
    checks.append(check("resolution_preserved", (out["width"], out["height"]) == (src_info.width, src_info.height),
                        f"{out['width']}x{out['height']} vs source {src_info.width}x{src_info.height}"))
    checks.append(check("frame_count_matches_source", out["frames"] == src["frames"] == expected_frames,
                        f"output {out['frames']} / source {src['frames']} / decoded by pipeline {expected_frames}"))
    checks.append(check("fps_matches_source", out["rate"] == src_info.rate, f"output {out['rate']} vs source {src_info.rate}"))
    facts.update({"frames": out["frames"], "fps": str(out["rate"]), "width": out["width"], "height": out["height"]})
    src_audio, out_audio = audio_facts(source), audio_facts(output)
    facts["audio_source"], facts["audio_output"] = src_audio, out_audio
    if not src_audio["streams"]:
        checks.append(check("audio_preserved", not out_audio["streams"], "source has no audio stream; output has none either"))
    else:
        problems = []
        if len(out_audio["streams"]) != len(src_audio["streams"]):
            problems.append(f"{len(out_audio['streams'])} audio stream(s) vs source {len(src_audio['streams'])}")
        else:
            for k, (a, b) in enumerate(zip(src_audio["streams"], out_audio["streams"])):
                if abs(a["duration"] - b["duration"]) > AUDIO_TOLERANCE_S:
                    problems.append(f"stream {k} duration {b['duration']:.3f}s vs source {a['duration']:.3f}s")
                if (a["channels"], str(a["sample_rate"])) != (b["channels"], str(b["sample_rate"])):
                    problems.append(f"stream {k} layout {b['channels']}ch/{b['sample_rate']} vs {a['channels']}ch/{a['sample_rate']}")
        checks.append(check("audio_preserved", not problems,
                            "; ".join(problems) or f"{len(out_audio['streams'])} stream(s), {out_audio['streams'][0]['codec']} "
                            f"{out_audio['streams'][0]['duration']:.3f}s (source {src_audio['streams'][0]['codec']} "
                            f"{src_audio['streams'][0]['duration']:.3f}s)"))
    return checks, facts


# ----------------------------------------------------------------------------------------------- pixel audit
def pixel_audit(source: Path, output: Path, records: list[FrameRecord]) -> dict:
    """Independent of the pipeline's own bookkeeping: decode source and output side by side and measure how many pixels inside the
    target face box changed. A target-present frame whose box is (nearly) untouched is unswapped no matter what the report says."""
    cap_a, cap_b = cv2.VideoCapture(str(source)), cv2.VideoCapture(str(output))
    changed_by_frame: dict[int, float] = {}
    background: list[float] = []
    note = None
    try:
        for record in records:
            ok_a, a = cap_a.read()
            ok_b, b = cap_b.read()
            if not (ok_a and ok_b):
                note = f"decode ended at frame {record.index} (source ok={ok_a}, output ok={ok_b})"
                break
            if a.shape != b.shape:
                note = f"frame {record.index}: source {a.shape} vs output {b.shape}"
                break
            if not (record.target_present and record.box):
                continue
            scale = PIXEL_SIDE / max(a.shape[:2])
            if scale < 1:
                a, b = (cv2.resize(x, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) for x in (a, b))
            else:
                scale = 1.0
            diff = np.abs(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.int16) - cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.int16))
            x1, y1, x2, y2 = (int(round(v * scale)) for v in record.box)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(diff.shape[1], x2), min(diff.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            region = diff[y1:y2, x1:x2]
            changed_by_frame[record.index] = float((region > PIXEL_DELTA).mean())
            outside = np.ones(diff.shape, bool)
            outside[max(0, y1 - (y2 - y1) // 2):y2 + (y2 - y1) // 2, max(0, x1 - (x2 - x1) // 2):x2 + (x2 - x1) // 2] = False
            if outside.any():
                background.append(float((diff[outside] > PIXEL_DELTA).mean()))
    finally:
        cap_a.release()
        cap_b.release()
    values = np.array(list(changed_by_frame.values())) if changed_by_frame else np.zeros(0)
    return {"frames_audited": len(changed_by_frame), "note": note,
            "unswapped_by_pixels": sorted(i for i, v in changed_by_frame.items() if v < PIXEL_MIN_CHANGED),
            "changed_fraction_median": float(np.median(values)) if len(values) else None,
            "changed_fraction_min": float(values.min()) if len(values) else None,
            "background_changed_fraction_mean": float(np.mean(background)) if background else None}


# ---------------------------------------------------------------------------------------------------- coverage
def runs(indices: list[int]) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for index in indices:
        if result and index == result[-1][1] + 1:
            result[-1] = (result[-1][0], index)
        else:
            result.append((index, index))
    return result


def coverage_summary(records: list[FrameRecord], target_id: int, identities: int, audit: dict) -> tuple[list[dict], dict]:
    present = [r for r in records if r.target_present]
    unswapped = [r.index for r in present if not r.swapped]
    swapped_set = {r.index for r in present if r.swapped}
    flicker = [(a, b) for a, b in runs(unswapped) if (a - 1) in swapped_set and (b + 1) in swapped_set]
    by_class: dict[str, dict] = {}
    for name, _low, _high in POSE_CLASSES:
        rows = [r for r in present if r.pose_class == name]
        by_class[name] = {"target_frames": len(rows), "swapped": sum(r.swapped for r in rows),
                          "coverage": (sum(r.swapped for r in rows) / len(rows)) if rows else None}
    unknown = [r for r in present if r.pose_class is None]
    other_ids = sorted({i for r in records for i in r.other_canonical_ids})
    other_frames = [r.index for r in records if r.other_canonical_ids]
    unbound_frames = [r.index for r in records if r.unbound_faces]
    skip_codes: dict[str, int] = {}
    for r in present:
        for code, message in zip(r.skip_codes, r.skip_messages or [""] * len(r.skip_codes)):
            key = f"{code}: {message}" if message else code
            skip_codes[key] = skip_codes.get(key, 0) + 1
        for code in r.withheld:
            skip_codes[code] = skip_codes.get(code, 0) + 1
    modes: dict[str, int] = {}
    for r in records:
        if r.swapped and r.geometry_mode:
            modes[r.geometry_mode] = modes.get(r.geometry_mode, 0) + 1
    stats = {"frames": len(records), "target_frames": len(present), "swapped_target_frames": len(swapped_set),
             "coverage": (len(swapped_set) / len(present)) if present else 0.0, "unswapped_target_frames": len(unswapped),
             "unswapped_runs": [[a, b] for a, b in runs(unswapped)], "flicker_runs": [[a, b] for a, b in flicker],
             "by_pose_class": by_class, "pose_unknown_frames": len(unknown), "skip_or_withhold_reasons": skip_codes,
             "geometry_modes": modes, "consolidated_identities": identities, "target_canonical_id": target_id,
             "other_canonical_ids_seen": other_ids, "frames_with_other_canonical_ids": len(other_frames),
             "frames_with_unbound_faces": len(unbound_frames), "pixel_audit": audit,
             "mean_ms_per_frame": float(np.mean([r.total_ms for r in records])) if records else 0.0}
    checks = [
        check("target_found", len(present) > 0, f"{len(present)} of {len(records)} frames carry the locked target"),
        check("zero_unswapped_target_frames", not unswapped,
              f"{len(unswapped)} target frame(s) not swapped" + (f", runs {stats['unswapped_runs'][:6]}" if unswapped else "")),
        check("zero_flicker_frames", not flicker, f"{sum(b - a + 1 for a, b in flicker)} frame(s) in {len(flicker)} swapped-unswapped-swapped run(s)"),
        check("pixel_audit_zero_unswapped", not audit["unswapped_by_pixels"] and audit["note"] is None,
              f"{len(audit['unswapped_by_pixels'])} of {audit['frames_audited']} audited target boxes unchanged "
              f"(< {PIXEL_MIN_CHANGED:.0%} pixels changed by > {PIXEL_DELTA} levels)" + (f"; {audit['note']}" if audit["note"] else "")),
        check("single_identity", identities == 1, f"{identities} canonical identit{'y' if identities == 1 else 'ies'} after global consolidation"),
        check("no_face_2_3_4_5_splits", not other_frames, f"canonical IDs other than Face {target_id}: {other_ids or 'none'} "
              f"on {len(other_frames)} frame(s)"),
        check("no_unbound_faces", not unbound_frames, f"{len(unbound_frames)} frame(s) with a face no canonical ID claimed"),
    ]
    return checks, stats


# ------------------------------------------------------------------------------------------------- per video
def process_video(video: Path, output_dir: Path, loaded, catalog, config, args) -> dict:
    began = time.perf_counter()
    row: dict = {"video": video.name, "input": str(video), "checks": [], "passed": False}
    output = output_dir / f"{video.stem}.mp4"
    session = ReferencePipeline(catalog, config)
    processor = None
    try:
        info = probe_video(video)
        row["source"] = {"width": info.width, "height": info.height, "fps": str(info.rate), "declared_frames": info.frames,
                         "audio_streams": info.audio_streams, "audio_codec": info.audio_codec, "vfr": info.vfr,
                         "color_space": info.color_space, "rotation": info.rotation}
        say(f"\n=== {video.name}: {info.width}x{info.height} @ {float(info.rate):.3f} fps, {info.frames} frames, "
            f"audio={info.audio_codec or 'none'}")
        policy = TrackPolicy()
        session.prepare_session(loaded.document, detection_threshold=args.detection_threshold, track_policy=policy,
                                restoration_weight=args.restoration_weight, texture_size=args.texture_size)
        mark = time.perf_counter()
        last = [0.0]

        def progress_a(done: int) -> None:
            if time.perf_counter() - last[0] > 20:
                last[0] = time.perf_counter()
                say(f"  A identity scan {done}/{info.frames}")

        consolidation, scanned = identity_scan(session, video, progress_a)
        row["identity_scan_seconds"] = round(time.perf_counter() - mark, 1)
        row["scanned_frames"] = scanned
        binder, target_id, identity_summary = build_binder(consolidation)
        row["identity"] = identity_summary
        say(f"  A done: {scanned} frames, {len(consolidation.identities)} identit"
            f"{'y' if len(consolidation.identities) == 1 else 'ies'}, target = Face {target_id} ({row['identity_scan_seconds']}s)")

        session.prepare_session(loaded.document, track_ids={target_id}, sources_by_track={target_id: loaded.document},
                                detection_threshold=args.detection_threshold, track_policy=policy, locked_targets=binder,
                                restoration_weight=args.restoration_weight, texture_size=args.texture_size)
        session.configure_temporal(Settings())
        mark = time.perf_counter()
        prepass = session.scan_locked_targets(video)
        row["locked_prepass"] = prepass
        row["locked_prepass_seconds"] = round(time.perf_counter() - mark, 1)
        say(f"  B locked pre-pass: {prepass} ({row['locked_prepass_seconds']}s)")

        processor = SessionFrameProcessor(session, Photorealistic3DSwapCore(SwapCoreOptions(
            restoration_weight=args.restoration_weight)))
        records: list[FrameRecord] = []
        rate = float(info.rate)
        started = [time.perf_counter()]

        def frame_fn(frame: np.ndarray, index: int) -> np.ndarray:
            t0 = time.perf_counter()
            record = FrameRecord(index)
            session.set_frame_timestamp(index / rate)
            tracked = session.track_frame(frame)
            for face in tracked.faces:
                if face.track_id == target_id:
                    record.target_present = True
                    record.box = [float(v) for v in face.observation.box]
                    record.yaw = face_yaw(face.observation)
                    record.pose_class = classify(record.yaw)
                elif face.track_id >= UNBOUND_BASE:
                    record.unbound_faces += 1
                else:
                    record.other_canonical_ids.append(int(face.track_id))
            output_frame = frame
            try:
                result = session.infer_frame(frame, tracked=tracked)
                output_frame, report = processor.process_result(frame, result, index)
                record.swapped = any(f.track_id == target_id for f in report.faces)
                record.geometry_mode = next((f.geometry.get("mode") for f in report.faces if f.track_id == target_id), None)
                record.skip_codes = [str(item.get("code")) for item in report.skipped]
                record.skip_messages = [str(item.get("message") or item.get("details") or "")[:160] for item in report.skipped]
                record.withheld = [str(item["reason"]) for item in report.withheld]
            except DomainError as error:
                if error.code not in SessionFrameProcessor.FIT_SKIP_CODES:
                    raise
                record.skip_codes.append(error.code)
                record.skip_messages.append(str(error.message)[:160])
            record.total_ms = (time.perf_counter() - t0) * 1000
            records.append(record)
            return output_frame

        def on_progress(done: int, fps: float, _decode_ms: float, _encode_ms: float) -> None:
            if time.perf_counter() - last[0] > 20:
                last[0] = time.perf_counter()
                swapped = sum(r.swapped for r in records)
                say(f"  C render {done}/{info.frames} frames, {fps:.2f} fps, swapped {swapped}/{sum(r.target_present for r in records)} target")

        mark = time.perf_counter()
        result = stream_video(video, output, frame_fn, info=info, prefer_nvenc=config.hardware_profile != "cpu",
                              audio_policy="copy", progress=on_progress)
        row["render_seconds"] = round(time.perf_counter() - mark, 1)
        row["encoder"] = result.encoder
        row["encoder_fallback"] = result.encoder_fallback
        row["audio_remux"] = result.audio_policy
        say(f"  C done: {result.frames} frames in {row['render_seconds']}s ({result.frames / max(row['render_seconds'], 1e-6):.2f} fps), "
            f"encoder {result.encoder}, audio {result.audio_policy}")

        checks, facts = integrity_checks(video, output, info, scanned)
        row["output"] = {"path": str(output), **facts}
        say("  D integrity: " + ", ".join(f"{c['name']}={'ok' if c['passed'] else 'FAIL'}" for c in checks))
        mark = time.perf_counter()
        audit = pixel_audit(video, output, records)
        say(f"  E pixel audit: {audit['frames_audited']} target boxes, median changed {audit['changed_fraction_median']}, "
            f"{len(audit['unswapped_by_pixels'])} unchanged ({time.perf_counter() - mark:.0f}s)")
        coverage_checks, coverage = coverage_summary(records, target_id, len(consolidation.identities), audit)
        row["coverage"] = coverage
        row["checks"] = checks + coverage_checks
        row["per_frame"] = [{"i": r.index, "target": r.target_present, "swapped": r.swapped, "yaw": None if r.yaw is None else round(r.yaw, 1),
                             "class": r.pose_class, "skip": r.skip_codes, "withheld": r.withheld} for r in records]
    except BaseException as error:                                      # one bad video must not hide the others
        if isinstance(error, KeyboardInterrupt):
            raise
        row["fatal"] = {"kind": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        row["checks"].append(check("pipeline_completed", False, f"{type(error).__name__}: {error}"))
        say(f"  FATAL {type(error).__name__}: {error}")
    finally:
        if processor is not None:
            processor.release()
        session.release_session()
    row["passed"] = bool(row["checks"]) and all(c["passed"] for c in row["checks"])
    row["seconds"] = round(time.perf_counter() - began, 1)
    return row


# ----------------------------------------------------------------------------------------------------- report
def render_report(receipt: dict) -> str:
    lines = ["", "=" * 118, "STAGE 5 VERIFICATION REPORT".center(118), "=" * 118]
    face = receipt.get("faceset", {})
    lines.append(f"Faceset {face.get('id')}: {face.get('views')} views, dense68 on {len(face.get('views_with_dense68', []))}, "
                 f"models {face.get('landmark_models')}")
    lines.append(f"Runtime: provider {receipt['runtime']['provider']}, profile {receipt['runtime']['hardware_profile']}; "
                 f"output {receipt['output_dir']}")
    lines.append("")
    header = (f"{'video':9s}{'res':>10s}{'fps':>8s}{'frames':>7s}{'target':>7s}{'swapped':>8s}{'cover':>7s}{'frontal':>9s}{'angled':>9s}"
              f"{'profile':>9s}{'unswp':>6s}{'flick':>6s}{'IDs':>4s}{'split':>6s}{'audio':>7s}{'s/video':>9s}  result")
    lines.append(header)
    for row in receipt["videos"]:
        cov, src = row.get("coverage"), row.get("source", {})
        if cov is None:
            lines.append(f"{row['video']:9s} ABORTED: {row.get('fatal', {}).get('message', '')[:90]}")
            continue
        def pc(name):
            item = cov["by_pose_class"][name]
            return "n/a" if item["coverage"] is None else f"{item['swapped']}/{item['target_frames']}"
        audio = next((c for c in row["checks"] if c["name"] == "audio_preserved"), None)
        audio_text = ("n/a" if not src.get("audio_streams") else "ok") if audio and audio["passed"] else "FAIL"
        lines.append(f"{row['video']:9s}{src['width']}x{src['height']:<5}{float(Fraction(src['fps'])):8.3f}{cov['frames']:7d}"
                     f"{cov['target_frames']:7d}{cov['swapped_target_frames']:8d}{cov['coverage']:7.1%}{pc('frontal'):>9s}{pc('angled'):>9s}"
                     f"{pc('profile'):>9s}{cov['unswapped_target_frames']:6d}{sum(b - a + 1 for a, b in cov['flicker_runs']):6d}"
                     f"{cov['consolidated_identities']:4d}{cov['frames_with_other_canonical_ids']:6d}{audio_text:>7s}"
                     f"{row['seconds']:9.0f}  {'PASS' if row['passed'] else 'FAIL'}")
    lines.append("")
    for row in receipt["videos"]:
        failed = [c for c in row["checks"] if not c["passed"]]
        lines.append(f"[{'PASS' if row['passed'] else 'FAIL'}] {row['video']}")
        for item in row["checks"]:
            lines.append(f"      {'ok  ' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
        cov = row.get("coverage")
        if cov:
            lines.append(f"      geometry modes {cov['geometry_modes']}; skip/withhold reasons {cov['skip_or_withhold_reasons'] or 'none'}; "
                         f"mean {cov['mean_ms_per_frame']:.0f} ms/frame")
            if cov["unswapped_runs"]:
                lines.append(f"      unswapped target runs (frame ranges): {cov['unswapped_runs'][:12]}")
        if failed and row.get("fatal"):
            lines.append("      " + row["fatal"]["traceback"].strip().splitlines()[-1])
    lines.append("")
    lines.append(f"{sum(r['passed'] for r in receipt['videos'])}/{len(receipt['videos'])} videos pass every assertion. "
                 f"Overall: {'PASS' if receipt['passed'] else 'FAIL'}")
    lines.append("The gate certifies files, counts, coverage and ID stability; it does not certify visual realism - watch the outputs.")
    return "\n".join(lines)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--videos", type=Path, default=DEFAULT_VIDEOS, help="directory of .mp4 inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory the swapped .mp4 files are written to")
    parser.add_argument("--facesets", type=Path, default=DEFAULT_FACESETS, help="directory that holds the faceset archive")
    parser.add_argument("--faceset", default="harjot")
    parser.add_argument("--only", nargs="*", help="video file names to process (default: every .mp4)")
    parser.add_argument("--provider", default="CUDAExecutionProvider", choices=("auto", "CPUExecutionProvider", "CUDAExecutionProvider",
                                                                                 "DmlExecutionProvider", "TensorrtExecutionProvider"))
    parser.add_argument("--profile", default=None, choices=("cpu", "6gb", "12gb"))
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--restoration-weight", type=float, default=0.35)
    parser.add_argument("--detection-threshold", type=float, default=0.4)
    parser.add_argument("--resume", action="store_true",
                        help="reuse the finished row of a video whose output file still exists (also re-prints the batch report)")
    return parser.parse_args(argv)


def row_path(video: Path) -> Path:
    return ROW_DIR / f"{video.stem}.json"


def load_row(video: Path, output_dir: Path) -> dict | None:
    """A finished row from an earlier invocation, valid only while its output file is still on disk."""
    path = row_path(video)
    try:
        row = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    return row if (output_dir / f"{video.stem}.mp4").is_file() and "checks" in row else None


def main(argv=None) -> int:
    args = parse_args(argv)
    began = time.perf_counter()
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)                                  # task 1
    ROW_DIR.mkdir(parents=True, exist_ok=True)
    every = sorted(args.videos.glob("*.mp4"), key=natural)
    videos = [v for v in every if not args.only or v.name in set(args.only)]
    if not videos:
        say(f"no .mp4 under {args.videos}")
        return 2
    profile = args.profile or ("12gb" if args.provider in ("CUDAExecutionProvider", "TensorrtExecutionProvider") else "cpu")
    config = RuntimeConfig(provider=args.provider, hardware_profile=profile)
    root = application_root()
    receipt = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": "scripts/verify_e2e_test1.py",
               "command": [sys.executable, *sys.argv], "output_dir": str(output_dir), "input_dir": str(args.videos),
               "runtime": {"provider": config.provider, "hardware_profile": config.hardware_profile},
               "parameters": {"texture_size": args.texture_size, "restoration_weight": args.restoration_weight,
                              "detection_threshold": args.detection_threshold},
               "videos": [], "passed": False}
    exit_code = 2
    try:
        pending = [v for v in videos if not (args.resume and load_row(v, output_dir) is not None)]
        loaded = catalog = None
        if pending:
            catalog = ModelCatalog(root)
            catalog.rescan_local()
            archive = args.facesets / f"{args.faceset}.fsz"
            if not archive.is_file():
                raise FileNotFoundError(f"faceset archive not found: {archive}")
            say(f"FACESET {args.faceset} ({archive})")
            loaded, summary = load_source(root, catalog, args.faceset)
            if Path(summary["archive"]).resolve() != archive.resolve():
                raise RuntimeError(f"the production resolver loaded {summary['archive']}, not {archive}")
            say(f"  {summary['views']} views, dense68 on {summary['views_with_dense68']}, landmark models "
                f"{summary['landmark_models']}, bank coverage {(summary['bank'] or {}).get('coverage')}")
            if not summary["views_with_dense68"]:
                raise RuntimeError("the faceset has no dense68 view: nothing to reconstruct from")
            receipt["faceset"] = summary
        for video in videos:
            row = load_row(video, output_dir) if args.resume else None
            if row is None:
                row = process_video(video, output_dir, loaded, catalog, config, args)
                row["faceset"] = receipt["faceset"]
                row_path(video).write_text(json.dumps(_jsonable(row), indent=1), "utf-8")
            else:
                say(f"=== {video.name}: reusing finished row ({row_path(video).name})")
            receipt.setdefault("faceset", row.get("faceset", {}))
            receipt["videos"].append(row)
            atomic_json(root, RECEIPT, _jsonable(receipt))                        # a crash later still leaves what finished
        receipt["passed"] = bool(receipt["videos"]) and all(r["passed"] for r in receipt["videos"])
        exit_code = 0 if receipt["passed"] else 1
    except BaseException as error:
        if isinstance(error, KeyboardInterrupt):
            raise
        receipt["fatal"] = {"kind": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        say(receipt["fatal"]["traceback"])
    # The report always covers every video that has a finished row, so a one-video invocation still prints the batch so far.
    done = {r["video"] for r in receipt["videos"]}
    for video in every:
        extra = load_row(video, output_dir) if video.name not in done else None
        if extra is not None:
            receipt["videos"].append(extra)
    receipt["videos"].sort(key=lambda r: natural(Path(r["video"])))
    receipt["passed"] = bool(receipt["videos"]) and all(r["passed"] for r in receipt["videos"]) and "fatal" not in receipt
    if "fatal" not in receipt:
        exit_code = 0 if receipt["passed"] else 1
    receipt["seconds"] = round(time.perf_counter() - began, 1)
    receipt.setdefault("faceset", {})
    report = render_report(receipt) if receipt["videos"] else "no video finished"
    receipt["report_text"] = report
    atomic_json(root, RECEIPT, _jsonable(receipt))
    (output_dir / "stage5_verification_report.txt").write_text(report + "\n", "utf-8")
    (output_dir / "stage5_verification_report.json").write_text(json.dumps(_jsonable(receipt), indent=1), "utf-8")
    say(report)
    say(f"\nreceipt: app/{RECEIPT}   report: {output_dir / 'stage5_verification_report.txt'}   ({receipt['seconds']}s)")
    return exit_code


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return str(value)
    return value


if __name__ == "__main__":
    sys.exit(main())
