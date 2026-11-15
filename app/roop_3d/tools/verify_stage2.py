"""Stage 2 gate: whole-timeline target detection on every .mp4 of a directory + `harjot` reference-bank loading.

Run from app/:
    roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage2 [--videos DIR] [--faceset harjot]

Real execution only: the production ``MediaService.inspect`` (adaptive sampling, robust multi-scale detector, ArcFace,
tracker) and the production FaceSet loader. Videos must live under ``roop_3d/.runtime/media`` for the service to accept them;
files from another directory are hard-linked there (copied when the volumes differ), never modified.
Exit code 0 only when every video produced at least one real detection and the bank loaded. Receipt:
``roop_3d/.runtime/stage2_verification.json``.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json
from roop_3d.faceset.locator import resolve_faceset
from roop_3d.faceset_loader import LandmarkEngine, load_faceset
from roop_3d.media.schema import InspectRequest
from roop_3d.media.service import MediaService
from roop_3d.models.catalog import ModelCatalog

ANGLED_YAW = 25.0         # |yaw| >= 25: three-quarter
PROFILE_YAW = 55.0        # |yaw| >= 55: steep profile


def stage_video(root: Path, source: Path) -> str:
    target = root / "roop_3d/.runtime/media/stage2" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        target.unlink(missing_ok=True)
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    return target.relative_to(root).as_posix()


def bank_scores(service: MediaService, bank, relative: str, inspection) -> list[dict]:
    """Score each track's most-turned and most-frontal real observation against the harjot reference bank."""
    from roop_3d.core.pose import as_pose3d
    out = []
    for track in inspection.tracks:
        picks = {"frontal": min(track.observations, key=lambda o: abs(o.yaw_degrees), default=None),
                 "turned": max(track.observations, key=lambda o: abs(o.yaw_degrees), default=None)}
        for label, observation in picks.items():
            if observation is None:
                continue
            image = service.frames.read(relative, frame_index=observation.frame).image
            height, width = image.shape[:2]
            faces = service.analyzer.analyze(image, thumbnails=False)
            best = None
            for face in faces:
                x1, y1, x2, y2 = (float(v) for v in face.box)
                box = (x1 / width, y1 / height, x2 / width, y2 / height)
                overlap = (min(box[2], observation.bbox[2]) - max(box[0], observation.bbox[0])) * \
                          (min(box[3], observation.bbox[3]) - max(box[1], observation.bbox[1]))
                if overlap > 0 and (best is None or overlap > best[0]):
                    best = (overlap, face)
            if best is None or best[1].embedding is None:
                out.append({"track_id": track.track_id, "view": label, "frame": observation.frame, "scored": False})
                continue
            pose = as_pose3d(best[1].pose_legacy)
            score = bank.similarity(best[1].embedding, pose.yaw, pose.pitch)
            out.append({"track_id": track.track_id, "view": label, "frame": observation.frame, "scored": True,
                        "yaw": round(float(pose.yaw), 1), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in score.items()}})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path(r"G:\pinokio\roop-keep\single"))
    parser.add_argument("--faceset", default="harjot")
    parser.add_argument("--budget", type=float, default=900.0, help="per-video scan time budget in seconds")
    parser.add_argument("--only", nargs="*", help="video file names to run (default: all .mp4)")
    args = parser.parse_args()
    root = application_root()
    catalog = ModelCatalog(root)
    catalog.rescan_local()
    receipt: dict = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "videos": [], "faceset": None}
    failures: list[str] = []

    # ------------------------------------------------------------------------------------------ FaceSet bank
    started = time.perf_counter()
    try:
        resolved = resolve_faceset(root, args.faceset)
        engine = LandmarkEngine(root)
        loaded = load_faceset(resolved.archive, catalog=catalog, root=root, engine=engine, faceset_id=resolved.faceset_id)
        if loaded.bank is None:
            raise RuntimeError("; ".join(loaded.notes) or "no reference bank")
        bank = loaded.bank
        summary = bank.summary()
        receipt["faceset"] = {"id": resolved.faceset_id, "archive": str(resolved.archive), "source": resolved.source,
                              "views_in_archive": len(loaded.document.views), "seconds": round(time.perf_counter() - started, 2),
                              "bank": summary, "notes": loaded.notes}
        print(f"FACESET {resolved.faceset_id} <- {resolved.archive} ({resolved.source}): {len(loaded.document.views)} views, "
              f"{summary['usable']} usable, rejected={summary['rejected']}, coverage={summary['coverage']}")
        for ref in summary["references"]:
            print(f"  view {ref['view_index']}: yaw {ref['yaw']:+.1f} pitch {ref['pitch']:+.1f} roll {ref['roll']:+.1f} "
                  f"quality {ref['quality']:.3f} centroid_cos {ref['centroid_cosine']:.3f} [{ref['pose_source']}]")
        print(f"  centroid |c| = {summary['centroid_norm']}   UV view order = {summary['uv_view_order']}")
    except Exception as error:
        failures.append(f"faceset {args.faceset}: {type(error).__name__}: {error}")
        print(f"FAIL faceset {args.faceset}: {type(error).__name__}: {error}", flush=True)
        bank = None

    # --------------------------------------------------------------------------------------------- videos
    service = MediaService(root, catalog)
    videos = sorted(args.videos.glob("*.mp4"), key=lambda p: [int(t) if t.isdigit() else t for t in
                                                              __import__("re").split(r"(\d+)", p.name)])
    if args.only:
        videos = [v for v in videos if v.name in set(args.only)]
    if not videos:
        failures.append(f"no .mp4 under {args.videos}")
    for video in videos:
        row: dict = {"video": video.name, "error": None}
        began = time.perf_counter()
        try:
            relative = stage_video(root, video)
            inspection = service.inspect(InspectRequest(path=relative, identity_lock=True, sampling="adaptive",
                                                        time_budget_seconds=args.budget))
            observations = [o for t in inspection.tracks for o in t.observations]
            yaws = np.array([abs(o.yaw_degrees) for o in observations]) if observations else np.zeros(0)
            row.update(decoded_frames=inspection.decoded_frames, analysed_frames=inspection.sampled_frames,
                       truncated=inspection.truncated, tracks=len(inspection.tracks), detections=len(observations),
                       frontal=int((yaws < ANGLED_YAW).sum()), angled_3q=int(((yaws >= ANGLED_YAW) & (yaws < PROFILE_YAW)).sum()),
                       profile=int((yaws >= PROFILE_YAW).sum()),
                       yaw_range=[round(min(o.yaw_degrees for o in observations), 1), round(max(o.yaw_degrees for o in observations), 1)]
                       if observations else None,
                       min_confidence=round(min(o.confidence for o in observations), 3) if observations else None,
                       coverage_seconds=[round(min(t.first_timestamp_seconds for t in inspection.tracks), 2),
                                         round(max(t.last_timestamp_seconds for t in inspection.tracks), 2)] if inspection.tracks else None,
                       warning=next((w for w in inspection.warnings if w.startswith(("Adaptive", "Bounded"))), None),
                       inspection_id=inspection.inspection_id)
            if not observations:
                row["error"] = "no face detected in any sampled frame"
            elif inspection.truncated:
                row["error"] = "scan truncated before the end of the video"
            if bank is not None and observations:
                row["bank_match"] = bank_scores(service, bank, relative, inspection)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["seconds"] = round(time.perf_counter() - began, 1)
        receipt["videos"].append(row)
        if row["error"]:
            failures.append(f"{video.name}: {row['error']}")
        status = "FAIL" if row["error"] else "PASS"
        if row.get("detections") is not None:
            print(f"{status} {video.name}: {row['detections']} detections over {row['analysed_frames']}/{row['decoded_frames']} frames, "
                  f"{row['tracks']} track(s); frontal {row['frontal']}, 3/4 {row['angled_3q']}, profile {row['profile']}; "
                  f"yaw {row['yaw_range']}; span {row['coverage_seconds']}s; {row['seconds']}s" + (f"  !! {row['error']}" if row["error"] else ""),
                  flush=True)
        else:
            print(f"{status} {video.name}: {row['error']}", flush=True)
        for match in row.get("bank_match", []):
            if match["scored"]:
                print(f"    {match['track_id']} {match['view']:7s} frame {match['frame']:5d} yaw {match['yaw']:+6.1f}: "
                      f"centroid {match['centroid_cosine']:+.3f} pose-matched {match['pose_matched_cosine']:+.3f} score {match['score']:+.3f}")
    receipt["failures"] = failures
    receipt["passed"] = not failures
    atomic_json(root, "roop_3d/.runtime/stage2_verification.json", receipt)
    total = sum(v.get("detections") or 0 for v in receipt["videos"])
    print(f"\n{len(receipt['videos'])} video(s), {total} real detections, "
          f"{sum(1 for v in receipt['videos'] if v['error'])} failed; faceset bank {'OK' if bank is not None else 'FAILED'}.")
    for failure in failures:
        print("  FAILURE:", failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
