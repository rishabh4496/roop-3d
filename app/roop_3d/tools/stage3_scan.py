"""Per-frame raw-detection cache for tracker development and replay (Stage 3).

Run from app/:
    roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.stage3_scan [--videos DIR] [--only s.mp4 ...]

Every decoded frame of each video goes through the production detector/recognizer stack (``FaceTracker.process``);
the raw per-frame ``FaceObservation`` list the tracker saw (box, score, landmarks, ArcFace embedding, pose, 6-DoF
translation) and the scene-cut flag are written to ``roop_3d/.runtime/stage3_cache/<video>.npz`` so association can be
replayed deterministically without re-running inference. The IDs the tracker produced in this pass are stored as the
``live_ids`` baseline.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.engine.detection.face_tracker import FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy
from roop_3d.identity.consolidation import TrackletRecorder
from roop_3d.media.service import MediaService
from roop_3d.media.video_reader import VideoReader
from roop_3d.models.catalog import ModelCatalog
from roop_3d.tools.verify_stage2 import stage_video

CACHE = "roop_3d/.runtime/stage3_cache"


def natural(path: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", path.name)]


def scan_video(service: MediaService, root: Path, source: Path, threshold: float = 0.4):
    """Run the production stack on every frame; returns (cache path, tracklet recorder, {frame: [(raw id, box)]})."""
    relative = stage_video(root, source)
    path = service.path(relative)
    captured: list = []
    tracker = FaceTracker(service.detector.catalog, threshold=threshold, detector=service.detector,
                          policy=TrackPolicy(), stabilize=False)
    original = tracker.observe

    def observe(image, **kwargs):
        captured[:] = original(image, **kwargs)
        return captured

    tracker.observe = observe
    rows: list[dict] = []
    recorder = TrackletRecorder()
    tracked_by_frame: dict[int, list] = {}
    frames_meta: list[tuple[float, bool, int, int]] = []
    began = time.perf_counter()
    try:
        with closing(VideoReader(path).frames()) as frames:
            for index, frame in enumerate(frames):
                image = frame.image
                tracked = tracker.process(image)
                recorder.observe(index, tracked)
                tracked_by_frame[index] = [(t.track_id, np.asarray(t.observation.box, np.float32).copy()) for t in tracked]
                frames_meta.append((float(frame.timestamp.seconds), bool(tracker.scene_cut), image.shape[1], image.shape[0]))
                for obs in captured:
                    # The tracker may stabilise copies; match its output back by box value.
                    live = next((t.track_id for t in tracked if np.allclose(t.observation.box, obs.box, atol=1e-3)), -1)
                    pose6 = obs.pose6dof
                    rows.append(dict(frame=index, box=obs.box, score=obs.score, lm5=np.asarray(obs.landmarks5).reshape(-1),
                                     emb=obs.embedding, pose=np.asarray(obs.pose, np.float32),
                                     t3d=(np.asarray(pose6.t, np.float32) if pose6 is not None else np.full(3, np.nan, np.float32)),
                                     live=live))
                if index % 100 == 0:
                    print(f"  {source.name}: frame {index} ({time.perf_counter() - began:.0f}s)", flush=True)
    finally:
        tracker.release()
    out = root / CACHE / (source.stem + ".npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    np.savez_compressed(
        out,
        frame=np.array([r["frame"] for r in rows], np.int32),
        box=np.array([r["box"] for r in rows], np.float32).reshape(n, 4),
        score=np.array([r["score"] for r in rows], np.float32),
        lm5=np.array([r["lm5"] for r in rows], np.float32).reshape(n, 10),
        emb=np.array([r["emb"] for r in rows], np.float32).reshape(n, 512),
        pose=np.array([r["pose"] for r in rows], np.float32).reshape(n, 3),
        t3d=np.array([r["t3d"] for r in rows], np.float32).reshape(n, 3),
        live_ids=np.array([r["live"] for r in rows], np.int32),
        frame_time=np.array([m[0] for m in frames_meta], np.float64),
        frame_cut=np.array([m[1] for m in frames_meta], bool),
        size=np.array(frames_meta[0][2:] if frames_meta else (0, 0), np.int32),
        total_frames=np.int32(len(frames_meta)))
    print(f"DONE {source.name}: {len(frames_meta)} frames, {n} detections, "
          f"{len({r['live'] for r in rows if r['live'] >= 0})} live ids, {time.perf_counter() - began:.0f}s", flush=True)
    return out, recorder, tracked_by_frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path(r"G:\pinokio\roop-keep\single"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    root = application_root()
    catalog = ModelCatalog(root)
    catalog.rescan_local()
    service = MediaService(root, catalog)
    videos = sorted(args.videos.glob("*.mp4"), key=natural)
    if args.only:
        videos = [v for v in videos if v.name in set(args.only)]
    for video in videos:
        if (root / CACHE / (video.stem + ".npz")).exists():
            print(f"SKIP {video.name} (cached)")
            continue
        scan_video(service, root, video, args.threshold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
