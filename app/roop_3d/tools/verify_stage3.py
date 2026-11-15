"""Stage 3 gate: persistent cross-frame target tracking and single-ID locking on every .mp4 of a directory.

Run from app/:
    roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage3 [--videos DIR] [--only s.mp4 ...] [--live]

* default: *replay* the per-frame detections cached by ``tools.stage3_scan`` through the production tracker association
  (``FaceTracker.assign`` with the identity-lock policy) and the global consolidation. Fast and deterministic.
* ``--live``: decode every frame and run the full production stack (detector -> ArcFace -> ``FaceTracker.process`` ->
  consolidation), refreshing the cache.
* ``--calibrate``: print the cosine statistics the merge thresholds in ``TrackPolicy`` are calibrated against.

The ground-truth proxy for "the target person" on a single-subject clip is, on every frame, the largest detection scoring
>= 0.5. A frame counts as tracked only if that detection carries the video's one canonical ID; an ID switch is a change
of canonical ID between consecutive target frames. The online column shows the same measures *before* consolidation.
Receipt: ``roop_3d/.runtime/stage3_verification.json``. Exit code 0 only when every video has exactly one target ID and
zero ID switches.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.storage import atomic_json
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker
from roop_3d.engine.detection.track_policy import TrackPolicy, yaw_bin
from roop_3d.identity.locked_target import LockedTargetBinder
from roop_3d.identity.consolidation import TrackletRecorder, consolidate
from roop_3d.tools.stage3_scan import CACHE, natural

TARGET_SCORE = 0.5


def load_cache(root: Path, stem: str) -> dict:
    return dict(np.load(root / CACHE / f"{stem}.npz"))


def observations_by_frame(data: dict) -> dict[int, list[FaceObservation]]:
    result: dict[int, list[FaceObservation]] = {}
    for k, frame in enumerate(data["frame"].tolist()):
        t = data["t3d"][k]
        pose6 = None if np.isnan(t).any() else SimpleNamespace(t=t.astype(np.float64))
        result.setdefault(frame, []).append(FaceObservation(
            data["box"][k].astype(np.float32), data["lm5"][k].reshape(5, 2).astype(np.float32),
            np.zeros((106, 2), np.float32), data["emb"][k].astype(np.float32),
            tuple(float(v) for v in data["pose"][k]), score=float(data["score"][k]), pose6dof=pose6))
    return result


def replay(data: dict, policy: TrackPolicy, threshold: float = 0.4):
    """Run the cached detections through the production association; returns (recorder, {frame: [(raw_id, box)]})."""
    tracker = FaceTracker(policy=policy, threshold=threshold, stabilize=False)
    recorder = TrackletRecorder()
    per_frame = observations_by_frame(data)
    tracked_by_frame: dict[int, list[tuple[int, np.ndarray]]] = {}
    for frame in range(int(data["total_frames"])):
        if data["frame_cut"][frame]:
            tracker.mark_scene_cut()
        tracker.frame_number = frame
        tracked = tracker.assign(per_frame.get(frame, []))
        recorder.observe(frame, tracked)
        tracked_by_frame[frame] = [(t.track_id, t.observation.box) for t in tracked]
    return recorder, tracked_by_frame, tracker


def target_boxes(data: dict) -> dict[int, np.ndarray]:
    """Ground-truth proxy: per frame, the largest detection scoring >= TARGET_SCORE."""
    frames, boxes, scores = data["frame"], data["box"], data["score"]
    target: dict[int, np.ndarray] = {}
    for frame in range(int(data["total_frames"])):
        rows = np.where((frames == frame) & (scores >= TARGET_SCORE))[0]
        if len(rows):
            areas = (boxes[rows, 2] - boxes[rows, 0]) * (boxes[rows, 3] - boxes[rows, 1])
            target[frame] = boxes[rows[int(np.argmax(areas))]]
    return target


def evaluate(name: str, data: dict, tracked_by_frame, consolidation) -> dict:
    total = int(data["total_frames"])
    target = target_boxes(data)
    raw_seq, final_seq, untracked = [], [], 0
    for frame in sorted(target):
        match = next((raw for raw, box in tracked_by_frame.get(frame, []) if np.allclose(box, target[frame], atol=1e-3)), None)
        if match is None:
            untracked += 1
            continue
        raw_seq.append(match)
        final_seq.append(consolidation.canonical(match))
    def switches(sequence):
        return sum(1 for a, b in zip(sequence, sequence[1:]) if a != b)
    counts: dict = {}
    for item in final_seq:
        counts[item] = counts.get(item, 0) + 1
    main = max((key for key in counts if key is not None), key=lambda key: counts[key], default=None)
    return {
        "video": name, "total_frames": total, "frames_with_target_detection": len(target),
        "tracked_target_frames": counts.get(main, 0) if main is not None else 0,
        "target_frames_without_track": untracked + counts.get(None, 0),
        "unique_target_ids": len({item for item in final_seq if item is not None}),
        "id_switches": switches(final_seq),
        "online_raw_ids_on_target": len(set(raw_seq)), "online_id_switches": switches(raw_seq),
        "raw_tracks": len(consolidation.canonical_of), "identities": consolidation.unique_ids,
        "canonical_ids": [identity.canonical_id for identity in consolidation.identities],
        "suppressed": {str(key): value for key, value in consolidation.suppressed.items()},
        "merges": [{"left": d.left, "right": d.right, "reason": d.reason, "cosine": None if d.cosine is None else round(d.cosine, 3),
                    "gap": d.gap_frames} for d in consolidation.decisions if d.merged],
        "face1_label": "Face 1" if main == 1 else (f"Face {main}" if main is not None else None),
    }


def bind_check(data: dict, policy: TrackPolicy, consolidation, *, prepass: bool, threshold: float = 0.4) -> dict:
    """Re-run association with a *fresh* tracker (its raw IDs differ from the first run's) through a ``LockedTargetBinder``
    built from the consolidated signatures, exactly as the render replays an inspection. Every target frame must carry the
    one canonical ID; ``prepass=False`` exercises the embedding/continuity fallback alone."""
    signatures = {identity.canonical_id: np.stack(([identity.centroid] if identity.centroid is not None else []) + list(identity.anchors))
                  for identity in consolidation.identities if identity.centroid is not None}
    binder = LockedTargetBinder(signatures, policy)
    per_frame = observations_by_frame(data)
    total = int(data["total_frames"])

    def run() -> dict[int, list]:
        tracker = FaceTracker(policy=policy, threshold=threshold, stabilize=False)
        out = {}
        for frame in range(total):
            if data["frame_cut"][frame]:
                tracker.mark_scene_cut()
            tracker.frame_number = frame
            faces = binder.bind(tracker.assign(per_frame.get(frame, [])), frame)
            out[frame] = [(face.track_id, face.observation.box) for face in faces]
        return out

    if prepass:
        binder.start_prepass()
        run()
        binder.finish_prepass()
    bound = run()
    target = target_boxes(data)
    sequence, unmatched = [], 0
    for frame in sorted(target):
        match = next((track for track, box in bound.get(frame, []) if np.allclose(box, target[frame], atol=1e-3)), None)
        if match is None:
            unmatched += 1
        else:
            sequence.append(match)
    expected = min(signatures) if signatures else None
    wrong = sum(1 for item in sequence if item != expected)
    return {"mode": "pre-pass table + fallback" if prepass else "embedding/continuity fallback only",
            "target_frames_bound_to_face_1": len(sequence) - wrong, "target_frames_bound_elsewhere": wrong + unmatched,
            "id_switches": sum(1 for a, b in zip(sequence, sequence[1:]) if a != b), "decisions": dict(binder.stats)}


def calibrate(root: Path, stems: list[str]) -> dict:
    """Cosine statistics: same-person across yaw bins within a video, different-person prototype cosine across videos."""
    protos: dict[str, dict[int, np.ndarray]] = {}
    within: dict[str, list[float]] = {}
    for stem in stems:
        data = load_cache(root, stem)
        frames, boxes, scores, emb, pose = (data[k] for k in ("frame", "box", "score", "emb", "pose"))
        chosen = []
        for frame in np.unique(frames):
            rows = np.where((frames == frame) & (scores >= TARGET_SCORE))[0]
            if len(rows):
                areas = (boxes[rows, 2] - boxes[rows, 0]) * (boxes[rows, 3] - boxes[rows, 1])
                chosen.append(rows[int(np.argmax(areas))])
        chosen = np.array(chosen, int)
        if not len(chosen):
            continue
        by_bin: dict[int, list[int]] = {}
        for k in chosen:
            by_bin.setdefault(yaw_bin(float(pose[k][0])), []).append(k)
        protos[stem] = {}
        for key, rows in by_bin.items():
            vector = emb[rows].mean(axis=0)
            protos[stem][key] = (vector / np.linalg.norm(vector))
        base = protos[stem].get(0)
        if base is not None:
            for key, rows in by_bin.items():
                within.setdefault(f"|bin|={abs(key)}", []).extend(float(emb[k] @ base) for k in rows)
    cross: dict[str, list[float]] = {}
    names = sorted(protos)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for ka, va in protos[a].items():
                for kb, vb in protos[b].items():
                    if abs(ka - kb) <= 1:
                        cross.setdefault(f"|bin|={min(abs(ka), abs(kb))}", []).append(float(va @ vb))
    def summary(values):
        arr = np.array(values)
        return {"n": len(values), "p50": round(float(np.percentile(arr, 50)), 3), "p95": round(float(np.percentile(arr, 95)), 3),
                "p99": round(float(np.percentile(arr, 99)), 3), "max": round(float(arr.max()), 3), "min": round(float(arr.min()), 3)}
    return {"same_person_obs_vs_frontal_prototype": {k: summary(v) for k, v in sorted(within.items())},
            "different_video_prototype_pairs": {k: summary(v) for k, v in sorted(cross.items())}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path(r"G:\pinokio\roop-keep\single"))
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--live", action="store_true", help="run the full production stack on every frame (slow)")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.4)
    args = parser.parse_args()
    root = application_root()
    videos = sorted(args.videos.glob("*.mp4"), key=natural)
    if args.only:
        videos = [v for v in videos if v.name in set(args.only)]
    stems = [v.stem for v in videos]
    if args.calibrate:
        print(json.dumps(calibrate(root, stems), indent=1))
        return 0
    policy = TrackPolicy()
    service = None
    if args.live:
        from roop_3d.media.service import MediaService
        from roop_3d.models.catalog import ModelCatalog
        from roop_3d.tools.stage3_scan import scan_video
        catalog = ModelCatalog(root)
        catalog.rescan_local()
        service = MediaService(root, catalog)
    receipt = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "mode": "live" if args.live else "replay",
               "policy_fingerprint": policy.fingerprint(), "target_proxy": f"largest detection with score >= {TARGET_SCORE}",
               "videos": []}
    failures = []
    for video in videos:
        began = time.perf_counter()
        live = None
        if service is not None:
            (root / CACHE / f"{video.stem}.npz").unlink(missing_ok=True)
            _, live_recorder, live_tracked = scan_video(service, root, video, args.threshold)
            live = (live_recorder, live_tracked)
        if not (root / CACHE / f"{video.stem}.npz").exists():
            failures.append(f"{video.name}: no cached detections (run tools.stage3_scan)")
            print(f"FAIL {video.name}: no cache")
            continue
        data = load_cache(root, video.stem)
        recorder, tracked_by_frame = live if live is not None else replay(data, policy, args.threshold)[:2]
        result = consolidate(recorder.tracklets(), policy)
        row = evaluate(video.name, data, tracked_by_frame, result)
        row["binding_prepass"] = bind_check(data, policy, result, prepass=True, threshold=args.threshold)
        row["binding_fallback_only"] = bind_check(data, policy, result, prepass=False, threshold=args.threshold)
        row["seconds"] = round(time.perf_counter() - began, 1)
        # The render always runs the pre-pass (``scan_locked_targets`` / the angle scan) before binding, so that variant is the
        # gate. The fallback-only variant (a frame the pre-pass table missed) is reported, not gated: a new tracker ID born
        # in a steep profile has no local appearance evidence and is deliberately left unbound rather than guessed.
        bound_ok = row["binding_prepass"]["id_switches"] == 0 and row["binding_prepass"]["target_frames_bound_elsewhere"] == 0
        row["passed"] = (row["unique_target_ids"] == 1 and row["id_switches"] == 0 and row["identities"] == 1
                         and row["target_frames_without_track"] == 0 and bound_ok)
        if not row["passed"]:
            failures.append(f"{video.name}: {row['unique_target_ids']} target ID(s), {row['id_switches']} switch(es), "
                            f"{row['identities']} identities, {row['binding_prepass']['target_frames_bound_elsewhere']} frames "
                            "not bound to Face 1")
        receipt["videos"].append(row)
    receipt["failures"], receipt["passed"] = failures, not failures
    atomic_json(root, "roop_3d/.runtime/stage3_verification.json", receipt)
    header = (f"{'video':8s} {'frames':>6s} {'target det':>10s} {'tracked':>8s} {'IDs':>4s} {'switches':>8s} | online: {'raw IDs':>7s} "
              f"{'switches':>8s} | locked binding (pre-pass / fallback): bound, elsewhere, switches")
    print(header)
    for row in receipt["videos"]:
        a, b = row["binding_prepass"], row["binding_fallback_only"]
        print(f"{row['video']:8s} {row['total_frames']:6d} {row['frames_with_target_detection']:10d} {row['tracked_target_frames']:8d} "
              f"{row['unique_target_ids']:4d} {row['id_switches']:8d} | {row['online_raw_ids_on_target']:7d} {row['online_id_switches']:8d} | "
              f"{a['target_frames_bound_to_face_1']}/{a['target_frames_bound_elsewhere']}/{a['id_switches']} , "
              f"{b['target_frames_bound_to_face_1']}/{b['target_frames_bound_elsewhere']}/{b['id_switches']}  "
              f"{row['identities']} identity {'PASS' if row['passed'] else 'FAIL'}")
    for failure in failures:
        print("FAILURE:", failure)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
