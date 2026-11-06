"""Stage 9 acceptance: persistent queue + checkpoint recovery + segment join + NVHardwareVideoWriter.

Scenario (real ffmpeg, real files, no mocks):
  1. Build a 3 s / 90 frame / 30 fps clip with a 440 Hz track and a keyframe every second.
  2. Queue a job that splits into 3 GOP-aligned segments, pause it from the queue in the MIDDLE of segment 2,
     "restart the app" (new QueueManager), resume, and confirm segment 0 was NOT re-rendered (same sha256) and
     the job continued from segment 1.
  3. Concatenate with ``-c copy`` + source audio and check frame count, duration, audio, and that every
     segment_XXXX.mp4 is still on disk afterwards.
  4. Trimmed job (In 0.5 s / Out 2.5 s): 60 frames and ~2 s of audio.
  5. NVHardwareVideoWriter: encoder choice (NVENC or reported CPU fallback) and a baked .cube LUT changes pixels
     while ``lut_bake=False`` does not.
Exit code 0 only if every check passes.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys

import av
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from roop_3d.job_queue import SegmentedRenderRunner, count_video_packets  # noqa: E402
from roop_3d.media.nv_writer import NVHardwareVideoWriter  # noqa: E402
from roop_3d.queue_manager import CanonicalJobState, QueueManager  # noqa: E402
from roop_3d.segment_dispatcher import find_ffmpeg  # noqa: E402

SCRATCH = ROOT / "app/roop_3d/.runtime/stage9_verify"
FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    print(("PASS " if condition else "FAIL ") + message)
    if not condition:
        FAILURES.append(message)


def make_clip(path: Path) -> None:
    subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=s=320x240:r=30:d=3",
                    "-f", "lavfi", "-i", "sine=f=440:d=3", "-c:v", "libx264", "-g", "30", "-keyint_min", "30",
                    "-sc_threshold", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True)


def duration_of(path: Path, kind: str) -> float:
    with av.open(str(path)) as c:
        s = c.streams.video[0] if kind == "video" else c.streams.audio[0]
        return float(s.duration * s.time_base) if s.duration else float(c.duration) / 1e6


def main() -> int:
    shutil.rmtree(SCRATCH, ignore_errors=True)
    media = SCRATCH / "media"
    media.mkdir(parents=True)
    clip = media / "target.mp4"
    make_clip(clip)
    root = SCRATCH / "root"
    root.mkdir()
    seen: list[str] = []

    def record(event: dict) -> None:
        if event.get("action") == "state_change":
            seen.append(event["state"])

    queue = QueueManager(root, notify=record)
    runner = SegmentedRenderRunner(queue, interrupt_poll_frames=1)

    # ---- 2. multi-segment job, pause mid-segment-2, restart, resume ------------------------------------------
    job, _ = queue.submit({"target_path": str(clip), "max_segment_seconds": 1.0, "output_name": "full.mp4",
                           "settings_3d": {"render_mode": "reenactment_3d", "pose_weight": 0.8},
                           "execution_provider": "CUDAExecutionProvider"}, name="stage9 multi-segment")
    check(job.state == CanonicalJobState.QUEUED, "job starts QUEUED")
    queue.notify = record
    calls = {"n": 0}

    def pausing_transform(frame, index):
        calls["n"] += 1
        if index == 45:  # frame 15 of segment 1 (0-based) -> mid-segment
            queue.pause(job.job_id)
        return frame

    first = runner.run(job.job_id, pausing_transform)
    check(first["state"] == "PAUSED", f"pause detected mid-render (got {first['state']})")
    ck = runner.load_checkpoint(job.job_id)
    check(ck is not None and len(ck.segments) == 3, "checkpoint holds a 3-segment manifest")
    check([s.status for s in ck.segments] == ["COMPLETED", "PENDING", "PENDING"], "only segment 0 finished before the pause")
    seg0 = root / ck.segments[0].path
    sha_before, mtime_before = ck.segments[0].sha256, seg0.stat().st_mtime_ns
    check(not list(seg0.parent.glob("*.part*")), "partial segment discarded, no .part left behind")
    check(ck.execution_provider == "CUDAExecutionProvider" and ck.settings_3d["pose_weight"] == 0.8,
          "checkpoint keeps settings and execution provider")
    check(ck.frame_progress == {"completed_frames": 30, "total_frames": 90}, f"frame progress {ck.frame_progress}")
    check(queue.get(job.job_id).checkpoint_path == runner.checkpoint_rel(job.job_id), "queue row points at job_checkpoint.json")

    queue = QueueManager(root, notify=record)  # simulated app restart: state must come back from SQLite
    runner = SegmentedRenderRunner(queue, interrupt_poll_frames=1)
    check(queue.get(job.job_id).state == CanonicalJobState.PAUSED, "PAUSED survives a restart")
    queue.resume(job.job_id)
    rendered_after: list[int] = []
    result = runner.run(job.job_id, lambda frame, index: (rendered_after.append(index), frame)[1])
    check(result["state"] == "COMPLETED", f"resumed job completes (got {result['state']})")
    check(min(rendered_after) == 30, f"resume started at frame {min(rendered_after)}, not 0")
    check(seg0.stat().st_mtime_ns == mtime_before and runner.load_checkpoint(job.job_id).segments[0].sha256 == sha_before,
          "segment_0000.mp4 untouched by the resume")

    # ---- 3. join verification ----------------------------------------------------------------------------------
    out = root / result["output_path"]
    frames, duration = count_video_packets(out), duration_of(out, "video")
    check(frames == 90, f"joined video has 90 frames (got {frames})")
    check(abs(duration - 3.0) < 0.1, f"joined duration {duration:.3f}s ~ 3.0s")
    with av.open(str(out)) as c:
        check(len(c.streams.audio) == 1, "source audio muxed into the final file")
    check(abs(duration_of(out, "audio") - 3.0) < 0.25, "audio length matches video")
    names = sorted(p.name for p in (root / runner.work_dir(job.job_id) / "segments").glob("*"))
    check(names == ["segment_0000.mp4", "segment_0001.mp4", "segment_0002.mp4"], f"all segments preserved: {names}")
    check(result["segments_preserved"] and result["verified"], "result reports verified + preserved")
    check(queue.get(job.job_id).state == CanonicalJobState.COMPLETED, "queue state COMPLETED")
    check({"PREFLIGHT", "ANALYZING", "RENDERING", "PAUSED", "MUXING", "COMPLETED"} <= set(seen), f"lifecycle states observed: {sorted(set(seen))}")
    try:
        SegmentedRenderRunner(queue).cleanup_segments("nonexistent")
        check(False, "cleanup refuses unknown job")
    except Exception:
        check(True, "cleanup refuses without a verified output")

    # ---- 4. timeline trim -----------------------------------------------------------------------------------------
    trim, _ = queue.submit({"target_path": str(clip), "max_segment_seconds": 1.0, "trim_in_seconds": 0.5,
                            "trim_out_seconds": 2.5, "output_name": "trim.mp4"}, name="stage9 trim")
    res = runner.run(trim.job_id)
    tout = root / res["output_path"]
    check(res["state"] == "COMPLETED" and count_video_packets(tout) == 60, f"trimmed render has 60 frames (got {count_video_packets(tout)})")
    check(abs(duration_of(tout, "audio") - 2.0) < 0.25, f"trimmed audio ~2.0 s (got {duration_of(tout, 'audio'):.3f})")

    # ---- retry path through RETRYING ---------------------------------------------------------------------------
    bad, _ = queue.submit({"target_path": str(media / "missing.mp4")}, name="stage9 failing")
    try:
        runner.run(bad.job_id)
    except Exception:
        pass
    check(queue.get(bad.job_id).state == CanonicalJobState.FAILED, "missing media -> FAILED")
    queue.transition(bad.job_id, "RETRYING")
    check(queue.get(bad.job_id).state == CanonicalJobState.RETRYING, "FAILED -> RETRYING allowed")
    queue.cancel(bad.job_id)
    check(queue.get(bad.job_id).state == CanonicalJobState.CANCELLED, "per-job cancel works")

    # ---- 5. NVHardwareVideoWriter -------------------------------------------------------------------------------
    cube = media / "invert.cube"
    cube.write_text("LUT_3D_SIZE 2\n" + "".join(f"{1 - r} {1 - g} {1 - b}\n" for b in (0, 1) for g in (0, 1) for r in (0, 1)))
    white = np.full((240, 320, 3), 235, np.uint8)

    def encode(name: str, **kw) -> tuple[float, NVHardwareVideoWriter]:
        target = media / name
        with NVHardwareVideoWriter(target, 320, 240, 30, **kw) as writer:
            for _ in range(12):
                writer.write(white)
        with av.open(str(target)) as c:
            mean = float(next(c.decode(video=0)).to_ndarray(format="rgb24").mean())
        return mean, writer

    plain_mean, plain = encode("plain.mp4")
    print(f"INFO encoder={plain.encoder} fallback_reason={plain.fallback_reason}")
    check(plain.encoder in ("h264_nvenc", "libx264"), "encoder is NVENC or the libx264 fallback")
    check(plain.encoder == "h264_nvenc" or plain.fallback_reason, "CPU fallback is explained")
    check(plain_mean > 200, f"no-LUT output stays bright ({plain_mean:.0f})")
    baked_mean, baked = encode("baked.mp4", lut_path=cube, lut_bake=True)
    check(baked.lut_applied and baked_mean < 60, f"baked LUT inverts pixels ({baked_mean:.0f})")
    kept_mean, kept = encode("kept.mp4", lut_path=cube, lut_bake=False)
    check(not kept.lut_applied and kept_mean > 200 and (media / "kept.cube").is_file(), "lut_bake=False leaves pixels, ships .cube sidecar")
    cpu_mean, cpu = encode("cpu.mp4", prefer_hardware=False)
    check(cpu.encoder == "libx264" and cpu_mean > 200, "forced CPU path works")
    hevc_mean, hevc = encode("hevc.mp4", codec="hevc")
    check(hevc.encoder in ("hevc_nvenc", "libx265") and hevc_mean > 200, f"HEVC encodes with {hevc.encoder}")

    print(f"\n{'ALL CHECKS PASSED' if not FAILURES else str(len(FAILURES)) + ' CHECK(S) FAILED'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(SCRATCH / "root", ignore_errors=True)
    sys.exit(code)
