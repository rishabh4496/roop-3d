"""Defect #3 saturation benchmark: serial vs asynchronous multi-stream pipeline, real models, real hardware counters.

    roop_3d/.venv/Scripts/python.exe -m roop_3d.tools.benchmark_async_pipeline CLIP.mp4 --frames 90 --mode both

Both modes execute the production :class:`ReferenceStages` (track / swap / composite / write); only the scheduler
differs.  GPU / CPU / PCIe numbers are sampled from NVML while the steady state runs (warm-up frames are excluded) and
the report says plainly whether the >80 % sustained-GPU target was met -- it never rounds a miss up to a pass.

``CLIP`` should be a clip the app can decode; make 1080p / 4K variants with ffmpeg (``-vf scale=1920:1080``).
Run from ``app/``.  Output goes to ``roop_3d/.runtime/benchmarks/``.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from roop_3d.async_pipeline import AsyncPipeline, HardwareMonitor, Item, PinnedFramePool, monitor  # noqa: E402
from roop_3d.async_pipeline.live import JpegEncoder, LiveFrameHub  # noqa: E402
from roop_3d.async_pipeline.reference import ReferenceStages  # noqa: E402

WARMUP = 3


def build_session(provider: str, profile: str, faceset: Path):
    from roop_3d.engine.runtime import RuntimeConfig
    from roop_3d.engine.volumetric.session import ReferencePipeline
    from roop_3d.faceset.faceset_reader import FaceSetReader
    from roop_3d.models.catalog import ModelCatalog
    session = ReferencePipeline(ModelCatalog(), config=RuntimeConfig(hardware_profile=profile, provider=provider))
    session.prepare_session(FaceSetReader().read(faceset), retarget="auto")
    return session


def run(mode: str, clip: Path, frames: int, session, queue: int, live: bool, scratch: Path, keep: int = 0) -> dict:
    from roop_3d.media.video_reader import VideoReader
    reader = VideoReader(clip, hardware="auto")
    request = SimpleNamespace(workstation_parameters=None, parameter_curves=None, parameters_at=lambda index: None)
    scratch.mkdir(parents=True, exist_ok=True)
    marks: list[float] = []
    kept: list[np.ndarray] = []

    def write_frame(index, image, shape):
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        (scratch / f"{index:06d}.png").write_bytes(encoded.tobytes())

    stages = ReferenceStages(session, request=request, working_transfer=lambda: reader.working.transfer, write_frame=write_frame)
    hub = None
    if live:
        hub = LiveFrameHub(JpegEncoder(quality=80, max_edge=1280))
        import asyncio
        loop = asyncio.new_event_loop()
        hub.subscribe(loop)       # a watcher is attached so publishing actually encodes
    hardware = monitor().acquire()
    pipeline = None
    swapped = 0
    began = time.perf_counter()
    try:
        source = reader.frames()
        if mode == "async":
            pipeline = AsyncPipeline("bench", None, queue_size=queue)
            first = {}

            def items():
                pool = None
                try:
                    for number, decoded in enumerate(source):
                        if number >= frames:
                            return
                        if pool is None:
                            pool = PinnedFramePool(decoded.image.shape, queue + 6, cancel=pipeline.cancel_event)
                            pipeline.add_pool(pool)
                        slot, buffer = pool.acquire()
                        np.copyto(buffer, decoded.image)
                        yield Item(index=decoded.index, payload=replace(decoded, image=buffer),
                                   cleanups=[lambda s=slot: pool.release(s)])
                finally:
                    source.close()

            pipeline.set_source(items())
            for name, function in stages.stages():
                pipeline.add_stage(name, function, stream=name != "write")
            with pipeline:
                for item in pipeline.results():
                    marks.append(time.perf_counter())
                    swapped += bool(item.meta["result"].replacements)
                    if len(kept) < keep:
                        kept.append(item.payload.copy())
                    if hub:
                        hub.publish(item.payload, frame_index=item.index, timestamp_s=item.index / 30)
                    item.release()
                    if len(marks) == WARMUP:
                        hardware.history.clear()           # sample only the steady state
        else:
            for number, decoded in enumerate(source):
                if number >= frames:
                    break
                item = Item(index=decoded.index, payload=decoded)
                for name, function in stages.stages():
                    item.payload = function(item)
                marks.append(time.perf_counter())
                swapped += bool(item.meta["result"].replacements)
                if len(kept) < keep:
                    kept.append(item.payload.copy())
                if hub:
                    hub.publish(item.payload, frame_index=item.index, timestamp_s=item.index / 30)
                if len(marks) == WARMUP:
                    hardware.history.clear()
            source.close()
    finally:
        snapshot = hardware.snapshot()
        hardware.release()
        if hub:
            hub.stop()
    steady = len(marks) - WARMUP
    wall = marks[-1] - marks[WARMUP - 1]
    gpu_samples = [h["gpu_util_pct"] for h in hardware.history if h.get("gpu_util_pct") is not None]
    return {
        "mode": mode, "frames": len(marks), "swapped_frames": swapped, "steady_fps": round(steady / wall, 3),
        "steady_wall_s": round(wall, 2), "total_wall_s": round(time.perf_counter() - began, 2),
        "gpu_util": {"mean": round(statistics.fmean(gpu_samples), 1) if gpu_samples else None,
                     "p10": sorted(gpu_samples)[int(0.1 * (len(gpu_samples) - 1))] if gpu_samples else None,
                     "min": min(gpu_samples) if gpu_samples else None, "samples": len(gpu_samples),
                     "share_of_samples_ge_80": round(sum(v >= 80 for v in gpu_samples) / len(gpu_samples), 3) if gpu_samples else None},
        "hardware_window": snapshot["window"], "hardware_backend": snapshot["backend"],
        "pipeline": pipeline.snapshot() if pipeline else None,
        "live_hub": hub.stats() if hub else None,
        "_frames": kept,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip", type=Path)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--mode", choices=("serial", "async", "both"), default="both")
    parser.add_argument("--queue", type=int, default=8)
    parser.add_argument("--provider", default="TensorrtExecutionProvider")
    parser.add_argument("--profile", default="12gb", choices=("6gb", "12gb"))
    parser.add_argument("--faceset", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="also publish every frame to the JPEG live hub")
    parser.add_argument("--label", default="")
    parser.add_argument("--verify", type=int, default=0, metavar="N",
                        help="keep the first N output frames of each mode and compare them pixel by pixel")
    args = parser.parse_args()
    from roop_3d.audit.project_paths import legacy_root
    faceset = args.faceset or legacy_root() / "app/facesets/akansha.fsz"
    info = cv2.VideoCapture(str(args.clip))
    resolution = f"{int(info.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(info.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
    info.release()
    report = {"clip": args.clip.name, "resolution": resolution, "frames_requested": args.frames,
              "provider": args.provider, "queue": args.queue, "runs": []}
    for mode in (("serial", "async") if args.mode == "both" else (args.mode,)):
        print(f"--- {mode} @ {resolution} ---", flush=True)
        session = build_session(args.provider, args.profile, faceset)
        try:
            result = run(mode, args.clip, args.frames, session, args.queue, args.live,
                         APP / f"roop_3d/.runtime/bench_{mode}", keep=args.verify)
        finally:
            session.release_session()
            for path in (APP / f"roop_3d/.runtime/bench_{mode}").glob("*.png"):
                path.unlink()
        report["runs"].append(result)
        gpu = result["gpu_util"]
        print(f"  {result['steady_fps']:.2f} fps   GPU util mean {gpu['mean']}%  p10 {gpu['p10']}%  "
              f">=80% in {gpu['share_of_samples_ge_80']} of samples", flush=True)
    if args.verify and len(report["runs"]) == 2:
        a, b = report["runs"][0]["_frames"], report["runs"][1]["_frames"]
        diffs = [np.abs(x.astype(np.int16) - y.astype(np.int16)) for x, y in zip(a, b)]
        report["equivalence"] = {
            "frames_compared": len(diffs), "identical_frames": sum(int(not d.any()) for d in diffs),
            "max_abs_diff": int(max(d.max() for d in diffs)) if diffs else None,
            "mean_abs_diff": round(float(np.mean([d.mean() for d in diffs])), 5) if diffs else None,
            "pixels_differing_pct": round(float(np.mean([(d > 0).mean() for d in diffs])) * 100, 4) if diffs else None}
        print("equivalence:", report["equivalence"], flush=True)
    for run_result in report["runs"]:
        run_result.pop("_frames", None)
    if len(report["runs"]) == 2:
        serial, asynchronous = report["runs"]
        report["speedup"] = round(asynchronous["steady_fps"] / serial["steady_fps"], 3)
        mean = asynchronous["gpu_util"]["mean"]
        report["target_gpu_ge_80_sustained"] = bool(mean is not None and mean >= 80
                                                   and (asynchronous["gpu_util"]["share_of_samples_ge_80"] or 0) >= 0.8)
        print(f"speedup x{report['speedup']}   sustained >80% GPU target met: {report['target_gpu_ge_80_sustained']}")
    out = APP / f"roop_3d/.runtime/benchmarks/async_pipeline_{args.label or resolution}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("report:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
