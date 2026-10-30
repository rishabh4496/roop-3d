"""Real 100-frame reference campaign; all media and biometric data remain local.

Example: python -B -u -m roop_3d.tests.manual_stage06_acceptance --provider cuda --profile 6gb
This writes ignored artifacts only. It never changes source media or model files.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import gc
import os
from pathlib import Path
import time
import tracemalloc

import av
import cv2
import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.core.storage import atomic_bytes, atomic_json, owned
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.preprocess import transform_points
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.media.probe import probe
from roop_3d.media.video_writer import encode_video


def process_memory() -> dict[str, int]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                          "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                          "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage", "PrivateUsage")]]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        current = ctypes.windll.kernel32.GetCurrentProcess
        current.restype = wintypes.HANDLE
        call = ctypes.windll.psapi.GetProcessMemoryInfo
        call.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        call.restype = wintypes.BOOL
        if not call(current(), ctypes.byref(value), value.cb):
            raise OSError("Process memory measurement failed")
        return {"private_bytes": value.PrivateUsage, "resident_bytes": value.WorkingSetSize}
    values = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    return {"private_bytes": int(values["RssAnon"].split()[0]) * 1024,
            "resident_bytes": int(values["VmRSS"].split()[0]) * 1024}


def make_input(path: Path, image: np.ndarray, frames: int = 100) -> None:
    with av.open(str(path), "w") as container:
        video = container.add_stream("ffv1", rate=10)
        video.width, video.height, video.pix_fmt = image.shape[1], image.shape[0], "yuv420p"
        video.codec_context.time_base = Fraction(1, 1000)
        audio = container.add_stream("pcm_s16le", rate=48000)
        audio.layout = "mono"
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base, frame.duration = index * 100, Fraction(1, 1000), 100
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode(None):
            container.mux(packet)
        # Non-silent generated audio makes accidental track loss observable.
        for index in range(frames):
            signal = (np.sin(np.arange(4800) * 2 * np.pi * 440 / 48000) * 4000).astype(np.int16)[None]
            frame = av.AudioFrame.from_ndarray(signal, format="s16", layout="mono")
            frame.sample_rate, frame.time_base, frame.pts = 48000, Fraction(1, 48000), index * 4800
            for packet in audio.encode(frame):
                container.mux(packet)
        for packet in audio.encode(None):
            container.mux(packet)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--profile", choices=("cpu", "6gb", "12gb"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    arguments = parser.parse_args()
    root = application_root()
    campaign = f"stage06-{arguments.provider}-{arguments.profile}"
    input_path = owned(root, f"roop_3d/.runtime/acceptance/{campaign}-input.mkv")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    output_name = f"{campaign}-aligned-preview.mov"
    # New outputs are required; unique campaign runs use a timestamp suffix.
    output_name = output_name.replace(".mov", f"-{time.time_ns()}.mov")
    source_path = legacy_root() / "app/facesets/akansha.fsz"
    target_path = legacy_root() / "app/facesets/anushree.fsz"
    before = {"source": sha256_file(source_path)[1], "target": sha256_file(target_path)[1]}
    reader = FaceSetReader()
    source, target = reader.read(source_path), reader.read(target_path)
    view = min(target.views, key=lambda item: abs(item.yaw))
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    make_input(input_path, image)
    config = RuntimeConfig(provider="CUDAExecutionProvider" if arguments.provider == "cuda" else "CPUExecutionProvider",
                           hardware_profile=arguments.profile, cpu_threads=arguments.cpu_threads)
    session = ReferencePipeline(config=config)
    samples: list[dict] = []
    began = time.perf_counter()
    try:
        session.prepare_session(source, retarget="required")
        for _ in range(3):
            initial = session.infer_frame(image)
        face = initial.observations[0].observation
        before_embedding = session.tracker.recognizer.embed(image, face.landmarks5)
        before_cosine = float(np.dot(before_embedding, source.canonical_embedding))
        del initial
        gc.collect()
        tracemalloc.start()
        python_baseline = tracemalloc.get_traced_memory()[0]
        last = None
        def transform(decoded):
            nonlocal last
            last = session.infer_frame(decoded.image)
            assert len(last.replacements) == 1 and last.replacements[0].track_id == 0
            assert len(session.portrait.tracks) == 1 and len(session.tracker.active_ids) == 1
            memory = process_memory()
            samples.append({"frame": decoded.index, **memory, "latency": asdict(last.latency),
                            "python_live_bytes": tracemalloc.get_traced_memory()[0]})
            if len(samples) % 10 == 0:
                print(f"{len(samples)}/100 real frames; private={memory['private_bytes'] // 1048576} MiB; cached_tracks=1", flush=True)
            return last.replacements[0].image
        manifest = encode_video(input_path, output_name, transform=transform, hardware="software")
        assert len(samples) == manifest["verified_frames"] == 100
        assert len(manifest["audio_tracks"]) == 1 and manifest["audio_tracks"][0]["action"] == "copy"
        output_path = owned(root, f"roop_3d/outputs/{output_name}")
        assert len(probe(output_path).audio) == 1
        output = last.replacements[0]
        points = transform_points(last.observations[0].observation.landmarks5,
                                  cv2.invertAffineTransform(output.inverse)).astype(np.float32)
        after_embedding = session.tracker.recognizer.embed(output.image, points)
        after_cosine = float(np.dot(after_embedding, source.canonical_embedding))
        assert after_cosine > before_cosine + 0.2
        memory = [sample["private_bytes"] for sample in samples]
        start_window, end_window = np.median(memory[20:40]), np.median(memory[-20:])
        native_growth = int(end_window - start_window)
        # Compare the same point of the live encode loop in both windows.
        # The cold baseline has no decoder/encoder frames in flight yet.
        python_start = np.median([sample["python_live_bytes"] for sample in samples[20:40]])
        python_end = np.median([sample["python_live_bytes"] for sample in samples[-20:]])
        python_growth = int(python_end - python_start)
        assert native_growth <= 32 * 1048576, f"Private memory failed plateau gate: {native_growth} bytes"
        assert python_growth <= 2 * 1048576, f"Retained Python objects grew by {python_growth} bytes"
        capabilities = session.capability()
        assert capabilities["portrait"]["models"]["warp_spade"]["executions"] == 103
        assert capabilities["identity_swapper"]["executions"] == 103
        assert all(sha256_file(path)[1] == before[key] for key, path in (("source", source_path), ("target", target_path)))
        for model in [session.swapper.model, *session.portrait.models.values(), session.tracker.dense.model106,
                      session.tracker.dense.model203]:
            model.catalog.verify(model.record)
        ok, jpeg = cv2.imencode(".jpg", np.hstack((cv2.resize(image, (512, 512)), output.image)))
        assert ok
        preview = f"roop_3d/.runtime/acceptance/{campaign}-comparison.jpg"
        atomic_bytes(root, preview, jpeg.tobytes())
        report = {"status": "passed", "timestamp": datetime.now(timezone.utc).isoformat(),
                  "frames": 100, "warmup_frames": 3, "profile": arguments.profile, "provider": arguments.provider,
                  "cpu_threads": arguments.cpu_threads, "source_sha256": before["source"], "target_sha256": before["target"],
                  "source_cosine_before": before_cosine, "source_cosine_after": after_cosine,
                  "private_memory_growth_bytes": native_growth, "python_live_growth_bytes": python_growth,
                  "python_cold_baseline_bytes": python_baseline,
                  "memory_comparison_windows": {"start_frames": [20, 39], "end_frames": [80, 99]},
                  "private_plateau_gate_bytes": 32 * 1048576, "python_live_gate_bytes": 2 * 1048576,
                  "samples": samples, "capabilities": capabilities,
                  "gpu_budget_bytes": config.gpu_budget_bytes if arguments.provider == "cuda" else 0,
                  "output_manifest": manifest, "output": str(output_path.relative_to(root)),
                  "comparison": preview, "scope": "Aligned portrait preview; full-frame Stage 09 compositor is not claimed",
                  "elapsed_seconds": time.perf_counter() - began,
                  "note": "No growth beyond defined bounds across this 100-frame campaign; not a proof for arbitrary inputs or durations"}
        atomic_json(root, f"roop_3d/.runtime/acceptance/{campaign}.json", report)
        print({key: report[key] for key in ("status", "frames", "source_cosine_before", "source_cosine_after",
                                            "private_memory_growth_bytes", "python_live_growth_bytes", "elapsed_seconds")}, flush=True)
    finally:
        tracemalloc.stop()
        session.release_session()
        assert not session.prepared and not session.portrait.tracks and not session.portrait.models


if __name__ == "__main__":
    main()
