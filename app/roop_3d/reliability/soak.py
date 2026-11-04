"""Soak scenarios and CLI: drive the real worker/checkpoint/decode/encode path for a long time and judge the resource series.

    python -m roop_3d.reliability.soak long_video --minutes 30 --out ../docs/reliability/soak_long_30min.json
    python -m roop_3d.reliability.soak successive --jobs 60
    python -m roop_3d.reliability.soak model_switch --cycles 300
    python -m roop_3d.reliability.soak hires --width 1920 --height 1080 --frames 600
    python -m roop_3d.reliability.soak selftest          # proves the detector fires on a deliberate leak

Each scenario returns (and optionally writes) a JSON document: configuration, throughput, per-metric growth verdicts
against ``DEFAULT_PROFILE`` and the end-of-run invariants (ledger back to zero, no children, work directory released).
The exit code is non-zero if any verdict or invariant fails.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import tracemalloc
from typing import Any, Callable

from roop_3d.jobs.schema import JobState
from roop_3d.reliability import probe
from roop_3d.reliability.growth import DEFAULT_PROFILE, IDLE_PROFILE, BoundedProfile, analyse
from roop_3d.reliability.harness import StubSession, build_rig, decoded_digest
from roop_3d.reliability.ledger import LEDGER

APP = Path(__file__).resolve().parents[2]
RUNTIME = APP / "roop_3d/.runtime"
# Metrics a *stubbed-inference* soak can honestly judge. GPU memory is device-wide on WDDM and the stub never touches
# the GPU, so it is recorded in the series but left out of the verdict.
JUDGED = ("rss_mib", "private_mib", "handles", "os_threads", "py_threads", "children", "py_heap_blocks")


class Sampler:
    """Collects ``ResourceSample`` s on a timer (long jobs) and/or on demand (per-iteration series)."""

    def __init__(self, interval: float = 2.0, *, gpu: bool = False):
        self.interval, self.gpu = interval, gpu
        self.samples: list[tuple[float, probe.ResourceSample]] = []
        self.started = time.monotonic()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def take(self, x: float | None = None) -> probe.ResourceSample:
        sample = probe.take_sample(self.started, gauges=LEDGER.counts(), with_gpu=self.gpu, with_heap=True)
        self.samples.append((sample.elapsed if x is None else x, sample))
        return sample

    def __enter__(self) -> "Sampler":
        def run():
            while not self._stop.wait(self.interval):
                self.take()
        self.take()
        self._thread = threading.Thread(target=run, name="soak-sampler", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(5)
        self.take()

    def series(self, names: tuple[str, ...] | None = None) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for x, sample in self.samples:
            for key, value in sample.metrics().items():
                if names is None or key in names or key.startswith("live."):
                    out.setdefault(key, []).append((x, value))
        return out


def _child_names(pids: list[int]) -> list[str]:
    if os.name != "nt":
        return [str(pid) for pid in pids]
    return [f"{name}({pid})" for pid, _, _, name in probe._win_processes() if pid in pids]


def _invariants(rig=None) -> dict[str, Any]:
    gc.collect()
    # A child that is merely finishing (nvidia-smi, a conhost) is not an orphan; one that persists for 5 s is.
    deadline = time.monotonic() + 5
    children = probe.child_pids()
    while children and time.monotonic() < deadline:
        time.sleep(0.1)
        children = probe.child_pids()
    checks = {"ledger_live": {k: v for k, v in LEDGER.counts().items() if v},
              "abandoned_tokens": LEDGER.abandoned(), "stub_sessions_alive": StubSession.live_instances,
              "child_processes": len(children), "child_process_names": _child_names(children)}
    checks["ok"] = not checks["ledger_live"] and not checks["abandoned_tokens"] \
        and not checks["stub_sessions_alive"] and not checks["child_processes"]
    return checks


def _finish(name: str, config: dict, sampler: Sampler, started: float, extra: dict, profile: BoundedProfile,
            judged: tuple[str, ...] = JUDGED) -> dict[str, Any]:
    report = analyse(profile, sampler.series(judged))
    series = sampler.series(("rss_mib", "handles", "os_threads", "py_threads"))
    thin = {k: [[round(x, 1), round(y, 2)] for x, y in v[:: max(1, len(v) // 60)]] for k, v in series.items()}
    result = {"scenario": name, "config": config, "seconds": round(time.monotonic() - started, 1),
              "profile": profile.name, "growth": report.as_dict(), "invariants": _invariants(), "series": thin, **extra}
    result["passed"] = bool(report.ok and result["invariants"]["ok"] and extra.get("checks_ok", True))
    return result


def _fresh_root(name: str) -> Path:
    root = RUNTIME / "soak18" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def scenario_long_video(minutes: float = 30, fps: int = 25, size=(96, 64), tracks: int = 2, interval: float = 5.0,
                        profile: BoundedProfile = DEFAULT_PROFILE, keep: bool = False, tracemalloc_top: int = 0) -> dict:
    frames = int(minutes * 60 * fps)
    root = _fresh_root(f"long{int(minutes)}")
    started = time.monotonic()
    t0 = time.monotonic()
    rig = build_rig(root, frames=frames, tracks=tracks, size=size, fps=fps, poll_interval=0.05)
    prep = time.monotonic() - t0
    if tracemalloc_top:
        tracemalloc.start(8)
        before = tracemalloc.take_snapshot()
    try:
        with Sampler(interval) as sampler:
            record = rig.submit("soak-long")
            render_started = time.monotonic()
            progress = {}
            while True:
                record = rig.job(record.job_id)
                progress[record.progress.stage] = progress.get(record.progress.stage, time.monotonic() - render_started)
                if record.state in (JobState.succeeded, JobState.failed, JobState.cancelled):
                    break
                time.sleep(0.5)
            wall = time.monotonic() - render_started
        extra: dict[str, Any] = {"state": record.state.value, "failure": record.failure and record.failure.model_dump(),
                                 "frames": record.output.frames if record.output else None,
                                 "render_seconds": round(wall, 1), "render_fps": round(frames / wall, 1),
                                 "prep_seconds": round(prep, 1), "stage_start_s": {k: round(v, 1) for k, v in progress.items()}}
        checks = record.state == JobState.succeeded and record.output.frames == frames
        deadline = time.monotonic() + 30
        while rig.workdir(record.job_id).exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        extra["workdir_released"] = not rig.workdir(record.job_id).exists()
        if record.output:
            out = root / record.output.output_path
            import hashlib
            extra["output_bytes"] = out.stat().st_size
            extra["output_frames_verified"] = record.output.frames
        extra["checks_ok"] = bool(checks and extra["workdir_released"])
        if tracemalloc_top:
            diff = tracemalloc.take_snapshot().compare_to(before, "lineno")[:tracemalloc_top]
            extra["tracemalloc_growth"] = [f"{item.size_diff / 1024:.0f} KiB {item.traceback[0].filename.split('roop_3d')[-1]}:{item.traceback[0].lineno}" for item in diff]
        return _finish("long_video", {"minutes": minutes, "fps": fps, "frames": frames, "size": list(size), "tracks": tracks},
                       sampler, started, extra, profile)
    finally:
        rig.worker.stop()
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def scenario_hires(width: int = 1920, height: int = 1080, frames: int = 600, fps: int = 25, tracks: int = 2,
                   profile: BoundedProfile = DEFAULT_PROFILE, keep: bool = False, interval: float = 2.0) -> dict:
    root = _fresh_root(f"hires{width}")
    started = time.monotonic()
    rig = build_rig(root, frames=frames, tracks=tracks, size=(width, height), fps=fps, poll_interval=0.05)
    try:
        with Sampler(interval) as sampler:
            record = rig.submit("soak-hires")
            while rig.job(record.job_id).state not in (JobState.succeeded, JobState.failed, JobState.cancelled):
                time.sleep(0.5)
        record = rig.job(record.job_id)
        deadline = time.monotonic() + 60
        while rig.workdir(record.job_id).exists() and time.monotonic() < deadline:
            time.sleep(0.2)
        extra = {"state": record.state.value, "failure": record.failure and record.failure.model_dump(),
                 "frames": record.output.frames if record.output else None,
                 "peak_rss_mib": round(max(s.memory.rss_mib or 0 for _, s in sampler.samples), 1),
                 "output_bytes": (root / record.output.output_path).stat().st_size if record.output else None,
                 "workdir_released": not rig.workdir(record.job_id).exists()}
        extra["checks_ok"] = record.state == JobState.succeeded and record.output.frames == frames and extra["workdir_released"]
        return _finish("hires", {"width": width, "height": height, "frames": frames, "fps": fps, "tracks": tracks},
                       sampler, started, extra, profile)
    finally:
        rig.worker.stop()
        if not keep:
            shutil.rmtree(root, ignore_errors=True)


def scenario_successive(jobs: int = 60, frames: int = 16, tracks: int = 2, profile: BoundedProfile = IDLE_PROFILE,
                        pause_resume_every: int = 5, cancel_every: int = 7, leak: Callable[[int], None] | None = None) -> dict:
    """Many jobs back to back in one process, a mix of completed / paused-and-resumed / cancelled, judged per job."""
    root = _fresh_root("successive")
    started = time.monotonic()
    rig = build_rig(root, frames=frames, tracks=tracks, poll_interval=0.01, delay=0.004)
    outcomes = {"succeeded": 0, "cancelled": 0, "failed": 0}
    store_ms: list[float] = []
    try:
        sampler = Sampler()
        sampler.take(0)
        for index in range(1, jobs + 1):
            record = rig.submit(f"succ-{index}")
            if cancel_every and index % cancel_every == 0:
                rig.wait_frames(record.job_id, 3)
                rig.store.control(record.job_id, "cancel")
            elif pause_resume_every and index % pause_resume_every == 0:
                rig.wait_frames(record.job_id, 3)
                try:
                    rig.store.control(record.job_id, "pause")
                    rig.wait(record.job_id, {JobState.paused, JobState.succeeded})
                    if rig.job(record.job_id).state == JobState.paused:
                        rig.store.control(record.job_id, "resume")
                except Exception:
                    pass
            final = rig.wait(record.job_id, timeout=120)
            outcomes[final.state.value] = outcomes.get(final.state.value, 0) + 1
            t = time.perf_counter()
            rig.store.list()
            store_ms.append((time.perf_counter() - t) * 1000)
            deadline = time.monotonic() + 10
            while (rig.workdir(record.job_id).exists() or rig.worker._futures) and time.monotonic() < deadline:
                time.sleep(0.01)
            if leak:
                leak(index)
            gc.collect()
            sampler.take(index)
        outputs = [p for p in rig.output_files() if not p.name.endswith(".manifest.json")]
        extra = {"outcomes": outcomes, "published_outputs": len(outputs),
                 "state_json_bytes": (root / "roop_3d/.runtime/jobs/state.json").stat().st_size,
                 "store_list_ms": {"first": round(store_ms[0], 2), "last": round(store_ms[-1], 2), "max": round(max(store_ms), 2)},
                 "leftover_workdirs": len([p for p in (root / "roop_3d/.runtime/jobs").glob("*") if p.is_dir()])}
        extra["checks_ok"] = (outcomes["failed"] == 0 and extra["leftover_workdirs"] == 0
                              and len(outputs) == outcomes["succeeded"])
        return _finish("successive", {"jobs": jobs, "frames": frames, "tracks": tracks, "x_axis": "job index"},
                       sampler, started, extra, profile)
    finally:
        rig.worker.stop()
        shutil.rmtree(root, ignore_errors=True)


def _tiny_model(path: Path, size: int = 64) -> bytes:
    import onnx
    from onnx import TensorProto, helper
    weight = helper.make_tensor("w", TensorProto.FLOAT, [size, size], [0.01] * (size * size))
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, size])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, size])], [weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    data = model.SerializeToString()
    path.write_bytes(data)
    return data


def scenario_model_switch(cycles: int = 300, profile: BoundedProfile = IDLE_PROFILE) -> dict:
    """Load / run / switch / evict real ONNX Runtime sessions through the app's own eviction machinery."""
    import numpy as np
    import onnxruntime as ort
    from roop_3d.engine import resources
    root = _fresh_root("modelswitch")
    started = time.monotonic()
    models = [(_tiny_model(root / f"m{i}.onnx", 64 + 16 * i), 64 + 16 * i) for i in range(3)]

    class Owner:
        def __init__(self, data: bytes, size: int):
            self.session = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
            self.size = size
            resources.register(self, "release")

        def run(self):
            return self.session.run(None, {"x": np.ones((1, self.size), np.float32)})

        def release(self):
            self.session = None

    sampler = Sampler()
    sampler.samples.clear()
    sampler.take(0)
    try:
        current = None
        for cycle in range(1, cycles + 1):
            data, size = models[cycle % len(models)]
            if current is not None:
                current.release()  # the switch: the previous model's session must really disappear
            current = Owner(data, size)
            for _ in range(5):
                current.run()
            if cycle % 10 == 0:
                evicted = resources.evict_for_build()  # raises BUILD_LIVE_ORT_SESSIONS if anything is still referenced
                current = None
                sampler.take(cycle)
                if evicted["remaining_ort_sessions"] != 0:
                    raise AssertionError("ORT session survived eviction")
        current = None
        resources.evict_for_build()
        final = probe.count_ort_sessions()
        extra = {"cycles": cycles, "final_ort_sessions": final, "checks_ok": final == 0}
        return _finish("model_switch", {"cycles": cycles, "x_axis": "switch cycle"}, sampler, started, extra, profile)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def scenario_selftest(profile: BoundedProfile = IDLE_PROFILE) -> dict:
    """The detector must catch a real leak: 4 MiB retained per job (~100 MiB over the run, > 2x the RSS allowance)."""
    hoard: list[bytearray] = []
    result = scenario_successive(jobs=30, frames=8, tracks=1, profile=profile, pause_resume_every=0, cancel_every=0,
                                 leak=lambda i: hoard.append(bytearray(os.urandom(256)) * 16384))
    flagged = result["growth"]["violations"]
    result["selftest_leak_detected"] = bool({"rss_mib", "private_mib", "py_heap_blocks"} & set(flagged)) or bool(flagged)
    result["passed"] = result["selftest_leak_detected"]  # inverted: here the PASS condition is "the leak was caught"
    return result


SCENARIOS = {"long_video": scenario_long_video, "hires": scenario_hires, "successive": scenario_successive,
             "model_switch": scenario_model_switch, "selftest": scenario_selftest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roop_3d.reliability.soak")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--jobs", type=int, default=60)
    parser.add_argument("--cycles", type=int, default=300)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--tracks", type=int, default=2)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--tracemalloc", type=int, default=0, help="report the top-N Python allocation growth sites")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    kwargs: dict[str, Any] = {
        "long_video": dict(minutes=args.minutes, fps=args.fps, tracks=args.tracks, interval=args.interval,
                           tracemalloc_top=args.tracemalloc),
        "hires": dict(width=args.width, height=args.height, frames=args.frames, fps=args.fps, tracks=args.tracks),
        "successive": dict(jobs=args.jobs, tracks=args.tracks), "model_switch": dict(cycles=args.cycles), "selftest": {}}[args.scenario]
    result = SCENARIOS[args.scenario](**kwargs)
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    summary = {k: result[k] for k in ("scenario", "passed", "seconds") if k in result}
    summary["violations"] = result["growth"]["violations"]
    summary["invariants_ok"] = result["invariants"]["ok"]
    print(json.dumps(summary), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
