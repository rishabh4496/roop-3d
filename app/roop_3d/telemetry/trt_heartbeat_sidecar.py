"""Out-of-process TensorRT build heartbeat: the only way to keep a terminal line alive during a cold engine build.

ONNX Runtime holds Python's GIL for the whole of ``InferenceSession(...)`` with the TensorRT provider (measured: a 7 s build
let a 20 Hz Python thread run twice). While it does, no thread of the server process can run, including the one that would
print the heartbeat and the event loop that serves the WebSocket. This module therefore runs as a separate process with its
own GIL:

    python -m roop_3d.telemetry.trt_heartbeat_sidecar --model M --precision P --shape S --parent PID --started T

It prints ``[TRT-BUILDER] ⚙ Compiling ... Elapsed: 14.0s | VRAM: 3.2 GB`` once a second to the terminal it inherited, in place
on a TTY, and measures VRAM itself (the parent's own figure where the driver reports one, else the device's). It prints nothing
for the first ``--grace`` seconds, so a build that hits the engine cache and finishes at once leaves no trace. It exits when its
stdin closes (the parent finished, or died) and erases its line on the way out.

Imports are stdlib-only on purpose: starting it must be cheap.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time

from roop_3d.telemetry.trt_heartbeat import heartbeat_line


def _vram(parent_pid: int) -> tuple[float | None, str | None]:
    from roop_3d.reliability.probe import gpu_memory
    reading = gpu_memory(max_age=0.5, pid=parent_pid)
    if not reading.available:
        return None, None
    if reading.per_process and reading.process_mib is not None:
        return reading.process_mib / 1024, "process"
    return (reading.used_mib / 1024 if reading.used_mib is not None else None), "device"


def run(model: str, precision: str | None, shape: str | None, parent_pid: int, started: float, grace: float, interval: float,
        stdin=None, out=None, clock=time.time) -> None:
    stdin = sys.stdin if stdin is None else stdin
    out = sys.stdout if out is None else out
    stop = threading.Event()

    def watch_stdin() -> None:
        try:
            while stdin.read(1):
                pass
        except (OSError, ValueError):
            pass
        stop.set()
    threading.Thread(target=watch_stdin, daemon=True).start()

    in_place = out.isatty()
    drawn = False
    while not stop.is_set():
        elapsed = clock() - started
        if elapsed >= grace:
            try:
                vram_gb, scope = _vram(parent_pid)
            except Exception:
                vram_gb, scope = None, None
            line = heartbeat_line(model, precision, shape, elapsed, vram_gb, scope)
            out.write(("\r\x1b[K" + line) if in_place else line + "\n")
            out.flush()
            drawn = True
        stop.wait(interval)
    if drawn and in_place:
        out.write("\r\x1b[K")
        out.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="roop_3d.telemetry.trt_heartbeat_sidecar")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--shape", default=None)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--started", type=float, default=None, help="wall-clock time the build began (default: now)")
    parser.add_argument("--grace", type=float, default=1.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, OSError):
        pass
    run(args.model, args.precision, args.shape, args.parent, args.started if args.started is not None else time.time(),
        args.grace, args.interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
