"""Heartbeat while TensorRT builds an engine, so neither the terminal nor the React UI looks frozen.

**What the build does to the process (measured, not assumed).** ``ort.InferenceSession(...)`` with the TensorRT provider holds
Python's GIL for the whole build: in a 7 s cold build a 20 Hz Python thread woke twice (largest gap 6.99 s). Nothing in the
server process can run in that window, neither a heartbeat thread nor the asyncio loop that serves ``/ws/v1/pipeline/status``.
A heartbeat that only ticks inside the process would therefore be silent for exactly the 15-60 s it exists for. So:

* **Terminal** - a sidecar *process* (``trt_heartbeat_sidecar``) prints the 1 Hz ``[TRT-BUILDER]`` line. It has its own GIL.
  If it cannot be started, an in-process thread prints instead (correct for any build that does release the GIL).
* **WebSocket / UI** - a ``trt_compilation_progress`` *start* frame is published before the native call, followed by a short
  ``settle`` sleep (which releases the GIL) so the event loop can actually deliver it. The UI then counts the elapsed time itself
  from that frame. Frames with a server-measured ``elapsed_seconds`` and ``vram_gb`` follow once a second whenever the process
  is able to run (a build that releases the GIL); a ``finished`` frame closes the window. During a GIL-holding build the server
  cannot send VRAM or anything else, and the UI says so rather than showing a stale number.

No percentage is reported anywhere: TensorRT exposes no build progress.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Protocol

log = logging.getLogger("roop3d.trt_heartbeat")

PRECISION_LABEL = {"FP16": "FP16+FP32 Mix", "FP32": "FP32", "FP8": "FP8", "INT8": "INT8", "BF16": "BF16"}


def precision_label(precision: str | None) -> str:
    """``FP16`` builds always keep an FP32 fallback for layer-norm and unsafe nodes (``trt_layer_norm_fp32_fallback``),
    so they are genuinely mixed; every other value is shown as given."""
    if not precision:
        return "precision n/a"
    return PRECISION_LABEL.get(precision.upper(), precision)


def format_shape(shape: Any) -> str | None:
    """``(4, 3, 512, 512)`` -> ``"4x3x512x512"``; accepts a name->shape mapping (first input) or an already formatted text."""
    if shape is None:
        return None
    if isinstance(shape, str):
        return shape or None
    if isinstance(shape, dict):
        shape = next((value for value in shape.values() if value), None)
        if shape is None:
            return None
    try:
        return "x".join(str(int(part)) for part in shape)
    except (TypeError, ValueError):
        return None


def heartbeat_line(model: str, precision: str | None, opt_shape: str | None, elapsed: float, vram_gb: float | None,
                   vram_scope: str | None) -> str:
    vram = "n/a" if vram_gb is None else f"{vram_gb:.1f} GB" + (" (dev)" if vram_scope == "device" else "")
    return (f"[TRT-BUILDER] ⚙ Compiling {model} ({precision_label(precision)} | opt={opt_shape or 'n/a'}) ... "
            f"Elapsed: {elapsed:.1f}s | VRAM: {vram}")


def _default_vram() -> tuple[float | None, str | None]:
    from roop_3d.reliability.probe import gpu_memory
    reading = gpu_memory(max_age=0.5)
    if not reading.available:
        return None, None
    if reading.per_process and reading.process_mib is not None:
        return reading.process_mib / 1024, "process"
    return (reading.used_mib / 1024 if reading.used_mib is not None else None), "device"


class TerminalHandle(Protocol):
    def stop(self) -> None: ...


class TrtCompilationHeartbeat:
    """``begin`` when a build starts; ``end`` when it finishes, fails or moves to the next phase. Idempotent.

    ``spawn_terminal(info)`` starts the out-of-process terminal printer and returns a handle (or ``None``); without one the
    in-process thread prints through ``beat``. ``settle`` is how long ``begin`` sleeps after publishing the start frame."""

    def __init__(self, *, beat: Callable[[str], None], done: Callable[[str], None], publish: Callable[[dict], None],
                 interval: float = 1.0, clock: Callable[[], float] = time.monotonic,
                 vram: Callable[[], tuple[float | None, str | None]] = _default_vram,
                 spawn_terminal: Callable[[dict], TerminalHandle | None] | None = None, settle: float = 0.15):
        self._beat, self._done, self._publish = beat, done, publish
        self.interval, self.clock, self._vram = interval, clock, vram
        self._spawn_terminal, self.settle = spawn_terminal, settle
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._terminal: TerminalHandle | None = None
        self._info: dict[str, Any] | None = None
        self._last: dict[str, Any] | None = None

    @property
    def active(self) -> bool:
        with self._lock:
            return self._info is not None

    def snapshot(self) -> dict[str, Any] | None:
        """The current frame while a build runs (for a client that connects mid-build), with ``elapsed_seconds`` brought up to date."""
        with self._lock:
            if self._info is None or self._last is None:
                return None
            return {**self._last, "elapsed_seconds": round(self.clock() - self._info["started"], 1)}

    def begin(self, model: str, precision: str | None = None, opt_shape: Any = None) -> None:
        info = {"model": model, "precision": precision, "opt_shape": format_shape(opt_shape)}
        with self._lock:
            if self._info is not None and self._info["model"] == model:
                # Same build, reported again (e.g. the shape became known): keep the clock, refresh what is shown.
                self._info.update({k: v for k, v in info.items() if v is not None})
                return
        self.end("next model")
        info.update(started=self.clock(), wall=time.time())
        with self._lock:
            self._info = info
            self._stop = stop = threading.Event()
        frame = self._frame(info, 0.0, "compiling", vram=(None, None))   # no probe before the native call: it must start at once
        with self._lock:
            self._last = frame
        self._safe(self._publish, frame)
        if self.settle > 0:
            time.sleep(self.settle)   # releases the GIL, so the event loop can send the start frame before the build takes it
        terminal = None
        if self._spawn_terminal is not None:
            try:
                terminal = self._spawn_terminal(info)
            except Exception:
                log.exception("TensorRT terminal heartbeat could not start; falling back to the in-process line")
        if terminal is None:   # in-process printing: show the build at once instead of one interval later
            self._safe(self._beat, heartbeat_line(model, precision, info["opt_shape"], 0.0, *self._safe_vram()))
        with self._lock:
            self._terminal = terminal
            thread = self._thread = threading.Thread(target=self._run, args=(stop, terminal is None), name="roop3d-trt-heartbeat", daemon=True)
        thread.start()

    def end(self, reason: str = "finished") -> None:
        with self._lock:
            info, self._info = self._info, None
            thread, self._thread = self._thread, None
            terminal, self._terminal = self._terminal, None
            self._last = None
            self._stop.set()
        if info is None:
            return
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        if terminal is not None:
            self._safe(terminal.stop)   # the sidecar erases its own line first, so the closing line below starts clean
        elapsed = self.clock() - info["started"]
        self._safe(self._done, f"[TRT-BUILDER] ■ {info['model']} build window closed after {elapsed:.1f}s ({reason}); "
                                "the job or log reports whether the engine was written")
        frame = self._frame(info, elapsed, "finished")
        frame["reason"] = reason
        self._safe(self._publish, frame)

    # -- internals ------------------------------------------------------------------------------------------
    def _frame(self, info: dict[str, Any], elapsed: float, status: str, *, vram: tuple[float | None, str | None] | None = None) -> dict[str, Any]:
        vram_gb, scope = vram if vram is not None else self._safe_vram()
        return {"type": "trt_compilation_progress", "status": status, "model": info["model"], "precision": info["precision"],
                "precision_label": precision_label(info["precision"]), "opt_shape": info["opt_shape"],
                "elapsed_seconds": round(elapsed, 1), "vram_gb": None if vram_gb is None else round(vram_gb, 2), "vram_scope": scope,
                "started_at": info["wall"]}

    def _safe_vram(self) -> tuple[float | None, str | None]:
        try:
            return self._vram()
        except Exception:
            return None, None

    @staticmethod
    def _safe(call: Callable[..., object], *args: object) -> None:
        try:
            call(*args)
        except Exception:
            log.exception("TensorRT heartbeat sink failed")  # telemetry must never affect the build

    def _tick(self, stop: threading.Event, *, print_line: bool) -> None:
        with self._lock:
            info = dict(self._info) if self._info is not None else None
        if info is None or stop.is_set():
            return
        elapsed = self.clock() - info["started"]
        frame = self._frame(info, elapsed, "compiling")
        if print_line:
            self._safe(self._beat, heartbeat_line(info["model"], info["precision"], info["opt_shape"], elapsed, frame["vram_gb"], frame["vram_scope"]))
        with self._lock:
            if self._info is None or stop.is_set():
                return
            self._last = frame
        self._safe(self._publish, frame)

    def _run(self, stop: threading.Event, print_line: bool) -> None:
        # Reached only when the process can run. During a GIL-holding build this thread is simply not scheduled until the
        # build ends; the sidecar covers the terminal and the UI counts its own clock from the start frame.
        while not stop.wait(self.interval):
            self._tick(stop, print_line=print_line)
