"""Pinokio terminal HUD: UTF-8/unbuffered enforcement, one live in-place line, milestone logs and the boot/job report.

Everything here is presentation of values measured elsewhere. A value that was not measured prints ``--``; nothing is
estimated to fill a gap. The HUD is *unarmed* until ``arm()`` is called by the real server entry point, so importing it
(tests, tools, the job worker under a fake runner) never writes to a terminal.

The Pinokio launcher (``start.js``) watches this terminal's output for ``http://<ip>:<port>`` and for a few failure
tokens. ``defuse()`` therefore rewrites those tokens in anything the HUD prints, and the boot report is only printed
after the server's own ``ready`` line, so the first URL Pinokio sees is always the real loopback address.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Callable, Mapping, TextIO

# ---------------------------------------------------------------------------------------------- environment lock
TERMINAL_ENVIRONMENT: dict[str, str] = {"PYTHONUNBUFFERED": "1", "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def enforce_terminal_environment(environ: dict[str, str] | None = None, *, streams: bool = True) -> dict[str, str]:
    """Force unbuffered, UTF-8 Python I/O for this process and every child it starts (children inherit ``os.environ``).

    Returns the variables that had to change. ``streams=True`` also reconfigures this process's stdout/stderr (setting
    ``PYTHONUTF8`` after start-up cannot change the running interpreter) and turns on ANSI processing for a legacy
    Windows console. Idempotent; never raises."""
    target = os.environ if environ is None else environ
    changed = {key: value for key, value in TERMINAL_ENVIRONMENT.items() if target.get(key) != value}
    target.update(changed)
    if streams:
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True, write_through=True)  # type: ignore[union-attr]
            except (AttributeError, ValueError, OSError):
                pass  # a captured/closed stream (pytest, a service wrapper) keeps what it has
        _enable_windows_vt()
    return changed


def child_environment(base: Mapping[str, str] | None = None) -> dict[str, str]:
    """``base`` (default ``os.environ``) with the terminal contract applied, for an explicit ``subprocess`` ``env=``."""
    return {**(os.environ if base is None else base), **TERMINAL_ENVIRONMENT}


def _enable_windows_vt() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if kernel.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:  # not a console (pty, pipe, service): ANSI is then the host's business
        pass


# ------------------------------------------------------------------------------------------------- ANSI helpers
ESC = "\x1b"
CLEAR_LINE = "\r\x1b[K"
RESET = f"{ESC}[0m"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_STYLES = {"dim": "2", "bold": "1", "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36"}

# Tokens `start.js` treats as a failed start, and the numeric-host URL it captures. A diagnostic must never trip them.
_TRIPWIRES = (("error:", "error ="), ("PORT_IN_USE", "PORT IN USE"), ("INVALID_SETTINGS", "INVALID SETTINGS"),
              ("ROOP3D_GUARD_ERROR", "ROOP3D GUARD ERR"), ("ModuleNotFoundError", "Module-NotFound"))


def defuse(text: str) -> str:
    for token, replacement in _TRIPWIRES:
        text = text.replace(token, replacement)
    return text


def paint(text: str, *styles: str, enabled: bool = True) -> str:
    codes = ";".join(_STYLES[name] for name in styles if name in _STYLES)
    return f"{ESC}[{codes}m{text}{RESET}" if enabled and codes else text


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


def truncate_visible(text: str, width: int) -> str:
    """Cut to ``width`` visible columns without splitting an ANSI sequence; closes styling if it cut anything."""
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    out, shown, index = [], 0, 0
    while index < len(text) and shown < width:
        match = _ANSI.match(text, index)
        if match:
            out.append(match.group(0))
            index = match.end()
            continue
        out.append(text[index])
        shown += 1
        index += 1
    return "".join(out) + RESET


def fmt_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds + 0.5)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def fmt_gb(mib: float | None) -> str:
    return "--" if mib is None else f"{mib / 1024:.1f}"


def fmt_ms(value: float | None) -> str:
    return "--" if value is None else f"{value:.1f}"


# ------------------------------------------------------------------------------------------ resource measurement
@dataclass(frozen=True)
class ResourceReading:
    cpu_pct: float | None = None
    vram_used_mib: float | None = None
    vram_total_mib: float | None = None
    #: "process" when the driver reports this process's own memory, "device" for the whole GPU (WDDM reports only that).
    vram_scope: str | None = None


class _CpuMeter:
    """System-wide CPU utilisation between two reads. ``None`` where the OS offers no dependency-free counter."""

    def __init__(self, source: Callable[[], tuple[int, int] | None] | None = None):
        self._source = source or self._os_counters
        self._last: tuple[int, int] | None = None

    @staticmethod
    def _os_counters() -> tuple[int, int] | None:
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes
                idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
                if not ctypes.windll.kernel32.GetSystemTimes(*(ctypes.byref(v) for v in (idle, kernel, user))):  # type: ignore[attr-defined]
                    return None
                value = lambda t: (t.dwHighDateTime << 32) | t.dwLowDateTime  # noqa: E731
                return value(idle), value(kernel) + value(user)  # kernel time already includes idle time
            except Exception:
                return None
        try:
            with open("/proc/stat", "r", encoding="ascii") as stream:
                fields = [int(part) for part in stream.readline().split()[1:]]
            return fields[3] + (fields[4] if len(fields) > 4 else 0), sum(fields[:8])
        except (OSError, ValueError, IndexError):
            return None

    def read(self) -> float | None:
        now = self._source()
        previous, self._last = self._last, now
        if now is None or previous is None:
            return None
        idle, total = now[0] - previous[0], now[1] - previous[1]
        if total <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))


class ResourceSampler:
    """Samples CPU and VRAM on its own thread so the render loop never waits for ``nvidia-smi``.

    ``latest()`` is a lock-protected read of the last reading; ``peak_vram_mib`` is the maximum seen since ``reset_peak``."""

    def __init__(self, *, interval: float = 1.0, gpu: Callable[[], Any] | None = None, cpu: _CpuMeter | None = None):
        self.interval = interval
        self._gpu = gpu or self._default_gpu
        self._cpu = cpu or _CpuMeter()
        self._lock = threading.Lock()
        self._reading = ResourceReading()
        self._peak: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._users = 0

    @staticmethod
    def _default_gpu() -> Any:
        from roop_3d.reliability.probe import gpu_memory
        return gpu_memory(max_age=0.5)

    def sample_once(self) -> ResourceReading:
        used = total = scope = None
        try:
            gpu = self._gpu()
            if getattr(gpu, "available", False):
                per_process = bool(getattr(gpu, "per_process", False)) and gpu.process_mib is not None
                used, scope = (gpu.process_mib, "process") if per_process else (gpu.used_mib, "device")
                total = gpu.total_mib
        except Exception:
            pass  # a failed probe is an unmeasured value, not a failed job
        reading = ResourceReading(self._cpu.read(), used, total, scope)
        with self._lock:
            self._reading = reading
            if used is not None:
                self._peak = used if self._peak is None else max(self._peak, used)
        return reading

    def latest(self) -> ResourceReading:
        with self._lock:
            return self._reading

    @property
    def peak_vram_mib(self) -> float | None:
        with self._lock:
            return self._peak

    def reset_peak(self) -> None:
        with self._lock:
            self._peak = self._reading.vram_used_mib

    def acquire(self) -> None:
        """Start the sampling thread for the first user; the last ``release`` stops it."""
        with self._lock:
            self._users += 1
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="roop3d-hud-sampler", daemon=True)
            self._thread.start()

    def release(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users:
                return
            self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self.interval)


# ------------------------------------------------------------------------------------------------- frame HUD model
#: HUD column -> the measured pipeline stage that feeds it (``core.stage_timing.STAGES``). ``None`` means the pipeline
#: does not time that step on its own: the compositor reports Inpaint+Cage and LapBlend together as ``masking``, which
#: the bar shows as ``Comp`` instead of splitting a measurement it does not have.
STAGE_COLUMNS: tuple[tuple[str, str | None], ...] = (
    ("Det", "detection"), ("3D-LBS", "pose_tps"), ("Inpaint+Cage", None), ("3DSwap+SH", "inswapper"),
    ("Enhancer", "restorer"), ("LapBlend", None))
SWAP_MODE_LABEL = {"full_craniofacial": "FULL_CRANIOFACIAL_LOCK", "adaptive_skull": "ADAPTIVE_SKULL", "inner_face": "INNER_FACE"}


def swap_mode_label(value: Any) -> str:
    return SWAP_MODE_LABEL.get(str(value), str(value).upper() if value else "--")


#: What each ``audio_policy`` does to the soundtrack, in the words the remux milestone and the report use.
AUDIO_POLICY_TEXT = {"copy": "copy - lossless stream copy, no re-encode", "transcode": "transcode - re-encoded",
                     "drop": "drop - output has no audio"}


@dataclass
class JobHud:
    job_id: str
    swap_mode: str
    total: int | None
    started: float
    budget_mib: float | None = None
    samples: deque = None  # type: ignore[assignment]  # (monotonic time, frames done)
    last: tuple[float, int] | None = None
    current_fps: float | None = None
    last_frame: int = 0
    last_timings: dict | None = None
    muxing: bool = False
    stage: str = "render"          # the phase whose frames the live line is counting

    def __post_init__(self) -> None:
        self.samples = deque(maxlen=512)


def rolling_fps(samples: deque, window: float) -> float | None:
    if len(samples) < 2:
        return None
    newest_t, newest_f = samples[-1]
    oldest_t, oldest_f = next(((t, f) for t, f in samples if newest_t - t <= window), samples[0])
    span = newest_t - oldest_t
    return (newest_f - oldest_f) / span if span > 0 and newest_f >= oldest_f else None


def bar(percent: float | None, width: int = 14, *, color: bool = True) -> str:
    if percent is None:
        return "░" * width
    filled = max(0, min(width, int(round(width * percent / 100.0))))
    return paint("█" * filled, "green", enabled=color) + paint("░" * (width - filled), "dim", enabled=color)


#: Live-line label for the phases that count frames but are not the render itself.
PHASE_LABEL = {"angle_scan": "SCAN"}


def render_frame_line(job: JobHud, *, frame: int, rolling: float | None, reading: ResourceReading, peak_mib: float | None,
                      timings: Mapping[str, float] | None, width: int, color: bool = True) -> str:
    """One status line, composed of segments dropped from the right until it fits ``width`` columns.

    The first three segments (job and mode, progress, rate) are never dropped."""
    timings = timings or {}
    pct = (100.0 * frame / job.total) if job.total else None
    remaining = (job.total - frame) if job.total else None
    eta = (remaining / rolling) if (remaining is not None and rolling and rolling > 0) else None
    vram_tag = "VRAM" if reading.vram_scope == "process" else "VRAM(dev)" if reading.vram_scope == "device" else "VRAM"
    stage_cells = [f"{label} {fmt_ms(timings.get(key)) if key else '--'}" for label, key in STAGE_COLUMNS]
    segments = [
        f"{paint('▶', 'cyan', enabled=color)} {paint(job.job_id[:8], 'bold', enabled=color)} {paint(job.swap_mode, 'magenta', enabled=color)}"
        + (f" {paint(PHASE_LABEL[job.stage], 'yellow', enabled=color)}" if job.stage in PHASE_LABEL else ""),
        f"{bar(pct, color=color)} {'--' if pct is None else f'{pct:5.1f}'}% {frame}/{job.total if job.total else '?'}",
        f"{fmt_ms(job.current_fps)}/{fmt_ms(rolling)} fps",
        f"ETA {fmt_duration(eta)}",
        f"{vram_tag} {fmt_gb(reading.vram_used_mib)}/{fmt_gb(peak_mib)}/{fmt_gb(job.budget_mib)}GB",
        f"CPU {'--' if reading.cpu_pct is None else f'{reading.cpu_pct:.0f}'}%",
        " ".join(stage_cells) + f" Comp {fmt_ms(timings.get('masking'))} ms",
    ]
    separator = paint(" │ ", "dim", enabled=color)
    keep = len(segments)
    while keep > 3 and visible_len(separator.join(segments[:keep])) > width:
        keep -= 1
    return truncate_visible(separator.join(segments[:keep]), width)


# ----------------------------------------------------------------------------------------------------------- HUD
class PinokioTerminalHUD:
    """Owns the terminal's *live line*: a frame bar or a TensorRT heartbeat that is rewritten in place, with permanent
    lines (milestones, reports, log records) scrolling above it. Thread-safe; every write is one locked operation."""

    def __init__(self, stream: TextIO | None = None, *, color: bool | None = None, in_place: bool | None = None,
                 clock: Callable[[], float] = time.monotonic, sampler: ResourceSampler | None = None,
                 min_redraw: float = 0.1, plain_interval: float = 5.0, rolling_window: float = 5.0,
                 width: Callable[[], int] | None = None):
        self._stream = stream
        self._color_flag, self._in_place_flag = color, in_place
        self.clock = clock
        self.sampler = sampler or ResourceSampler()
        self.min_redraw, self.plain_interval, self.rolling_window = min_redraw, plain_interval, rolling_window
        self._width = width
        self._lock = threading.RLock()
        self.armed = False
        self.state: Any = None
        self._jobs: dict[str, JobHud] = {}
        self._live: dict[str, str] = {}          # slot -> rendered line; "trt" outranks "bar"
        self._shown: str | None = None
        self._last_draw = 0.0
        self._last_plain = 0.0
        self.recent_milestones: deque = deque(maxlen=200)

    # -- configuration --------------------------------------------------------------------------------------
    @property
    def stream(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stdout

    def _is_tty(self) -> bool:
        try:
            return bool(self.stream.isatty())
        except (AttributeError, ValueError, OSError):
            return False

    @property
    def color(self) -> bool:
        if self._color_flag is not None:
            return self._color_flag
        return (not os.environ.get("NO_COLOR")) and (self._is_tty() or bool(os.environ.get("FORCE_COLOR")))

    @property
    def in_place(self) -> bool:
        return self._in_place_flag if self._in_place_flag is not None else self._is_tty()

    def columns(self) -> int:
        if self._width is not None:
            return self._width()
        return max(40, shutil.get_terminal_size((120, 24)).columns - 1)

    def arm(self, state: Any = None, *, stream: TextIO | None = None) -> "PinokioTerminalHUD":
        """Enable output. ``ROOP3D_HUD=0`` keeps the HUD silent (CI, log capture) without changing any behaviour."""
        with self._lock:
            if os.environ.get("ROOP3D_HUD", "").strip().lower() in {"0", "off", "false", "no"}:
                return self
            self.state = state if state is not None else self.state
            if stream is not None:
                self._stream = stream
            self.armed = True
        return self

    def disarm(self) -> None:
        with self._lock:
            self._clear_live()
            self.armed = False
            self._live.clear()
            self._jobs.clear()

    # -- low level output -----------------------------------------------------------------------------------
    def _raw(self, text: str) -> None:
        try:
            self.stream.write(text)
            self.stream.flush()
        except UnicodeEncodeError:
            try:
                encoding = getattr(self.stream, "encoding", None) or "ascii"
                self.stream.write(text.encode(encoding, "replace").decode(encoding, "replace"))
                self.stream.flush()
            except Exception:
                pass
        except (OSError, ValueError):
            pass  # a closed or broken terminal must never fail a render

    def _clear_live(self) -> None:
        if self._shown is not None and self.in_place:
            self._raw(CLEAR_LINE)
        self._shown = None

    def _current_live(self) -> str | None:
        return self._live.get("trt") or self._live.get("bar")

    def _draw_live(self, *, force: bool = False) -> None:
        line = self._current_live()
        if line is None:
            return
        now = self.clock()
        if self.in_place:
            if not force and now - self._last_draw < self.min_redraw:
                return
            self._raw(CLEAR_LINE + truncate_visible(line, self.columns()))
            self._shown, self._last_draw = line, now
        elif force or now - self._last_plain >= self.plain_interval:
            self._raw(_ANSI.sub("", line) + "\n")  # no cursor control available: a plain line, at a low rate
            self._last_plain = now

    def line(self, text: str) -> None:
        """Print one permanent line above the live line."""
        if not self.armed:
            return
        with self._lock:
            self._clear_live()
            self._raw(defuse(text) + "\n")
            if self.in_place:
                self._draw_live(force=True)

    def block(self, text: str) -> None:
        """Print a multi-line block atomically (a report), so concurrent frames cannot interleave with it."""
        self.line(text.rstrip("\n"))

    # -- logging coexistence --------------------------------------------------------------------------------
    def suspend(self) -> None:
        self._lock.acquire()
        self._clear_live()

    def resume(self) -> None:
        try:
            if self.in_place:
                self._draw_live(force=True)
        finally:
            self._lock.release()

    # -- milestones -----------------------------------------------------------------------------------------
    def milestone(self, tag: str, event: str, **fields: Any) -> str | None:
        """``[TAG] event key=value ...`` — one structured, greppable line. Returns the line (also kept in memory)."""
        if not self.armed:
            return None
        try:
            parts = " ".join(f"{key}={value}" for key, value in fields.items() if value is not None)
            text = f"{paint(f'[{tag}]', 'cyan', enabled=self.color)} {paint(event, 'bold', enabled=self.color)} {parts}".rstrip()
            self.recent_milestones.append({"at": time.time(), "tag": tag, "event": event, **{k: v for k, v in fields.items() if v is not None}})
            self.line(text)
            return text
        except Exception:  # called from the render path: a bad field must never fail a job
            return None

    # -- jobs -----------------------------------------------------------------------------------------------
    def _budget_mib(self) -> float | None:
        """The effective VRAM budget the hardware probe computed for this machine and settings; ``None`` without a GPU."""
        try:
            from roop_3d.core.hardware import probe_hardware
            from roop_3d.core.settings import Settings
            settings = getattr(self.state, "settings", None) or Settings()
            gpus = probe_hardware(settings).gpus
            return float(gpus[0].effective_vram_budget_mib) if gpus else None
        except Exception:
            return None

    def print_report(self, trigger: str, *, job: dict | None = None, wait: bool = False) -> None:
        """Print the 14-section runtime report. Built on a worker thread by default so a job or the server never waits for it."""
        if not self.armed:
            return

        def work() -> None:
            try:
                from roop_3d.telemetry.runtime_report import build_runtime_report, render_report
                report = build_runtime_report(getattr(self.state, "root", None), self.state, job=job, trigger=trigger)
                self.block(render_report(report, width=self.columns() + 1, ansi=self.color))
            except Exception as error:
                self.line(f"[REPORT] runtime report unavailable ({type(error).__name__})")
        if wait:
            work()
        else:
            threading.Thread(target=work, name="roop3d-runtime-report", daemon=True).start()

    def boot(self, state: Any = None) -> None:
        """Server boot: print the report for ``state``. Does nothing unless ``arm()`` ran (only ``main()`` arms), and must be
        called after the server's own ``ready`` line so Pinokio captures the real loopback URL first."""
        if not self.armed:
            return
        self.state = state if state is not None else self.state
        self.print_report("server boot")

    def job_started(self, job_id: str, *, swap_mode: Any = None, total_frames: int | None = None,
                    budget_mib: float | None = None, report: bool = True) -> None:
        if not self.armed:
            return
        budget = budget_mib if budget_mib is not None else self._budget_mib()
        with self._lock:
            self._jobs[job_id] = JobHud(job_id, swap_mode_label(swap_mode), total_frames, self.clock(), budget)
        self.sampler.reset_peak()
        self.sampler.acquire()
        self.milestone("JOB", "started", job=job_id[:8], mode=swap_mode_label(swap_mode), frames=total_frames)
        if report:
            self.print_report("job start", job={"job_id": job_id, "swap_mode": str(swap_mode), "total_frames": total_frames})

    def frame(self, job_id: str, *, frame: int, total: int | None, stage: str, stage_timings_ms: Mapping[str, float] | None = None) -> None:
        if not self.armed:
            return
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            if stage not in ("render", "angle_scan"):
                self._stage_change(job, stage, frame)
                return
            now = self.clock()
            if job.stage != stage:
                # A new phase has its own frame counter, rate and ETA.
                job.stage, job.last, job.last_frame, job.current_fps = stage, None, 0, None
                job.samples.clear()
                if stage == "angle_scan":
                    self.milestone("SCAN", "pre-pass: tracking the target's head angles", job=job.job_id[:8], frames=total)
                elif job.total and total:
                    job.total = total
            if total and (not job.total or stage == "angle_scan"):
                job.total = total
            if job.last is not None and now > job.last[0] and frame >= job.last[1]:
                job.current_fps = (frame - job.last[1]) / (now - job.last[0])
            job.last, job.last_frame = (now, frame), frame
            job.samples.append((now, frame))
            if stage_timings_ms:
                job.last_timings = dict(stage_timings_ms)
            rolling = rolling_fps(job.samples, self.rolling_window)
            line = render_frame_line(job, frame=frame, rolling=rolling, reading=self.sampler.latest(),
                                     peak_mib=self.sampler.peak_vram_mib, timings=job.last_timings, width=self.columns(),
                                     color=self.color)
            self._live["bar"] = line
            self._draw_live(force=bool(job.total and frame >= job.total))

    def _stage_change(self, job: JobHud, stage: str, frame: int) -> None:
        if stage == "mux" and not job.muxing:
            job.muxing = True
            self._live.pop("bar", None)  # the bar is finished; the encoder prints nothing until it is done
            self.milestone("MUX", "encode and audio remux started", job=job.job_id[:8], frames=frame)

    def job_finished(self, job_id: str, outcome: str, **fields: Any) -> None:
        if not self.armed:
            return
        with self._lock:
            job = self._jobs.pop(job_id, None)
            self._live.pop("bar", None)
        if job is None:
            return
        elapsed = self.clock() - job.started
        rolling = rolling_fps(job.samples, self.rolling_window)
        self.milestone("JOB", outcome, job=job_id[:8], frames=job.last_frame or None, seconds=f"{elapsed:.1f}",
                       peak_vram_gb=fmt_gb(self.sampler.peak_vram_mib) if self.sampler.peak_vram_mib is not None else None,
                       last_fps=None if rolling is None else f"{rolling:.1f}", **fields)
        self.sampler.release()

    # -- TensorRT heartbeat ---------------------------------------------------------------------------------
    def spawn_trt_sidecar(self, info: Mapping[str, Any]) -> "TrtSidecar | None":
        """Start the out-of-process heartbeat printer for a TensorRT build, or ``None`` (caller then prints in-process).

        Needed because ONNX Runtime holds the GIL for the whole build (see ``trt_heartbeat``): only another process can print
        while it runs. It writes to the terminal this process inherited, so a custom HUD stream cannot use it."""
        if not self.armed or self._stream is not None or os.environ.get("ROOP3D_TRT_SIDECAR", "1").strip().lower() in {"0", "off", "false", "no"}:
            return None
        try:
            import subprocess
            from roop_3d.audit.project_paths import application_root
            command = [sys.executable, "-B", "-m", "roop_3d.telemetry.trt_heartbeat_sidecar", "--model", str(info["model"]),
                       "--parent", str(os.getpid()), "--started", repr(float(info["wall"]))]
            for flag, key in (("--precision", "precision"), ("--shape", "opt_shape")):
                if info.get(key):
                    command += [flag, str(info[key])]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, env=child_environment(), cwd=str(application_root()))
        except Exception:
            return None
        with self._lock:
            self._shown = ""   # the sidecar owns the live line now; the next permanent line must erase it
        return TrtSidecar(process)

    def trt_beat(self, line: str) -> None:
        if not self.armed:
            return
        with self._lock:
            self._live["trt"] = line
            self._draw_live(force=True)

    def trt_done(self, final_line: str) -> None:
        if not self.armed:
            return
        with self._lock:
            self._live.pop("trt", None)
        self.line(final_line)


class TrtSidecar:
    """Handle on the heartbeat printer process. ``stop`` closes its stdin (its cue to erase its line and exit); a stuck one is killed."""

    def __init__(self, process: Any):
        self.process = process

    def stop(self) -> None:
        import subprocess
        try:
            self.process.stdin.close()
        except (OSError, ValueError, AttributeError):
            pass
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)


class HudLogHandler(logging.StreamHandler):
    """Drop-in for the server's stdout ``StreamHandler``: log records clear the live line, print, and redraw it."""

    def __init__(self, stream: TextIO | None = None, hud: PinokioTerminalHUD | None = None):
        super().__init__(stream)
        self.hud = hud

    def emit(self, record: logging.LogRecord) -> None:
        hud = self.hud if self.hud is not None else HUD
        if not hud.armed:
            super().emit(record)
            return
        hud.suspend()
        try:
            super().emit(record)
        finally:
            hud.resume()


#: The process-wide HUD. Unarmed until the server entry point calls ``HUD.arm(...)``.
HUD = PinokioTerminalHUD()
