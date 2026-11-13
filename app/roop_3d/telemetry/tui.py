"""Live terminal dashboard for headless/CLI renders: stage bars, FPS, frame-time percentiles, provider, VRAM, CPU/RAM, ETA.

Three layers, each usable without a terminal:

* :class:`FrameStats` - pure rolling-window math (FPS, p50/p95/p99 frame time, ETA). No I/O.
* :class:`SystemProbe` - one probe snapshot. A value that cannot be measured is ``None`` and its reason is listed in
  ``unavailable`` (never ``0``). Reuses ``core.hardware.probe_hardware`` (GPU, RAM, providers) and, when ``torch`` is already
  imported, its CUDA allocator counters for VRAM allocated/reserved.
* :func:`make_dashboard` - the REAL selection between a Rich ``Live`` dashboard and plain line output: Rich is used only when it is
  importable AND the output stream is a TTY AND ``headless`` is false; otherwise plain, ANSI-free lines.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import sys
import time
from typing import Any, Callable, Mapping, TextIO

import numpy as np

STAGE_BARS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Decode", ("decode",)),
    ("3D Track/Detect", ("detection", "pose_tps")),
    ("3D Mesh/Swap", ("inswapper",)),
    ("Restore/Blend", ("masking", "restorer")),
    ("Encode", ("encode",)),
)


class FrameStats:
    """Rolling per-frame wall-time window. ``record(ms)`` per finished frame; everything else is derived."""

    def __init__(self, window: int = 240, total_frames: int | None = None):
        if window < 1:
            raise ValueError("window must be positive")
        self.window = window
        self.total_frames = total_frames
        self.done = 0
        self._ms: deque[float] = deque(maxlen=window)

    def record(self, frame_ms: float) -> None:
        if not np.isfinite(frame_ms) or frame_ms < 0:
            raise ValueError("frame_ms must be finite and non-negative")
        self._ms.append(float(frame_ms))
        self.done += 1

    def percentiles(self) -> dict[str, float | None]:
        """Linear-interpolated percentiles of the window; ``None`` before the first frame."""
        if not self._ms:
            return {"p50": None, "p95": None, "p99": None}
        p50, p95, p99 = np.percentile(np.fromiter(self._ms, float), [50, 95, 99])
        return {"p50": float(p50), "p95": float(p95), "p99": float(p99)}

    def fps(self) -> float | None:
        """Frames per second over the window (``1000 / mean frame ms``); ``None`` before the first frame or if all frames took 0 ms."""
        if not self._ms:
            return None
        mean = sum(self._ms) / len(self._ms)
        return 1000.0 / mean if mean > 0 else None

    def eta_seconds(self) -> float | None:
        """Remaining frames at the windowed FPS; ``None`` when the total or the rate is unknown."""
        fps = self.fps()
        if self.total_frames is None or fps is None:
            return None
        return max(0, self.total_frames - self.done) / fps


def stage_shares(timings_ms: Mapping[str, float]) -> list[tuple[str, float | None, float | None]]:
    """``[(bar label, mean ms or None, share of measured total or None)]`` in pipeline order; unmeasured stages are ``None``."""
    ms: dict[str, float | None] = {}
    for label, keys in STAGE_BARS:
        present = [float(timings_ms[k]) for k in keys if k in timings_ms]
        ms[label] = sum(present) if present else None
    total = sum(v for v in ms.values() if v is not None)
    return [(label, value, (value / total if value is not None and total > 0 else None)) for label, value in ms.items()]


def format_eta(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60}:{seconds % 60:02d}"


# ------------------------------------------------------------------------------------------------------ system probe
@dataclass
class SystemSnapshot:
    provider: str | None = None
    gpu_name: str | None = None
    vram_total_mib: int | None = None
    vram_free_mib: int | None = None
    vram_allocated_mib: float | None = None
    vram_reserved_mib: float | None = None
    cpu_percent: float | None = None
    ram_total_mib: int | None = None
    ram_available_mib: int | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class SystemProbe:
    """Hardware snapshot with the process CPU load computed from ``time.process_time`` deltas (no psutil dependency)."""

    def __init__(self, hardware_probe: Callable[[], Any] | None = None, *, provider: str | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self._hardware_probe, self._provider, self._clock = hardware_probe, provider, clock
        self._last = (clock(), time.process_time())

    def snapshot(self) -> SystemSnapshot:
        out = SystemSnapshot(provider=self._provider)
        try:
            if self._hardware_probe is None:
                from roop_3d.core.hardware import probe_hardware
                hardware = probe_hardware()
            else:
                hardware = self._hardware_probe()
        except Exception as error:
            hardware = None
            for key in ("vram", "ram", "provider"):
                out.unavailable[key] = f"hardware probe failed: {type(error).__name__}"
        if hardware is not None:
            if hardware.gpus:
                gpu = hardware.gpus[0]
                out.gpu_name, out.vram_total_mib, out.vram_free_mib = gpu.name, gpu.total_vram_mib, gpu.free_vram_mib
            else:
                out.unavailable["vram"] = "no GPU was verified by the hardware probe"
            if hardware.system_ram_total_mib is not None:
                out.ram_total_mib, out.ram_available_mib = hardware.system_ram_total_mib, hardware.system_ram_available_mib
            else:
                out.unavailable["ram"] = "system RAM was not measured"
            if out.provider is None:
                active = (hardware.tensorrt_mixed_precision or {}).get("active_provider")
                out.provider = active or (hardware.advertised_providers[0] if hardware.advertised_providers else None)
                if out.provider is None:
                    out.unavailable["provider"] = "no execution provider advertised"
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                out.vram_allocated_mib = torch.cuda.memory_allocated() / 1048576
                out.vram_reserved_mib = torch.cuda.memory_reserved() / 1048576
            else:
                out.unavailable["vram_allocated_reserved"] = "torch CUDA allocator is not loaded in this process (ONNX Runtime owns VRAM)"
        except Exception as error:
            out.unavailable["vram_allocated_reserved"] = f"torch CUDA query failed: {type(error).__name__}"
        now, cpu = self._clock(), time.process_time()
        elapsed = now - self._last[0]
        if elapsed > 0.05:
            out.cpu_percent = 100.0 * (cpu - self._last[1]) / elapsed / (os.cpu_count() or 1)
            self._last = (now, cpu)
        else:
            out.unavailable["cpu"] = "sampling interval too short"
        return out


# ----------------------------------------------------------------------------------------------------------- output
def _num(value: float | None, unit: str = "", digits: int = 1) -> str:
    return "--" if value is None else f"{value:.{digits}f}{unit}"


def plain_line(stats: FrameStats, timings_ms: Mapping[str, float], system: SystemSnapshot, stage: str = "") -> str:
    """One ANSI-free status line; unmeasured values print ``--`` and are never zero."""
    pct = stats.percentiles()
    total = f"/{stats.total_frames}" if stats.total_frames else ""
    bars = " ".join(f"{label.split('/')[0].replace(' ', '')}={_num(ms, 'ms')}" for label, ms, _ in stage_shares(timings_ms))
    vram = ("--" if system.vram_total_mib is None else
            f"{_num(system.vram_allocated_mib, '', 0)}/{_num(system.vram_reserved_mib, '', 0)}/{system.vram_total_mib}MiB(free {system.vram_free_mib})")
    return (f"frame {stats.done}{total} stage={stage or '--'} fps={_num(stats.fps())} "
            f"ft p50/p95/p99={_num(pct['p50'])}/{_num(pct['p95'])}/{_num(pct['p99'])}ms eta={format_eta(stats.eta_seconds())} "
            f"provider={system.provider or '--'} vram(alloc/res/total)={vram} cpu={_num(system.cpu_percent, '%', 0)} "
            f"ram_avail={_num(system.ram_available_mib, 'MiB', 0)} | {bars}")


class PlainDashboard:
    """Line-oriented reporter (selected when Rich is missing, stdout is not a TTY, or ``--headless``)."""
    kind = "plain"

    def __init__(self, stream: TextIO | None = None, *, every_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic):
        self.stream, self.every, self._clock, self._last = stream or sys.stdout, every_seconds, clock, None

    def __enter__(self) -> "PlainDashboard":
        return self

    def __exit__(self, *_exc) -> None:
        return None

    def update(self, stats: FrameStats, timings_ms: Mapping[str, float], system: SystemSnapshot, stage: str = "", *, force: bool = False) -> None:
        now = self._clock()
        if not force and self._last is not None and now - self._last < self.every:
            return
        self._last = now
        print(plain_line(stats, timings_ms, system, stage), file=self.stream, flush=True)


class RichDashboard:
    """Rich ``Live`` dashboard. Constructed only when Rich is importable and the output is an interactive terminal."""
    kind = "rich"

    def __init__(self, stream: TextIO | None = None):
        from rich.console import Console
        from rich.live import Live
        self._console = Console(file=stream or sys.stdout)
        self._live = Live(console=self._console, refresh_per_second=4, transient=False)

    def __enter__(self) -> "RichDashboard":
        self._live.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._live.__exit__(*exc)

    def update(self, stats: FrameStats, timings_ms: Mapping[str, float], system: SystemSnapshot, stage: str = "", *, force: bool = False) -> None:
        from rich.console import Group
        from rich.panel import Panel
        from rich.progress_bar import ProgressBar
        from rich.table import Table
        pct = stats.percentiles()
        head = Table.grid(padding=(0, 2))
        head.add_row("Stage", stage or "--", "Frames", f"{stats.done}" + (f"/{stats.total_frames}" if stats.total_frames else ""),
                     "ETA", format_eta(stats.eta_seconds()))
        head.add_row("FPS", _num(stats.fps()), "Frame ms p50/p95/p99",
                     f"{_num(pct['p50'])}/{_num(pct['p95'])}/{_num(pct['p99'])}", "Provider", system.provider or "--")
        head.add_row("VRAM alloc/res", f"{_num(system.vram_allocated_mib, '', 0)}/{_num(system.vram_reserved_mib, '', 0)} MiB",
                     "VRAM free/total", f"{_num(system.vram_free_mib, '', 0)}/{_num(system.vram_total_mib, '', 0)} MiB",
                     "CPU / RAM free", f"{_num(system.cpu_percent, '%', 0)} / {_num(system.ram_available_mib, ' MiB', 0)}")
        bars = Table.grid(padding=(0, 1))
        for label, ms, share in stage_shares(timings_ms):
            bars.add_row(f"{label:<16}", ProgressBar(total=100, completed=100 * (share or 0), width=28), _num(ms, " ms"))
        overall = ProgressBar(total=max(stats.total_frames or 1, 1), completed=min(stats.done, stats.total_frames or stats.done), width=48)
        self._live.update(Panel(Group(head, bars, overall), title="roop-3d render"), refresh=force)


def rich_available() -> bool:
    try:
        import rich  # noqa: F401
        return True
    except ImportError:
        return False


def make_dashboard(*, headless: bool = False, stream: TextIO | None = None, is_tty: bool | None = None,
                   rich_importable: bool | None = None) -> PlainDashboard | RichDashboard:
    """Rich only if importable AND ``stream`` is a TTY AND not ``headless``; otherwise the plain line reporter."""
    stream = stream or sys.stdout
    tty = bool(getattr(stream, "isatty", lambda: False)()) if is_tty is None else is_tty
    usable = rich_available() if rich_importable is None else rich_importable
    if usable and tty and not headless:
        return RichDashboard(stream)
    return PlainDashboard(stream)
