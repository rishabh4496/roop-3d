"""The 14-section structured runtime report (terminal at boot / job start, ``GET /api/v1/system/runtime-report``, UI drawer).

One collector per section. A collector reads *live* state (settings, hardware probe, registries, caches, counters) and
returns rows of ``label -> text``. It never raises (a failure becomes an ``UNKNOWN`` section naming the exception type),
never contacts the network, and never prints a secret: home-directory paths are masked as ``~`` and the share-mode
token is not read at all. A value that was not measured is the text ``not measured``; nothing here is a placeholder.

Section statuses: ``OK`` (verified or healthy), ``INFO`` (configuration, nothing to judge), ``WARN`` (needs attention),
``UNAVAILABLE`` (capability absent on this machine), ``UNKNOWN`` (the collector itself failed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
import platform
import shutil
import sys
import textwrap
import time
from typing import Any, Callable

from roop_3d import __version__
from roop_3d.telemetry.pinokio_terminal import AUDIO_POLICY_TEXT, TERMINAL_ENVIRONMENT, defuse, paint

NOT_MEASURED = "not measured"
SECTION_TITLES: tuple[tuple[str, str], ...] = (
    ("host", "Host / OS / Venv Isolation"),
    ("cpu", "CPU Cores & Worker Pool Allocation"),
    ("gpu", "GPU Architecture & Compute Capability"),
    ("providers", "Execution Provider Chain & Mixed Precision"),
    ("trt_cache", "TensorRT Engine Cache"),
    ("ring", "Zero-Copy OrtValue Ring Buffer"),
    ("vram", "VRAM Budget & OOM-Guard Thresholds"),
    ("governance", "Model Governance & SHA-256 Verification"),
    ("topology", "3D Head Topology & Anatomical Sub-Mesh"),
    ("swap_lock", "Full-Craniofacial Swap Lock"),
    ("camera_sh", "Physical Camera & SH Relighting"),
    ("gop_audio", "GOP Video Slicer & Audio Remux"),
    ("offline", "Offline Isolation Audit"),
    ("loopback", "Loopback Server & Deep-Link URLs"),
)
_ARCHITECTURES = {"5.": "Maxwell", "6.": "Pascal", "7.0": "Volta", "7.2": "Volta", "7.5": "Turing", "8.0": "Ampere", "8.6": "Ampere",
                  "8.7": "Ampere", "8.9": "Ada Lovelace", "9.": "Hopper", "10.": "Blackwell", "12.": "Blackwell"}
DEEP_LINKS = ("home", "faceswap", "batch", "processing", "facemgr", "extras", "gallery", "history", "settings")


@dataclass
class Section:
    number: int
    key: str
    title: str
    status: str = "INFO"
    rows: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def add(self, label: str, value: Any) -> None:
        self.rows.append((label, show(value)))

    def public(self) -> dict[str, Any]:
        return {"number": self.number, "key": self.key, "title": self.title, "status": self.status,
                "rows": [{"label": label, "value": value} for label, value in self.rows], "notes": list(self.notes)}


def show(value: Any) -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple, set)):
        return ", ".join(show(item) for item in value) or "none"
    return str(value)


def tilde(path: Any) -> str:
    """Mask the user's home directory so a pasted report does not carry the account name."""
    text = str(path)
    home = str(Path.home())
    if home and text.lower().startswith(home.lower()):
        text = "~" + text[len(home):]
    return text.replace("\\", "/")


def mib(value: float | int | None) -> str:
    return NOT_MEASURED if value is None else f"{value / 1024:.1f} GB ({int(value)} MiB)"


class _Context:
    """Everything a collector may read; the hardware probe is taken once per report."""

    def __init__(self, root: Path | None, state: Any, job: dict | None):
        self.state, self.job = state, job
        self.settings = getattr(state, "settings", None)
        if self.settings is None:
            from roop_3d.core.settings import Settings
            self.settings = Settings()
        if root is None:
            from roop_3d.audit.project_paths import application_root
            root = application_root()
        self.root = Path(root)
        self._hardware: Any = False

    @property
    def hardware(self) -> Any:
        if self._hardware is False:
            try:
                from roop_3d.core.hardware import probe_hardware
                self._hardware = probe_hardware(self.settings)
            except Exception:
                self._hardware = None
        return self._hardware

    @property
    def gpu(self) -> Any:
        hardware = self.hardware
        return hardware.gpus[0] if hardware is not None and hardware.gpus else None

    def attr(self, name: str) -> Any:
        return getattr(self.state, name, None) if self.state is not None else None


def _available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


# ----------------------------------------------------------------------------------------------- collectors
def _host(c: _Context, s: Section) -> None:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    s.add("Operating system", f"{_os_name()} ({platform.machine()})")
    s.add("Python", f"{platform.python_implementation()} {platform.python_version()}")
    s.add("Interpreter", tilde(sys.executable))
    s.add("Virtual environment", f"active · {tilde(sys.prefix)}" if in_venv else "NOT active (system interpreter)")
    s.add("Application root", tilde(c.root))
    contract = {key: os.environ.get(key) for key in TERMINAL_ENVIRONMENT}
    holds = all(contract[key] == value for key, value in TERMINAL_ENVIRONMENT.items())
    s.add("Terminal contract", " ".join(f"{key}={contract[key] or 'unset'}" for key in TERMINAL_ENVIRONMENT))
    s.add("stdout encoding / TTY", f"{getattr(sys.stdout, 'encoding', None) or 'unknown'} / {_isatty()}")
    s.add("Process id", os.getpid())
    s.status = "OK" if in_venv and holds else "WARN"
    if not in_venv:
        s.notes.append("Running outside a virtual environment: packages are not isolated from the system interpreter.")
    if not holds:
        s.notes.append("PYTHONUNBUFFERED/PYTHONUTF8/PYTHONIOENCODING are not all set; output may lag or garble.")


def _os_name() -> str:
    """``platform.release()`` says "10" on Windows 11 until Python 3.12; the build number (>= 22000) is the real discriminator."""
    system, release = platform.system(), platform.release()
    if system == "Windows":
        try:
            build = int(platform.version().split(".")[-1])
            return f"Windows {'11' if build >= 22000 else release} (build {build})"
        except ValueError:
            pass
    return f"{system} {release}"


def _isatty() -> str:
    try:
        return "tty" if sys.stdout.isatty() else "not a tty"
    except (AttributeError, ValueError):
        return "not a tty"


def _cpu(c: _Context, s: Section) -> None:
    hardware, settings = c.hardware, c.settings
    logical = hardware.logical_cpu_count if hardware else os.cpu_count()
    s.add("Logical / physical cores", f"{show(logical)} / {show(hardware.physical_cpu_count if hardware else None)}")
    s.add("System RAM", f"{mib(hardware.system_ram_total_mib) if hardware else NOT_MEASURED} · available {mib(hardware.system_ram_available_mib) if hardware else NOT_MEASURED}")
    s.add("Execution threads / restorer threads", f"{settings.execution_threads} / {settings.restorer_threads}")
    s.add("Restorer pool size · stream batch", f"{settings.restorer_pool_size} · {settings.stream_batch_size}")
    s.add("Concurrent jobs (max)", settings.max_concurrent_jobs)
    try:
        from roop_3d.execution.worker_pools import _cpu_workers
        s.add("Decode pool default workers", _cpu_workers(None))
    except Exception:
        s.add("Decode pool default workers", None)
    benchmark = c.attr("thread_benchmark")
    snapshot = benchmark.snapshot() if benchmark is not None else None
    if snapshot and snapshot.get("recommendation"):
        best = snapshot["recommendation"]
        s.add("Last pool benchmark", f"{best['execution_threads']} threads × pool {best['pool_size']} · {best['samples_per_second']:.1f} samples/s ({snapshot.get('provider')})")
        s.notes.append(snapshot.get("note", "Synthetic workload; not model or video FPS."))
    else:
        s.add("Last pool benchmark", f"{snapshot['state'].lower()} - never run" if snapshot and snapshot.get("state") == "IDLE" else "none yet")
    s.status = "WARN" if logical and settings.execution_threads > logical else "INFO"
    if s.status == "WARN":
        s.notes.append("execution_threads exceeds the logical core count.")


def _architecture(capability: str | None) -> str | None:
    if not capability:
        return None
    for prefix in sorted(_ARCHITECTURES, key=len, reverse=True):
        if capability.startswith(prefix):
            return _ARCHITECTURES[prefix]
    return None


def _gpu(c: _Context, s: Section) -> None:
    hardware, gpu = c.hardware, c.gpu
    if gpu is None or hardware is None:
        s.status = "UNAVAILABLE"
        s.add("GPU", "none detected - CPU profile")
        s.notes.append("No NVIDIA GPU was reported by the driver probe.")
        return
    architecture = _architecture(gpu.compute_capability)
    s.add("Device", f"{gpu.name} (index {gpu.index})")
    s.add("Compute capability", f"{show(gpu.compute_capability)}" + (f" · {architecture} (inferred from capability)" if architecture else ""))
    s.add("VRAM total / free", f"{mib(gpu.total_vram_mib)} / {mib(gpu.free_vram_mib)}")
    s.add("Driver · CUDA runtime · max CUDA", f"{gpu.driver_version} · {show(hardware.cuda_runtime_version)} · {show(hardware.driver_supported_cuda_version)}")
    s.add("TensorRT", hardware.tensorrt_version)
    s.add("Tensor-core precisions", hardware.tensor_core_precisions or None)
    s.add("NVENC / NVDEC (FFmpeg registration)", f"{show(hardware.nvenc_available)} / {show(hardware.nvdec_available)}")
    s.status = "OK"
    s.notes.append("Tensor-core precisions come from the capability table, not from running each precision.")


def _providers(c: _Context, s: Section) -> None:
    hardware, settings = c.hardware, c.settings
    from roop_3d.core.hardware import resolve_accelerator
    advertised = hardware.advertised_providers if hardware else None
    resolved = resolve_accelerator(settings.preferred_provider, advertised, bool(c.gpu))
    s.add("ONNX Runtime", hardware.ort_version if hardware else None)
    s.add("Advertised providers", advertised)
    s.add("Preferred → resolved", f"{settings.preferred_provider} → {resolved}")
    s.add("Precision policy", settings.precision_policy)
    mixed = hardware.tensorrt_mixed_precision if hardware else {}
    s.add("TRT mixed precision (FP16+FP32)", f"available={show(mixed.get('available'))} active={show(mixed.get('active'))} configured={show(mixed.get('configured'))}")
    if mixed.get("diagnostic"):
        s.add("Mixed-precision diagnostic", mixed["diagnostic"])
    try:
        from roop_3d.engine import active_models
        loaded = {stage: (f"{v['model_id']} on {str(v['provider']).replace('ExecutionProvider', '')} {v.get('precision', '')}".strip() if v else "nothing loaded")
                  for stage, v in active_models.snapshot().items()}
        for stage, text in loaded.items():
            s.add(f"Loaded · {stage}", text)
    except Exception:
        pass
    s.add("Provider execution verified", hardware.provider_execution_verified if hardware else None)
    s.status = "OK" if resolved != "CPUExecutionProvider" else "WARN"
    s.notes.append("An advertised provider is not proof it executes; sessions report what they actually built with.")
    if resolved == "CPUExecutionProvider":
        s.notes.append("Resolved to CPU: GPU rendering is unavailable with the current provider selection.")


def _dir_stats(path: Path, pattern: str) -> tuple[int, int, float | None]:
    files = [item for item in path.glob(pattern) if item.is_file()] if path.is_dir() else []
    newest = max((item.stat().st_mtime for item in files), default=None)
    return len(files), sum(item.stat().st_size for item in files), newest


def _trt_cache(c: _Context, s: Section) -> None:
    from roop_3d.audit.project_paths import application_root
    app = application_root()
    strict_dir, bundle_dir = app / "models/.trt_cache", app / "roop_3d/cache/engines"
    engines, engine_bytes, engine_newest = _dir_stats(strict_dir, "*.engine")
    bundles, bundle_bytes, bundle_newest = _dir_stats(bundle_dir, "*.bundle")
    newest = max((t for t in (engine_newest, bundle_newest) if t is not None), default=None)
    s.add("Strict builder cache (*.engine)", f"{engines} file(s) · {engine_bytes / 1048576:.0f} MiB · app/models/.trt_cache")
    s.add("EPContext bundles (*.bundle)", f"{bundles} file(s) · {bundle_bytes / 1048576:.0f} MiB · app/roop_3d/cache/engines")
    s.add("Newest engine written", datetime.fromtimestamp(newest, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if newest else "never")
    last = getattr(c.attr("trt_builder"), "last_build", None) or {}
    s.add("Last strict build", {k: last[k] for k in ("model", "precision", "compile_seconds") if k in last} or "none this session")
    tracker = c.attr("pipeline")
    snapshot = tracker.snapshot() if tracker is not None else None
    s.add("Compilation now", f"{snapshot.engine_state} · {snapshot.engine_model or '-'}" if snapshot else NOT_MEASURED)
    s.status = "INFO" if engines + bundles else "WARN"
    if not engines + bundles:
        s.notes.append("Cold cache: the first use of each TensorRT model compiles its engine (typically 15-60 s, shown live as [TRT-BUILDER]).")


def _ring(c: _Context, s: Section) -> None:
    settings = c.settings
    torch = _available("torch")
    s.add("Runtime frame ring", f"{settings.stream_ring_slots} slots × batch {settings.stream_batch_size} (pinned host memory)")
    s.add("Render backend", settings.render_backend)
    s.add("GPU OrtValue ring (torch-backed)", "available" if torch else "unavailable - torch is not installed")
    try:
        from roop_3d.core.transfer_ledger import TRANSFERS
        ledger = TRANSFERS.snapshot()
        total = ledger["total"]
        s.add("ORT boundary copies (since start)", (f"H2D {total['h2d_count']} · D2H {total['d2h_count']} · intermediate {total['intermediate_copies']}"
                                                    if ledger["measured"] and ledger["gpu_runs"] else "no GPU run measured yet"))
    except Exception:
        s.add("ORT boundary copies (since start)", None)
    s.status = "OK" if torch else "UNAVAILABLE"
    if not torch:
        s.notes.append("Without torch the pipeline copies at every ONNX Runtime boundary (host ↔ device); it is not zero-copy.")


def _vram(c: _Context, s: Section) -> None:
    settings, gpu = c.settings, c.gpu
    from roop_3d.engine.vram_governor import VRAMGovernor
    import inspect
    from roop_3d.engine.vram_watchdog import VRAM_OOM_THRESHOLD_PCT, VRAMWatchdog
    pressure = inspect.signature(VRAMWatchdog.__init__).parameters["pressure_pct"].default
    s.add("Resolved profile", c.hardware.resolved_hardware_profile if c.hardware else None)
    s.add("Effective budget", mib(gpu.effective_vram_budget_mib) if gpu else "0 (CPU)")
    s.add("Reserve · user limit · user budget", f"{settings.memory_safety_reserve_mb} MiB · {settings.vram_limit_mb or 'auto'} · {settings.vram_budget_mb or 'auto'}")
    s.add("Per-stream admission reservation", f"{VRAMGovernor.PER_STREAM_RESERVATION_MIB} MiB (admission policy, not a measured model size)")
    s.add("Watchdog thresholds", f"pressure {pressure:.0f}% · near-OOM {VRAM_OOM_THRESHOLD_PCT:.0f}%")
    s.add("OOM recovery", "checkpoint, then one retry on the CPU profile")
    try:
        from roop_3d.reliability.probe import gpu_memory
        live = gpu_memory(max_age=1.0)
        if live.available and live.total_mib:
            used_pct = 100.0 * (live.used_mib or 0) / live.total_mib
            s.add("Live device usage", f"{live.used_mib:.0f} / {live.total_mib:.0f} MiB ({used_pct:.0f}%)")
            s.status = "WARN" if used_pct >= pressure else "OK"
            if s.status == "WARN":
                s.notes.append("Device VRAM is already above the pressure threshold before a job starts.")
        else:
            s.add("Live device usage", NOT_MEASURED)
            s.status = "UNAVAILABLE" if gpu is None else "INFO"
    except Exception:
        s.add("Live device usage", None)
        s.status = "INFO"


def _governance(c: _Context, s: Section) -> None:
    catalog = c.attr("models")
    rows = catalog.public() if catalog is not None else []
    installed = [row for row in rows if row.get("installed", row.get("sha256") is not None)]
    hashed = [row for row in installed if row.get("sha256")]
    verified = [row for row in installed if str(row.get("state", "")).startswith("VERIFIED")]
    s.add("Catalogue entries", len(rows))
    s.add("Installed on disk", len(installed))
    s.add("With a recorded SHA-256", len(hashed))
    s.add("Verified (state VERIFIED_*)", len(verified))
    missing = [str(row.get("name") or row.get("model_id")) for row in installed if not row.get("sha256")]
    s.add("Installed without a recorded hash", missing[:6] + (["…"] if len(missing) > 6 else []) or "none")
    for row in hashed[:4]:
        s.add(f"  {row.get('name') or row.get('model_id')}", f"{str(row['sha256'])[:16]}… · {row.get('state', NOT_MEASURED)}")
    s.status = "WARN" if missing else ("OK" if hashed else "INFO")
    if not rows:
        s.notes.append("No model catalogue is attached to this process.")
    s.notes.append("A hash proves the file is the one recorded; it is not a statement about the model's quality or licence.")


def _topology(c: _Context, s: Section) -> None:
    from roop_3d.tools.check_topology_assets import check
    report = check()
    for item in report["topologies"]:
        s.add(f"Topology · {item['id']}", f"{item['state']}{' (default)' if item.get('default') else ''} · {item['license_id']}")
    try:
        from roop_3d.reconstruction.backbone_status import backbone_status
        backbone = backbone_status()
        if backbone["active"]:
            s.add("Dense-face backbone", f"{backbone['active']} (installed bundle)")
        else:
            reason = next((row["reason"] for row in backbone["backbones"] if row["id"] == backbone["default"]), "")
            s.add("Dense-face backbone", f"none active · {reason}".strip(" ·"))
    except Exception:
        s.add("Dense-face bundle", None)
    try:
        from roop_3d.reconstruction import anatomical_pack
        s.add("Anatomical sub-mesh pack", f"module present · {len(anatomical_pack.AnatomicalFullHeadAsset.__dataclass_fields__)} asset fields")
    except Exception:
        s.add("Anatomical sub-mesh pack", "unavailable")
    s.status = "OK" if report["default_ready"] else "WARN"
    if not report["default_ready"]:
        s.notes.append("The default head topology is not ready; run roop_3d.tools.check_topology_assets for the fix.")


def _swap_lock(c: _Context, s: Section) -> None:
    from roop_3d.core.workstation import WorkstationParameters
    live = c.job is not None and c.job.get("swap_mode")
    params = WorkstationParameters()
    scope = str(c.job["swap_mode"]) if live else params.swap_scope
    s.add("Source of values", "this job" if live else "configured defaults (no job running)")
    s.add("Swap scope", scope)
    s.add("Skull blend (source share)", params.skull_blend)
    for label, name in (("Neck protection", "neck_protection"), ("Hair matting", "hair_matting"), ("Preserve ears/glasses", "preserve_ears_glasses"),
                        ("Boundary ribbon matting", "boundary_ribbon_matting"), ("Jaw feather (px)", "jaw_feather_px")):
        s.add(label, getattr(params, name, None))
    modules = {name: _available(f"roop_3d.rendering.{name}") for name in ("silhouette_replacement", "perimeter_cage", "silhouette_inpaint")}
    s.add("Silhouette stages", ", ".join(f"{name}={'present' if ok else 'missing'}" for name, ok in modules.items()))
    locked = scope in {"full_craniofacial", "FULL_CRANIOFACIAL_LOCK"} and params.skull_blend >= 1.0
    s.add("Full-craniofacial lock", "engaged (100% source skull/jaw/forehead)" if locked else "not engaged")
    s.status = "OK" if locked and all(modules.values()) else "INFO"
    s.notes.append("Values are the configured request parameters; this section does not measure swap quality.")


def _camera_sh(c: _Context, s: Section) -> None:
    from roop_3d.rendering.physical_camera import FOCAL_RANGE_MM, SENSOR_WIDTH_MM_DEFAULT
    from roop_3d.core.workstation import WorkstationParameters
    params = WorkstationParameters()
    s.add("SH lighting model", "9-band (order 2) spherical harmonics · Ramamoorthi & Hanrahan basis")
    s.add("Relighting default", f"{'on' if params.relight else 'off'} · strength {params.relight_strength}")
    s.add("Ambient fill (manual key light)", params.ambient_fill)
    s.add("Camera model", f"full-frame sensor {SENSOR_WIDTH_MM_DEFAULT:g} mm · focal range {FOCAL_RANGE_MM[0]:g}-{FOCAL_RANGE_MM[1]:g} mm")
    s.add("Per-footage calibration", "estimated per target at render time (sharpness, blur, CA, grain, ISO); none stored")
    s.status = "INFO"
    s.notes.append("Calibration is measured from the footage when a job runs; nothing is pre-calibrated for your camera.")


def _gop_audio(c: _Context, s: Section) -> None:
    hardware, settings = c.hardware, c.settings
    from roop_3d.engine.pipeline_runner import MasterPipelineRunner
    s.add("FFmpeg", tilde(hardware.ffmpeg_path) if hardware and hardware.ffmpeg_path else "not found")
    s.add("NVENC encoders", (hardware.ffmpeg_nvenc_encoders[:4] if hardware and hardware.ffmpeg_nvenc_encoders else "none"))
    s.add("Checkpoint / GOP cadence", f"every {MasterPipelineRunner.CHECKPOINT_EVERY} frames (GOP-aligned resume point)")
    s.add("Stream segment length", f"{settings.stream_segment_frames} frames")
    s.add("Audio policy", AUDIO_POLICY_TEXT.get(settings.audio_policy, settings.audio_policy))
    s.add("Timing policy", settings.video_timing_policy)
    s.add("GOP chunk stitcher", "present (execution.worker_pools) - the job runner encodes once and does not use it")
    ok = bool(hardware and hardware.ffmpeg_path)
    s.status = "OK" if ok else "UNAVAILABLE"
    if not ok:
        s.notes.append("FFmpeg was not found: video output and audio remux are unavailable.")


def _offline(c: _Context, s: Section) -> None:
    from roop_3d.core.integrity import isolation_status
    audit = isolation_status(c.root)
    s.add("Isolation audit", audit["status"])
    s.add("Audit time · files checked", f"{show(audit.get('checked_at'))} · {show(audit.get('checked_files'))}")
    s.add("Audit scope", audit["scope"])
    s.add("Remote processing", "enabled" if c.settings.remote_processing_enabled else "disabled")
    s.add("Bind host", c.settings.host)
    proxies = [name for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY") if os.environ.get(name)]
    s.add("Proxy variables set", proxies or "none")
    s.status = "OK" if audit["status"] == "passed" and not c.settings.remote_processing_enabled else "WARN"
    if audit["status"] != "passed":
        s.notes.append("There is no passing isolation audit on record, so offline operation is configured but not verified.")


def _loopback(c: _Context, s: Section) -> None:
    host, port = c.settings.host, c.settings.port
    base = f"http://{host}:{port}"
    s.add("Bind", f"{host}:{port} (loopback only)")
    s.add("Workstation", f"{base}/")
    for screen in DEEP_LINKS:
        s.add(f"  deep link · {screen}", f"{base}/#/{screen}")
    s.add("Health", f"{base}/api/v1/health")
    s.add("Pipeline status stream", f"ws://{host}:{port}/ws/v1/pipeline/status")
    s.add("This report", f"{base}/api/v1/system/runtime-report")
    share = c.attr("share")
    s.add("LAN share mode", "ON - remote peers need the per-launch token printed once at start" if share is not None else "off")
    s.status = "WARN" if share is not None else "OK"
    if share is not None:
        s.notes.append("Share mode listens on every interface. The access token is never part of this report.")


_COLLECTORS: tuple[Callable[[_Context, Section], None], ...] = (
    _host, _cpu, _gpu, _providers, _trt_cache, _ring, _vram, _governance, _topology, _swap_lock, _camera_sh, _gop_audio, _offline, _loopback)


def build_runtime_report(root: Path | None = None, state: Any = None, *, job: dict | None = None, trigger: str = "on demand") -> dict[str, Any]:
    """Collect all 14 sections. ``job`` is ``{"job_id", "swap_mode", "total_frames"}`` when produced at job start."""
    context = _Context(root, state, job)
    sections: list[Section] = []
    for number, ((key, title), collector) in enumerate(zip(SECTION_TITLES, _COLLECTORS), start=1):
        section = Section(number, key, title)
        try:
            collector(context, section)
        except Exception as error:  # a diagnostic must degrade, never fail the caller
            section = Section(number, key, title, "UNKNOWN", [("collector", f"failed: {type(error).__name__}")],
                              ["This section could not be collected; the others are unaffected."])
        sections.append(section)
    counts = {status: sum(1 for item in sections if item.status == status) for status in ("OK", "INFO", "WARN", "UNAVAILABLE", "UNKNOWN")}
    report = {"report_id": f"r3d-{int(time.time())}", "generated_at": datetime.now(timezone.utc).isoformat(), "app": "roop-3d",
              "version": __version__, "trigger": trigger, "job": job, "counts": counts,
              "overall": "ATTENTION" if counts["WARN"] or counts["UNAVAILABLE"] or counts["UNKNOWN"] else "OK",
              "sections": [item.public() for item in sections]}
    report["text"] = render_report(report, ansi=False)
    return report


# ----------------------------------------------------------------------------------------------------- rendering
_STATUS_STYLE = {"OK": "green", "INFO": "cyan", "WARN": "yellow", "UNAVAILABLE": "yellow", "UNKNOWN": "red"}
LABEL_WIDTH = 38


def render_report(report: dict[str, Any], *, width: int | None = None, ansi: bool = True) -> str:
    """Box-drawn text. ``ansi=False`` is the copyable form (no escape codes); both defuse launcher trip-wire tokens."""
    width = max(72, min(width or shutil.get_terminal_size((110, 24)).columns, 130))
    inner = width - 4
    value_width = max(20, inner - LABEL_WIDTH - 2)
    out = []
    head = f" ROOP 3D RUNTIME REPORT · v{report['version']} · {report['trigger']} · {report['overall']} "
    if report.get("job"):
        head += f"· job {str(report['job'].get('job_id', ''))[:8]} "
    out.append("╔" + paint(head, "bold", enabled=ansi) + "═" * max(1, width - 2 - len(head)) + "╗")
    for section in report["sections"]:
        status = section["status"]
        title = f" {section['number']:>2} · {section['title'].upper()} "
        badge = f" {status} "
        fill = max(1, width - 2 - len(title) - len(badge))
        out.append("┌" + paint(title, "bold", enabled=ansi) + "─" * fill + paint(badge, _STATUS_STYLE.get(status, "dim"), "bold", enabled=ansi) + "┐")
        for row in section["rows"]:
            wrapped = textwrap.wrap(row["value"], value_width, break_long_words=True, break_on_hyphens=False) or [""]
            for index, piece in enumerate(wrapped):
                label = row["label"] if index == 0 else ""
                cell = f"{label[:LABEL_WIDTH]:<{LABEL_WIDTH}}  {piece}"
                out.append("│ " + paint(cell[:LABEL_WIDTH], "dim", enabled=ansi) + cell[LABEL_WIDTH:].ljust(inner - LABEL_WIDTH) + " │")
        for note in section["notes"]:
            for index, piece in enumerate(textwrap.wrap(note, inner - 3) or [""]):
                text = ("! " if index == 0 else "  ") + piece
                out.append("│ " + paint(text.ljust(inner), "yellow", enabled=ansi) + " │")
        out.append("└" + "─" * (width - 2) + "┘")
    return defuse("\n".join(out))
