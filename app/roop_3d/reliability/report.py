"""Local, redacted diagnostic report.

The report answers "what happened on this machine" without ever containing the *content* of what was processed:

* no pixels, thumbnails, masks, landmarks, embeddings or FaceSet contents - those keys are dropped, and any numeric
  vector that looks like an embedding is replaced;
* no media or FaceSet names - they become stable short hashes (``media:3fa9c1d2``) so two entries about the same file
  can still be correlated by a person reading the report, but the name cannot be recovered;
* no user names or absolute paths - the application root becomes ``<APP>``, the home directory ``~``, any other
  absolute path ``<PATH>``;
* secrets are removed by the shared ``redact`` and log sanitizer.

The report stays on this machine: it is returned to the local caller and written under ``roop_3d/.runtime/diagnostics``.
Nothing in this module opens a network connection, and the route that serves it is part of the loopback-only API.
"""
from __future__ import annotations

from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any

from roop_3d import __version__
from roop_3d.core.redaction import redact
from roop_3d.core.storage import atomic_json
from roop_3d.reliability import disk, probe
from roop_3d.reliability.ledger import LEDGER
from roop_3d.reliability.monitor import MONITOR
from roop_3d.reliability.processes import REGISTRY, _bind_state

REPORT_SCHEMA = "roop3d.diagnostics/1"
DIAGNOSTICS_RELATIVE = "roop_3d/.runtime/diagnostics"

# Keys whose values are biometric or media content. They are removed outright, at any depth.
_CONTENT_KEYS = {"embedding", "embeddings", "canonical_embedding", "medoid_embedding", "landmarks", "thumbnail_base64",
                 "thumbnail", "image_bytes", "mask_base64", "image", "crop", "pixels", "face_crop", "mesh",
                 "vertices", "texture", "pose_cache", "keyframe_curves", "manual_mask"}
# Model names are intentionally NOT here: "which model, which version" is the point of the report.
_NAME_KEYS = {"faceset_id", "faceset_ids", "track_id", "subject_name", "target_path", "filename", "source_path",
              "inspection_id"}
_MEDIA_EXT = r"(?:mp4|mov|mkv|avi|webm|png|jpe?g|webp|bmp|tiff?|fsz|npz|wav|mp3|flac|srt|ass)"
_MEDIA_NAME = re.compile(rf"[^\\/\s\"']+\.{_MEDIA_EXT}\b", re.IGNORECASE)
_ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/][^\s\"']*|(?<![\w:])/(?:home|Users|root|mnt|media|tmp|var|opt)/[^\s\"']*)")
_LONG_B64 = re.compile(r"[A-Za-z0-9+/=]{120,}")


def _short(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:8]


class Redactor:
    def __init__(self, root: Path, extra_names: tuple[str, ...] = ()):
        self.root = str(Path(root))
        self.home = str(Path.home())
        try:
            self.user = getpass.getuser()
        except Exception:
            self.user = ""
        self.extra = tuple(name for name in extra_names if len(name) >= 3)
        self.stats = {"content_fields_removed": 0, "names_hashed": 0, "paths_masked": 0, "vectors_removed": 0}

    def text(self, value: str) -> str:
        for variant in {self.root, self.root.replace("\\", "/"), self.root.replace("/", "\\")}:
            if variant and variant in value:
                value = value.replace(variant, "<APP>")
                self.stats["paths_masked"] += 1
        for variant in {self.home, self.home.replace("\\", "/")}:
            if variant and variant in value:
                value = value.replace(variant, "~")
                self.stats["paths_masked"] += 1
        for name in self.extra:
            if name in value:
                value = value.replace(name, f"id:{_short(name)}")
                self.stats["names_hashed"] += 1
        # Media file names: keep the extension (useful for triage), hide the name.
        value = _MEDIA_NAME.sub(lambda m: f"media:{_short(m.group(0))}{os.path.splitext(m.group(0))[1].lower()}", value)
        if _ABS_PATH.search(value):
            value = _ABS_PATH.sub("<PATH>", value)
            self.stats["paths_masked"] += 1
        if self.user and len(self.user) >= 3:
            value = re.sub(re.escape(self.user), "<user>", value, flags=re.IGNORECASE)
        return _LONG_B64.sub("<binary>", value)

    def value(self, item: Any, key: str = "") -> Any:
        lowered = key.lower()
        if lowered in _CONTENT_KEYS:
            self.stats["content_fields_removed"] += 1
            return "<removed:biometric-or-media-content>"
        if isinstance(item, dict):
            return {self.text(str(k)): self.value(v, str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple, set)):
            numbers = [x for x in item if isinstance(x, (int, float)) and not isinstance(x, bool)]
            if len(item) >= 8 and len(numbers) == len(item) and any(isinstance(x, float) for x in numbers) and lowered not in {"series"}:
                self.stats["vectors_removed"] += 1
                return f"<removed:numeric-vector len={len(item)}>"
            return [self.value(x, key) for x in item]
        if isinstance(item, str):
            if lowered in _NAME_KEYS:
                self.stats["names_hashed"] += 1
                return f"id:{_short(item)}"
            return self.text(item)
        if isinstance(item, bytes):
            self.stats["content_fields_removed"] += 1
            return f"<removed:{len(item)} bytes>"
        return item

    def apply(self, document: dict[str, Any]) -> dict[str, Any]:
        cleaned = self.value(redact(document))
        cleaned["redaction"] = {"applied": True, **self.stats,
                                "policy": "no media, faces, embeddings, names or absolute paths"}
        return cleaned


# ------------------------------------------------------------------------------------------------- sections
def _hardware(settings: Any) -> dict[str, Any]:
    try:
        from roop_3d.core.hardware import probe_hardware
        hw = probe_hardware(settings)
    except Exception as error:
        return {"state": "UNAVAILABLE", "error": type(error).__name__}
    return {
        "cpu": {"logical": hw.logical_cpu_count, "physical": hw.physical_cpu_count, "name": platform.processor() or None},
        "ram_mib": {"total": hw.system_ram_total_mib, "available": hw.system_ram_available_mib},
        "gpus": [{"index": g.index, "name": g.name, "vram_total_mib": g.total_vram_mib, "vram_free_mib": g.free_vram_mib,
                  "vram_budget_mib": g.effective_vram_budget_mib, "compute_capability": g.compute_capability} for g in hw.gpus],
        "probe_state": hw.gpu_probe_state, "notes": hw.probe_notes[:10],
        "_drivers": {"nvidia_driver": hw.gpus[0].driver_version if hw.gpus else None,
                     "cuda_runtime": hw.cuda_runtime_version, "driver_supported_cuda": hw.driver_supported_cuda_version,
                     "tensorrt": hw.tensorrt_version, "onnxruntime": hw.ort_version},
        "_providers": {"advertised": hw.advertised_providers or [], "verified": hw.provider_execution_verified,
                       "ffmpeg_nvenc": hw.ffmpeg_nvenc_encoders[:6], "ffmpeg_nvdec": hw.ffmpeg_nvdec_decoders[:6]},
    }


def _libraries() -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": sys.version.split()[0]}
    for module in ("numpy", "cv2", "onnxruntime", "av", "fastapi", "pydantic"):
        try:
            versions[module] = getattr(__import__(module), "__version__", None)
        except Exception:
            versions[module] = None
    try:
        import av
        versions["ffmpeg_libs"] = ",".join(f"{name}={'.'.join(map(str, v[:3]))}" if isinstance(v, tuple) else f"{name}={v}"
                                           for name, v in sorted(av.library_versions.items())[:4])
    except Exception:
        pass
    return versions


def _models(catalog: Any) -> dict[str, Any]:
    try:
        records = catalog.public()
    except Exception as error:
        return {"state": "UNAVAILABLE", "error": type(error).__name__}
    rows, states = [], {}
    for record in records:
        state = str(record.get("state", "UNKNOWN"))
        states[state] = states.get(state, 0) + 1
        rows.append({"name": record.get("name"), "role": record.get("role") or record.get("task"),
                     "state": state, "version": record.get("version"), "bytes": record.get("size") or record.get("byte_size"),
                     "sha256": str(record.get("sha256") or "")[:16] or None})
    try:
        from roop_3d.models.catalog import render_dependencies
        render_dependencies(catalog)
        render_ready = True
    except Exception as error:
        render_ready = f"missing: {getattr(error, 'code', type(error).__name__)}"
    return {"count": len(rows), "by_state": states, "render_dependencies": render_ready, "models": rows}


def _jobs(store: Any) -> dict[str, Any]:
    try:
        records = store.list()
    except Exception as error:
        return {"state": "UNAVAILABLE", "error": getattr(error, "code", type(error).__name__)}
    by_state: dict[str, int] = {}
    failures: dict[str, int] = {}
    recent = []
    for record in records:
        by_state[record.state.value] = by_state.get(record.state.value, 0) + 1
        if record.failure is not None:
            failures[record.failure.code] = failures.get(record.failure.code, 0) + 1
    for record in records[-10:]:
        diagnostics = record.diagnostics.models if record.diagnostics else {}
        recent.append({
            "job": record.job_id[:8], "state": record.state.value, "revision": record.revision,
            "frame": record.progress.frame, "total_frames": record.progress.total_frames,
            "fps": round(record.progress.fps, 2), "stage": record.progress.stage,
            "stage_timings_ms": dict(record.progress.stage_timings_ms),
            "failure": record.failure.model_dump(mode="json") if record.failure else None,
            "tracks": len(record.request.face_mappings), "engine": record.request.engine_id,
            "providers": {name: model.provider for name, model in diagnostics.items()},
            "fallbacks": {name: [f.reason[:120] for f in model.fallback_history] for name, model in diagnostics.items()
                          if model.fallback_history},
            "age_s": round((datetime.now(timezone.utc) - record.created_at).total_seconds())})
    return {"total": len(records), "by_state": by_state, "failure_codes": failures, "recent": recent}


def _disk(root: Path, settings: Any) -> dict[str, Any]:
    usage = {}
    for label, relative in (("outputs", settings.output_root), ("job_workdirs", "roop_3d/.runtime/jobs"),
                            ("quarantine", "roop_3d/.runtime/quarantine"), ("cache", "roop_3d/.runtime/cache"),
                            ("media", "roop_3d/.runtime/media"), ("diagnostics", DIAGNOSTICS_RELATIVE)):
        usage[label + "_mib"] = round(disk.tree_bytes(root / relative) / probe.MIB, 1)
    try:
        total = __import__("shutil").disk_usage(root)
        usage.update(volume_total_mib=total.total // probe.MIB, volume_free_mib=total.free // probe.MIB,
                     volume_used_pct=round(100 * total.used / total.total, 1))
    except OSError:
        pass
    return usage


def build_report(state: Any, *, audit: dict[str, Any] | None = None, include_series: bool = False) -> dict[str, Any]:
    """``state`` is ``app.state``. Returns the *redacted* document; the raw one never leaves this function."""
    root: Path = state.root
    settings = state.settings
    started = time.monotonic()
    hardware = _hardware(settings)
    drivers = hardware.pop("_drivers", {})
    providers = hardware.pop("_providers", {})
    growth = MONITOR.growth()
    gpu = probe.gpu_memory()
    memory = probe.process_memory()
    names: list[str] = []
    try:
        names = [item.faceset_id for record in state.jobs.list() for item in record.request.face_mappings]
        names += [Path(record.request.target_path).stem for record in state.jobs.list()]
    except Exception:
        pass
    document = {
        "schema": REPORT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy": {"local_only": True, "transmitted": False, "contains_media": False, "contains_biometrics": False},
        "application": {"name": "Roop 3D", "version": __version__, "platform": platform.platform(),
                        "pid": os.getpid(), "uptime_s": round(time.monotonic() - MONITOR.started, 1)},
        "hardware": hardware,
        "drivers": {**drivers, "libraries": _libraries(), "gpu_driver_via_smi": gpu.driver_version},
        "models": _models(getattr(state, "models", None)),
        "execution_providers": {"configured": {"preferred": settings.preferred_provider,
                                               "hardware_profile": settings.hardware_profile,
                                               "execution_mode": settings.execution_mode},
                                **providers},
        "pipeline_stage_timings_ms": MONITOR.summary()["stage_timings_ms"],
        "errors_and_warnings": {"log_counts": MONITOR.warnings.counts(), "recent": list(MONITOR.warnings.records)[-25:],
                                "audit_unresolved": (audit or {}).get("unresolved")},
        "memory": {
            "process": {"rss_mib": memory.rss_mib, "private_mib": memory.private_mib, "peak_rss_mib": memory.peak_rss_mib,
                        "handles": memory.handles, "os_threads": memory.os_threads},
            "gpu": {"available": gpu.available, "used_mib": gpu.used_mib, "total_mib": gpu.total_mib,
                    "process_mib": gpu.process_mib, "per_process_accounting": gpu.per_process,
                    "note": None if gpu.per_process else "driver does not report per-process VRAM (WDDM); device total shown"},
            "torch_cuda": probe.torch_cuda_state(),
            "peaks": MONITOR.summary()["peaks"], "job_peaks": MONITOR.summary()["job_peaks"],
            "growth": growth.as_dict()},
        "resources": {"ledger": LEDGER.snapshot(), "orphan_guard": dict(_bind_state), "tracked_children": REGISTRY.live_count()},
        "jobs": _jobs(state.jobs),
        "disk": _disk(root, settings),
        "reliability_audit": audit,
    }
    if include_series:
        document["memory"]["series"] = {k: [[round(x, 1), round(y, 2)] for x, y in v[-120:]] for k, v in MONITOR.series().items()}
    document["generation_ms"] = round((time.monotonic() - started) * 1000, 1)
    return Redactor(root, tuple(dict.fromkeys(names))).apply(document)


def save_report(root: Path, document: dict[str, Any], keep: int = 20) -> str:
    """Write the report locally (atomic) and prune old ones. Returns the relative path."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    relative = f"{DIAGNOSTICS_RELATIVE}/diagnostics-{stamp}.json"
    atomic_json(root, relative, document)
    directory = root / DIAGNOSTICS_RELATIVE
    old = sorted(directory.glob("diagnostics-*.json"))[:-keep]
    for path in old:
        path.unlink(missing_ok=True)
    return relative


def dumps(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, sort_keys=True, default=str)
