"""Swap-engine readiness: one honest row per supported swapper, shared by ``/api/v1/engines/readiness`` and ``/api/v1/health``.

A row is ``READY`` only when ``ModelCatalog.swapper_eligibility`` admits it (a supported adapter exists, the registry does not
block every record for it, and a weight resolves under the app/models local-use policy or a registry-verified artifact). A
swapper whose weights are on disk is therefore never reported unavailable because it has not been *loaded* yet, and one
whose weights are genuinely absent stays ``UNAVAILABLE`` with the catalog's own reason. Nothing here loads a graph.

``session`` carries what ONNX Runtime really did the last time that weight was opened: the provider that built the session
(TensorRT -> CUDA -> CPU), the providers it fell back from and why, and the tensor dtypes the graph expects, so an FP16
export and an FP32 export are distinguishable per provider.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

SUMMARY_TTL_SECONDS = 20.0


def _session_for(path: str | None, model_id: str) -> dict[str, Any] | None:
    from roop_3d.core.ort_session import session_selection
    names = [model_id]
    if path:
        names = [Path(path).name, Path(path).stem, *names]
    return session_selection(*names)


def swap_engine_report(catalog: Any) -> list[dict[str, Any]]:
    from roop_3d.engine import active_models
    active = (active_models.snapshot() or {}).get("swapper")
    rows: list[dict[str, Any]] = []
    for item in catalog.swapper_readiness():
        artifact = item.get("artifact") or {}
        path = artifact.get("path")
        ready = bool(item.get("selectable"))
        loaded = bool(active and active.get("model_id") == item["model_id"])
        session = _session_for(path, item["model_id"]) if ready else None
        rows.append({
            "model_id": item["model_id"], "architecture": item.get("architecture"),
            "native_resolution": item.get("native_resolution"),
            "status": "READY" if ready else "UNAVAILABLE",
            "readiness": item.get("readiness"), "policy": item.get("policy"),
            "reason": None if ready and item.get("readiness") in ("READY_REGISTRY_VERIFIED", "LOCAL_UNREVIEWED") else item.get("reason"),
            "blocked": bool(item.get("blocked")),
            "weight": ({"path": path, "filename": artifact.get("filename"), "size_bytes": artifact.get("size_bytes"),
                        "sha256": artifact.get("sha256"), "trust_basis": artifact.get("trust_basis"),
                        "registry_model_id": artifact.get("registry_model_id"),
                        "local_use_policy": bool(artifact.get("local_use_policy"))} if path else None),
            "loaded": loaded, "loaded_provider": active.get("provider") if loaded else None,
            "loaded_precision": active.get("precision") if loaded else None,
            "session": session,
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"state": "READY" if any(row["status"] == "READY" for row in rows) else "NONE_AVAILABLE",
            "ready": [row["model_id"] for row in rows if row["status"] == "READY"],
            "unavailable": [{"model_id": row["model_id"], "readiness": row["readiness"], "reason": row["reason"]}
                            for row in rows if row["status"] != "READY"],
            "checked_at": time.time()}


_lock = threading.Lock()
_cache: dict[int, tuple[float, dict[str, Any]]] = {}
_running: set[int] = set()


def _refresh(catalog: Any) -> None:
    key = id(catalog)
    try:
        value = summarize(swap_engine_report(catalog))
    except Exception as error:  # a broken catalog must never take /health down
        value = {"state": "ERROR", "ready": [], "unavailable": [], "error": type(error).__name__, "checked_at": time.time()}
    with _lock:
        _cache[key] = (time.monotonic(), value)
        _running.discard(key)


def swap_engine_summary(catalog: Any, *, block: bool = False) -> dict[str, Any]:
    """Cached ``{state, ready, unavailable}`` for the health payload.

    The first scan hashes the weights (seconds), so it runs on a background thread and health reports ``SCANNING`` until it
    lands; ``block=True`` waits for it (tests, diagnostics)."""
    key = id(catalog)
    with _lock:
        cached = _cache.get(key)
        fresh = cached is not None and time.monotonic() - cached[0] < SUMMARY_TTL_SECONDS
        start = not fresh and key not in _running
        if start and not block:
            _running.add(key)
    if fresh:
        return cached[1]  # type: ignore[index]
    if block:
        _refresh(catalog)
        return _cache[key][1]
    if start:
        threading.Thread(target=_refresh, args=(catalog,), name="swap-engine-readiness", daemon=True).start()
    return cached[1] if cached is not None else {"state": "SCANNING", "ready": [], "unavailable": [], "checked_at": None}
