"""Read-only component diagnostics: artifact availability is not a loaded session or a forward proof.

The optional .runtime/stage1_verification.json has schema_version=1, generated_at,
and results[{component, path, size_bytes, mtime_ns, provider, execution_verified,
error}]. Only paths already resolved by the pure-3D registry are inspected. A
receipt applies only while its exact source path, size and mtime still match;
it never implies a session is resident in this server or enables the renderer.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import time
from typing import Any

from roop_3d.audit.isolation import safe_path
from roop_3d.core.ort_session import session_selection, session_selections
from roop_3d.server.true3d import renderer_status

REPORT_RELATIVE = "roop_3d/.runtime/stage1_verification.json"
_LOG = logging.getLogger("roop3d.components")


def _error(error: Exception) -> dict[str, Any]:
    return {"code": getattr(error, "code", type(error).__name__),
            "message": getattr(error, "message", str(error)),
            "status": getattr(error, "status", 503)}


def _report(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        path = safe_path(root, REPORT_RELATIVE)
        if not path.exists():
            return {"state": "not_run", "generated_at": None, "error": None}, []
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("Stage 1 verification report exceeds 2 MiB")
        value = json.loads(path.read_text("utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("results"), list):
            raise ValueError("Stage 1 verification report requires schema_version=1 and results[]")
        generated_at = value.get("generated_at")
        return {"state": "recorded", "generated_at": generated_at if isinstance(generated_at, str) else None, "error": None}, [
            row for row in value["results"] if isinstance(row, dict)]
    except Exception as error:
        return {"state": "invalid", "generated_at": None, "error": _error(error)}, []


def _identity(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    target = Path(path).resolve(strict=True)
    stat = target.stat()
    if not target.is_file():
        raise ValueError("Resolved component artifact is not a regular file")
    return {"path": str(target), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _same_path(left: Any, right: str) -> bool:
    # Never open/stat a report- or session-supplied path; it is only compared to the resolved inventory.
    return isinstance(left, str) and Path(left).is_absolute() and Path(left) == Path(right)


def _forward(row: dict[str, Any], identity: dict[str, Any] | None,
             results: list[dict[str, Any]]) -> dict[str, Any]:
    empty = {"state": "not_run", "provider": None, "error": None}
    if identity is None:
        return empty
    candidates = [result for result in results if result.get("component") == row["component"]
                  and _same_path(result.get("path"), identity["path"])]
    if not candidates:
        return empty
    result = candidates[-1]
    if (type(result.get("size_bytes")) is not int or type(result.get("mtime_ns")) is not int
            or result["size_bytes"] != identity["size_bytes"] or result["mtime_ns"] != identity["mtime_ns"]):
        return {**empty, "state": "stale", "error": "Resolved file identity differs from the saved forward receipt"}
    error = result.get("error")
    provider = result.get("provider")
    if error is not None and not isinstance(error, str) or provider is not None and not isinstance(provider, str):
        return {**empty, "state": "failed", "error": "Forward receipt has invalid provider/error fields"}
    verified = result.get("execution_verified") is True and error is None
    return {"state": "verified" if verified else "failed", "provider": provider, "error": error}


def _decorate(row: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    row = dict(row)
    try:
        identity = _identity(row.get("path"))
    except Exception as error:
        identity = None
        if row.get("error") is None:
            row.update(state="UNAVAILABLE", error=_error(error))
    diagnostic = session_selection(*(str(value) for value in
        (row.get("filename"), row.get("model_id")) if value)) if row.get("kind") == "weight" else None
    if row.get("kind") == "weight" and identity:
        # A registry filename and a user's compatible local-export filename can differ.
        matches = [entry for entry in session_selections().values() if _same_path(entry.get("path"), identity["path"])]
        if matches:
            diagnostic = max(matches, key=lambda entry: entry.get("selected_at", 0))
    live = {"load_status": "not_loaded", "active_provider": None, "error": None,
            "fallback_errors": [], "fallback_reason": None, "warnings": [], "requested_providers": [],
            "execution_verified": False, "session_path": None}
    if diagnostic and identity and _same_path(diagnostic.get("path"), identity["path"]):
        # A last session selection is not evidence about bytes replaced since that selection.
        selected_at = diagnostic.get("selected_at")
        fresh = (diagnostic.get("source_size_bytes") == identity["size_bytes"]
                 and diagnostic.get("source_mtime_ns") == identity["mtime_ns"])
        if diagnostic.get("load_status") == "failed":
            fresh = isinstance(selected_at, (int, float)) and identity["mtime_ns"] <= int(selected_at * 1e9)
        if fresh:
            live.update({key: diagnostic.get(key, default) for key, default in live.items()})
        else:
            live.update(load_status="stale", error="Resolved file changed after the recorded session selection")
    if row.get("kind") != "weight":
        live["load_status"] = "not_applicable"
    if row.get("component") == "detector" and live["load_status"] == "not_loaded" and identity:
        from roop_3d.engine import active_models
        active = active_models.snapshot().get("detector")
        if active and _same_path(active.get("model_path"), identity["path"]) and identity["mtime_ns"] <= int(active["loaded_at"] * 1e9):
            live.update(load_status="loaded", active_provider=active["provider"], session_path=identity["path"])
    row.update(file_identity=identity, runtime=live, last_forward=_forward(row, identity, results))
    row["assets"] = [_decorate(asset, results) for asset in row.get("assets", [])]
    return row


def component_status(state) -> dict[str, Any]:
    """Resolve local dependencies and read diagnostics; never create inference sessions or execute a forward."""
    from roop_3d.models.pure3d import resolve_components

    root = Path(state.root)
    report, results = _report(root)
    rows = [_decorate(row, results) for row in resolve_components(state.models, root=root)]
    renderer = renderer_status()
    # UV synthesis is a genuine standalone mathematical implementation; production rasterization is a separate gate.
    source = Path(__file__).resolve().parents[1] / "rendering" / "cpu_perspective_raster.py"
    uv_renderer = {"component": "uv_renderer", "label": "UV renderer / mesh rasterization", "kind": "implementation",
                   "required": True, "state": "READY" if renderer["wired"] else "UNAVAILABLE",
                   "model_id": None, "filename": source.name, "path": str(source), "sha256": None,
                   "size_bytes": source.stat().st_size, "trust_basis": "IMPLEMENTATION", "integrity_verified": False,
                   "license": None, "assets": [], "alternatives": [], "acquisition": None,
                   "error": None if renderer["wired"] else {"code": renderer["code"], "message": renderer["message"], "status": 503}}
    rows.insert(6, _decorate(uv_renderer, results))
    # Log changes, not every poll. The application owns this bounded per-component read-back cache.
    previous = getattr(state, "_component_diagnostic_log", {})
    current = {}
    for row in [*rows, *(asset for parent in rows for asset in parent.get("assets", []))]:
        diagnostic = {key: row.get(key) for key in ("component", "state", "path", "runtime", "error", "last_forward")}
        signature = json.dumps(diagnostic, sort_keys=True)
        current[row["component"]] = signature
        if previous.get(row["component"]) != signature:
            _LOG.info("Component diagnostic: %s", signature)
    state._component_diagnostic_log = current
    try:
        import onnxruntime as ort
        advertised = ort.get_available_providers()
        provider_error = None
    except Exception as error:
        advertised, provider_error = [], _error(error)
    return {"schema_version": 1, "checked_at": time.time(), "components": rows,
            "renderer": renderer, "render_ready": bool(renderer["wired"]), "verification_report": report,
            "providers": {"advertised": advertised, "error": provider_error,
                          "availability_is_load_proof": False},
            "note": "Resolved files, session initialization and actual forwards are separate evidence. Saved forwards are not resident sessions or production render readiness."}
