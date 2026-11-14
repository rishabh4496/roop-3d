"""Render-engine readiness: one honest row for the true-3D full-head render path.

There is no 2D swapper to list any more. The single row reports whether the FullHeadEngine render path is wired into the
session (``roop_3d.pipeline.true3d_renderer.is_wired()``) and what is missing otherwise. Nothing here loads a graph and no
"ready" is ever fabricated: while the renderer is not wired the state is ``NONE_AVAILABLE`` with the reason
``3D renderer not wired``.
"""
from __future__ import annotations

import time
from typing import Any

RENDERER_ID = "true3d_head_renderer"


def render_engine_report(catalog: Any = None) -> list[dict[str, Any]]:
    from roop_3d.core.true3d_guard import NOT_WIRED_CODE, NOT_WIRED_MESSAGE, True3DArchitectureGuard, renderer_is_wired
    wired = renderer_is_wired()
    return [{
        "model_id": RENDERER_ID, "architecture": "full_head_perspective_zbuffer", "native_resolution": None,
        "status": "READY" if wired else "UNAVAILABLE",
        "readiness": "WIRED" if wired else NOT_WIRED_CODE, "policy": None,
        "reason": None if wired else f"{NOT_WIRED_MESSAGE}: the FullHeadEngine render path is not connected to the session",
        "blocked": False, "weight": None, "loaded": wired, "loaded_provider": None, "loaded_precision": None, "session": None,
        "requirements": True3DArchitectureGuard().describe(),
    }]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"state": "READY" if any(row["status"] == "READY" for row in rows) else "NONE_AVAILABLE",
            "ready": [row["model_id"] for row in rows if row["status"] == "READY"],
            "unavailable": [{"model_id": row["model_id"], "readiness": row["readiness"], "reason": row["reason"]}
                            for row in rows if row["status"] != "READY"],
            "checked_at": time.time()}


def render_engine_summary(catalog: Any = None, *, block: bool = False) -> dict[str, Any]:
    """``{state, ready, unavailable}`` for the health payload (cheap: no weights are hashed)."""
    return summarize(render_engine_report(catalog))


# Names kept for callers written against the former swap-engine report.
swap_engine_report = render_engine_report
swap_engine_summary = render_engine_summary
