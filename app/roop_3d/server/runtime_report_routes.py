"""`GET /api/v1/system/runtime-report`: the 14-section structured runtime report as JSON plus copy-ready plain text.

The same collectors feed the Pinokio terminal at boot and job start, so the terminal, this endpoint and the Settings drawer
cannot disagree. Read-only; it probes local hardware and files and never contacts the network.
"""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(prefix="/api/v1/system", tags=["diagnostics"])


@router.get("/runtime-report")
def runtime_report(request: Request) -> dict:
    from roop_3d.telemetry.runtime_report import build_runtime_report
    state = request.app.state
    return build_runtime_report(getattr(state, "root", None), state, trigger="on demand")
