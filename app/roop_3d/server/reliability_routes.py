"""Stage 18 REST surface: local, redacted diagnostics plus the crash-consistency audit.

All endpoints sit behind the same loopback host/origin gate as the rest of the API (``IntakeMiddleware``). Nothing
here sends data anywhere: the report is returned to the caller and, when ``save`` is requested, written under
``roop_3d/.runtime/diagnostics`` on this machine.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from roop_3d.core.settings import Contract
from roop_3d.reliability.ledger import LEDGER
from roop_3d.reliability.monitor import MONITOR
from roop_3d.reliability.recovery import ReliabilityAuditor
from roop_3d.reliability.report import build_report, save_report

router = APIRouter(prefix="/api/v1/reliability", tags=["reliability"])

# While jobs run, only temp artefacts older than this are considered stale, so a live render is never disturbed.
RUNTIME_MIN_AGE_S = 600.0


class RepairRequest(Contract):
    deep: bool = False
    # The only destructive repair. Quarantines an unreadable job database and starts empty; never implicit.
    reset_job_store: bool = False


def _auditor(request: Request) -> ReliabilityAuditor:
    state = request.app.state
    worker = state.worker
    return ReliabilityAuditor(
        state.root, state.jobs, lambda: state.settings,
        active_job_ids=(lambda: {job for job, _ in list(worker._futures.values())}) if worker is not None else None,
        runner=worker.runner if worker is not None else None)


@router.get("/report")
def diagnostics_report(request: Request, series: bool = False, save: bool = False) -> dict[str, Any]:
    """The redacted, local-only diagnostic report. ``save=true`` also keeps a copy on disk."""
    audit = _auditor(request).audit(min_age_s=RUNTIME_MIN_AGE_S).as_dict()
    document = build_report(request.app.state, audit=audit, include_series=series)
    document["startup_reconciliation"] = getattr(request.app.state, "startup_reconciliation", None)
    document["orphan_sweep_at_start"] = getattr(request.app.state, "orphan_sweep", [])
    if save:
        document["saved_to"] = save_report(request.app.state.root, document)
    return document


@router.get("/resources")
def live_resources() -> dict[str, Any]:
    """Cheap live view for dashboards: latest sample, peaks, lifecycle ledger and the growth verdict."""
    return {"monitor": MONITOR.summary(), "ledger": LEDGER.snapshot(), "growth": MONITOR.growth().as_dict()}


@router.get("/audit")
def audit(request: Request, deep: bool = False) -> dict[str, Any]:
    """Read-only crash-consistency audit. ``deep=true`` also re-hashes every succeeded output."""
    return _auditor(request).audit(deep=deep, min_age_s=RUNTIME_MIN_AGE_S).as_dict()


@router.post("/repair")
def repair(request: Request, payload: RepairRequest) -> dict[str, Any]:
    """Apply every safe repair found by the audit. Active jobs are never touched."""
    return _auditor(request).repair(deep=payload.deep, min_age_s=RUNTIME_MIN_AGE_S, queue=request.app.state.queue,
                                    reset_job_store=payload.reset_job_store).as_dict()
