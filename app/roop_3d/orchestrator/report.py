"""Explainable decision report: structured JSON with a deterministic digest, plus a rendered text view.

The report carries no timestamps and no live measurements, so identical inputs yield byte-identical output.
"""
from __future__ import annotations

from typing import Any

from roop_3d.orchestrator.compile import compile_plan
from roop_3d.orchestrator.plan import Admission
from roop_3d.orchestrator.registry import CapabilityRegistry
from roop_3d.orchestrator.selector import Decision, Selection, _reason_dicts
from roop_3d.orchestrator.schema import ReasonCode, digest_of

REPORT_SCHEMA = "roop3d.orchestrator.report.v1"


def _admission(admission: Admission | None) -> dict[str, Any] | None:
    if admission is None:
        return None
    return {"admitted": admission.admitted, "peak_vram_mib": admission.peak_vram_mib, "peak_source": admission.peak_source,
            "budget_mib": admission.budget_mib, "reserve_mib": admission.reserve_mib, "schedule": admission.schedule,
            "residency": [{"engine_id": e, "residency": r} for e, r in admission.residency],
            "reasons": [{"code": c.value, "detail": d} for c, d in admission.reasons], "notes": list(admission.notes)}


def _selected(registry: CapabilityRegistry, decision: Decision) -> dict[str, Any] | None:
    if decision.plan is None:
        return None
    reason = decision.reason_code
    selected: dict[str, Any] = {}
    for item in decision.plan.engines:
        engine = registry.engine(item.engine_id)
        selected[item.category] = {"engine_id": item.engine_id, "provider": item.provider, "precision": item.precision,
                                   "reason_code": reason, "fallback": engine.fallback.model_dump(mode="json"),
                                   "peak_vram_source": ("declared" if engine.declared_peak_vram_mib is not None else
                                                        "none" if engine.residency == "none" else "unknown")}
    return selected


def _decision(registry: CapabilityRegistry, decision: Decision) -> dict[str, Any]:
    evidence = decision.evidence
    compiled = compile_plan(decision.plan) if decision.plan else None
    return {"scene": decision.segment.as_dict(), "status": "FEASIBLE" if decision.feasible else "INFEASIBLE",
            "basis": decision.basis, "reason_code": decision.reason_code,
            "plan_digest": decision.plan.digest if decision.plan else None,
            "selected": _selected(registry, decision),
            "score": decision.score, "score_note": decision.score_note,
            "evidence": None if evidence is None else {
                "status": evidence.status, "code": evidence.code.value if evidence.code else None,
                "detail": evidence.detail,
                "evidence_id": evidence.record.evidence_id if evidence.record else None},
            "admission": _admission(decision.admission),
            "runner": None if compiled is None else compiled.as_dict(),
            "problems": _reason_dicts(decision.problems),
            "rejected": decision.rejected, "notes": list(decision.notes)}


def build_report(registry: CapabilityRegistry, selection: Selection) -> dict[str, Any]:
    request = selection.request
    body = {"schema": REPORT_SCHEMA, "status": selection.status, "mode": request.mode, "tier": request.tier,
            "profile_id": request.profile_id, "snapshot": selection.snapshot,
            "candidates_considered": selection.candidates_considered,
            "engine_eligibility": {eid: _reason_dicts(reasons) for eid, reasons in sorted(selection.eligibility.items())},
            "decisions": [_decision(registry, d) for d in selection.decisions],
            "reason_vocabulary": sorted(code.value for code in ReasonCode)}
    return {**body, "report_digest": digest_of(body)}


def render_text(report: dict[str, Any]) -> str:
    lines = [f"Orchestrator report {report['report_digest'][:12]}  mode={report['mode']} tier={report['tier']} "
             f"profile={report['profile_id']}  status={report['status']}",
             f"snapshot {report['snapshot']['snapshot_digest'][:12]}"]
    for item in report["decisions"]:
        scene = item["scene"]
        lines.append(f"\nScene {scene['scene_id']}  tracks={','.join(scene['tracks']) or '-'}  traits={','.join(scene['traits'])}"
                     f"  -> {item['status']} ({item['basis']}, {item['reason_code']})")
        for category, choice in (item["selected"] or {}).items():
            lines.append(f"  + {category}: {choice['engine_id']} [{choice['provider']}/{choice['precision']}]")
        for problem in item["problems"]:
            lines.append(f"  ! {problem['code']}: {problem['detail']}")
        for row in item["rejected"]:
            lines.append(f"  - {row['category']}: {row['engine_id']}  " +
                         "; ".join(f"{r['code']}({r['detail']})" for r in row["reasons"]))
        if item["runner"] and item["runner"]["unsupported"]:
            lines.append("  unsupported: " + ", ".join(item["runner"]["unsupported"]))
    return "\n".join(lines)
