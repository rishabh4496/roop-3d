"""Evidence store and the exact-match rule that decides whether a measurement applies to a plan.

Only ``measured_with == "real_models"`` records count. The Stage 11 simulated harness records are
rejected: its profiles apply pixel filters and ``time.sleep`` and never load a Model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydantic import ValidationError

from roop_3d.core.errors import DomainError
from roop_3d.orchestrator.plan import Plan
from roop_3d.orchestrator.schema import EvidenceFile, EvidenceRecord, ReasonCode

EVIDENCE_RELATIVE = "roop_3d/.runtime/orchestrator/evidence.json"


@dataclass(frozen=True)
class EvidenceMatch:
    status: str                      # valid | none | stale | not_real | veto
    record: EvidenceRecord | None
    code: ReasonCode | None
    detail: str

    @property
    def usable(self) -> bool:
        return self.status == "valid"


class EvidenceStore:
    def __init__(self, records: Iterable[EvidenceRecord] = ()):
        self.records = tuple(sorted(records, key=lambda r: (r.recorded_at, r.evidence_id)))

    @classmethod
    def load(cls, path: Path | None = None) -> "EvidenceStore":
        if path is None:
            from roop_3d.audit.project_paths import application_root
            path = application_root() / EVIDENCE_RELATIVE
        if not path.is_file():
            return cls()
        try:
            return cls(EvidenceFile.model_validate_json(path.read_text("utf-8")).records)
        except (OSError, ValidationError):
            raise DomainError("ORCHESTRATOR_EVIDENCE_INVALID", "Orchestrator evidence file is invalid", 503) from None

    def match(self, plan: Plan) -> EvidenceMatch:
        same_tier = [r for r in self.records if r.tier == plan.tier]
        exact = [r for r in same_tier if r.plan_digest == plan.digest]
        if not exact:
            ids = plan.engine_ids()
            related = [r for r in same_tier if {e.engine_id for e in r.engines} == ids]
            if related:
                return EvidenceMatch("stale", related[-1], ReasonCode.EVIDENCE_STALE,
                                     f"{related[-1].evidence_id} was measured against a different Engine entry, Model hash, "
                                     "provider or precision")
            return EvidenceMatch("none", None, ReasonCode.NO_EVIDENCE_FOR_PROFILE,
                                 f"no real measurement of this plan on {plan.tier}")
        real = [r for r in exact if r.measured_with == "real_models"]
        if not real:
            return EvidenceMatch("not_real", exact[-1], ReasonCode.EVIDENCE_NOT_REAL,
                                 f"{exact[-1].evidence_id} was not measured with real models")
        record = real[-1]
        expected = {(e.engine_id, e.entry_digest, e.model_sha256s, e.precision, e.provider) for e in plan.engines}
        got = {(e.engine_id, e.entry_digest, tuple(sorted(e.model_sha256s)), e.precision, e.provider) for e in record.engines}
        if expected != got:
            return EvidenceMatch("stale", record, ReasonCode.EVIDENCE_STALE, f"{record.evidence_id} does not match the plan")
        if not record.safety_vetoes_passed:
            return EvidenceMatch("veto", record, ReasonCode.SAFETY_VETO, f"{record.evidence_id} failed a safety veto")
        return EvidenceMatch("valid", record, None, record.evidence_id)
