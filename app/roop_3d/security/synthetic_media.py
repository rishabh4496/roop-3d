"""Synthetic media disclosure, cryptographic provenance, identity honesty, and safety safeguards."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from pydantic import Field

from roop_3d import __version__
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract

NCII_PROHIBITED_TERMS: frozenset[str] = frozenset({
    "ncii", "non-consensual", "deepnude", "undress", "revenge_porn",
    "intimate_leak", "deceptive_leak", "impersonate_defame", "fraudulent_id",
})


class SyntheticMediaDisclosure(Contract):
    """Visible and auditable disclosure accompanying synthetic media artifacts."""
    synthetic_media: Literal[True] = True
    processor: str = "Roop 3D"
    processor_version: str = __version__
    processing: Literal["local"] = "local"
    notice: str = "Synthetic media processed locally by Roop 3D"
    digital_source_type: str = "http://cv.iptc.org/newscodes/digitalsourcetype/compositeWithTrainedAlgorithmicMedia"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AuditRecord(Contract):
    """Immutable audit trail of render settings, models, and inputs."""
    job_id: str
    software_version: str = __version__
    hardware_profile: str
    engine_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_digests: dict[str, str] = Field(default_factory=dict)
    render_settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class IdentityHonestyPolicy:
    """Enforces truthful claims: SHA-256 establishes byte integrity, not person identity."""

    DISALLOWED_CLAIM_PATTERNS = (
        r"(?i)cryptographic[_\s]+proof[_\s]+of[_\s]+(person|identity|human)",
        r"(?i)sha-?256[_\s]+proves[_\s]+(identity|person|human)",
        r"(?i)hash[_\s]+proves[_\s]+(identity|individual)",
        r"(?i)biometric[_\s]+proof[_\s]+via[_\s]+hash",
    )

    @classmethod
    def audit_claim(cls, claim: dict[str, Any] | str) -> None:
        """Reject any claim asserting that an ordinary SHA-256 hash proves a person's identity."""
        text = json.dumps(claim) if isinstance(claim, dict) else str(claim)
        for pattern in cls.DISALLOWED_CLAIM_PATTERNS:
            if re.search(pattern, text):
                raise DomainError(
                    "DISHONEST_IDENTITY_CLAIM",
                    "Ordinary SHA-256 hashes establish file byte integrity only, never proof of a person's identity",
                    400,
                )

        if isinstance(claim, dict):
            if claim.get("identity_proof_type") == "sha256" or claim.get("proves_person_identity") is True:
                raise DomainError(
                    "DISHONEST_IDENTITY_CLAIM",
                    "A hash digest cannot be claimed as person identity proof",
                    400,
                )

    @classmethod
    def honest_integrity_statement(cls, sha256_digest: str) -> dict[str, str]:
        """Return the standard, truthful integrity claim for an artifact."""
        return {
            "sha256": sha256_digest,
            "verification_type": "artifact_byte_integrity",
            "limitation": "Proves file byte identity against local registry or input; does not establish legal or biometric person identity.",
        }


class SafetyGuard:
    """Safeguards against Non-Consensual Intimate Imagery (NCII) and deceptive impersonation."""

    @classmethod
    def inspect_request(
        cls,
        *,
        request_metadata: dict[str, Any] | None = None,
        face_labels: tuple[str, ...] | list[str] = (),
        notes: str = "",
    ) -> None:
        """Scan inputs for deceptive impersonation flags and NCII indicators."""
        corpus = [notes]
        if request_metadata:
            corpus.append(json.dumps(request_metadata))
        corpus.extend(face_labels)

        combined = " ".join(corpus).lower()
        for term in NCII_PROHIBITED_TERMS:
            if term in combined:
                raise DomainError(
                    "SAFETY_POLICY_VIOLATION",
                    f"Request contains prohibited term/intent '{term}': NCII and deceptive impersonation are blocked",
                    403,
                )


def build_provenance_manifest(
    *,
    input_sha256: str,
    output_sha256: str,
    output_path: str,
    job_id: str,
    engine_id: str,
    hardware_profile: str,
    model_digests: dict[str, str],
    render_settings: dict[str, Any],
    c2pa_result: dict[str, Any] | None = None,
    authentication_mode: str = "none",
) -> dict[str, Any]:
    """Assemble a standard, tamper-evident synthetic media provenance manifest."""
    disclosure = SyntheticMediaDisclosure().model_dump()
    audit = AuditRecord(
        job_id=job_id,
        hardware_profile=hardware_profile,
        engine_id=engine_id,
        input_sha256=input_sha256,
        output_sha256=output_sha256,
        model_digests=model_digests,
        render_settings=render_settings,
    ).model_dump()

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "input_sha256": input_sha256,
        "output_sha256": output_sha256,
        "output_path": output_path,
        "disclosure": disclosure,
        "audit": audit,
        "integrity_notice": IdentityHonestyPolicy.honest_integrity_statement(output_sha256),
    }

    if c2pa_result:
        manifest["c2pa"] = c2pa_result

    if authentication_mode == "hmac-sha256" and os.environ.get("ROOP3D_PROVENANCE_KEY"):
        key = os.environ["ROOP3D_PROVENANCE_KEY"].encode()
        body = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        manifest["authentication"] = {
            "algorithm": "hmac-sha256",
            "signature": hmac.new(key, body, hashlib.sha256).hexdigest(),
        }

    return manifest
