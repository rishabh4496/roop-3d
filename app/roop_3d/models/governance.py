"""Fail-closed compatibility and lifecycle assessment for registered artifacts."""
from __future__ import annotations

from importlib import metadata
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version, InvalidVersion

from roop_3d.models.registry import ModelRecord

LIFECYCLE_STATES = frozenset({"NOT_INSTALLED", "DOWNLOADING", "VERIFYING", "LICENSE_RESTRICTED",
                              "READY", "INCOMPATIBLE", "QUARANTINED", "FAILED"})


class ModelCompatibilityScanner:
    """Evaluate declared compatibility without loading model code or weights."""

    def scan(self, record: ModelRecord, *, installed: bool | None = None,
             integrity_verified: bool | None = None, hardware: dict[str, Any] | None = None,
             precision: str | None = None) -> dict[str, Any]:
        issues: list[str] = []
        compat = record.dependency_compatibility
        state = str(compat.get("state", "UNKNOWN")).upper() if isinstance(compat, dict) else "UNKNOWN"
        if state == "INCOMPATIBLE":
            issues.append("Declared dependency compatibility is incompatible")
        elif state != "COMPATIBLE":
            issues.append("Dependency compatibility is unknown")
        for requirement in compat.get("requirements", ()) if isinstance(compat, dict) else ():
            if not isinstance(requirement, dict) or not requirement.get("name") or not requirement.get("specifier"):
                issues.append("Malformed dependency requirement")
                continue
            try:
                installed_version = metadata.version(str(requirement["name"]))
                if Version(installed_version) not in SpecifierSet(str(requirement["specifier"])):
                    issues.append(f"Dependency version mismatch: {requirement['name']}")
            except (metadata.PackageNotFoundError, InvalidVersion, ValueError):
                issues.append(f"Dependency missing or version unknown: {requirement['name']}")

        required_vram = record.minimum_vram_gib
        if required_vram is None:
            hardware_status = str(record.hardware_requirements.get("state", "UNKNOWN")).upper()
            if hardware_status != "VERIFIED":
                issues.append("Hardware requirements are unknown")
        elif hardware and hardware.get("vram_total_gb") is not None:
            if float(hardware["vram_total_gb"]) < required_vram:
                issues.append("Available GPU memory is below the registered minimum")
        elif not hardware:
            issues.append("Runtime hardware has not been scanned")
        if hardware:
            profiles = record.hardware_requirements.get("profiles")
            profile = hardware.get("profile")
            if profiles and profile and profile not in profiles:
                issues.append("Hardware profile is not supported by this model")
        if precision and record.precision_requirements and precision.upper() not in {
                item.upper() for item in record.precision_requirements}:
            issues.append("Requested precision is not supported by this model")

        extension = record.serialization_format.upper()
        if extension in {"PICKLE", "PYTORCH", "TORCH", "PT", "PTH", "CKPT", "PKL", "PICKLE"}:
            issues.append("Pickle-based serialization is not admitted")
        if record.security_status != "VERIFIED":
            issues.append("Artifact security review is incomplete")
        if not record.precision_requirements:
            issues.append("Precision requirements are unknown")
        if any(value in {"", "UNKNOWN"} for value in
               (record.original_repository, record.source_revision, record.architecture,
                record.required_base_model, record.attribution_requirements)):
            issues.append("Required model provenance or attribution metadata is unknown")
        if record.license_id == "UNKNOWN" or not record.license_url or record.commercial_use == "unknown" or record.redistribution_permission == "unknown":
            issues.append("License and redistribution terms are not fully reviewed")
        base_license_incomplete = (record.required_base_model not in {"NONE", "UNKNOWN"} and
            (record.base_model_license_id == "UNKNOWN" or not record.base_model_license_url or
             record.base_model_commercial_use == "unknown" or
             record.base_model_redistribution_permission == "unknown" or
             record.base_model_attribution_requirements == "UNKNOWN"))
        if base_license_incomplete:
            issues.append("Required base model license terms are not fully reviewed")

        if installed is False:
            lifecycle = "NOT_INSTALLED"
        elif installed is None:
            lifecycle = "NOT_INSTALLED" if record.status == "verified_remote" else "FAILED"
        elif not record.expected_sha256 or not record.byte_size or integrity_verified is False:
            lifecycle = "QUARANTINED" if integrity_verified is False else "FAILED"
        elif (record.status == "blocked" or record.license_id == "UNKNOWN" or not record.license_url
              or record.commercial_use == "unknown" or record.redistribution_permission == "unknown"
              or base_license_incomplete):
            lifecycle = "LICENSE_RESTRICTED"
        elif any(word in issue.lower() for issue in issues for word in ("incompatible", "mismatch", "below", "not admitted", "not supported")):
            lifecycle = "INCOMPATIBLE"
        elif issues:
            lifecycle = "INCOMPATIBLE"
        else:
            lifecycle = "READY"
        return {"lifecycle_state": lifecycle, "compatibility": "COMPATIBLE" if not issues else "UNKNOWN_OR_INCOMPATIBLE",
                "issues": issues, "integrity_verified": integrity_verified,
                "license_status": "REVIEWED" if record.license_id != "UNKNOWN" else "UNKNOWN",
                "source_revision": record.source_revision,
                "security_status": record.security_status}
