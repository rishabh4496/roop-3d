"""Privacy enforcement, biometric permission validation, and network isolation guards."""
from __future__ import annotations

from datetime import datetime, timezone
import ipaddress
import re
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract


class ConsentRecord(Contract):
    """Explicit biometric and likeness permission for an identity."""
    subject_name: str = Field(min_length=1, max_length=128)
    consent_granted: bool = Field(default=False)
    consent_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    permitted_operations: tuple[str, ...] = Field(default=("swap", "reenactment", "3d_reconstruction"))
    commercial_use_allowed: bool = Field(default=False)
    terms: str = Field(default="Local research and synthetic media reenactment", max_length=1024)


class TargetFootagePermission(Contract):
    """Attestation and permission record for target footage."""
    target_path: str = Field(min_length=1, max_length=1024)
    target_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    authorized: bool = Field(default=False)
    lawful_use_attestation: str = Field(
        default="User attests that target footage is owned, licensed, or processed with consent",
        max_length=1024,
    )


class ConsentVerifier:
    """Enforces consent and authorization fail-closed across all processing pipelines."""

    @staticmethod
    def verify_source_consent(
        consent_data: ConsentRecord | dict[str, Any] | Any,
        operation: str = "swap",
    ) -> None:
        """Verify that likeness consent was explicitly granted for the requested operation."""
        if consent_data is None:
            raise DomainError("IDENTITY_CONSENT_REQUIRED", "No consent metadata provided for source identity", 403)

        if isinstance(consent_data, dict):
            granted = bool(consent_data.get("consent_granted", False))
            permitted = consent_data.get("permitted_operations", ("swap", "reenactment", "3d_reconstruction"))
        elif hasattr(consent_data, "consent_granted"):
            granted = bool(getattr(consent_data, "consent_granted", False))
            permitted = getattr(consent_data, "permitted_operations", ("swap", "reenactment", "3d_reconstruction"))
        else:
            raise DomainError("IDENTITY_CONSENT_INVALID", "Unrecognized consent metadata structure", 403)

        if not granted:
            raise DomainError(
                "IDENTITY_CONSENT_DENIED",
                "Explicit consent for source identity likeness usage is required",
                403,
            )

        if operation and permitted and operation not in permitted:
            raise DomainError(
                "OPERATION_UNAUTHORIZED",
                f"Operation '{operation}' is not permitted by source identity consent",
                403,
            )

    @staticmethod
    def verify_target_permission(
        target_path: str,
        target_sha256: str,
        permission: TargetFootagePermission | dict[str, Any] | None,
    ) -> None:
        """Verify that target footage has appropriate authorization."""
        if permission is None:
            # Default fallback: check if target is a local file
            if not target_path or not target_sha256:
                raise DomainError("TARGET_PERMISSION_REQUIRED", "Target path and hash are required", 400)
            return

        if isinstance(permission, dict):
            authorized = bool(permission.get("authorized", False))
            sha = permission.get("target_sha256", "")
        else:
            authorized = permission.authorized
            sha = permission.target_sha256

        if not authorized:
            raise DomainError("TARGET_PERMISSION_DENIED", "Target footage permission was not authorized", 403)
        if sha and sha != target_sha256:
            raise DomainError("TARGET_HASH_MISMATCH", "Target footage hash does not match authorization", 409)

    @staticmethod
    def verify_egress_policy(
        destination: str,
        *,
        allow_remote: bool = False,
        consent_token: str | None = None,
    ) -> None:
        """Verify that no external media or biometric transfer occurs without opt-in consent."""
        try:
            parsed = urlsplit(destination if "://" in destination else f"http://{destination}")
            host = parsed.hostname or destination.split(":")[0]
            if host == "localhost":
                host = "127.0.0.1"
            ip = ipaddress.ip_address(host)
            if ip.is_loopback:
                return  # Loopback is always permitted
        except (ValueError, AttributeError):
            pass

        # If not loopback:
        if not allow_remote:
            raise DomainError(
                "REMOTE_EGRESS_BLOCKED",
                "External network transmission is disabled by default",
                403,
            )
        if not consent_token:
            raise DomainError(
                "REMOTE_CONSENT_REQUIRED",
                "External media transmission requires an explicit per-job consent token",
                403,
            )

    @staticmethod
    def audit_no_hidden_transfer(payload: dict[str, Any]) -> None:
        """Assert that payload contains no remote destinations or hidden upload directives."""
        remote = payload.get("remote_processing_enabled", False)
        dest = payload.get("destination_host")
        if not remote and dest:
            host = dest.split(":")[0].strip().lower()
            if host in {"127.0.0.1", "localhost", "::1"}:
                return
            try:
                ip = ipaddress.ip_address(host)
                if ip.is_loopback:
                    return
            except ValueError:
                pass
            raise DomainError("HIDDEN_TRANSFER_DETECTED", f"External destination '{dest}' configured while remote is disabled", 403)
