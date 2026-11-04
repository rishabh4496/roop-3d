"""Roop 3D security, privacy, and synthetic media protection module."""
from __future__ import annotations

from roop_3d.security.content_validator import (
    validate_faceset_archive_bytes,
    validate_image_header,
    validate_safe_filename,
    validate_safe_path,
    validate_video_container,
)
from roop_3d.security.privacy import (
    ConsentRecord,
    ConsentVerifier,
    TargetFootagePermission,
)
from roop_3d.security.synthetic_media import (
    AuditRecord,
    IdentityHonestyPolicy,
    SafetyGuard,
    SyntheticMediaDisclosure,
    build_provenance_manifest,
)
from roop_3d.security.artifact_cleaner import SafeArtifactCleaner
from roop_3d.security.log_sanitizer import (
    SecurityRedactionFilter,
    install_log_sanitizer,
    sanitize_log_message,
)

__all__ = [
    "validate_faceset_archive_bytes",
    "validate_image_header",
    "validate_safe_filename",
    "validate_safe_path",
    "validate_video_container",
    "ConsentRecord",
    "ConsentVerifier",
    "TargetFootagePermission",
    "AuditRecord",
    "IdentityHonestyPolicy",
    "SafetyGuard",
    "SyntheticMediaDisclosure",
    "build_provenance_manifest",
    "SafeArtifactCleaner",
    "SecurityRedactionFilter",
    "install_log_sanitizer",
    "sanitize_log_message",
]
