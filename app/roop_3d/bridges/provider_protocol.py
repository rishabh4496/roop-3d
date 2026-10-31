"""Typed remote boundary with per-job signed consent and safe diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Literal, Protocol
from urllib.parse import urlsplit

from pydantic import Field, field_validator
from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract


def safe_filename(value: str) -> str:
    reserved = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if (not value or len(value) > 255 or value in {".", ".."}
            or any(c in value for c in '/\\:%?*"<>|') or value.endswith((" ", "."))
            or any(ord(c) < 32 for c in value) or value.split(".")[0].casefold() in reserved):
        raise ValueError("A portable basename is required")
    return value


class MediaDescriptor(Contract):
    filename: str
    size_bytes: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    _filename = field_validator("filename")(safe_filename)


class RemoteJob(Contract):
    job_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,96}$")
    provider_id: str = Field(pattern=r"^[a-z0-9_-]{1,64}$")
    destination_host: str = Field(min_length=1, max_length=253)
    files: tuple[MediaDescriptor, ...] = Field(min_length=1, max_length=256)
    consent_token: str | None = Field(default=None, repr=False, max_length=16384)

    @field_validator("destination_host")
    @classmethod
    def host(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", value) or ".." in value:
            raise ValueError("Use an exact lowercase destination hostname")
        return value

    def consent_payload(self) -> dict:
        if len({item.filename for item in self.files}) != len(self.files):
            raise DomainError("CONSENT_MANIFEST_INVALID", "Consent file names must be unique", 422)
        return {"job_id": self.job_id, "provider_id": self.provider_id,
                "destination_host": self.destination_host, "files": [item.model_dump() for item in self.files]}


@dataclass(frozen=True)
class ProviderCapabilities:
    state: Literal["NOT_CONFIGURED", "VERIFIED_REMOTE", "UNSUPPORTED"]
    provider_id: str
    destination_host: str | None = None


@dataclass(frozen=True)
class ProviderStatus:
    job_id: str
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]


class ProviderProtocol(Protocol):
    def capabilities(self) -> ProviderCapabilities: ...
    def validate_consent(self, job: RemoteJob) -> None: ...
    async def submit(self, job: RemoteJob) -> ProviderStatus: ...
    async def poll(self, job: RemoteJob) -> ProviderStatus: ...
    async def cancel(self, job: RemoteJob) -> ProviderStatus: ...
    async def fetch_output(self, job: RemoteJob) -> bytes: ...


class ConsentAuthority:
    """Only a trusted user-consent handler issues tokens. Keys never enter settings."""
    def __init__(self, secret: bytes | None = None, *, clock=time.time):
        self._secret = secret if secret is not None else os.environ.get("ROOP3D_CONSENT_SECRET", "").encode()
        if len(self._secret) < 32:
            raise DomainError("CONSENT_KEY_UNAVAILABLE", "A consent signing key of at least 32 bytes is required", 503)
        self._clock = clock

    @staticmethod
    def _encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    def issue(self, job: RemoteJob, *, acknowledged: bool, lifetime_seconds: int = 300) -> str:
        if acknowledged is not True or type(lifetime_seconds) is not int or not 1 <= lifetime_seconds <= 900:
            raise DomainError("CONSENT_REQUIRED", "Explicit acknowledgement and a bounded lifetime are required", 403)
        now = int(self._clock())
        payload = {"version": 1, **job.consent_payload(), "issued_at": now, "expires_at": now + lifetime_seconds}
        body = self._encode(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        return body + "." + self._encode(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())

    def validate(self, job: RemoteJob, *, remote_processing_enabled: bool) -> None:
        if remote_processing_enabled is not True or not job.consent_token:
            raise DomainError("REMOTE_CONSENT_REQUIRED", "Remote processing and signed per-job consent are required", 403)
        try:
            body, signature = job.consent_token.split(".")
            expected = self._encode(hmac.new(self._secret, body.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(signature, expected):
                raise ValueError()
            payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            issued, expiry = payload.pop("issued_at"), payload.pop("expires_at")
            if (type(issued) is not int or type(expiry) is not int
                    or not issued <= self._clock() < expiry or not 0 < expiry - issued <= 900
                    or payload != {"version": 1, **job.consent_payload()}):
                raise ValueError()
        except (ValueError, KeyError, TypeError, UnicodeError, AttributeError):
            raise DomainError("REMOTE_CONSENT_INVALID", "Consent is invalid, expired, or does not match the upload", 403) from None


class ConsentGatedProvider:
    """Every future provider must be exposed through this I/O gate."""
    def __init__(self, adapter: ProviderProtocol, authority: ConsentAuthority, *, remote_processing_enabled: bool = False):
        self._adapter, self._authority = adapter, authority
        self.remote_processing_enabled = remote_processing_enabled

    def capabilities(self) -> ProviderCapabilities:
        return self._adapter.capabilities()

    def validate_consent(self, job: RemoteJob) -> None:
        self._authority.validate(job, remote_processing_enabled=self.remote_processing_enabled)
        cap = self.capabilities()
        if cap.state == "NOT_CONFIGURED":
            raise DomainError("NOT_CONFIGURED", "No remote provider is configured", 503)
        if cap.state != "VERIFIED_REMOTE" or cap.provider_id != job.provider_id or cap.destination_host != job.destination_host:
            raise DomainError("CONSENT_DESTINATION_MISMATCH", "Consent does not match a verified provider destination", 403)
        self._adapter.validate_consent(job)

    async def submit(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)
        return await self._adapter.submit(job)

    async def poll(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)
        return await self._adapter.poll(job)

    async def cancel(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)
        return await self._adapter.cancel(job)

    async def fetch_output(self, job: RemoteJob) -> bytes:
        self.validate_consent(job)
        return await self._adapter.fetch_output(job)


def redact_secrets(value: str) -> str:
    value = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", value)
    value = re.sub(r"(?i)\b(?:consent_token|access_token|api_key|signature|token)[\"']?\s*[=:]\s*[\"']?[^\s,;\"']+", "secret=[REDACTED]", value)
    value = re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)?", "[REDACTED]", value)
    def url(match):
        try:
            parsed = urlsplit(match.group())
            return f"{parsed.scheme}://{parsed.hostname}/[REDACTED_URL]"
        except ValueError:
            return "[REDACTED_URL]"
    return re.sub(r"https?://[^\s\"'<>]+", url, value)


class SecretRedactionFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg, record.args = redact_secrets(record.getMessage()), ()
        if record.exc_info:
            record.msg += " [exception details redacted]"
            record.exc_info, record.exc_text = None, None
        record.stack_info = None
        return True
