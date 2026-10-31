"""Disabled remote adapter. Contains no URLs, credentials, HTTP client, or uploads."""
from roop_3d.bridges.provider_protocol import ProviderCapabilities, ProviderStatus, RemoteJob
from roop_3d.core.errors import DomainError


class CloudStub:
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities("NOT_CONFIGURED", "cloud_stub")

    def validate_consent(self, job: RemoteJob) -> None:
        raise DomainError("NOT_CONFIGURED", "No remote provider is configured", 503)

    async def submit(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)

    async def poll(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)

    async def cancel(self, job: RemoteJob) -> ProviderStatus:
        self.validate_consent(job)

    async def fetch_output(self, job: RemoteJob) -> bytes:
        self.validate_consent(job)
