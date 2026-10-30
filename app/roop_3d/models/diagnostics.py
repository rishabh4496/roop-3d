"""Small public provider diagnostics; media and golden tensors are never included."""
from typing import Literal

from pydantic import Field

from roop_3d.core.settings import Contract

Provider = Literal["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]


class ProviderFailure(Contract):
    provider: Provider
    precision: Literal["fp32", "fp16"]
    reason: str = Field(min_length=1, max_length=4096)


class ModelDiagnostic(Contract):
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    golden_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: Provider
    requested_provider: Provider
    precision: Literal["fp32", "fp16"]
    executions: int = Field(ge=0)
    node_providers: dict[Provider, int] = Field(max_length=3)
    fallback_history: tuple[ProviderFailure, ...] = Field(default=(), max_length=8)


class JobDiagnostics(Contract):
    models: dict[str, ModelDiagnostic] = Field(default_factory=dict, max_length=32)
