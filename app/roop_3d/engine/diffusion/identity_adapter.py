"""Family-locked identity adapter declarations; no model is loaded at import time."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from roop_3d.core.errors import DomainError

ModelFamily = Literal["sdxl", "flux"]


@dataclass(frozen=True)
class IdentityAdapterSpec:
    model_id: str
    family: ModelFamily
    embedding_dimensions: int | None


# These identifiers encode the family explicitly. The generic manifest entry
# ``pulid`` deliberately remains ambiguous and cannot be loaded by either path.
_ADAPTERS = {
    "instantid": IdentityAdapterSpec("instantid", "sdxl", 512),
    "ip-adapter-faceid-sdxl": IdentityAdapterSpec("ip-adapter-faceid-sdxl", "sdxl", 512),
    "pulid-sdxl": IdentityAdapterSpec("pulid-sdxl", "sdxl", 512),
    "pulid-flux": IdentityAdapterSpec("pulid-flux", "flux", None),
}


def adapter_spec(adapter_id: str) -> IdentityAdapterSpec:
    try:
        return _ADAPTERS[adapter_id]
    except KeyError:
        if adapter_id == "pulid":
            raise DomainError("DIFFUSION_ADAPTER_FAMILY_UNKNOWN",
                              "PuLID must be registered as pulid-sdxl or pulid-flux", 422) from None
        raise DomainError("DIFFUSION_ADAPTER_UNKNOWN", "Identity adapter is not in the family contract", 422) from None


def validate_adapter(family: ModelFamily, adapter_id: str,
                     embedding: np.ndarray | None = None) -> IdentityAdapterSpec:
    """Reject family mixing before touching optional libraries or model files."""
    spec = adapter_spec(adapter_id)
    if spec.family != family:
        raise DomainError("DIFFUSION_FAMILY_MISMATCH",
                          f"Adapter {adapter_id} belongs to {spec.family}, not {family}", 422)
    if embedding is not None:
        if (not isinstance(embedding, np.ndarray) or embedding.ndim != 1
                or spec.embedding_dimensions is not None and embedding.shape != (spec.embedding_dimensions,)
                or not np.issubdtype(embedding.dtype, np.number) or not np.isfinite(embedding).all()):
            expected = (f"a finite {spec.embedding_dimensions}-D embedding" if spec.embedding_dimensions
                        else "a finite one-dimensional embedding")
            raise DomainError("DIFFUSION_IDENTITY_EMBEDDING_INVALID", f"Expected {expected}", 422)
    return spec
