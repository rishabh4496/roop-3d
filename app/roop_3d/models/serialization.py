"""Restricted model deserialization helpers. No pickle fallback is permitted."""
from __future__ import annotations

import inspect
from typing import Any, BinaryIO

from roop_3d.core.errors import DomainError


def load_restricted_torch_weights(stream: BinaryIO) -> Any:
    """Load tensor-only checkpoint data with PyTorch's restricted weights loader.

    The caller must supply a stream opened through the verified model provider.
    Older PyTorch releases without ``weights_only`` fail closed.
    """
    try:
        import torch
    except ImportError:
        raise DomainError("PYTORCH_UNAVAILABLE", "PyTorch is not installed for this optional checkpoint", 503) from None
    try:
        supports_restricted_loading = "weights_only" in inspect.signature(torch.load).parameters
    except (TypeError, ValueError):
        supports_restricted_loading = False
    if not supports_restricted_loading:
        raise DomainError("UNSAFE_CHECKPOINT_LOADER", "This PyTorch version lacks restricted weights_only loading", 409)
    try:
        return torch.load(stream, map_location="cpu", weights_only=True)
    except Exception:
        raise DomainError("CHECKPOINT_REJECTED", "Restricted PyTorch checkpoint loading failed", 422) from None
