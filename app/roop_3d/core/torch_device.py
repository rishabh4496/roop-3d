"""Lazy optional PyTorch device initialization with real tensor execution probes."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

from roop_3d.core.errors import DomainError

_LOG = logging.getLogger("roop_3d.torch")


@dataclass(frozen=True)
class TorchDevice:
    device: object
    provider: str
    fallback_errors: tuple[str, ...]


@lru_cache(maxsize=4)
def select_torch_device(preferred: str = "auto") -> TorchDevice:
    """CUDA -> torch-directml -> CPU; an explicit device never silently changes."""
    if preferred not in ("auto", "cuda", "dml", "cpu"):
        raise ValueError("PyTorch device must be auto, cuda, dml or cpu")
    try:
        import torch
    except ImportError as error:
        raise DomainError("PYTORCH_NOT_INSTALLED",
                          f"Optional PyTorch runtime is not installed in this interpreter: {error}", 503) from error
    errors = []
    for candidate in (("cuda", "dml", "cpu") if preferred == "auto" else (preferred,)):
        try:
            if candidate == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("torch.cuda.is_available() is false")
                device, provider = torch.device("cuda"), "CUDAExecutionProvider"
            elif candidate == "dml":
                import torch_directml
                device, provider = torch_directml.device(), "DmlExecutionProvider"
            else:
                device, provider = torch.device("cpu"), "CPUExecutionProvider"
            # Allocation alone is not proof: execute and synchronize an operation.
            probe = torch.ones((2, 2), dtype=torch.float32, device=device)
            value = (probe @ probe).sum().item()
            if value != 8.0:
                raise RuntimeError(f"PyTorch tensor probe returned {value}, expected 8")
            _LOG.info("PyTorch initialized: device=%s provider=%s fallback_errors=%s", device, provider, errors)
            return TorchDevice(device, provider, tuple(errors))
        except Exception as error:
            detail = f"{candidate}: {type(error).__name__}: {error}"
            errors.append(detail)
            _LOG.warning("PyTorch device probe failed: %s", detail)
    raise DomainError("PYTORCH_DEVICE_UNAVAILABLE", "; ".join(errors), 503)
