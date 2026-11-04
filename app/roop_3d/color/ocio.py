"""Optional OpenColorIO adapter. Explicit opt-in: nothing in the default render path imports this module.

Why optional: the native colour maths in this package (checked against published constants) covers every
conversion the pipeline performs, so OCIO is not a dependency. It is useful as (a) an independent
reference that the native maths is cross-validated against, and (b) an explicit transform for users who
want an ACES-style or studio-config conversion in finishing.

Why only built-in configs: an OCIO config *file* can reference LUT files by path. Only ``ocio://`` URIs
(configs compiled into the library) are accepted, so no file chosen by a request is ever opened.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from roop_3d.core.errors import DomainError

BUILTIN_CONFIGS = ("ocio://default", "ocio://cg-config-latest", "ocio://studio-config-latest")


def _import():
    try:
        import PyOpenColorIO as ocio
    except ImportError:
        return None
    return ocio


def is_available() -> bool:
    return _import() is not None


def capability() -> dict:
    ocio = _import()
    if ocio is None:
        return {"state": "UNSUPPORTED", "default_path": False,
                "reason": "PyOpenColorIO is not installed; it is optional and the default pipeline does not use it",
                "install": "pip install opencolorio (built-in ocio:// configs only; no config files are ever opened)"}
    return {"state": "AVAILABLE", "version": ocio.__version__, "default_path": False, "configs": list(BUILTIN_CONFIGS),
            "note": "explicit opt-in; native colour maths remains the default and is cross-validated against it"}


@dataclass(frozen=True)
class OcioTransform:
    """A source -> destination colour-space conversion in a built-in OCIO config."""
    source: str
    destination: str
    config_uri: str = "ocio://studio-config-latest"

    def __post_init__(self) -> None:
        if self.config_uri not in BUILTIN_CONFIGS:
            raise DomainError("OCIO_CONFIG_DENIED", f"Only built-in OCIO configs are accepted: {BUILTIN_CONFIGS}", 422)

    def _processor(self):
        ocio = _import()
        if ocio is None:
            raise DomainError("OCIO_UNAVAILABLE", "PyOpenColorIO is not installed", 503)
        config = ocio.Config.CreateFromFile(self.config_uri)
        try:
            return config, config.getProcessor(self.source, self.destination).getDefaultCPUProcessor()
        except Exception as error:  # OCIO raises its own Exception type
            raise DomainError("OCIO_TRANSFORM_INVALID", f"{self.source} -> {self.destination}: {error}", 422) from None

    def fingerprint(self) -> dict:
        config, processor = self._processor()
        return {"config": self.config_uri, "ocio_version": _import().__version__, "source": self.source,
                "destination": self.destination, "config_cache_id": config.getCacheID(),
                "processor_cache_id": processor.getCacheID()}

    def apply_float(self, rgb: np.ndarray) -> np.ndarray:
        """Convert an (..., 3) float RGB array; returns a new float32 array of the same shape."""
        _, processor = self._processor()
        flat = np.ascontiguousarray(np.asarray(rgb, np.float32).reshape(-1, 3).copy())
        processor.applyRGB(flat)
        return flat.reshape(np.shape(rgb))

    def apply_bgr8(self, bgr: np.ndarray) -> np.ndarray:
        """Convert uint8 BGR (as the pipeline carries it) through the transform; result is clipped and rounded."""
        rgb = bgr[..., ::-1].astype(np.float32) / 255.0
        converted = self.apply_float(rgb)
        return np.ascontiguousarray(np.rint(np.clip(converted, 0.0, 1.0)[..., ::-1] * 255.0).astype(np.uint8))


def cross_validate() -> dict:
    """Maximum absolute difference between the native maths and OCIO's, per conversion (needs OCIO)."""
    from roop_3d.color import gamut, transfer
    if not is_available():
        raise DomainError("OCIO_UNAVAILABLE", "PyOpenColorIO is not installed", 503)
    ramp = np.repeat(np.linspace(0, 1, 256, dtype=np.float32)[:, None], 3, axis=1)
    basis = np.eye(3, dtype=np.float32)
    nits = np.array([0.1, 1, 10, 50, 100, 203, 500, 1000], np.float32)
    display = np.repeat((nits / 100.0)[:, None], 3, axis=1)  # display spaces: 1.0 == 100 cd/m2

    def diff(mine, theirs) -> float:
        return float(np.abs(np.asarray(mine, np.float64) - np.asarray(theirs, np.float64)).max())

    lin709 = "Linear Rec.709 (sRGB)"
    return {
        "srgb_eotf": diff(transfer.srgb_eotf(ramp[:, 0]), OcioTransform("sRGB Encoded Rec.709 (sRGB)", lin709).apply_float(ramp)[:, 0]),
        "bt1886_gamma24": diff(transfer.to_linear("bt1886", ramp[:, 0]), OcioTransform("Gamma 2.4 Encoded Rec.709", lin709).apply_float(ramp)[:, 0]),
        "gamma22": diff(transfer.to_linear("gamma22", ramp[:, 0]), OcioTransform("Gamma 2.2 Encoded Rec.709", lin709).apply_float(ramp)[:, 0]),
        "bt2020_to_bt709_matrix": diff(gamut.conversion_matrix(9, 1), OcioTransform("Linear Rec.2020", lin709).apply_float(basis).T),
        "bt709_to_bt2020_matrix": diff(gamut.conversion_matrix(1, 9), OcioTransform(lin709, "Linear Rec.2020").apply_float(basis).T),
        "pq_inverse_eotf": diff(transfer.pq_inverse_eotf(nits), OcioTransform("Linear Rec.2020", "Rec.2100-PQ - Display").apply_float(display)[:, 0]),
        "hlg_oetf_with_inverse_ootf": diff(transfer.hlg_oetf((nits / 1000.0) ** (1 / 1.2)),
                                           OcioTransform("Linear Rec.2020", "Rec.2100-HLG - Display").apply_float(display)[:, 0]),
        # OCIO's "Camera Rec.709" uses the rounded textbook constants 1.099 / 0.018; ours are the exact continuous pair.
        "bt709_oetf_rounded_constants": diff(transfer.bt709_oetf(np.linspace(0, 1, 257)),
                                             OcioTransform(lin709, "Camera Rec.709").apply_float(
                                                 np.repeat(np.linspace(0, 1, 257, dtype=np.float32)[:, None], 3, axis=1))[:, 0]),
    }
