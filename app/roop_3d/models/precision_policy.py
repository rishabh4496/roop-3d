"""Model-specific precision decisions. FP16 is a candidate until measured parity."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from roop_3d.core.errors import DomainError
from roop_3d.models.export_validation import ParityReport, Tolerance


@dataclass(frozen=True)
class PrecisionPolicy:
    model_name: str
    fp16_candidate: bool
    sensitive_reason: str | None
    fp32: Tolerance
    fp16: Tolerance
    excluded_trt_ops: tuple[str, ...] = ()

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def tolerance(self, precision: str) -> Tolerance:
        if precision == "fp16" and self.fp16_candidate:
            return self.fp16
        if precision == "fp32":
            return self.fp32
        raise DomainError("PRECISION_NOT_PERMITTED", "Sensitive model must retain FP32", 422)

    def admit(self, precision: str, reports: tuple[ParityReport, ...]) -> str:
        expected = self.tolerance(precision)
        if not reports or any(not r.passed or not r.outputs or any(o.failures for o in r.outputs)
                              or r.tolerance != expected for r in reports):
            raise DomainError("PRECISION_PARITY_REQUIRED", "Every golden case must pass this model's exact policy", 422)
        return precision


def policy_for(model_name: str, role: str | None) -> PrecisionPolicy:
    fp32 = Tolerance()
    fp16 = Tolerance(0.02, 0.002)
    sensitive, candidate, excluded = None, False, ()
    if role == "recognizer":
        fp32, fp16 = Tolerance(0.001, 0.00005, 0.995), Tolerance(0.02, 0.002, 0.995)
        candidate = True
    elif role == "identity_swapper":
        fp32 = Tolerance(0.001, 0.00005, image_psnr_db=65, image_ssim=0.9995, border_absolute=0.001)
        fp16 = Tolerance(0.05, 0.0025, image_psnr_db=40, image_ssim=0.995, border_absolute=0.025)
        candidate = True
    elif role == "liveportrait_appearance":
        candidate = True
    elif role == "liveportrait_warp":
        fp32 = Tolerance(0.001, 0.00005, image_psnr_db=65, image_ssim=0.9995, border_absolute=0.0005)
        sensitive, excluded = "3-D warp coordinates and output borders retain FP32", ("GridSample",)
    elif role in ("liveportrait_motion", "liveportrait_stitch"):
        fp32 = Tolerance(0.0005, 0.00005)
        sensitive = "Pose/regression heads and stitching offsets retain FP32"
    elif role == "detector" or "landmark" in model_name or model_name == "2d106det.onnx":
        fp32 = Tolerance(0.0005, 0.00005)
        sensitive = "Detection/landmark regression retains FP32 to avoid jitter"
    else:
        sensitive = "Unreviewed model precision remains FP32"
    return PrecisionPolicy(model_name, candidate, sensitive, fp32, fp16, excluded)


def require_no_custom_ops(requested: bool = False) -> None:
    if requested:
        raise DomainError("CUSTOM_OPS_NOT_APPROVED", "No custom kernel has profiler and parity approval; standard providers remain available", 422)
