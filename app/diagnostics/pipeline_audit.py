"""Unified Pipeline Diagnostic & Telemetry Framework (Non-breaking).

Audits cross-stage tensor shapes, active ONNX execution providers per node,
Host-to-Device (H2D) and Device-to-Host (D2H) copy counts, and enhancer
governance/bypass events. In debug mode (ROOP_AUDIT_DEBUG=1 or fail_loud=True),
it fails loudly with actionable diagnostics instead of silently degrading quality.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

log = logging.getLogger("roop3d.diagnostics.pipeline_audit")


class AuditError(RuntimeError):
    """Base class for diagnostic audit failures in debug mode."""


class StageQualityDegradationError(AuditError):
    """Raised when an enhanced high-resolution tensor is silently downsampled."""


class ExecutionProviderFallbackError(AuditError):
    """Raised when an engine silently falls back to CPU nodes under TensorRT/CUDA."""


class EnhancerBypassError(AuditError):
    """Raised when an enhancer is silently bypassed or routed to an unverified fallback."""


@dataclass
class AuditConfig:
    """Diagnostic configuration driven by environment variables or programmatic overrides."""
    debug_mode: bool = field(
        default_factory=lambda: os.getenv("ROOP_AUDIT_DEBUG", "0").lower() in ("1", "true", "yes")
    )
    fail_loud: bool = field(
        default_factory=lambda: os.getenv("ROOP_AUDIT_FAIL_LOUD", "0").lower() in ("1", "true", "yes")
    )
    strict_trt: bool = field(
        default_factory=lambda: os.getenv("ROOP_AUDIT_STRICT_TRT", "0").lower() in ("1", "true", "yes")
    )
    max_history_entries: int = 1000
    downsample_warning_ratio: float = 0.5
    min_enhanced_resolution: int = 512
    max_downsampled_target: int = 128


@dataclass
class TensorShapeRecord:
    stage: str
    tag: str
    shape: tuple[int, ...]
    dtype: str
    frame_index: int | None = None
    is_exit: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class TensorShapeTracker:
    """Tracks entering and exiting tensor shapes per stage and catches silent downsampling."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.history: list[TensorShapeRecord] = []
        self.degradation_events: list[dict[str, Any]] = []

    def record(
        self,
        stage: str,
        tag: str,
        tensor_or_shape: Any,
        is_exit: bool = False,
        frame_index: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[int, ...]:
        shape: tuple[int, ...]
        dtype: str = "unknown"
        if hasattr(tensor_or_shape, "shape"):
            shape = tuple(int(d) for d in tensor_or_shape.shape)
            dtype = str(getattr(tensor_or_shape, "dtype", "unknown"))
        elif isinstance(tensor_or_shape, (tuple, list)):
            shape = tuple(int(d) for d in tensor_or_shape)
        else:
            shape = ()

        rec = TensorShapeRecord(
            stage=stage,
            tag=tag,
            shape=shape,
            dtype=dtype,
            frame_index=frame_index,
            is_exit=is_exit,
            metadata=metadata or {},
        )
        if len(self.history) < self.config.max_history_entries:
            self.history.append(rec)
        return shape

    def check_stage3_downsampling(
        self,
        entering_shape: tuple[int, ...],
        enhanced_shape: tuple[int, ...],
        exiting_shape: tuple[int, ...],
        frame_index: int | None = None,
        enhancer_name: str = "unknown",
    ) -> None:
        """Verifies if a 512+ enhanced tensor was downsampled back to 128 or input resolution."""
        # Detect height and width from typical NCHW or HWC shapes
        def get_hw(s: tuple[int, ...]) -> tuple[int, int]:
            if len(s) == 4:
                return s[2], s[3]
            if len(s) == 3:
                return s[0], s[1]
            if len(s) == 2:
                return s[0], s[1]
            return 0, 0

        enh_h, enh_w = get_hw(enhanced_shape)
        exit_h, exit_w = get_hw(exiting_shape)

        if (
            enh_h >= self.config.min_enhanced_resolution
            and exit_h <= self.config.max_downsampled_target
        ):
            event = {
                "event": "stage3_silent_downsample",
                "enhancer": enhancer_name,
                "entering_shape": entering_shape,
                "enhanced_shape": enhanced_shape,
                "exiting_shape": exiting_shape,
                "frame_index": frame_index,
                "detail": (
                    f"Stage 3 output was enhanced at {enh_h}x{enh_w} but downsampled back to "
                    f"{exit_h}x{exit_w} prior to inverse affine warping!"
                ),
            }
            self.degradation_events.append(event)
            log.warning("[AUDIT_ALERT] %s: %s", event["event"], event["detail"])

            if self.config.debug_mode and self.config.fail_loud:
                raise StageQualityDegradationError(event["detail"])


class ExecutionProviderAuditor:
    """Inspects ONNX Runtime sessions, detecting CPU fallback nodes under TensorRT/CUDA."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.session_reports: dict[str, dict[str, Any]] = {}
        self.fallback_events: list[dict[str, Any]] = []

    def inspect_session(
        self,
        name: str,
        session: Any,
        expected_provider: str | None = None,
    ) -> dict[str, Any]:
        """Examines active providers and engine metadata for an onnxruntime InferenceSession."""
        if session is None:
            report = {"name": name, "status": "SESSION_NONE", "providers": []}
            self.session_reports[name] = report
            return report

        providers = list(getattr(session, "get_providers", lambda: [])())
        provider_options = {}
        if hasattr(session, "get_provider_options"):
            try:
                provider_options = session.get_provider_options()
            except Exception as e:
                provider_options = {"error": str(e)}

        active_primary = providers[0] if providers else "UNKNOWN"
        has_fallback = False
        fallback_details = []

        if expected_provider and active_primary != expected_provider:
            has_fallback = True
            fallback_details.append(
                f"Expected primary provider '{expected_provider}' but active is '{active_primary}'"
            )

        # Inspect TensorRT profiling log if available
        profile_file = None
        cpu_nodes: list[str] = []
        if hasattr(session, "end_profiling"):
            try:
                profile_file = session.end_profiling()
                if profile_file and Path(profile_file).is_file():
                    cpu_nodes = self._parse_profile_for_cpu_nodes(Path(profile_file))
            except Exception:
                pass

        if cpu_nodes and active_primary == "TensorrtExecutionProvider":
            has_fallback = True
            fallback_details.append(
                f"TensorRT session has {len(cpu_nodes)} unaccelerated nodes executing on CPU!"
            )

        report = {
            "name": name,
            "active_primary": active_primary,
            "all_providers": providers,
            "provider_options": provider_options,
            "has_fallback": has_fallback,
            "fallback_details": fallback_details,
            "cpu_fallback_nodes": cpu_nodes[:20],
            "total_cpu_fallback_nodes": len(cpu_nodes),
            "profile_file": profile_file,
        }
        self.session_reports[name] = report

        if has_fallback:
            self.fallback_events.append(report)
            msg = f"[AUDIT_ALERT] EP Fallback in {name}: {'; '.join(fallback_details)}"
            log.warning(msg)
            if self.config.debug_mode and (self.config.fail_loud or self.config.strict_trt):
                raise ExecutionProviderFallbackError(msg)

        return report

    @staticmethod
    def _parse_profile_for_cpu_nodes(profile_path: Path) -> list[str]:
        """Parses ONNX Runtime profiling JSON trace to find nodes executed on CPU."""
        cpu_nodes: list[str] = []
        try:
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and item.get("cat") == "Node":
                    args = item.get("args", {})
                    provider = args.get("provider", "")
                    node_name = item.get("name", "unknown")
                    if "cpu" in provider.lower():
                        cpu_nodes.append(f"{node_name} ({args.get('op_name', 'Op')})")
        except Exception as e:
            log.debug("Failed parsing profile json %s: %s", profile_path, e)
        return cpu_nodes


class CopyCounter:
    """Tracks Host-to-Device (H2D) and Device-to-Host (D2H) copy bottlenecks."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.h2d_count: int = 0
        self.d2h_count: int = 0
        self.cpu_opencv_ops: int = 0
        self.copy_events: list[dict[str, Any]] = []

    def record_h2d(self, tag: str, shape: tuple[int, ...] | None = None, bytes_count: int = 0) -> None:
        self.h2d_count += 1
        if len(self.copy_events) < self.config.max_history_entries:
            self.copy_events.append({"type": "H2D", "tag": tag, "shape": shape, "bytes": bytes_count})

    def record_d2h(self, tag: str, shape: tuple[int, ...] | None = None, bytes_count: int = 0) -> None:
        self.d2h_count += 1
        if len(self.copy_events) < self.config.max_history_entries:
            self.copy_events.append({"type": "D2H", "tag": tag, "shape": shape, "bytes": bytes_count})

    def record_cpu_op(self, op_name: str, details: str = "") -> None:
        self.cpu_opencv_ops += 1
        if len(self.copy_events) < self.config.max_history_entries:
            self.copy_events.append({"type": "CPU_OP", "op": op_name, "details": details})

    def get_summary(self) -> dict[str, int]:
        return {
            "h2d_copies": self.h2d_count,
            "d2h_copies": self.d2h_count,
            "total_sync_copies": self.h2d_count + self.d2h_count,
            "cpu_opencv_ops": self.cpu_opencv_ops,
        }


class EnhancerBypassDetector:
    """Detects and alerts on enhancer bypasses, <=320px short-circuits, and governance collapses."""

    def __init__(self, config: AuditConfig):
        self.config = config
        self.bypass_events: list[dict[str, Any]] = []

    def audit_invocation(
        self,
        enhancer_name: str,
        crop_shape: tuple[int, ...],
        native_face_pixels: int | None = None,
        restoration_blend: float = 1.0,
        fidelity_weight: float | None = None,
        catalog_verified: bool = True,
        is_bypassed: bool = False,
        bypass_reason: str | None = None,
    ) -> dict[str, Any]:
        """Analyzes an enhancer invocation for silent short-circuits or governance issues."""
        h, w = (crop_shape[0], crop_shape[1]) if len(crop_shape) >= 2 else (0, 0)
        effective_native = native_face_pixels if native_face_pixels is not None else min(h, w)
        reasons = []

        if not catalog_verified:
            reasons.append("Model weights not VERIFIED_LOCAL in catalog")

        if restoration_blend <= 0.0:
            reasons.append(f"restoration_blend is {restoration_blend} (silent no-op)")

        if enhancer_name.lower() in ("ultramax", "ultramax_bfr") and effective_native <= 320:
            reasons.append(
                f"<=320px native face-crop bypass rule triggered (native: {effective_native}px <= 320px)"
            )

        if enhancer_name.lower() in ("gfpgan", "gfpgan_1_4"):
            reasons.append("GFPGAN model has no registered execution adapter; routes to fallback")

        if is_bypassed or reasons:
            all_reasons = (reasons + [bypass_reason]) if bypass_reason else reasons
            event = {
                "enhancer": enhancer_name,
                "crop_shape": crop_shape,
                "native_face_pixels": effective_native,
                "restoration_blend": restoration_blend,
                "fidelity_weight": fidelity_weight,
                "reasons": all_reasons,
                "detail": f"Enhancer '{enhancer_name}' bypassed: {'; '.join(all_reasons)}",
            }
            self.bypass_events.append(event)
            log.warning("[AUDIT_ALERT] %s", event["detail"])

            if self.config.debug_mode and self.config.fail_loud:
                raise EnhancerBypassError(event["detail"])

            return event

        return {"enhancer": enhancer_name, "status": "EXECUTED"}


class PipelineAuditor:
    """Central singleton coordinator orchestrating full pipeline telemetry and auditing."""

    def __init__(self, config: AuditConfig | None = None):
        self.config = config or AuditConfig()
        self.shapes = TensorShapeTracker(self.config)
        self.providers = ExecutionProviderAuditor(self.config)
        self.copies = CopyCounter(self.config)
        self.enhancers = EnhancerBypassDetector(self.config)
        self.stage_timings: dict[str, list[float]] = {}

    def reset(self) -> None:
        self.shapes = TensorShapeTracker(self.config)
        self.providers = ExecutionProviderAuditor(self.config)
        self.copies = CopyCounter(self.config)
        self.enhancers = EnhancerBypassDetector(self.config)
        self.stage_timings.clear()

    def generate_report(self) -> dict[str, Any]:
        """Compiles an exhaustive audit report across all 3 stages and rendering loops."""
        return {
            "debug_mode": self.config.debug_mode,
            "fail_loud": self.config.fail_loud,
            "strict_trt": self.config.strict_trt,
            "copy_summary": self.copies.get_summary(),
            "stage3_downsampling_events": self.shapes.degradation_events,
            "total_downsampling_events": len(self.shapes.degradation_events),
            "ep_fallback_events": self.providers.fallback_events,
            "total_ep_fallback_events": len(self.providers.fallback_events),
            "enhancer_bypass_events": self.enhancers.bypass_events,
            "total_enhancer_bypass_events": len(self.enhancers.bypass_events),
            "session_reports": self.providers.session_reports,
        }

    def save_report(self, destination: Path | str) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(self.generate_report(), indent=2), encoding="utf-8")
        return dest


# Global default instance
audit_pipeline = PipelineAuditor()
