"""Compile a Pipeline plan onto the existing execution path; never a second runner.

A plan becomes ``Settings`` overrides and ``WorkstationParameters`` fields for the reference or stream
runner. Anything the runner cannot honour is listed under ``unsupported`` (Stage 4 precedent) and makes the
plan *inexpressible*; the selector treats inexpressible plans as structurally invalid, so a plan the runner
would silently alter is never selected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from roop_3d.orchestrator.plan import Plan

STREAM_ENGINES = frozenset({"ffmpeg_stream_source", "ffmpeg_stream_writer"})
PROJECTION_ENGINES = frozenset({"identity_asset_builder"})
OCCLUSION_ENGINES = frozenset({"bisenet_face_parser", "xseg_visible_face"})
DEFAULT_STRENGTHS: dict[str, float] = {}


@dataclass(frozen=True)
class CompiledPlan:
    backend: str                              # reference | stream
    settings: dict[str, Any]
    workstation: dict[str, Any]
    unsupported: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())

    @property
    def expressible(self) -> bool:
        return not self.unsupported

    def as_dict(self) -> dict[str, Any]:
        return {"backend": self.backend, "settings": self.settings, "workstation_parameters": self.workstation,
                "unsupported": list(self.unsupported), "notes": list(self.notes)}


def compile_plan(plan: Plan, strengths: dict[str, float] | None = None) -> CompiledPlan:
    ids = plan.engine_ids()
    values = {**DEFAULT_STRENGTHS, **(strengths or {})}
    backend = "stream" if ids & STREAM_ENGINES else "reference"
    settings: dict[str, Any] = {"render_backend": backend}
    workstation: dict[str, Any] = {}
    unsupported: list[str] = []
    notes: list[str] = []
    if "true3d_head_renderer" not in ids:
        unsupported.append("3D_HEAD_RENDERER_REQUIRED: the plan has no true-3D head render engine")
    if backend == "stream":
        if ids & ({"identity_asset_builder", "liveportrait_retarget"} | OCCLUSION_ENGINES):
            unsupported.append("STREAM_BACKEND_LACKS_3D_EXPRESSION_AND_OCCLUSION_ENGINES")
        notes.append("stream backend jobs must stay 'basic': no workstation parameters are emitted")
    else:
        if not ids & OCCLUSION_ENGINES:
            unsupported.append("REFERENCE_RUNNER_ALWAYS_ATTACHES_FACE_PARSER: plan has no occlusion engine")
        if "xseg_visible_face" in ids:
            workstation["occlusion_mode"] = "robust"
        elif "bisenet_face_parser" in ids:
            workstation["occlusion_mode"] = "standard"
    if any(e.provider == "HardwareCodec" for e in plan.engines if e.category in ("video_decoding", "video_encoding")):
        notes.append("HardwareCodec is a preference: the runner records nvdec_state/nvenc_state and may use software; "
                     "ROOP_STRICT_NVDEC/NVENC are environment settings the plan cannot set")
    if backend == "reference" and any(e.provider == "TensorrtExecutionProvider" for e in plan.engines):
        unsupported.append("REFERENCE_RUNNER_ACCEPTS_CPU_OR_CUDA_ONLY: TensorRT is a stream-backend provider")
    return CompiledPlan(backend, settings, workstation, tuple(unsupported), tuple(notes))
