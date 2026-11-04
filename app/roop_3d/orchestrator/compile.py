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

STREAM_ENGINES = frozenset({"ffmpeg_stream_source", "ffmpeg_stream_writer", "inswapper_batch_stream",
                            "stream_restorer_gpen", "stream_restorer_codeformer", "stream_restorer_ultramax"})
PROJECTION_ENGINES = frozenset({"identity_asset_builder", "pose_blend_projector"})
OCCLUSION_ENGINES = frozenset({"bisenet_face_parser", "xseg_visible_face"})
RESTORER_POLICY = {"gpen_reference": "gpen", "codeformer_reference": "codeformer", "stream_restorer_gpen": "gpen",
                   "stream_restorer_codeformer": "codeformer", "stream_restorer_ultramax": "ultramax"}
DEFAULT_STRENGTHS = {"restoration_blend": 1.0, "tps_expression_weight": 1.0}


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
    restorers = sorted(ids & RESTORER_POLICY.keys())
    if restorers:
        settings["face_restoration_policy"] = RESTORER_POLICY[restorers[0]]
    if backend == "stream":
        if ids & ({"identity_asset_builder", "pose_blend_projector", "liveportrait_retarget", "tps_expression",
                   "reenactment_3d"} | OCCLUSION_ENGINES):
            unsupported.append("STREAM_BACKEND_LACKS_3D_EXPRESSION_AND_OCCLUSION_ENGINES")
        notes.append("stream backend jobs must stay 'basic': no workstation parameters are emitted")
    else:
        if not ids & OCCLUSION_ENGINES:
            unsupported.append("REFERENCE_RUNNER_ALWAYS_ATTACHES_FACE_PARSER: plan has no occlusion engine")
        if not ids & PROJECTION_ENGINES and "liveportrait_retarget" not in ids:
            unsupported.append("REFERENCE_RUNNER_RETARGETS_BY_DEFAULT: omitting liveportrait_retarget needs a 3D projection engine")
        if "reenactment_3d" in ids:
            workstation["render_mode"] = "reenactment_3d"
        elif "identity_asset_builder" in ids:
            workstation["render_mode"] = "pose_3d"
        if "pose_blend_projector" in ids:
            workstation["pose_mode"] = "mesh_blend"
        if "tps_expression" in ids:
            workstation["tps_expression_weight"] = values["tps_expression_weight"]
            notes.append("tps_expression_weight set to %.2f; change it per job if desired" % values["tps_expression_weight"])
        if "xseg_visible_face" in ids:
            workstation["occlusion_mode"] = "robust"
        elif "bisenet_face_parser" in ids:
            workstation["occlusion_mode"] = "standard"
        if ids & {"gpen_reference", "codeformer_reference"}:
            workstation["restorer_model"] = RESTORER_POLICY[restorers[0]]
            workstation["restoration_blend"] = values["restoration_blend"]
            notes.append("restoration_blend set to %.2f" % values["restoration_blend"])
    if any(e.provider == "HardwareCodec" for e in plan.engines if e.category in ("video_decoding", "video_encoding")):
        notes.append("HardwareCodec is a preference: the runner records nvdec_state/nvenc_state and may use software; "
                     "ROOP_STRICT_NVDEC/NVENC are environment settings the plan cannot set")
    if backend == "reference" and any(e.provider == "TensorrtExecutionProvider" for e in plan.engines):
        unsupported.append("REFERENCE_RUNNER_ACCEPTS_CPU_OR_CUDA_ONLY: TensorRT is a stream-backend provider")
    return CompiledPlan(backend, settings, workstation, tuple(unsupported), tuple(notes))
