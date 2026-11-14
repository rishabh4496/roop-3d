"""Full-frame adapter between volumetric reconstruction, semantic parsing, and mesh compositing.

Only full-frame mesh rasters and frame-space masks cross this boundary. Cropped replacement images and affine paste-back
are not accepted. The current ReferencePipeline fails closed because its true-3D renderer is not wired.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
import numpy as np

from roop_3d.compositor.masks import INCLUDED_SKIN_FACE_CLASSES, STRICT_FORBIDDEN_CLASSES, BisenetFaceParser
from roop_3d.core.errors import DomainError
from roop_3d.core.true3d_guard import FRAME_PIPELINE_STAGES, True3DFramePipelineError, require_true3d_frame_pipeline
from roop_3d.pipeline.photorealistic_swap_core import FacePacket, FaceReport, Photorealistic3DSwapCore

log = logging.getLogger("roop3d.pipeline")

STEP_TO_STAGE = {}



@dataclass
class FrameReport:
    frame_index: int
    scene_cut: bool = False
    faces: list[FaceReport] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)       # faces that could NOT be swapped this frame, with the reason
    withheld: list[dict] = field(default_factory=list)      # faces the identity gate refused to swap
    stage_ms: dict[str, float] = field(default_factory=dict)




class SessionFrameProcessor:
    """Run the true-3D render, semantic-mask, and blend stages for one frame."""

    FIT_SKIP_CODES = frozenset({
        "DENSE_HEAD_LANDMARKS_REQUIRED", "HEAD_CORRESPONDENCE_REQUIRED", "HEAD_CORRESPONDENCE_INVALID",
        "HEAD_CAMERA_INVALID", "HEAD_POSE_FAILED", "HEAD_POSE_REJECTED",
        "HEAD_VISIBLE_LANDMARKS_INSUFFICIENT", "HEAD_BEHIND_CAMERA",
        "TRUE3D_RENDER_INVALID", "TRUE3D_MESH_NORMALS_REQUIRED", "TRUE3D_SEMANTIC_MASK_INVALID",
        "TRUE3D_SEMANTIC_MASK_EMPTY", "TRUE3D_OCCLUDER_MASK_INVALID", "TRUE3D_SEMANTIC_PARSER_FAILED",
    })

    def __init__(self, session, core: Photorealistic3DSwapCore):
        self.session, self.core = session, core
        self.parser = None
        self.frames = 0
        self.totals: dict[str, float] = {}
        self.faces_swapped = 0
        self.faces_skipped = 0
        self.faces_withheld = 0
        self.geometry_modes: dict[str, int] = {}
        self.mask_tiers: dict[str, int] = {}

    def process(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, FrameReport]:
        began = time.perf_counter()
        try:
            tracked = self.session.track_frame(frame)
            result = self.session.infer_frame(frame, tracked=tracked)
        except DomainError as error:
            if error.code not in self.FIT_SKIP_CODES:
                raise
            report = FrameReport(frame_index)
            diagnostic = {"event": "true3d_frame_skipped", "code": error.code, "message": error.message}
            log.warning("true3d_frame_skipped=%s", json.dumps(diagnostic, sort_keys=True))
            report.skipped.append(diagnostic)
            self._finish_report(report, began)
            return frame, report
        return self.process_result(frame, result, frame_index)

    def process_result(self, frame: np.ndarray, result, frame_index: int) -> tuple[np.ndarray, FrameReport]:
        """Composite only renderer results carrying a full-frame raster and semantic masks."""
        began = time.perf_counter()
        report = FrameReport(frame_index, scene_cut=bool(result.scene_cut))
        report.stage_ms["detection"] = float(result.latency.detection_ms)
        report.stage_ms["head_render"] = float(result.latency.swap_ms + result.latency.retarget_ms)
        report.withheld = [{"track_id": item.track_id, "reason": item.reason} for item in result.withheld]
        output = frame
        for replacement in result.replacements:
            try:
                packet, parse_ms = self._packet(frame, replacement)
                output, face = self.core.process_face(output, packet, frame_index=frame_index)
            except True3DFramePipelineError as error:
                diagnostic = {"event": "true3d_frame_skipped", "track_id": replacement.track_id,
                              "code": error.code, "details": error.details}
                log.warning("true3d_frame_skipped=%s", json.dumps(diagnostic, sort_keys=True))
                report.skipped.append(diagnostic)
                continue
            except DomainError as error:
                if error.code not in self.FIT_SKIP_CODES:
                    raise
                diagnostic = {"event": "true3d_frame_skipped", "track_id": replacement.track_id,
                              "code": error.code, "message": error.message}
                log.warning("true3d_frame_skipped=%s", json.dumps(diagnostic, sort_keys=True))
                report.skipped.append(diagnostic)
                continue
            mode = face.geometry["mode"]
            tier = packet.mask_diagnostics["tier"]
            self.geometry_modes[mode] = self.geometry_modes.get(mode, 0) + 1
            self.mask_tiers[tier] = self.mask_tiers.get(tier, 0) + 1
            report.faces.append(face)
            report.stage_ms["masking"] = report.stage_ms.get("masking", 0.0) + parse_ms
        self._finish_report(report, began)
        return output, report

    def _finish_report(self, report: FrameReport, began: float) -> None:
        self.frames += 1
        self.faces_swapped += len(report.faces)
        self.faces_skipped += len(report.skipped)
        self.faces_withheld += len(report.withheld)
        for stage, milliseconds in report.stage_ms.items():
            self.totals[stage] = self.totals.get(stage, 0.0) + milliseconds
        report.stage_ms["total"] = (time.perf_counter() - began) * 1000

    def mean_stage_ms(self) -> dict[str, float]:
        return {stage: round(total / max(self.frames, 1), 3) for stage, total in self.totals.items()}

    def _packet(self, frame: np.ndarray, replacement) -> tuple[FacePacket, float]:
        if not replacement.full_head:
            raise True3DFramePipelineError(
                "Renderer output is not attested as a complete 3D head",
                {"expected": list(FRAME_PIPELINE_STAGES[:3]), "observed": list(replacement.pipeline_stages)},
            )
        require_true3d_frame_pipeline(replacement.pipeline_stages, through=FRAME_PIPELINE_STAGES[2])
        began = time.perf_counter()
        if self.parser is None:
            self.parser = BisenetFaceParser(self.session.catalog, self.session.config)
        try:
            probabilities = np.asarray(self.parser.predict(frame), np.float32)
        except Exception as error:
            raise DomainError("TRUE3D_SEMANTIC_PARSER_FAILED",
                              f"Full-frame semantic parser failed: {type(error).__name__}", 503) from error
        if (probabilities.ndim != 3 or probabilities.shape[:2] != frame.shape[:2]
                or probabilities.shape[2] <= max((*INCLUDED_SKIN_FACE_CLASSES, *STRICT_FORBIDDEN_CLASSES))
                or not np.isfinite(probabilities).all()):
            raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID",
                              "Semantic probabilities must be finite full-frame class planes", 422)
        skin = np.clip(np.sum(probabilities[..., INCLUDED_SKIN_FACE_CLASSES], axis=-1), 0.0, 1.0)
        hair = np.clip(np.sum(probabilities[..., STRICT_FORBIDDEN_CLASSES], axis=-1), 0.0, 1.0)
        stages = (*replacement.pipeline_stages, FRAME_PIPELINE_STAGES[3])
        require_true3d_frame_pipeline(stages, through=FRAME_PIPELINE_STAGES[3])
        packet = FacePacket(
            track_id=replacement.track_id,
            rendered_frame=np.asarray(replacement.rendered_frame),
            coverage=np.asarray(replacement.coverage, np.float32),
            skin_mask=skin,
            hair_mask=hair,
            normals=np.asarray(replacement.normals, np.float32),
            pipeline_stages=stages,
            occluder=None if replacement.occluder is None else np.asarray(replacement.occluder, np.float32),
            mask_diagnostics={"tier": "bisenet_full_frame", "hair_exclusion": True},
        )
        return packet, (time.perf_counter() - began) * 1000
    def release(self) -> None:
        if self.parser is not None:
            self.parser.release()
            self.parser = None
