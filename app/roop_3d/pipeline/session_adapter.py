"""Full-frame adapter between volumetric reconstruction, semantic parsing, and mesh compositing.

Only full-frame mesh rasters and frame-space masks cross this seam. Expanded
face ROIs are used solely for semantic inference, never for replacement placement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import time
import cv2
import numpy as np

from roop_3d.compositor.matte import refine_occluder
from roop_3d.compositor.masks import BisenetFaceParser
from roop_3d.compositor.occlusion_model import VisibleFaceNet
from roop_3d.compositor.stage4 import semantic_face_masks
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
        self.visible_nets: tuple[VisibleFaceNet, ...] = ()
        self.last_packet: FacePacket | None = None
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
        self.last_packet = None
        self.core.last_alpha = None
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
        self.last_packet = None
        self.core.last_alpha = None
        if not replacement.full_head:
            raise True3DFramePipelineError(
                "Renderer output is not attested as a complete 3D head",
                {"expected": list(FRAME_PIPELINE_STAGES[:3]), "observed": list(replacement.pipeline_stages)},
            )
        require_true3d_frame_pipeline(replacement.pipeline_stages, through=FRAME_PIPELINE_STAGES[2])
        began = time.perf_counter()
        shape = frame.shape[:2]
        coverage = _mask_plane(replacement.coverage, shape, "coverage")
        mesh_skin = _mask_plane(replacement.skin_mask, shape, "renderer skin")
        mesh_hair = _mask_plane(replacement.hair_mask, shape, "renderer hair")
        ys, xs = np.nonzero(coverage > 0)
        if not len(xs):
            raise DomainError("TRUE3D_SEMANTIC_MASK_EMPTY", "The head raster has no covered pixels to parse", 422)
        width, height = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        # A square contextual ROI preserves the parser's facial scale/aspect and
        # includes the complete forehead, jaw and both visible ears.
        side = int(np.ceil(max(width, height) * 1.25))
        cx, cy = 0.5 * (float(xs.min()) + float(xs.max())), 0.5 * (float(ys.min()) + float(ys.max()))
        left, top = int(np.floor(cx - side / 2)), int(np.floor(cy - side / 2))
        right, bottom = left + side, top + side
        x0, y0 = max(0, left), max(0, top)
        x1, y1 = min(shape[1], right), min(shape[0], bottom)
        roi = np.s_[y0:y1, x0:x1]
        crop = frame[roi]
        # Keep a square inference field even when a head crosses the image edge.
        # Off-frame padding is discarded before any full-frame mask is produced.
        padded = cv2.copyMakeBorder(crop, y0 - top, bottom - y1, x0 - left, right - x1,
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        visible_roi = np.s_[y0 - top:y1 - top, x0 - left:x1 - left]
        if self.parser is None:
            self.parser = BisenetFaceParser(self.session.catalog, self.session.config, normalization="imagenet")
        if not self.visible_nets:
            self.visible_nets = tuple(VisibleFaceNet(self.session.catalog, self.session.config, model_name=name)
                                      for name in ("xseg.onnx", "occluder.onnx"))
        try:
            probabilities = np.asarray(self.parser.predict(padded), np.float32)
            if probabilities.shape[:2] != padded.shape[:2]:
                raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", "Semantic parser output must match its face ROI", 503)
            face, forbidden, skin_only, ears = semantic_face_masks(probabilities[visible_roi])
            visible = [net.predict(padded)[visible_roi] for net in self.visible_nets]
        except DomainError:
            # Preserve missing weights, graph, hash and license diagnostic codes.
            raise
        except Exception as error:
            raise DomainError("TRUE3D_SEMANTIC_PARSER_FAILED",
                              f"Face ROI masking failed: {type(error).__name__}: {error}", 503) from error
        for plane in visible:
            _mask_plane(plane, crop.shape[:2], "visible-face network")
        neural_visible = np.minimum(visible[0], visible[1])
        # Visible-face networks were trained with a face-only silhouette. The
        # explicitly labelled ears are anatomical skin, not foreground objects;
        # retain them while keeping all parser/external occluders authoritative.
        neural_visible = np.where(ears, 1.0, neural_visible).astype(np.float32)
        ramp = np.clip((neural_visible - 0.08) / 0.52, 0, 1)
        network_occluder = 1.0 - ramp * ramp * (3.0 - 2.0 * ramp)
        network_occluder = np.maximum(
            network_occluder, refine_occluder(network_occluder, crop, radius=2, contrast=1.5))
        network_occluder[neural_visible <= 0.08] = 1.0
        skin = np.zeros(shape, np.float32)
        hair = np.ones(shape, np.float32)
        occluder = np.ones(shape, np.float32)
        photometric_skin = np.zeros(shape, np.float32)
        skin[roi] = np.where(coverage[roi] > 0, np.minimum(face, mesh_skin[roi]), 0.0)
        hair[roi] = np.maximum(forbidden, mesh_hair[roi])
        occluder[roi] = network_occluder
        photometric_skin[roi] = skin_only
        if replacement.occluder is not None:
            occluder = np.maximum(occluder, _mask_plane(replacement.occluder, shape, "renderer occluder"))
        stages = (*replacement.pipeline_stages, FRAME_PIPELINE_STAGES[3])
        require_true3d_frame_pipeline(stages, through=FRAME_PIPELINE_STAGES[3])
        packet = FacePacket(
            track_id=replacement.track_id,
            rendered_frame=np.asarray(replacement.rendered_frame),
            coverage=coverage,
            skin_mask=skin,
            hair_mask=hair,
            normals=np.asarray(replacement.normals, np.float32),
            pipeline_stages=stages,
            occluder=occluder,
            mask_diagnostics={
                "tier": "bisenet_face_roi_xseg_and_occluder", "roi": [x0, y0, x1, y1],
                "hair_exclusion": True, "background_exclusion": True, "ears_included": True,
                "parser_normalization": "imagenet", "neural_models": [net.capability() for net in self.visible_nets],
                "ear_pixels": int(ears.sum()), "_skin_only_mask": photometric_skin,
            },
        )
        self.last_packet = packet
        return packet, (time.perf_counter() - began) * 1000
    def release(self) -> None:
        if self.parser is not None:
            self.parser.release()
            self.parser = None
        for net in self.visible_nets:
            net.release()
        self.visible_nets = ()
        self.last_packet = None


def _mask_plane(value, shape: tuple[int, int], name: str) -> np.ndarray:
    plane = np.asarray(value, np.float32)
    if plane.shape != shape or not np.isfinite(plane).all() or np.any((plane < 0) | (plane > 1)):
        raise DomainError("TRUE3D_SEMANTIC_MASK_INVALID", f"{name} must be a finite matching plane in 0..1", 422)
    return plane
