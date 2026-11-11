"""Per-frame glue between a prepared ``ReferencePipeline`` session and :class:`Photorealistic3DSwapCore`.

The session does the real work it always did: SCRFD detection + 106 landmarks + tracking, the identity gate, and the
neural swap (``inswapper``). This adapter turns each ``Replacement`` into a :class:`FacePacket` (picking the FaceSet
photograph nearest in pose, running the parser / occluder networks on a full-head crop) and lets the core do geometry,
rectification, relighting, restoration and the Laplacian composite. Nothing here composites a mesh.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
import time
from typing import Any

import cv2
import numpy as np

from roop_3d.core.errors import DomainError
from roop_3d.engine.volumetric.preprocess import landmark_crop
from roop_3d.pipeline import image_ops as ops
from roop_3d.pipeline.photorealistic_swap_core import FacePacket, FaceReport, Photorealistic3DSwapCore
from roop_3d.reconstruction.expression_tps import layout_is_plausible
from roop_3d.rendering.silhouette_replacement import DonorHistory

log = logging.getLogger("roop3d.pipeline")

# Pipeline stage names the job contract / HUD already know (core.stage_timing.STAGES). The core's five steps map onto them
# the way the HUD columns are labelled: geometry+rectify -> pose_tps ("3D-LBS"), relight+blend -> masking, restore -> restorer.
OCCLUSION_THRESHOLD = 0.55          # the compositor's own default semantic-confidence gate
STEP_TO_STAGE = {"geometry": "pose_tps", "rectify": "pose_tps", "relight": "masking", "blend": "masking", "restore": "restorer"}


@dataclass
class FrameReport:
    frame_index: int
    scene_cut: bool = False
    faces: list[FaceReport] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)       # faces that could NOT be swapped this frame, with the reason
    withheld: list[dict] = field(default_factory=list)      # faces the identity gate refused to swap
    stage_ms: dict[str, float] = field(default_factory=dict)


class SourceViews:
    """Nearest-pose FaceSet photograph per track, with its 106 landmarks (stored, else measured once and cached)."""

    def __init__(self, session, *, cache_size: int = 24):
        self.session = session
        self._cache: OrderedDict[tuple[str, int], tuple[np.ndarray, np.ndarray | None]] = OrderedDict()
        self._cache_size = cache_size

    def pick(self, document, pose: tuple[float, float, float]) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """``(photo_bgr, landmarks106 | None, note)``; a note other than ``"ok"`` says why geometry cannot use the view."""
        views = [view for view in document.views if not view.is_synthetic]
        if not views:
            return None, None, "FACESET_HAS_NO_REAL_VIEWS"
        yaw, pitch = float(pose[0]), float(pose[1])
        best = min(views, key=lambda view: (view.yaw - yaw) ** 2 + 1.5 * (view.pitch - pitch) ** 2)
        key = (document.faceset_id, best.index)
        if key not in self._cache:
            image = cv2.imdecode(np.frombuffer(best.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                return None, None, "FACESET_VIEW_UNDECODABLE"
            points = None
            if best.landmarks_106 is not None and best.landmarks_trusted:
                points = np.asarray(best.landmarks_106, np.float64)
            else:
                try:
                    measured = self.session._detect_landmarks106(image)
                    points = None if measured is None else np.asarray(measured, np.float64)
                except Exception:                                   # a detector failure must not fail the render
                    log.debug("view landmark measurement failed", exc_info=True)
            if points is not None and not layout_is_plausible(points):
                points = None
            self._cache[key] = (image, points)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        image, points = self._cache[key]
        return image, points, "ok" if points is not None else "VIEW_LANDMARKS_UNAVAILABLE"


class SessionFrameProcessor:
    """``process(frame, index)`` -> ``(swapped frame, FrameReport)`` using a prepared session and a swap core."""

    def __init__(self, session, core: Photorealistic3DSwapCore, *, history_depth: int = 2, analysis_size: int = 512):
        self.session, self.core = session, core
        self.views = SourceViews(session)
        self.history = DonorHistory(history_depth)
        self.analysis_size = analysis_size
        self.frames = 0
        self.totals: dict[str, float] = {}
        self.fallbacks: dict[str, int] = {}
        self.faces_swapped = 0
        self.faces_skipped = 0
        self.faces_withheld = 0

    # -------------------------------------------------------------------------------------------------- per frame
    def process(self, frame: np.ndarray, frame_index: int) -> tuple[np.ndarray, FrameReport]:
        report = FrameReport(frame_index)
        began = time.perf_counter()
        tracked = self.session.track_frame(frame)
        result = self.session.infer_frame(frame, tracked=tracked)
        report.scene_cut = bool(result.scene_cut)
        if report.scene_cut:
            self.history.clear()
        report.stage_ms["detection"] = float(result.latency.detection_ms)
        report.stage_ms["inswapper"] = float(result.latency.swap_ms + result.latency.retarget_ms)
        report.withheld = [{"track_id": item.track_id, "reason": item.reason} for item in result.withheld]
        observations = {item.track_id: item.observation for item in result.observations}
        out = frame
        swapped_tracks: set[int] = set()
        for replacement in result.replacements:
            observation = observations.get(replacement.track_id)
            if observation is None:
                report.skipped.append({"track_id": replacement.track_id, "reason": "NO_OBSERVATION"})
                continue
            try:
                packet, parse_ms = self._packet(frame, replacement, observation)
                donors = self.history.donors(replacement.track_id, frame.shape[:2], frame_index, max_gap=4)
                out, face = self.core.process_face(out, packet, donors=donors, frame_index=frame_index)
            except DomainError as error:
                report.skipped.append({"track_id": replacement.track_id, "reason": error.code})
                continue
            report.faces.append(face)
            swapped_tracks.add(replacement.track_id)
            for step, milliseconds in face.steps_ms.items():
                stage = STEP_TO_STAGE[step]
                report.stage_ms[stage] = report.stage_ms.get(stage, 0.0) + milliseconds
            report.stage_ms["masking"] = report.stage_ms.get("masking", 0.0) + parse_ms
            for item in face.fallbacks:
                self.fallbacks[item] = self.fallbacks.get(item, 0) + 1
        # the donor history holds UNSWAPPED frames (background the old head covered is what a donor must show)
        for replacement in result.replacements:
            if replacement.track_id in swapped_tracks and observations.get(replacement.track_id) is not None:
                self.history.push(replacement.track_id, frame, self._head_mask(frame.shape[:2], observations[replacement.track_id]), frame_index)
        self.history.retain(set(tracked.active_ids))
        self.frames += 1
        self.faces_swapped += len(report.faces)
        self.faces_skipped += len(report.skipped)
        self.faces_withheld += len(report.withheld)
        for stage, milliseconds in report.stage_ms.items():
            self.totals[stage] = self.totals.get(stage, 0.0) + milliseconds
        report.stage_ms["total"] = (time.perf_counter() - began) * 1000
        return out, report

    def mean_stage_ms(self) -> dict[str, float]:
        return {stage: round(total / max(self.frames, 1), 3) for stage, total in self.totals.items()}

    # ---------------------------------------------------------------------------------------------------- packets
    def _head_mask(self, shape: tuple[int, int], observation) -> np.ndarray | None:
        points = np.asarray(observation.landmarks106, np.float64)
        if not layout_is_plausible(points):
            return None
        return ops.polygon_mask(ops.head_polygon(points, self.core.options.forehead_lift), shape)

    def _packet(self, frame: np.ndarray, replacement, observation) -> tuple[FacePacket, float]:
        landmarks = np.asarray(observation.landmarks106, np.float64)
        image = np.asarray(replacement.image)
        if image.ndim != 3 or image.shape[0] != image.shape[1]:
            raise DomainError("SWAP_CORE_CROP_INVALID", "The neural swap must be a square crop", 422)
        document = self.session._document_for(replacement.track_id)
        view = view_landmarks = None
        if document is not None:
            view, view_landmarks, _ = self.views.pick(document, observation.pose)
        began = time.perf_counter()
        structure = skin = occluder = protect = None
        try:
            structure, skin, occluder, protect = self._parse(frame, landmarks, replacement)
        except DomainError:
            raise
        except Exception:                                           # a parser failure degrades the masks, never the render
            log.warning("semantic parsing failed for a face; continuing without hair/skin/occluder masks", exc_info=True)
        parse_ms = (time.perf_counter() - began) * 1000
        packet = FacePacket(
            track_id=replacement.track_id, swapped_crop=image, crop_to_frame=np.asarray(replacement.inverse, np.float64),
            target_landmarks=landmarks,
            crop_landmarks5=None if replacement.landmarks5 is None else np.asarray(replacement.landmarks5, np.float32),
            crop_coverage=None if replacement.coverage is None else np.asarray(replacement.coverage, np.float32),
            source_landmarks=view_landmarks, source_view=view if view_landmarks is not None else None,
            structure_mask=structure, skin_mask=skin, occluder=occluder, protect=protect)
        return packet, parse_ms

    def _parse(self, frame: np.ndarray, landmarks: np.ndarray, replacement):
        """Hair / ears / neck, skin and front occluders from the parser + XSeg + occluder networks, in FRAME space."""
        if not layout_is_plausible(landmarks):
            return None, None, None, None
        crop = landmark_crop(frame, landmarks, self.analysis_size, expansion=2.3, vertical_shift=-0.15)
        points = (landmarks @ np.asarray(crop.matrix, np.float64)[:, :2].T) + np.asarray(crop.matrix, np.float64)[:, 2]
        analyzer = self.session.compositor.analyzer
        analysis = analyzer.analyze(crop.image, points, feather_px=3.0, confidence_threshold=OCCLUSION_THRESHOLD,
                                    track_id=replacement.track_id)
        h, w = frame.shape[:2]
        inverse = np.asarray(crop.inverse, np.float32)

        def to_frame(plane, border=0.0) -> np.ndarray:
            return cv2.warpAffine(np.asarray(plane, np.float32), inverse, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=border)

        structure = skin = None
        if analysis.parser_trusted and analysis.regions is not None:
            regions = analysis.regions
            structure = np.clip(to_frame(regions.get("hair")) + to_frame(regions.get("ears")) + to_frame(regions.get("neck")), 0, 1)
            skin = np.clip(to_frame(analysis.skin), 0, 1)
        occluder = None if analysis.occluder is None else np.clip(to_frame(analysis.occluder), 0, 1)
        protect = None
        return structure, skin, occluder, protect
