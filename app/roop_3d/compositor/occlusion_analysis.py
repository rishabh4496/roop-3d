"""Unified face + occlusion analysis for one target face crop.

Evidence from real local footage (``docs/OCCLUSION_STAGE5.md``) drives three design rules:

1. The BiSeNet parser is *not* trusted blindly. On side profiles and heavily blurred faces it can
   return almost no face class while still being numerically "confident". Its face coverage is
   therefore cross-checked against the visible-face networks; on disagreement the parser is
   ignored and the networks define the face shape (tier ``neural``).
2. Segmentation uncertainty never invents foreground. Two visible-face networks are combined with
   ``min`` (a pixel is face only if both agree), so disagreement keeps the original target pixels.
3. Every result carries a confidence, a tier and human-readable reasons; callers must never be
   told a stronger tier than what ran.

Tiers: ``full`` (parser + networks), ``semantic`` (parser only), ``neural`` (networks define shape),
``geometric`` (landmark hull only, occlusion unverified).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import cv2
import numpy as np

from roop_3d.compositor.face_regions import FaceRegions, build_regions, restoration_protection
from roop_3d.compositor.masks import FaceParser, build_face_mask, landmark_hull_mask
from roop_3d.compositor.occlusion import combine_occluders
from roop_3d.compositor.occlusion_model import DEFAULT_VISIBLE_NET, VisibleFaceNet
from roop_3d.core.errors import DomainError

DEFAULT_NETS = (DEFAULT_VISIBLE_NET, "face_occluder.onnx")


@dataclass(frozen=True)
class FaceAnalysis:
    face_alpha: np.ndarray                 # hull ∩ semantic face (feathered), before any occluder
    occluder: np.ndarray | None            # front-object probability; None when occlusion is unverified
    skin: np.ndarray                       # skin map for texture work (zeros without a trusted parser)
    regions: FaceRegions | None
    protect: np.ndarray | None             # restoration protection map (eyes, brows, mouth, teeth estimate)
    confidence: float
    tier: str
    reasons: tuple[str, ...] = ()
    nets_used: tuple[str, ...] = ()
    parser_trusted: bool = False
    visible_fraction: float | None = None  # fraction of the face hull the networks call visible
    timing_ms: dict[str, float] = field(default_factory=dict)


def _interior_weight(hull: np.ndarray, fraction: float = 0.06, border_trust: float = 0.6) -> np.ndarray:
    """How much the networks' "not face" vote counts at each hull pixel.

    The networks' face outlines differ from the landmark hull, so at the hull border their vote is
    discounted to ``border_trust`` instead of 0: a hand reaching the border must still be able to
    occlude, while a merely tighter outline only costs a partial vote. Outside the hull it is 0.
    """
    side = min(hull.shape)
    radius = max(1, int(round(side * fraction)))
    core = cv2.erode((hull > 0.5).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
    interior = cv2.GaussianBlur(core.astype(np.float32), (2 * radius + 1,) * 2, radius / 2.0)
    inside = np.clip(hull * 2.0, 0.0, 1.0)
    return (inside * (border_trust + (1.0 - border_trust) * interior)).astype(np.float32)


class OcclusionAnalyzer:
    def __init__(self, catalog=None, config=None, *, parser: FaceParser | None = None,
                 net_names: tuple[str, ...] = DEFAULT_NETS, parser_normalization: str = "imagenet"):
        from roop_3d.engine.runtime import RuntimeConfig
        from roop_3d.compositor.masks import BisenetFaceParser

        self.catalog, self.config = catalog, config or RuntimeConfig()
        # The robust path measures its own preprocessing (ImageNet RGB); an injected parser is used as given.
        self.parser = parser if parser is not None else BisenetFaceParser(catalog, self.config,
                                                                          normalization=parser_normalization)
        self.nets = [VisibleFaceNet(catalog, self.config, model_name=name) for name in net_names]
        self._unavailable: set[str] = set()
        self._trust_streak: dict[int, int] = {}
        self.frames_analysed = 0

    def retain(self, track_ids) -> None:
        """Forget per-track state of faces that are no longer active."""
        keep = set(track_ids)
        self._trust_streak = {key: value for key, value in self._trust_streak.items() if key in keep}

    # -- model access ---------------------------------------------------------------------------------
    def _predict_nets(self, crop: np.ndarray) -> tuple[list[np.ndarray], list[str]]:
        maps, names = [], []
        for net in self.nets:
            if net.model_name in self._unavailable:
                continue
            if not net.available():
                self._unavailable.add(net.model_name)
                continue
            try:
                maps.append(net.predict(crop))
                names.append(net.model_name)
            except (DomainError, RuntimeError, OSError, ValueError):
                self._unavailable.add(net.model_name)
        return maps, names

    # -- analysis -------------------------------------------------------------------------------------
    def analyze(self, crop: np.ndarray, landmarks: np.ndarray | None, *, feather_px: float = 5.0,
                confidence_threshold: float = 0.55, explicit_front: np.ndarray | None = None,
                border_trust: float = 0.6, track_id: int | None = None, trust_frames: int = 3,
                mesh_support: np.ndarray | None = None) -> FaceAnalysis:
        began = time.perf_counter()
        height, width = crop.shape[:2]
        shape = (height, width)
        hull = landmark_hull_mask(shape, landmarks, feather_px=feather_px)
        support = hull > 0.05
        reasons: list[str] = []
        timing: dict[str, float] = {}

        mark = time.perf_counter()
        probabilities = None
        if self.parser is not None:
            try:
                probabilities = self.parser.predict(crop)
            except (DomainError, RuntimeError, OSError, ValueError) as error:
                reasons.append(f"PARSER_UNAVAILABLE:{getattr(error, 'code', type(error).__name__)}")
        timing["parser"] = (time.perf_counter() - mark) * 1000

        mark = time.perf_counter()
        visibles, names = self._predict_nets(crop)
        timing["networks"] = (time.perf_counter() - mark) * 1000
        visible = np.minimum.reduce(visibles) if visibles else None
        visible_fraction = None
        if visible is not None and support.any():
            visible_fraction = float((visible[support] * hull[support]).sum() / max(float(hull[support].sum()), 1e-6))

        # Landmark sanity: on extreme roll, blur or profile the 106-point hull can collapse to a small cluster.
        # If the networks see a face several times larger than the hull, the networks define the shape instead.
        hull_implausible = False
        if visible is not None:
            visible_area, hull_area = int((visible > 0.5).sum()), int((hull > 0.5).sum())
            if visible_area > 0 and hull_area < 0.5 * visible_area:
                hull_implausible = True
                reasons.append("HULL_IMPLAUSIBLE_USING_NETWORKS")

        # Semantic face (no occluder removal here: occluders are unified separately below).
        semantic = None
        parser_trusted = False
        semantic_confidence = 0.0
        if probabilities is not None and not hull_implausible:
            semantic = build_face_mask(shape, landmarks, probabilities=probabilities,
                                       confidence_threshold=confidence_threshold, feather_px=feather_px,
                                       occluder_class_ids=())
            semantic_confidence = float(semantic.confidence)
            parser_trusted = semantic.source.startswith("bisenet")
            if parser_trusted and support.any():
                coverage = float((semantic.alpha[support] * hull[support]).sum() / max(float(hull[support].sum()), 1e-6))
                if coverage < 0.35 and (visible_fraction is None or visible_fraction > 0.5):
                    parser_trusted = False
                    reasons.append("PARSER_DISAGREES_WITH_NETWORKS" if visible is not None else "PARSER_FACE_COVERAGE_LOW")
            elif not parser_trusted:
                reasons.append("PARSER_LOW_CONFIDENCE")
        # Hysteresis: trust is lost immediately but regained only after ``trust_frames`` consistent frames, so a
        # borderline parser cannot flip the whole mask strategy (and the matte with it) frame to frame.
        if track_id is not None:
            streak = self._trust_streak.get(track_id, 0) + 1 if parser_trusted else 0
            self._trust_streak[track_id] = streak
            if parser_trusted and streak < trust_frames:
                parser_trusted = False
                reasons.append("PARSER_TRUST_WARMUP")

        explicit = None if explicit_front is None else np.clip(np.asarray(explicit_front, np.float32), 0, 1)
        zeros = np.zeros(shape, np.float32)
        if parser_trusted:
            regions = build_regions(probabilities)
            # Strict Hair, Ear & Neck Exclusion: hard-subtract forbidden classes
            forbidden_static = np.clip(
                regions.get("glasses") + regions.get("hat") + regions.get("cloth")
                + regions.get("neck") + regions.get("ears"), 0, 1
            )
            semantic_front = combine_occluders(forbidden_static, regions.get("hair") * support)
            face_alpha = semantic.alpha
            skin = regions.get("skin") * support
            protect = restoration_protection(regions, crop)
            if visible is not None:
                net_front = (1.0 - visible) * _interior_weight(hull, border_trust=border_trust) * support
                tier = "full"
            else:
                net_front, tier = None, "semantic"
                reasons.append("NETWORKS_UNAVAILABLE")
            front = combine_occluders(semantic_front, net_front, explicit)
        elif visible is not None and hull_implausible:
            regions, skin, protect = None, zeros, None
            radius = max(1, int(round(min(shape) * 0.02)))
            grown = cv2.dilate((visible > 0.5).astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2))
            blur = max(3, int(feather_px * 3) | 1)
            face_alpha = cv2.GaussianBlur(grown, (blur, blur), max(feather_px, 0.5))
            front = explicit                                                    # the visible region is already unoccluded
            tier = "neural"
        elif visible is not None:
            regions, skin, protect = None, zeros, None
            face_alpha = hull
            # The networks define both shape and occlusion here, so they act across the whole hull.
            front = combine_occluders((1.0 - visible) * support, explicit)
            tier = "neural"
        else:
            regions, skin, protect = None, zeros, None
            face_alpha = hull
            front = explicit
            tier = "geometric"
            reasons.append("OCCLUSION_UNVERIFIED")

        if mesh_support is not None and probabilities is not None:
            # Hair/glasses may lie above or outside the target landmark hull.
            # A full-head mesh still needs their semantic foreground veto there.
            from roop_3d.compositor.occlusion import semantic_occluder_mask
            support_head = np.asarray(mesh_support, np.float32)
            if support_head.shape != shape or not np.isfinite(support_head).all():
                raise ValueError("Full-head support must match the analysis canvas")
            # Face-coverage trust is a different question from a confident hair,
            # hat or glasses prediction. Waiting for face-trust warmup paints
            # over the hairline in the very first full-head frame of a shot.
            foreground = semantic_occluder_mask(probabilities)
            foreground *= probabilities.max(axis=2) >= confidence_threshold
            front = combine_occluders(front, foreground * np.clip(support_head, 0, 1))

        # Confidence is the weakest verified component; it never rises above what ran.
        parts = [1.0]
        if tier in {"full", "semantic"}:
            parts.append(semantic_confidence)
        if len(visibles) > 1:
            interior = _interior_weight(hull, border_trust=border_trust) > 0.8
            if interior.any():
                parts.append(float(1.0 - np.abs(visibles[0] - visibles[1])[interior].mean() * 2.0))
        if tier == "geometric":
            parts.append(0.3)
        confidence = float(np.clip(min(parts), 0.0, 1.0))
        if visible_fraction is not None and visible_fraction < 0.15:
            reasons.append("FACE_MOSTLY_HIDDEN")
        timing["total"] = (time.perf_counter() - began) * 1000
        self.frames_analysed += 1
        return FaceAnalysis(face_alpha=np.clip(face_alpha, 0, 1).astype(np.float32),
                            occluder=None if front is None else np.clip(front, 0, 1).astype(np.float32),
                            skin=np.clip(skin, 0, 1).astype(np.float32), regions=regions, protect=protect,
                            confidence=confidence, tier=tier, reasons=tuple(reasons), nets_used=tuple(names),
                            parser_trusted=parser_trusted, visible_fraction=visible_fraction, timing_ms=timing)

    def capability(self) -> dict:
        parser = self.parser.capability() if hasattr(self.parser, "capability") else {"weights": "UNKNOWN"}
        if hasattr(self.parser, "normalization"):
            parser = {**parser, "preprocessing": f"{self.parser.normalization} RGB (measured against XSeg on real crops)",
                      "class_mapping": "CHECKED_ON_REAL_FRAMES (anatomical order of brows < eyes < nose < lips; "
                                       "tests/test_occlusion_real_models.py)"}
        return {"parser": parser, "visible_face_networks": [net.capability() for net in self.nets],
                "unavailable": sorted(self._unavailable), "frames_analysed": self.frames_analysed,
                "teeth": "HEURISTIC_ONLY (the parser has no teeth class)",
                "foreground_objects": "visible-face networks only (no object-class model is installed)"}

    def release(self) -> None:
        if self.parser is not None and hasattr(self.parser, "release"):
            self.parser.release()
        for net in self.nets:
            net.release()
        self._unavailable.clear()
