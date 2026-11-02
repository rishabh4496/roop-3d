"""FaceSet quality and coverage analysis (Stage 3, Phase A).

Everything here is MEASURED from the supplied images, landmarks, poses and embeddings. Quantities that
cannot be measured with what is installed are reported as ``None`` / "unavailable" rather than guessed
(notably semantic occlusion: no verified occluder model is wired into this app, so occlusion is only measured
from image-border truncation and from an optional caller-supplied mask).

The report answers three questions: which views may enter a reconstruction (and why the others may not),
which pose ranges are covered, and what additional images would improve the result.
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from pydantic import Field

from roop_3d.core.errors import DomainError
from roop_3d.core.settings import Contract
from roop_3d.faceset.pose_matrix import view_pose
from roop_3d.faceset.schemas import FaceSetDocument, FaceView
from roop_3d.reconstruction.multiview_fit import estimate_pose_from_landmarks

IDENTITY_LINK_SIMILARITY = 0.30      # ArcFace cosine similarity above which two views are linked as one person
DUPLICATE_HASH_DISTANCE = 3          # dHash Hamming distance (of 64 bits)
DUPLICATE_POSE_DEG = 3.0             # a perceptual-hash duplicate must also share its pose within this many degrees
MIN_SHARPNESS = 0.03                 # reader normalisation: min(1, laplacian_var / 500)
MIN_FACE_PIXELS = 40.0               # bbox width in pixels
MAX_CLIPPED_FRACTION = 0.35
POSE_MISMATCH_YAW_DEG = 25.0
MAX_ABS_YAW_DEG = 88.0

# (name, target yaw in the standard frame: + = nose toward IMAGE right, tolerance, requirement)
YAW_TARGETS: tuple[tuple[str, float, str], ...] = (
    ("profile_image_left", -75.0, "recommended"),
    ("three_quarter_image_left", -45.0, "required"),
    ("frontal", 0.0, "required"),
    ("three_quarter_image_right", 45.0, "required"),
    ("profile_image_right", 75.0, "recommended"),
)
YAW_TOLERANCE_DEG = 20.0
PITCH_TARGETS: tuple[tuple[str, float], ...] = (("looking_down", -20.0), ("level", 0.0), ("looking_up", 20.0))
PITCH_TOLERANCE_DEG = 12.0


class ViewQuality(Contract):
    index: int
    yaw: float = Field(description="Standard frame, + = nose toward image right")
    pitch: float = Field(description="Standard frame, + = looking up")
    roll: float
    landmark_layout: str
    sharpness: float = Field(description="min(1, Laplacian variance / 500) over the face box")
    mean_luma: float
    clipped_dark: float
    clipped_bright: float
    exposure_score: float
    landmark_score: float
    border_truncation: float = Field(description="Fraction of the face box outside the image")
    occlusion_fraction: float | None = Field(default=None, description="None = not measurable (no mask supplied)")
    identity_distance: float = Field(description="1 - cosine similarity to the main identity cluster centroid")
    identity_cluster: int
    duplicate_of: int | None = None
    usable: bool
    exclusion_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class FaceSetCoverageReport(Contract):
    faceset_id: str
    view_count: int
    usable_count: int
    usable_indices: list[int]
    readiness: str = Field(description="insufficient | blocked_inconsistent_identity | front_only | partial_coverage | full_coverage "
                                       "(pose COVERAGE only; the reconstruction level actually achieved is reported by the fit)")
    yaw_coverage: dict[str, list[int]]
    pitch_coverage: dict[str, list[int]]
    yaw_range_deg: tuple[float, float] | None
    yaw_span_deg: float
    pitch_span_deg: float
    missing: list[str]
    requests: list[str] = Field(description="Human-readable image requests, most valuable first")
    identity: dict[str, Any]
    duplicate_groups: list[list[int]]
    expression_spread: dict[str, float]
    capabilities: dict[str, str]
    views: list[ViewQuality]

    def summary(self) -> str:
        head = (f"{self.faceset_id}: {self.usable_count}/{self.view_count} usable views, readiness={self.readiness}, "
                f"yaw {self.yaw_range_deg}, span {self.yaw_span_deg:.0f} deg")
        return head + ("".join(f"\n  - {r}" for r in self.requests) if self.requests else "")


# --------------------------------------------------------------------------- per-view measurements
def _decode(view: FaceView) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def _face_roi(image: np.ndarray, bbox: Sequence[float]) -> tuple[np.ndarray, float, float]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox)
    area = max((x2 - x1) * (y2 - y1), 1e-6)
    ix1, iy1, ix2, iy2 = max(x1, 0.0), max(y1, 0.0), min(x2, float(w)), min(y2, float(h))
    inside = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    truncation = float(np.clip(1.0 - inside / area, 0.0, 1.0))
    if ix2 - ix1 < 8 or iy2 - iy1 < 8:
        return image, truncation, float(w)
    return image[int(iy1):int(math.ceil(iy2)), int(ix1):int(math.ceil(ix2))], truncation, x2 - x1


def _is_placeholder(view: FaceView, width: int, height: int) -> bool:
    """The reader's fabricated stand-in landmarks (fixed 64 px layout, or fractions of the image)."""
    pts = np.asarray(view.landmarks, np.float64)
    if pts.shape != (5, 2):
        return False
    fixed = np.array([(16.0, 32.0), (48.0, 32.0), (32.0, 48.0), (20.0, 60.0), (44.0, 60.0)])
    frac = np.array([(0.3 * width, 0.4 * height), (0.7 * width, 0.4 * height), (0.5 * width, 0.55 * height),
                     (0.35 * width, 0.75 * height), (0.65 * width, 0.75 * height)])
    return bool(np.allclose(pts, fixed, atol=1e-6) or np.allclose(pts, frac, atol=1e-3))


def _landmark_checks(view: FaceView, width: int, height: int) -> tuple[float, list[str], list[str], str]:
    """-> (score 0..1, blocking issues, warnings, layout)."""
    blocking: list[str] = []
    warnings: list[str] = []
    layout = "68pt" if view.landmarks_68 is not None else "5pt"
    pts5 = np.asarray(view.landmarks, np.float64)
    pts = np.asarray(view.landmarks_68, np.float64) if view.landmarks_68 is not None else pts5
    if not view.landmarks_trusted or _is_placeholder(view, width, height):
        return 0.0, ["LANDMARKS_NOT_OBSERVED"], warnings, layout
    if pts5.shape != (5, 2) or not np.isfinite(pts).all():
        return 0.0, ["LANDMARKS_INVALID"], warnings, layout
    inside = ((pts[:, 0] >= 0) & (pts[:, 0] <= width) & (pts[:, 1] >= 0) & (pts[:, 1] <= height)).mean()
    if inside < 0.8:
        blocking.append("LANDMARKS_OUTSIDE_IMAGE")
    score = float(inside)
    # de-roll, then eyes must be above the nose, nose above the mouth, and the eyes must be separated
    roll = math.radians(view.roll)
    c, s = math.cos(roll), math.sin(roll)
    rot = pts5 @ np.array([[c, s], [-s, c]])
    eye_y, nose_y, mouth_y = rot[:2, 1].mean(), rot[2, 1], rot[3:, 1].mean()
    if not (eye_y < nose_y < mouth_y):
        blocking.append("LANDMARK_TOPOLOGY_INVALID")
        score *= 0.2
    iod = float(np.linalg.norm(pts5[0] - pts5[1]))
    if iod < 4.0:
        warnings.append("EYES_NEARLY_COINCIDENT_PROFILE_OR_DEGENERATE")
    estimated = estimate_pose_from_landmarks(pts5)
    stored = view_pose(view)
    if abs(estimated[0] - stored.yaw) > POSE_MISMATCH_YAW_DEG:
        blocking.append("POSE_LANDMARK_MISMATCH")
        score *= 0.5
    elif abs(estimated[0] - stored.yaw) > POSE_MISMATCH_YAW_DEG / 2:
        warnings.append("POSE_LANDMARK_DISAGREE")
    if abs(stored.yaw) > MAX_ABS_YAW_DEG:
        blocking.append("POSE_OUT_OF_RANGE")
    return float(np.clip(score, 0.0, 1.0)), blocking, warnings, layout


# --------------------------------------------------------------------------- identity consistency
def _identity_clusters(embeddings: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Single-link clusters at cosine similarity >= IDENTITY_LINK_SIMILARITY; returns (label per view, stats)."""
    n = len(embeddings)
    unit = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9)
    sim = unit @ unit.T
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= IDENTITY_LINK_SIMILARITY:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(n)]
    order = sorted(set(roots), key=lambda r: (-roots.count(r), r))
    labels = np.array([order.index(r) for r in roots])
    off = sim[~np.eye(n, dtype=bool)] if n > 1 else np.array([1.0])
    stats = {"cluster_sizes": [int((labels == k).sum()) for k in range(len(order))],
             "min_pairwise_similarity": float(off.min()), "mean_pairwise_similarity": float(off.mean()),
             "link_similarity": IDENTITY_LINK_SIMILARITY}
    return labels, stats


# --------------------------------------------------------------------------- public API
def analyze_faceset(
    document: FaceSetDocument,
    *,
    occlusion_masks: Mapping[int, np.ndarray] | None = None,
) -> FaceSetCoverageReport:
    """Measure every view of a FaceSet and decide what a reconstruction may use."""
    if not document.views:
        raise DomainError("FACESET_EMPTY", "FaceSet contains no views to analyse", 422)
    views = list(document.views)
    n = len(views)
    embeddings = np.stack([np.asarray(v.embedding, np.float64) for v in views])
    labels, id_stats = _identity_clusters(embeddings)
    sizes = id_stats["cluster_sizes"]
    inconsistent = n >= 2 and (sizes[0] == 1 or (len(sizes) > 1 and sizes[1] >= 2))
    main = np.where(labels == 0)[0]
    centroid = embeddings[main].mean(axis=0)
    centroid /= max(np.linalg.norm(centroid), 1e-9)

    decoded = [_decode(v) for v in views]
    quality_rank = np.argsort([-v.quality.score for v in views], kind="stable")
    sha = [hashlib.sha256(v.image_bytes).hexdigest() for v in views]
    hashes = [(_dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) if img is not None else None) for img in decoded]

    duplicate_of: dict[int, int] = {}
    kept: list[int] = []
    for i in quality_rank:
        i = int(i)
        twin = None
        for j in kept:
            same_bytes = sha[i] == sha[j]
            vi, vj = view_pose(views[i]), view_pose(views[j])
            # A coarse perceptual hash cannot tell two poses of one person on one background apart, so a hash match
            # only counts as a duplicate when the pose is also (nearly) the same. Identical bytes always count.
            same_look = (hashes[i] is not None and hashes[j] is not None
                         and bin(hashes[i] ^ hashes[j]).count("1") <= DUPLICATE_HASH_DISTANCE
                         and abs(vi.yaw - vj.yaw) <= DUPLICATE_POSE_DEG and abs(vi.pitch - vj.pitch) <= DUPLICATE_POSE_DEG)
            same_pose_id = (abs(vi.yaw - vj.yaw) < 1.5 and abs(vi.pitch - vj.pitch) < 1.5
                            and float(embeddings[i] @ embeddings[j]) > 0.995 * float(np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])))
            if same_bytes or same_look or same_pose_id:
                twin = j
                break
        if twin is None:
            kept.append(i)
        else:
            duplicate_of[i] = twin

    per_view: list[ViewQuality] = []
    for i, view in enumerate(views):
        img = decoded[i]
        reasons: list[str] = []
        warns: list[str] = []
        pose = view_pose(view)
        if img is None:
            per_view.append(ViewQuality(
                index=view.index, yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll, landmark_layout="5pt", sharpness=0.0,
                mean_luma=0.0, clipped_dark=0.0, clipped_bright=0.0, exposure_score=0.0, landmark_score=0.0,
                border_truncation=0.0, identity_distance=1.0, identity_cluster=int(labels[i]), usable=False,
                exclusion_reasons=["IMAGE_UNDECODABLE"]))
            continue
        h, w = img.shape[:2]
        roi, truncation, face_px = _face_roi(img, view.bbox)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sharp = float(min(1.0, cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0))
        luma = float(gray.mean())
        dark, bright = float((gray < 8).mean()), float((gray > 247).mean())
        exposure = float(np.clip(1.0 - 2.0 * (dark + bright) - 0.5 * abs(luma - 128.0) / 128.0, 0.0, 1.0))
        lm_score, lm_block, lm_warn, layout = _landmark_checks(view, w, h)
        reasons += lm_block
        warns += lm_warn
        if sharp < MIN_SHARPNESS:
            reasons.append("TOO_BLURRY")
        elif sharp < 0.1:
            warns.append("LOW_SHARPNESS")
        if dark + bright > MAX_CLIPPED_FRACTION or luma < 25 or luma > 235:
            reasons.append("BAD_EXPOSURE")
        elif exposure < 0.5:
            warns.append("MARGINAL_EXPOSURE")
        if face_px < MIN_FACE_PIXELS:
            reasons.append("FACE_TOO_SMALL")
        if truncation > 0.15:
            reasons.append("FACE_TRUNCATED_BY_IMAGE_EDGE")
        elif truncation > 0.02:
            warns.append("FACE_PARTLY_OUTSIDE_IMAGE")
        occ = None
        if occlusion_masks is not None and view.index in occlusion_masks:
            m = np.asarray(occlusion_masks[view.index])
            x1, y1, x2, y2 = (int(round(v)) for v in view.bbox)
            region = m[max(y1, 0):max(y2, 1), max(x1, 0):max(x2, 1)]
            occ = float(1.0 - (region > 128).mean()) if region.size else None
            if occ is not None and occ > 0.35:
                reasons.append("FACE_HEAVILY_OCCLUDED")
            elif occ is not None and occ > 0.1:
                warns.append("FACE_PARTLY_OCCLUDED")
        if i in duplicate_of:
            reasons.append("DUPLICATE_IMAGE")
        if inconsistent:
            reasons.append("IDENTITY_SET_INCONSISTENT")
        elif labels[i] != 0:
            reasons.append("IDENTITY_OUTLIER")
        dist = float(1.0 - (embeddings[i] / max(np.linalg.norm(embeddings[i]), 1e-9)) @ centroid)
        per_view.append(ViewQuality(
            index=view.index, yaw=pose.yaw, pitch=pose.pitch, roll=pose.roll, landmark_layout=layout, sharpness=sharp,
            mean_luma=luma, clipped_dark=dark, clipped_bright=bright, exposure_score=exposure, landmark_score=lm_score,
            border_truncation=truncation, occlusion_fraction=occ, identity_distance=dist, identity_cluster=int(labels[i]),
            duplicate_of=views[duplicate_of[i]].index if i in duplicate_of else None, usable=not reasons,
            exclusion_reasons=reasons, warnings=warns))

    usable =[q for q in per_view if q.usable]
    yaw_cov: dict[str, list[int]] = {name: [] for name, _, _ in YAW_TARGETS}
    pitch_cov: dict[str, list[int]] = {name: [] for name, _ in PITCH_TARGETS}
    for q in usable:
        name, target, _ = min(YAW_TARGETS, key=lambda t: abs(q.yaw - t[1]))
        if abs(q.yaw - target) <= YAW_TOLERANCE_DEG:
            yaw_cov[name].append(q.index)
        pname, ptarget = min(PITCH_TARGETS, key=lambda t: abs(q.pitch - t[1]))
        if abs(q.pitch - ptarget) <= PITCH_TOLERANCE_DEG:
            pitch_cov[pname].append(q.index)

    yaws = [q.yaw for q in usable]
    pitches = [q.pitch for q in usable]
    yaw_span = float(max(yaws) - min(yaws)) if yaws else 0.0
    pitch_span = float(max(pitches) - min(pitches)) if pitches else 0.0

    missing: list[str] = []
    requests: list[str] = []
    guidance = {
        "frontal": "Add a sharp, evenly lit frontal portrait (looking straight at the camera).",
        "three_quarter_image_left": "Add a three-quarter portrait with the head turned toward the image LEFT (about -45 deg yaw).",
        "three_quarter_image_right": "Add a three-quarter portrait with the head turned toward the image RIGHT (about +45 deg yaw).",
        "profile_image_left": "Add a side profile with the nose pointing toward the image LEFT (about -75 deg yaw) to capture nose, jaw and ear depth.",
        "profile_image_right": "Add a side profile with the nose pointing toward the image RIGHT (about +75 deg yaw) to capture nose, jaw and ear depth.",
    }
    order = {"required": 0, "recommended": 1}
    for name, _, need in sorted(YAW_TARGETS, key=lambda t: order[t[2]]):
        if not yaw_cov[name]:
            missing.append(name)
            requests.append(f"[{need}] {guidance[name]}")
    if usable:
        for pname, _ in PITCH_TARGETS:
            if pname != "level" and not pitch_cov[pname]:
                missing.append(pname)
                requests.append(f"[optional] Add a portrait {pname.replace('_', ' ')} (about {dict(PITCH_TARGETS)[pname]:+.0f} deg pitch) for vertical coverage.")

    openness = [float(v.expression.get("openness", 0.0)) for v in views if "openness" in v.expression]
    expression_spread = {"openness_min": min(openness) if openness else 0.0, "openness_max": max(openness) if openness else 0.0}
    if openness and expression_spread["openness_max"] - expression_spread["openness_min"] < 0.1 and len(usable) > 1:
        requests.append("[optional] All views share one expression; add a neutral/closed-mouth and an open-mouth view to separate identity from expression.")

    if not usable:
        readiness = "blocked_inconsistent_identity" if inconsistent else "insufficient"
    elif yaw_span < 25.0:
        readiness = "front_only"
    elif len(usable) >= 3 and yaw_cov["frontal"] and yaw_cov["three_quarter_image_left"] and yaw_cov["three_quarter_image_right"]:
        readiness = "full_coverage"
    else:
        readiness = "partial_coverage"
    if inconsistent and readiness != "blocked_inconsistent_identity":
        readiness = "blocked_inconsistent_identity"

    identity = {"status": "inconsistent" if inconsistent else ("single_view" if n == 1 else "consistent"), **id_stats,
                "excluded_outliers": [q.index for q in per_view if "IDENTITY_OUTLIER" in q.exclusion_reasons]}
    if inconsistent:
        requests.insert(0, "[blocking] The FaceSet appears to contain more than one person. Remove the images of other people "
                           "or split them into separate FaceSets; no identity asset was built from a mixed set.")
    groups: dict[int, list[int]] = {}
    for dup, orig in duplicate_of.items():
        groups.setdefault(orig, [views[orig].index]).append(views[dup].index)

    return FaceSetCoverageReport(
        faceset_id=document.faceset_id, view_count=n, usable_count=len(usable), usable_indices=[q.index for q in usable],
        readiness=readiness, yaw_coverage=yaw_cov, pitch_coverage=pitch_cov,
        yaw_range_deg=(float(min(yaws)), float(max(yaws))) if yaws else None, yaw_span_deg=yaw_span,
        pitch_span_deg=pitch_span, missing=missing, requests=requests, identity=identity,
        duplicate_groups=[sorted(g) for g in groups.values()], expression_spread=expression_spread,
        capabilities={
            "landmarks": "68pt" if any(v.landmarks_68 is not None for v in views) else "5pt (SCRFD)",
            "pose": "5-point scaled-orthographic estimate, typical error a few degrees (not metric)",
            "occlusion": "border truncation + optional supplied mask; semantic occluder model: unavailable",
            "identity_embedding": "stored ArcFace embeddings",
        },
        views=per_view)
