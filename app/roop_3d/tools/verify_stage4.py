"""Stage 4 gate: three real head angles through the volumetric MakeHuman render and the semantic compositor.

Run from app/:
    roop_3d/.venv/Scripts/python.exe -B -m roop_3d.tools.verify_stage4 [--input-dir DIR] [--output-dir DIR]
        [--max-side 960] [--texture-size 1024] [--restoration-weight 0.35] [--provider auto]

Real execution only: no mocks, no replayed renders, no silent fallbacks.

* The ``harjot`` FaceSet goes through the production loader (``resolve_faceset`` + ``load_faceset`` + ``LandmarkEngine``:
  trusted 1k3d68 dense landmarks per source view).
* One ``ReferencePipeline`` is prepared with the requested UV ``texture_size`` / ``restoration_weight``.
* For each of three head-angle classes (frontal, ~45 deg three-quarter, steep profile) candidate frames are *proposed* from the
  stage-3 detector cache (a hint list only; its poses are never trusted), then each candidate is decoded fresh from its video,
  downscaled to ``--max-side`` and sent through the real stack: ``track_frame`` (SCRFD, ArcFace, 1k3d68 dense landmarks) ->
  ``infer_frame`` (MakeHuman dense68 PnP, perspective Z-buffer raster, SH relighting) -> ``composite_frame`` (BiSeNet ROI + XSeg +
  occluder mattes, skin photometry, Laplacian blend).
* Candidates are ranked by the *solved* yaw of the renderer's dense68 pose, never by the cached five-point detector estimate
  (which saturates at +-90 deg on rolled heads). A class passes only when that solved yaw lies inside its window and the
  roll-independent turn angle of the solved rotation agrees with it.
* Every verification shot is an independent hard cut: the tracker is reset, ``scene_cut`` is set so the renderer forgets pose
  history, and the target (the largest face) carries the canonical track id 1, which is how one renderer (one identity fit and UV
  build) is reused for all three frames - the same labelling the locked-target binder gives the single target in production.

Outputs (paths are relative to app/; the output directory must stay inside this checkout):
    roop_3d/.runtime/stage4_verification.json
    roop_3d/outputs/stage4/{frontal,three_quarter,profile}_comparison.png   Original | wireframe + hair-excluded alpha | raw 3D | final

Exit code 0 only when all three classes were solved and every gate check passed; 1 on a gate failure; 2 on an aborted run.
The gate fails on: no replacement, a skipped or fallback composite, a pose not solved from the real dense68 landmarks, a flat or
inconsistent depth/normal volume, a mesh below the guard's head-vertex floor, UV evidence that is missing, any edit of a forbidden
pixel (hair, background, neck, clothing, occluders, outside the mesh), a no-op composite, or an implausibly dark/bright raw render.
Edge, seam and forehead/jaw coverage metrics are INDICATORS ONLY: they never certify seam-free blending. Open the three PNG
strips and inspect them - a passing gate does not certify visual quality.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Mapping

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root
from roop_3d.core.errors import DomainError
from roop_3d.core.pose import rotation_from_euler
from roop_3d.core.storage import atomic_bytes, atomic_json
from roop_3d.core.true3d_guard import DEFAULT_POLICY, FRAME_PIPELINE_STAGES
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.faceset.locator import resolve_faceset
from roop_3d.faceset_loader import LandmarkEngine, load_faceset
from roop_3d.media.frame_service import VideoFrameService
from roop_3d.models.catalog import ModelCatalog
from roop_3d.reconstruction.stage4_head import HeadLayer, VolumetricHeadRenderer
from roop_3d.tools.stage3_scan import CACHE, natural

RECEIPT = "roop_3d/.runtime/stage4_verification.json"
DEFAULT_OUTPUT = "roop_3d/outputs/stage4"
DEFAULT_INPUT = Path(r"G:\pinokio\roop-keep\single")
CANONICAL_TRACK_ID = 1                  # "Face 1": the label of the single locked target in every verification shot
PNP, HEURISTIC = "tracker_dense_pnp", "five_point_heuristic"

# ---- candidate proposal from the stage-3 detector cache (hints only; every candidate is re-detected fresh) ----
TARGET_SCORE = 0.5                      # stage-3 target proxy: the largest detection scoring >= 0.5
MIN_CANDIDATE_SCORE = 0.6               # the proposed row's own detection score (the single-face test still uses TARGET_SCORE)
MAX_CACHED_ABS_ROLL = 30.0              # in-plane rolled heads saturate / mis-estimate the cached heuristic yaw
BORDER_FRACTION = 0.02                  # the face box must keep this fraction of the frame away from every border
MIN_FACE_PX = 72.0                      # shorter face-box side at the working resolution
NMS_FRAMES = 12                         # candidates of one video are at least this many frames apart
HEURISTIC_PER_CLASS = 2                 # cached rows without a tracker PnP pose: at most this many candidates per class
DEEP_LEAD_IN = 20                       # retry lead-in when the short one found no confident target (arms detector recovery)
# A composite refusal that merely reflects the frame (nothing of the face is visible after the semantic exclusions). Every other
# skip code is a pipeline fault and fails the case instead of moving on to another frame.
DATA_DEPENDENT_SKIPS = frozenset({"TRUE3D_SEMANTIC_MASK_EMPTY"})

# ---- gate thresholds (heuristic floors: a flat plane, an empty or black raster and a no-op composite sit far below them) ----
MIN_THICKNESS_RATIO = 0.12              # smallest / largest principal std of the visible surface point cloud
MAX_DEPTH_NORMAL_MEDIAN_DEG = 22.0      # angle between normals derived from the depth map and the rasterized normals
MIN_DEPTH_NORMAL_WITHIN_35 = 0.75
MIN_CAMERA_FACING_FRACTION = 0.95
MIN_UNIT_NORMAL_FRACTION = 0.999
MIN_OBSERVED_UV_RATIO = 0.02
MIN_CHANGED_FRACTION = 0.02             # of the alpha>0.5 support, pixels the composite must really change
MIN_MEAN_ABS_DELTA = 0.5                # 8-bit levels over the alpha>0.5 support
RAW_LUMA_RATIO = (0.45, 2.2)            # raw 3D mean luma / original mean luma over the replacement support
FINAL_LUMA_RATIO = (0.65, 1.5)
MAX_RAW_BLACK_FRACTION = 0.20           # fraction of raw support pixels below luma 12 (an empty / unlit atlas)
MIN_ALPHA_SUPPORT_PIXELS = 256

PANEL_LONG_SIDE = 560
FONT = cv2.FONT_HERSHEY_SIMPLEX


class ContractError(RuntimeError):
    """The pipeline broke a contract this tool relies on; any verdict would be meaningless, so the run aborts."""


class RendererBuildError(RuntimeError):
    """The per-track renderer could not be built (identity fit / UV synthesis); other frames cannot change that."""

    def __init__(self, error: DomainError):
        super().__init__(f"{error.code}: {error.message}")
        self.error = error


@dataclass(frozen=True)
class ClassSpec:
    name: str                           # output file stem
    label: str
    target: float                       # degrees of |yaw|
    accept: tuple[float, float]         # hard window on the solved |yaw|
    ideal: tuple[float, float]          # inside this band no further candidate is evaluated
    coarse: tuple[float, float]         # window on the cached |yaw|, used only to propose candidates


def class_specs(profile_min: float) -> tuple[ClassSpec, ...]:
    return (
        ClassSpec("frontal", "frontal", 0.0, (0.0, 15.0), (0.0, 8.0), (0.0, 20.0)),
        ClassSpec("three_quarter", "three-quarter (~45 deg)", 45.0, (35.0, 55.0), (40.0, 50.0), (28.0, 62.0)),
        ClassSpec("profile", "steep profile", max(80.0, profile_min + 5.0), (profile_min, 95.0),
                  (profile_min + 5.0, 95.0), (profile_min - 15.0, 120.0)),
    )


@dataclass(frozen=True)
class Candidate:
    video: Path
    video_order: int
    frame: int
    timestamp: float
    box: tuple[float, float, float, float]
    source_size: tuple[int, int]
    detection_score: float
    cached_yaw: float
    cached_pose_source: str
    face_px: float
    quality: float

    def summary(self) -> dict[str, Any]:
        return {"video": self.video.name, "frame": self.frame, "timestamp_s": self.timestamp, "source_size": list(self.source_size),
                "box_xyxy": list(self.box), "detection_score": self.detection_score, "cached_yaw_deg": self.cached_yaw,
                "cached_pose_source": self.cached_pose_source, "face_min_side_px_at_working_size": self.face_px,
                "quality": self.quality}


@dataclass
class Attempt:
    spec: str
    candidate: Candidate
    order: int
    status: str = "pending"             # rendered | rejected
    stage: str = "decode"
    error: dict[str, Any] | None = None
    seconds: float = 0.0
    built_renderer: bool = False
    tracked_frames: int = 0
    decoded_frame: int | None = None
    timestamp: float | None = None
    image: np.ndarray | None = None
    target: Any = None
    result: Any = None
    layer: HeadLayer | None = None
    tracker_pose: dict[str, Any] | None = None
    solved: dict[str, Any] | None = None
    in_window: bool = False
    consistent: bool = False
    distance: float = math.inf
    note: str | None = None

    def drop_arrays(self) -> None:
        self.image = self.target = self.result = self.layer = None

    def summary(self) -> dict[str, Any]:
        return {**self.candidate.summary(), "status": self.status, "stage": self.stage, "error": self.error,
                "decoded_frame": self.decoded_frame, "timestamp_decoded_s": self.timestamp,
                "frames_tracked_including_lead_in": self.tracked_frames, "renderer_built_in_this_call": self.built_renderer,
                "tracker_pose": self.tracker_pose, "solved_pose": self.solved,
                "solved_yaw_in_class_window": self.in_window, "turn_angle_consistent": self.consistent,
                "abs_yaw_distance_to_class_target_deg": self.distance if math.isfinite(self.distance) else None,
                "seconds": round(self.seconds, 2), "note": self.note}


@dataclass
class Composite:
    final: np.ndarray | None = None
    report: Any = None
    packet: Any = None
    alpha: np.ndarray | None = None
    error: dict[str, Any] | None = None

    def refusal_code(self) -> str | None:
        if self.error is not None:
            return str(self.error["code"])
        if self.report is None:
            return "NO_FRAME_REPORT"
        if self.report.skipped:
            return str(self.report.skipped[0].get("code", "SKIPPED"))
        return None


# ------------------------------------------------------------------------------------------------------ small helpers
def jsonable(value: Any) -> Any:
    """Strict-JSON view of numpy / path / mapping values; non-finite floats become null (the file is written allow_nan=False)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, str):
        return value
    return repr(value)


def stats(values: Any) -> dict[str, Any]:
    data = np.asarray(values, np.float64).reshape(-1)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return {"count": 0}
    p05, median, p95 = np.percentile(data, (5, 50, 95))
    return {"count": int(data.size), "min": float(data.min()), "p05": float(p05), "median": float(median),
            "mean": float(data.mean()), "p95": float(p95), "max": float(data.max())}


def error_record(error: BaseException) -> dict[str, Any]:
    if isinstance(error, DomainError):
        return {"code": error.code, "message": error.message, "status": error.status}
    return {"code": type(error).__name__, "message": str(error)}


def luma(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)


def fit_to_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    """The working frame: the decoded frame itself, or an INTER_AREA downscale so that its long side is ``max_side``."""
    height, width = image.shape[:2]
    scale = min(1.0, max_side / float(max(height, width)))
    if scale >= 1.0:
        return np.ascontiguousarray(image.copy())
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.ascontiguousarray(cv2.resize(image, size, interpolation=cv2.INTER_AREA))


def owned_relative(root: Path, path: Path) -> str:
    resolved = (path if path.is_absolute() else root / path).resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        raise SystemExit(f"--output-dir must stay inside the application checkout {root}: {resolved}") from None


def environment() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(),
                            "numpy": np.__version__, "opencv": cv2.__version__}
    try:
        import onnxruntime as ort
        info["onnxruntime"] = ort.__version__
        info["onnxruntime_available_providers"] = ort.get_available_providers()
    except Exception as error:                                   # reporting only; the models report their own providers
        info["onnxruntime"] = f"unavailable: {type(error).__name__}: {error}"
    return info


# ------------------------------------------------------------------------------------------------------------ arguments
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="directory of .mp4 target videos")
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT),
                        help="PNG strip directory (relative paths are relative to app/; must stay inside the checkout)")
    parser.add_argument("--faceset", default="harjot", help="source FaceSet id (production resolver)")
    parser.add_argument("--video", nargs="*", help="video file names to consider (default: every .mp4 of --input-dir)")
    parser.add_argument("--max-side", type=int, default=960, help="long side of the working frame (default 960)")
    parser.add_argument("--texture-size", type=int, default=1024, help="canonical UV atlas edge in texels (default 1024)")
    parser.add_argument("--restoration-weight", type=float, default=0.35, help="UV restoration weight 0..1 (default 0.35)")
    parser.add_argument("--candidates", type=int, default=6, help="candidate frames evaluated per class at most (default 6)")
    parser.add_argument("--lead-in", type=int, default=3,
                        help="real frames tracked before each candidate so the tracker's staged recovery is armed (default 3)")
    parser.add_argument("--detection-threshold", type=float, default=0.4, help="tracker detection threshold (stage-3 scan default)")
    parser.add_argument("--profile-min-yaw", type=float, default=70.0, help="smallest solved |yaw| accepted as steep profile")
    parser.add_argument("--provider", choices=("auto", "CPUExecutionProvider", "CUDAExecutionProvider", "DmlExecutionProvider"),
                        default="auto")
    parser.add_argument("--profile", choices=("cpu", "6gb", "12gb"),
                        help="hardware profile (default: 6gb for CUDA, cpu otherwise, as verify_stage1)")
    args = parser.parse_args(argv)
    if args.max_side < 256:
        parser.error("--max-side must be at least 256")
    if not 64 <= args.texture_size <= 4096:
        parser.error("--texture-size must be within 64..4096")
    if not 0.0 <= args.restoration_weight <= 1.0:
        parser.error("--restoration-weight must be within 0..1")
    if args.candidates < 1 or args.lead_in < 0:
        parser.error("--candidates must be >= 1 and --lead-in >= 0")
    if not 0.1 <= args.detection_threshold <= 0.99:
        parser.error("--detection-threshold must be within 0.1..0.99")
    if not 60.0 <= args.profile_min_yaw <= 88.0:
        parser.error("--profile-min-yaw must be within 60..88")
    return args


def runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    profile = args.profile or ("6gb" if args.provider == "CUDAExecutionProvider" else "cpu")
    return RuntimeConfig(provider=args.provider, hardware_profile=profile)


# ----------------------------------------------------------------------------------------------- candidate proposal
def rank_pool(rows: list[Candidate], spec: ClassSpec, anchor: int | None, limit: int) -> list[Candidate]:
    """Deterministic order: anchor video first, tracker-PnP rows before heuristic rows, nearest cached yaw bucket (5 deg), then
    detection quality; frames of one video closer than NMS_FRAMES are suppressed."""
    def key(candidate: Candidate):
        distance = abs(abs(candidate.cached_yaw) - spec.target)
        return (0 if candidate.video_order == anchor else 1, 0 if candidate.cached_pose_source == PNP else 1,
                int(round(distance / 5.0)), -candidate.quality, candidate.video_order, candidate.frame)

    chosen: list[Candidate] = []
    heuristic = 0
    for candidate in sorted(rows, key=key):
        if any(other.video_order == candidate.video_order and abs(other.frame - candidate.frame) < NMS_FRAMES for other in chosen):
            continue
        if candidate.cached_pose_source != PNP:
            if heuristic >= HEURISTIC_PER_CLASS:
                continue
            heuristic += 1
        chosen.append(candidate)
        if len(chosen) >= limit:
            break
    return chosen


def choose_anchor(rows: dict[str, list[Candidate]], specs: tuple[ClassSpec, ...]) -> int | None:
    """The video that offers tracker-PnP candidates for every class (so the three strips show one subject), best total quality."""
    targets = {spec.name: spec.target for spec in specs}
    coverage: dict[int, dict[str, float]] = {}
    for name, candidates in rows.items():
        for candidate in candidates:
            if candidate.cached_pose_source != PNP:
                continue
            closeness = 1.0 - min(1.0, abs(abs(candidate.cached_yaw) - targets[name]) / 40.0)
            slot = coverage.setdefault(candidate.video_order, {})
            slot[name] = max(slot.get(name, 0.0), candidate.quality * closeness)
    complete = [order for order, slot in coverage.items() if len(slot) == len(rows)]
    return max(complete, key=lambda order: (sum(coverage[order].values()), -order)) if complete else None


def propose_candidates(root: Path, videos: list[Path], specs: tuple[ClassSpec, ...], *, max_side: int,
                       limit: int) -> tuple[dict[str, list[Candidate]], dict[str, Any]]:
    """Coarse per-class candidate frames from the stage-3 cache. Only single-confident-face, uncut frames whose face box is
    inside the frame and large enough at the working resolution qualify. Cached rows whose pose came from the tracker's dense
    PnP are preferred; the five-point heuristic rows (which saturate on rolled heads) are a capped last resort. Nothing here
    decides the class of a frame: the solved yaw of the fresh render does."""
    rows: dict[str, list[Candidate]] = {spec.name: [] for spec in specs}
    considered: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for order, video in enumerate(videos):
        cache = root / CACHE / f"{video.stem}.npz"
        if not cache.is_file():
            skipped.append({"video": video.name, "reason": f"missing {CACHE}/{video.stem}.npz (run roop_3d.tools.stage3_scan)"})
            continue
        with np.load(cache) as data:
            frames = np.asarray(data["frame"], np.int64)
            boxes = np.asarray(data["box"], np.float64).reshape(-1, 4)
            scores = np.asarray(data["score"], np.float64)
            poses = np.asarray(data["pose"], np.float64).reshape(-1, 3)
            translations = np.asarray(data["t3d"], np.float64).reshape(-1, 3)
            size = np.asarray(data["size"], np.int64)
            times = np.asarray(data["frame_time"], np.float64)
            cuts = np.asarray(data["frame_cut"], bool)
        width, height = int(size[0]), int(size[1])
        if len(frames) == 0 or width <= 0 or height <= 0:
            skipped.append({"video": video.name, "reason": "the stage-3 cache holds no detections"})
            continue
        confident = scores >= TARGET_SCORE
        faces_per_frame = np.bincount(frames[confident], minlength=int(frames.max()) + 1)
        scale = min(1.0, max_side / float(max(width, height)))
        face_px = scale * np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
        inside = ((boxes[:, 0] >= BORDER_FRACTION * width) & (boxes[:, 2] <= (1.0 - BORDER_FRACTION) * width)
                  & (boxes[:, 1] >= BORDER_FRACTION * height) & (boxes[:, 3] <= (1.0 - BORDER_FRACTION) * height))
        timed = frames < len(times)
        cut_frame = np.zeros(len(frames), bool)
        cut_known = frames < len(cuts)
        cut_frame[cut_known] = cuts[frames[cut_known]]
        keep = (confident & (faces_per_frame[frames] == 1) & inside & timed & ~cut_frame & (scores >= MIN_CANDIDATE_SCORE)
                & (face_px >= MIN_FACE_PX) & np.isfinite(poses[:, 0]) & (np.abs(poses[:, 2]) < MAX_CACHED_ABS_ROLL))
        measured = np.isfinite(translations).all(axis=1)
        for row in np.nonzero(keep)[0]:
            yaw, frame = float(poses[row, 0]), int(frames[row])
            candidate = Candidate(
                video=video, video_order=order, frame=frame, timestamp=float(times[frame]),
                box=tuple(float(value) for value in boxes[row]), source_size=(width, height),
                detection_score=float(scores[row]), cached_yaw=yaw,
                cached_pose_source=PNP if measured[row] else HEURISTIC, face_px=float(face_px[row]),
                quality=float(scores[row] * min(1.0, face_px[row] / 160.0)))
            for spec in specs:
                if spec.coarse[0] <= abs(yaw) <= spec.coarse[1]:
                    rows[spec.name].append(candidate)
        considered.append({"video": video.name, "source_size": [width, height],
                           "single_confident_face_frames": int(keep.sum()),
                           "of_which_tracker_dense_pnp": int((keep & measured).sum())})
    anchor = choose_anchor(rows, specs)
    pools = {spec.name: rank_pool(rows[spec.name], spec, anchor, limit) for spec in specs}
    info = {
        "policy": ("stage-3 cache rows -> single confident face, uncut, detection score >= "
                   f"{MIN_CANDIDATE_SCORE:g}, cached |roll| < {MAX_CACHED_ABS_ROLL:g} deg, face box inside the frame and >= "
                   f"{MIN_FACE_PX:g}px at the working size; tracker-PnP rows first (anchor video preferred so the strips show one "
                   f"subject), at most {HEURISTIC_PER_CLASS} five-point-heuristic rows per class; frames >= {NMS_FRAMES} apart. "
                   "The class of a frame is decided by the solved yaw of the fresh dense68 render, never by these hints."),
        "cache_dir": CACHE, "videos_with_cache": considered, "videos_without_cache": skipped,
        "anchor_video": None if anchor is None else videos[anchor].name,
        "proposed": {spec.name: [candidate.summary() for candidate in pools[spec.name]] for spec in specs},
    }
    return pools, info


# ------------------------------------------------------------------------------------------------ frame driving
class FrameReader:
    """Random access to real video frames through the production ``VideoFrameService`` (colour, rotation, HDR handled there),
    returned at the working resolution."""

    def __init__(self, max_side: int):
        self.service = VideoFrameService(lambda value: Path(value))
        self.max_side = max_side

    def read(self, video: Path, index: int):
        decoded = self.service.read(str(video), frame_index=index)
        return fit_to_max_side(decoded.image, self.max_side), decoded

    def close(self) -> None:
        self.service.close()


def pick_target(tracked, threshold: float):
    """The largest confident face of the frame (the stage-3 target proxy)."""
    faces = [face for face in tracked.faces if face.swap_allowed and float(face.observation.score) >= threshold]
    if not faces:
        return None

    def key(face):
        x1, y1, x2, y2 = (float(value) for value in face.observation.box)
        return ((x2 - x1) * (y2 - y1), float(face.observation.score), -int(face.track_id))
    return max(faces, key=key)


def tracker_pose(observation) -> dict[str, Any]:
    pose6 = observation.pose6dof
    return {"source": observation.pose_source, "fallback_reason": observation.pose_fallback_reason,
            "detector_score": float(observation.score),
            "dense_pnp_yaw_deg": None if pose6 is None else float(pose6.yaw_deg),
            "dense_pnp_pitch_deg": None if pose6 is None else float(pose6.pitch_deg),
            "dense_pnp_roll_deg": None if pose6 is None else float(pose6.roll_deg),
            "dense_pnp_reprojection_rms_px": None if pose6 is None else float(pose6.reproj_rms_px),
            "legacy_pose_tuple_pitch_down_positive": [float(value) for value in observation.pose]}


def solved_pose(layer: HeadLayer) -> dict[str, Any]:
    """The renderer's own solved pose (MakeHuman dense68 PnP) plus the roll-independent turn angle of its rotation."""
    keys = ("yaw_deg", "pitch_deg", "roll_deg", "source", "reprojection_rms_px", "inliers", "dense_landmark_count")
    pose = layer.diagnostics.get("pose") if isinstance(layer.diagnostics, Mapping) else None
    if not isinstance(pose, Mapping) or any(key not in pose for key in keys):
        raise ContractError(f"HeadLayer.diagnostics['pose'] must carry {keys}; got "
                            f"{sorted(pose) if isinstance(pose, Mapping) else repr(pose)}")
    yaw, pitch, roll = (float(pose[key]) for key in ("yaw_deg", "pitch_deg", "roll_deg"))
    rotation = rotation_from_euler(yaw, pitch, roll)
    off_axis = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2, 2])))))
    return {"yaw_deg": yaw, "pitch_deg": pitch, "roll_deg": roll, "off_axis_deg": off_axis, "source": pose["source"],
            "dense_failure": pose.get("dense_failure"), "reprojection_rms_px": float(pose["reprojection_rms_px"]),
            "inliers": int(pose["inliers"]), "dense_landmark_count": int(pose["dense_landmark_count"]),
            "expression": pose.get("expression"), "translation_m": pose.get("translation_m"),
            "camera_intrinsics": pose.get("camera_intrinsics")}


def judge_pose(spec: ClassSpec, solved: Mapping[str, Any]) -> tuple[bool, bool, float]:
    """(solved |yaw| inside the class window, turn angle consistent with it, |yaw| distance to the class target)."""
    abs_yaw = abs(float(solved["yaw_deg"]))
    low, high = spec.accept
    in_window = low <= abs_yaw <= high
    consistent = max(0.0, low - 10.0) <= float(solved["off_axis_deg"]) <= high + 15.0
    return in_window, consistent, abs(abs_yaw - spec.target)


def is_ideal(spec: ClassSpec, attempt: Attempt) -> bool:
    return (attempt.status == "rendered" and attempt.consistent
            and spec.ideal[0] <= abs(float(attempt.solved["yaw_deg"])) <= spec.ideal[1])


def run_attempt(session: ReferencePipeline, reader: FrameReader, spec: ClassSpec, candidate: Candidate, order: int,
                args: argparse.Namespace) -> Attempt:
    """Decode a candidate and its lead-in fresh, track, and render it. A refusal the pipeline raises as DomainError for this frame
    (no face, dense PnP rejected, camera intersection, ...) rejects only this candidate; a failure to build the renderer or any
    other exception aborts the run."""
    attempt = Attempt(spec.name, candidate, order)
    began = time.perf_counter()
    try:
        tracker = session.tracker
        if tracker is None:
            raise ContractError("ReferencePipeline has no tracker after prepare_session")
        tracker.reset_tracking()
        tracked = image = decoded = None
        for index in range(max(0, candidate.frame - args.lead_in), candidate.frame + 1):
            attempt.stage = "decode"
            image, decoded = reader.read(candidate.video, index)
            session.set_frame_timestamp(float(decoded.timestamp_sec))
            attempt.stage = "track"
            tracked = session.track_frame(image)
            attempt.tracked_frames += 1
        attempt.image = image
        attempt.decoded_frame, attempt.timestamp = int(decoded.frame_index), float(decoded.timestamp_sec)
        attempt.stage = "target"
        target = pick_target(tracked, args.detection_threshold)
        if target is None:
            attempt.status = "rejected"
            attempt.error = {"code": "TARGET_FACE_NOT_FOUND",
                             "message": f"no face scoring >= {args.detection_threshold} ({len(tracked.faces)} tracked face(s))"}
            return attempt
        observation = target.observation
        attempt.tracker_pose = tracker_pose(observation)
        dense = observation.landmarks68
        if dense is None or np.asarray(dense).shape != (68, 3) or not np.isfinite(np.asarray(dense)).all():
            attempt.status = "rejected"
            attempt.error = {"code": "DENSE68_LANDMARKS_MISSING",
                             "message": "the tracker produced no finite 1k3d68 landmarks for the target face"}
            return attempt
        # An independent shot: the locked-target label for the single target, and a hard cut so the renderer's pose history
        # (One-Euro state of an earlier, unrelated frame) is forgotten.
        labelled = replace(target, track_id=CANONICAL_TRACK_ID)
        shot = replace(tracked, faces=(labelled,), active_ids=frozenset({CANONICAL_TRACK_ID}), scene_cut=True)
        attempt.target = labelled
        attempt.stage = "render"
        attempt.built_renderer = CANONICAL_TRACK_ID not in session.full_head_engines
        try:
            result = session.infer_frame(image, tracked=shot)
        except DomainError as error:
            if CANONICAL_TRACK_ID not in session.full_head_engines:
                raise RendererBuildError(error) from error
            attempt.status, attempt.error = "rejected", error_record(error)
            return attempt
        if len(result.replacements) != 1:
            attempt.status = "rejected"
            attempt.error = {"code": "NO_REPLACEMENT",
                             "message": f"infer_frame returned {len(result.replacements)} replacement(s) for the selected face; "
                                        f"withheld={[(item.track_id, item.reason) for item in result.withheld]}"}
            return attempt
        engine = session.full_head_engines.get(CANONICAL_TRACK_ID)
        if not isinstance(engine, VolumetricHeadRenderer) or engine.last_layer is None:
            raise ContractError("session.full_head_engines[1] must be a VolumetricHeadRenderer exposing last_layer after "
                                f"infer_frame; got {type(engine).__name__}")
        attempt.result, attempt.layer = result, engine.last_layer
        attempt.solved = solved_pose(attempt.layer)
        attempt.in_window, attempt.consistent, attempt.distance = judge_pose(spec, attempt.solved)
        attempt.status, attempt.stage = "rendered", "rendered"
        return attempt
    except DomainError as error:
        attempt.status, attempt.error = "rejected", error_record(error)
        return attempt
    finally:
        attempt.seconds = time.perf_counter() - began


def evaluate_class(session: ReferencePipeline, reader: FrameReader, spec: ClassSpec, pool: list[Candidate],
                   args: argparse.Namespace) -> list[Attempt]:
    attempts: list[Attempt] = []
    for order, candidate in enumerate(pool[:args.candidates]):
        attempt = run_attempt(session, reader, spec, candidate, order, args)
        if attempt.status == "rejected" and (attempt.error or {}).get("code") == "TARGET_FACE_NOT_FOUND" \
                and args.lead_in < DEEP_LEAD_IN:
            # The primary detector pass may not score a steep profile on its own; track a longer stretch first so the
            # tracker's staged recovery (armed when a track is alive) is active at the candidate, as in a real job.
            deep = argparse.Namespace(**{**vars(args), "lead_in": DEEP_LEAD_IN})
            attempt = run_attempt(session, reader, spec, candidate, order, deep)
        attempts.append(attempt)
        if attempt.status == "rendered":
            solved = attempt.solved
            verdict = "in window" if attempt.in_window and attempt.consistent else "outside window"
            print(f"    {candidate.video.name} f{candidate.frame}: solved yaw {solved['yaw_deg']:+.1f} pitch {solved['pitch_deg']:+.1f} "
                  f"roll {solved['roll_deg']:+.1f} (turn {solved['off_axis_deg']:.1f}) rms {solved['reprojection_rms_px']:.2f}px "
                  f"-> {verdict} [{attempt.seconds:.1f}s{', incl. renderer build' if attempt.built_renderer else ''}]", flush=True)
        else:
            error = attempt.error or {}
            print(f"    {candidate.video.name} f{candidate.frame}: rejected at {attempt.stage}: {error.get('code')}: "
                  f"{error.get('message')} [{attempt.seconds:.1f}s]", flush=True)
        if is_ideal(spec, attempt):
            break
    return attempts


# ------------------------------------------------------------------------------------------------------- compositing
def compositor_state(session: ReferencePipeline):
    """The compositor's per-frame evidence: its adapter (``last_packet``) and swap core (``last_alpha``) hang off the session."""
    processor = getattr(session, "_frame_processor", None)
    if processor is None or not hasattr(processor, "core"):
        raise ContractError("session.composite_frame did not leave a SessionFrameProcessor (_frame_processor) to read the packet "
                            "and alpha from")
    return processor


def run_composite(session: ReferencePipeline, attempt: Attempt) -> Composite:
    session.last_frame_report = None            # never read a stale report
    try:
        final = session.composite_frame(attempt.image, attempt.result, frame_index=attempt.decoded_frame,
                                        timestamp_seconds=attempt.timestamp)
    except DomainError as error:
        return Composite(error=error_record(error))
    report = session.last_frame_report
    processor = compositor_state(session)
    return Composite(final=final, report=report, packet=processor.last_packet, alpha=processor.core.last_alpha)


# ------------------------------------------------------------------------------------------------------ evidence
def structural_validation(validation: Any) -> tuple[bool, str]:
    if not isinstance(validation, Mapping):
        return False, "diagnostics carry no architecture_validation mapping"
    if "problems" in validation:
        return not validation["problems"], f"problems={validation['problems']}"
    for key in ("ok", "valid", "passed"):
        if key in validation:
            return bool(validation[key]), f"{key}={validation[key]}"
    return False, "architecture_validation has neither 'problems' nor an ok/valid/passed flag"


def depth_normal_agreement(depth: np.ndarray, normals: np.ndarray, geometric: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float) -> dict[str, Any]:
    """Compare the rasterized normals with normals re-derived from the depth map alone (central differences of the back-projected
    point cloud). A painted-on volume (flat depth, shaded normals) or a depth map without matching shading disagrees."""
    rows, cols = np.nonzero(geometric)
    y0, y1 = max(int(rows.min()) - 2, 0), min(int(rows.max()) + 3, depth.shape[0])
    x0, x1 = max(int(cols.min()) - 2, 0), min(int(cols.max()) + 3, depth.shape[1])
    mask = geometric[y0:y1, x0:x1]
    z = np.where(mask, depth[y0:y1, x0:x1], np.nan)
    u = np.arange(x0, x1, dtype=np.float64)[None, :]
    v = np.arange(y0, y1, dtype=np.float64)[:, None]
    cloud = np.stack(((u - cx) * z / fx, (v - cy) * z / fy, z), axis=-1)
    tangent_u = cloud[1:-1, 2:] - cloud[1:-1, :-2]
    tangent_v = cloud[2:, 1:-1] - cloud[:-2, 1:-1]
    cross = np.cross(tangent_u, tangent_v)
    length = np.linalg.norm(cross, axis=-1)
    centre = z[1:-1, 1:-1]
    # Skip silhouette bleed and occlusion jumps: a step above that of a surface slanted 80 deg is not a smooth surface.
    smooth = ((np.abs(tangent_u[..., 2]) <= 2.0 * (centre / fx) * 5.67) & (np.abs(tangent_v[..., 2]) <= 2.0 * (centre / fy) * 5.67))
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))[1:-1, 1:-1] > 0
    valid = interior & np.isfinite(length) & (length > 1e-12) & smooth
    count = int(valid.sum())
    if count < 200:
        return {"status": "insufficient_smooth_surface", "valid_pixels": count}
    # +z of the rasterized normals points toward the viewer while camera depth grows away from it, so the camera-frame normal
    # -(Tu x Tv) is re-expressed with a flipped z: (-cross_x, -cross_y, +cross_z).
    derived = np.stack((-cross[..., 0], -cross[..., 1], cross[..., 2]), axis=-1)[valid] / length[valid][:, None]
    rendered = normals[y0:y1, x0:x1][1:-1, 1:-1][valid]
    rendered = rendered / np.maximum(np.linalg.norm(rendered, axis=1, keepdims=True), 1e-12)
    angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", derived, rendered), -1.0, 1.0)))
    return {"status": "measured", "valid_pixels": count, "median_angle_deg": float(np.median(angle)),
            "p90_angle_deg": float(np.percentile(angle, 90)), "fraction_within_35deg": float(np.mean(angle <= 35.0))}


def volume_evidence(layer: HeadLayer) -> dict[str, Any]:
    """Depth / normal evidence that the layer is a real volume: finite perspective depth, unit camera-facing normals, a point cloud
    that is not a plane, and normals that agree with the depth map."""
    depth = np.asarray(layer.depth, np.float64)
    normals = np.asarray(layer.normals, np.float64)
    coverage = np.asarray(layer.coverage, np.float64)
    pose = layer.diagnostics.get("pose") if isinstance(layer.diagnostics, Mapping) else None
    intrinsics = pose.get("camera_intrinsics") if isinstance(pose, Mapping) else None
    if not isinstance(intrinsics, Mapping) or any(name not in intrinsics for name in ("fx", "fy", "cx", "cy")):
        raise ContractError("HeadLayer.diagnostics['pose']['camera_intrinsics'] must carry fx, fy, cx, cy")
    fx, fy, cx, cy = (float(intrinsics[name]) for name in ("fx", "fy", "cx", "cy"))
    geometric = np.isfinite(depth) & (depth > 0.0)
    out: dict[str, Any] = {
        "geometric_pixels": int(geometric.sum()),
        "coverage_pixels_gt0": int(np.count_nonzero(coverage > 0)),
        "coverage_pixels_without_depth": int(np.count_nonzero((coverage > 0) & ~geometric)),
        "finite_depth_values_outside_head": int(np.count_nonzero(np.isfinite(depth) & ~geometric)),
        "normals_finite": bool(np.isfinite(normals).all()),
        "normal_vectors_outside_head": int(np.count_nonzero((np.linalg.norm(normals, axis=2) > 1e-6) & ~geometric)),
        "camera_intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy},
    }
    if out["geometric_pixels"] < MIN_ALPHA_SUPPORT_PIXELS:
        out["status"] = "insufficient_geometry"
        return out
    rows, cols = np.nonzero(geometric)
    z = depth[rows, cols]
    points = np.column_stack(((cols - cx) * z / fx, (rows - cy) * z / fy, z))
    sample = points[::max(1, len(z) // 40000)]
    centered = sample - sample.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    deviation = singular / math.sqrt(len(sample))
    out.update(
        status="measured", depth_m=stats(z), principal_std_m=[float(value) for value in deviation],
        thickness_ratio=float(deviation[2] / max(deviation[0], 1e-12)), plane_residual_rms_m=float(deviation[2]),
        head_extent_m=float(np.ptp(centered @ axes[0])))
    unit = normals[rows, cols]
    lengths = np.linalg.norm(unit, axis=1)
    to_camera = -points / np.maximum(np.linalg.norm(points, axis=1, keepdims=True), 1e-12)
    facing = np.einsum("ij,ij->i", unit * np.array([1.0, 1.0, -1.0]), to_camera) / np.maximum(lengths, 1e-12)
    mean_direction = unit.mean(axis=0)
    mean_direction = mean_direction / max(float(np.linalg.norm(mean_direction)), 1e-12)
    spread = np.degrees(np.arccos(np.clip((unit / np.maximum(lengths[:, None], 1e-12)) @ mean_direction, -1.0, 1.0)))
    out.update(
        normal_length=stats(lengths), unit_normal_fraction=float(np.mean(np.abs(lengths - 1.0) <= 0.02)),
        camera_facing_fraction=float(np.mean(facing > 0.0)), normal_view_cosine=stats(facing),
        normal_angle_to_mean_deg=stats(spread), normal_fraction_deviating_over_15deg=float(np.mean(spread > 15.0)),
        depth_normal_agreement=depth_normal_agreement(depth, normals, geometric, fx, fy, cx, cy))
    return out


def mesh_evidence(layer: HeadLayer) -> dict[str, Any]:
    projected = np.asarray(layer.projected_vertices, np.float64)
    faces = np.asarray(layer.faces, np.int64)
    height, width = layer.depth.shape
    diagnostics = layer.diagnostics
    inside = ((projected[:, 0] >= 0) & (projected[:, 0] < width) & (projected[:, 1] >= 0) & (projected[:, 1] < height))
    validation = diagnostics.get("architecture_validation")
    return {
        "projected_vertices": int(len(projected)), "triangles": int(len(faces)),
        "projected_vertices_finite": bool(np.isfinite(projected).all()),
        "triangle_indices_valid": bool(faces.size > 0 and faces.min() >= 0 and faces.max() < len(projected)),
        "unreferenced_vertices": int(len(projected) - len(np.unique(faces))) if faces.size else int(len(projected)),
        "projected_vertices_inside_frame": int(inside.sum()),
        "vertex_bounds_xyxy": [float(projected[:, 0].min()), float(projected[:, 1].min()),
                               float(projected[:, 0].max()), float(projected[:, 1].max())],
        "guard_min_head_vertices": DEFAULT_POLICY.min_vertices,
        "model": diagnostics.get("model"), "architecture_validation": validation,
        "projection_roi": diagnostics.get("projection_roi"), "rasterization": diagnostics.get("rasterization"),
    }


def uv_evidence(layer: HeadLayer, engine: VolumetricHeadRenderer) -> dict[str, Any]:
    diagnostics = layer.diagnostics
    uv = diagnostics.get("uv_synthesis") or {}
    observed, symmetry, harmonic = (float(uv.get(key, 0.0)) for key in ("observed_ratio", "symmetry_ratio", "harmonic_ratio"))
    per_view = []
    for item in uv.get("per_view", []):
        semantic = item.get("semantic_mask") or {}
        per_view.append({"view_index": item.get("view_index"), "observed_uv_ratio": item.get("observed_uv_ratio"),
                         "observed_texels": item.get("observed_texels"),
                         "semantic_face_fraction": semantic.get("semantic_face_fraction"),
                         "forbidden_fraction": semantic.get("forbidden_fraction"),
                         "visible_face_fraction": semantic.get("visible_face_fraction")})
    return {"observed_ratio": observed, "symmetry_ratio": symmetry, "harmonic_ratio": harmonic,
            "completed_ratio": symmetry + harmonic, "accounted_ratio": observed + symmetry + harmonic,
            "view_count": uv.get("view_count"), "views_with_observed_uv": uv.get("views_with_observed_uv"),
            "unseen_texels_are_observations": uv.get("unseen_texels_are_observations"),
            "semantic_exclusion": uv.get("semantic_exclusion"), "per_view": per_view,
            "texture_restoration": diagnostics.get("texture_restoration"),
            "renderer_texture_size": int(engine.texture_size), "renderer_restoration_weight": float(engine.restoration_weight)}


def mask_planes(packet) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coverage = np.asarray(packet.coverage, np.float32)
    skin = np.asarray(packet.skin_mask, np.float32)
    hair = np.asarray(packet.hair_mask, np.float32)
    occluder = np.zeros_like(coverage) if packet.occluder is None else np.asarray(packet.occluder, np.float32)
    return coverage, skin, hair, occluder


def semantic_evidence(packet, alpha: np.ndarray) -> dict[str, Any]:
    coverage, skin, hair, occluder = mask_planes(packet)
    mesh = coverage > 0
    allowed = mesh & (skin > 0) & (hair < 0.5) & (occluder < 0.5)
    diagnostics = packet.mask_diagnostics
    return {
        "frame_pixels": int(coverage.size), "semantic_roi_xyxy": diagnostics.get("roi"),
        "mesh_coverage_pixels_gt0": int(mesh.sum()), "mesh_coverage_pixels_ge_half": int((coverage > 0.5).sum()),
        "mesh_pixels_semantic_skin": int((mesh & (skin > 0)).sum()),
        "mesh_pixels_excluded_hair_and_forbidden_classes": int((mesh & (hair >= 0.5)).sum()),
        "mesh_pixels_excluded_occluder": int((mesh & (occluder >= 0.5)).sum()),
        "mesh_pixels_excluded_non_skin": int((mesh & (skin <= 0)).sum()),
        "allowed_pixels": int(allowed.sum()), "alpha_pixels_gt0": int((alpha > 0).sum()),
        "alpha_pixels_ge_half": int((alpha > 0.5).sum()), "alpha_pixels_full": int((alpha >= 0.999).sum()),
        "ear_pixels": diagnostics.get("ear_pixels"), "tier": diagnostics.get("tier"),
        "hair_exclusion": diagnostics.get("hair_exclusion"), "background_exclusion": diagnostics.get("background_exclusion"),
        "ears_included": diagnostics.get("ears_included"), "parser_normalization": diagnostics.get("parser_normalization"),
        "neural_models": diagnostics.get("neural_models"),
        "forbidden_class_note": "hair_mask carries BiSeNet background/glasses/earrings/neck/cloth/hair/hat plus the mesh hair",
    }


def forbidden_evidence(original: np.ndarray, final: np.ndarray, packet, alpha: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Hard equality of every forbidden pixel: outside the mesh, non-skin semantics, hair/background/neck/clothing classes and
    occluders must be bit-identical to the input. Returns the evidence and the per-pixel changed mask."""
    coverage, skin, hair, occluder = mask_planes(packet)
    allowed = (coverage > 0) & (skin > 0) & (hair < 0.5) & (occluder < 0.5)
    forbidden = ~allowed
    changed = np.any(final != original, axis=2)
    roi = packet.mask_diagnostics.get("roi")
    in_roi = np.zeros(changed.shape, bool)
    if roi is not None:
        x0, y0, x1, y1 = (int(value) for value in roi)
        in_roi[y0:y1, x0:x1] = True
    groups = {"outside_mesh_coverage": coverage <= 0, "non_skin_semantics": skin <= 0,
              "hair_background_neck_cloth_hat_glasses": (hair >= 0.5) & in_roi, "occluders": (occluder >= 0.5) & in_roi,
              "outside_semantic_roi": ~in_roi}
    evidence = {
        "forbidden_pixels": int(forbidden.sum()), "forbidden_pixels_changed": int((forbidden & changed).sum()),
        "changed_pixels_total": int(changed.sum()), "changed_with_zero_alpha": int((changed & (alpha <= 0)).sum()),
        "alpha_support_outside_allowed": int(((alpha > 0) & forbidden).sum()),
        "by_group": {name: {"pixels": int(mask.sum()), "changed": int((mask & changed).sum())} for name, mask in groups.items()},
    }
    return evidence, changed


def changed_evidence(original: np.ndarray, final: np.ndarray, changed: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    support = alpha > 0.5
    delta = np.abs(final.astype(np.int16) - original.astype(np.int16))
    inside = delta[support]
    count = int((changed & support).sum())
    return {"alpha_support_pixels": int(support.sum()), "masked_changed_pixels": count,
            "masked_changed_fraction": count / max(int(support.sum()), 1),
            "masked_changed_pixels_alpha_gt0": int((changed & (alpha > 0)).sum()),
            "mean_abs_delta_levels": float(inside.mean()) if inside.size else 0.0,
            "p95_abs_delta_levels": float(np.percentile(inside, 95)) if inside.size else 0.0,
            "max_abs_delta_levels": int(delta.max())}


def tone_evidence(original: np.ndarray, raw: np.ndarray, final: np.ndarray, alpha: np.ndarray) -> dict[str, Any]:
    """A dark, empty or flat raster must not pass as a replacement: compare raw 3D and final luminance with the scene's own skin."""
    support = alpha > 0.5
    count = int(support.sum())
    if count < 64:
        return {"status": "insufficient_support", "support_pixels": count}
    original_luma, raw_luma, final_luma = (luma(image)[support] for image in (original, raw, final))
    return {"status": "measured", "support_pixels": count,
            "luma_mean": {"original": float(original_luma.mean()), "raw_3d": float(raw_luma.mean()), "final": float(final_luma.mean())},
            "luma_std": {"original": float(original_luma.std()), "raw_3d": float(raw_luma.std()), "final": float(final_luma.std())},
            "raw_to_original_luma_ratio": float(raw_luma.mean() / max(original_luma.mean(), 1.0)),
            "final_to_original_luma_ratio": float(final_luma.mean() / max(original_luma.mean(), 1.0)),
            "raw_black_fraction": float(np.mean(raw_luma < 12.0)),
            "channel_mean_bgr": {"original": [float(value) for value in original[support].mean(axis=0)],
                                 "raw_3d": [float(value) for value in raw[support].mean(axis=0)],
                                 "final": [float(value) for value in final[support].mean(axis=0)]}}


def landmark_regions(points: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray] | None:
    """Forehead band above the brows and jaw band (jawline to lower lip) from the target's dense68 landmarks; approximate by
    construction (the far side of a turned head is a network estimate)."""
    p = np.asarray(points, np.float64)[:, :2]
    if p.shape != (68, 2) or not np.isfinite(p).all():
        return None
    brows, jaw = p[17:27], p[0:17]
    brow_center, mouth_center, chin = brows.mean(axis=0), p[48:68].mean(axis=0), p[8]
    axis = brow_center - mouth_center
    length = float(np.linalg.norm(axis))
    if length < 4.0:
        return None
    up = axis / length
    right = np.array([-up[1], up[0]])
    face_height = float(np.dot(brow_center - chin, up))
    span = float(np.linalg.norm(brows[-1] - brows[0]))
    if face_height < 8.0 or span < 8.0:
        return None
    lower = np.vstack((brows[0] - 0.12 * span * right, brows, brows[-1] + 0.12 * span * right)) + 0.05 * face_height * up
    forehead = np.vstack((lower, lower[::-1] + 0.5 * face_height * up))
    jaw_band = np.vstack((jaw[2:15], p[[54, 55, 56, 57, 58, 59, 48]]))
    masks: dict[str, np.ndarray] = {}
    for name, polygon in (("forehead", forehead), ("jaw", jaw_band)):
        mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
        masks[name] = mask > 0
    return masks


def region_evidence(landmarks: np.ndarray, packet, alpha: np.ndarray, geometric: np.ndarray, changed: np.ndarray) -> dict[str, Any]:
    masks = landmark_regions(landmarks, alpha.shape)
    if masks is None:
        return {"status": "dense68_region_geometry_degenerate"}
    coverage, skin, hair, occluder = mask_planes(packet)
    explained = (hair >= 0.5) | (occluder >= 0.5) | (skin <= 0)
    shortfall = geometric & (alpha <= 0.5) & ~explained
    out: dict[str, Any] = {"status": "measured",
                           "note": "landmark-derived approximations; indicators of coverage, not a certificate"}
    for name, region in masks.items():
        pixels = int(region.sum())
        if pixels < 24:
            out[name] = {"pixels": pixels, "status": "degenerate_region"}
            continue
        mesh_pixels = int((geometric & region).sum())
        out[name] = {
            "pixels": pixels,
            "mesh_geometric_fraction": float(geometric[region].mean()),
            "mesh_coverage_ge_half_fraction": float((coverage[region] > 0.5).mean()),
            "alpha_ge_half_fraction": float((alpha[region] > 0.5).mean()),
            "alpha_ge_half_of_mesh_fraction": float(((alpha > 0.5) & geometric & region).sum() / max(mesh_pixels, 1)),
            "excluded_hair_or_forbidden_fraction": float((hair[region] >= 0.5).mean()),
            "excluded_occluder_fraction": float((occluder[region] >= 0.5).mean()),
            "excluded_non_skin_fraction": float((skin[region] <= 0).mean()),
            "changed_fraction": float(changed[region].mean()),
            "unexplained_alpha_shortfall_fraction": float(shortfall[region].mean()),
        }
    return out


def edge_metrics(original: np.ndarray, final: np.ndarray, raw: np.ndarray, alpha: np.ndarray, packet) -> dict[str, Any]:
    """Boundary indicators of the replacement support (alpha > 0.5): colour step across the boundary in the final frame against
    the same step in the untouched original, raw-3D-to-scene mismatch the blend has to hide, gradient energy added in the
    boundary band and the alpha ramp width. Indicators only - they do not prove the absence of seams or halos."""
    support = alpha > 0.5
    if int(support.sum()) < MIN_ALPHA_SUPPORT_PIXELS:
        return {"status": "insufficient_support"}
    ring = 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring + 1, 2 * ring + 1))
    support_u8 = support.astype(np.uint8)
    inner_ring = support & ~(cv2.erode(support_u8, kernel) > 0)
    outer_ring = (cv2.dilate(support_u8, kernel) > 0) & ~support
    contour = support & ~(cv2.erode(support_u8, np.ones((3, 3), np.uint8)) > 0)
    if not contour.any() or not outer_ring.any():
        return {"status": "no_boundary"}

    def lab(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)

    def window_mean(image_lab: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        weight = mask.astype(np.float32)
        numerator = cv2.GaussianBlur(image_lab * weight[..., None], (0, 0), 3.0)
        denominator = cv2.GaussianBlur(weight, (0, 0), 3.0)
        return numerator / np.maximum(denominator[..., None], 1e-6), denominator

    def step(inside: np.ndarray, outside: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        inner_mean, inner_weight = window_mean(inside, inner_ring)
        outer_mean, outer_weight = window_mean(outside, outer_ring)
        valid = contour & (inner_weight > 0.02) & (outer_weight > 0.02)
        return np.linalg.norm(inner_mean - outer_mean, axis=-1), valid

    lab_original, lab_final, lab_raw = lab(original), lab(final), lab(raw)
    final_step, valid = step(lab_final, lab_final)
    original_step, _ = step(lab_original, lab_original)
    raw_step, _ = step(lab_raw, lab_original)
    if not valid.any():
        return {"status": "no_measurable_boundary"}
    coverage, skin, hair, occluder = mask_planes(packet)
    near_forbidden = cv2.dilate(((hair >= 0.5) | (occluder >= 0.5)).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    band = inner_ring | outer_ring

    def gradient(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    gradient_final, gradient_original = float(gradient(final)[band].mean()), float(gradient(original)[band].mean())
    changed = np.any(final != original, axis=2)
    ramp = (alpha > 0.02) & (alpha < 0.98)

    def group(mask: np.ndarray) -> dict[str, Any]:
        selected = valid & mask
        return {"contour_pixels": int(selected.sum()),
                "final_step_deltaE": stats(final_step[selected]), "original_step_deltaE": stats(original_step[selected]),
                "raw_3d_vs_original_deltaE": stats(raw_step[selected])}

    return {
        "status": "measured", "ring_px": ring, "contour_pixels": int(contour.sum()), "measured_contour_pixels": int(valid.sum()),
        "all_boundary": group(np.ones(valid.shape, bool)), "next_to_hair_or_occluder": group(near_forbidden),
        "not_next_to_hair_or_occluder": group(~near_forbidden),
        "final_minus_original_step_deltaE_mean": float(final_step[valid].mean() - original_step[valid].mean()),
        "boundary_band_gradient": {"final": gradient_final, "original": gradient_original,
                                   "excess": gradient_final - gradient_original,
                                   "ratio": gradient_final / max(gradient_original, 1e-6)},
        "alpha_ramp_width_px": float(ramp.sum() / max(int(contour.sum()), 1)),
        "changed_pixels_in_zero_alpha_outer_ring": int((changed & outer_ring & (alpha <= 0)).sum()),
        "note": "indicators only; absence of a measured step does not prove seam-free blending - inspect the strip",
    }


# --------------------------------------------------------------------------------------------------------- gate
def assess(spec: ClassSpec, attempt: Attempt, composite: Composite, engine: VolumetricHeadRenderer,
           args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    layer, result, original = attempt.layer, attempt.result, attempt.image
    replacement = result.replacements[0]
    final, packet, alpha, report = composite.final, composite.packet, composite.alpha, composite.report
    checks: list[dict[str, Any]] = []

    def check(name: str, ok: Any, detail: str = "") -> None:
        checks.append({"name": name, "passed": bool(ok), "detail": detail})

    evidence: dict[str, Any] = {}
    solved = attempt.solved
    landmarks = attempt.target.observation.landmarks68
    # ---- real dense68 target pose ---------------------------------------------------------------------------------
    check("target_dense68_landmarks_measured",
          landmarks is not None and np.asarray(landmarks).shape == (68, 3) and np.isfinite(np.asarray(landmarks)).all(),
          f"tracker pose source={attempt.tracker_pose['source']}")
    check("pose_solved_from_dense68",
          solved["source"] == "makehuman_pnp_dense68" and solved["dense_landmark_count"] == 68 and solved["dense_failure"] is None,
          f"source={solved['source']} dense_landmarks={solved['dense_landmark_count']} dense_failure={solved['dense_failure']} "
          f"rms={solved['reprojection_rms_px']:.2f}px inliers={solved['inliers']}")
    check(f"solved_yaw_is_{spec.name}", attempt.in_window and attempt.consistent,
          f"solved yaw {solved['yaw_deg']:+.1f} (|yaw| window {spec.accept[0]:g}..{spec.accept[1]:g}), "
          f"roll-independent turn {solved['off_axis_deg']:.1f}")
    evidence["pose"] = {"solved": solved, "tracker": attempt.tracker_pose, "class_window_abs_yaw_deg": list(spec.accept),
                        "class_target_abs_yaw_deg": spec.target}
    # ---- render result contract ------------------------------------------------------------------------------------
    stages = tuple(replacement.pipeline_stages)
    check("replacement_is_full_head_with_stage_trace",
          replacement.full_head is True and stages == FRAME_PIPELINE_STAGES[:3] and replacement.track_id == CANONICAL_TRACK_ID,
          f"full_head={replacement.full_head} stages={list(stages)} track_id={replacement.track_id}")
    shape = original.shape
    same = {"rendered_frame": np.array_equal(np.asarray(replacement.rendered_frame), layer.image),
            "coverage": np.array_equal(np.asarray(replacement.coverage, np.float32), layer.coverage),
            "normals": np.array_equal(np.asarray(replacement.normals, np.float32), layer.normals)}
    check("replacement_carries_the_renderer_layer",
          all(same.values()) and replacement.rendered_frame.shape == shape and replacement.rendered_frame.dtype == np.uint8,
          f"identical to HeadLayer: {same}")
    # ---- mesh --------------------------------------------------------------------------------------------------------
    mesh = mesh_evidence(layer)
    evidence["mesh"] = mesh
    check("mesh_is_full_head_vertex_set",
          mesh["projected_vertices"] >= DEFAULT_POLICY.min_vertices and mesh["projected_vertices_finite"]
          and mesh["triangle_indices_valid"],
          f"{mesh['projected_vertices']} projected vertices (guard floor {DEFAULT_POLICY.min_vertices}), "
          f"{mesh['triangles']} triangles, {mesh['projected_vertices_inside_frame']} inside the frame")
    structural_ok, structural_detail = structural_validation(mesh["architecture_validation"])
    check("architecture_guard_validated_fitted_head", structural_ok, structural_detail)
    # ---- volume ------------------------------------------------------------------------------------------------------
    volume = volume_evidence(layer)
    evidence["volume"] = volume
    measured = volume.get("status") == "measured"
    check("volume_measurable", measured, f"geometric pixels={volume['geometric_pixels']}")
    if measured:
        agreement = volume["depth_normal_agreement"]
        check("depth_is_finite_where_covered", volume["coverage_pixels_without_depth"] == 0 and volume["normals_finite"]
              and volume["finite_depth_values_outside_head"] == 0,
              f"coverage without depth={volume['coverage_pixels_without_depth']}, finite depth outside head="
              f"{volume['finite_depth_values_outside_head']}")
        check("volume_is_not_planar", volume["thickness_ratio"] >= MIN_THICKNESS_RATIO,
              f"point-cloud thickness ratio {volume['thickness_ratio']:.3f} (floor {MIN_THICKNESS_RATIO}), "
              f"plane residual {volume['plane_residual_rms_m'] * 1000:.1f} mm over {volume['head_extent_m'] * 1000:.0f} mm")
        check("normals_are_unit_and_face_the_camera",
              volume["unit_normal_fraction"] >= MIN_UNIT_NORMAL_FRACTION and volume["camera_facing_fraction"] >= MIN_CAMERA_FACING_FRACTION,
              f"unit {volume['unit_normal_fraction']:.4f}, camera-facing {volume['camera_facing_fraction']:.3f}")
        check("normals_agree_with_depth", agreement.get("status") == "measured"
              and agreement["median_angle_deg"] <= MAX_DEPTH_NORMAL_MEDIAN_DEG
              and agreement["fraction_within_35deg"] >= MIN_DEPTH_NORMAL_WITHIN_35,
              "depth-derived vs rasterized normals: " + (
                  f"median {agreement['median_angle_deg']:.1f} deg, within 35 deg {agreement['fraction_within_35deg']:.3f}"
                  if agreement.get("status") == "measured" else str(agreement)))
    # ---- UV ----------------------------------------------------------------------------------------------------------
    uv = uv_evidence(layer, engine)
    evidence["uv"] = uv
    restoration = uv["texture_restoration"] or {}
    expected_status = "applied" if args.restoration_weight > 0 else "disabled"
    check("uv_observed_texels_present", uv["observed_ratio"] >= MIN_OBSERVED_UV_RATIO
          and uv["unseen_texels_are_observations"] is False,
          f"observed {uv['observed_ratio']:.3f} symmetry {uv['symmetry_ratio']:.3f} harmonic {uv['harmonic_ratio']:.3f}; "
          f"unseen_texels_are_observations={uv['unseen_texels_are_observations']}")
    check("uv_restoration_matches_request", restoration.get("status") == expected_status
          and uv["renderer_texture_size"] == args.texture_size
          and math.isclose(uv["renderer_restoration_weight"], args.restoration_weight, abs_tol=1e-9),
          f"restoration={restoration.get('status')} (expected {expected_status}); renderer texture_size="
          f"{uv['renderer_texture_size']} restoration_weight={uv['renderer_restoration_weight']}")
    # ---- composite ---------------------------------------------------------------------------------------------------
    face_report = report.faces[0] if report is not None and report.faces else None
    check("composite_not_skipped", report is not None and len(report.faces) == 1 and not report.skipped and not report.withheld,
          "faces=%s skipped=%s withheld=%s" % (None if report is None else len(report.faces),
                                               None if report is None else report.skipped, None if report is None else report.withheld))
    if face_report is not None:
        check("composite_ran_all_five_stages_without_fallback",
              tuple(face_report.pipeline_stages) == FRAME_PIPELINE_STAGES and not face_report.fallbacks,
              f"stages={list(face_report.pipeline_stages)} fallbacks={face_report.fallbacks}")
        evidence["compositor"] = {"face_report": face_report.as_dict(), "photometry": face_report.relight,
                                  "detail_finish": face_report.restore, "stage_ms": report.stage_ms}
    usable = (final is not None and packet is not None and alpha is not None and final.shape == shape and final.dtype == np.uint8)
    check("compositor_state_available", usable,
          f"final={None if final is None else (final.shape, str(final.dtype))} packet={packet is not None} "
          f"alpha={None if alpha is None else alpha.shape}")
    if not usable:
        return evidence, checks
    # ---- semantics, forbidden equality, change -----------------------------------------------------------------------
    evidence["semantics"] = semantic_evidence(packet, alpha)
    forbidden, changed = forbidden_evidence(original, final, packet, alpha)
    evidence["forbidden"] = forbidden
    check("forbidden_pixels_bit_identical",
          forbidden["forbidden_pixels_changed"] == 0 and forbidden["changed_with_zero_alpha"] == 0
          and forbidden["alpha_support_outside_allowed"] == 0,
          f"{forbidden['forbidden_pixels']} forbidden pixels, {forbidden['forbidden_pixels_changed']} changed; "
          f"{forbidden['changed_with_zero_alpha']} changed at zero alpha; "
          f"{forbidden['alpha_support_outside_allowed']} alpha pixels outside the allowed set")
    changes = changed_evidence(original, final, changed, alpha)
    evidence["changed"] = changes
    check("replacement_changed_masked_pixels",
          changes["alpha_support_pixels"] >= MIN_ALPHA_SUPPORT_PIXELS and changes["masked_changed_fraction"] >= MIN_CHANGED_FRACTION
          and changes["mean_abs_delta_levels"] >= MIN_MEAN_ABS_DELTA,
          f"{changes['masked_changed_pixels']} of {changes['alpha_support_pixels']} alpha>0.5 pixels changed "
          f"({changes['masked_changed_fraction']:.3f}), mean |delta| {changes['mean_abs_delta_levels']:.2f} levels")
    tone = tone_evidence(original, np.asarray(layer.image), final, alpha)
    evidence["tone"] = tone
    tone_ok = (tone.get("status") == "measured"
               and RAW_LUMA_RATIO[0] <= tone["raw_to_original_luma_ratio"] <= RAW_LUMA_RATIO[1]
               and FINAL_LUMA_RATIO[0] <= tone["final_to_original_luma_ratio"] <= FINAL_LUMA_RATIO[1]
               and tone["raw_black_fraction"] <= MAX_RAW_BLACK_FRACTION)
    check("raw_and_final_tone_plausible", tone_ok,
          "raw/original luma %.2f (window %s), final/original %.2f (window %s), raw near-black %.3f (max %.2f)" % (
              tone.get("raw_to_original_luma_ratio", float("nan")), RAW_LUMA_RATIO, tone.get("final_to_original_luma_ratio", float("nan")),
              FINAL_LUMA_RATIO, tone.get("raw_black_fraction", float("nan")), MAX_RAW_BLACK_FRACTION)
          if tone.get("status") == "measured" else str(tone))
    # ---- indicators (never gated) ------------------------------------------------------------------------------------
    geometric = np.isfinite(layer.depth)
    replacement_real = changes["masked_changed_fraction"] >= MIN_CHANGED_FRACTION and tone_ok
    evidence["regions"] = region_evidence(np.asarray(landmarks), packet, alpha, geometric, changed)
    evidence["edge"] = edge_metrics(original, final, np.asarray(layer.image), alpha, packet)
    evidence["indicators_reliable"] = bool(replacement_real)
    evidence["indicator_notice"] = ("region coverage and edge metrics are indicators only. They are meaningful solely for a real, "
                                    "plausibly lit replacement and never certify seam-free blending or forehead/jaw coverage.")
    return evidence, checks


# ------------------------------------------------------------------------------------------------------ strip image
def head_roi(geometric: np.ndarray) -> tuple[int, int, int, int]:
    height, width = geometric.shape
    rows, cols = np.nonzero(geometric)
    if not len(rows):
        return 0, 0, width, height
    x0, x1, y0, y1 = int(cols.min()), int(cols.max()) + 1, int(rows.min()), int(rows.max()) + 1
    pad = int(round(0.30 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(width, x1 + pad), min(height, y1 + pad)
    x1 = min(width, max(x1, x0 + 96))
    y1 = min(height, max(y1, y0 + 96))
    return max(0, x1 - max(x1 - x0, 96)), max(0, y1 - max(y1 - y0, 96)), x1, y1


def tint(image: np.ndarray, weight: np.ndarray, color: tuple[int, int, int], strength: float) -> np.ndarray:
    blend = (np.clip(weight, 0.0, 1.0) * strength).astype(np.float32)[..., None]
    return np.clip(image.astype(np.float32) * (1.0 - blend) + np.asarray(color, np.float32) * blend, 0, 255).astype(np.uint8)


def front_edges(layer: HeadLayer) -> np.ndarray:
    """Unique edges of the camera-facing triangles whose centroid lies on the rasterized head. Winding contract of
    VolumetricHeadRenderer: it rasterizes faces[:, ::-1], so in the stored order a camera-facing triangle has a positive signed
    image-space area (y down). Hidden front-facing triangles (nose over far cheek) can still appear; this is a debug overlay."""
    projected = np.asarray(layer.projected_vertices, np.float64)
    faces = np.asarray(layer.faces, np.int64)
    a, b, c = projected[faces[:, 0]], projected[faces[:, 1]], projected[faces[:, 2]]
    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    centroid = (a + b + c) / 3.0
    height, width = layer.depth.shape
    on_frame = (centroid[:, 0] >= 0) & (centroid[:, 0] < width) & (centroid[:, 1] >= 0) & (centroid[:, 1] < height)
    px = np.clip(np.rint(centroid[:, 0]).astype(np.int64), 0, width - 1)
    py = np.clip(np.rint(centroid[:, 1]).astype(np.int64), 0, height - 1)
    keep = (area > 0) & on_frame & np.isfinite(np.asarray(layer.depth)[py, px])
    triangles = faces[keep]
    if not len(triangles):
        return np.zeros((0, 2), np.int64)
    edges = np.concatenate((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (24, 24, 24), -1)
    cv2.putText(out, text, (6, 17), FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_strip(original: np.ndarray, raw: np.ndarray, final: np.ndarray, layer: HeadLayer, alpha: np.ndarray, packet,
                 caption: list[str]) -> np.ndarray:
    """Original | wireframe + hair-excluded alpha | raw 3D | final, cropped to the head with one shared ROI and scale."""
    depth = np.asarray(layer.depth)
    geometric = np.isfinite(depth)
    x0, y0, x1, y1 = head_roi(geometric)
    roi_w, roi_h = x1 - x0, y1 - y0
    scale = min(3.0, PANEL_LONG_SIDE / float(max(roi_w, roi_h)))
    size = (max(1, int(round(roi_w * scale))), max(1, int(round(roi_h * scale))))
    sx, sy = size[0] / float(roi_w), size[1] / float(roi_h)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA

    def view(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image[y0:y1, x0:x1], size, interpolation=interpolation)

    def plane(values: np.ndarray) -> np.ndarray:
        return cv2.resize(np.asarray(values, np.float32)[y0:y1, x0:x1], size, interpolation=cv2.INTER_LINEAR)

    _, _, hair, occluder = mask_planes(packet)
    mesh_view = plane(geometric.astype(np.float32)) > 0.5
    alpha_view = plane(alpha)
    wire = tint(view(original), alpha_view, (60, 200, 60), 0.40)
    wire = tint(wire, ((plane(hair) >= 0.5) & mesh_view).astype(np.float32), (50, 50, 230), 0.40)
    wire = tint(wire, ((plane(occluder) >= 0.5) & mesh_view).astype(np.float32), (230, 120, 40), 0.40)
    contours, _ = cv2.findContours((alpha_view > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(wire, contours, -1, (80, 255, 80), 1, cv2.LINE_AA)
    edges = front_edges(layer)
    if len(edges):
        vertices = (np.asarray(layer.projected_vertices, np.float64) - np.array([x0, y0], np.float64)) * np.array([sx, sy])
        segments = np.rint(np.stack((vertices[edges[:, 0]], vertices[edges[:, 1]]), axis=1)).astype(np.int32).reshape(-1, 2, 1, 2)
        lines = wire.copy()
        cv2.polylines(lines, list(segments), False, (0, 255, 255), 1, cv2.LINE_AA)
        wire = cv2.addWeighted(lines, 0.6, wire, 0.4, 0.0)
    cv2.putText(wire, "green alpha / red hair+forbidden / blue occluder / yellow front mesh", (6, size[1] - 8), FONT, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)
    labels = ("1 Original", "2 Wireframe + hair-excluded alpha", "3 Raw 3D raster", "4 Final composite")
    panels = [label_panel(image, text) for image, text in zip((view(original), wire, view(raw), view(final)), labels)]
    gap = np.full((size[1], 4, 3), 40, np.uint8)
    row = np.hstack([panels[0], gap, panels[1], gap, panels[2], gap, panels[3]])
    footer = np.full((22 * len(caption) + 8, row.shape[1], 3), 24, np.uint8)
    for index, line in enumerate(caption):
        cv2.putText(footer, line, (8, 20 + 22 * index), FONT, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([row, footer])


# ---------------------------------------------------------------------------------------------------- case assembly
def finish_case(session: ReferencePipeline, spec: ClassSpec, attempts: list[Attempt], args: argparse.Namespace, root: Path,
                output_relative: str) -> dict[str, Any]:
    case: dict[str, Any] = {"name": spec.name, "label": spec.label, "target_abs_yaw_deg": spec.target,
                            "accept_abs_yaw_deg": list(spec.accept), "status": "no_render", "passed": False, "checks": []}
    rendered = [attempt for attempt in attempts if attempt.status == "rendered"]
    if not rendered:
        case["failure"] = "no candidate rendered: " + "; ".join(
            f"{a.candidate.video.name} f{a.candidate.frame} {(a.error or {}).get('code')}" for a in attempts)
        case["attempts"] = [attempt.summary() for attempt in attempts]
        return case
    ranked = sorted(rendered, key=lambda attempt: (not (attempt.in_window and attempt.consistent), attempt.distance, attempt.order))
    for attempt in ranked:
        composite = run_composite(session, attempt)
        refusal = composite.refusal_code()
        if refusal in DATA_DEPENDENT_SKIPS:
            attempt.note = f"composite refused this frame ({refusal}); the next ranked candidate was used"
            continue
        engine = session.full_head_engines.get(CANONICAL_TRACK_ID)
        if not isinstance(engine, VolumetricHeadRenderer):
            raise ContractError(f"session.full_head_engines[{CANONICAL_TRACK_ID}] is {type(engine).__name__}, not a "
                                "VolumetricHeadRenderer, at assessment time")
        evidence, checks = assess(spec, attempt, composite, engine, args)
        passed = all(item["passed"] for item in checks)
        case.update(status="verified", chosen={**attempt.candidate.summary(), "decoded_frame": attempt.decoded_frame,
                                               "working_size": [int(attempt.image.shape[1]), int(attempt.image.shape[0])]},
                    passed=passed, checks=checks, evidence=evidence,
                    frame_latency_ms=None if attempt.result is None else {
                        "detection": attempt.result.latency.detection_ms, "render": attempt.result.latency.swap_ms,
                        "total": attempt.result.latency.total_ms})
        usable = (composite.final is not None and composite.packet is not None and composite.alpha is not None
                  and "forbidden" in evidence)
        if usable:
            solved, mesh, volume, uv = attempt.solved, evidence["mesh"], evidence["volume"], evidence["uv"]
            changes = evidence.get("changed", {})
            caption = [
                f"{spec.label} | {attempt.candidate.video.name} frame {attempt.decoded_frame} t={attempt.timestamp:.2f}s | solved yaw "
                f"{solved['yaw_deg']:+.1f} pitch {solved['pitch_deg']:+.1f} roll {solved['roll_deg']:+.1f} turn {solved['off_axis_deg']:.1f}"
                f" | dense68 PnP rms {solved['reprojection_rms_px']:.2f}px",
                f"vertices {mesh['projected_vertices']} triangles {mesh['triangles']} | thickness "
                f"{volume.get('thickness_ratio', float('nan')):.2f} | UV observed {uv['observed_ratio']:.0%} symmetry "
                f"{uv['symmetry_ratio']:.0%} harmonic {uv['harmonic_ratio']:.0%} | masked changed "
                f"{changes.get('masked_changed_pixels', 0)} px | forbidden changed {evidence['forbidden']['forbidden_pixels_changed']}"
                f" | GATE {'PASS' if passed else 'FAIL'}",
                "Metrics are indicators only. A passing gate does not certify seam-free blending or visual quality: inspect the panels."]
            strip = render_strip(attempt.image, np.asarray(attempt.layer.image), composite.final, attempt.layer, composite.alpha,
                                 composite.packet, caption)
            ok, encoded = cv2.imencode(".png", strip)
            if not ok:
                raise ContractError(f"PNG encoding of the {spec.name} strip failed")
            relative = f"{output_relative}/{spec.name}_comparison.png"
            atomic_bytes(root, relative, encoded.tobytes())
            case["strip"] = {"path": relative, "width": int(strip.shape[1]), "height": int(strip.shape[0])}
        case["attempts"] = [item.summary() for item in attempts]
        return case
    case["failure"] = "every rendered candidate was refused by the composite as data-dependent (no visible face after exclusions)"
    case["attempts"] = [attempt.summary() for attempt in attempts]
    return case


# ------------------------------------------------------------------------------------------------------------ main
def load_source(root: Path, catalog: ModelCatalog, name: str):
    started = time.perf_counter()
    resolved = resolve_faceset(root, name)
    engine = LandmarkEngine(root, catalog=catalog)
    loaded = load_faceset(resolved.archive, catalog=catalog, root=root, engine=engine, faceset_id=resolved.faceset_id)
    views = loaded.document.views
    summary = {"id": resolved.faceset_id, "archive": str(resolved.archive), "source": resolved.source, "views": len(views),
               "views_with_dense68": [int(view.index) for view in views if view.landmarks_68 is not None],
               "landmark_models": loaded.landmark_models, "samples": [sample.summary() for sample in loaded.samples],
               "notes": loaded.notes, "bank": loaded.bank.summary() if loaded.bank is not None else None,
               "seconds": round(time.perf_counter() - started, 2)}
    return loaded, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = application_root()
    output_relative = owned_relative(root, args.output_dir)
    specs = class_specs(args.profile_min_yaw)
    began = time.perf_counter()
    receipt: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tool": "roop_3d.tools.verify_stage4",
        "command": [sys.executable, "-B", "-m", "roop_3d.tools.verify_stage4", *(argv if argv is not None else sys.argv[1:])],
        "parameters": {"input_dir": str(args.input_dir), "output_dir": output_relative, "faceset": args.faceset,
                       "videos": args.video, "max_side": args.max_side, "texture_size": args.texture_size,
                       "restoration_weight": args.restoration_weight, "candidates_per_class": args.candidates,
                       "lead_in_frames": args.lead_in, "detection_threshold": args.detection_threshold,
                       "profile_min_abs_yaw_deg": args.profile_min_yaw, "provider": args.provider,
                       "hardware_profile": args.profile},
        "environment": environment(), "classes": {spec.name: {"label": spec.label, "target_abs_yaw_deg": spec.target,
                                                              "accept_abs_yaw_deg": list(spec.accept)} for spec in specs},
        "cases": [], "global_checks": [], "failures": [], "passed": False,
        "limitations": [
            "Frames are non-consecutive independent shots: the tracker is reset per candidate, a hard cut is signalled so the "
            "renderer keeps no pose history, and the largest face is labelled track 1 so a single renderer (identity fit + UV) serves "
            "all three. Temporal smoothing, identity gating and multi-face handling are not exercised.",
            "Candidate frames are proposed from the stage-3 cache (hints). The class of each frame is decided by the solved yaw of "
            "the fresh dense68 render; if no candidate lands in a class window that class fails (raise --candidates or relax "
            "--profile-min-yaw explicitly).",
            "Frames are downscaled to --max-side before tracking, rendering and compositing; metrics are at that resolution.",
            "Volume, tone and change thresholds are heuristic floors that reject a flat, empty, dark or no-op result; they are not "
            "calibrated quality scores.",
            "Forehead/jaw regions come from the target's dense68 landmarks (the far side of a turned head is a network estimate) "
            "and edge statistics are boundary indicators. Neither certifies seam-free blending or complete forehead/jaw coverage; "
            "visual inspection of the three PNG strips is required.",
            "The wireframe draws edges of camera-facing triangles on the rasterized head; hidden front-facing triangles can appear.",
            "No identity similarity or perceptual quality is measured; the source is a different person from every target."]}
    session: ReferencePipeline | None = None
    reader: FrameReader | None = None
    fatal: dict[str, Any] | None = None
    try:
        catalog = ModelCatalog(root)
        catalog.rescan_local()
        config = runtime_config(args)
        receipt["runtime"] = {"provider": config.provider, "hardware_profile": config.hardware_profile}

        print(f"FACESET {args.faceset}", flush=True)
        loaded, receipt["faceset"] = load_source(root, catalog, args.faceset)
        print(f"  {receipt['faceset']['views']} views, dense68 on views {receipt['faceset']['views_with_dense68']}, "
              f"landmark models {receipt['faceset']['landmark_models']} ({receipt['faceset']['seconds']}s)", flush=True)

        videos = sorted(args.input_dir.glob("*.mp4"), key=natural)
        if args.video:
            videos = [video for video in videos if video.name in set(args.video)]
        if not videos:
            raise ContractError(f"no .mp4 under {args.input_dir}" + (f" matching {args.video}" if args.video else ""))
        pools, receipt["selection"] = propose_candidates(root, videos, specs, max_side=args.max_side, limit=args.candidates)
        print(f"CANDIDATES anchor video {receipt['selection']['anchor_video']}: " + ", ".join(
            f"{name} {len(pool)}" for name, pool in pools.items()), flush=True)

        session = ReferencePipeline(catalog, config)
        capability = session.prepare_session(loaded.document, detection_threshold=args.detection_threshold,
                                             restoration_weight=args.restoration_weight, texture_size=args.texture_size)
        receipt["session"] = {"capability_after_prepare": capability}
        renderer_wired = bool(capability.get("renderer", {}).get("wired"))
        receipt["global_checks"].append({"name": "renderer_wired", "passed": renderer_wired,
                                         "detail": str(capability.get("renderer"))})
        if not renderer_wired:
            raise ContractError(f"the true-3D renderer is not wired into the session: {capability.get('renderer')}")
        reader = FrameReader(args.max_side)

        for spec in specs:
            print(f"CLASS {spec.name} ({spec.label}; solved |yaw| {spec.accept[0]:g}..{spec.accept[1]:g}, "
                  f"{min(len(pools[spec.name]), args.candidates)} candidate(s))", flush=True)
            attempts = evaluate_class(session, reader, spec, pools[spec.name], args) if pools[spec.name] else []
            case = finish_case(session, spec, attempts, args, root, output_relative)
            receipt["cases"].append(case)
            for attempt in attempts:
                attempt.drop_arrays()
            if case["status"] == "verified":
                solved = case["chosen"]
                failed = [item for item in case["checks"] if not item["passed"]]
                print(f"  {'PASS' if case['passed'] else 'FAIL'} {spec.name}: {solved['video']} frame {solved['decoded_frame']} "
                      f"-> {case.get('strip', {}).get('path')}", flush=True)
                for item in failed:
                    print(f"    FAILED {item['name']}: {item['detail']}", flush=True)
            else:
                print(f"  FAIL {spec.name}: {case.get('failure')}", flush=True)

        engines = dict(session.full_head_engines)
        receipt["session"]["head_renderers_built"] = len(engines)
        receipt["session"]["capability_after_render"] = session.capability()
        first = next((a for case in receipt["cases"] for a in case.get("attempts", []) if a.get("renderer_built_in_this_call")), None)
        receipt["session"]["renderer_build_plus_first_render_seconds"] = None if first is None else first["seconds"]
        engine = engines.get(CANONICAL_TRACK_ID)
        last_layer = engine.last_layer if isinstance(engine, VolumetricHeadRenderer) else None
        if last_layer is not None:
            receipt["renderer"] = {key: last_layer.diagnostics.get(key) for key in
                                   ("model", "architecture_validation", "identity_fit", "source_views", "uv_synthesis",
                                    "texture_restoration", "limitations")}
        receipt["global_checks"].append({"name": "one_identity_build_reused", "passed": len(engines) == 1,
                                         "detail": f"{len(engines)} head renderer(s) cached in the session"})
        loaded_ok = bool(receipt["faceset"]["views_with_dense68"]) and bool(receipt["faceset"]["landmark_models"].get("1k3d68"))
        receipt["global_checks"].append({"name": "harjot_loaded_with_dense68_views", "passed": loaded_ok,
                                         "detail": f"dense68 views {receipt['faceset']['views_with_dense68']}, models "
                                                   f"{receipt['faceset']['landmark_models']}"})
        receipt["global_checks"].append({"name": "three_classes_verified",
                                         "passed": len(receipt["cases"]) == len(specs)
                                         and all(case["status"] == "verified" for case in receipt["cases"]),
                                         "detail": ", ".join(f"{case['name']}={case['status']}" for case in receipt["cases"])})
    except RendererBuildError as error:
        fatal = {"kind": "renderer_build_failed", **error_record(error.error), "traceback": traceback.format_exc()}
    except Exception as error:                                   # an aborted run still leaves a receipt with the traceback
        fatal = {"kind": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        cleanup: list[str] = []
        for name, resource, method in (("frame reader", reader, "close"), ("session", session, "release_session")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as error:
                    cleanup.append(f"{name}: {type(error).__name__}: {error}")
        receipt["cleanup_errors"] = cleanup

    failures: list[str] = []
    for item in receipt["global_checks"]:
        if not item["passed"]:
            failures.append(f"{item['name']}: {item['detail']}")
    for case in receipt["cases"]:
        if case["status"] != "verified":
            failures.append(f"{case['name']}: {case.get('failure')}")
        failures.extend(f"{case['name']}: {item['name']}: {item['detail']}" for item in case["checks"] if not item["passed"])
    if fatal is not None:
        receipt["fatal"] = fatal
        failures.append(f"aborted: {fatal['kind']}: {fatal.get('message')}")
    receipt["failures"] = failures
    receipt["passed"] = fatal is None and not failures
    receipt["seconds"] = round(time.perf_counter() - began, 1)
    atomic_json(root, RECEIPT, jsonable(receipt))

    print(f"\nreceipt: {RECEIPT}")
    for case in receipt["cases"]:
        if "strip" in case:
            print(f"strip:   {case['strip']['path']}")
    print("GATE " + ("PASS" if receipt["passed"] else "FAIL")
          + " (the gate does not certify seam-free blending or visual quality; inspect the strips)")
    for failure in failures:
        print("  FAILURE:", failure)
    if fatal is not None:
        print(fatal["traceback"])
        return 2
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
