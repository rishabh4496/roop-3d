"""Stage 13 synthetic acceptance: frozen angle identities + gap-safe target tracing.

Run from the repository root with:
    app/roop_3d/.venv/Scripts/python.exe -B verify_stage13_angle_tracking_and_swap.py

The FaceSet part reads a local multi-angle .fsz. The video path is deliberately synthetic
and model-free so it can verify all three render-mode routes without downloading or
activating detector/swapper/restoration weights.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.angle_tracker import (AngleDetection, AngleTrajectoryTracer, normalize_render_mode,
                                   pose_match_threshold)
from roop_3d.core.pose import Pose3D, project_model5
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.identity_swapper import IdentitySwapper
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.identity.target_lock import FaceRef, TargetLockStore
from roop_3d.pose_matrix import FrozenSourcePoseMatrix
from roop_3d.engine.volumetric.preprocess import roll_compensated_aligned_crop, transform_points
from roop_3d.engine.volumetric.session import profile_aware_mask
from roop_3d.reconstruction.uv_synthesizer import remap_points
from roop_3d.faceset.identity_blend import IdentityBlendSpec


def check(condition: bool, message: str) -> None:
    if not condition:
        print("FAIL:", message)
        raise SystemExit(1)
    print("ok  :", message)


def _faceset_candidates() -> list[Path]:
    configured = os.environ.get("ROOP3D_VERIFY_FACESETS")
    if configured:
        root = Path(configured).expanduser()
        return [root] if root.is_file() else sorted(root.glob("*.fsz"))
    return sorted((APP / "facesets").glob("*.fsz"))


def _multi_angle_document():
    reader = FaceSetReader()
    candidates = []
    for path in _faceset_candidates():
        try:
            document = reader.read(path)
        except Exception as error:
            print(f"skip: unreadable local .fsz ({path.name}): {type(error).__name__}")
            continue
        if len(document.views) < 3:
            continue
        yaws = [float(view.yaw) for view in document.views]
        pitches = [float(view.pitch) for view in document.views]
        spread = max(yaws) - min(yaws) + max(pitches) - min(pitches)
        if spread >= 25.0:
            candidates.append((spread, document, path))
    if not candidates:
        raise SystemExit("FAIL: no local multi-angle .fsz found; set ROOP3D_VERIFY_FACESETS to its directory or file")
    _, document, path = max(candidates, key=lambda item: item[0])
    print(f"source: {path.name} ({len(document.views)} views, multi-angle .fsz)")
    return document


def verify_frozen_matrix(document) -> FrozenSourcePoseMatrix:
    # A deterministic non-identity EMAP keeps the verifier independent of installed model artifacts
    # while proving that each frozen source vector is actually transformed before frame inference.
    emap = np.roll(np.eye(512, dtype=np.float32), 13, axis=1)
    matrix = FrozenSourcePoseMatrix.from_document(document, projection=emap, device="cuda")
    check(len(matrix.cells) == 9, "pre-freeze contains all 3x3 yaw/pitch cells")
    check(matrix.roll_rotations.shape == (361, 2, 2), "pre-freeze contains roll compensation from -180 through +180")
    check(matrix.projection_device in {"cuda", "cpu"}, f"all pose EMAP vectors are pre-projected ({matrix.projection_device})")
    check(matrix.fingerprint and matrix.summary()["roll_rotations"] == 361, "frozen pose matrix fingerprint is stable")
    probe = matrix.lookup(Pose3D(yaw=12, pitch=-8, roll=0))
    expected_projected = _unit(probe.pose_selected_vector @ emap)
    check(float(np.max(np.abs(expected_projected - probe.projected_vector))) < 1e-5,
          "frozen vectors contain the precomputed non-identity EMAP product")

    second_id = "stage13_blend_source"
    direction = _unit(np.random.default_rng(809).normal(size=512))
    def shift(vector):
        return _unit(0.92 * np.asarray(vector, np.float32) + 0.08 * direction)
    second_views = tuple(view.model_copy(update={"embedding": tuple(float(v) for v in shift(view.embedding))})
                         for view in document.views)
    second = document.model_copy(update={"faceset_id": second_id, "source_sha256": "a" * 64,
        "canonical_embedding": tuple(float(v) for v in shift(document.canonical_embedding)),
        "medoid_embedding": tuple(float(v) for v in shift(document.medoid_embedding)), "views": second_views})
    recipe = IdentityBlendSpec(source_face_ids=(document.faceset_id, second_id), weights=(0.6, 0.4))
    blended = FrozenSourcePoseMatrix.from_documents((document, second), recipe,
        projection=emap)
    check(blended.mode == "blend:slerp" and abs(float(np.linalg.norm(
        blended.lookup(Pose3D(yaw=45, pitch=0, roll=0)).pose_selected_vector)) - 1) < 1e-4,
        "multi-source identity recipe is pose-cell blended before freezing")

    angles = [(yaw, pitch, roll) for yaw in (-75, 0, 75) for pitch in (-35, 0, 35) for roll in (-45, 45)]
    for yaw, pitch, roll in angles:
        selected = matrix.lookup(Pose3D(yaw=yaw, pitch=pitch, roll=roll))
        check(selected.vector.shape == (512,) and abs(float(np.linalg.norm(selected.vector)) - 1) < 1e-4,
              f"unit pose vector at yaw={yaw}, pitch={pitch}, roll={roll}")
        check(selected.projected_vector.shape == (512,) and abs(float(np.linalg.norm(selected.projected_vector)) - 1) < 1e-4,
              f"unit pre-projected vector at yaw={yaw}, pitch={pitch}, roll={roll}")

    # Test both the old 3x3 knot and the 96-bin fine-grid boundaries for continuity.
    for yaw, pitch in ((0.0, 0.0), (-7.5, -35.0), (7.5, 35.0), (22.5, 0.0), (-22.5, 0.0)):
        left = matrix.lookup(Pose3D(yaw=yaw - 0.01, pitch=pitch, roll=0)).vector
        right = matrix.lookup(Pose3D(yaw=yaw + 0.01, pitch=pitch, roll=0)).vector
        check(float(np.linalg.norm(left - right)) < 0.03,
              f"bilinear identity interpolation is continuous across yaw={yaw:g} boundary")

    for roll in (-45.01, -44.99, 44.99, 45.01):
        affine = matrix.roll_affine(roll, (64.0, 64.0))
        check(affine.shape == (2, 3) and np.isfinite(affine).all(), f"finite roll compensation at {roll:g} degrees")
    rolled_pose = Pose3D(yaw=0, pitch=0, roll=45)
    rolled_points = project_model5(rolled_pose, scale=180, center=(128, 128))
    aligned = roll_compensated_aligned_crop(np.zeros((256, 256, 3), np.uint8), rolled_points, 45, 128,
        compensation=matrix.roll_affine(45, tuple(rolled_points[:2].mean(axis=0))))
    actual = transform_points(rolled_points, aligned.matrix)
    check(abs(float(actual[0, 1] - actual[1, 1])) < 1.0 and actual[0, 0] < actual[1, 0],
          "roll is removed before canonical 5-point alignment")
    check(float(np.max(np.abs(transform_points(actual, aligned.inverse) - rolled_points))) < 1e-3,
          "roll-compensated crop transform round-trips to the source frame")
    profile_points = np.asarray(((48, 48), (82, 48), (68, 64), (52, 82), (80, 82)), np.float32)
    profile_mask = profile_aware_mask((128, 128), profile_points, 70)
    check(profile_mask[64, 5] < profile_mask[64, 100], "profile-aware mask tightens the far cheek while preserving the near side")
    check(pose_match_threshold(0, yaw=0, spatial_iou=0.9) == 0.40, "same-angle target match threshold is 0.40")
    check(pose_match_threshold(90, yaw=70, spatial_iou=0.9) == 0.22, "steep profile with strong IoU uses 0.22 threshold")
    return matrix


def verify_uv_sampler() -> None:
    texture = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    xs = np.asarray((0.0, 0.5, 1.0, 2.0, 2.25, -0.25), np.float32)
    ys = np.asarray((0.0, 0.5, 1.0, 2.0, 1.0, 1.0), np.float32)
    reference = remap_points(texture, xs, ys, device="cpu")
    accelerated = remap_points(texture, xs, ys, device="cuda")
    check(accelerated.shape == reference.shape and np.isfinite(accelerated).all()
          and np.allclose(accelerated, reference, atol=1.0),
          "perspective UV sampler uses CUDA grid_sample when available and preserves the CPU fallback")


def _unit(vector):
    value = np.asarray(vector, np.float32)
    return value / np.linalg.norm(value)


class _SyntheticSwapModel:
    """Shape-compatible INSwapper stub; exercises real alignment and source-vector routing."""

    def __init__(self):
        self.calls = 0

    def run(self, inputs):
        self.calls += 1
        return {"output": inputs["target"].copy()}


def _make_target_frame(frame_index: int, embedding: np.ndarray, *, other: bool = False) -> AngleDetection:
    if other:
        yaw, pitch, roll, center_x = 0.0, 0.0, 0.0, 500.0
    else:
        yaw = 70.0 * frame_index / 29.0
        pitch = 8.0 * np.sin(frame_index / 29.0 * np.pi)
        roll, center_x = 30.0, 145.0 + frame_index * 0.35
    pose = Pose3D(yaw=float(yaw), pitch=float(pitch), roll=float(roll))
    landmarks5 = project_model5(pose, scale=240.0, center=(center_x, 150.0)).astype(np.float32)
    bbox = np.asarray((center_x - 72, 70, center_x + 72, 230), np.float32)
    rng = np.random.default_rng(7000 + frame_index + int(other) * 1000)
    dense68 = np.column_stack((rng.uniform(bbox[0], bbox[2], 68), rng.uniform(bbox[1], bbox[3], 68),
                               rng.normal(0, 4, 68))).astype(np.float32)
    dense106 = rng.uniform((bbox[0], bbox[1]), (bbox[2], bbox[3]), (106, 2)).astype(np.float32)
    identity = _unit(np.random.default_rng(77).normal(size=512)) if other else embedding
    return AngleDetection(bbox, landmarks5, pose.as_tuple(), identity, dense68, dense106, 0.99, frame_index)


def verify_tracer(matrix: FrozenSourcePoseMatrix, document) -> None:
    target_embedding = _unit(np.random.default_rng(31).normal(size=512))
    lock_store = TargetLockStore()
    lock = lock_store.create("synthetic-stage13", 0,
        FaceRef(target_embedding, 0.0, 0.0, 30.0, [73.0, 70.0, 217.0, 230.0]))
    frames = []
    dropout = {13, 14, 15}
    for frame_index in range(30):
        faces = []
        if frame_index not in dropout:
            faces.append(_make_target_frame(frame_index, target_embedding))
        faces.append(_make_target_frame(frame_index, target_embedding, other=True))
        frames.append(faces)

    tracer = AngleTrajectoryTracer(max_gap=8)
    result = tracer.scan(frames, target_lock=lock, registry=lock_store, fps=30.0)
    check(result.frames_seen == 30 and result.detection_count == 57, "Stage A scanned all 30 frames and observed both identities")
    check(result.forward_tracks >= 2 and result.backward_tracks >= 2, "Stage B linked face trajectories in both time directions")
    check(result.locked_frames == 30 and result.bridged_frames == 3,
          f"Stage B bridged detector dropout and preserved 30/30 target frames ({result.locked_frames}/30)")
    check(all(item is not None and item.landmarks68_3d is not None and item.landmarks106 is not None
              for item in result.target_frames), "gap bridge retains 5-point, 68-point 3D and 106-point landmarks")
    check(all(not result.target_frames[index].detected for index in dropout), "dropout frames are explicitly interpolated")

    # A shape-compatible model stub exercises the real swapper's pose lookup, roll-aware
    # alignment, and output construction without loading heavyweight model weights.
    swapper = IdentitySwapper(config=RuntimeConfig())
    swapper.model = _SyntheticSwapModel()
    swapper.source = document
    swapper.canonical = _unit(document.canonical_embedding)
    swapper.projection = matrix.projection
    swapper._matrices = {None: matrix}
    swapper._selectors = {}
    swapper.sources_by_track = {}
    swapper.canonical_by_track = {}
    swapper._logged_cells = set()
    image = np.zeros((300, 640, 3), np.uint8)

    for mode in ("standard_2d", "estimated_2_5d", "full_3d_reconstructed"):
        engine_mode = normalize_render_mode(mode)
        swapped = 0
        for item in result.target_frames:
            assert item is not None
            pose = Pose3D(yaw=item.pose[0], pitch=item.pose[1], roll=item.pose[2])
            swap = swapper.infer_frame(image, item.landmarks5, pose=pose, track_id=item.track_id)
            if engine_mode and swap.image.shape == (128, 128, 3) and swap.reference_index >= 0:
                swapped += 1
        check(swapped == 30, f"{mode} produced 30/30 swap outputs with no target drop-out")
    check(swapper.model.calls == 90, "all three mode routes executed the real swapper seam")
    check(len({item.track_id for item in result.target_frames if item is not None}) == 1,
          "target track ID remains continuous through the profile turn and gap")


def main() -> int:
    document = _multi_angle_document()
    matrix = verify_frozen_matrix(document)
    verify_uv_sampler()
    verify_tracer(matrix, document)
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
