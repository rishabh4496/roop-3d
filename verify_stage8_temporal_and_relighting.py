"""Stage 8 synthetic acceptance: temporal stability, transfer, expression, and manual matte overrides."""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))

from roop_3d.compositor.occlusion import subtract_occluders
from roop_3d.compositor.pipeline import FrameCompositor
from roop_3d.compositor.stream_restoration import MaskSeamWatchdog
from roop_3d.engine.detection.face_tracker import FaceObservation, FaceTracker, TrackedFace
from roop_3d.reconstruction.expression_tps import layout_is_plausible, warp_expression
from roop_3d.relighting_3d import NeuralRelighter3D
from roop_3d.temporal import TemporalMaskEMA, TrackTemporalStabilizer


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expression_landmarks() -> np.ndarray:
    points = np.zeros((106, 2), np.float64)
    def ellipse(indices, cx, cy, rx, ry):
        angle = np.linspace(0, 2 * np.pi, len(indices), endpoint=False)
        points[list(indices)] = np.column_stack((cx + rx * np.cos(angle), cy + ry * np.sin(angle)))
    ellipse(range(33), 0, 0, 50, 57)
    ellipse(range(33, 43), -19, -10, 10, 3)
    ellipse(range(43, 52), -19, -19, 10, 2)
    ellipse(range(52, 72), 0, 20, 15, 6)
    ellipse(range(72, 87), 0, 5, 5, 8)
    ellipse(range(87, 97), 19, -10, 10, 3)
    ellipse(range(97, 106), 19, -19, 10, 2)
    return points


def main() -> int:
    rng = np.random.default_rng(8308)
    stabilizer = TrackTemporalStabilizer(min_cutoff=.2, beta=.03, process_noise=2, measurement_noise=4)
    base_box = np.array([20, 14, 92, 94], np.float32)
    base5 = np.array([[38, 42], [72, 42], [55, 58], [42, 73], [68, 73]], np.float32)
    base68 = np.column_stack((np.linspace(24, 88, 68), np.linspace(18, 91, 68))).astype(np.float32)
    base106 = np.column_stack((np.linspace(21, 91, 106), np.linspace(15, 93, 106))).astype(np.float32)
    box_raw, box_out, lm5_raw, lm5_out, lm68_raw, lm68_out, lm106_raw, lm106_out = ([] for _ in range(8))
    pose_raw, pose_out, shape_raw, shape_out, expression_raw, expression_out = ([] for _ in range(6))
    tracker = FaceTracker.__new__(FaceTracker)
    tracker.stabilize = True
    tracker.frame_index = 0
    tracker.timestamp_seconds = 0.0
    tracker.stabilizer = stabilizer
    for frame in range(10):
        timestamp = frame / 30.0
        noise = rng.normal(0, 1.5, size=4).astype(np.float32)
        box = base_box + noise
        points5 = base5 + rng.normal(0, 1.5, base5.shape).astype(np.float32)
        points68 = base68 + rng.normal(0, 1.5, base68.shape).astype(np.float32)
        points106 = base106 + rng.normal(0, 1.5, base106.shape).astype(np.float32)
        shape = np.full(100, .2, np.float32) + rng.normal(0, .08, 100).astype(np.float32)
        expression = np.full(50, .1, np.float32) + rng.normal(0, .12, 50).astype(np.float32)
        pose = (2 + float(rng.normal(0, 1.8)), -1 + float(rng.normal(0, 1.8)), float(rng.normal(0, 1.8)))
        embedding = np.zeros(512, np.float32); embedding[0] = 1
        raw = FaceObservation(box, points5, points106, embedding, pose,
                              landmarks68=points68, shape_coefficients=shape,
                              expression_coefficients=expression)
        tracker.timestamp_seconds = timestamp
        tracked = tracker._stabilize([TrackedFace(7, raw, frame + 1, 0)], {7: (raw, 0, frame + 1)})[0]
        filtered = tracked.observation
        box_raw.append(box); box_out.append(filtered.box)
        lm5_raw.append(points5); lm5_out.append(filtered.landmarks5)
        lm68_raw.append(points68); lm68_out.append(filtered.landmarks68)
        lm106_raw.append(points106); lm106_out.append(filtered.landmarks106)
        pose_raw.append(pose); pose_out.append(filtered.pose)
        shape_raw.append(shape); shape_out.append(filtered.shape_coefficients)
        expression_raw.append(expression); expression_out.append(filtered.expression_coefficients)

    variance = lambda values: float(np.var(np.asarray(values), axis=0).mean())
    reductions = {
        "bbox": variance(box_out) / variance(box_raw),
        "landmarks5": variance(lm5_out) / variance(lm5_raw),
        "landmarks68": variance(lm68_out) / variance(lm68_raw),
        "landmarks106": variance(lm106_out) / variance(lm106_raw),
        "pose": variance(pose_out) / variance(pose_raw),
        "shape_coefficients": variance(shape_out) / variance(shape_raw),
        "expression_coefficients": variance(expression_out) / variance(expression_raw),
    }
    check(all(ratio < .8 for ratio in reductions.values()), f"Temporal variance was not reduced enough: {reductions}")

    from roop_3d.engine.stage16_pipeline import Stage16Options, Stage16Pipeline
    from roop_3d.flame_3d_engine import EyeGaze, Face3D, FlameParameters
    from roop_3d.reconstruction.geometry_types import euler_to_rotation_matrix
    class SyntheticFlame:
        def __init__(self): self.raw_psi = []
        def extract(self, rgb, *, require_gaze=False):
            if float(np.mean(rgb)) < 100:
                params = FlameParameters(np.ones(100), np.zeros(50), np.zeros(3), np.eye(3))
            else:
                psi = np.full(50, .35, np.float32) + rng.normal(0, .12, 50).astype(np.float32)
                self.raw_psi.append(psi)
                params = FlameParameters(np.full(100, 2., np.float32) + rng.normal(0, .1, 100).astype(np.float32),
                    psi, rng.normal(0, .05, 3).astype(np.float32),
                    euler_to_rotation_matrix(*rng.normal(0, 1.5, 3)),
                    gaze=EyeGaze(*rng.normal(0, .08, 4)))
            return Face3D(params, None)
    synthetic_flame = SyntheticFlame()
    class SyntheticSwapper:
        def swap(self, image, embedding): return type('Swap', (), {'rgb': image.copy(), 'mask': np.ones(image.shape[:2], np.uint8)})()
    class SyntheticPortrait:
        def retarget(self, image, driving, **kwargs): return image
    class ShapePass:
        def condition(self, image, source, target): return image
    stage16 = Stage16Pipeline(synthetic_flame, SyntheticSwapper(), SyntheticPortrait(), shape=ShapePass())
    stage16_psi = []
    for frame in range(10):
        source_rgb = np.zeros((256, 256, 3), np.uint8)
        target_rgb = np.full((256, 256, 3), 150, np.uint8)
        result = stage16.process(source_rgb, target_rgb, np.zeros(512, np.float32),
            options=Stage16Options(shape_retention_weight=1), track_id=7, timestamp_seconds=frame / 30)
        stage16_psi.append(result.flame.psi)
    coefficient_ratio = variance(stage16_psi) / variance(synthetic_flame.raw_psi)
    check(coefficient_ratio < .8, f"Stage16 3DMM expression coefficients were not stabilized: {coefficient_ratio}")
    reductions["stage16_3dmm_expression"] = coefficient_ratio

    mask_ema = TemporalMaskEMA(alpha=.65)
    raw_masks, stable_masks = [], []
    for frame in range(10):
        mask = np.zeros((64, 64), np.float32)
        left = 18 + int(round(rng.normal(0, 1.2)))
        mask[14:50, left:left + 28] = 1
        raw_masks.append(mask)
        stable_masks.append(mask_ema.update(7, mask, frame=frame, kind="xseg"))
    mask_variance_ratio = variance(stable_masks) / variance(raw_masks)
    check(mask_variance_ratio < 1, f"Temporal mask EMA did not reduce jitter: {mask_variance_ratio}")

    watchdog = MaskSeamWatchdog()
    geometric = np.zeros((64, 64), np.float32); geometric[10:54, 10:54] = 1
    raw_occluder = np.zeros_like(geometric); raw_occluder[26:38, 2:62] = .8
    watchdog.observe(7, geometric, kind="geometric")
    baseline_drift = watchdog.observe(7, raw_occluder, kind="occluder")
    check(baseline_drift == 0, "Occluder mask was compared with the geometric face baseline")
    feathered = watchdog.feather(7, np.clip(geometric * (1 - raw_occluder), 0, 1), kind="composite")
    check(feathered.shape == geometric.shape and np.isfinite(feathered).all(), "Seam watchdog produced an invalid mask")

    try:
        import torch
    except ImportError:
        from roop_3d.compositor.color_match import estimate_lab_stats, apply_lab_stats
        source_bgr = np.full((32, 32, 3), (82, 82, 82), np.uint8)
        target_bgr = np.empty_like(source_bgr); target_bgr[:] = (102, 122, 158)
        skin_np = np.ones((32, 32), np.float32)
        stats = estimate_lab_stats(source_bgr, target_bgr, skin_np)
        matched_np = apply_lab_stats(source_bgr, stats, skin_np)
        lab_before = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).mean((0, 1))
        lab_target = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float32).mean((0, 1))
        lab_after = cv2.cvtColor(matched_np, cv2.COLOR_BGR2LAB).astype(np.float32).mean((0, 1))
        check(np.isfinite(matched_np).all() and np.linalg.norm(lab_after - lab_target) < np.linalg.norm(lab_before - lab_target),
              "Reinhard LAB transfer did not move source toward target skin colour")
        lab_backend = "CPU fallback (Torch unavailable)"
    else:
        from roop_3d.gpu_compositor import color_transfer, rgb_to_lab
        source = torch.full((1, 3, 32, 32), .32, dtype=torch.float32)
        target = torch.empty_like(source); target[:, 0].fill_(.62); target[:, 1].fill_(.48); target[:, 2].fill_(.40)
        skin = torch.ones((1, 1, 32, 32), dtype=torch.float32)
        matched = color_transfer(source, target, skin, strength=1.0)
        lab_before = rgb_to_lab(source).mean((-2, -1))
        lab_target = rgb_to_lab(target).mean((-2, -1))
        lab_after = rgb_to_lab(matched).mean((-2, -1))
        check(torch.isfinite(matched).all().item() and torch.linalg.vector_norm(lab_after - lab_target) < torch.linalg.vector_norm(lab_before - lab_target),
              "Reinhard LAB tensor transfer did not move the source toward target skin colour")
        lab_backend = "GPU tensor implementation" if torch.cuda.is_available() else "Torch tensor implementation"

    height, width = 64, 64
    y, x = np.mgrid[0:height, 0:width]
    swapped = np.stack((70 + x, 90 + y, np.full_like(x, 120)), axis=-1).clip(0, 255).astype(np.uint8)
    target_bgr = np.stack((40 + 2 * x, 60 + y, 100 + x // 2), axis=-1).clip(0, 255).astype(np.uint8)
    face = np.zeros((height, width), np.float32); face[5:-5, 5:-5] = 1
    relit, normals, shading, metadata = NeuralRelighter3D().relight(swapped, target_bgr, skin_mask=face,
        face_mask=face, relight_strength=.8, specular_sheen=.1, shadow_preservation=.5)
    check(relit.shape == swapped.shape and np.isfinite(relit).all() and np.isfinite(normals).all()
          and np.isfinite(shading).all() and "sh_coeffs" in metadata, "SH relighting returned invalid output")

    src106 = expression_landmarks()
    check(layout_is_plausible(src106), "Synthetic expression landmark layout is invalid")
    dst106 = src106.copy(); dst106[52:72, 1] += 4; dst106[33:43, 1] += 1
    canvas = np.zeros((128, 128, 3), np.uint8)
    cv2.circle(canvas, (64, 64), 30, (120, 160, 190), -1)
    expression = warp_expression(canvas, src106 + 64, dst106 + 64, .75)
    check(expression.applied and expression.image.shape == canvas.shape and np.any(expression.image != canvas),
          "Target micro-expression landmarks were not restored")

    include = np.zeros((32, 32), np.float32); include[5:12, 5:12] = .8
    exclude = np.zeros_like(include); exclude[8:10, 8:10] = 1
    decoded_include, decoded_exclude = FrameCompositor._manual_mask((include, exclude), include.shape)
    for frame in range(10):
        base_alpha = np.full(include.shape, .2, np.float32)
        composite = subtract_occluders(np.maximum(base_alpha, decoded_include), decoded_exclude)
        check(abs(float(composite[6, 6]) - .8) < 1e-5 and composite[8, 8] == 0,
              f"Manual include/exclude overrides were not blended on frame {frame}")

    print({"status": "PASS", "frames": 10, "temporal_variance_ratio": reductions,
           "mask_variance_ratio": mask_variance_ratio, "reinhard_lab": "PASS (" + lab_backend + ")", "sh_relighting": "PASS",
           "expression_restoration": "PASS", "manual_mask_override": "PASS"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
