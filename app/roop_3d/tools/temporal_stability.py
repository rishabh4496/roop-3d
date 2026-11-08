"""Reproducible 120-frame synthetic benchmark for temporal detail stabilization.

Run from ``app/`` with the project Python environment:
    python -m roop_3d.tools.temporal_stability --frames 120

This is a controlled algorithm benchmark, not a claim about a particular source video,
GPU, UltraMax weights, or whole-render quality. It reports the requested 70% gate honestly.
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

from roop_3d.compositor.detail_temporal import TemporalDetailStabilizer


def _highpass(image: np.ndarray) -> np.ndarray:
    value = image.astype(np.float32)
    return value - cv2.GaussianBlur(value, (0, 0), 1.15, borderType=cv2.BORDER_REFLECT101)


def benchmark_temporal_stability(*, frames: int = 120, seed: int = 7, side: int = 128,
                                 cut_frame: int = 60, flow_side: int = 96,
                                 temporal_alpha: float = .25) -> dict:
    if (frames < 3 or side < 32 or flow_side < 16 or not 0 <= cut_frame < frames
            or not 0 <= temporal_alpha <= .95):
        raise ValueError("Use >=3 frames, side >= 32, flow_side >= 16, temporal_alpha in 0..0.95, and an in-range cut frame")
    rng = np.random.default_rng(seed)
    # Smooth facial guide with a stable fine texture for measurable subpixel flow.
    guide_noise = rng.normal(128, 32, (side, side)).clip(0, 255).astype(np.uint8)
    guide_base = cv2.GaussianBlur(guide_noise, (0, 0), 1.2)
    guide_base = cv2.cvtColor(guide_base, cv2.COLOR_GRAY2BGR)
    pore_field = rng.normal(0, 8, (side, side, 3)).astype(np.float32)
    cut_guide_noise = rng.integers(0, 256, (side, side), dtype=np.uint8)
    cut_guide_base = cv2.cvtColor(cv2.GaussianBlur(cut_guide_noise, (0, 0), 1.0), cv2.COLOR_GRAY2BGR)
    cut_pore_field = rng.normal(0, 8, (side, side, 3)).astype(np.float32)
    base = np.full((side, side, 3), 128, np.uint8)
    stabilizer = TemporalDetailStabilizer(temporal_alpha=temporal_alpha, flow_side=flow_side, max_side=side,
                                          use_cuda_flow=False)
    raw_residuals, filtered_residuals, outputs, inputs, reports = [], [], [], [], []
    for index in range(frames):
        # Keep a separate, persistent shot after the cut. Motion remains known exactly.
        post_cut = index >= cut_frame
        source_guide = cut_guide_base if post_cut else guide_base
        source_detail = cut_pore_field if post_cut else pore_field
        offset = 0.35 * (index - cut_frame if post_cut else index)
        affine = np.array(((1, 0, offset), (0, 1, 0.18 * np.sin(index / 9))), np.float32)
        guide = cv2.warpAffine(source_guide, affine, (side, side), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT101)
        stable_detail = cv2.warpAffine(source_detail, affine, (side, side), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_REFLECT101)
        noise = rng.normal(0, 7, (side, side, 3)).astype(np.float32)
        enhanced = np.clip(base.astype(np.float32) + stable_detail + noise, 0, 255).round().astype(np.uint8)
        filtered, report = stabilizer.update(0, enhanced, base, guide, frame_index=index,
                                             scene_cut=index == cut_frame)
        reports.append(report)
        outputs.append(filtered)
        inputs.append(enhanced)
        raw_residuals.append(_highpass(enhanced) - _highpass(base))
        filtered_residuals.append(_highpass(filtered) - _highpass(base))

    raw_warp, filtered_warp, raw_lap, filtered_lap = [], [], [], []
    cut_ghost = int(np.abs(outputs[cut_frame].astype(np.int16) - inputs[cut_frame].astype(np.int16)).max())
    for index in range(1, frames):
        if index in (cut_frame, cut_frame + 1):
            continue
        dx, dy = .35, .18 * (np.sin(index / 9) - np.sin((index - 1) / 9))
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        # Current pixel samples the previous image at p - forward translation.
        map_x, map_y = xx - dx, yy - dy
        valid = (map_x >= 2) & (map_x < side - 2) & (map_y >= 2) & (map_y < side - 2)
        for current, errors, lap_errors in (
            (raw_residuals, raw_warp, raw_lap),
            (filtered_residuals, filtered_warp, filtered_lap),
        ):
            warped = cv2.remap(current[index - 1], map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            delta = current[index] - warped
            errors.append(float(np.abs(delta[valid]).mean()))
            lap_now = cv2.Laplacian(current[index], cv2.CV_32F, ksize=3)
            lap_prev = cv2.Laplacian(warped, cv2.CV_32F, ksize=3)
            lap_errors.append(float(np.abs(lap_now[valid] - lap_prev[valid]).mean()))

    raw_e, filtered_e = float(np.mean(raw_warp)), float(np.mean(filtered_warp))
    raw_l, filtered_l = float(np.mean(raw_lap)), float(np.mean(filtered_lap))
    warp_reduction = 100 * (raw_e - filtered_e) / max(raw_e, 1e-9)
    lap_reduction = 100 * (raw_l - filtered_l) / max(raw_l, 1e-9)
    return {
        "frames": frames,
        "synthetic_only": True,
        "backend": next((item["backend"] for item in reports
                          if item["backend"] != "none_no_history"), "none"),
        "temporal_alpha": temporal_alpha,
        "history_weight": 1.0 - temporal_alpha,
        "flow_side": flow_side,
        "cut_frame": cut_frame,
        "flow_confidence_mean": float(np.mean([item["flow_confidence"] for item in reports])),
        "history_influenced_frames": int(sum(item["influenced_fraction"] > 0 for item in reports)),
        "reset_reasons": {reason: sum(item["reset_reason"] == reason for item in reports)
                          for reason in sorted({item["reset_reason"] for item in reports if item["reset_reason"]})},
        "temporal_warp_error": {"raw": raw_e, "stabilized": filtered_e,
                                 "reduction_percent": warp_reduction},
        "laplacian_flicker": {"raw": raw_l, "stabilized": filtered_l,
                               "reduction_percent": lap_reduction},
        "scene_cut_max_pixel_delta": cut_ghost,
        "scene_cut_ghosting_free": cut_ghost == 0.0,
        "requested_70_percent_gate_met": warp_reduction >= 70.0 and lap_reduction >= 70.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--side", type=int, default=128)
    parser.add_argument("--cut-frame", type=int, default=60)
    parser.add_argument("--flow-side", type=int, default=96)
    parser.add_argument("--temporal-alpha", type=float, default=.25)
    args = parser.parse_args()
    print(json.dumps(benchmark_temporal_stability(frames=args.frames, seed=args.seed,
                                                   side=args.side, cut_frame=args.cut_frame,
                                                   flow_side=args.flow_side,
                                                   temporal_alpha=args.temporal_alpha), indent=2))


if __name__ == "__main__":
    main()
