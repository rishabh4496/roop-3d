"""Physical camera matching, CC0 lighting probes, and UV-flow texture denoising.

This module stays renderer-neutral and uses OpenCV/NumPy only. The compact
environment probes in ``assets/stage10_cc0_probes.npz`` are low-resolution,
linear-light reductions of the CC0 Poly Haven HDRIs listed in the adjacent
provenance JSON. The tileable pore normal is deterministic procedural CC0 data;
ambientCG does not currently publish a human-skin pore material, so no unrelated
surface texture is represented as skin.

Call ``PhysicalCameraMatcher.match_rendered_head`` on the isolated rendered
head layer before passing that layer and its alpha to the final compositor.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from roop_3d.rendering.physical_buffers import evaluate_sh_basis

SENSOR_WIDTH_MM_DEFAULT = 36.0
FOCAL_RANGE_MM = (18.0, 135.0)
PROBE_BUNDLE_PATH = Path(__file__).resolve().parent / "assets" / "stage10_cc0_probes.npz"
PROBE_BUNDLE_SCHEMA = "roop3d.stage10.cc0_preview_assets.v1"
ENVIRONMENT_PROBES = (
    {"name": "Neutral Studio", "asset_id": "studio_small_08", "kelvin": 6000.0,
     "md5": "de3ba64222895aca876b1d1c2e0cf81a", "page_url": "https://polyhaven.com/a/studio_small_08"},
    {"name": "Warm Indoor", "asset_id": "studio_small_09", "kelvin": 2750.0,
     "md5": "d5d7eb9d26d341d6aa4d11c1a54fd97c", "page_url": "https://polyhaven.com/a/studio_small_09"},
    {"name": "Cool Overcast", "asset_id": "white_studio_06", "kelvin": 5500.0,
     "md5": "d9950d6316c57b1b92494d909bd1a92a", "page_url": "https://polyhaven.com/a/white_studio_06"},
    {"name": "Golden Hour", "asset_id": "signal_hill_sunrise", "kelvin": 5400.0,
     "md5": "6245187232b805d255988ae27ed31e1f", "page_url": "https://polyhaven.com/a/signal_hill_sunrise"},
    {"name": "Cinematic Rim", "asset_id": "studio_small_04", "kelvin": 5500.0,
     "md5": "4d58f1d1b8f919f2bb4de1b748d11270", "page_url": "https://polyhaven.com/a/studio_small_04"},
)


@dataclass(frozen=True)
class OpticalMeasurements:
    sharpness_variance: float
    motion_blur_vector_px: tuple[float, float]
    blue_edge_shift_px: float
    red_edge_shift_px: float
    grain_sigma: float
    estimated_iso: int


@dataclass(frozen=True)
class OpticalMatchInfo:
    measurements: OpticalMeasurements
    focal_length_mm: float
    depth_of_field_sigma_px: float
    directional_blur_length_px: float
    applied_grain_sigma: float


@dataclass(frozen=True)
class RimLightLobe:
    direction_xyz: tuple[float, float, float]
    intensity: float
    angular_radius_deg: float
    color_bgr: tuple[float, float, float]


@dataclass(frozen=True)
class KeyLightEstimate:
    direction_xyz: tuple[float, float, float]
    azimuth_deg: float
    elevation_deg: float
    intensity: float
    kelvin_temp: float
    sh_coefficients: np.ndarray
    rim_lobes: tuple[RimLightLobe, ...]


@dataclass(frozen=True)
class EnvironmentProbe:
    name: str
    asset_id: str
    environment_bgr: np.ndarray  # compact equirectangular radiance map, linear BGR
    sh_coefficients: np.ndarray  # second-order 9x3 coefficients in BGR order
    white_balance_kelvin: float
    license_id: str = "CC0-1.0"


@dataclass(frozen=True)
class CC0PreviewAssets:
    probes: tuple[EnvironmentProbe, ...]
    micro_pore_normal: np.ndarray  # tangent-space xyz in [-1, 1]
    metadata: Mapping[str, Any]


class PhysicalCameraMatcher:
    """Fit pinhole camera scale and reproduce the target shot's optical finish."""

    def __init__(
        self,
        focal_length_mm: float = 50.0,
        *,
        sensor_width_mm: float = SENSOR_WIDTH_MM_DEFAULT,
        focal_range_mm: tuple[float, float] = FOCAL_RANGE_MM,
    ) -> None:
        if sensor_width_mm <= 0 or focal_range_mm[0] <= 0 or focal_range_mm[1] <= focal_range_mm[0]:
            raise ValueError("invalid sensor or focal-length bounds")
        self.sensor_width_mm = float(sensor_width_mm)
        self.focal_range_mm = tuple(map(float, focal_range_mm))
        self.focal_length_mm = self._clamp_focal(focal_length_mm)

    def _clamp_focal(self, focal_length_mm: float) -> float:
        if not math.isfinite(focal_length_mm):
            raise ValueError("focal_length_mm must be finite")
        return float(np.clip(focal_length_mm, *self.focal_range_mm))

    @staticmethod
    def estimate_focal_length_mm(
        camera_points_xyz: np.ndarray,
        observed_pixels_xy: np.ndarray,
        image_shape: tuple[int, ...],
        *,
        sensor_width_mm: float = SENSOR_WIDTH_MM_DEFAULT,
        principal_point_xy: tuple[float, float] | None = None,
        focal_range_mm: tuple[float, float] = FOCAL_RANGE_MM,
    ) -> float:
        """Robustly fit focal length from tracked metric 3D points and 2D pixels.

        Points must already be in the tracked camera frame, with positive z.
        This makes perspective scale identifiable from the head's metric shape
        instead of conflating focal length and arbitrary model scale.
        """
        xyz = np.asarray(camera_points_xyz, dtype=np.float64)
        pixels = np.asarray(observed_pixels_xy, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or pixels.shape != (len(xyz), 2) or len(xyz) < 4:
            raise ValueError("camera_points_xyz and observed_pixels_xy must have shapes (N,3)/(N,2), N >= 4")
        if not np.isfinite(xyz).all() or not np.isfinite(pixels).all() or np.any(xyz[:, 2] <= 1e-6):
            raise ValueError("camera points and observed pixels must be finite and in front of the camera")
        height, width = int(image_shape[0]), int(image_shape[1])
        if width <= 0 or height <= 0 or sensor_width_mm <= 0:
            raise ValueError("image dimensions and sensor width must be positive")
        cx, cy = principal_point_xy or (width * 0.5, height * 0.5)
        rays = np.concatenate((xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]))
        offsets = np.concatenate((pixels[:, 0] - cx, pixels[:, 1] - cy))
        weights = np.ones_like(rays)
        focal_px = 0.0
        for _ in range(8):
            denominator = float(np.sum(weights * rays * rays))
            if denominator < 1e-12:
                raise ValueError("tracked points do not constrain focal length")
            focal_px = float(np.sum(weights * rays * offsets) / denominator)
            residual = offsets - focal_px * rays
            scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
            weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(residual), 1e-9))
        focal_mm = focal_px * float(sensor_width_mm) / width
        return float(np.clip(focal_mm, *focal_range_mm))

    def estimate_focal_length(
        self,
        camera_points_xyz: np.ndarray,
        observed_pixels_xy: np.ndarray,
        image_shape: tuple[int, ...],
        **kwargs: Any,
    ) -> float:
        self.focal_length_mm = self.estimate_focal_length_mm(
            camera_points_xyz, observed_pixels_xy, image_shape,
            sensor_width_mm=self.sensor_width_mm, focal_range_mm=self.focal_range_mm, **kwargs,
        )
        return self.focal_length_mm

    @staticmethod
    def estimate_radial_distortion(
        camera_points_xyz: np.ndarray,
        observed_pixels_xy: np.ndarray,
        focal_length_mm: float,
        image_shape: tuple[int, ...],
        *,
        sensor_width_mm: float = SENSOR_WIDTH_MM_DEFAULT,
        principal_point_xy: tuple[float, float] | None = None,
    ) -> tuple[float, float]:
        """Least-squares estimate of OpenCV radial k1/k2 from tracked points."""
        xyz = np.asarray(camera_points_xyz, np.float64)
        observed = np.asarray(observed_pixels_xy, np.float64)
        h, w = int(image_shape[0]), int(image_shape[1])
        if xyz.shape != (len(observed), 3) or observed.ndim != 2 or observed.shape[1] != 2 or len(xyz) < 6:
            raise ValueError("at least six corresponding camera points and pixels are required")
        if np.any(xyz[:, 2] <= 1e-6) or not np.isfinite(xyz).all() or not np.isfinite(observed).all():
            raise ValueError("distortion inputs must be finite and in front of the camera")
        fx = float(focal_length_mm) / sensor_width_mm * w
        fy = float(focal_length_mm) / (sensor_width_mm * h / w) * h
        cx, cy = principal_point_xy or (w * 0.5, h * 0.5)
        x, y = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
        radius2 = x * x + y * y
        design = np.concatenate((np.stack((fx * x * radius2, fx * x * radius2 ** 2), axis=1),
                                 np.stack((fy * y * radius2, fy * y * radius2 ** 2), axis=1)), axis=0)
        ideal = np.concatenate((fx * x + cx, fy * y + cy))
        actual = np.concatenate((observed[:, 0], observed[:, 1]))
        solution, *_ = np.linalg.lstsq(design, actual - ideal, rcond=None)
        return float(solution[0]), float(solution[1])

    @staticmethod
    def project_points(
        camera_points_xyz: np.ndarray,
        image_shape: tuple[int, ...],
        focal_length_mm: float,
        *,
        sensor_width_mm: float = SENSOR_WIDTH_MM_DEFAULT,
        principal_point_xy: tuple[float, float] | None = None,
        distortion: tuple[float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    ) -> np.ndarray:
        """Pinhole projection with OpenCV-compatible radial/tangential distortion."""
        xyz = np.asarray(camera_points_xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all() or np.any(xyz[:, 2] <= 1e-6):
            raise ValueError("camera_points_xyz must be finite (N,3) points with positive z")
        height, width = int(image_shape[0]), int(image_shape[1])
        if width <= 0 or height <= 0 or focal_length_mm <= 0 or sensor_width_mm <= 0:
            raise ValueError("image dimensions, focal length and sensor width must be positive")
        fx = float(focal_length_mm) / float(sensor_width_mm) * width
        fy = float(focal_length_mm) / (float(sensor_width_mm) * height / width) * height
        cx, cy = principal_point_xy or (width * 0.5, height * 0.5)
        x, y = xyz[:, 0] / xyz[:, 2], xyz[:, 1] / xyz[:, 2]
        r2 = x * x + y * y
        k1, k2, p1, p2, k3 = map(float, distortion)
        radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.stack((fx * xd + cx, fy * yd + cy), axis=1).astype(np.float32)

    @staticmethod
    def measure_target_optics(
        frame_bgr: np.ndarray,
        previous_frame_bgr: np.ndarray | None = None,
        *,
        mask: np.ndarray | None = None,
    ) -> OpticalMeasurements:
        """Measure target sharpness, inter-frame flow, edge CA, and fine sensor noise."""
        frame = _image_u8(frame_bgr, "frame_bgr")
        height, width = frame.shape[:2]
        valid = np.ones((height, width), bool) if mask is None else np.asarray(mask) > 0
        if valid.shape != (height, width):
            raise ValueError("mask must match the target frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        sharpness = float(np.var(lap[valid])) if np.any(valid) else 0.0

        flow_vector = (0.0, 0.0)
        if previous_frame_bgr is not None:
            previous = _image_u8(previous_frame_bgr, "previous_frame_bgr")
            if previous.shape != frame.shape:
                raise ValueError("adjacent frames must have identical dimensions")
            prev_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 19, 3, 5, 1.1, 0)
            flow_samples = flow[valid]
            if len(flow_samples):
                magnitude = np.linalg.norm(flow_samples, axis=1)
                keep = magnitude <= np.percentile(magnitude, 90)
                robust = flow_samples[keep] if np.any(keep) else flow_samples
                median = np.median(robust, axis=0)
                flow_vector = (float(median[0]), float(median[1]))

        blue_shift, red_shift = _estimate_chromatic_edge_shifts(frame, valid)
        grain_sigma, estimated_iso = _estimate_sensor_noise(frame, valid)
        # Laplacian variance includes sensor noise; subtract its expected
        # response for the 3x3 OpenCV Laplacian before using it as a blur target.
        sharpness = max(0.0, sharpness - 20.0 * grain_sigma * grain_sigma)
        return OpticalMeasurements(sharpness, flow_vector, blue_shift, red_shift, grain_sigma, estimated_iso)

    def match_rendered_head(
        self,
        rendered_head: np.ndarray,
        target_frame_bgr: np.ndarray,
        *,
        previous_frame_bgr: np.ndarray | None = None,
        alpha: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        focus_distance_m: float | None = None,
        f_number: float = 2.8,
        shutter_angle_deg: float = 180.0,
        target_mask: np.ndarray | None = None,
        random_seed: int = 0,
    ) -> tuple[np.ndarray, OpticalMatchInfo]:
        """Apply target-matched optics to an isolated rendered BGR/BGRA head.

        ``depth_m`` is camera distance in metres; ``focus_distance_m`` and
        ``f_number`` enable spatially varying lens-circle-of-confusion blur.
        The returned image keeps its original channel count and should be
        composited only after this method returns.
        """
        source = np.asarray(rendered_head)
        if source.ndim != 3 or source.shape[2] not in (3, 4) or source.dtype not in (np.uint8, np.float32, np.float64):
            raise ValueError("rendered_head must be uint8/float BGR or BGRA")
        target = _image_u8(target_frame_bgr, "target_frame_bgr")
        if source.shape[:2] != target.shape[:2]:
            raise ValueError("rendered head and target frame must have equal dimensions")
        if f_number <= 0 or not 0.0 <= shutter_angle_deg <= 360.0:
            raise ValueError("f_number must be positive and shutter angle within 0..360 degrees")
        height, width = source.shape[:2]
        float_scale = 255.0 if source.dtype == np.uint8 or float(np.nanmax(source[..., :3])) > 1.5 else 1.0
        rgb = source[..., :3].astype(np.float32) / float_scale
        if alpha is None and source.shape[2] == 4:
            alpha_values = source[..., 3].astype(np.float32) / (255.0 if source.dtype == np.uint8 or np.nanmax(source[..., 3]) > 1.5 else 1.0)
        elif alpha is None:
            alpha_values = np.ones((height, width), np.float32)
        else:
            alpha_values = np.array(alpha, dtype=np.float32, copy=True)
            if alpha_values.shape != (height, width):
                raise ValueError("alpha must match rendered_head dimensions")
            if alpha_values.max(initial=0.0) > 1.5:
                alpha_values /= 255.0
        alpha_values = np.clip(alpha_values, 0.0, 1.0)

        face_mask = target_mask if target_mask is not None else (alpha_values > 0.05)
        measurements = self.measure_target_optics(target, previous_frame_bgr, mask=face_mask)
        sigma_match = _sharpness_match_sigma(rgb, alpha_values, target, measurements.sharpness_variance)

        if depth_m is not None:
            depth = np.asarray(depth_m, np.float32)
            if depth.shape != (height, width):
                raise ValueError("depth_m must match rendered_head dimensions")
            if focus_distance_m is None or focus_distance_m <= 0 or focus_distance_m <= self.focal_length_mm / 1000.0:
                raise ValueError("positive focus_distance_m beyond the focal plane is required with depth_m")
            sigma_dof = _circle_of_confusion_sigma(depth, alpha_values, focus_distance_m, f_number,
                                                    self.focal_length_mm, self.sensor_width_mm, width)
            sigma_map = np.sqrt(sigma_dof * sigma_dof + sigma_match * sigma_match)
        else:
            sigma_dof = np.zeros((height, width), np.float32)
            sigma_map = np.full((height, width), sigma_match, np.float32)
        premultiplied = rgb * alpha_values[..., None]
        optical_layer = np.concatenate((premultiplied, alpha_values[..., None]), axis=2)
        optical_layer = _spatial_blur(optical_layer, sigma_map)
        premultiplied, alpha_values = optical_layer[..., :3], optical_layer[..., 3]

        flow_x, flow_y = measurements.motion_blur_vector_px
        flow_magnitude = math.hypot(flow_x, flow_y)
        blur_length = min(64.0, flow_magnitude * shutter_angle_deg / 360.0)
        if blur_length > 0.35:
            direction = (flow_x / max(flow_magnitude, 1e-8), flow_y / max(flow_magnitude, 1e-8))
            optical_layer = np.concatenate((premultiplied, alpha_values[..., None]), axis=2)
            optical_layer = _directional_blur(optical_layer, direction, blur_length)
            premultiplied, alpha_values = optical_layer[..., :3], optical_layer[..., 3]
        alpha_values = np.clip(alpha_values, 0.0, 1.0)
        color = premultiplied / np.maximum(alpha_values[..., None], 1e-5)
        color = _apply_chromatic_aberration(color, alpha_values, measurements.blue_edge_shift_px,
                                            measurements.red_edge_shift_px)
        applied_grain = min(24.0, measurements.grain_sigma)
        if applied_grain > 0.02:
            rng = np.random.default_rng(random_seed)
            grain_unit = applied_grain / 255.0
            shared = rng.normal(0.0, grain_unit, (height, width, 1)).astype(np.float32)
            chroma = rng.normal(0.0, grain_unit * 0.28, (height, width, 3)).astype(np.float32)
            color += (shared + chroma) * alpha_values[..., None]
        color = np.clip(color, 0.0, 1.0)
        output = color * float_scale
        if source.shape[2] == 4:
            alpha_out = alpha_values * (255.0 if source.dtype == np.uint8 or np.nanmax(source[..., 3]) > 1.5 else 1.0)
            output = np.concatenate((output, alpha_out[..., None]), axis=2)
        output = np.rint(output).clip(0, 255).astype(np.uint8) if source.dtype == np.uint8 else output.astype(source.dtype)
        dof_value = float(np.max(sigma_dof[alpha_values > 0.01])) if np.any(alpha_values > 0.01) else 0.0
        info = OpticalMatchInfo(measurements, self.focal_length_mm, dof_value, blur_length, applied_grain)
        return output, info


class DominantKeyLightAligner:
    """Estimate a directional key and bright rim lobes from a face SH fit."""

    def estimate(
        self,
        target_frame_bgr: np.ndarray,
        normals: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        regularization: float = 0.02,
    ) -> KeyLightEstimate:
        image = _image_u8(target_frame_bgr, "target_frame_bgr")
        n = np.asarray(normals, dtype=np.float32)
        if n.shape != (*image.shape[:2], 3) or not np.isfinite(n).all():
            raise ValueError("normals must be finite and match image dimensions with xyz channels")
        if regularization < 0:
            raise ValueError("regularization must be nonnegative")
        valid = np.ones(image.shape[:2], bool) if mask is None else np.asarray(mask) > 0
        if valid.shape != image.shape[:2]:
            raise ValueError("mask must match target frame")
        valid &= np.linalg.norm(n, axis=2) > 1e-6
        if int(valid.sum()) < 32:
            raise ValueError("at least 32 valid normal samples are required for SH fitting")
        linear = _srgb_to_linear(image.astype(np.float32) / 255.0)
        basis = evaluate_sh_basis(n[valid])
        weight = np.sqrt(np.clip(np.linalg.norm(n[valid], axis=1), 0.0, 1.0))
        a = basis * weight[:, None]
        b = linear[valid] * weight[:, None]
        gram = a.T @ a + (regularization * len(a) + 1e-6) * np.eye(9, dtype=np.float32)
        sh = np.linalg.solve(gram, a.T @ b).astype(np.float32)
        luma_weights = np.array([0.114, 0.587, 0.299], np.float32)
        first_order = sh[1:4] @ luma_weights
        direction = first_order / max(float(np.linalg.norm(first_order)), 1e-8)
        intensity = float(np.linalg.norm(first_order))
        azimuth = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
        elevation = math.degrees(math.atan2(float(direction[2]), math.hypot(float(direction[0]), float(direction[1]))))
        kelvin = _estimate_kelvin(sh[0])
        rim_lobes = _estimate_rim_lobes(image, n, valid, linear)
        return KeyLightEstimate(tuple(map(float, direction)), azimuth, elevation, intensity, kelvin, sh, rim_lobes)


class TemporalSpatialSwapDenoiser:
    """Edge-preserving texture accumulation guided by a current-to-previous UV flow."""

    def __init__(
        self,
        *,
        temporal_strength: float = 0.68,
        color_sigma: float = 14.0,
        spatial_sigma: float = 1.4,
        max_flow_px: float = 48.0,
    ) -> None:
        if not 0 <= temporal_strength <= 0.95 or color_sigma <= 0 or spatial_sigma < 0 or max_flow_px <= 0:
            raise ValueError("invalid temporal/spatial denoiser parameters")
        self.temporal_strength = float(temporal_strength)
        self.color_sigma = float(color_sigma)
        self.spatial_sigma = float(spatial_sigma)
        self.max_flow_px = float(max_flow_px)
        self._previous: np.ndarray | None = None

    def reset(self) -> None:
        self._previous = None

    def process(
        self,
        current_bgr: np.ndarray,
        current_to_previous_uv_flow: np.ndarray | None = None,
        *,
        confidence: np.ndarray | None = None,
        mesh_mask: np.ndarray | None = None,
        scene_cut: bool = False,
    ) -> np.ndarray:
        """Denoise one UV-space frame while preserving face edges and UV seams.

        Flow is normalized UV displacement from the current texel to the
        previous-frame sampling coordinate: +u moves right and +v moves up.
        Flow therefore is converted to pixels as ``(+du*W, -dv*H)``.
        """
        current_raw = np.asarray(current_bgr)
        if current_raw.ndim != 3 or current_raw.shape[2] != 3 or current_raw.dtype not in (np.uint8, np.float32, np.float64):
            raise ValueError("current_bgr must be an HxWx3 uint8 or float image")
        current = current_raw.astype(np.float32)
        if current_raw.dtype != np.uint8 and float(np.nanmax(current)) <= 1.5:
            current *= 255.0
        height, width = current.shape[:2]
        mask = np.ones((height, width), bool) if mesh_mask is None else np.asarray(mesh_mask) > 0
        if mask.shape != (height, width):
            raise ValueError("mesh_mask must match current_bgr")
        if confidence is None:
            conf = np.ones((height, width), np.float32)
        else:
            conf = np.asarray(confidence, np.float32)
            if conf.shape != (height, width) or not np.isfinite(conf).all():
                raise ValueError("confidence must be finite and match current_bgr")
            conf = np.clip(conf, 0.0, 1.0)
        if self.spatial_sigma > 0:
            spatial = cv2.bilateralFilter(current.astype(np.uint8), 5, self.color_sigma, self.spatial_sigma)
            base = current * 0.78 + spatial.astype(np.float32) * 0.22
        else:
            base = current
        if scene_cut or self._previous is None or current_to_previous_uv_flow is None:
            output = base
        else:
            flow = np.asarray(current_to_previous_uv_flow, dtype=np.float32)
            if flow.shape != (height, width, 2) or not np.isfinite(flow).all():
                raise ValueError("current_to_previous_uv_flow must be finite HxWx2 normalized UV offsets")
            flow_px = flow.copy()
            flow_px[..., 0] *= width
            flow_px[..., 1] *= -height
            magnitude = np.linalg.norm(flow_px, axis=2)
            over_limit = magnitude > self.max_flow_px
            flow_px[over_limit] *= (self.max_flow_px / np.maximum(magnitude[over_limit], 1e-6))[:, None]
            grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
            warped_previous = cv2.remap(self._previous, grid_x + flow_px[..., 0], grid_y + flow_px[..., 1],
                                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            delta = np.mean(np.abs(base - warped_previous), axis=2)
            edge_weight = np.exp(-delta / self.color_sigma)
            temporal = self.temporal_strength * conf * edge_weight * mask
            output = base * (1.0 - temporal[..., None]) + warped_previous * temporal[..., None]
        output = np.clip(output, 0.0, 255.0).astype(np.uint8)
        self._previous = output.astype(np.float32)
        return output


def synthesize_micro_pore_normal(resolution: int = 256, seed: int = 41) -> np.ndarray:
    """Generate a repeatable tileable tangent-space micro-pore normal map.

    Pores are smooth, randomly jittered circular dimples with a faint raised
    rim. Periodic distance and wrapped gradients make opposite edges tile.
    The output is linear tangent-space XYZ in [-1, 1], not sRGB color data.
    """
    if resolution < 32:
        raise ValueError("micro-pore map resolution must be at least 32")
    rng = np.random.default_rng(seed)
    grid_y, grid_x = np.mgrid[0:resolution, 0:resolution].astype(np.float32)
    height = np.zeros((resolution, resolution), np.float32)
    count = max(96, resolution * resolution // 230)
    for _ in range(count):
        cx, cy = rng.uniform(0, resolution, 2)
        radius = rng.uniform(0.55, 1.55) * resolution / 256.0
        depth = rng.uniform(0.018, 0.050)
        dx = (grid_x - cx + resolution * 0.5) % resolution - resolution * 0.5
        dy = (grid_y - cy + resolution * 0.5) % resolution - resolution * 0.5
        r2 = (dx * dx + dy * dy) / max(radius * radius, 0.2)
        center_dimple = np.exp(-2.5 * r2)
        raised_rim = np.exp(-0.85 * (r2 - 1.6) ** 2)
        height += -depth * center_dimple + depth * 0.20 * raised_rim
    # Low-amplitude cross-scale relief avoids a repeated dot matrix look.
    for frequency, amplitude, phase in ((0.11, 0.0018, 0.4), (0.23, 0.0008, 1.7)):
        height += amplitude * (np.sin(grid_x * frequency + phase) * np.sin(grid_y * frequency * 1.13 - phase))
    dx = (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1)) * 0.5
    dy = (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0)) * 0.5
    normals = np.stack((-dx * 8.0, -dy * 8.0, np.ones_like(height)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-8)
    return normals.astype(np.float32)


def sample_micro_pore_normal(
    uv: np.ndarray,
    pore_normal: np.ndarray,
    *,
    repeat: float = 12.0,
) -> np.ndarray:
    """Sample a periodic pore tile at arbitrary UV coordinates."""
    coords = np.asarray(uv, np.float32)
    texture = np.asarray(pore_normal, np.float32)
    if coords.shape[-1] != 2 or texture.ndim != 3 or texture.shape[2] != 3 or repeat <= 0:
        raise ValueError("uv must end in 2, pore_normal must be HxWx3, and repeat positive")
    height, width = texture.shape[:2]
    map_x = np.mod(coords[..., 0] * repeat, 1.0) * (width - 1)
    map_y = np.mod(1.0 - coords[..., 1] * repeat, 1.0) * (height - 1)
    sampled = cv2.remap(texture, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_WRAP)
    sampled /= np.maximum(np.linalg.norm(sampled, axis=-1, keepdims=True), 1e-8)
    return sampled


def build_cc0_probe_bundle(
    source_dir: Path | str,
    output_path: Path | str = PROBE_BUNDLE_PATH,
    *,
    environment_shape: tuple[int, int] = (32, 64),
    pore_resolution: int = 256,
) -> Path:
    """Reduce pinned local Poly Haven 1K HDRIs and synthesize the CC0 pore tile."""
    source_root = Path(source_dir)
    maps, coefficients = [], []
    for probe in ENVIRONMENT_PROBES:
        source = source_root / f"{probe['asset_id']}_1k.hdr"
        if not source.is_file():
            raise FileNotFoundError(f"Missing CC0 source HDRI: {source}")
        source_md5 = hashlib.md5(source.read_bytes()).hexdigest()
        if source_md5 != probe["md5"]:
            raise ValueError(f"CC0 source HDRI digest mismatch for {probe['asset_id']}: {source_md5}")
        hdr = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if hdr is None or hdr.ndim != 3 or hdr.shape[2] != 3:
            raise ValueError(f"Could not decode Radiance HDRI: {source}")
        reduced = cv2.resize(hdr.astype(np.float32), (environment_shape[1], environment_shape[0]), interpolation=cv2.INTER_AREA)
        maps.append(reduced)
        coefficients.append(_project_environment_sh(reduced))
    pore_map = synthesize_micro_pore_normal(pore_resolution)
    payload = {
        "schema": np.asarray(PROBE_BUNDLE_SCHEMA),
        "probe_names": np.asarray([probe["name"] for probe in ENVIRONMENT_PROBES]),
        "probe_asset_ids": np.asarray([probe["asset_id"] for probe in ENVIRONMENT_PROBES]),
        "probe_kelvin": np.asarray([probe["kelvin"] for probe in ENVIRONMENT_PROBES], np.float32),
        "environment_maps_bgr": np.asarray(maps, np.float16),
        "sh_coefficients_bgr": np.asarray(coefficients, np.float32),
        "micro_pore_normal_xyz": pore_map.astype(np.float16),
        "metadata": np.asarray(json.dumps({
            "schema": PROBE_BUNDLE_SCHEMA,
            "license": "CC0-1.0",
            "environment_license": "CC0-1.0",
            "environment_source": "Poly Haven HDRIs, reduced from official 1K Radiance downloads",
            "pore_license": "CC0-1.0",
            "pore_source": "Deterministic procedural periodic dimple synthesis by Roop 3D contributors",
            "ambientcg_note": "ambientCG publishes CC0 assets; its current catalog did not expose a human-skin pore map, so this tile is procedural and is not misattributed to a different ambientCG material.",
            "ambientcg_license_url": "https://docs.ambientcg.com/license/",
            "ambientcg_catalog_query": "https://ambientcg.com/api/v3/assets?type=material&q=skin&limit=100&include=title,tags",
            "environment_shape": list(environment_shape),
            "pore_resolution": pore_resolution,
            "probes": [dict(probe, license="CC0-1.0",
                             download_url=f"https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/{probe['asset_id']}_1k.hdr")
                       for probe in ENVIRONMENT_PROBES],
        }, sort_keys=True)),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
    provenance = json.loads(str(payload["metadata"].item()))
    provenance["bundle_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_name(target.name + ".json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return target


def load_cc0_preview_assets(path: Path | str = PROBE_BUNDLE_PATH) -> CC0PreviewAssets:
    """Load and validate the five compact HDRI probes and pore normal tile."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"CC0 preview bundle not found: {target}")
    sidecar = target.with_name(target.name + ".json")
    if not sidecar.is_file():
        raise ValueError(f"CC0 preview provenance sidecar is missing: {sidecar}")
    provenance = json.loads(sidecar.read_text("utf-8"))
    actual_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    if provenance.get("bundle_sha256") != actual_sha256:
        raise ValueError("CC0 preview bundle digest does not match its provenance sidecar")
    with np.load(target, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    if str(arrays["schema"].item()) != PROBE_BUNDLE_SCHEMA:
        raise ValueError("unsupported CC0 probe bundle schema")
    metadata = json.loads(str(arrays["metadata"].item()))
    if metadata.get("license") != "CC0-1.0":
        raise ValueError("preview bundle is not declared CC0")
    if metadata != {key: value for key, value in provenance.items() if key != "bundle_sha256"}:
        raise ValueError("CC0 preview metadata differs from its provenance sidecar")
    names = arrays["probe_names"].tolist()
    ids = arrays["probe_asset_ids"].tolist()
    maps = arrays["environment_maps_bgr"].astype(np.float32)
    sh = arrays["sh_coefficients_bgr"].astype(np.float32)
    kelvin = arrays["probe_kelvin"].astype(np.float32)
    if maps.ndim != 4 or maps.shape[0] != 5 or maps.shape[-1] != 3 or sh.shape != (5, 9, 3) or len(names) != 5:
        raise ValueError("preview bundle has invalid environment probe shapes")
    probes = tuple(EnvironmentProbe(str(name), str(asset_id), maps[index], sh[index], float(kelvin[index]))
                   for index, (name, asset_id) in enumerate(zip(names, ids)))
    pore = arrays["micro_pore_normal_xyz"].astype(np.float32)
    if pore.ndim != 3 or pore.shape[-1] != 3 or not np.isfinite(pore).all():
        raise ValueError("preview bundle has invalid pore normal map")
    return CC0PreviewAssets(probes, pore, metadata)


def _image_u8(image: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"{name} must be an HxWx3 BGR image")
    if value.dtype == np.uint8:
        return value
    if value.dtype not in (np.float32, np.float64) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite uint8 or float BGR")
    scale = 255.0 if value.max(initial=0.0) > 1.5 else 1.0
    return np.rint(np.clip(value * scale, 0.0, 255.0)).astype(np.uint8)


def _estimate_chromatic_edge_shifts(image: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    threshold = max(12.0, float(np.percentile(magnitude[valid], 88)) if valid.any() else 12.0)
    ys, xs = np.where(valid & (magnitude >= threshold))
    if len(xs) < 8:
        return 0.0, 0.0
    if len(xs) > 3000:
        select = np.linspace(0, len(xs) - 1, 3000).astype(np.int64)
        ys, xs = ys[select], xs[select]
    nx = gx[ys, xs] / np.maximum(magnitude[ys, xs], 1e-6)
    ny = gy[ys, xs] / np.maximum(magnitude[ys, xs], 1e-6)
    radial_x = xs.astype(np.float32) - (image.shape[1] - 1) * 0.5
    radial_y = ys.astype(np.float32) - (image.shape[0] - 1) * 0.5
    radial_norm = np.maximum(np.hypot(radial_x, radial_y), 1e-6)
    radial_alignment = nx * radial_x / radial_norm + ny * radial_y / radial_norm
    offsets = np.linspace(-2.5, 2.5, 11, dtype=np.float32)
    channel_shifts: dict[int, float] = {}
    for channel in (0, 2):
        cgx = cv2.Sobel(image[..., channel].astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        cgy = cv2.Sobel(image[..., channel].astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        # Sample channel gradients along each luminance-edge normal and locate the local edge peak.
        samples = np.empty((len(xs), len(offsets)), np.float32)
        for index, offset in enumerate(offsets):
            map_x = (xs.astype(np.float32) + nx * offset).reshape(-1, 1)
            map_y = (ys.astype(np.float32) + ny * offset).reshape(-1, 1)
            sx = cv2.remap(cgx, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]
            sy = cv2.remap(cgy, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[:, 0]
            samples[:, index] = np.abs(sx * nx + sy * ny)
        floor = np.min(samples, axis=1, keepdims=True)
        weights = np.maximum(samples - floor, 0.0)
        shift_normal = np.sum(weights * offsets[None, :], axis=1) / np.maximum(np.sum(weights, axis=1), 1e-6)
        shift_radial = shift_normal * radial_alignment
        usable = (np.max(samples, axis=1) > 5.0) & (np.abs(radial_alignment) > 0.25)
        channel_shifts[channel] = float(np.median(shift_radial[usable])) if np.any(usable) else 0.0
    return float(np.clip(channel_shifts[0], -2.5, 2.5)), float(np.clip(channel_shifts[2], -2.5, 2.5))


def _estimate_sensor_noise(frame_bgr: np.ndarray, valid: np.ndarray) -> tuple[float, int]:
    color = frame_bgr.astype(np.float32)
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(image, (0, 0), 1.25)
    residual = image - smooth
    gx = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gx, gy)
    # Chroma-difference residual cancels real luminance texture and exposes
    # uncorrelated channel noise. A smooth luma estimate covers correlated
    # noise only when the frame actually contains sufficiently flat patches.
    color_high = color - cv2.GaussianBlur(color, (0, 0), 1.0)
    chroma = np.concatenate((color_high[..., 0][valid] - color_high[..., 1][valid],
                             color_high[..., 2][valid] - color_high[..., 1][valid]))
    chroma -= np.median(chroma) if chroma.size else 0.0
    chroma_sigma = 1.4826 * float(np.median(np.abs(chroma))) / math.sqrt(2.0) if chroma.size else 0.0
    mean = cv2.boxFilter(image, cv2.CV_32F, (7, 7), normalize=True)
    mean_square = cv2.boxFilter(image * image, cv2.CV_32F, (7, 7), normalize=True)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    candidates = valid & (local_std <= 8.0) & (gradient <= 16.0)
    if int(candidates.sum()) >= 64:
        values = residual[candidates]
        centered = values - np.median(values)
        luma_sigma = 1.4826 * float(np.median(np.abs(centered)))
    else:
        luma_sigma = 0.0
    sigma = max(chroma_sigma, luma_sigma)
    sigma = float(np.clip(sigma, 0.0, 32.0))
    iso = int(np.clip(round(100.0 * (sigma / 1.8) ** 2), 50, 12800))
    return sigma, iso


def _sharpness_match_sigma(rendered: np.ndarray, alpha: np.ndarray, target: np.ndarray, target_sharpness: float) -> float:
    mask = alpha > 0.35
    if int(mask.sum()) < 32 or target_sharpness <= 1e-6:
        return 0.0
    gray = cv2.cvtColor(np.clip(rendered * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    def variance(image: np.ndarray) -> float:
        lap = cv2.Laplacian(image, cv2.CV_32F, ksize=3)
        return float(np.var(lap[mask]))

    if variance(gray) <= target_sharpness:
        return 0.0  # Matching never invents sharpness or extra blur.
    low, high = 0.0, 8.0
    for _ in range(12):
        middle = 0.5 * (low + high)
        blurred = cv2.GaussianBlur(gray, (0, 0), middle)
        if variance(blurred) > target_sharpness:
            low = middle
        else:
            high = middle
    return float(high)


def _circle_of_confusion_sigma(
    depth: np.ndarray,
    alpha: np.ndarray,
    focus_distance_m: float,
    f_number: float,
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width: int,
) -> np.ndarray:
    z = np.asarray(depth, np.float32)
    valid = (alpha > 0.01) & np.isfinite(z) & (z > 1e-4)
    z_safe = np.where(valid, z, focus_distance_m)
    focal_m = focal_length_mm / 1000.0
    sensor_m = sensor_width_mm / 1000.0
    coc_m = (
        focal_m * focal_m / (f_number * max(focus_distance_m - focal_m, 1e-6))
    ) * np.abs(z_safe - focus_distance_m) / np.maximum(z_safe, 1e-4)
    coc_px = coc_m * image_width / sensor_m
    sigma = np.clip(coc_px / 2.355, 0.0, 8.0)
    return np.where(valid, sigma, 0.0).astype(np.float32)


def _spatial_blur(image: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
    source = np.asarray(image, np.float32)
    if source.ndim == 2:
        source = source[..., None]
        squeeze = True
    else:
        squeeze = False
    sigma = np.asarray(sigma_map, np.float32)
    levels = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0, 8.0], np.float32)
    high = np.searchsorted(levels, sigma, side="right").clip(1, len(levels) - 1)
    low = high - 1
    denom = np.maximum(levels[high] - levels[low], 1e-6)
    blend = np.clip((sigma - levels[low]) / denom, 0.0, 1.0)
    output = np.zeros_like(source)
    for level_index, level in enumerate(levels):
        if level_index == 0:
            blurred = source
        else:
            blurred = cv2.GaussianBlur(source, (0, 0), float(level))
            if blurred.ndim == 2:
                blurred = blurred[..., None]
        weight = np.where(low == level_index, 1.0 - blend, 0.0) + np.where(high == level_index, blend, 0.0)
        output += blurred * weight[..., None]
    return output[..., 0] if squeeze else output


def _directional_blur(image: np.ndarray, direction: tuple[float, float], length_px: float) -> np.ndarray:
    if length_px <= 0.35:
        return np.asarray(image)
    radius = max(1, int(math.ceil(length_px / 2.0)))
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.float32)
    cx, cy = radius, radius
    dx, dy = direction[0] * length_px * 0.5, direction[1] * length_px * 0.5
    steps = max(3, int(math.ceil(length_px * 2)))
    for t in np.linspace(-1.0, 1.0, steps):
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        kernel[y, x] += 1.0
    kernel /= max(float(kernel.sum()), 1e-8)
    return cv2.filter2D(np.asarray(image, np.float32), -1, kernel, borderType=cv2.BORDER_REFLECT_101)


def _apply_chromatic_aberration(
    color: np.ndarray,
    alpha: np.ndarray,
    blue_shift: float,
    red_shift: float,
) -> np.ndarray:
    height, width = alpha.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx, cy = (width - 1) * 0.5, (height - 1) * 0.5
    dx, dy = xx - cx, yy - cy
    radius = np.sqrt(dx * dx + dy * dy)
    dx /= np.maximum(radius, 1e-6)
    dy /= np.maximum(radius, 1e-6)
    output = np.asarray(color, np.float32).copy()
    # BGR channels: blue at index 0, red at index 2. Alpha remains the tracked mesh coverage.
    for channel, shift in ((0, blue_shift), (2, red_shift)):
        if abs(shift) < 0.03:
            continue
        map_x = xx - dx * shift
        map_y = yy - dy * shift
        output[..., channel] = cv2.remap(output[..., channel], map_x, map_y, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_REPLICATE)
    return output


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _estimate_kelvin(ambient_bgr: np.ndarray) -> float:
    b, g, r = np.maximum(np.asarray(ambient_bgr, np.float64), 1e-5)
    total = r + g + b
    x = r / total
    y = g / total
    n = (x - 0.3320) / (0.1858 - y + 1e-8)
    kelvin = 449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33
    return float(np.clip(kelvin, 2000.0, 12000.0))


def _estimate_rim_lobes(image: np.ndarray, normals: np.ndarray, valid: np.ndarray, linear: np.ndarray) -> tuple[RimLightLobe, ...]:
    luma = linear @ np.array([0.114, 0.587, 0.299], np.float32)
    valid_values = luma[valid]
    if len(valid_values) < 64:
        return ()
    cutoff = float(np.percentile(valid_values, 94.0))
    rim = valid & (np.abs(normals[..., 2]) < 0.72) & (luma >= cutoff)
    y, x = np.where(rim)
    if len(x) < 8:
        return ()
    directions = normals[y, x].astype(np.float64)
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-8)
    weights = np.maximum(luma[y, x].astype(np.float64) - cutoff, 1e-5)
    first = int(np.argmax(weights))
    seeds = [directions[first]]
    if len(directions) > 16:
        seeds.append(directions[int(np.argmin(directions @ seeds[0]))])
    centers = np.asarray(seeds)
    labels = np.zeros(len(directions), np.int32)
    for _ in range(8):
        labels = np.argmax(directions @ centers.T, axis=1)
        for index in range(len(centers)):
            selected = labels == index
            if np.any(selected):
                center = np.average(directions[selected], axis=0, weights=weights[selected])
                centers[index] = center / max(np.linalg.norm(center), 1e-8)
    lobes = []
    for index, center in enumerate(centers):
        selected = labels == index
        if int(selected.sum()) < 4:
            continue
        angular = np.arccos(np.clip(directions[selected] @ center, -1.0, 1.0))
        intensity = float(np.average(weights[selected], weights=weights[selected]))
        color = np.average(linear[y[selected], x[selected]], axis=0, weights=weights[selected])
        lobes.append(RimLightLobe(tuple(map(float, center)), intensity,
                                  float(np.degrees(np.sqrt(np.average(angular * angular, weights=weights[selected])))),
                                  tuple(map(float, color))))
    return tuple(sorted(lobes, key=lambda lobe: -lobe.intensity))


def _project_environment_sh(environment_bgr: np.ndarray) -> np.ndarray:
    image = np.asarray(environment_bgr, np.float32)
    height, width = image.shape[:2]
    row, column = np.mgrid[0:height, 0:width]
    latitude = np.pi * (0.5 - (row + 0.5) / height)
    longitude = 2.0 * np.pi * ((column + 0.5) / width - 0.5)
    cos_lat = np.cos(latitude)
    directions = np.stack((cos_lat * np.sin(longitude), np.sin(latitude), cos_lat * np.cos(longitude)), axis=-1)
    basis = evaluate_sh_basis(directions.reshape(-1, 3)).astype(np.float64)
    weights = np.broadcast_to(np.maximum(cos_lat, 1e-6), (height, width)).reshape(-1).astype(np.float64)
    radiance = image.reshape(-1, 3).astype(np.float64)
    gram = basis.T @ (basis * weights[:, None]) + np.eye(9) * 1e-7
    return np.linalg.solve(gram, basis.T @ (radiance * weights[:, None])).astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-cc0-bundle", type=Path, metavar="SOURCE_DIR",
                        help="reduce the five verified Poly Haven 1K HDR files in SOURCE_DIR into the bundled compact asset")
    parser.add_argument("--out", type=Path, default=PROBE_BUNDLE_PATH)
    args = parser.parse_args(argv)
    if args.build_cc0_bundle:
        print(build_cc0_probe_bundle(args.build_cc0_bundle, args.out))
        return 0
    assets = load_cc0_preview_assets(args.out)
    print(json.dumps({"ok": True, "probes": len(assets.probes), "pore_resolution": list(assets.micro_pore_normal.shape[:2]),
                      "license": assets.metadata["license"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
