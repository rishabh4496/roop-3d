"""3D Surface Normal Estimation, 9-Band Spherical Harmonics Lighting, and Physically Based Relighting.

Stage 17 Implementation:
1. High-resolution per-pixel 3D Surface Normal Map N(x, y) combining low-frequency geometric
   normals (from fitted 3DMM / FLAME / canonical head) and high-frequency detail normals
   (monocular depth / photometric gradients).
2. 9-Band (2nd-Order) Spherical Harmonics (SH) illumination extraction via regularized least-squares,
   recovering ambient light, directional key light angle, and shadow falloff.
3. Albedo decomposition and diffuse relighting with cast shadow preservation and Blinn-Phong
   subsurface specular skin sheen.
"""
from __future__ import annotations

import math
import sys
from typing import Any

import cv2
import numpy as np

# Ramamoorthi & Hanrahan (2001) irradiance constants
_C1 = 0.429043
_C2 = 0.511664
_C3 = 0.743125
_C4 = 0.886227
_C5 = 0.247708


class SurfaceNormalEstimator:
    """Estimates high-resolution 3D surface normal maps N(x, y) in [-1, 1]^3."""

    def __init__(self, high_freq_weight: float = 0.25):
        self.high_freq_weight = high_freq_weight

    def estimate_from_depth(self, depth: np.ndarray) -> np.ndarray:
        """Derive 3D surface normals from a dense depth map Z(x, y)."""
        z = np.asarray(depth, dtype=np.float32)
        h, w = z.shape[:2]
        dz_dy, dz_dx = np.gradient(z, 2.0 / max(h, 1), 2.0 / max(w, 1))
        # In right-handed CV camera coordinates (+X right, +Y down, +Z into scene, normal pointing out towards camera)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(nx)
        normals = np.stack([nx, ny, nz], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(norm, 1e-8)

    def estimate_geometric_normals(self, height: int, width: int,
                                   landmarks: np.ndarray | None = None) -> np.ndarray:
        """Generate smooth low-frequency geometric surface normals based on 3DMM facial geometry."""
        y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
        r2 = x * x + y * y
        head_z = np.sqrt(np.maximum(1.0 - np.minimum(r2, 0.95), 0.0))

        # Nose bridge: 3D ridge along the vertical midline
        nose_width = 0.08
        nose_z = 0.28 * np.exp(-(x * x) / (2.0 * nose_width * nose_width)) * np.maximum(0.0, 1.0 - np.abs(y) / 0.45)
        # Brow ridge and cheekbones
        brow_z = 0.08 * np.exp(-((y + 0.3) ** 2) / (2.0 * 0.06 * 0.06)) * (1.0 - 0.5 * x * x)
        cheek_z = 0.10 * np.exp(-((y - 0.1) ** 2) / 0.08) * np.exp(-((np.abs(x) - 0.42) ** 2) / 0.06)

        total_z = head_z + nose_z + brow_z + cheek_z
        dz_dy, dz_dx = np.gradient(total_z, 2.0 / height, 2.0 / width)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(nx)
        normals = np.stack([nx, ny, nz], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(norm, 1e-8)

    def extract_photometric_details(self, image_bgr: np.ndarray) -> np.ndarray:
        """Derive high-frequency detail normal perturbations from image luminance."""
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        # High-pass spatial filter (texture minus low-pass smooth base)
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.5, sigmaY=2.5)
        high_freq = gray - blurred

        # High-frequency spatial gradients
        gx = cv2.Sobel(high_freq, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(high_freq, cv2.CV_32F, 0, 1, ksize=3)

        return np.stack([-gx, -gy, np.zeros_like(gx)], axis=-1)

    def compute_normals(self, image_bgr: np.ndarray,
                        depth_map: np.ndarray | None = None,
                        landmarks: np.ndarray | None = None) -> np.ndarray:
        """Combine low-frequency geometric normals with high-frequency surface details."""
        h, w = image_bgr.shape[:2]
        if depth_map is not None:
            base_normals = self.estimate_from_depth(depth_map)
        else:
            base_normals = self.estimate_geometric_normals(h, w, landmarks)

        detail = self.extract_photometric_details(image_bgr)
        fused = base_normals + self.high_freq_weight * detail
        # Ensure normal vector always points towards camera (nz > 0)
        fused[..., 2] = np.maximum(fused[..., 2], 0.05)
        norm = np.linalg.norm(fused, axis=-1, keepdims=True)
        return fused / np.maximum(norm, 1e-8)

    @staticmethod
    def encode_normal_map_rgb(normals: np.ndarray) -> np.ndarray:
        """Encodes normal map [-1, 1]^3 to uint8 BGR image [0, 255]."""
        # Map [-1, 1] -> [0, 255]
        # In BGR: B = nz, G = ny, R = nx
        mapped = ((normals + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        # Store as standard normal map format (R=nx, G=ny, B=nz) for OpenCV BGR display
        bgr = np.zeros_like(mapped)
        bgr[..., 0] = mapped[..., 2]  # B: nz
        bgr[..., 1] = mapped[..., 1]  # G: ny
        bgr[..., 2] = mapped[..., 0]  # R: nx
        return bgr

    @staticmethod
    def decode_normal_map_rgb(bgr_image: np.ndarray) -> np.ndarray:
        """Decodes uint8 BGR normal map back to float32 [-1, 1]^3."""
        rgb = bgr_image.astype(np.float32) / 255.0
        nx = rgb[..., 2] * 2.0 - 1.0
        ny = rgb[..., 1] * 2.0 - 1.0
        nz = rgb[..., 0] * 2.0 - 1.0
        normals = np.stack([nx, ny, nz], axis=-1)
        norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        return normals / np.maximum(norm, 1e-8)


class SphericalHarmonicsLighting:
    """9-Band (2nd-Order) Spherical Harmonics lighting estimator and shader."""

    @staticmethod
    def evaluate_basis(normals: np.ndarray) -> np.ndarray:
        """Computes the 9 Ramamoorthi & Hanrahan SH basis functions for surface normals.

        Input: normals (..., 3) unit vectors (x, y, z).
        Output: basis (..., 9) float32 evaluated basis terms.
        """
        x = normals[..., 0]
        y = normals[..., 1]
        z = normals[..., 2]

        return np.stack([
            _C4 * np.ones_like(x),           # L00 (Ambient)
            2.0 * _C2 * x,                   # L11 (Linear X: key light left/right)
            2.0 * _C2 * y,                   # L1-1 (Linear Y: key light up/down)
            2.0 * _C2 * z,                   # L10 (Linear Z: key light depth)
            2.0 * _C1 * (x * y),             # L2-2
            2.0 * _C1 * (x * z),             # L21
            2.0 * _C1 * (y * z),             # L2-1
            _C1 * (x * x - y * y),           # L22
            _C3 * (3.0 * z * z - 1.0)        # L20
        ], axis=-1).astype(np.float32)

    def estimate_lighting(self, normals: np.ndarray, image_bgr: np.ndarray,
                          mask: np.ndarray | None = None,
                          regularization: float = 0.03) -> np.ndarray:
        """Estimates the 9x3 SH lighting coefficients L from normals and observed BGR image.

        Output: L (9, 3) float32 coefficients for B, G, R channels.
        """
        h, w = normals.shape[:2]
        if mask is None:
            # Default to inner face ellipse
            my, mx = np.mgrid[-1.0:1.0:complex(h), -1.0:1.0:complex(w)]
            mask = (mx * mx + my * my) <= 0.85
        else:
            mask = np.asarray(mask, dtype=bool)

        valid_idx = mask & np.all(np.isfinite(normals), axis=-1)
        if np.count_nonzero(valid_idx) < 32:
            # Default neutral illumination
            return self.create_directional_lighting(np.array([0.0, 0.0, 1.0]), ambient=0.5, intensity=0.5)

        basis_samples = self.evaluate_basis(normals[valid_idx])  # (M, 9)
        img_samples = (image_bgr[valid_idx].astype(np.float32) / 255.0)  # (M, 3) in BGR

        num_samples = len(basis_samples)
        lambda_reg = regularization * num_samples
        ata = basis_samples.T @ basis_samples + lambda_reg * np.eye(9, dtype=np.float32)
        atb = basis_samples.T @ img_samples  # (9, 3)

        coeffs = np.linalg.solve(ata, atb)  # (9, 3)
        return coeffs.astype(np.float32)

    def evaluate_shading(self, normals: np.ndarray, sh_coeffs: np.ndarray) -> np.ndarray:
        """Evaluates diffuse irradiance from surface normals and SH coefficients.

        Output: (H, W, 3) float32 shading values in BGR.
        """
        basis = self.evaluate_basis(normals)  # (H, W, 9)
        shading = basis @ sh_coeffs           # (H, W, 3)
        return np.maximum(shading, 0.01)

    def extract_key_light_direction(self, sh_coeffs: np.ndarray) -> np.ndarray:
        """Extracts the dominant directional key light vector (lx, ly, lz) from linear SH terms."""
        # Weighted luminance of linear coefficients (BGR weights: 0.114 B + 0.587 G + 0.299 R)
        bgr_weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
        luma_coeffs = sh_coeffs @ bgr_weights  # (9,)

        lx = luma_coeffs[1]
        ly = luma_coeffs[2]
        lz = luma_coeffs[3]
        direction = np.array([lx, ly, lz], dtype=np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return direction / norm

    def extract_color_temperature(self, sh_coeffs: np.ndarray) -> float:
        """Estimates the correlated color temperature (CCT in Kelvin) from ambient SH color."""
        bgr = np.maximum(sh_coeffs[0], 1e-4)
        b, g, r = bgr[0], bgr[1], bgr[2]
        tot = r + g + b
        x = r / tot
        y = g / tot
        # McCamy's polynomial approximation for CCT
        n = (x - 0.3320) / (0.1858 - y + 1e-8)
        cct = 449.0 * (n ** 3) + 3525.0 * (n ** 2) + 6823.3 * n + 5520.33
        return float(np.clip(cct, 2000.0, 12000.0))

    def create_directional_lighting(self, direction: np.ndarray,
                                    ambient: float = 0.25,
                                    intensity: float = 0.75,
                                    color_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
                                    cct: float = 5500.0) -> np.ndarray:
        """Constructs 9x3 SH coefficients for a directional key light and ambient base."""
        dir_norm = np.asarray(direction, dtype=np.float32)
        dir_norm = dir_norm / max(float(np.linalg.norm(dir_norm)), 1e-8)
        lx, ly, lz = float(dir_norm[0]), float(dir_norm[1]), float(dir_norm[2])

        # Color tint from temperature (Kelvin)
        k_r, k_g, k_b = self._kelvin_to_rgb(cct)
        cr = color_rgb[0] * k_r
        cg = color_rgb[1] * k_g
        cb = color_rgb[2] * k_b
        tint_bgr = np.array([cb, cg, cr], dtype=np.float32)

        coeffs = np.zeros((9, 3), dtype=np.float32)
        # L00 (Ambient)
        coeffs[0] = (ambient / _C4) * tint_bgr
        # L11, L1-1, L10 (Linear terms for directional key light)
        dir_scale = intensity / (2.0 * _C2)
        coeffs[1] = (dir_scale * lx) * tint_bgr
        coeffs[2] = (dir_scale * ly) * tint_bgr
        coeffs[3] = (dir_scale * lz) * tint_bgr
        # Oblique falloff terms
        quad_scale = (intensity * 0.25) / _C1
        coeffs[4] = (quad_scale * lx * ly) * tint_bgr
        coeffs[5] = (quad_scale * lx * lz) * tint_bgr
        coeffs[6] = (quad_scale * ly * lz) * tint_bgr
        coeffs[7] = (quad_scale * (lx * lx - ly * ly)) * tint_bgr
        coeffs[8] = ((intensity * 0.15) / _C3 * (3.0 * lz * lz - 1.0)) * tint_bgr

        return coeffs

    @staticmethod
    def _kelvin_to_rgb(kelvin: float) -> tuple[float, float, float]:
        """Tanner-Helland Planckian approximation for color temperature to RGB."""
        k = np.clip(kelvin, 1000.0, 40000.0) / 100.0
        # Red
        if k <= 66.0:
            r = 255.0
        else:
            r = 329.698727446 * ((k - 60.0) ** -0.1332047592)
        # Green
        if k <= 66.0:
            g = 99.4708025861 * math.log(max(k, 1.0)) - 161.1195681661
        else:
            g = 288.1221695283 * ((k - 60.0) ** -0.0755148492)
        # Blue
        if k >= 66.0:
            b = 255.0
        elif k <= 19.0:
            b = 0.0
        else:
            b = 138.5177312231 * math.log(max(k - 10.0, 1.0)) - 305.0447927307

        rgb = np.clip([r, g, b], 0.0, 255.0) / 255.0
        # Normalize so max channel is 1.0
        return tuple(rgb / np.max(rgb))


class NeuralRelighter3D:
    """Combines 3D surface normals, SH scene illumination, and skin specular shading."""

    def __init__(self, normal_estimator: SurfaceNormalEstimator | None = None,
                 sh_lighting: SphericalHarmonicsLighting | None = None):
        self.normal_estimator = normal_estimator or SurfaceNormalEstimator()
        self.sh_lighting = sh_lighting or SphericalHarmonicsLighting()

    def compute_shadow_mask(self, target_bgr: np.ndarray,
                            target_normals: np.ndarray,
                            target_sh: np.ndarray,
                            mask: np.ndarray | None = None) -> np.ndarray:
        """Preserves hard cast shadows (hat brim, microphone, blinds) via luminance ratio."""
        target_gray = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        pred_shading = self.sh_lighting.evaluate_shading(target_normals, target_sh)
        pred_gray = (pred_shading @ np.array([0.114, 0.587, 0.299], dtype=np.float32))

        # Hard cast shadows cause observed luminance to drop below smooth low-frequency SH prediction
        ratio = target_gray / np.maximum(pred_gray, 0.05)
        shadow_mask = np.clip(ratio, 0.0, 1.0)
        # Soften edges slightly to prevent aliasing
        shadow_mask = cv2.GaussianBlur(shadow_mask, (3, 3), 1.0)
        return shadow_mask

    def compute_blinn_phong_specular(self, normals: np.ndarray,
                                     light_dir: np.ndarray,
                                     k_spec: float = 0.25,
                                     shininess: float = 24.0) -> np.ndarray:
        """Subsurface specular skin sheen (soft Blinn-Phong highlight)."""
        if k_spec <= 1e-4:
            return np.zeros_like(normals)

        view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        half_vec = light_dir + view_dir
        half_vec /= max(float(np.linalg.norm(half_vec)), 1e-8)

        n_dot_h = np.maximum(normals @ half_vec, 0.0)
        spec = (n_dot_h ** shininess) * k_spec
        # Gentle subsurface scatter tint (slightly warm/soft sheen)
        sheen_color = np.array([0.85, 0.92, 1.0], dtype=np.float32)  # BGR
        return spec[..., None] * sheen_color * 255.0

    def relight(self, swapped_bgr: np.ndarray,
                target_bgr: np.ndarray,
                swap_normals: np.ndarray | None = None,
                target_normals: np.ndarray | None = None,
                skin_mask: np.ndarray | None = None,
                face_mask: np.ndarray | None = None,
                *,
                relight_strength: float = 0.8,
                specular_sheen: float = 0.25,
                shadow_preservation: float = 0.85,
                lighting_mode: str = "auto",
                manual_light_yaw: float = 35.0,
                manual_light_pitch: float = 25.0,
                ambient_fill: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Executes full 3D relighting transfer.

        Returns:
            relit_bgr: Relit swapped face (uint8).
            normal_vis: RGB-encoded 3D surface normal map (uint8).
            sh_lighting_vis: Extracted SH shading map and shadow mask (uint8).
            metadata: Detected light direction, CCT, etc.
        """
        h, w = swapped_bgr.shape[:2]
        if swap_normals is None:
            swap_normals = self.normal_estimator.compute_normals(swapped_bgr)
        if target_normals is None:
            target_normals = self.normal_estimator.compute_normals(target_bgr)

        # 1. SH Lighting Extraction
        if lighting_mode == "manual":
            # Virtual manual key light from yaw and pitch
            yaw_rad = math.radians(manual_light_yaw)
            pitch_rad = math.radians(manual_light_pitch)
            dir_x = math.sin(yaw_rad) * math.cos(pitch_rad)
            dir_y = -math.sin(pitch_rad)  # in CV: +Y is down
            dir_z = math.cos(yaw_rad) * math.cos(pitch_rad)
            manual_dir = np.array([dir_x, dir_y, dir_z], dtype=np.float32)
            target_sh = self.sh_lighting.create_directional_lighting(manual_dir, ambient=ambient_fill, intensity=0.7)
            key_light_dir = manual_dir / np.linalg.norm(manual_dir)
            cct = 5500.0
        else:
            target_sh = self.sh_lighting.estimate_lighting(target_normals, target_bgr, skin_mask)
            key_light_dir = self.sh_lighting.extract_key_light_direction(target_sh)
            cct = self.sh_lighting.extract_color_temperature(target_sh)

        source_sh = self.sh_lighting.estimate_lighting(swap_normals, swapped_bgr, skin_mask)

        # 2. Evaluate diffuse shading maps
        target_shading = self.sh_lighting.evaluate_shading(swap_normals, target_sh)
        source_shading = self.sh_lighting.evaluate_shading(swap_normals, source_sh)

        # Compute luminance shading ratio (preserves exact underlying skin hue)
        target_luma = target_shading @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
        source_luma = source_shading @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
        shading_ratio = target_luma / np.maximum(source_luma, 0.02)
        shading_ratio = np.clip(shading_ratio, 0.15, 3.5)

        # Blend with relight strength
        eff_ratio = (1.0 - relight_strength) + relight_strength * shading_ratio

        # 3. Cast shadow preservation
        shadow_mask = self.compute_shadow_mask(target_bgr, target_normals, target_sh, skin_mask)
        eff_shadow = (1.0 - shadow_preservation) + shadow_preservation * shadow_mask

        # 4. Apply diffuse shading on HSV Value (Luminance) channel to strictly preserve Hue
        swap_hsv = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        orig_v = swap_hsv[..., 2]
        new_v = np.clip(orig_v * eff_ratio * eff_shadow, 0.0, 255.0)
        swap_hsv[..., 2] = new_v
        diffuse_relit = cv2.cvtColor(swap_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 5. Specular skin sheen
        spec = self.compute_blinn_phong_specular(swap_normals, key_light_dir, k_spec=specular_sheen)
        relit_bgr = np.clip(diffuse_relit + spec, 0.0, 255.0).astype(np.uint8)

        # Mask blending if face_mask is provided
        if face_mask is not None:
            alpha = np.asarray(face_mask, dtype=np.float32)
            if alpha.ndim == 2:
                alpha = alpha[..., None]
            if alpha.max() > 1.0:
                alpha /= 255.0
            relit_bgr = np.clip(relit_bgr * alpha + swapped_bgr.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)

        # Visualizations
        normal_vis = SurfaceNormalEstimator.encode_normal_map_rgb(swap_normals)
        sh_vis = np.clip((target_shading * eff_shadow[..., None]) * 255.0, 0, 255).astype(np.uint8)

        metadata = {
            "key_light_direction": [float(v) for v in key_light_dir],
            "color_temperature_k": float(cct),
            "sh_coeffs": target_sh.tolist(),
            "relight_strength": float(relight_strength),
            "specular_sheen": float(specular_sheen),
            "shadow_preservation": float(shadow_preservation),
        }

        return relit_bgr, normal_vis, sh_vis, metadata


# Top-level instance
relighter = NeuralRelighter3D()

# Register compatibility alias so legacy import paths map to roop_3d
sys.modules.setdefault("roop.relighting_3d", sys.modules[__name__])
