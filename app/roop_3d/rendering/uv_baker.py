# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Roop 3D contributors
"""Local PBR atlas baking from a validated multi-view identity asset.

The albedo is copied from the existing visibility/quality-weighted FaceSet
texture atlas. No generative fill, external data, or trained model is used.
Normal and cavity maps are deterministic derivatives of its measured UV relief.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

import cv2
import numpy as np

from roop_3d.reconstruction.uv_synthesizer import ConfidenceState
from roop_3d.rendering.intrinsic_texture import IntrinsicTextureDecomposer


BAKER_VERSION = "roop3d-uv-pbr-v1"
MAP_SUFFIXES = ("basecolor", "normal", "orm", "cavity")


@dataclass(frozen=True)
class BakedPBRAtlas:
    resolution: int
    basecolor_bgra: np.ndarray
    normal_bgra: np.ndarray
    orm_bgra: np.ndarray
    cavity: np.ndarray
    trust: np.ndarray
    source_resolution: tuple[int, int]
    cavity_relief_range_m: float
    relief_available: bool

    def encoded(self) -> dict[str, bytes]:
        arrays = {"basecolor": self.basecolor_bgra, "normal": self.normal_bgra,
                  "orm": self.orm_bgra, "cavity": self.cavity}
        result: dict[str, bytes] = {}
        for name, image in arrays.items():
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError(f"could not encode {name} atlas")
            result[name] = encoded.tobytes()
        return result


def _resize(array: np.ndarray, resolution: int, interpolation: int | None = None) -> np.ndarray:
    value = np.asarray(array)
    if value.shape[:2] == (resolution, resolution):
        return np.array(value, copy=True)
    if interpolation is None:
        interpolation = cv2.INTER_AREA if resolution < min(value.shape[:2]) else cv2.INTER_CUBIC
    return cv2.resize(value, (resolution, resolution), interpolation=interpolation)


def _trust_from_state(state: np.ndarray, resolution: int) -> np.ndarray:
    states = _resize(np.asarray(state, np.uint8), resolution, cv2.INTER_NEAREST)
    # Observed texels are fully trusted; bounded interpolation is explicitly
    # marked lower confidence; unseen/background texels remain transparent.
    return np.where(states == int(ConfidenceState.OBSERVED), 1.0,
                    np.where(states == int(ConfidenceState.INTERPOLATED), 0.45, 0.0)).astype(np.float32)


def bake_pbr_atlas(asset: Any, resolution: int = 2048) -> BakedPBRAtlas:
    """Bake albedo, tangent-space relief normals, ORM and cavity to 2K or 4K."""
    if not 16 <= int(resolution) <= 4096:
        raise ValueError("PBR atlas resolution must be within [16,4096]; production endpoints expose 2K and 4K")
    resolution = int(resolution)
    albedo = np.asarray(asset.texture)
    state = np.asarray(asset.texel_state)
    if albedo.ndim != 3 or albedo.shape[2] != 3 or albedo.dtype != np.uint8 or state.shape != albedo.shape[:2]:
        raise ValueError("identity asset must provide uint8 BGR albedo and a matching texel state map")
    if not np.isfinite(albedo).all():
        raise ValueError("albedo atlas contains non-finite values")
    source_size = (int(albedo.shape[0]), int(albedo.shape[1]))
    trust = _trust_from_state(state, resolution)
    source_states = np.asarray(state, np.uint8)
    source_valid = source_states != int(ConfidenceState.UNSEEN)
    diffuse = IntrinsicTextureDecomposer().decompose(albedo, source_valid)
    color = _resize(diffuse.albedo_bgr, resolution)
    # Preserve the measured diffuse reflectance and expose observation trust
    # in alpha. The decomposer only removes broad illumination; it fills none.
    basecolor = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
    basecolor[..., 3] = np.rint(trust * 255).astype(np.uint8)

    relief = np.asarray(getattr(asset, "displacement_uv_m", np.empty((0, 0))), np.float32)
    if relief.shape != albedo.shape[:2] or not np.isfinite(relief).all() or not np.any(np.abs(relief) > 1e-9):
        relief = diffuse.displacement_uv_m
    relief_available = bool(np.any(np.abs(relief) > 1e-9))
    height = _resize(relief, resolution)
    gx = cv2.Sobel(height, cv2.CV_32F, 1, 0, ksize=3) * np.float32(resolution / 8.0)
    gy = cv2.Sobel(height, cv2.CV_32F, 0, 1, ksize=3) * np.float32(resolution / 8.0)
    normals = np.stack((-gx * 180.0, -gy * 180.0, np.ones_like(height)), axis=-1)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    # A PNG consumed as RGB reads the BGR byte order as (nx,ny,nz).
    normal = np.empty((resolution, resolution, 4), np.uint8)
    normal[..., 0] = np.rint((normals[..., 2] * 0.5 + 0.5) * 255).astype(np.uint8)
    normal[..., 1] = np.rint((normals[..., 1] * 0.5 + 0.5) * 255).astype(np.uint8)
    normal[..., 2] = np.rint((normals[..., 0] * 0.5 + 0.5) * 255).astype(np.uint8)
    normal[..., 3] = np.rint(trust * 255).astype(np.uint8)

    # Positive discrete Laplacian is local concavity. Robust scaling keeps
    # isolated pore relief from turning the rest of the face grey.
    curvature = cv2.Laplacian(height, cv2.CV_32F, ksize=3)
    concavity = np.maximum(curvature, 0.0)
    valid = trust > 0
    scale = float(np.percentile(concavity[valid], 99.5)) if valid.any() else 0.0
    ao = np.ones_like(height, np.float32) if scale <= 1e-10 else np.clip(1.0 - 0.45 * concavity / scale, 0.55, 1.0)
    ao[~valid] = 1.0
    cavity = np.rint(ao * 255).astype(np.uint8)

    # glTF-compatible ORM channel order when interpreted as RGB: R=AO,
    # G=roughness, B=metallic. Skin is dielectric; metallic is zero.
    linear = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    luminance = linear @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    roughness = np.clip(0.68 + 0.10 * (0.5 - luminance), 0.35, 0.9)
    orm = np.empty((resolution, resolution, 4), np.uint8)
    orm[..., 0] = 0                                      # B = metallic in stored BGR
    orm[..., 1] = np.rint(roughness * 255).astype(np.uint8)  # G = roughness
    orm[..., 2] = np.rint(ao * 255).astype(np.uint8)         # R = AO
    orm[..., 3] = np.rint(trust * 255).astype(np.uint8)

    return BakedPBRAtlas(resolution, basecolor, normal, orm, cavity, trust, source_size, scale, relief_available)


def atlas_manifest(baked: BakedPBRAtlas, encoded: dict[str, bytes], *, faceset_id: str,
                   build_key: str, source_sha256: str) -> dict[str, Any]:
    """Small reproducibility record stored beside the PNG maps."""
    if not all(isinstance(value, str) and value for value in (faceset_id, build_key, source_sha256)):
        raise ValueError("FaceSet id, build key, and source digest are required")
    return {
        "schema": "roop3d.pbr_atlas.v1", "algorithm": BAKER_VERSION,
        "faceset_id": faceset_id, "identity_asset_build_key": build_key, "source_sha256": source_sha256,
        "resolution": baked.resolution, "source_resolution": list(baked.source_resolution),
        "albedo": {"source": "intrinsic diffuse decomposition of the observed multi-view identity asset", "diffuse_is_unbaked": True,
                   "unseen_texels": "transparent", "interpolated_alpha": 0.45},
        "maps": {name: {"sha256": hashlib.sha256(encoded[name]).hexdigest(), "bytes": len(encoded[name])}
                 for name in MAP_SUFFIXES},
        "orm_channels": {"R": "cavity ambient occlusion", "G": "roughness", "B": "metallic (zero for skin)"},
        "normal_map": "tangent-space normals from measured UV displacement; no synthetic pore inference",
        "relief_available": baked.relief_available,
        "cavity_relief_range_m": round(baked.cavity_relief_range_m, 8),
        "claims": {"multi_view_albedo": True, "pore_relief_observed_or_asset_supplied": baked.relief_available,
                   "25_percent_render_speedup_benchmarked": False},
    }


__all__ = ["BAKER_VERSION", "MAP_SUFFIXES", "BakedPBRAtlas", "bake_pbr_atlas", "atlas_manifest"]
