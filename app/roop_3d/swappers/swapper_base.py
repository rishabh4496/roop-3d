"""Multi-Swapper Architecture with Unified Interface and PixelBoost Integration.

Supports hot-swapping between:
- InSwapper-128
- HyperSwap-256 (checkpoints 1a, 1b, 1c)
- SimSwap-512
- Ghost-256
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, Tuple, Union

import numpy as np

from roop_3d.swappers.pixel_boost import PixelBoost

SwapperModelId = Literal[
    "inswapper_128",
    "hyperswap_1a_256",
    "hyperswap_1b_256",
    "hyperswap_1c_256",
    "simswap_unofficial_512",
    "ghost_1_256",
]

EmbeddingMode = Literal["normed_emap", "normed", "converted_norm", "converted_raw"]


@dataclass
class SwapperConfig:
    model_id: str
    name: str
    resolution: int
    embedding_mode: EmbeddingMode
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    denormalize: bool
    precision_policy: str
    supports_dynamic_batch: bool


SWAPPER_CONFIGS: dict[str, SwapperConfig] = {
    "inswapper_128": SwapperConfig(
        model_id="inswapper_128",
        name="inswapper_128.onnx",
        resolution=128,
        embedding_mode="normed_emap",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        denormalize=False,
        precision_policy="FP32",
        supports_dynamic_batch=True,
    ),
    "hyperswap_1a_256": SwapperConfig(
        model_id="hyperswap_1a_256",
        name="hyperswap_1a_256.onnx",
        resolution=256,
        embedding_mode="normed",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        denormalize=True,
        precision_policy="FP16",
        supports_dynamic_batch=True,
    ),
    "hyperswap_1b_256": SwapperConfig(
        model_id="hyperswap_1b_256",
        name="hyperswap_1b_256.onnx",
        resolution=256,
        embedding_mode="normed",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        denormalize=True,
        precision_policy="FP16",
        supports_dynamic_batch=True,
    ),
    "hyperswap_1c_256": SwapperConfig(
        model_id="hyperswap_1c_256",
        name="hyperswap_1c_256.onnx",
        resolution=256,
        embedding_mode="normed",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        denormalize=True,
        precision_policy="FP16",
        supports_dynamic_batch=True,
    ),
    "simswap_unofficial_512": SwapperConfig(
        model_id="simswap_unofficial_512",
        name="simswap_unofficial_512.onnx",
        resolution=512,
        embedding_mode="converted_norm",
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        denormalize=False,
        precision_policy="FP16",
        supports_dynamic_batch=True,
    ),
    "ghost_1_256": SwapperConfig(
        model_id="ghost_1_256",
        name="ghost_1_256.onnx",
        resolution=256,
        embedding_mode="converted_raw",
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        denormalize=True,
        precision_policy="FP16",
        supports_dynamic_batch=True,
    ),
}


class BaseSwapper(ABC):
    """Abstract base class for all identity face swappers."""

    def __init__(self, config: SwapperConfig):
        self.config = config
        self.resolution = config.resolution
        self.pixel_boost: PixelBoost | None = None

    def enable_pixel_boost(self, boost_size: int = 512) -> None:
        """Enable PixelBoost sub-patch tiling to the specified resolution."""
        if boost_size > self.resolution:
            self.pixel_boost = PixelBoost(model_size=self.resolution, boost_size=boost_size)
        else:
            self.pixel_boost = None

    def prepare_input_tensor(self, crop: np.ndarray) -> np.ndarray:
        """Normalize input crop according to swapper model specification."""
        arr = np.asarray(crop, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[-1] == 3:  # (H, W, 3) BGR/RGB -> (1, 3, H, W)
            arr = arr.transpose(2, 0, 1)[np.newaxis, ...]
        elif arr.ndim == 3:  # (3, H, W) -> (1, 3, H, W)
            arr = arr[np.newaxis, ...]

        # Rescale from 0..255 to 0..1
        if arr.max() > 1.0:
            arr = arr / 255.0

        mean = np.array(self.config.mean, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.array(self.config.std, dtype=np.float32).reshape(1, 3, 1, 1)
        return (arr - mean) / std

    def postprocess_output_tensor(self, out: np.ndarray) -> np.ndarray:
        """Denormalize and clip swapper model output back to 0..1 float32."""
        arr = np.asarray(out, dtype=np.float32)
        if self.config.denormalize:
            arr = (arr + 1.0) / 2.0
        return np.clip(arr, 0.0, 1.0)

    @abstractmethod
    def swap_core(self, target_patches: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """Run batched inference over target patches with source embedding."""
        pass

    def swap(self, target_crop: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """Execute swap pass with optional PixelBoost sub-patch tiling."""
        if self.pixel_boost is not None:
            # 1. Unfold oversized crop into batched patches
            patches, coords = self.pixel_boost.unfold_sub_patches(target_crop)
            # 2. Normalize and run single batched inference
            norm_patches = self.prepare_input_tensor(patches)
            swapped_patches = self.swap_core(norm_patches, source_embedding)
            denorm_patches = self.postprocess_output_tensor(swapped_patches)
            # 3. Reconstruct with raised-cosine feather blending
            reconstructed = self.pixel_boost.reconstruct_from_sub_patches(denorm_patches, coords)
            return reconstructed

        # Single patch path
        norm_input = self.prepare_input_tensor(target_crop)
        swapped = self.swap_core(norm_input, source_embedding)
        return self.postprocess_output_tensor(swapped)


class SwapperFactory:
    """Factory and registry manager for instantiating multi-swapper engines."""

    @staticmethod
    def get_swapper(
        model_id: str = "inswapper_128",
        pixel_boost_size: int | None = None,
        inference_backend: Any = None,
    ) -> BaseSwapper:
        config = SWAPPER_CONFIGS.get(model_id)
        if not config:
            raise ValueError(f"Unknown swapper model_id '{model_id}'. Available: {list(SWAPPER_CONFIGS.keys())}")

        class GenericSwapper(BaseSwapper):
            def __init__(self, cfg: SwapperConfig, backend: Any):
                super().__init__(cfg)
                self.backend = backend

            def swap_core(self, target_patches: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
                if self.backend is not None:
                    return self.backend(target_patches, source_embedding)
                # Synthetic reference identity path for testing/mock execution
                return target_patches

        swapper = GenericSwapper(config, inference_backend)
        if pixel_boost_size and pixel_boost_size > config.resolution:
            swapper.enable_pixel_boost(pixel_boost_size)
        return swapper
