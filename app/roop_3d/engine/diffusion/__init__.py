"""Optional, lazily loaded diffusion VFX backends and deterministic ROI tiling."""

from .base import BackendCapability, BackendState, DiffusionBackend, MemoryPlan
from .flux_backend import FluxBackend
from .sdxl_backend import SDXLBackend
from .tile_manager import Tile, TileManager

__all__ = [
    "BackendCapability", "BackendState", "DiffusionBackend", "FluxBackend",
    "MemoryPlan", "SDXLBackend", "Tile", "TileManager",
]
