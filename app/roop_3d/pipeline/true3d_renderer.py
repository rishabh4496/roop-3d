"""Hook module for the true-3D full-head render path (``roop_3d.server.true3d`` looks this module up).

The FullHeadEngine render path (perspective K[R|t], Z-buffered rasterization, PBR + subsurface scattering or 3D Gaussian
shading, over a watertight FLAME / MakeHuman head) is NOT wired into the session / job runners yet. This module reports
that truthfully: ``is_wired()`` is False and every entry point raises ``True3DRendererUnavailable``. When the renderer is
connected, flip ``is_wired`` (and implement the three entry points); the guard in ``roop_3d.core.true3d_guard`` keeps
2D / 2.5D inputs out either way.
"""
from __future__ import annotations

from typing import Any

from roop_3d.core.true3d_guard import (
    NOT_WIRED_MESSAGE,
    True3DArchitectureGuard,
    True3DRendererUnavailable,
)

#: The pieces that exist and are reusable once the renderer is connected.
BUILDING_BLOCKS = (
    "roop_3d.reconstruction.full_head.FullHeadEngine",
    "roop_3d.reconstruction.full_head.GPUHeadRasterizer",
    "roop_3d.rendering.physical_buffers.PhysicalBufferRenderer",
    "roop_3d.rendering.uv_baker",
    "roop_3d.enhancers.uv_detailer.UVTextureDetailer",
)


def is_wired() -> bool:
    return False


def describe() -> dict[str, Any]:
    return {"wired": False, "message": NOT_WIRED_MESSAGE, "building_blocks": list(BUILDING_BLOCKS),
            "guard": True3DArchitectureGuard().describe()}


def _unavailable(what: str):
    raise True3DRendererUnavailable(
        f"{NOT_WIRED_MESSAGE}: '{what}' needs the FullHeadEngine render path, which is not connected to the session yet.",
        {"operation": what})


def render_preview(state, request, cancel, *, raw: bool = False, final_only: bool = False) -> dict:
    _unavailable("render_preview")


def open_live_session(state, config, true3d):
    _unavailable("open_live_session")


def render_frame(*args, **kwargs):
    _unavailable("render_frame")
