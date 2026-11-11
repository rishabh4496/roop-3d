"""Stage 12 interchange: binary glTF serialization and the optional, GPL-isolated Blender bridge.

Imports are lazy so ``import roop_3d.io`` costs nothing.
"""
from __future__ import annotations

_EXPORTS = {
    "GLBBinarySerializer": "glb_export", "HeadTrack": "glb_export", "MorphTarget": "glb_export", "parse_glb": "glb_export",
    "BlenderMCPBridge": "blender_bridge", "BlenderBridgeError": "blender_bridge",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
