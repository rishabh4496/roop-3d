"""Stage 17: photorealistic 3D swap core and the FFmpeg-piped end-to-end video job runner.

Imports are lazy so ``import roop_3d.pipeline`` costs nothing (the server only pays for the runner when the
``photoreal_3d_ffmpeg`` engine is selected)::

    from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
    from roop_3d.pipeline.video_job_runner import EndToEndVideoJobRunner

The FFmpeg runner refuses work until ``true3d_renderer.is_wired()`` and ``true3d_guard`` admit the full-head path. Its compositor
accepts only a perspective mesh raster with full-frame coverage, normals and semantic masks; the bounded image-space ROI is a
compositing optimization, not a face warp. ``pipeline.image_ops`` supplies image/mask and multiband-compositing primitives;
there is no 106-point similarity/TPS replacement fallback. Semantic parsing and configured visible-face networks are mandatory;
missing or invalid models fail with structured diagnostics. MakeHuman's legacy identity-asset rig remains distinct from the
native BFM/FLAME backbone and is not substituted for it. The compositor restores zero-alpha scene pixels after reconstruction.
"""
from __future__ import annotations

__all__ = ["Photorealistic3DSwapCore", "EndToEndVideoJobRunner"]


def __getattr__(name: str):
    if name == "Photorealistic3DSwapCore":
        from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
        return Photorealistic3DSwapCore
    if name == "EndToEndVideoJobRunner":
        from roop_3d.pipeline.video_job_runner import EndToEndVideoJobRunner
        return EndToEndVideoJobRunner
    raise AttributeError(name)
