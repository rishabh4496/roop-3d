"""Stage 17: photorealistic 3D swap core and the FFmpeg-piped end-to-end video job runner.

Imports are lazy so ``import roop_3d.pipeline`` costs nothing (the server only pays for the runner when the
``photoreal_3d_ffmpeg`` engine is selected)::

    from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
    from roop_3d.pipeline.video_job_runner import EndToEndVideoJobRunner
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
