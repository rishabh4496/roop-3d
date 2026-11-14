"""Stage 17: photorealistic 3D swap core and the FFmpeg-piped end-to-end video job runner.

Imports are lazy so ``import roop_3d.pipeline`` costs nothing (the server only pays for the runner when the
``photoreal_3d_ffmpeg`` engine is selected)::

    from roop_3d.pipeline.photorealistic_swap_core import Photorealistic3DSwapCore
    from roop_3d.pipeline.video_job_runner import EndToEndVideoJobRunner

The FFmpeg path preserves mesh/projection coverage, presence fades and upstream enhancement from the session. Standard
replacements still use 106-point similarity/TPS geometry and a heuristic forehead, not a fitted 3D reconstruction.
Semantic parsing and every configured visible-face network are required; unavailable or invalid models fail the job
with structured diagnostics rather than silently dropping occlusion. Parser trust uncertainty is retained in reports.
MakeHuman's legacy identity-asset rig is distinct from the native BFM/FLAME backbone and is not substituted for it.
The compositor normalizes layered alpha and restores zero-alpha background pixels after multiband reconstruction.
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
