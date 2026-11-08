"""Switches for the asynchronous pipeline.  Environment variables, so no settings-schema or UI-contract churn.

``ROOP3D_ASYNC_PIPELINE=0``  run the same four stage functions inline on one thread (the pre-Defect-#3 behaviour)
``ROOP3D_PIPELINE_QUEUE=n``  bounded queue depth between stages, 1..64 (default 8: caps in-flight frames / VRAM)
``ROOP3D_PINNED_FRAMES=0``   do not stage decoded frames in page-locked host buffers
"""
from __future__ import annotations

import os

from roop_3d.async_pipeline.buffers import DEFAULT_QUEUE_SIZE


def async_pipeline_enabled() -> bool:
    return os.environ.get("ROOP3D_ASYNC_PIPELINE", "1") != "0"


def pinned_enabled() -> bool:
    return os.environ.get("ROOP3D_PINNED_FRAMES", "1") != "0"


def async_queue_depth() -> int:
    try:
        return max(1, min(64, int(os.environ.get("ROOP3D_PIPELINE_QUEUE", DEFAULT_QUEUE_SIZE))))
    except ValueError:
        return DEFAULT_QUEUE_SIZE
