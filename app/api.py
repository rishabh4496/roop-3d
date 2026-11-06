"""Unified FastAPI backend entrypoint for Roop 3D.

Exposes the unified application and central state store managing source faces,
.fsz facesets, target media, render modes, and active processors.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure app directory and repository root are present on sys.path
_current_dir = Path(__file__).resolve().parent
_root_dir = _current_dir.parent
for _p in (str(_current_dir), str(_root_dir)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI
from roop_3d.core.settings import Settings
from roop_3d.server.app import create_app
from roop_3d.server.unified_state import (
    ActiveProcessorsPayload,
    RenderMode,
    RenderModePayload,
    SourceFacePayload,
    UnifiedStateStore,
    router as state_router,
    state,
)


def create_unified_app(settings: Optional[Settings] = None, *, repo_root: Optional[Path] = None) -> FastAPI:
    """Create the unified FastAPI application with all 3D, faceset, timeline, and state routes."""
    resolved_root = repo_root or _current_dir
    app_instance = create_app(settings=settings, repo_root=resolved_root)
    app_instance.state.unified_state = state
    return app_instance


# Main module-level app instance
app = create_unified_app()

__all__ = [
    "app",
    "state",
    "create_unified_app",
    "RenderMode",
    "UnifiedStateStore",
    "RenderModePayload",
    "SourceFacePayload",
    "ActiveProcessorsPayload",
]
