"""Roop 3D: isolated local video-processing application."""
import sys

__version__ = "0.1.0"

# Register compatibility alias so references to legacy module paths map to roop_3d
sys.modules.setdefault("roop", sys.modules[__name__])
