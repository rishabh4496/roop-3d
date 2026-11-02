"""Shim module forwarding to roop_3d."""
import sys
from roop_3d import *  # noqa: F401,F403
sys.modules.setdefault("roop", sys.modules[__name__])
