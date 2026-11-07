"""Standalone entrypoint; the import path contains only this app's source."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure app directory is on sys.path
_app_dir = Path(__file__).resolve().parent
if str(_app_dir) not in sys.path:
    sys.path.insert(0, str(_app_dir))

# Automatically inject CUDA / cuDNN / TensorRT DLLs & ensure runtime directories
from roop_3d.core.dll_injection import setup_environment
setup_environment(_app_dir.parent)

from roop_3d.server.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
