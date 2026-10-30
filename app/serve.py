"""Standalone entrypoint; the import path contains only this app's source."""
from roop_3d.server.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
