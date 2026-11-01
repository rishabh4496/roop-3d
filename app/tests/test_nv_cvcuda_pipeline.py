"""Requested Stage 14 gate path; implementation lives in the owned test package."""
from pathlib import Path
import sys

# Normalize before import: project boundary checks deliberately reject '..'.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from roop_3d.tests.test_nv_cvcuda_pipeline import *  # noqa: F401,F403
