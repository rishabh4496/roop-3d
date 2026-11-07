"""Stage 3 integration gate; --benchmark-gpu checks installed TensorRT model throughput."""
from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module=r"albumentations(?:\..*)?")
sys.path.insert(0, str(Path(__file__).absolute().parent / "app"))
from roop_3d.tools.verify_stream_integration import main

if __name__ == "__main__":
    raise SystemExit(main())
