"""Explicit local TRT builds; never called merely to boot the API."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "app"))
from roop_3d.trt_session_builder import BatchProfile, TensorRTSessionBuilder
from roop_3d.models.dynamic_shapes import static_batch_reason


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("models", type=Path, nargs="+")
    result.add_argument("--batches", default="1,4,8")
    result.add_argument("--opt-batch", type=int, default=4)
    result.add_argument("--hardware-profile", choices=("6gb", "12gb"), default="6gb")
    result.add_argument("--shapes", default="{}", help="JSON input shape overrides")
    return result


def main() -> int:
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        batches = tuple(int(b) for b in args.batches.split(","))
        shapes = {k: tuple(v) for k, v in json.loads(args.shapes).items()}
        for model in args.models:
            profile = BatchProfile((1,), 1) if static_batch_reason(model.name) else BatchProfile(batches, args.opt_batch)
            builder = TensorRTSessionBuilder(hardware_profile=args.hardware_profile,
                                            telemetry=lambda event: print(json.dumps(event), flush=True))
            session = builder.build(model, profile=profile, shapes=shapes)
            print(json.dumps({"model": model.name, **builder.last_build}), flush=True)
            del session
        return 0
    except Exception as error:
        print(f"TRT_PREBUILD_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
