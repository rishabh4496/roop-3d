"""Verify and publish an owned dynamic ONNX model; reference weights are immutable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "app"))

from roop_3d.models.dynamic_shapes import convert, StaticBatchRequired
from roop_3d.models.owned_files import owned_path


def parse_batches(value: str) -> tuple[int, ...]:
    batches = tuple(int(v) for v in value.split(","))
    if not batches or tuple(sorted(set(batches))) != batches or batches[0] != 1 or batches[-1] < 2:
        raise argparse.ArgumentTypeError("Declare sorted positive batches including 1 and a multi-batch size")
    return batches


def make_synthetic(path: Path) -> Path:
    """A real static Reshape graph, not a fake conversion result."""
    import onnx
    import numpy as np
    path = owned_path(path)
    if path.exists():
        raise ValueError("Synthetic fixture already exists; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)
    graph = onnx.helper.make_graph([
        onnx.helper.make_node("Reshape", ["image", "shape"], ["output"])
    ], "stage2_synthetic", [onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 2, 2])],
       [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 12])],
       [onnx.numpy_helper.from_array(np.array([1, 12], dtype=np.int64), "shape")])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-batches", type=parse_batches, default=(1, 4, 8))
    parser.add_argument("--shapes", default="{}", help='JSON concrete input shapes, e.g. {"images":[1,3,640,640]}')
    parser.add_argument("--synthetic", action="store_true", help="Create a synthetic static model at MODEL (must not exist)")
    parser.add_argument("--grid-sample-4d", action="store_true",
                        help="Lower volumetric GridSample into 4D spatial sampling plus depth interpolation")
    args = parser.parse_args()
    try:
        if args.grid_sample_4d:
            from roop_3d.models.grid_sample import convert_grid_sample
            if args.output is None or args.synthetic:
                parser.error("--grid-sample-4d requires --output and an existing reference model")
            changed = convert_grid_sample(args.model, args.output)
            print(f"GRIDSAMPLE_4D_CONVERTED nodes={len(changed)} provider_parity=UNVERIFIED output={args.output}")
            return 0
        if args.synthetic:
            make_synthetic(args.model)
        result = convert(args.model, args.output, batches=args.check_batches,
                         overrides={k: tuple(v) for k, v in json.loads(args.shapes).items()})
        print(f"VERIFIED_DYNAMIC model={result.name} batches={','.join(map(str, args.check_batches))} finite=true parity=true")
        return 0
    except StaticBatchRequired as error:
        print(f"STATIC_BATCH_1 model={args.model.name}: {error}")
        return 0  # Explicitly classified, never advertised as dynamically converted.
    except Exception as error:
        print(f"DYNAMIC_CONVERSION_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
