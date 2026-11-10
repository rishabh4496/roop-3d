"""Write `<name>.static.onnx`: the exported encoder with its output shapes stated statically.

The upstream exports leave sliced/Tanh output dims symbolic, so the runtime contract check fails when ONNX Runtime
runs with graph optimisation disabled (shape inference then never resolves them). This copies each graph, sets the
declared contract dims on the graph outputs ONLY (weights/nodes untouched) and proves equivalence on seeded inputs.

Run from app/:
  python -B -m roop_3d.tools.annotate_onnx_shapes --role expression --graph .../expression.onnx
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

from roop_3d.models.owned_files import owned_path
from roop_3d.reconstruction.native_backbone import GRAPH_CONTRACTS


def annotate(role: str, graph: Path) -> dict:
    source = owned_path(graph)
    target = owned_path(source.with_name(source.stem + ".static.onnx"))
    if target.exists():
        raise ValueError(f"{target.name} already exists; remove it deliberately to regenerate")
    inputs, outputs = GRAPH_CONTRACTS[role]
    model = onnx.load(str(source), load_external_data=True)
    declared = model.graph.output
    if {o.name for o in declared} != outputs.keys():
        raise ValueError("Graph outputs do not match the contract names")
    for output in declared:
        dims = output.type.tensor_type.shape.dim
        shape = outputs[output.name]
        if len(dims) != len(shape):
            raise ValueError(f"{output.name}: rank {len(dims)} differs from contract {shape}")
        for dim, value in zip(dims, shape):
            if dim.HasField("dim_value") and dim.dim_value != value:
                raise ValueError(f"{output.name}: static dim {dim.dim_value} contradicts contract {value}")
            dim.ClearField("dim_param")
            dim.dim_value = value
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    # Equivalence on two seeded inputs (same weights; only declared shapes differ).
    rng = np.random.default_rng(7)
    session_a = ort.InferenceSession(str(source), providers=["CPUExecutionProvider"])
    session_b = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    worst = 0.0
    for _ in range(2):
        feed = {name: rng.standard_normal(shape).astype(np.float32) * .5 for name, shape in inputs.items()}
        for left, right in zip(session_a.run(None, feed), session_b.run(None, feed)):
            if left.shape != right.shape or not np.isfinite(right).all():
                raise ValueError("Annotated graph output shape/finite mismatch")
            worst = max(worst, float(np.abs(left - right).max()))
    if worst != 0.0:
        raise ValueError(f"Annotated graph differs from the original by {worst}")
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = {"role": role, "source": source.name, "source_sha256": digest(source), "file": target.name,
              "sha256": digest(target), "max_abs_difference_on_two_seeded_inputs": worst,
              "scope": "output shape annotation only; no weights or nodes changed"}
    target.with_suffix(".static.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role", required=True, choices=sorted(GRAPH_CONTRACTS))
    parser.add_argument("--graph", required=True)
    args = parser.parse_args()
    print(json.dumps(annotate(args.role, Path(args.graph)), indent=2))


if __name__ == "__main__":
    main()
