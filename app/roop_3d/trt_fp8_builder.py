"""Explicit E4M3FN Q/DQ admission, never an implicit FP8 builder flag.

Calibration/export is an offline operation using representative local images.
Only a separately supplied quantized artifact is admitted; reference weights are
never rewritten. Native build/canary and parity validation remain mandatory.
"""
from __future__ import annotations

from pathlib import Path

HEAVY_RESTORERS = frozenset(("gpen-bfr-1024", "ultramax_bfr", "restoreformer_plus_plus", "swin2_sr_x4"))


def model_name(path):
    name = Path(path).stem.casefold()
    while name.endswith(("_dynamic", ".batch")):
        name = name.removesuffix("_dynamic").removesuffix(".batch")
    return name


def validate_swapper_fp32(model):
    """Reject pre-quantized or cast-down swapper graphs, including subgraphs."""
    import onnx
    low = {onnx.TensorProto.FLOAT16, onnx.TensorProto.BFLOAT16,
           onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.FLOAT8E4M3FNUZ,
           onnx.TensorProto.FLOAT8E5M2, onnx.TensorProto.FLOAT8E5M2FNUZ}
    def check(graph):
        if any(t.data_type in low for t in graph.initializer) or any(
                v.type.tensor_type.elem_type in low for v in (*graph.input, *graph.output, *graph.value_info)):
            raise ValueError("inswapper_128 must remain FP32")
        for node in graph.node:
            if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
                raise ValueError("inswapper_128 must remain FP32")
            for attr in node.attribute:
                if (attr.name == "to" and attr.i in low) or (attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type in low):
                    raise ValueError("inswapper_128 must remain FP32")
                if attr.type == onnx.AttributeProto.GRAPH:
                    check(attr.g)
                for graph in attr.graphs:
                    check(graph)
    check(model.graph)


def validate_fp8_graph(model, original_name, compute_capability, tensorrt_version):
    import onnx
    if model_name(original_name) not in HEAVY_RESTORERS:
        raise ValueError("FP8 is restricted to the four heavy restorers; inswapper is FP32 only")
    if tuple(compute_capability) < (8, 9) or int(tensorrt_version.split(".")[0]) != 10:
        raise ValueError("FP8 requires observed SM 8.9+ and TensorRT 10")
    onnx.checker.check_model(model)
    constants = {v.name: v for v in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = attr.t
    count = 0
    for node in model.graph.node:
        if node.op_type != "QuantizeLinear":
            continue
        if len(node.input) < 3 or node.input[2] not in constants:
            continue
        if constants[node.input[2]].data_type != onnx.TensorProto.FLOAT8E4M3FN:
            raise ValueError("FP8 Q/DQ requires E4M3FN zero points")
        zero = constants[node.input[2]]
        if any(zero.raw_data) or any(zero.int32_data):
            raise ValueError("FP8 requires symmetric zero points")
        import numpy as np
        scale = constants.get(node.input[1])
        if scale is None:
            raise ValueError("FP8 requires calibrated constant scales")
        values = onnx.numpy_helper.to_array(scale)
        if not values.size or not np.isfinite(values).all() or not (values > 0).all():
            raise ValueError("FP8 scales must be finite and positive")
        if not any(n.op_type == "DequantizeLinear" and n.input[0] == node.output[0]
                   and list(n.input[1:]) == list(node.input[1:]) for n in model.graph.node):
            raise ValueError("FP8 quantization must have matching explicit dequantization")
        count += 1
    if not count:
        raise ValueError("Model has no calibrated explicit E4M3FN Q/DQ pairs")
    return {"format": "e4m3fn", "qdq_pairs": count,
            "compute_capability": list(compute_capability), "tensorrt": tensorrt_version}
