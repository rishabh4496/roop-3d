"""Lower trilinear volumetric sampling to two standard 4-D GridSample nodes.

Depth planes are tiled along height with zero guard rows. Sampling the two
adjacent planes and interpolating depth preserves trilinear semantics without
custom operators. Only bilinear/linear, zeros/border exports are admitted.
Reference models are never modified. Importing this module loads no runtime.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from roop_3d.models.owned_files import model_source, owned_path


def lower_grid_sample_3d(model):
    import onnx
    from onnx import helper as h, numpy_helper as nh

    result = copy.deepcopy(model)
    if any(a.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS)
           for n in result.graph.node for a in n.attribute):
        raise ValueError("Control-flow subgraphs require a separately reviewed conversion")
    version = next((o.version for o in result.opset_import if o.domain == ""), 0)
    if not 16 <= version <= 20:
        raise ValueError("GridSample lowering supports reviewed ONNX opsets 16 through 20")
    inferred = onnx.shape_inference.infer_shapes(result)
    ranks = {v.name: len(v.type.tensor_type.shape.dim)
             for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)
             if v.type.tensor_type.HasField("shape")}
    types = {v.name: v.type.tensor_type.elem_type
             for v in (*inferred.graph.input, *inferred.graph.value_info)}
    used = {s for n in result.graph.node for s in (*n.input, *n.output)}
    used.update(v.name for v in result.graph.initializer)
    nodes = []
    replaced = []
    serial = 0

    def name(label):
        nonlocal serial
        while True:
            serial += 1
            candidate = f"roop3d_gs_{serial}_{label}"
            if candidate not in used:
                used.add(candidate)
                return candidate

    def const(value, dtype=np.int64):
        output = name("constant")
        result.graph.initializer.append(nh.from_array(np.asarray(value, dtype=dtype), output))
        return output

    def op(kind, *inputs, **attrs):
        output = name(kind)
        nodes.append(h.make_node(kind, list(inputs), [output], name=output, **attrs))
        return output

    for node in result.graph.node:
        if node.op_type != "GridSample" or node.domain:
            if "gridsample" in node.op_type.casefold() and node.domain:
                raise ValueError("Custom GridSample operators require a reviewed export")
            nodes.append(node)
            continue
        rank = ranks.get(node.input[0])
        if rank == 4:
            nodes.append(node)
            continue
        if rank != 5 or ranks.get(node.input[1]) != 5:
            raise ValueError(f"Cannot establish volumetric input/grid ranks for {node.name}")
        if types.get(node.input[0]) != onnx.TensorProto.FLOAT:
            raise ValueError("Convert the FP32 reference graph before any FP16 optimization")
        attrs = {a.name: h.get_attribute_value(a) for a in node.attribute}
        mode = attrs.get("mode", b"bilinear")
        padding = attrs.get("padding_mode", b"zeros").decode()
        align = int(attrs.get("align_corners", 0))
        if mode not in (b"bilinear", b"linear") or padding not in ("zeros", "border"):
            raise ValueError("Only trilinear sampling with zeros/border padding is supported")
        data, grid = node.input
        zero, one, two, half = [const(v, np.float32) for v in (0, 1, 2, .5)]
        i_one, i_two = const([1]), const([2])
        shape = op("Shape", data)
        dims = [op("Gather", shape, const([i]), axis=0) for i in range(5)]
        n, c, d, height, width = dims
        df, hf, wf = [op("Cast", dim, to=onnx.TensorProto.FLOAT) for dim in (d, height, width)]
        gshape = op("Shape", grid)
        outdims = [op("Gather", gshape, const([i]), axis=0) for i in (1, 2, 3)]
        outdepth, outheight, outwidth = outdims
        merged_height = op("Mul", outdepth, outheight)
        tileheight = op("Add", height, i_two)
        totalheight = op("Mul", d, tileheight)
        thf = op("Cast", tileheight, to=onnx.TensorProto.FLOAT)
        totalf = op("Cast", totalheight, to=onnx.TensorProto.FLOAT)
        # Guard rows prevent bilinear filtering from leaking into adjacent planes.
        padded = op("Pad", data, const([0, 0, 0, 1, 0, 0, 0, 0, 1, 0]), mode="constant")
        tiled = op("Reshape", padded, op("Concat", n, c, totalheight, width, axis=0))
        gx, gy, gz = [op("Gather", grid, const(i), axis=4) for i in range(3)]

        def pixel(g, size):
            scale = op("Sub", size, one) if align else size
            v = op("Div", op("Mul", op("Add", g, one), scale), two)
            return v if align else op("Sub", v, half)

        y, z = pixel(gy, hf), pixel(gz, df)
        if padding == "border":
            y = op("Clip", y, zero, op("Sub", hf, one))
            z = op("Clip", z, zero, op("Sub", df, one))
        else:
            y = op("Clip", y, const(-1, np.float32), hf)
            z = op("Clip", z, const(-1, np.float32), df)
        lower = op("Floor", z)
        upper = op("Add", lower, one)
        fraction = op("Sub", z, lower)
        samples = []
        for depth, weight in ((lower, op("Sub", one, fraction)), (upper, fraction)):
            valid = op("And", op("GreaterOrEqual", depth, zero), op("Less", depth, df))
            weight = op("Mul", weight, op("Cast", valid, to=onnx.TensorProto.FLOAT))
            depth = op("Clip", depth, zero, op("Sub", df, one))
            tile_y = op("Add", op("Add", y, one), op("Mul", depth, thf))
            if align:
                tile_y = op("Sub", op("Div", op("Mul", tile_y, two), op("Sub", totalf, one)), one)
            else:
                tile_y = op("Sub", op("Div", op("Mul", op("Add", tile_y, half), two), totalf), one)
            xy = op("Concat", op("Unsqueeze", gx, const([4])),
                    op("Unsqueeze", tile_y, const([4])), axis=4)
            flat_grid = op("Reshape", xy, op("Concat", n, merged_height, outwidth, i_two, axis=0))
            # Opset 20 renamed bilinear to linear; rank and interpolation stay 2-D.
            sampled = op("GridSample", tiled, flat_grid, mode="linear" if version == 20 else "bilinear",
                         padding_mode=padding, align_corners=align)
            sampled = op("Reshape", sampled, op("Concat", n, c, outdepth, outheight, outwidth, axis=0))
            samples.append(op("Mul", sampled, op("Unsqueeze", weight, i_one)))
        nodes.append(h.make_node("Add", samples, list(node.output), name=name("depth_interpolation")))
        replaced.append(node.name or node.output[0])
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    onnx.checker.check_model(result)
    return result, tuple(replaced)


def convert_grid_sample(source: Path, destination: Path) -> tuple[str, ...]:
    """Create a new owned derivative, never overwrite either existing artifact."""
    import onnx
    import os
    import uuid
    source, destination = model_source(source), owned_path(destination)
    if source == destination or destination.exists():
        raise ValueError("Choose a new derivative path; reference and existing outputs are immutable")
    model = onnx.load(source, load_external_data=False)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError("External tensor data needs a separately verified bundle")
    model, changed = lower_grid_sample_3d(model)
    if not changed:
        raise ValueError("No volumetric GridSample nodes found")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned_path(destination.with_name(destination.name + f".{uuid.uuid4().hex}.tmp"))
    try:
        onnx.save(model, temporary)
        # Exclusive publication also protects against an output appearing mid-conversion.
        with open(owned_path(destination), "xb") as target, open(temporary, "rb") as stream:
            import shutil
            shutil.copyfileobj(stream, target)
    finally:
        if temporary.exists():
            os.unlink(owned_path(temporary))
    return changed
