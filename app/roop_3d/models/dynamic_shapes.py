"""Conservative batch graph surgery, admitted only after real ORT execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import uuid

import numpy as np

from roop_3d.audit.isolation import sha256_file
from roop_3d.models.owned_files import owned_path, model_source
from roop_3d.audit.project_paths import project_root


class DynamicShapeError(RuntimeError):
    pass


class StaticBatchRequired(DynamicShapeError):
    pass


def static_batch_reason(name: str) -> str | None:
    if "gpen" in name.casefold() and any(size in name for size in ("256", "512")):
        return "GPEN internal weight reshaping/grouped convolution requires static batch-size-1 compilation"
    return None


def runtime_inputs(model):
    initializers = {v.name for v in model.graph.initializer}
    return [v for v in model.graph.input if v.name not in initializers]


def input_shapes(model, batch: int, overrides: dict[str, tuple[int, ...]] | None = None):
    result = {}
    for value in runtime_inputs(model):
        dims = value.type.tensor_type.shape.dim
        shape = list((overrides or {}).get(value.name, tuple(d.dim_value for d in dims)))
        if len(shape) != len(dims):
            raise DynamicShapeError(f"Rank mismatch for {value.name}")
        if shape and value.name not in ("w", "fidelity_weight"):
            shape[0] = batch
        if any(d <= 0 for d in shape):
            raise DynamicShapeError(f"Supply concrete nonbatch dimensions for {value.name} with --shapes")
        result[value.name] = tuple(shape)
    return result


def sample_inputs(model, batch: int, overrides=None, *, seed: int = 83, fidelity: float = 0.5):
    import onnx
    rng = np.random.default_rng(seed)
    shapes = input_shapes(model, batch, overrides)
    result = {}
    for value in runtime_inputs(model):
        dtype = onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        shape = shapes[value.name]
        if value.name in ("w", "fidelity_weight"):
            result[value.name] = np.full(shape, fidelity, dtype=dtype)
        elif np.issubdtype(dtype, np.floating):
            result[value.name] = rng.uniform(-0.5, 0.5, shape).astype(dtype)
        else:
            raise DynamicShapeError(f"Explicit representative inputs required for non-floating input {value.name}")
    return result


def assert_finite(outputs) -> None:
    if not outputs or any(not np.isfinite(np.asarray(v)).all() for v in outputs):
        raise DynamicShapeError("Inference produced empty or non-finite outputs")


def surgery(model, model_name: str):
    import onnx
    reason = static_batch_reason(model_name)
    if reason:
        raise StaticBatchRequired(reason)
    result = copy.deepcopy(model)
    graph = result.graph
    # A live fidelity input is mandatory; do not invent the location of a frozen w.
    if "codeformer" in model_name.casefold():
        inputs = {v.name: v for v in graph.input}
        weight = next((n for n in ("w", "fidelity_weight") if n in inputs), None)
        if weight is None:
            raise DynamicShapeError("CodeFormer export must supply w as its second runtime input; re-export the source model")
        required = {v.name for v in graph.output}
        for node in reversed(graph.node):
            if required.intersection(node.output):
                required.update(node.input)
        if weight not in required:
            raise DynamicShapeError("CodeFormer fidelity input is disconnected/frozen; re-export required")
        for initializer in list(graph.initializer):
            if initializer.name == weight:
                graph.initializer.remove(initializer)
        ordered = [v for v in graph.input if v.name != weight]
        ordered.insert(1, inputs[weight])
        del graph.input[:]
        graph.input.extend(ordered)
    try:
        inferred = onnx.shape_inference.infer_shapes(result)
    except (onnx.shape_inference.InferenceError, ValueError):
        inferred = result
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
    constants = {v.name: onnx.numpy_helper.to_array(v) for v in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    constants[node.output[0]] = onnx.numpy_helper.to_array(attr.t)
    # Clone each reshape target rather than mutate constants also used by weights.
    for index, node in enumerate(graph.node):
        if node.op_type == "Squeeze":
            source_shape = shapes.get(node.input[0], [])
            axes = constants.get(node.input[1]) if len(node.input) > 1 else next(
                (np.asarray(a.ints) for a in node.attribute if a.name == "axes"), None)
            if (axes is not None and list(axes) == [0] and len(source_shape) >= 2
                    and source_shape[0] == 1 and all(d > 0 for d in source_shape[1:])):
                # Detector exports squeeze batch 1. Flatten batch/anchors instead,
                # preserving per-sample concatenation verified below.
                name = f"roop3d_squeeze_shape_{index}_{uuid.uuid4().hex}"
                graph.initializer.append(onnx.numpy_helper.from_array(
                    np.array([-1, *source_shape[2:]], np.int64), name))
                del node.attribute[:]
                node.op_type = "Reshape"
                del node.input[1:]
                node.input.append(name)
                continue
        if node.op_type != "Reshape" or node.input[1] not in constants:
            continue
        target = constants[node.input[1]]
        source_shape = shapes.get(node.input[0], [])
        if target.ndim == 1 and len(target) > 1 and target[0] == 1 and source_shape and source_shape[0] == 1:
            replacement = target.copy()
            replacement[0] = 0  # ONNX copy input dimension, never a guessed weight size.
            name = f"roop3d_batch_shape_{index}_{uuid.uuid4().hex}"
            graph.initializer.append(onnx.numpy_helper.from_array(replacement, name))
            node.input[1] = name
    for value in runtime_inputs(result):
        dims = value.type.tensor_type.shape.dim
        if dims and value.name not in ("w", "fidelity_weight"):
            if dims[0].dim_value not in (0, 1):
                raise DynamicShapeError(f"{value.name} is not a batch-1 input")
            dims[0].dim_param = "batch"
    # Static intermediates and squeezed/flattened detector output shapes are stale.
    del graph.value_info[:]
    for value in graph.output:
        for index, dim in enumerate(value.type.tensor_type.shape.dim):
            dim.dim_param = f"{value.name}_dynamic_{index}"
    onnx.checker.check_model(result)
    return result


def validate_conversion(original, converted, batches=(1, 4, 8), overrides=None) -> dict:
    from roop_3d.models.native_libraries import register_native_libraries
    register_native_libraries()
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    reference = ort.InferenceSession(original.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    candidate = ort.InferenceSession(converted.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    checked = []
    for batch in batches:
        feeds = sample_inputs(converted, batch, overrides)
        try:
            actual = candidate.run(None, feeds)
        except Exception as error:
            if checked and batch > 1 and "Reshape" in str(error):
                raise StaticBatchRequired("Batch-1 passed, but an internal reshape rejects multi-batch inputs; compile static batch-size-1") from error
            raise
        assert_finite(actual)
        samples = []
        for index in range(batch):
            single = {n: v if n in ("w", "fidelity_weight") else v[index:index + 1] for n, v in feeds.items()}
            outputs = reference.run(None, single)
            assert_finite(outputs)
            samples.append(outputs)
        for index, output in enumerate(actual):
            pieces = [v[index] for v in samples]
            if batch == 1:
                expected = pieces[0]
            elif pieces[0].ndim and output.shape == (pieces[0].shape[0] * batch, *pieces[0].shape[1:]):
                expected = np.concatenate(pieces, axis=0)
            elif output.shape == (batch, *pieces[0].shape):
                expected = np.stack(pieces)
            else:
                raise DynamicShapeError(f"Output {index} does not preserve independent batch samples: {output.shape}")
            np.testing.assert_allclose(output, expected, rtol=1e-4, atol=1e-5)
        checked.append(batch)
    # Verify w stays a runtime input at more than one value (without claiming source-export parity).
    if "codeformer" in original.graph.name.casefold() or any(v.name in ("w", "fidelity_weight") for v in runtime_inputs(converted)):
        for fidelity in (0.0, 1.0):
            feeds = sample_inputs(converted, 1, overrides, fidelity=fidelity)
            actual, expected = candidate.run(None, feeds), reference.run(None, feeds)
            assert_finite(actual)
            assert_finite(expected)
            for a, b in zip(actual, expected):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-5)
    return {"checked_batches": checked, "finite": True, "single_sample_parity": True, "provider": "CPUExecutionProvider"}


def convert(source: Path, destination: Path | None = None, *, batches=(1, 4, 8), overrides=None) -> Path:
    import onnx
    if not batches or tuple(sorted(set(batches))) != tuple(batches) or batches[0] != 1 or batches[-1] < 2:
        raise DynamicShapeError("Verification requires batch 1 and at least one larger batch")
    source = model_source(source)
    default_directory = source.parent if source.is_relative_to(project_root()) else project_root() / "app/models"
    destination = owned_path(destination or default_directory / (source.stem + "_dynamic.onnx"))
    if destination.suffix.casefold() != ".onnx":
        raise DynamicShapeError("Dynamic model output must use the .onnx extension")
    if destination == source:
        raise DynamicShapeError("Conversion cannot overwrite the reference model")
    if destination.exists() or destination.with_suffix(".json").exists():
        raise DynamicShapeError("Destination exists; choose a new owned output path")
    reason = static_batch_reason(source.name)
    receipt = owned_path(destination.with_suffix(".json"))
    if reason:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"state": "STATIC_BATCH_1", "reason": reason, "source_sha256": sha256_file(source)[1]}) + "\n")
        raise StaticBatchRequired(reason)
    source_digest = sha256_file(source)[1]
    original = onnx.load(source, load_external_data=False)
    if any(v.data_location == onnx.TensorProto.EXTERNAL for v in original.graph.initializer):
        raise DynamicShapeError("External tensor files require a separately verified export; no path traversal is permitted")
    candidate = surgery(original, source.name)
    try:
        evidence = validate_conversion(original, candidate, batches, overrides)
    except StaticBatchRequired as error:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"state": "STATIC_BATCH_1", "reason": str(error), "source_sha256": source_digest}) + "\n")
        raise
    if sha256_file(source)[1] != source_digest:
        raise DynamicShapeError("Source model changed during conversion")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned_path(destination.with_name(destination.name + f".{uuid.uuid4().hex}.part"))
    try:
        onnx.save(candidate, temporary)
        owned_path(destination)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            owned_path(temporary).unlink()
    receipt.write_text(json.dumps({"state": "VERIFIED_DYNAMIC", "source_sha256": source_digest,
                                   "output_sha256": sha256_file(destination)[1], **evidence}, indent=2) + "\n")
    return destination
