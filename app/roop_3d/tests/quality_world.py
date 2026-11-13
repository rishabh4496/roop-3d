"""Offline arithmetic ONNX fixtures, not learned-model or photographic-quality evidence.

The real catalog, hash/governance checks and CPU runtime execute these graphs.
Only graph inputs and known image arithmetic are synthetic.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.local_provider import LocalWeightProvider
from roop_3d.models.registry import Locator, Manifest, ModelRegistry
from roop_3d.tests.test_stage03_registry import reviewed_artifact


def graph_catalog(root: Path, name: str, graph: onnx.ModelProto, *, role: str) -> ModelCatalog:
    root.mkdir(parents=True, exist_ok=True)
    graph.ir_version = 8
    path = root / name
    onnx.save(graph, path)

    def contract(value):
        tensor = value.type.tensor_type
        return {"name": value.name, "dtype": TensorProto.DataType.Name(tensor.elem_type),
                "shape": [dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                          for dim in tensor.shape.dim]}

    record = reviewed_artifact(
        path.read_bytes(), model_id="quality-fixture", name=name, filenames=(name,), role=role,
        role_evidence="Generated arithmetic graph; no pretrained inference claim",
        locator=Locator(scope="fixture", path=name),
        input_output_contract={"inputs": [contract(value) for value in graph.graph.input],
                               "outputs": [contract(value) for value in graph.graph.output]},
    )
    registry = ModelRegistry(root, manifest=Manifest(evidence_scope="Offline analytic quality fixture", models=(record,)))
    catalog = ModelCatalog(root=root, cache_root=root, registry=registry)
    catalog.provider = LocalWeightProvider(registry, root=root, scopes={"fixture": root})
    return catalog


def swap_catalog(root: Path) -> ModelCatalog:
    """A registered 256px swapper contract with an observable, input-dependent color inversion."""
    graph = helper.make_model(helper.make_graph(
        [helper.make_node("Neg", ["target"], ["negative"]),
         helper.make_node("ReduceMean", ["source"], ["identity_mean"], keepdims=0),
         helper.make_node("Add", ["negative", "identity_mean"], ["output"])],
        "analytic_swapper",
        [helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, 256, 256]),
         helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 512])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 256, 256])]),
        opset_imports=[helper.make_opsetid("", 13)])
    return graph_catalog(root, "hyperswap_1a_256.onnx", graph, role="identity_swapper")


def restorer_catalog(root: Path, *, nonfinite: bool = False) -> ModelCatalog:
    """FP16 RGB image + scalar DOUBLE fidelity; invert contrast and add a known detail band.

    A finite input-dependent graph exercises dynamic fidelity and actual I/O binding.
    The nonfinite variant is an intentional malformed-output fixture for fail-closed tests.
    """
    detail = np.broadcast_to(
        (0.3 * np.sin(2 * np.pi * np.arange(512) / 4)).astype(np.float32), (1, 1, 512, 512)).copy()
    nodes = [helper.make_node("Cast", ["image"], ["image_f"], to=TensorProto.FLOAT),
             helper.make_node("Cast", ["w"], ["weight"], to=TensorProto.FLOAT),
             helper.make_node("Mul", ["image_f", "weight"], ["weighted"]),
             helper.make_node("Neg", ["weighted"], ["negative"]),
             helper.make_node("Mul", ["detail", "weight"], ["weighted_detail"]),
             helper.make_node("Add", ["negative", "weighted_detail"], ["with_detail"])]
    initializers = [onnx.numpy_helper.from_array(detail, "detail")]
    result = "with_detail"
    if nonfinite:
        initializers.append(onnx.numpy_helper.from_array(np.array(0, np.float32), "zero"))
        nodes.append(helper.make_node("Div", [result, "zero"], ["invalid"]))
        result = "invalid"
    nodes.append(helper.make_node("Cast", [result], ["output"], to=TensorProto.FLOAT16))
    graph = helper.make_model(helper.make_graph(
        nodes, "analytic_fp16_restorer",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT16, [1, 3, 512, 512]),
         helper.make_tensor_value_info("w", TensorProto.DOUBLE, [])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 3, 512, 512])], initializers),
        opset_imports=[helper.make_opsetid("", 13)])
    return graph_catalog(root, "codeformer.fp16.onnx", graph, role="face_restorer")
