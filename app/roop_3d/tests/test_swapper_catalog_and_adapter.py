from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from roop_3d.core.settings import Settings
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.swapper_engine import SwapperEngineAdapter
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import Manifest, ModelRegistry
from roop_3d.server.app import create_app


SWAPPERS = {
    "inswapper_128", "inswapper_128_fp16", "hyperswap_1a_256", "hyperswap_1b_256",
    "hyperswap_1c_256", "realswap_256", "realswap_512", "simswap_256",
    "simswap_512_unofficial", "ghost_256_unet_1", "ghost_256_unet_2",
    "ghost_256_unet_3", "blendswap_256", "uniface_256",
}


def test_models_api_lists_all_engine_catalog_entries(tmp_path: Path) -> None:
    app = create_app(Settings(), repo_root=tmp_path, worker_enabled=False)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        response = client.get("/api/models")
        assert response.status_code == 200
        rows = response.json()
        by_id = {row["id"]: row for row in rows}
        assert SWAPPERS <= by_id.keys()
        assert {"ultramax", "gpen_bfr_512", "gpen_bfr_1024", "codeformer", "gfpgan_v14", "restoreformer_pp"} <= by_id.keys()
        assert {"scrfd", "retinaface", "yoloface"} <= by_id.keys()
        assert {"xseg", "bisenet"} <= by_id.keys()
        assert {"realesrgan_x2plus", "realesrgan_x4plus", "4x_ultrasharp"} <= by_id.keys()
        assert {"swapper", "enhancer", "detector", "occluder", "upscaler"} <= {row["category"] for row in rows}
        registry = client.get("/api/v1/models/registry?view=legacy").json()
        assert SWAPPERS <= {row["model_id"] for row in registry}
        for model_id in SWAPPERS:
            assert by_id[model_id]["supports_3d_hybrid"] is True
            assert isinstance(by_id[model_id]["installed"], bool)
            assert by_id[model_id]["selectable"] is True
            assert {"native_resolution", "vram_loaded", "trt_cached", "download_url"} <= by_id[model_id].keys()
        rescan = client.post("/api/models/rescan")
        assert rescan.status_code == 200 and rescan.json()["models"] >= len(SWAPPERS)


def test_local_rescan_exposes_custom_extensions_and_unblocks_local_onnx(tmp_path: Path) -> None:
    package_models = tmp_path / "roop_3d"
    package_models.mkdir(parents=True)
    shutil.copy(Path(__file__).parents[1] / "supported_models.json", package_models / "supported_models.json")
    weights = tmp_path / "models"
    (weights / "known").mkdir(parents=True)
    (weights / "custom").mkdir()
    (weights / "known/hyperswap_1a_256.onnx").write_bytes(b"local model candidate")
    (weights / "custom/architecture.pth").write_bytes(b"pytorch candidate")
    (weights / "custom/engine.trt").write_bytes(b"tensorrt candidate")
    registry = ModelRegistry(tmp_path, manifest=Manifest(evidence_scope="test fixture", models=()))
    catalog = ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry)

    result = catalog.rescan_local()
    rows = {row["id"]: row for row in catalog.public()}
    assert result["unblock_local_models"] is True
    assert rows["hyperswap_1a_256"]["installed"] is True
    assert len(result["custom_local_models"]) == 2
    assert catalog.resolve_local_model("hyperswap_1a_256").name == "hyperswap_1a_256.onnx"
    assert all(row["installed"] for row in result["custom_local_models"])


class _Node:
    def __init__(self, name: str, shape: tuple[int, ...], node_type: str = "tensor(float)"):
        self.name, self.shape, self.type = name, shape, node_type


class _DummyGpuSession:
    def __init__(self, size: int, target_name: str, identity_name: str):
        self.target_name = target_name
        self.inputs = [_Node(target_name, (1, 3, size, size)), _Node(identity_name, (1, 512))]
        self.outputs = [_Node("output", (1, 3, size, size))]

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return ["CUDAExecutionProvider"]


def test_dummy_gpu_inference_respects_native_swapper_contracts() -> None:
    catalog = ModelCatalog()
    contracts = (("inswapper_128", 128, "target", "source"),
                 ("hyperswap_1a_256", 256, "input_face", "arcface_embedding"),
                 ("realswap_512", 512, "image", "identity_embedding"),
                 ("simswap_512_unofficial", 512, "aligned_face", "source_id"))
    for model_id, size, target_name, identity_name in contracts:
        seen = {}
        session = _DummyGpuSession(size, target_name, identity_name)

        def runner(feeds):
            seen.update(feeds)
            return {"output": feeds[target_name]}

        adapter = SwapperEngineAdapter(catalog, model_id, RuntimeConfig(), session=session, runner=runner)
        crop = np.full((size, size, 3), 90, np.uint8)
        output = adapter.infer(crop, np.ones(512, np.float32))
        assert session.get_providers()[0] == "CUDAExecutionProvider"
        assert seen[target_name].shape == (1, 3, size, size)
        assert seen[identity_name].shape == (1, 512)
        assert output.shape == (size, size, 3) and output.dtype == np.uint8
        assert np.isfinite(output).all()
        assert adapter.capability()["executions"] == 1
