"""Swap-engine availability: provider fallback, static NHWC occluder graphs, honest readiness rows, health + readiness routes."""
from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from roop_3d.core import ort_session
from roop_3d.core.ort_session import CPU, CUDA, TRT, ProviderInitializationError, SessionManager
from roop_3d.core.settings import Settings
from roop_3d.model_registry import UNIFIED_MANIFEST_MAP, UnifiedModelRegistry
from roop_3d.models.local_trust import MIN_TRUST_BYTES
from roop_3d.server.app import create_app

BIG = MIN_TRUST_BYTES + 4096


def put(root: Path, name: str, size: int = BIG, seed: int = 1) -> Path:
    path = root / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.random.default_rng(seed).bytes(size))
    return path


def tiny_model(path: Path, shape, *, dtype=None) -> Path:
    import onnx
    from onnx import TensorProto, helper
    dtype = dtype or TensorProto.FLOAT
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph([node], "g", [helper.make_tensor_value_info("x", dtype, shape)],
                              [helper.make_tensor_value_info("y", dtype, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    onnx.save(model, str(path))
    return path


# -- provider fallback --------------------------------------------------------------------------------------------------
@pytest.fixture
def cpu_only_runtime(monkeypatch):
    """An ONNX Runtime that advertises no GPU provider: what a missing TensorRT/CUDA EP or a failed build looks like."""
    import onnxruntime as ort
    monkeypatch.setattr(ort, "get_available_providers", lambda: [CPU])


def test_gpu_only_request_without_a_gpu_provider_is_an_error_by_default(tmp_path, cpu_only_runtime):
    model = tiny_model(tmp_path / "m.onnx", [1, 3, 8, 8])
    with pytest.raises(ProviderInitializationError) as error:
        SessionManager(cache_path=tmp_path / "trt").create(model, model_name="m.onnx", providers=(TRT, CUDA))
    assert len(error.value.attempts) == 2 and all("unavailable" in item for item in error.value.attempts)


def test_cpu_last_resort_serves_the_engine_and_records_the_provider_actually_chosen(tmp_path, cpu_only_runtime):
    model = tiny_model(tmp_path / "swapper_probe.onnx", [1, 3, 8, 8])
    info = SessionManager(cache_path=tmp_path / "trt").create(model, model_name="swapper_probe.onnx",
                                                              providers=(TRT, CUDA), cpu_last_resort=True)
    assert info.provider == CPU
    assert info.session.get_providers()[0] == CPU
    assert info.requested_providers == (TRT, CUDA, CPU)
    assert [item.split(":")[0] for item in info.fallback_errors] == [TRT, CUDA]
    selection = ort_session.session_selection("swapper_probe.onnx")
    assert selection["provider"] == CPU and selection["fell_back"] is True
    assert selection["requested_providers"] == [TRT, CUDA, CPU]
    # The dtype the session expects is read from the session of the provider that was chosen.
    assert selection["inputs"] == [{"name": "x", "dtype": "float32"}]
    assert ort_session.session_selection("swapper_probe")["provider"] == CPU


def test_fp16_io_graph_reports_float16_inputs_for_the_chosen_provider(tmp_path, cpu_only_runtime):
    from onnx import TensorProto
    model = tiny_model(tmp_path / "half.onnx", [1, 4], dtype=TensorProto.FLOAT16)
    SessionManager(cache_path=tmp_path / "trt").create(model, model_name="half.onnx", providers=(CUDA,), cpu_last_resort=True)
    assert ort_session.session_selection("half.onnx")["inputs"] == [{"name": "x", "dtype": "float16"}]


# -- channels-last occlusion nets ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["face_occluder.onnx", "xseg.onnx", "occluder.onnx"])
def test_channels_last_occlusion_graph_gets_a_static_batch_one_input(tmp_path, name):
    import onnx
    source = tiny_model(tmp_path / name, ["batch", 256, 256, 3])
    static, _ = ort_session._static_model_path(source, tmp_path / "cache", name)
    assert static != source
    dims = onnx.load(str(static)).graph.input[0].type.tensor_type.shape.dim
    assert [dim.dim_value for dim in dims] == [1, 256, 256, 3]
    assert ort_session._static_graph_inputs(onnx.load(str(static)))


# -- unified model readiness ---------------------------------------------------------------------------------------------
def test_unified_registry_reads_app_models_and_trusts_other_exports_but_not_corrupt_files(tmp_path):
    entry = UNIFIED_MANIFEST_MAP["hyperswap_1a_256"]
    registry = UnifiedModelRegistry(tmp_path)
    assert registry.inspect_model_readiness(entry)["readiness_state"] == "MISSING"
    assert "not found in app/models" in registry.inspect_model_readiness(entry)["reason"]

    put(tmp_path, entry.name)  # a different export of the same network, placed on purpose
    row = registry.inspect_model_readiness(entry)
    assert row["readiness_state"] == "DOWNLOADED (NEEDS TRT)"
    assert (row["trust_basis"], row["hash_valid"]) == ("LOCAL_USE_TRUST", False)
    assert row["model_path"].replace("\\", "/").endswith("models/hyperswap_1a_256.onnx")

    small = UNIFIED_MANIFEST_MAP["ghost_1_256"]
    (tmp_path / "models" / small.name).write_bytes(b"truncated download")
    assert registry.inspect_model_readiness(small)["readiness_state"] == "CORRUPT HASH"


def test_unified_registry_pinned_match_is_hash_valid_and_roop_ultimate_is_read_only_and_pin_only(tmp_path, monkeypatch):
    payload = np.random.default_rng(7).bytes(BIG)
    entry = dataclasses.replace(UNIFIED_MANIFEST_MAP["ghost_1_256"], expected_sha256=hashlib.sha256(payload).hexdigest(),
                                expected_size_bytes=len(payload))
    legacy = tmp_path / "ultimate" / ".cache" / "models"
    legacy.mkdir(parents=True)
    registry = UnifiedModelRegistry(tmp_path / "repo")
    monkeypatch.setattr(registry, "get_legacy_search_paths", lambda: [legacy])

    (legacy / entry.name).write_bytes(np.random.default_rng(8).bytes(BIG))  # same name, other bytes
    assert registry.inspect_model_readiness(entry)["readiness_state"] == "MISSING"  # never trusted by placement

    (legacy / entry.name).write_bytes(payload)  # same bytes as the pin
    row = registry.inspect_model_readiness(entry)
    assert row["hash_valid"] is True and row["trust_basis"] == "REGISTRY_PINNED_SHA256"
    assert Path(row["model_path"]).parent == legacy and (legacy / entry.name).read_bytes() == payload


# -- readiness rows + routes ---------------------------------------------------------------------------------------------
class FakeCatalog:
    def __init__(self, rows):
        self.rows = rows

    def swapper_readiness(self):
        return self.rows


def test_readiness_rows_mark_installed_engines_ready_and_missing_ones_unavailable_with_the_reason():
    from roop_3d.models import engine_readiness
    rows = engine_readiness.swap_engine_report(FakeCatalog([
        {"model_id": "hyperswap_1a_256", "architecture": "hyperswap", "native_resolution": 256, "selectable": True,
         "policy": "LOCAL_USE_TRUST", "readiness": "READY_REGISTRY_VERIFIED", "reason": None, "blocked": False,
         "artifact": {"path": "C:/w/hyperswap_1a_256.onnx", "filename": "hyperswap_1a_256.onnx", "size_bytes": 1,
                      "sha256": "a" * 64, "trust_basis": "LOCAL_DIGEST_UNREVIEWED", "local_use_policy": True}},
        {"model_id": "simswap_256", "architecture": "simswap", "native_resolution": 256, "selectable": False, "policy": None,
         "readiness": "NOT_INSTALLED", "reason": "no weights", "blocked": False, "artifact": None}]))
    assert [(row["model_id"], row["status"]) for row in rows] == [("hyperswap_1a_256", "READY"), ("simswap_256", "UNAVAILABLE")]
    assert rows[0]["weight"]["local_use_policy"] is True and rows[1]["weight"] is None
    assert rows[1]["reason"] == "no weights" and rows[1]["session"] is None
    summary = engine_readiness.summarize(rows)
    assert summary["ready"] == ["hyperswap_1a_256"] and summary["unavailable"][0]["model_id"] == "simswap_256"


def test_routes_list_installed_swappers_as_ready_and_missing_ones_honestly_unavailable(tmp_path):
    put(tmp_path, "hyperswap_1a_256.onnx")
    put(tmp_path, "inswapper_128_fp16.onnx", seed=2)
    app = create_app(Settings(hardware_profile="cpu", preferred_provider="CPUExecutionProvider"), repo_root=tmp_path)
    with TestClient(app, base_url="http://127.0.0.1:8030") as client:
        body = client.get("/api/v1/engines/readiness").json()
        engines = {row["model_id"]: row for row in body["engines"]}
        assert engines["hyperswap_1a_256"]["status"] == "READY" and engines["inswapper_128_fp16"]["status"] == "READY"
        assert engines["hyperswap_1a_256"]["weight"]["path"].replace("\\", "/").endswith("models/hyperswap_1a_256.onnx")
        for missing in ("simswap_256", "ghost_256_unet_1", "blendswap_256"):
            assert engines[missing]["status"] == "UNAVAILABLE"
            assert engines[missing]["readiness"] == "NOT_INSTALLED"
            assert "no installed local ONNX weights" in engines[missing]["reason"]
        # Registry-pinned artifacts that live in the read-only roop-ultimate cache are READY too, so only a lower bound holds.
        assert {"hyperswap_1a_256", "inswapper_128_fp16"} <= set(body["summary"]["ready"])
        assert not {"simswap_256", "ghost_256_unet_1", "blendswap_256"} & set(body["summary"]["ready"])

        health = client.get("/api/v1/health")
        assert health.status_code == 200
        deadline = __import__("time").monotonic() + 60
        while health.json()["swap_engines"]["state"] == "SCANNING" and __import__("time").monotonic() < deadline:
            __import__("time").sleep(0.2)
            health = client.get("/api/v1/health")
        swap = health.json()["swap_engines"]
        assert {"hyperswap_1a_256", "inswapper_128_fp16"} <= set(swap["ready"])
        assert {item["model_id"] for item in swap["unavailable"]} >= {"simswap_256", "ghost_256_unet_1"}


# -- CUDA FP16 graph conversion -------------------------------------------------------------------------------------------
def test_fp16_conversion_output_is_topologically_sorted_so_the_checker_accepts_it(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    from onnxruntime.transformers.float16 import convert_float_to_float16
    weight = helper.make_tensor("w", TensorProto.FLOAT, [4, 4], np.eye(4, dtype=np.float32).reshape(-1).tolist())
    graph = helper.make_graph([helper.make_node("MatMul", ["x", "w"], ["y"])], "g",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])], initializer=[weight])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    converted = convert_float_to_float16(model, keep_io_types=True)
    with pytest.raises(onnx.checker.ValidationError):  # the converter's own output is what used to break every CUDA FP16 session
        onnx.checker.check_model(converted)
    ort_session._topologically_ordered(converted)
    onnx.checker.check_model(converted)
    assert [node.op_type for node in converted.graph.node][0] == "Cast"


def test_fp32_locked_swapper_is_never_rewritten_to_mixed_precision(tmp_path):
    source = tiny_model(tmp_path / "inswapper_128_fp16.onnx", [1, 3, 8, 8])
    path, converted = ort_session._cuda_fp16_model_path(source, tmp_path / "cache")
    assert path == source.resolve() and converted is None
    assert not (tmp_path / "cache").exists()
