"""Weights the user placed in app/models are auto-trusted, never blocked, and reported by the read-back."""
import hashlib
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from roop_3d.engine import active_models
from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.local_trust import LocalTrust, MIN_TRUST_BYTES, family_key
from roop_3d.models.registry import Manifest, ModelRegistry
from roop_3d.tests.test_stage03_registry import artifact

BIG = MIN_TRUST_BYTES + 4096


def put(root: Path, name: str, size: int = BIG, seed: int = 1) -> Path:
    path = root / "models" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.random.default_rng(seed).bytes(size))
    return path


def catalog_for(tmp_path, *records, **kwargs) -> ModelCatalog:
    registry = ModelRegistry(manifest=Manifest(evidence_scope="Fixture", models=tuple(records)))
    return ModelCatalog(root=tmp_path, cache_root=tmp_path, registry=registry, **kwargs)


@pytest.mark.parametrize("file_name,model_id,registered", [
    ("GPEN-BFR-512.onnx", "gpen_bfr_512", "gpen_bfr_512.onnx"),
    ("gpen_bfr_512.fp16.onnx", "gpen_bfr_512", "gpen_bfr_512.onnx"),
    ("CodeFormer.fp16.onnx", "codeformer", "CodeFormerv0.1.onnx"),
    ("GFPGANv1.4.onnx", "gfpgan_v14", "GFPGANv1.4.onnx"),
    ("HyperSwap_1A_256.onnx", "hyperswap_1a_256", "hyperswap_1a_256.onnx"),
    ("inswapper_128.onnx", "inswapper_128", "inswapper_128.onnx"),
    ("SCRFD_10G_bnkps.onnx", "scrfd_10g", "scrfd_10g_bnkps.onnx"),
])
def test_physical_file_with_other_bytes_is_verified_approved_and_unblocked(tmp_path, caplog, file_name, model_id, registered):
    record = artifact(b"the pinned bytes", model_id=model_id, name=registered, filenames=[registered], role="face_restorer")
    catalog = catalog_for(tmp_path, record)
    path = put(tmp_path, file_name)
    with caplog.at_level(logging.INFO, logger="roop_3d.models.local_trust"):
        resolved = catalog.resolve_for_inference(model_id)
    assert resolved == path.resolve()
    assert f"Auto-trusted local weight file: {file_name}" in caplog.text
    # The runtime registered the file's own digest instead of comparing against the pinned one.
    assert catalog.provider.trusted_identity(resolved) == (hashlib.sha256(path.read_bytes()).hexdigest(), BIG)
    # The supported-model catalog may fold a registry record into its canonical row (scrfd_10g -> scrfd).
    row = next(item for item in catalog.public() if model_id in (item["id"], item["model_id"]) or item["id"] == "scrfd")
    assert (row["installed"], row["state"], row["verified"], row["governance_approved"], row["blocked"]) == (
        True, "VERIFIED_LOCAL", True, True, False)
    assert catalog.lifecycle(model_id)["lifecycle_state"] == "READY"
    with catalog.provider.open(model_id) as stream:
        assert len(stream.read()) == BIG


def test_files_under_one_megabyte_and_trt_cache_are_not_trusted(tmp_path):
    record = artifact(b"pinned", model_id="gpen_bfr_512", name="gpen_bfr_512.onnx", filenames=["gpen_bfr_512.onnx"])
    catalog = catalog_for(tmp_path, record)
    put(tmp_path, "gpen_bfr_512.onnx", size=MIN_TRUST_BYTES - 1)
    put(tmp_path, ".trt_cache/gpen_bfr_512.onnx")
    row = next(item for item in catalog.public() if item["id"] == "gpen_bfr_512")
    # (a small file is still *listed* as present by the visibility scan, but is neither verified nor trusted)
    assert row["state"] != "VERIFIED_LOCAL" and row["verified"] is False and row["local_trust"] is False
    with pytest.raises(Exception):
        catalog.resolve_for_inference("gpen_bfr_512")


def test_unknown_custom_file_is_listed_as_verified(tmp_path):
    catalog = catalog_for(tmp_path)
    put(tmp_path, "my_private_swapper.onnx")
    row = next(item for item in catalog.public() if item["name"] == "my_private_swapper")
    assert row["state"] == "VERIFIED_LOCAL" and row["verified"] and row["governance_approved"] and not row["blocked"]


def test_exact_pinned_file_still_resolves_without_trust(tmp_path):
    data = b"x" * BIG
    record = artifact(data, model_id="inswapper_128", name="inswapper_128.onnx", filenames=["inswapper_128.onnx"])
    catalog = catalog_for(tmp_path, record, search_paths=lambda: ("roop_3d/.runtime/weights",))
    path = tmp_path / "roop_3d/.runtime/weights/inswapper_128.onnx"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    assert catalog.provider.resolve("inswapper_128") == path
    assert catalog.provider.trusted_identity(path) is None


def test_locate_prefers_exact_name_then_onnx_over_other_formats(tmp_path):
    trust = LocalTrust(tmp_path / "models")
    put(tmp_path, "gpen_bfr_512.pth", seed=2)
    put(tmp_path, "gpen_bfr_512.fp16.onnx", seed=3)
    put(tmp_path, "gpen_bfr_512.onnx", seed=4)
    assert trust.locate(["gpen_bfr_512.onnx"]).name == "gpen_bfr_512.onnx"
    (tmp_path / "models/gpen_bfr_512.onnx").unlink()
    trust.invalidate()
    assert trust.locate(["GPEN-BFR-512.onnx"]).name == "gpen_bfr_512.fp16.onnx"
    assert family_key("CodeFormerv0.1") == family_key("codeformer.fp16") == "codeformer"
    assert family_key("hyperswap_1a_256") != family_key("hyperswap_1b_256")


def test_changed_local_file_is_rehashed(tmp_path):
    trust = LocalTrust(tmp_path / "models")
    path = put(tmp_path, "scrfd_10g.onnx", seed=5)
    first = trust.trust(path)
    path.write_bytes(np.random.default_rng(6).bytes(BIG + 10))
    assert trust.lookup(path) is None
    assert trust.trust(path).sha256 != first.sha256


def test_active_model_registry_logs_a_switch_and_ignores_stale_release(caplog):
    active_models.reset()
    with caplog.at_level(logging.INFO, logger="roop_3d.engine.active_models"):
        old = active_models.activate("swapper", "inswapper_128", "/m/inswapper_128.onnx", "CUDAExecutionProvider", "fp16")
        same = active_models.activate("swapper", "inswapper_128", "/m/inswapper_128.onnx", "CUDAExecutionProvider", "fp16")
        new = active_models.activate("swapper", "hyperswap_1c_256", "/m/hyperswap_1c_256.onnx", "CUDAExecutionProvider", "fp16")
    assert same is old and new.switches == 1
    assert "Switched swapper model -> hyperswap_1c_256 (/m/hyperswap_1c_256.onnx)" in caplog.text
    active_models.release("swapper", old.key)           # the old session releasing late must not erase the new one
    assert active_models.snapshot()["swapper"]["filename"] == "hyperswap_1c_256.onnx"
    active_models.release("swapper", new.key)
    assert active_models.snapshot()["swapper"] is None


def test_detector_rebuilds_session_when_the_file_changes_under_the_same_model_id():
    from roop_3d.media.detector import Detector
    active_models.reset()
    path = ["/models/a/scrfd_10g.onnx"]
    built = []

    class Catalog:
        registry = SimpleNamespace(get=lambda model_id: model_id)
        def detector(self, model_id="scrfd"):
            return {"model_id": model_id, "sha256": "same", "name": "scrfd_10g.onnx"}
        def _provider_path(self, record):
            return Path(path[0])

    class Session:
        def get_inputs(self):
            return [SimpleNamespace(name="input", shape=(1, 3, 640, 640), type="tensor(float)")]
        def run(self, _names, _feeds):
            return ([np.zeros((640 // s) ** 2 * 2, np.float32) for s in (8, 16, 32)] +
                    [np.zeros(((640 // s) ** 2 * 2, 4), np.float32) for s in (8, 16, 32)] +
                    [np.zeros(((640 // s) ** 2 * 2, 10), np.float32) for s in (8, 16, 32)])

    detector = Detector(Catalog(), lambda record: built.append(1) or Session(), model_id="scrfd",
                        provider="CUDAExecutionProvider", precision="fp16")
    frame = np.zeros((640, 640, 3), np.uint8)
    detector.detect(frame, .5)
    detector.detect(frame, .5)
    assert len(built) == 1
    assert active_models.snapshot()["detector"]["filename"] == "scrfd_10g.onnx"
    path[0] = "/models/b/scrfd_10g_fp16.onnx"
    detector.detect(frame, .5)
    assert len(built) == 2
    assert active_models.snapshot()["detector"]["model_path"].endswith("scrfd_10g_fp16.onnx")


def _identity_onnx(path: Path, dtype, size_pad: int) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    elem = {np.float16: TensorProto.FLOAT16, np.float32: TensorProto.FLOAT}[dtype]
    pad = numpy_helper.from_array(np.zeros(size_pad // np.dtype(dtype).itemsize, dtype), "unused_padding")
    graph = helper.make_graph([helper.make_node("Identity", ["input"], ["output"])], "fixture",
        [helper.make_tensor_value_info("input", elem, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", elem, [1, 3, 8, 8])], initializer=[pad])
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)


def test_fp16_restorer_with_a_foreign_digest_loads_and_runs(tmp_path):
    from roop_3d.engine.runtime import OrtModel, RuntimeConfig
    active_models.reset()
    contract = {"state": "VERIFIED", "inputs": [{"name": "input", "shape": [1, 3, 512, 512], "dtype": "FLOAT"}],
                "outputs": [{"name": "output", "shape": [1, 3, 512, 512], "dtype": "FLOAT"}]}
    record = artifact(b"pinned fp32 bytes", model_id="gpen_bfr_512", name="gpen_bfr_512.onnx",
                      filenames=["gpen_bfr_512.onnx"], role="face_restorer", input_output_contract=contract)
    catalog = catalog_for(tmp_path, record)
    _identity_onnx(tmp_path / "models" / "GPEN-BFR-512-fp16.onnx", np.float16, MIN_TRUST_BYTES + 1024)
    model = OrtModel(catalog, "gpen_bfr_512.onnx", RuntimeConfig())
    assert model.local_trust
    # The adapters' feed contract now describes the file that really runs.
    assert model.record["graph"]["inputs"][0]["dtype"] == "FLOAT16"
    assert model.record["sha256"] == catalog.provider.trusted_identity(model.path)[0]
    out = model.run({"input": np.full((1, 3, 8, 8), .25, np.float32)})["output"]
    assert out.dtype == np.float32 and np.allclose(out, .25)
    shown = active_models.snapshot()["enhancer"]
    assert shown["filename"] == "GPEN-BFR-512-fp16.onnx" and shown["model_id"] == "gpen_bfr_512"
    model.release()
    assert active_models.snapshot()["enhancer"] is None
