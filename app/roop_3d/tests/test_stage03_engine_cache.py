from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from roop_3d.core.errors import DomainError
from roop_3d.models.engine_cache import EngineCache, EngineKey, ShapeRange


def key(**changes) -> EngineKey:
    return EngineKey.model_validate({"artifact_sha256": "a" * 64, "precision": "fp16",
        "gpu_compute_capability": "8.9", "driver": "fixture-driver", "onnxruntime_version": "fixture-ort",
        "tensorrt_version": "fixture-trt", "shape_profile": {
            "input": {"minimum": [1, 3, 128, 128], "optimum": [1, 3, 128, 128], "maximum": [4, 3, 128, 128]}},
        **changes})


@pytest.mark.parametrize("changes", [
    {"artifact_sha256": "b" * 64}, {"precision": "fp32"}, {"gpu_compute_capability": "7.5"},
    {"driver": "other-driver"}, {"onnxruntime_version": "other-ort"}, {"tensorrt_version": "other-trt"},
    {"shape_profile": {"input": {"minimum": [1, 3, 256, 256], "optimum": [1, 3, 256, 256], "maximum": [1, 3, 256, 256]}}}])
def test_every_hardware_version_and_shape_component_changes_key(changes):
    assert key().digest() != key(**changes).digest()


def test_metadata_binds_actual_compiled_bytes_and_detects_corruption(tmp_path):
    cache, identity = EngineCache(tmp_path), key()
    assert cache.lookup(identity) is None
    artifact, metadata = cache.paths(identity)
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture compiled bytes; not a real engine")
    value = cache.record(identity)
    assert value.key == identity and cache.lookup(identity) == artifact
    assert metadata.parent == tmp_path / "roop_3d/cache/engines"
    document = json.loads(metadata.read_text())
    document["key"]["driver"] = "tampered-driver"
    metadata.write_text(json.dumps(document))
    with pytest.raises(DomainError) as failure:
        cache.lookup(identity)
    assert failure.value.code == "ENGINE_CACHE_INVALID"
    cache.record(identity)
    artifact.write_bytes(b"corrupt bytes")
    with pytest.raises(DomainError):
        cache.lookup(identity)


def test_shape_profiles_are_validated_and_onnx_cache_does_not_require_trt(tmp_path):
    with pytest.raises(ValidationError):
        ShapeRange(minimum=(2,), optimum=(1,), maximum=(4,))
    with pytest.raises(ValidationError):
        ShapeRange(minimum=(1, 3), optimum=(1,), maximum=(4,))
    cache = EngineCache(tmp_path)
    identity = key(tensorrt_version=None)
    assert cache.paths(identity, "onnx")[0].suffix == ".onnx"
    with pytest.raises(ValueError):
        cache.paths(identity, "engine")
