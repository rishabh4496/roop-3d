from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time
import pytest

from roop_3d.core.errors import DomainError
from roop_3d.models.engine_cache import EngineCache, EngineKey, ShapeRange


def key(**changes):
    values = dict(artifact_sha256="a"*64, precision="fp32", gpu_compute_capability="8.9",
                  driver="storage-test-driver", onnxruntime_version="storage-test-ort", tensorrt_version="storage-test-trt",
                  shape_profile={"x": ShapeRange(minimum=(1,512),optimum=(1,512),maximum=(1,512))},
                  build_signature="fixture-policy-and-golden")
    return EngineKey(**{**values, **changes})


def publish(cache, identity):
    with cache.transaction(identity) as (directory, existing):
        if existing is not None:
            return "hit"
        # Storage-unit payload only; this is never described as a TensorRT engine.
        (directory / "storage-fixture.engine").write_bytes(b"storage fixture, not a compiled engine")
        time.sleep(0.1)
        cache.publish_bundle(identity, directory, {"storage_test_only": True})
        return "built"


def test_atomic_bundle_publication_and_thread_single_writer(tmp_path):
    cache, identity = EngineCache(tmp_path), key()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: publish(cache, identity), range(4)))
    assert results.count("built") == 1 and results.count("hit") == 3
    assert cache.lookup_bundle(identity).receipt["storage_test_only"]
    assert not list((tmp_path / "roop_3d/cache/engines").glob("*.part"))
    assert cache.lookup_bundle(key(build_signature="other-golden")) is None
    (cache.bundle_path(identity) / "storage-fixture.engine").write_bytes(b"corrupt")
    with pytest.raises(DomainError): cache.lookup_bundle(identity)


def test_failed_build_never_publishes_and_lock_timeout_is_explicit(tmp_path):
    cache, identity = EngineCache(tmp_path), key()
    with pytest.raises(RuntimeError):
        with cache.transaction(identity) as (directory, _):
            (directory / "partial.engine").write_bytes(b"incomplete")
            raise RuntimeError("build failed")
    assert cache.lookup_bundle(identity) is None
    assert not list((tmp_path / "roop_3d/cache/engines").glob("*.part"))
    def contend():
        with pytest.raises(DomainError) as error:
            with cache.lock(identity.digest(), timeout=0.05):
                pytest.fail("A second thread entered a held engine lock")
        assert error.value.code == "ENGINE_BUILD_BUSY"
    with ThreadPoolExecutor(max_workers=1) as executor:
        with cache.lock(identity.digest()):
            executor.submit(contend).result(timeout=2)


def test_hardlinked_manifest_is_rejected(tmp_path):
    import os
    cache, identity = EngineCache(tmp_path), key()
    publish(cache, identity)
    manifest = cache.bundle_path(identity) / "manifest.json"
    os.link(manifest, tmp_path / "manifest-alias.json")
    with pytest.raises(DomainError) as error:
        cache.lookup_bundle(identity)
    assert error.value.code == "ENGINE_CACHE_INVALID"


def test_real_interprocess_lock_prevents_double_publication(tmp_path):
    identity = key()
    code = '''
import json,sys,time
from pathlib import Path
from roop_3d.models.engine_cache import EngineCache,EngineKey
c=EngineCache(Path(sys.argv[1]));k=EngineKey.model_validate_json(sys.argv[2])
with c.transaction(k) as (d,old):
 if old is None:
  (d/'storage-fixture.engine').write_bytes(b'storage fixture only')
  time.sleep(.5)
  c.publish_bundle(k,d,{'storage_test_only':True})
  print('built')
 else: print('hit')
'''
    processes = [subprocess.Popen([sys.executable, "-B", "-c", code, str(tmp_path), identity.model_dump_json()],
                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    try:
        outputs = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=20)
            assert process.returncode == 0, stderr
            outputs.append(stdout.strip())
        assert outputs.count("built") == 1 and outputs.count("hit") == 2
        assert EngineCache(tmp_path).lookup_bundle(identity) is not None
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
