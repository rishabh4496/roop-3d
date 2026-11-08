"""Measure whether the runtime path saturates the GPU: TensorRT FP16 + ``GpuIoBinding`` (device-resident tensors) + micro-batching.

Scope, stated so the number is not over-read: the workload is a synthetic FP16 convolution stack, because no model weights ship in
this repository. It certifies the *execution path* (provider, IOBinding, batching, worker overlap) and not the end-to-end swap
pipeline. Utilization is NVML ``utilization.gpu`` sampled at 10 Hz by the application's own ``HardwareMonitor``; the window mean
includes the ramp-up, so it understates steady state. Host<->device copies come from the application's own transfer ledger.

    python -m roop_3d.tools.gpu_saturation --batch 16 --workers 2
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import tempfile
import threading
import time
from typing import Any

CHANNELS, SIZE, LAYERS = 64, 128, 6


def available() -> tuple[bool, str]:
    """(usable, reason). Needs ONNX Runtime's TensorRT provider, its libraries, and an NVML-readable NVIDIA GPU."""
    try:
        from roop_3d.models.native_libraries import register_native_libraries
        register_native_libraries()
        import onnxruntime as ort
    except ImportError as error:
        return False, f"onnxruntime unavailable: {error}"
    if "TensorrtExecutionProvider" not in ort.get_available_providers():
        return False, "ONNX Runtime has no TensorrtExecutionProvider"
    from roop_3d.async_pipeline.hwmon import HardwareMonitor
    monitor = HardwareMonitor(interval=0.1).acquire()
    try:
        time.sleep(0.4)
        if monitor.snapshot().get("latest", {}).get("gpu_util_pct") is None:
            return False, "NVML reports no GPU utilization on this machine"
    finally:
        monitor.release()
    return True, ""


def _model(batch: int) -> bytes:
    import numpy as np
    from onnx import TensorProto, helper, numpy_helper
    rng = np.random.default_rng(0)
    nodes, inits, previous = [], [], "x"
    for index in range(LAYERS):
        weight = (rng.standard_normal((CHANNELS, CHANNELS, 3, 3)) * 0.05).astype(np.float16)
        inits.append(numpy_helper.from_array(weight, f"w{index}"))
        nodes.append(helper.make_node("Conv", [previous, f"w{index}"], [f"c{index}"], pads=[1, 1, 1, 1], kernel_shape=[3, 3]))
        nodes.append(helper.make_node("Relu", [f"c{index}"], [f"r{index}"]))
        previous = f"r{index}"
    shape = [batch, CHANNELS, SIZE, SIZE]
    graph = helper.make_graph(nodes, "saturation", [helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)],
                              [helper.make_tensor_value_info(previous, TensorProto.FLOAT16, shape)], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8
    return model.SerializeToString()


def measure(batch: int = 16, workers: int = 2, seconds: float = 8.0, cache_root: pathlib.Path | None = None) -> dict[str, Any]:
    """Run ``workers`` threads, each with its own IOBinding and a device-resident input, for ``seconds``; return what was measured."""
    from roop_3d.models.native_libraries import register_native_libraries
    register_native_libraries()                          # TensorRT / CUDA DLL directories must be registered before onnxruntime loads

    cache = pathlib.Path(tempfile.mkdtemp(dir=cache_root)) if cache_root else pathlib.Path(tempfile.mkdtemp())
    try:
        return _measure(cache, batch, workers, seconds)
    finally:
        shutil.rmtree(cache, ignore_errors=True)         # the engine cache is only useful to this run


def _measure(cache: pathlib.Path, batch: int, workers: int, seconds: float) -> dict[str, Any]:
    import numpy as np
    import onnxruntime as ort
    from roop_3d.async_pipeline.hwmon import HardwareMonitor
    from roop_3d.core.ort_session import GpuIoBinding
    from roop_3d.core.transfer_ledger import TRANSFERS

    options = {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache), "trt_max_workspace_size": 2 << 30}
    session = ort.InferenceSession(_model(batch), providers=[("TensorrtExecutionProvider", options), "CUDAExecutionProvider"])
    active = session.get_providers()[0]
    sample = np.random.default_rng(1).standard_normal((batch, CHANNELS, SIZE, SIZE)).astype(np.float16)
    binders = [GpuIoBinding(session) for _ in range(workers)]
    resident = [ort.OrtValue.ortvalue_from_numpy(sample, "cuda", 0) for _ in range(workers)]      # uploaded once, reused every call
    for binder, value in zip(binders, resident):
        for _ in range(10):
            binder.run({"x": value})
    TRANSFERS.install()
    TRANSFERS.reset()
    monitor = HardwareMonitor(interval=0.1, window=int(seconds * 10) + 50).acquire()
    try:
        time.sleep(0.5)
        counts = [0] * workers
        deadline = time.perf_counter() + seconds

        def work(index: int) -> None:
            while time.perf_counter() < deadline:
                binders[index].run({"x": resident[index]})
                counts[index] += batch

        began = time.perf_counter()
        threads = [threading.Thread(target=work, args=(index,)) for index in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        elapsed = time.perf_counter() - began
        window = monitor.snapshot().get("window", {}).get("gpu_util_pct", {})
    finally:
        monitor.release()
    totals = TRANSFERS.snapshot()["total"]
    return {"provider": active, "precision": "fp16", "batch": batch, "workers": workers, "seconds": round(elapsed, 2),
            "samples_per_second": round(sum(counts) / elapsed, 1), "gpu_util_mean_pct": window.get("mean"),
            "gpu_util_max_pct": window.get("max"), "intermediate_copies": totals["intermediate_copies"],
            "scope": "synthetic fp16 conv stack; execution path only, not the swap pipeline"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seconds", type=float, default=8.0)
    arguments = parser.parse_args()
    usable, reason = available()
    if not usable:
        print(json.dumps({"available": False, "reason": reason}))
        return 2
    print(json.dumps(measure(arguments.batch, arguments.workers, arguments.seconds)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
