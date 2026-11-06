"""Best-effort local Stage 2 model/provider smoke verification.

Only models whose immutable bytes are installed and admitted by the local model
catalog are loaded. Missing/unregistered weights are reported as such; they are
not represented as successful inference.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from roop_3d.models.catalog import ModelCatalog
from roop_3d.models.registry import ModelRegistry
from roop_3d.core.ort_session import SessionManager


TARGETS = {
    "detectors": ("scrfd_2.5g_kps.onnx", "scrfd_10g_bnkps.onnx", "retinaface_r50.onnx",
                  "retinaface_10g.onnx", "det_10g.onnx"),
    "swapper": ("inswapper_128.onnx", "inswapper_128_fp16.onnx"),
    "3d_landmarks": ("1k3d68.onnx", "2dfan4.onnx", "2d106det.onnx"),
    "xseg": ("xseg.onnx", "xseg_3.onnx"),
    "occluder": ("face_occluder.onnx", "occluder.onnx"),
    "gpen": ("gpen_bfr_512.onnx", "GPEN-BFR-512.onnx"),
    "ultramax": ("ultramax.onnx", "ultramax_512.onnx"),
    "other_registered_restorers": ("CodeFormerv0.1.onnx", "codeformer.fp16.onnx", "GFPGANv1.4.onnx"),
    "face_parser": ("bisenet_resnet_34.onnx",),
}


def _vram_mib() -> dict[str, int] | None:
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3, check=True)
        values = result.stdout.splitlines()[0].split(",")
        return {"used_mib": int(values[0].strip()), "total_mib": int(values[1].strip())}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        try:
            import torch
            if torch.cuda.is_available():
                return {"used_mib": round(torch.cuda.memory_allocated() / 1048576),
                        "total_mib": round(torch.cuda.get_device_properties(0).total_memory / 1048576)}
        except Exception:
            pass
    return None


def _feed(node, name: str):
    import numpy as np
    shape = []
    for axis, dimension in enumerate(node.shape):
        if isinstance(dimension, int) and dimension > 0:
            shape.append(dimension)
        elif axis == 0:
            shape.append(1)
        elif len(node.shape) == 4 and axis == 1:
            shape.append(3)
        elif len(node.shape) == 4 and axis in (2, 3):
            shape.append(512)
        else:
            shape.append(1)
    dtype = {"tensor(float16)": np.float16, "tensor(double)": np.float64,
             "tensor(int64)": np.int64, "tensor(int32)": np.int32}.get(node.type, np.float32)
    value = np.zeros(tuple(shape), dtype=dtype)
    if name.casefold() in ("w", "fidelity_weight"):
        value.fill(.5)
    return value


def _targets(registry: ModelRegistry):
    records = list(registry.records())
    result = []
    seen = set()
    for group, names in TARGETS.items():
        matches = [record for record in records if record.name.casefold() in {name.casefold() for name in names}]
        for record in matches:
            if record.model_id not in seen:
                result.append((group, record))
                seen.add(record.model_id)
        if not matches:
            result.append((group, None))
    # Include registered landmarks and other dense mesh model names even when an
    # upstream filename differs from the common InsightFace alias.
    for record in records:
        lowered = record.name.casefold()
        if record.model_id not in seen and (record.role == "dense_mesh" or
                any(token in lowered for token in ("1k3d68", "2dfan4", "landmark_106"))):
            result.append(("3d_landmarks", record))
            seen.add(record.model_id)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-only", action="store_true",
                        help="Skip TensorRT engine compilation and verify CUDA/CPU fallback")
    args = parser.parse_args()
    catalog = ModelCatalog()
    manager = SessionManager()
    registered_names = {record.name.casefold(): record.model_id for record in catalog.registry.records()}
    report = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "requested": [n for v in TARGETS.values() for n in v],
              "target_alias_coverage": {name: {"state": "REGISTERED" if name.casefold() in registered_names else "UNREGISTERED",
                                                  "model_id": registered_names.get(name.casefold())}
                                        for names in TARGETS.values() for name in names},
              "models": [], "vram_before_mib": _vram_mib()}
    failures = 0
    output = ROOT / "logs" / "stage2_gpu_model_verification.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    provider_order = ("CUDAExecutionProvider", "CPUExecutionProvider") if args.cuda_only else (
        "TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
    report["provider_order"] = list(provider_order)
    for group, record in _targets(catalog.registry):
        if record is None:
            report["models"].append({"group": group, "state": "UNREGISTERED", "provider": None})
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            continue
        item = {"group": group, "model_id": record.model_id, "name": record.name}
        try:
            path = catalog.resolve_for_inference(record.model_id)
        except Exception as error:
            item.update(state="UNAVAILABLE", detail=str(error))
            report["models"].append(item)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            continue
        try:
            info = manager.create(path, model_name=record.name, providers=provider_order)
            feeds = {node.name: _feed(node, node.name) for node in info.session.get_inputs()}
            info.session.run(None, feeds)  # warmup
            samples = []
            for _ in range(3):
                start = time.perf_counter()
                info.session.run(None, feeds)
                samples.append((time.perf_counter() - start) * 1000)
            latency = statistics.median(samples)
            item.update(state="EXECUTED", provider=info.provider, fallback_errors=list(info.fallback_errors),
                        input_names=list(feeds), output_names=[node.name for node in info.session.get_outputs()],
                        warmup_runs=1, measured_runs=len(samples), median_latency_ms=round(latency, 3),
                        fps=round(1000 / latency, 2) if latency else None)
        except Exception as error:
            failures += 1
            item.update(state="FAILED", detail=f"{type(error).__name__}: {error}")
        report["models"].append(item)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["vram_after_mib"] = _vram_mib()
    report["summary"] = {state: sum(model["state"] == state for model in report["models"])
                          for state in ("EXECUTED", "UNAVAILABLE", "UNREGISTERED", "FAILED")}
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report: {output}")
    # Missing weights are environment state, not script errors. Actual graph or
    # inference failures are nonzero and remain visible in the machine report.
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
