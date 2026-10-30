"""Capture real Stage 06 golden tensors; qualify providers and measure p50/p95.

All inputs, activations, profiles and reports remain inside ignored local state.
No optional package/model is downloaded and no custom kernel is required.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import replace
import gc
import json
import time

import cv2
import numpy as np

from roop_3d.audit.project_paths import application_root, legacy_root
from roop_3d.core.storage import atomic_json
from roop_3d.core.hardware import probe_hardware
from roop_3d.core.settings import Settings
from roop_3d.engine.resources import evict_for_build
from roop_3d.engine.runtime import RuntimeConfig
from roop_3d.engine.volumetric.session import ReferencePipeline
from roop_3d.engine.volumetric.preprocess import transform_points
from roop_3d.faceset.faceset_reader import FaceSetReader
from roop_3d.models.accelerated_runtime import AcceleratorFactory
from roop_3d.models.golden import GoldenBank
from roop_3d.models.export_validation import Tolerance, compare_outputs


def capture() -> GoldenBank:
    reader = FaceSetReader()
    source = reader.read(legacy_root() / "app/facesets/akansha.fsz")
    target = reader.read(legacy_root() / "app/facesets/anushree.fsz")
    bank = GoldenBank(cases_per_model=2)
    pipeline = ReferencePipeline(config=RuntimeConfig(model_factory=bank.factory))
    try:
        for view in sorted(target.views, key=lambda item: abs(item.yaw))[:2]:
            image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
            pipeline.prepare_session(source, retarget="required")
            output = pipeline.infer_frame(image)
            assert len(output.replacements) == 1
            pipeline.release_session()
    finally:
        pipeline.release_session()
        if reader._detector is not None:
            reader._detector.release_for_build()
        if reader._recognizer is not None:
            reader._recognizer.release_for_build()
    assert len(bank.models) == 10, sorted(bank.models)
    assert all(len(golden.cases) == 2 for golden in bank.models.values())
    bank.save(application_root())
    gc.collect()
    print("Captured two distinct real CPU golden cases for each of 10 Stage 06 models", flush=True)
    return bank


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("6gb", "12gb"), default="6gb")
    parser.add_argument("--reuse-golden", action="store_true")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--fp32-only", action="store_true", help="Qualify FP32 providers without attempting FP16 builds")
    args = parser.parse_args()
    root, started = application_root(), time.perf_counter()
    hardware = probe_hardware(Settings(hardware_profile=args.profile, preferred_provider="CUDAExecutionProvider"))
    bank = GoldenBank.load(root) if args.reuse_golden else capture()
    eviction = evict_for_build()
    factory = AcceleratorFactory(bank, profile=args.profile, allow_fp16=not args.fp32_only)
    receipts = []
    for name in sorted(bank.models):
        receipt = factory.qualify(name, iterations=args.iterations, warmups=1)
        receipts.append(receipt)
        print(f"{name}: {receipt['provider']} {receipt['precision']}; p50={receipt['selected_benchmark']['p50_ms']:.3f} ms; p95={receipt['selected_benchmark']['p95_ms']:.3f} ms", flush=True)
    reader = FaceSetReader()
    source = reader.read(legacy_root() / "app/facesets/akansha.fsz")
    target = reader.read(legacy_root() / "app/facesets/anushree.fsz")
    view = min(target.views, key=lambda item: abs(item.yaw))
    image = cv2.imdecode(np.frombuffer(view.image_bytes, np.uint8), cv2.IMREAD_COLOR)
    pipeline = ReferencePipeline(config=factory.runtime_config())
    try:
        pipeline.prepare_session(source, retarget="required")
        output = pipeline.infer_frame(image)
        assert len(output.replacements) == 1
        face, result = output.observations[0].observation, output.replacements[0]
        before = pipeline.tracker.recognizer.embed(image, face.landmarks5)
        points = transform_points(face.landmarks5, cv2.invertAffineTransform(result.inverse)).astype(np.float32)
        after = pipeline.tracker.recognizer.embed(result.image, points)
        source_vector = np.asarray(source.canonical_embedding)
        before_cos, after_cos = float(np.dot(before, source_vector)), float(np.dot(after, source_vector))
        assert after_cos > before_cos + 0.2
        online = pipeline.capability()
        online["detector"] = pipeline.tracker.detector._session.model.capability()
        online["recognizer"] = pipeline.tracker.recognizer._session.model.capability()
        diagnostics = factory.job_diagnostics().model_dump(mode="json")
        assert len(diagnostics["models"]) == 10
    finally:
        pipeline.release_session()
        if reader._detector is not None:
            reader._detector.release_for_build()
        if reader._recognizer is not None:
            reader._recognizer.release_for_build()
    reference = ReferencePipeline()
    try:
        reference.prepare_session(source, retarget="required")
        reference_output = reference.infer_frame(image)
        assert len(reference_output.replacements) == 1
        def image_tensor(value):
            return np.ascontiguousarray(value[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255
        image_parity = compare_outputs(
            {"crop": image_tensor(reference_output.replacements[0].image)},
            {"crop": image_tensor(result.image)},
            Tolerance(0.02, 0.002, image_psnr_db=40, image_ssim=0.995))
        image_parity.require()
    finally:
        reference.release_session()
    report = {"status": "passed", "profile": args.profile, "models": receipts, "golden_cases_per_model": 2,
              "fp16_candidates_attempted": not args.fp32_only,
              "hardware": hardware.model_dump(mode="json"), "pipeline_image_parity": asdict(image_parity),
              "job_diagnostics": diagnostics,
              "online_capabilities": online, "source_cosine_before": before_cos, "source_cosine_after": after_cos,
              "eviction": eviction, "iterations": args.iterations, "elapsed_seconds": time.perf_counter() - started,
              "scope": "Fixed local golden cases on this hardware/free-memory state; no general speed claim"}
    atomic_json(root, f"roop_3d/.runtime/acceptance/STAGE07_{args.profile}.json", report)
    print(json.dumps({"status": "passed", "profile": args.profile, "models": len(receipts),
                      "source_cosine_before": before_cos, "source_cosine_after": after_cos,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
