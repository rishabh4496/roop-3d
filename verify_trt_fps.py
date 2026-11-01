"""Benchmark synchronous ORT TRT calls after strict compilation/canary gates.

Reports model throughput including host transfers, not end-to-end video FPS.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).absolute().parent / "app"))
from roop_3d.models.dynamic_shapes import assert_finite, sample_inputs, static_batch_reason
from roop_3d.trt_session_builder import BatchProfile, TensorRTSessionBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--batch", type=int, choices=(1, 4, 8), default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--hardware-profile", choices=("6gb", "12gb"), default="6gb")
    parser.add_argument("--shapes", default="{}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        if args.iterations < 1:
            raise ValueError("Iterations must be positive")
        profile = BatchProfile((1,), 1) if static_batch_reason(args.model.name) else BatchProfile()
        if args.batch not in profile.batches:
            raise ValueError("Static model benchmark requires --batch 1")
        shapes = {k: tuple(v) for k, v in json.loads(args.shapes).items()}
        builder = TensorRTSessionBuilder(hardware_profile=args.hardware_profile,
                                        telemetry=lambda event: print(json.dumps(event), flush=True))
        session = builder.build(args.model, profile=profile, shapes=shapes)
        import onnx
        inputs = sample_inputs(onnx.load(args.model), args.batch, shapes)
        durations = []
        for _ in range(args.iterations):
            start = time.perf_counter()
            outputs = session.run(None, inputs)
            durations.append(time.perf_counter() - start)
            assert_finite(outputs)
        total = sum(durations)
        print(json.dumps({"model": args.model.name, "batch": args.batch, "iterations": args.iterations,
                          "scope": "model_inference_with_host_transfers_synthetic_inputs", "seconds": total,
                          "samples_per_second": args.iterations * args.batch / total,
                          "mean_latency_ms": 1000 * total / args.iterations, "build": builder.last_build}))
        return 0
    except Exception as error:
        print(f"TRT_BENCHMARK_FAILED: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
