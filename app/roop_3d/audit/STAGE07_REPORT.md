# Stage 07 - ONNX validation, safe TensorRT builds and benchmarks

Stage 07 software gates passed. All ten installed Stage 06 models were exercised on two fixed CPU golden cases. Final 6 GB and 12 GB policy replays selected nine TensorRT FP32 models and the original CPU warp/SPADE model. The preserved reference path remains the default; acceleration is configured explicitly before preparing inference sessions.

The campaigns used a physical 12 GB RTX 4070 (SM 8.9), not a physical 6 GB laptop GPU. They demonstrate the two budget policies on the observed free memory. Physical 6 GB qualification, production tail latency and dedicated peak VRAM remain UNKNOWN.

## Verification

- Full suite: **282 passed, 18 subtests passed, 0 failed, 0 skipped**; exit 0. One existing FastAPI/HTTPX deprecation warning.
- All 29 Stage 07 tests are included: numerical failures, cache identity changes, actual ORT execution/eviction, unmanaged session refusal, thread/process locks, timeout, failed-build cleanup, provider fallback, output reordering, persisted job diagnostics and p50/p95.
- Actual missing-TensorRT fallback was exercised before SDK installation. After installing the pinned optional SDK, real engines were compiled, parity checked, reloaded and profiled. Generated ONNX graphs and storage payloads in unit tests are not the real-model acceptance evidence.
- Final audit: exit 0, 68,391 comprehensive immutable files and the overlapping 2,646 historical files; no violations. Baseline manifests retain their original SHA-256 values. Runtime activity is reported separately under the existing policy.

Run from the application app directory:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests -q
roop_3d/.venv/Scripts/python.exe -B roop_3d/audit/verify_isolation.py --report
```

## Numerical and precision contract

Every newly activated optimized session replays the fixed golden tensors. Output names, shapes, dtype and finite values must match; max/mean absolute error, embedding cosine >= 0.995, PSNR/SSIM and image borders use the versioned per-model policy. Named outputs preserve semantic identity when EPContext reorders them; the detector facade preserves the source positional order.

Landmark/detector regression, motion, stitching and warp modules remain FP32. Actual FP16 builds of the swapper, appearance extractor and recognizer failed their original policies. No tolerance was relaxed and no FP16 model was promoted. The measured failures are retained in STAGE07_VALIDATION.json and the ignored local candidate report.

| FP16 candidate | Worst max absolute error | Worst mean absolute error | Failed gates |
|---|---:|---:|---|
| inswapper_128.onnx | 0.567816 | 0.0542187 | border_absolute, image_psnr, image_ssim, max_absolute, mean_absolute |
| liveportrait_appearance_feature_extractor.onnx | 1.30947 | 0.0243505 | max_absolute, mean_absolute |
| w600k_r50.onnx | 0.0177106 | 0.00461308 | mean_absolute |

## Build, cache and fallback

An inference/build fence releases all registered owned sessions, collects garbage and empties an already-loaded PyTorch CUDA allocator before compilation. Compilation refuses unmanaged ORT aliases or live Torch allocations. Torch was absent here, so its allocator branch is implemented but not claimed executed. Observed free VRAM is measured after eviction; workspace is capped by free memory minus reserve, the selected profile ceiling and the requested limit.

Cache identity covers source SHA-256, GPU SM, driver, ORT/TensorRT versions, precision, exact shape profile, workspace and the golden/policy digests. Thread and OS file locks serialize builds. Temporary directories and hashed manifests are published atomically. Online inference loads a verified embedded context and cannot silently compile an original graph. Driver caches and native temporary files are kept inside this checkout.

TensorRT TF32 settings remain consistent at build and execution. The v2 build signature invalidates earlier experimental contexts. Unsupported compilation or numerical/contract failures fall back through CUDA to CPU with the native reason and actual profiled provider recorded. The original 5-D GridSample warp remains CPU because this ORT CUDA path supports only 4-D input; no custom kernel is required.

The typed job diagnostics hook persists provider, precision, execution counts, golden identity and fallback history for active jobs, exposed through the existing job GET response. This stage adds no queue rendering worker.

## Measured profiles

Each table reports milliseconds from one warmup and five synchronous calls on the first golden case. Both golden cases must pass parity. Times include the host contract checks and NumPy output transfer; p95 uses linear interpolation over five samples. The machine also runs independent workloads. These are fixed-case observations, not a general speed, real-time FPS or production tail claim. Small stitching models can be slower on GPU.

### 6gb policy

Observed free VRAM: 4147 MiB; effective budget: 3379 MiB. Final crop PSNR 67.143 dB, SSIM 0.999897099. Source cosine 0.076508 -> 0.887301. Online providers: 1 CPUExecutionProvider, 9 TensorrtExecutionProvider.

| Model | Selected provider | CPU p50 / p95 ms | Selected p50 / p95 ms |
|---|---|---:|---:|
| 2d106det.onnx | TRT FP32 | 2.672 / 2.901 | 0.633 / 1.027 |
| inswapper_128.onnx | TRT FP32 | 2708.800 / 2751.381 | 32.029 / 41.511 |
| liveportrait_appearance_feature_extractor.onnx | TRT FP32 | 625.819 / 697.913 | 16.840 / 17.109 |
| liveportrait_landmark.onnx | TRT FP32 | 104.524 / 119.278 | 6.342 / 7.867 |
| liveportrait_motion_extractor.onnx | TRT FP32 | 143.604 / 168.056 | 9.409 / 10.418 |
| liveportrait_stitching.onnx | TRT FP32 | 0.068 / 0.127 | 0.598 / 0.614 |
| liveportrait_stitching_eye.onnx | TRT FP32 | 0.073 / 0.082 | 0.631 / 0.753 |
| liveportrait_warping_spade.onnx | CPU FP32 | 15352.108 / 19301.850 | 11823.669 / 18962.033 |
| scrfd_10g_bnkps.onnx | TRT FP32 | 231.498 / 233.768 | 8.311 / 9.993 |
| w600k_r50.onnx | TRT FP32 | 375.624 / 502.532 | 5.353 / 5.915 |

### 12gb policy

Observed free VRAM: 5643 MiB; effective budget: 4875 MiB. Final crop PSNR 67.143 dB, SSIM 0.999897099. Source cosine 0.076508 -> 0.887301. Online providers: 1 CPUExecutionProvider, 9 TensorrtExecutionProvider.

| Model | Selected provider | CPU p50 / p95 ms | Selected p50 / p95 ms |
|---|---|---:|---:|
| 2d106det.onnx | TRT FP32 | 2.412 / 3.182 | 0.717 / 2.996 |
| inswapper_128.onnx | TRT FP32 | 1151.884 / 1171.867 | 29.570 / 32.339 |
| liveportrait_appearance_feature_extractor.onnx | TRT FP32 | 294.440 / 302.545 | 9.455 / 12.099 |
| liveportrait_landmark.onnx | TRT FP32 | 43.613 / 46.968 | 2.643 / 3.002 |
| liveportrait_motion_extractor.onnx | TRT FP32 | 72.061 / 74.296 | 5.532 / 6.912 |
| liveportrait_stitching.onnx | TRT FP32 | 0.028 / 0.051 | 0.196 / 0.323 |
| liveportrait_stitching_eye.onnx | TRT FP32 | 0.031 / 0.038 | 0.094 / 0.330 |
| liveportrait_warping_spade.onnx | CPU FP32 | 8961.174 / 9070.765 | 8649.324 / 10192.368 |
| scrfd_10g_bnkps.onnx | TRT FP32 | 95.324 / 96.664 | 4.067 / 6.444 |
| w600k_r50.onnx | TRT FP32 | 100.458 / 148.692 | 4.323 / 6.571 |

## Reproduction and scope

Core dependencies and launchers are unchanged. The optional SDK dependencies are pinned with hashes in requirements-trt.lock, constrained to the existing core lock. All models load directly from the verified read-only registry; no model weights were downloaded or copied. Model provenance, license and commercial permissions remain separate evidence states.

Golden arrays, source media, embeddings, profiler traces and compiled engines remain ignored local state. Only scalar metrics, artifact hashes and contracts are in this report. Media and biometric data were never sent to a remote provider.

```powershell
roop_3d/.venv/Scripts/python.exe -B -u -m roop_3d.tests.manual_stage07_acceptance --profile 12gb --reuse-golden
roop_3d/.venv/Scripts/python.exe -B -u -m roop_3d.tests.manual_stage07_acceptance --profile 6gb --reuse-golden --fp32-only
roop_3d/.venv/Scripts/python.exe -B -u -m roop_3d.tests.manual_stage07_acceptance --profile 12gb --reuse-golden --fp32-only
```

The first command evaluates FP16 candidates; the final commands explicitly retain FP32. Remove --reuse-golden to capture new real CPU fixtures from the locally installed acceptance FaceSets. The general Python API can capture user-specific fixtures with GoldenBank.factory; biometric tensors stay in private runtime state.

```python
factory = AcceleratorFactory(GoldenBank.load(application_root()), profile="6gb", allow_fp16=False)
factory.qualify_all(iterations=5, warmups=1)
session = ReferencePipeline(config=factory.runtime_config())
try:
    session.prepare_session(source_faceset, retarget="auto")
    result = session.infer_frame(upright_bgr_frame)
    factory.record_job_diagnostics(job_store, active_job_id)
finally:
    session.release_session()
```

Qualification is offline and can evict existing owned sessions; prepare the inference session afterwards. Exact commands, exit codes, corrected failed runs, cache file hashes, per-case metrics, hardware snapshots and next gate status are in STAGE07_VALIDATION.json. The software gate is ready for the next numbered specification; physical 6 GB hardware and live Torch allocator qualification remain unverified.

## Files changed

- app/roop_3d/audit/STAGE07_REPORT.md
- app/roop_3d/audit/STAGE07_VALIDATION.json
- app/roop_3d/engine/detection/face_tracker.py
- app/roop_3d/engine/resources.py
- app/roop_3d/engine/runtime.py
- app/roop_3d/engine/volumetric/session.py
- app/roop_3d/faceset/recognizer.py
- app/roop_3d/jobs/schema.py
- app/roop_3d/jobs/store.py
- app/roop_3d/media/detector.py
- app/roop_3d/models/accelerated_runtime.py
- app/roop_3d/models/benchmark.py
- app/roop_3d/models/diagnostics.py
- app/roop_3d/models/engine_cache.py
- app/roop_3d/models/export_validation.py
- app/roop_3d/models/golden.py
- app/roop_3d/models/precision_policy.py
- app/roop_3d/models/trt_builder.py
- app/roop_3d/requirements-trt.lock
- app/roop_3d/tests/manual_stage07_acceptance.py
- app/roop_3d/tests/test_stage07_cache.py
- app/roop_3d/tests/test_stage07_diagnostics.py
- app/roop_3d/tests/test_stage07_runtime.py
- app/roop_3d/tests/test_stage07_validation.py

## Primary references

- [ORT TensorRT provider and embedded contexts](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [ORT CUDA provider](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
- [ORT profiling](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)
- [TensorRT precision control](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/precision-control.html)
- [CUDA cache environment](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)
- [NVIDIA pinned SDK package index](https://pypi.nvidia.com/tensorrt-cu12-libs/)
