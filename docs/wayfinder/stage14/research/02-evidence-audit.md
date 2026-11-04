# 02 Evidence audit (Stage 14 wayfinder research)

Facts only. No design decisions. Ticket: `docs/wayfinder/stage14/tickets/02-evidence-audit.md`.
Staleness rule under test (map.md "Settled while charting", Validated bullet): an exact-match benchmark record needs all Model sha256, precision, execution provider, hardware tier, and render-profile definition to match.

Path convention: `APP/` = `app/roop_3d/`. Line numbers are 1-based. `.runtime/` is git-ignored (`APP/.gitignore:14`); files under it exist only on this machine. `APP/audit/*.json` and `APP/docs/*.md` are tracked.

Method and limits. Read-only. I read source and JSON, and ran the project venv Python (`APP/.venv/Scripts/python.exe`) only to parse JSON and to instantiate `ModelRegistry` + `ModelCompatibilityScanner` (no pytest, no benchmark, no model load, no weight hashing). `git status` before and after: only the pre-existing untracked `GLOSSARY.md` and `docs/wayfinder/`.

---

## 1. Evidence stores that exist

| # | Store | Location | Tracked | What it is |
|---|-------|----------|---------|-----------|
| A | Stage 11 benchmark DB | `APP/.runtime/benchmarks/benchmark_database.json` (2 records) | no | `BenchmarkRecord` rows (`evaluation/benchmark_db.py:33-50`) |
| B | Stage 11 harness/metrics/regression/dataset/reports | `APP/evaluation/*.py` | yes | scoring framework over synthetic scenes |
| C | Model qualification receipts | `APP/.runtime/acceleration/{6gb,12gb}/<model sha256>.json` (10 per tier) | no | per-Model provider/precision/parity/latency, written at `models/accelerated_runtime.py:213` |
| D | Stage 07 acceptance campaigns | `APP/.runtime/acceptance/STAGE07_{6gb,12gb,fp16_candidates_12gb,...}.json`; summary `APP/audit/STAGE07_VALIDATION.json` | .runtime no, audit yes | same receipts plus hardware snapshot, FP16 attempts |
| E | Golden tensor bank | `APP/.runtime/golden/stage07/` (10 `.npz` + `manifest.json`) | no | CPU-reference input/output tensors for parity (`models/golden.py`) |
| F | Pipeline stage benchmarks | `APP/audit/STAGE2_BENCHMARKS.json`, `STAGE09_BENCHMARKS.json`, `STAGE15_BENCHMARKS.json` | yes | per-component and end-to-end timings, 12 GB host only |
| G | Occlusion acceptance | `APP/audit/OCCLUSION_STAGE5_VALIDATION.json`, `APP/docs/OCCLUSION_STAGE5.md` | yes | quality metrics on real footage, CPU provider |
| H | `ModelRecord.benchmark_status` | `models/registry.py:97`, manifests | yes | free-string status field |
| I | Stage 12 lifecycle scan | `models/governance.py:19-100`, `models/catalog.py:116-129` | yes | READY / not-READY gate |

---

## 2. Stage 11 benchmark database (store A) and framework (store B)

### 2.1 Record schema (what a benchmark record contains)

`BenchmarkRecord` fields (`evaluation/benchmark_db.py:33-50`): `benchmark_id, timestamp, git_commit, hardware_profile, gpu_name, total_vram_mib, execution_provider, models{name -> ModelProvenance(model_name, sha256, precision, framework)}, parameters, engine_profile, composite_overall_score, safety_vetoes_passed, scene_metrics, tags, notes`. `ModelProvenance` is at `:24-30`. There is no field for latency, peak VRAM, driver, ORT/TRT version, dataset id/digest, render-profile definition, or parity/policy digest. No validation of `sha256` format. Records are looked up by `engine_profile` string or tag only (`:139-152`); `get_baseline_record` returns the newest record tagged `baseline` (`:154-157`).

### 2.2 The two stored records

`benchmark_database.json`: `baseline_inswapper_std` (`engine_profile` `standard_inswapper`, composite 96.47, line 28-29) and `candidate_3d_aware` (`3d_aware`, composite 96.52, line 65-66). Both: `hardware_profile` `12gb`, GPU RTX 4070, `execution_provider` `CUDAExecutionProvider`, `parameters` `{}` (lines 27, 64), `scene_metrics` = `{"frontal": {"identity": ...}}` only (lines 31-34, 68-71), `git_commit` `475ed73e...` (lines 8, 45; that commit is the Stage 10 commit and precedes the Stage 11 commit `5d2a5d0` that added `evaluation/`).

Observations, each checkable:

1. Stored sha256 values are not valid SHA-256 digests: `e4a3f08c207a385557372336423b1fa109497f087249932306` is 50 chars, `a01292dc223bb3468903e1cfcf0007297e59ab4fc358a0a86` is 49 chars (lines 16, 22, 53, 59). `ModelRecord.expected_sha256` requires 64 hex (`models/registry.py:52`).
2. They do not match the registered artifacts. Registry: `inswapper_128.onnx` = `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af`; `scrfd_10g_bnkps.onnx` = `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` (same sha as `det_10g.onnx`). Only the first 8 hex of the inswapper value agree. The same hashes are the keyword defaults at `benchmark_db.py:109-110`, and `gpu_name`, `total_vram_mib`, `hardware_profile`, `execution_provider` equal the keyword defaults at `:93-96`. The stored records are therefore indistinguishable from "defaults used"; whether they were passed explicitly cannot be determined from the file.
3. `scrfd_10g_bnkps.onnx` is recorded as `FP16` (lines 23, 60). `policy_for(..., role="detector")` marks detectors `fp16_candidate=False` with FP32 retained (`models/precision_policy.py:58-60`), and `PrecisionPolicy.tolerance("fp16")` raises `PRECISION_NOT_PERMITTED` for such models (`:24-29`). The qualification receipt for the same sha records `fp32` and `fp16_state: UNSUPPORTED`.
4. Nothing in the record references an engine definition: `engine_profile` is a bare string drawn from the `EngineProfile` enum (`evaluation/harness.py:31-37`) which is a harness label, not a Render profile definition.
5. No production code reads or writes the DB. `BenchmarkDatabase`/`record_run` occur only in `evaluation/benchmark_db.py`, `evaluation/__init__.py`, `evaluation/reports.py` (type import) and `tests/test_stage11_evaluation_framework.py:313,315,402-403`. The ids `baseline_inswapper_std` / `candidate_3d_aware` occur only in the DB file and `docs/STAGE11_QUALITY_FRAMEWORK.md:101`; no script in the repo creates them.
6. Gate result recorded in the doc: candidate vs baseline `REJECTED`, +0.05 pts, regression "Temporal flicker increased by +51.3%" (`docs/STAGE11_QUALITY_FRAMEWORK.md:98-107`).

### 2.3 What the harness actually measures

`EvaluationHarness.render_frame_for_profile` (`evaluation/harness.py:74-196`) does not call any registered Model or adapter. Each profile is a pixel operation on the synthetic target frame (alpha tint toward a constant BGR colour, Gaussian blur, unsharp) followed by `time.sleep`: standard_inswapper 5 ms (`:113`), 3d_aware 12 ms (`:135`), liveportrait_enhanced 25 ms (`:152`), alternative_local 18 ms (`:165`), restored_output 35 ms (`:186`). `latency_ms` is the elapsed wall time (`:114`, passed at `:223`). The `metadata` strings name models/providers (`:120,172,193`) but nothing is loaded. Module docstring `:1-10` describes the six profiles in model terms.

`ObjectiveMetricsEvaluator.evaluate_frame` (`evaluation/metrics.py:205-`): with no `source_canonical_embedding` it derives identity similarity from mean pixel difference inside the face mask, clipped to 0.65-0.85 (`:237-243`), or 0.05 if the output equals the target; with an embedding but no recognizer it returns constant 0.75 (`:236`). `compute_composite_score` (`:391-`) weights identity 0.25, pose/geometry 0.20, temporal 0.15, compositing/background 0.20, perceptual 0.20 and defines vetoes (e.g. identity min < 0.28 at `:398`).

Consequence (fact): the Stage 11 composite scores, vetoes and latencies in store A/B are properties of the simulated profiles, not measurements of any Model, provider or tier. The doc's own prose is inconsistent with its table: table throughput 43.0 / 41.4 / 25.3 / 18.5 FPS (`docs/STAGE11_QUALITY_FRAMEWORK.md:73-74`) vs prose "62.5 FPS", "25.6 FPS", "15.2 FPS" (`:78-81`); prose lines are literals in `evaluation/reports.py:176-181`. The doc's hardware header and model-hash table are literals too (`reports.py:115,219-225`), and the table's hashes (e.g. `w600k_r50` `b6807d9b...`, `2d106det` `2a865db2...`, `gpen_bfr_512` `26d03d3c...`, `CodeFormerv0.1` `819fef79...`, `bisenet_resnet18` `075677ad...`) do not equal the registry sha256s (`w600k_r50` `4c06341c33c2...`, `2d106det` `f001b856447c...`, CodeFormerv0.1 `9aa48fc4b212...`).

### 2.4 Golden dataset (evaluation) coverage

`GoldenDatasetRegistry.get_canonical_dataset` (`evaluation/dataset.py:456-468`) returns 11 procedurally generated scene categories (`SceneCategory`, `:42-53`), default 12 frames each, 512x512 (384x384 for resolutions). Scenes carry `metadata={"seed":..., "procedural": True}` (`:445`); faces are drawn synthetically (`_draw_synthetic_face`, `:138`). Materialisation cache path `.runtime/golden_dataset` (`:453`) does not exist on disk. The docs call the dataset "controlled, consented" (`dataset.py:1`, STAGE11 doc `:24`), but the code generates it; no real-footage scene exists in this framework. The DB record contains one scene key (`frontal`, identity only); the other 10 categories have no stored result. The dataset has no id/digest recorded in `BenchmarkRecord`.

### 2.5 Regression gate

`RegressionDetector.evaluate_candidate` (`evaluation/regression.py:85-252`) compares two `SceneMetricSummary`/`CompositeQualityScore` pairs. Latency is checked at `:197-212` (default max +25%). Verdict logic `:224-236`. It takes no Model hash, provider or tier argument.

---

## 3. Model qualification receipts, golden bank, and precision parity (stores C, D, E)

### 3.1 Receipt contents

`AcceleratorFactory.qualify` (`models/accelerated_runtime.py:140-214`) writes one receipt per (profile, Model sha256). Fields (`:198-211`): `name, model_sha256, golden_digest, policy_digest, profile, provider, requested_provider, precision, engine_key{artifact_sha256, precision, gpu_compute_capability, driver, onnxruntime_version, tensorrt_version, shape_profile, build_signature}, shape_profile, fallback_history, node_providers, parity[], rejected_parity, reference_benchmark, selected_benchmark, fp16_state, sensitive_reason, build_diagnostics{free_vram_bytes, workspace_bytes, safety_reserve_bytes,...}, acceleration_state, selection`.

Benchmarks: `models/benchmark.py:21-32` (warmups default 2, iterations default 10; `qualify` overrides to warmups=1, iterations=5, `accelerated_runtime.py:141`); output p50/p95/min/max/samples; "synchronous session timing including output transfer" (`benchmark.py:1`). Timing is on `golden.cases[0]` only (`accelerated_runtime.py:187,195`): one fixed input, fixed shape profile (`shape_profile` in receipt). The reference benchmark is the untouched CPU provider on the same input (`:191-195`).

### 3.2 What is on disk (all dated 2026-10-30, ORT 1.23.2, TensorRT 10.9.0.34, driver 621.01, CC 8.9)

| Model (sha256 prefix) | tier | provider (resolved) | prec. | p50 ms (selected) | p50 ms (CPU ref) | note |
|---|---|---|---|---|---|---|
| inswapper_128 (e4a3f08c753c) | 6gb / 12gb | TensorRT | fp32 | 32.03 / 29.57 | 2708.80 / 1151.88 | fp16_state UNKNOWN |
| scrfd_10g_bnkps = det_10g (5838f7fe0536) | 6gb / 12gb | TensorRT | fp32 | 8.31 / 4.07 | 231.50 / 95.32 | fp16 UNSUPPORTED |
| w600k_r50 (4c06341c33c2) | 6gb / 12gb | TensorRT | fp32 | 5.35 / 4.32 | 375.62 / 100.46 | fp16 UNKNOWN |
| 2d106det (f001b856447c) | 6gb / 12gb | TensorRT | fp32 | 0.63 / 0.72 | 2.67 / 2.41 | fp16 UNSUPPORTED |
| liveportrait_landmark (31d22a504132) | 6gb / 12gb | TensorRT | fp32 | 6.34 / 2.64 | 104.52 / 43.61 | |
| liveportrait_appearance_feature_extractor (d070afccca7f) | 6gb / 12gb | TensorRT | fp32 | 16.84 / 9.45 | 625.82 / 294.44 | fp16 UNKNOWN |
| liveportrait_motion_extractor (219a46174297) | 6gb / 12gb | TensorRT | fp32 | 9.41 / 5.53 | 143.60 / 72.06 | |
| liveportrait_warping_spade (b0e7a566db8f) | 6gb / 12gb | **CPU** (`reference_fallback`) | fp32 | 11823.67 / 8649.32 | 15352.11 / 8961.17 | TRT: `TRT_CONTEXT_UNSUPPORTED` (GridSample excluded); CUDA: `PROVIDER_EXECUTION_FAILED` |
| liveportrait_stitching (8e33658425e3) | 6gb / 12gb | TensorRT | fp32 | 0.60 / 0.20 | 0.07 / 0.03 | TRT slower than CPU ref at 6gb |
| liveportrait_stitching_eye (b9a2086c4d75) | 6gb / 12gb | TensorRT | fp32 | 0.63 / 0.09 | 0.07 / 0.03 | |

Source: `APP/.runtime/acceleration/{6gb,12gb}/*.json` (`selected_benchmark.p50_ms`, `reference_benchmark.p50_ms`, `provider`, `selection`, `fallback_history`). All 10 Models use the same 10 shas in both tiers. Receipt sets lack: `liveportrait_stitching_lip`, `bisenet_*`, `xseg*`, `face_occluder`, `gpen_*`, `CodeFormerv0.1`, `hyperswap_*`, `hififace_*`, `adaface`, `glintr100`, `w600k_mbf`, `yunet`, `retinaface`, `yoloface`, restorer/upscaler/age Models.

Golden bank (store E): same 10 names/shas, 2 cases per model, reference provider CPU, ORT 1.23.2 (`.runtime/golden/stage07/manifest.json`; format `models/golden.py:76-96`). Capped at 1..8 cases per model (`golden.py:28-30,49`).

### 3.3 "6gb" evidence is a budget policy on a 12 GB card

Both tiers' files record one physical `NVIDIA GeForce RTX 4070`, `total_vram_mib` 12282 (`STAGE07_6gb.json:2205`, `STAGE07_12gb.json:2191`). Difference between tiers is `RuntimeConfig(hardware_profile=...)` / budget: "6gb" run had `free_vram_mib` 4147 and `effective_vram_budget_mib` 3379 (`STAGE07_6gb.json:2206,2209`); "12gb" run had free 5643, budget 4875 (`STAGE07_12gb.json:2192,2195`). `audit/STAGE07_REPORT.md:5`: "not a physical 6 GB laptop GPU ... Physical 6 GB qualification, production tail latency and dedicated peak VRAM remain UNKNOWN." Stage 2/9/15 JSONs mark `RTX_3060_Laptop` `UNAVAILABLE_LOCAL` / "benchmark not executed (anti-fabrication)" (`STAGE09_BENCHMARKS.json:43`, `STAGE15_BENCHMARKS.json:34`, `STAGE2_BENCHMARKS.json:43`). Tier boundaries in code: `resolve_hardware_profile` (`core/hardware.py:88-99`), ceilings 4600 / 12288 MiB (`core/hardware.py:84-85`), `RuntimeProfile = Literal["cpu","6gb","12gb"]` (`engine/vram_governor.py:15`), `RenderPolicy.hardware_profile` (`render_policy.py:26,35`).

### 3.4 CPU tier

No record in any store carries `hardware_profile`/`profile` equal to `cpu` (grep of `.runtime/acceptance`, `.runtime/acceleration`, `.runtime/benchmarks`, `audit/`). CPU-provider timings exist only as (a) `reference_benchmark` blocks inside the 6gb/12gb receipts (table above), (b) occlusion CPU timing in prose (section 5), (c) reenactment engine "~35 ms on CPU" in prose (`docs/REENACTMENT_STAGE4.md:115`). `STAGE07_cuda_*` files carry CUDA-provider runs (not read in detail).

### 3.5 Precision parity reports

- Parity machinery: `Tolerance` (`models/export_validation.py:16-32`), `compare_outputs` (`:94-139`), `ParityReport` (`:48-57`); admission `PrecisionPolicy.admit` (`precision_policy.py:31-36`). Receipts embed `parity[]` per golden case and `policy_digest`.
- FP32 parity passed for all 10 Models (receipt `parity[].passed`, `rejected_parity: {}` except below).
- FP16 attempts (`STAGE07_fp16_candidates_12gb.json`, `fp16_candidates_attempted: true`): inswapper TRT FP16 rejected (max_abs 0.557; failures max_absolute, mean_absolute, image_psnr, image_ssim, border_absolute); w600k_r50 TRT FP16 rejected (`mean_absolute` 0.01436); liveportrait_appearance TRT FP16 rejected (max_abs 1.309). Policy marks candidates only for recognizer, identity_swapper, liveportrait_appearance (`precision_policy.py:43-51`).
- Therefore the only precision with a passing parity report for any Model is **FP32**. No FP16/INT8 admission exists.
- Receipts in `.runtime/acceleration` are the latest `qualify()` outputs. The FP16-campaign file shows different providers for two Models than the final receipts (motion_extractor and scrfd resolve to CUDA there, `fallback_history` `INFERENCE_OUTPUT_CONTRACT: Output names differ`; final receipts show TensorRT). A receipt is overwritten per (profile, sha) at `accelerated_runtime.py:213`; history is kept only in the `STAGE07_*` campaign files.

---

## 4. `ModelRecord.benchmark_status` (store H) and lifecycle gate (store I)

- Field: `benchmark_status: str = "NOT_RUN"` (`models/registry.py:97`). Free string, no enum. Effective registry = `model_manifest.json` (101) + `stage16_models.json` (16) + `stage6_models.json` (4) = **121** records (`registry.py:161-169`). Present in the JSON for only the 4 Stage 06 records (`NOT_RUN`); absent in the other 117 and defaulted. **All 121 are `NOT_RUN`.**
- No writer: `benchmark_status` is referenced only at `models/registry.py:97`, `models/hub.py:236` (read), and `models/hub.py:276,296,315` (hardcoded `"NOT_RUN"` for adapter descriptors); the HTTP route `/api/settings/benchmark_status` at `server/api.py:30-31` is a different thing (benchmark job state). Nothing links receipts (store C) or the DB (store A) to `benchmark_status`.
- Fields that a Stage 14 hardware check would read, across all 121 records: `minimum_vram_gib` set 0, `recommended_vram_gib` set 0 (`registry.py:93-94`), `precision_requirements` non-empty 0, `supported_precision` non-empty 0 (`:89,96`), `supported_providers` non-empty 4 (all `("CPUExecutionProvider",)`: det_10g x3, scrfd_10g_bnkps), `hardware_requirements.state` all `UNKNOWN` (`:88`), `dependency_compatibility.state` all `UNKNOWN` (`:90`), `security_status` `VERIFIED` 0 (`:92`), `inference_availability` `UNKNOWN` 117 / `EXPERIMENTAL` 4 (`:98`).
- Lifecycle: `ModelCompatibilityScanner.scan` (`models/governance.py:19-100`) returns `READY` only when no issue is raised (`:94-95`); issues include unknown dependency compat (`:26-28`), unknown hardware requirements (`:40-44`), security not VERIFIED (`:62-63`), empty precision requirements (`:64-65`), unknown provenance/attribution (`:66-69`), unreviewed license (`:70-71`). My best-case run (assume `installed=True`, `integrity_verified=True`, `hardware={"vram_total_gb":12,"profile":"12gb"}`) over all 121 records: **0 READY, 113 LICENSE_RESTRICTED, 8 FAILED** (the 8 are `candidate`/`blocked` records without sha256: FLAME 2020, Wav2Lip, InstantID, PuLID, SDXL-Turbo, FLUX.1-Schnell, SMIRK_em1, emoca_flame). This matches `docs/STAGE12_MODEL_GOVERNANCE.md:69-73` ("zero models READY for inference").
- Fact for the staleness rule: under the "Validated" definition the lifecycle-READY precondition fails for every registered Model today, independent of any benchmark evidence. (Stated as a fact about the current manifests, not a recommendation.)
- Several `ModelRecord`s share one sha256 (e.g. `inswapper_128.onnx` records `4387a15d1f7f` and `6e6f38875f64`; `w600k_r50.onnx` x3; `det_10g.onnx` x3 + `scrfd_10g_bnkps.onnx`; `2d106det.onnx` x3). `model_id` differs per record; sha256 is the stable identity. Receipts key by sha256, not model_id. Different files can share a name: `gpen_bfr_512.onnx` sha `d5f066b9068a...` (and a `.batch` variant) vs `GPEN-BFR-512.onnx` sha `0960f8364887...`.

---

## 5. Component and end-to-end benchmarks (store F) and occlusion (store G)

All on the 12 GB RTX 4070 host only; no `Model sha256` field; component named by string.

| Component (JSON key) | provider string | p50 ms | source line |
|---|---|---|---|
| video decode (PyAV) | "CUDA / CPU Multi-Threaded" | 8.075 (115.4 fps) | `STAGE09_BENCHMARKS.json` `1_video_decoding` (~:50-60); Stage 2: 9.853 "CPU / FFmpeg" |
| detector scrfd_10g_bnkps.onnx | CUDAExecutionProvider | 15.374 | Stage 9 `:63-72`; Stage 2: 9.027 |
| dense landmarks 2d106det.onnx | CUDAExecutionProvider | 82.414 | Stage 9 `:76-85`; Stage 2 `3_dense_landmarks_106`: 1.273 (same component, 65x apart) |
| tracker (ArcFace embed + Hungarian + One-Euro) | Hybrid | 107.094 | Stage 9 `:88-98` |
| inswapper_128.onnx FP32 | CUDAExecutionProvider | 35.593 | Stage 9 `:101-111` |
| PoseBlendProjector (mesh + z-buffer) | "Vectorized 3D Rasterizer" (no Model) | 40.66 | Stage 9 `:114-123` |
| LivePortrait motion extractor / warping_spade | CUDAExecutionProvider | 228.036 / 212.78 | Stage 9 `6_motion_transfer` (receipts C show warping_spade CUDA failing and CPU fallback 8.6-11.8 s) |
| BisenetFaceParser "bisenet_resnet18.onnx" | CUDAExecutionProvider | 79.92 | Stage 9 `:143-152` (name is not a registry `name`; registry has `bisenet_resnet_18.onnx` remote and `resnet18.onnx`) |
| GPEN-BFR-512 / CodeFormerv0.1 | CUDAExecutionProvider | 154.324 / 165.39 | Stage 9 `:155-176` |
| encode "PyAV libx264 / NVENC" | "PyAV / FFmpeg" | 0.274 (3326 fps) | Stage 9 `:195-205` |

End-to-end: Stage 9 15 frames, 0.57 FPS, warm p50 1056.5 ms, `comparison_against_stage2.speedup_factor 0.28` (`STAGE09_BENCHMARKS.json:211,251`); Stage 15 10 frames 0.07 FPS (`STAGE15_BENCHMARKS.json:109`). Stage breakdown (Stage 9): detect_track 325, swap 31, retarget 427, restore 155, composite 125 ms p50.

Peak VRAM: only whole-process nvidia-smi polling in `run_benchmarks.py:160-181` -> `STAGE15_BENCHMARKS.json:110-111` (`initial_gpu_vram_mib` 3681, `peak_gpu_vram_mib` 3970, whole 10-frame pipeline, other processes included). Prose figure "1,938 MiB peak" for the full pipeline at `docs/INFERENCE_PROFILING_STAGE2.md:88`, with no source JSON found. Stage 9 JSON holds point-in-time `active_gpu_telemetry.vram_used_mib` (2973), not a peak. `audit/STAGE07_VALIDATION.json:5716` records `full_video_fps_or_peak_vram_claim: UNKNOWN`. Receipt `build_diagnostics.free_vram_bytes`/`workspace_bytes` are TensorRT build-time figures, not inference peaks. 6 GB per-Model VRAM limits in `INFERENCE_PROFILING_STAGE2.md:95-101` are allocator divisors (policy), not measurements; the same divisor idea is coded at `accelerated_runtime.py:103-106`.

Occlusion (store G): `OCCLUSION_STAGE5_VALIDATION.json` records sha256 for `bisenet_resnet_34.onnx` (4a0b8c95...), `xseg_3.onnx` (48ccd7e8...), `face_occluder.onnx` (ad9bfb90...) and scenario metrics on real footage with `acceptance.provider: CPUExecutionProvider`. No latency field in the JSON. Timing is prose only: CPU standard ~170-250 ms, robust ~440-720 ms per face per frame, "measured while a hashing job ran"; "A CUDA run on the RTX 4070 executed correctly but its timing was not isolated; no performance claim is made for either GPU profile" (`docs/OCCLUSION_STAGE5.md:170-171`). Occlusion coverage gaps stated in the doc: microphone crossing mouth `NOT_COVERED` (`:166`).

Hardware-codec evidence: NVDEC/NVENC native gate passed 2026-11-01 on synthetic workload; "execution evidence ... not production ... video FPS" (`docs/STREAM_STAGE3.md:174-181`). No NVDEC/NVENC timing record exists.

---

## 6. Render-profile definition: what exists to hash

- Render options in code: `render_mode` in `{standard, pose_3d, reenactment_3d}` (`core/workstation.py:26`; `RENDER_MODES` asserted at `tests/test_reenactment_engine.py:78`), `occlusion_mode` `{standard, robust}` (`:30`), `restorer_model` `{auto, gpen, codeformer}` (`:9`), `pose_mode` `{retarget, mesh_blend}` (`:16`), relight and other fields.
- No named, versioned Render-profile object exists. Closest hashable things: `settings_digest(Settings)` over the whole settings model (`jobs/project_store.py:21-22`), `JobRequest.identity_payload` digest (`engine/pipeline_runner.py:159-162`), `render_dependencies` model-sha map (`models/catalog.py:18-49`; names list `:24-29`; used at `pipeline_runner.py:360`).
- Stage 11 `EngineProfile` (6 values) is a harness enum and does not equal the above set (e.g. no `occlusion_mode`, `pose_3d` vs `3d_aware` naming).
- No evidence record in A-G carries any field identifying render_mode/occlusion_mode/restorer for the measurement, except Stage 9/15 JSON component names ("PoseBlendProjector", "GpenRestorer", "CodeFormerRestorer"), and prose in docs.

---

## 7. Gap table: exact-match record under the settled rule

Legend. Key = (every Model sha256 in the Engine, precision, execution provider, hardware tier, render-profile definition). "Model-key" = the key without render-profile definition. `R` = a store-C receipt exists for this sha on that tier. Engine composition is ticket 01's question; rows below are per Model or per code path with the composition named only where code/docs state it. For every row, the **strict** column (full key including render-profile definition) is **NO** in every cell because no store has a render-profile-definition field (section 6).

### 7.1 Per Model (category: Model(s) -> tier cells)

| Category | Model / component (sha prefix) | cpu | 6gb | 12gb | Model-key evidence (if tier-equal) |
|---|---|---|---|---|---|
| Face detection | scrfd_10g_bnkps / det_10g (5838f7fe0536) | none | R: TRT fp32, 8.31 ms | R: TRT fp32, 4.07 ms | receipts only; Stage 11 DB record has wrong sha, FP16; Stage 9 CUDA 15.4 ms has no sha |
| Face detection | yunet, retinaface_10g/r50, yoloface_8n, scrfd dims | none | none | none | none |
| Face recognition | w600k_r50 (4c06341c33c2) | none | R: TRT fp32, 5.35 ms | R: TRT fp32, 4.32 ms | fp16 attempted and rejected (12gb) |
| Face recognition | adaface_ir101, glintr100, sface, w600k_mbf, arcface_w600k_r50 | none | none | none | none |
| Dense landmarks | 2d106det (f001b856447c) | none | R: TRT fp32, 0.63 ms | R: TRT fp32, 0.72 ms | Stage 2 vs Stage 9 CUDA figures disagree 65x |
| Dense landmarks | liveportrait_landmark (31d22a504132) | none | R: TRT fp32 | R: TRT fp32 | receipts only |
| Identity swapping | inswapper_128 (e4a3f08c753c) | none | R: TRT fp32, 32.03 ms | R: TRT fp32, 29.57 ms | fp16 rejected; Stage 11 DB sha mismatched |
| Identity swapping | hyperswap_1a/1b/1c_256, hififace_unofficial_256, crossface_hififace, inswapper_128.batch | none | none | none | none (registry/identity only) |
| 3D reconstruction | PoseBlendProjector (no Model) | none | none | Stage 9: 40.66 ms, provider string "Vectorized 3D Rasterizer", no sha/render key | not Model-keyed |
| 3D reconstruction | flame_encoder candidates (SMIRK, EMOCA), FLAME 2020 | none | none | none | no sha256, lifecycle FAILED |
| Expression transfer | LivePortrait appearance (d070afccca7f), motion (219a46174297), stitching (8e336584...), stitching_eye (b9a2086c...) | none | R: TRT fp32 | R: TRT fp32 | per-Model only |
| Expression transfer | liveportrait_warping_spade (b0e7a566db8f) | none | R: **CPU** fp32 11.8 s | R: **CPU** fp32 8.6 s | TRT and CUDA both failed in receipt; Stage 9 reports CUDA 212.8 ms with no sha |
| Expression transfer | liveportrait_stitching_lip | none | none | none | no receipt, no golden |
| Occlusion segmentation | xseg_3, face_occluder, bisenet_resnet_34 (shas in section 5) | CPU quality metrics (real footage); latency in prose only | none | CUDA ran, timing "not isolated" | no latency record on any GPU tier |
| Occlusion segmentation | xseg, xseg_1, xseg_2, xseg trt surgeon fp16 | none | none | none | none |
| Restoration | gpen_bfr_512 (d5f066b9068a / GPEN-BFR-512 0960f8364887), CodeFormerv0.1 (9aa48fc4b212) | none | none | Stage 9 CUDA 154.3 / 165.4 ms, no sha | no receipt, no golden |
| Restoration | gfpgan, restoreformer++, gpen 256/1024, codeformer.fp16 | none | none | none | none |
| Video decoding | PyAV VideoReader / NvdecFrameSource (no Model) | none | none | Stage 9: 8.075 ms, provider string ambiguous "CUDA / CPU Multi-Threaded" | NVDEC: no timing |
| Video encoding | PyAV libx264 / NVENC (no Model) | none | none | Stage 9: 0.274 ms, provider "PyAV / FFmpeg" (codec used not recorded) | NVENC: no timing |

### 7.2 Per render profile (the Engines that existing code actually composes)

Only compositions stated in sources: standard swap path requires detector + recognizer + inswapper (`render_dependencies` required set, `models/catalog.py:30-38`); reenactment engine measured with real SCRFD/2d106 tracker + real inswapper (`docs/REENACTMENT_STAGE4.md:105-107`).

| Render profile | Models stated in sources | cpu | 6gb | 12gb | Gap |
|---|---|---|---|---|---|
| `render_mode=standard`, `occlusion_mode=standard`, no restorer | scrfd, w600k_r50, inswapper_128 | no | Model-keys: R x3 (TRT fp32) - Engine-level record: none | Model-keys: R x3 - Engine-level record: none | no Engine-level latency/quality record carries all three shas + render definition; Stage 11 DB `standard_inswapper` has wrong shas (FP16 scrfd) and simulated metrics |
| `render_mode=pose_3d` | not established in this audit | no | no | no | Stage 11 `3d_aware` record is simulated; models and render definition undetermined |
| `render_mode=reenactment_3d` | scrfd, 2d106det, inswapper_128 (per REENACTMENT_STAGE4) | no | Model-keys: R x3 | Model-keys: R x3 | "+40 ms/frame, engine ~35 ms on CPU" is prose with no sha/tier |
| `occlusion_mode=robust` (3 networks) | bisenet_resnet_34, xseg_3, face_occluder | CPU timing prose only | no | no | no GPU timing; measured under load |
| `restorer_model=gpen` or `codeformer` | gpen_bfr_512 / CodeFormerv0.1 | no | no | Stage 9 component only | no sha-keyed record |

**Totals.** Of the 10 Stage 14 categories x 3 tiers = 30 cells: 0 cpu cells have any tier-labelled record. 5 categories (detection, recognition, dense landmarks, identity swapping, expression transfer) have Model-level receipts on 6gb and on 12gb, i.e. 10 of 30 cells, and each of those categories has other registered Models with none (expression transfer is partial: warping_spade resolved to CPU, stitching_lip has no receipt). Occlusion, restoration, 3D reconstruction, decoding and encoding have no sha-keyed record in any tier. 0 cells have a record satisfying the full key including render-profile definition. 0 Models are lifecycle-READY (section 4), so the Validated precondition fails in all 30 cells independent of benchmark evidence.

---

## 8. Fields the benchmark record lacks, and where each could come from

(`BenchmarkRecord` = store A.)

| Needed by rule / Stage 14 | In `BenchmarkRecord`? | Where it exists today |
|---|---|---|
| Model sha256 (valid 64-hex) | present but malformed/mismatched | `ModelRecord.expected_sha256` (`registry.py:52`); receipts `model_sha256`; `render_dependencies()` map (`catalog.py:18-49`) |
| Precision | `ModelProvenance.precision` string, unvalidated (FP16 recorded for a detector the policy forbids) | receipt `precision` + `policy_digest` (`accelerated_runtime.py:198-199`); `PrecisionPolicy.digest` (`precision_policy.py:21`) |
| Execution provider | single top-level string, defaulted (`benchmark_db.py:96`) | receipt `provider`/`requested_provider`/`node_providers` (resolved, per Model); `AcceleratedModel.capability()` (`accelerated_runtime.py:315-321`); `session.get_providers()` |
| Hardware tier | `hardware_profile` string (defaulted `"12gb"`, `:95`) | receipt `profile`; `RuntimeConfig.hardware_profile`; `resolve_hardware_profile` (`core/hardware.py:88`); note tier = budget policy, not physical card (section 3.3) |
| Render-profile definition | absent (`engine_profile` string only; `parameters` empty) | no existing definition object; candidates for a hash: `Settings` render fields (`core/workstation.py:26,30,9,16`), `settings_digest` (`project_store.py:21`), job `identity_payload` digest (`pipeline_runner.py:159`) |
| Latency (p50/p95, warm/cold) | absent in record (only simulated value inside `SceneMetricSummary`, not stored) | receipts `selected_benchmark`/`reference_benchmark` (per Model, golden fixture, 5 iterations); `models/benchmark.py` helper; Stage 9 JSON component timings (no sha); `run_benchmarks.py` stage latencies |
| Peak VRAM | absent (`total_vram_mib` is card size) | no per-Model or per-Engine inference peak exists anywhere; whole-process polled peak in `STAGE15_BENCHMARKS.json:111` only; `ModelRecord.minimum_vram_gib`/`recommended_vram_gib` exist but are empty for all 121 (`registry.py:93-94`); allocator divisors `accelerated_runtime.py:103-106` are budgets |
| Driver / ORT / TRT versions, compute capability | absent | receipt `engine_key` (`engine_cache.py:28-35`), `hardware` in STAGE07 files; `core/hardware.py` probe |
| Golden/eval dataset id + digest | absent | `ModelGolden.digest()` (`golden.py:32-33`) in receipts as `golden_digest`; Stage 11 scenes have `seed` only (`dataset.py:445`) |
| Parity result | absent | receipt `parity[]`, `rejected_parity` |
| Measured vs simulated provenance | absent (nothing distinguishes a harness-simulated record) | `EngineOutput.metadata` strings (`harness.py`) are labels, not evidence |
| Multi-Model Engine aggregation | `models` dict allows multiple | per-Model receipts have no Engine-level roll-up |
| Staleness inputs (Model hash change, provider change) | not checkable (hash malformed) | `ModelRegistry` sha + receipt `model_sha256`/`policy_digest`/`golden_digest` comparison is done at `AcceleratedModel.__init__` (`accelerated_runtime.py:251-253`) |

---

## 9. Undetermined by this audit

1. Whether the two DB records were produced by passing the defaults explicitly or by omission (no producing script in repo; file mtime 2026-11-04 14:10 local).
2. Which Models make up the `pose_3d` and `3d_aware` render paths in the runner (ticket 01 / 04 territory); I did not trace `engine/volumetric/` or `reconstruction/`.
3. Whether the receipts' CUDA-vs-TRT provider differences between the FP16 campaign and the final receipts (section 3.5) reflect code changes or session options; not investigated.
4. Why Stage 9 reports CUDA `warping_spade_4d` at 212.8 ms while receipts for the same registry artifact show CUDA failing; cannot be reconciled because the Stage 9 JSON carries no sha or session options.
5. Stage 2 vs Stage 9 landmark latency difference (1.27 ms vs 82.4 ms): cause not determined.
6. Whether receipts remain valid under the current code (policy digest, golden digest recomputation not run; I did not load any model or run `qualify`).
7. The source of the "1,938 MiB" peak VRAM statement (`INFERENCE_PROFILING_STAGE2.md:88`); no measurement artifact found.
8. Whether any physical 6 GB or CPU-only-host run exists outside this repo; none recorded. The CPU tier has no `RuntimeProfile`-labelled measurement of any kind.
9. `STAGE07_cuda_{6,12}gb.json` and `stage06-*` acceptance files were not read in detail; they may add CUDA-provider receipts.
10. How `benchmark_status` should map to receipts/DB (no code path exists).
11. Where the 3 `liveportrait_stitching_lip`-class and restorer/segmentation Models would obtain golden fixtures (the golden bank captures from "genuine Stage 06 CPU inference", `golden.py:1`; only 10 Models captured).
