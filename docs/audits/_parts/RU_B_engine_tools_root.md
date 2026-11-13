# RU slice B â€” roop-ultimate `face_engine/`, `tools/`, `scripts/`, `benchmarks/`, `portable/`, `web_ui/`, `tests/`, CI/hooks, repo root

Reference: roop-ultimate @71ebb65 (read-only clone). Paths are relative to the repo root. All claims verified by reading the cited file unless tagged [INFERENCE].

## 0. Architecture of this slice

Two code bases live in roop-ultimate. `app/` (slice A) is the shipped Roop Ultimate app (FastAPI + `react-ui/` + `roop.*`). `face_engine/` (this slice) is a **standalone, newer 2D-only accelerator package**: ORT/TensorRT/CUDA runtime, model zoo, GPU detector/tracker/aligner/masker, batched swap/restore, media I/O, a small FastAPI server and its own `web_ui/` (`pyproject.toml:1-38`, `pytest.ini` 3 test trees). It is not imported by `app/` [INFERENCE from pyproject "standalone `face_engine` package only"; not grep-verified for app/].

**There is no 3D volumetric reconstruction anywhere in this slice.** grep for `3drecon|3dmm|fsz|volumetric|flame|deca|mesh` finds only `torch.meshgrid` calls, sidecar-file words, and the `1k3d68` *landmark* model (`face_engine/models/zoo.py:85`, a 68-point 3D-layout landmark net used by the app, FP32-locked `core/trt_compiler.py:269-273`). `.fsz` appears only in `tools/acceptance_stage15.py` (a person_a.fsz faceset file consumed by the app's API; header line 1). Pose is 2D-only: `estimate_yaw_5pt` geometry guess (`core/guardrails.py:90-97`), a profile-guard on Umeyama singular values (`pipeline/aligner.py:727-760`), and LivePortrait keypoints for expression (`processors/expression.py:1-25`). Stereo (`pipeline/stereo.py`) is SBS/TB 2D-per-eye epipolar coupling, not 3D reconstruction.

## 1. face_engine internals (detection / pose / warping / quality / runtime)

### 1.1 Detection (`pipeline/detector.py`, 1242 lines; `boosted/retinaface_gpu.py`)
- `BaseDetector`: input 640, score 0.5, NMS IoU 0.4 (`detector.py:325-327`); letterbox into a square canvas, `torchvision.ops.batched_nms` with frame index as group. Normalization per model measured on 240 frames: SCRFD `(x-127.5)/128` RGB, YOLOFace `x/255` (`detector.py:1-29`).
- `GPUSCRFDDetector` (tight canvas padded to a multiple of 32, TensorRT FP16): score 0.45, IoU 0.40 (`detector.py:626-627`).
- `AngleResilientSCRFD` (`detector.py:904-1000`): primary pass at current angle; **conditional rot90 sweep** (3 other angles batched at `sweep_size` 640) when no face >= 0.50, or face rolled > `max_pass_roll` 45 deg, or fewer faces than one of the last `lost_window` 2 frames, or tracker asked (`note_track_lost`). Dual thresholds high 0.50 / low 0.20, NMS 0.45, `preferred_hold` 10 frames, scene declared empty after `empty_after` 30 frames then swept every `empty_rescan` 15. This sweep is 90-degree-quantised in-plane roll only; **it is not a yaw/pitch (out-of-plane) detector**. Docstring quantifies SCRFD-at-0deg on a 90deg-rotated t1.jpg: 4/6 faces, 180deg: 2/6, landmarks 5-28% of face size off.
- `RetinaFaceR50Detector` (`boosted/retinaface_gpu.py`): vectorised anchors (`priors`), TensorRT, batched NMS; `strided_tracker` helper. Registered `retinaface_r50` (`models/zoo.py:39`).
- Model zoo (`models/zoo.py`, 461 lines): scrfd_10g_bnkps, retinaface_r50, yoloface_8n (640 fixed, AGPL), hrffa (**declared, `_UNAVAILABLE`**, `zoo.py:60`), 2dfan4 (68-pt, 256 input, 97.9 MB), 2d106det, 1k3d68, arcface_w600k_r50, hyperswap_256/512, alpha_256, alphaface_256 (**urls=() unavailable, `zoo.py:186`**), reswapper_128/256, inswapper_128(+fp16), hififace_256 + crossface_hififace, face_occluder_v3/xseg_3/dfl_xseg_v2/xseg, bisenet_resnet34, sam2_hiera_tiny (.pt), gpen_bfr_512/1024/2048, restoreformer_plus_plus, codeformer_fp16.
- Failure behaviour: guardrails never raise on empty frames (`core/guardrails.py:1-30`); `Face.kps` not clipped so edge-cut faces survive (`detector.py:54`).

### 1.2 Pose / angle thresholds (`core/guardrails.py`, `pipeline/aligner.py`)
- `estimate_yaw_5pt` (`guardrails.py:90`): geometry-only yaw from nose offset vs half-eye distance, `nose_depth=1.0` half eye distances; saturates when landmarks collapse. No pose network.
- `choose_alignment_points` (`guardrails.py:120-146`): use dense landmarks only if confidence >= `DENSE_MIN_CONFIDENCE` 0.35 and |yaw| <= `DENSE_MAX_YAW_DEG` 75 deg; else 5-point fallback. `MIN_DET` 1e-12, `MIN_SPREAD_PX` 0.5 (`guardrails.py:42-45`); `safe_affine` replaces singular/non-finite matrices.
- `ProfileGuardedAligner` (`aligner.py:807-838`, `profile_guarded_similarity_cuda:727`): closed-form 2x2 Umeyama (no cuSOLVER), floors `s2/s1` at `min_ratio` 0.15, scale only (docstring concedes a similarity cannot stretch; rotation unaffected); padding_mode "reflection". Templates: arcface_112, arcface_128 (+8px x-shift, 256 crops for HyperSwap), ffhq_512 (GPEN/RF++/CodeFormer), arcface_112_v1 (SimSwap 512) (`aligner.py:53-64`). `fit_error` exposed as a gate for profiles (`aligner.py:156`).
- **Wiring gap:** `AngleResilientSCRFD`, `RobustByteTracker`, `ProfileGuardedAligner`, `choose_alignment_points`, `estimate_yaw_5pt`, `EpipolarStereoCoupler`, `HybridVideoMasker`, provenance signer and `UltraRestorer` are referenced only by their own module, `__init__` re-exports and tests (grep over non-test `face_engine/`); `server/processing.py` uses `GPUSCRFDDetector` + `StridedFaceTracker` + `TemporalFaceTracker` only (`processing.py:127,203`). They are library-complete but not on the production render path [INFERENCE: grep shows no caller, dynamic import not excluded].

### 1.3 Temporal (`pipeline/tracker.py`, 1012 lines)
- `StridedFaceTracker`: full detection every N frames (UI 1-8; presets use 3), pyramidal GPU Lucas-Kanade between (`window` 21, `fb_threshold` 1.0 px, `min_valid_fraction` 0.5, 4 pyramid levels), forced re-detect on cut / lost face / no faces (`tracker.py:1-30,54-72`).
- `SceneCutDetector`: per-channel 32-bin histogram of 64-px-wide frame, 0.5*L1 averaged over channels, `cut_threshold` 0.5 (`tracker.py:70-118`); host read of 1 scalar per frame (`:405`).
- `TemporalFaceTracker`: EMA `alpha` 0.75 (weight on NEW), match cost 0.5*(1-IoU)+0.5*landmark dist, `max_cost` 0.7, `max_missed` 3 retention (`tracker.py:512-534`).
- `RobustByteTracker` (`:816`, Kalman XYAH `:742`): high 0.50 / low 0.20, match IoU high 0.20 (distance form), low 0.50, `max_lost_frames` 5, motion-compensated EMA alpha 0.75; docstring states plain EMA trails motion by `v(1-alpha)/alpha` (`tracker.py:672-714`).
- Quantified claims (synthetic): Stage9 jitter variance 39.72 -> 5.79 px^2 (-85.42%), 24 crossings 100% identity stable, 5-frame dropout recovered (`benchmarks/baselines/benchmark_stage9_temporal_tracking.json`). Stage3 detector 80 frames: every-frame 16.34 fps vs temporal 46.9 fps (2.87x) (`benchmark_stage3_detector.json`).

### 1.4 Swap / quality (`processors/`, `enhancers/`, `pipeline/masker.py`)
- `swapper.py` (720 lines): `IdentityEncoder` (ArcFace w600k_r50 aligned to arcface_112, 512-d L2-norm), HyperSwap 1a/1b/1c consumes vector directly; inswapper uses `normalize(v @ emap)`. **Pixel Boost** polyphase k*k sub-image swap (256x256/512x512/1024x1024) (`swapper.py:1-40`). Swap sessions built `trt_fp16=False`; FP16 overflow measured: HyperSwap identity distance 0.352->0.407, inswapper "rainbow" smudges (`swapper.py:28-31`). Swappers are 2D image-to-image warps on 256/128 crops: model-bound resolution.
- `masker.py` (637 lines): `M = Feather(Box) * XSeg * Regions * Valid` in the 256 crop (`masker.py:1-5`). XSEG_SIZE 256, PARSER_SIZE 512 (ffhq_512), XSeg threshold 0.1, feather sigma 1.0 px@256, box blur 0.3, BiSeNet-ResNet34 19 CelebAMask-HQ classes; default regions skin, nose, eyes, brows, lips, inner mouth (`masker.py:69-71,102-135`). Notes L/R eye labels unreliable. XSeg output is visible-face probability used un-inverted (`:12-18`).
- `boosted/xseg_masker.py`: morphological erode-then-blur feather on GPU; dual-stream swap || mask (`boosted/ultra_pipeline.py:61-90`); VRAM cap `set_per_process_memory_fraction(0.90)` (`ultra_pipeline.py:~25-30,50`).
- `pipeline/hybrid_masker.py`: SAM2 hiera-tiny memory tracking minus XSeg-3, bbox expanded 15%, dilation 3 px max_pool2d, gaussian sigma 0.5-4.0, 16-frame FIFO (`hybrid_masker.py:1-18`). Tests exist only in `test_modern_stages.py`.
- `processors/color.py`: modes via LAB mean/std transfer; region min 16 px; host (OpenCV 8-bit LAB) and CUDA paths equal within rounding (`color.py:1-25`).
- Restorers: `processors/enhancer.py` GPEN-BFR 512/1024/2048 + RestoreFormer++; `enhancers/ultra_engine.py` Ultra Restore: ScaleAwareEnhancerRouter (bypass / GPEN-512 17.8 ms / GPEN-1024 32.3 ms per face, `scale_router.py:47-52`), LAB lock to the original crop, `STATS_SIZE` 128, `COLLAPSE_STD` 2.0 guard, frequency-split fusion (crop sigma 2.5, 17-px kernel = `2*ceil(3sigma)+1`, `frequency.py:1-30`), BiSeNet region weights (`semantic_fusion.py`). Measured table (`ultra_engine.py:40-50`): linear alpha 0.8 identity med 0.916/p10 0.693 vs split sigma 2.5 0.991/0.959, texture 232% vs 160%, 27.3 vs 32.2 ms/face. GPEN TRT engines require final linear + style pixel-norm pinned FP32 or 1024 NaNs (`ultra_engine.py:3-8`). Engines batch-1 (StyleGAN modulated conv).
- `processors/expression.py`: LivePortrait expression transfer `x_d = x_s + scale*w*(exp_target-exp_s)`, groups lips(6,12,14,17,19,20)/eyes(11,13,15,16,18)/other, blink-sync 97% of frames vs 33% (`expression.py:1-30`); `utils/gridsample5d.py` rewrites 5-D GridSample for TensorRT.
- `pipeline/provenance.py`: offline HMAC-signed manifests in MP4/PNG/WebP + DCT watermark (370 lines); not wired.

### 1.5 Compute (`core/`)
- `config.py:49`: default provider order TensorRT -> CUDA -> CPU; TRT workspace 4 GiB, `trt_fp16_enable` True, engine cache `.cache/trt_engines` (`config.py:54-64`); `strict` mode turns silent ORT EP drop into `ProviderFallbackError` (`config.py:130-135`, `execution.py`). Env: FACE_ENGINE_DEVICE_ID / CACHE_DIR / MODELS_DIR.
- `core/trt_compiler.py` (710 lines): AOT engine build + verify (NaN reject, fidelity) + json sidecar keyed on TRT version/SM/graph stamp; whole-model FP16 locks xseg, 2d106det, 1k3d68 to FP32 (`trt_compiler.py:269-273`). `zero_copy_engine.py`: ORT IOBinding/OrtValue with profile min 256 / max 512 (`:1-20`; hard-coded `get_default_face_shape_profile`). `utils/onnx_batch.py` dynamic batch + FP16 + InstanceNorm decomposition. `cuda_streams.py` (579 lines): triple-buffered decode/inference/encode on three streams, event hand-off, ring buffer pinned+VRAM; NVDEC "loses once inference shares the GPU and deadlocks in-process" so software PyAV decode is the shipped path (`cuda_streams.py:8-13`).
- Measured presets (`server/processing.py:90-96`): RTX 4070, 1080p, 2 faces, 600 frames, spec targets 60/30/12 fps vs measured 56.2/44.5/8.7 (Stage 4-6 sequential 50.3/40.4/8.6).

### 1.6 Media / jobs (`media/`, `server/`)
- Output: libx264 `-preset medium -crf 18 -pix_fmt yuv420p +faststart`, BT.709/BT.2020 tagging; ffmpeg 8.1 colour-tag workaround (`ffmpeg_pipe.py:9-14,64-68,199-236`); NVENC writer with x264 fallback (`encoder.py`); lossless demux to `.m4a`/`.mka` sidecars (not raw .aac, 21.3 ms late) (`demuxer.py:17`); A/V sync check tolerance one frame period (`guardrails.py:169-206`).
- `worker_pool.py`: GOP-aligned segments, one spawned process per chunk, seamless concat; `ipc_pool.py` shared-memory ring (613 lines).
- `server/state.py`: **one in-memory Job**, states preparing/rendering/completed/failed/cancelled, `threading.Event` cancel, job thread (`state.py:113-126,366-406`). No persisted queue, no checkpoint/resume in face_engine (the app has pause/resume via root `pause.js`/`resume.js`).
- REST (`server/api.py`): `/api/project/load`, `/api/project`, `/api/project/assign`, `/api/detect/faces`, `/api/options`, `/api/pipeline/start|stop|status`, `/api/outputs/{filename}`, `/api/media/target`, `/api/thumbnails/{name}`; preview routes in `preview.py`; `/ws/telemetry` in `telemetry.py` (NVML else nvidia-smi sampler). `RenderParams` is `extra="forbid"`: detector (scrfd/retinaface/yolo), swapper, pixel_boost, mask_types box/occlusion/region, enhancer_blend 0-100, provider, mask padding, mask_blur 0.3, match_threshold 0.3, detection_stride 1-8, temporal_smoothing, workers 1-8 (`processing.py:59-83`).

## 2. Precision and CUDA policy documents

- `PRECISION_POLICY.md` (226 lines): model-specific precision matrix (safe/candidate/unsafe/not-validated/unsupported). Facts: GPEN-1024/2048 and GFPGAN **FP32-only**; all swap nets FP16 `unsafe`, mixed `candidate`; XSeg mixed `unsafe` (IoU min 0.81); landmarks 2d106det/1k3d68 FP32 override (drift 1.2-8.8 px); RestoreFormer++ FP32 default (21.63 -> 39.31 ms/face, +82%). Swap INT8 on hyperswap_1a (RTX 4070, TRT 10.9.0.34, 186 faces): INT8 3.69 ms but identity -0.026 (59.1% of faces lose >0.02) => unsafe; FP8 unsupported on Ada (0 layers run FP8, 14.48 ms). RTX 3060 results **pending**. Enforced via `precision_policy.PRECISION_OVERRIDES` (app side, slice A).
- `CUDA_EXECUTION_POLICY.md` (174 lines): stream policy by VRAM (<7 GB => 1 stream, 0 TRT aux; >=7 GB => <=2 streams, <=1 aux); CUDA-graph GPEN-256 Pro filter rejected (23% slower); fork-join XSeg on second stream +15.9% single pair but +2.2-3.0% (noise) with 2-4 frame workers; `cv2->grid_sample` and `dilate->max_pool2d` rejected (ellipse vs square error up to 1.0); end-to-end 7.74 fps / 13.58 fps render phase.
- `PERFORMANCE_BASELINE.md` (261 lines): locked 2026-08-29 baseline, i9-14900K / RTX 4070 12 GB / 32 GB; final Stage-15: 7.74 fps e2e 600 frames, VRAM peak 12,073 MB, decode 21.3 / detect 18.7 / swap 97.0 / mask 35.2 / enhance 99.8 / blend 37.1 ms per call; software decode 204 fps, hevc_nvenc 880 fps. Explicit not-run list (benchmark_pipeline k1.mp4, RTX 3060).
- `OPTIMIZATION_PLAN.md` (717 lines): 16-phase plan + gates A-E (adversarial review, targets, Rubin, i9 CPU, unified scheduler). Current gate: physical RTX 3060 Laptop validation of phases 0-4 pending.

## 3. Benchmarks (`benchmarks/baselines`, `scripts/benchmarks`, `tools/benchmark_*`, `tests/benchmark_pipeline.py`)

Stage0 baseline (RTX 4070, 13 scenarios, `benchmark_stage0_summary.csv`; **synthetic fixtures**, [INFERENCE] from `tests/benchmark_pipeline.py` SyntheticFace/`generate_synthetic_video_fixture`): frontal 11.62 fps / ID 0.9854; 45-degree 11.62 / 0.9069; **extreme profile 4.36 fps, 25 detection failures, ID 0.467**; small face ID 0.767; multi-interacting faces 0.33 fps ID 0.4238; partial occlusion ID 0.8027; high-motion landmark instability 0.8553; 1080p 0.50 fps; 4K 0.64 fps; VRAM saturates at ~12.1 GB from scenario 7 on. Stage files 2-9 cover lifecycle, detector, alignment, HyperSwap audit v2 (quality via `tools/quality_harness.py` after earlier typed-in values were declared INVALID), restore-ultra profiles FAST/BALANCED/QUALITY/ULTRA, xseg3, compositing, temporal. `trace_frame_analysis.json`: 1280x720 single frame 286.3 ms.

## 4. Per-file inventory

Status legend used in purpose text: files under `face_engine/` are real implementations unless noted in section 6; `tools/apply_*.py` are one-shot source patchers (see 6); `tools/test_*.py`, `tests/*` are tests/probes. Bytes = inventory bytes; lines = `wc -l`.


### repo root (42 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `.clinerules` | 438 | 7 | agent-rules stub (7 lines) pointing to AGENTS.md; directives ignored |  |
| `.cursorrules` | 438 | 7 | same agent-rules stub |  |
| `.gitattributes` | 415 | 9 | LF enforcement for shell hooks |  |
| `.gitignore` | 3455 | 119 | IDE |  |
| `.windsurfrules` | 438 | 7 | same agent-rules stub |  |
| `AGENTS.md` | 53647 | 767 | Roop Ultimate â€” agent rules (single source of truth) |  |
| `CLAUDE.md` | 471 | 9 | Roop Ultimate - Claude Code |  |
| `CUDA_EXECUTION_POLICY.md` | 13316 | 174 | Phase 6 â€” CUDA Streams and CUDA Graphs |  |
| `ENVIRONMENT` | 3176 | 89 | Pinokio env template: PINOKIO_SHARE_*, GRADIO_TEMP_DIR, HF_HOME, TORCH_HOME etc. (89 lines) |  |
| `GEMINI.md` | 466 | 9 | Roop Ultimate - Gemini |  |
| `LICENSE` | 35184 | 661 | AGPL-3.0 text |  |
| `NOTICE.md` | 6369 | 124 | Notices and attribution |  |
| `OPTIMIZATION_PLAN.md` | 19354 | 717 | ROOP-ULTIMATE PERFORMANCE OPTIMIZATION PLAN |  |
| `PERFORMANCE_BASELINE.md` | 12526 | 261 | ROOP-ULTIMATE PERFORMANCE BASELINE |  |
| `PRECISION_POLICY.md` | 17735 | 226 | Phase 5 â€” Model-Specific Precision Policy |  |
| `QWEN.md` | 464 | 9 | Roop Ultimate - Qwen |  |
| `README.md` | 46600 | 823 | Roop Ultimate |  |
| `conftest.py` | 6399 | 139 | pytest configuration for both suites (app/tests and tests/). |  |
| `icon.png` | 429149 | 1729 | Pinokio icon (binary) |  |
| `install.js` | 6452 | 188 | Install Python dependencies for the backend (app/ is already in the repo) |  |
| `link.js` | 129 | 10 | Pinokio dedupe/link script (10 lines) |  |
| `metadata.json` | 315 | 10 | Pinokio app metadata |  |
| `package-lock.json` | 80382 | 2383 | npm lockfile |  |
| `package.json` | 699 | 27 | root npm manifest (react-ui typecheck/mock server) |  |
| `pause.js` | 916 | 25 | Pause the running job from the Pinokio sidebar â€” POSTs /api/pause so the |  |
| `pinokio.js` | 7813 | 195 | app/env is created before the later Python, model-support, and React |  |
| `pinokio.json` | 234 | 7 | Pinokio manifest |  |
| `pinokio_batch_runner.py` | 14222 | 309 | Run the retained video folders with one fresh GPU process per video. |  |
| `pyproject.toml` | 1333 | 38 | Packaging for the standalone `face_engine` package only. The Roop Ultimate |  |
| `pytest.ini` | 1355 | 27 | pytest config: testpaths app/tests tests face_engine/tests, --import-mode=importlib, pythonpath app . |  |
| `requirements.txt` | 551 | 20 | face_engine runtime deps (21 root req) |  |
| `reset.js` | 639 | 32 | The production build is what the backend actually serves, so leaving it |  |
| `resume.js` | 722 | 23 | Resume a paused job from the Pinokio sidebar â€” POSTs /api/resume so the |  |
| `run.py` | 708 | 17 | Root entry point forwarding execution to app/run.py. |  |
| `start.js` | 316 | 5 | The canonical Pinokio entry point launches React UI 1.0, the production |  |
| `start_legacy.js` | 2260 | 69 | DEPRECATED. The Gradio interface (app/ui) is frozen: it gets no new features and is kept only so an existin |  |
| `start_react.js` | 7400 | 177 | Never infer readiness from app/env. Verify the final package contract |  |
| `stop.js` | 1108 | 29 | Graceful "Stop Swap" â€” POSTs to the running backend's /api/stop so the |  |
| `torch.js` | 716 | 19 | Runtime provisioning is intentionally unconditional. Pinokio's documented |  |
| `tsconfig.json` | 321 | 13 | root TS config for mock-server typecheck |  |
| `update.js` | 3522 | 127 | Follow the branch this checkout is on (not a hard-coded main) and fetch |  |
| `update_manifest.json` | 1941 | 62 | generated dependency/runtime contract (tools/gen_update_manifest.py) |  |

### .githooks (1 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `.githooks/pre-commit` | 2548 | 48 | Regenerate update_manifest.json from the index and stage it, so a commit that |  |

### .github (1 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `.github/workflows/ci.yml` | 4218 | 101 | Vite 8 needs ^20.19 |  |

### benchmarks (12 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `benchmarks/baselines/benchmark_results.json` | 8164 | 295 | 2026-09-29 RTX4070 benchmark_pipeline results, 20 arms, 300-frame slices |  |
| `benchmarks/baselines/benchmark_stage0_results.json` | 62024 | 2117 | Stage0 13-scenario full result (pipeline_config, hardware_profile, per-scenario metrics) |  |
| `benchmarks/baselines/benchmark_stage0_summary.csv` | 1794 | 14 | Stage0 13-scenario summary: fps, latency, VRAM, identity sim, landmark instability |  |
| `benchmarks/baselines/benchmark_stage2_lifecycle.json` | 2198 | 77 | Stage2 model-lifecycle timings for 5 models |  |
| `benchmarks/baselines/benchmark_stage3_detector.json` | 1726 | 62 | Stage3 detector: full-every-frame 16.34 fps vs temporal 46.9 fps (2.87x) etc., 80 frames |  |
| `benchmarks/baselines/benchmark_stage4_alignment.json` | 1214 | 40 | Stage4 alignment strategies: raw Umeyama vs EMA vs pose-aware stabilizer jitter |  |
| `benchmarks/baselines/benchmark_stage5_hyperswap.json` | 175509 | 5559 | Stage5 HyperSwap audit v2 via quality_harness (5 clips, 300 frames; supersedes invalid earlier file) |  |
| `benchmarks/baselines/benchmark_stage6_restore_ultra.json` | 3277 | 91 | Stage6 restore profiles FAST/BALANCED/QUALITY/ULTRA latency + VRAM |  |
| `benchmarks/baselines/benchmark_stage7_xseg3.json` | 1279 | 34 | Stage7 xseg_3 (256 NHWC) latency/quality |  |
| `benchmarks/baselines/benchmark_stage8_compositing.json` | 3909 | 127 | Stage8 compositing: dE, seam, halo, dark-scene metrics (synthetic) |  |
| `benchmarks/baselines/benchmark_stage9_temporal_tracking.json` | 1691 | 56 | Stage9 jitter -85.42%, 5-frame dropout recovery, cut guard (synthetic) |  |
| `benchmarks/baselines/trace_frame_analysis.json` | 1759 | 67 | 1280x720 single-frame trace: 286.3 ms total, per-stage ms, H2D/D2H counts |  |

### face_engine (5 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/README.md` | 71774 | 1234 | face_engine |  |
| `face_engine/__init__.py` | 1869 | 41 | face_engine: accelerator runtime and model registry for the face pipeline. |  |
| `face_engine/benchmark.py` | 21473 | 482 | Benchmark the face_engine render: per-stage latency, sustained fps, VRAM, CPU, sync locks. |  |
| `face_engine/requirements.txt` | 752 | 12 | extra deps: av>=17,<18, kornia>=0.8,<0.9, kornia_rs>=0.2 |  |
| `face_engine/run.py` | 16128 | 346 | One-command launcher for face_engine: models, TensorRT engines, then the service. |  |

### face_engine/boosted (6 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/boosted/__init__.py` | 809 | 15 | The boosted pipeline: RetinaFace R50 + HyperSwap-256 + DFL XSeg (+ GPEN-1024 "Ultra"). |  |
| `face_engine/boosted/benchmark_boosted.py` | 10746 | 244 | Benchmark the boosted pipeline: per-engine TensorRT latency, fps per preset, VRAM, CPU. |  |
| `face_engine/boosted/retinaface_gpu.py` | 9116 | 196 | RetinaFace ResNet-50 on the GPU: TensorRT engine, vectorized anchor decoding, batched NMS. |  |
| `face_engine/boosted/trt_compiler.py` | 5179 | 109 | AOT TensorRT engines for the boosted pipeline: R50, HyperSwap-256, DFL XSeg, GPEN-1024. |  |
| `face_engine/boosted/ultra_pipeline.py` | 6363 | 138 | The boosted render: R50 + HyperSwap-256 |  |
| `face_engine/boosted/xseg_masker.py` | 2295 | 53 | DFL XSeg with morphological feathering, on the GPU. |  |

### face_engine/core (9 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/core/__init__.py` | 73 | 1 | package marker |  |
| `face_engine/core/base_processor.py` | 4075 | 108 | Abstract base processor and execution options contract for roop-ultimate. |  |
| `face_engine/core/config.py` | 8475 | 198 | Typed configuration for the face engine runtime. |  |
| `face_engine/core/cuda_streams.py` | 27682 | 579 | Triple-buffered decode / inference / encode on three CUDA streams. |  |
| `face_engine/core/execution.py` | 32784 | 759 | ORT session mgmt, explicit provider fallback (ProviderFallbackError), ManagedSession, VRAM query, GPU DLL registration |  |
| `face_engine/core/guardrails.py` | 10480 | 224 | Edge-case guardrails: degenerate geometry, extreme pose, zero-face frames, A/V sync. |  |
| `face_engine/core/registry.py` | 9154 | 231 | Declarative model registry. |  |
| `face_engine/core/trt_compiler.py` | 33633 | 710 | Ahead-of-time TensorRT engine compilation, verification and lookup. |  |
| `face_engine/core/zero_copy_engine.py` | 7776 | 212 | Zero-overhead inference execution engine with TensorRT and CUDA zero-copy buffering. |  |

### face_engine/enhancers (5 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/enhancers/__init__.py` | 1592 | 42 | Ultra Restore: frequency-split detail injection, scale routing, region weights. |  |
| `face_engine/enhancers/frequency.py` | 8156 | 164 | Frequency-split detail injection: the restorer's texture on the swap's light. |  |
| `face_engine/enhancers/scale_router.py` | 6842 | 164 | Scale-aware routing: spend restorer compute only where the frame can show it. |  |
| `face_engine/enhancers/semantic_fusion.py` | 11657 | 251 | Region-weighted detail injection: how much restorer texture each part of the face gets. |  |
| `face_engine/enhancers/ultra_engine.py` | 21952 | 448 | Ultra Restore: routed GPEN restoration, LAB lock, region-weighted frequency-split fusion. |  |

### face_engine/media (9 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/media/__init__.py` | 2484 | 45 | Video ingestion, GPU decode/encode, demux/remux, shared-memory IPC, segment workers. |  |
| `face_engine/media/capturer.py` | 17268 | 421 | Video ingestion: container metadata, frame-accurate decoding, GOP-aligned segments. |  |
| `face_engine/media/decoder.py` | 20388 | 466 | Video decoding into batches of CUDA tensors: software (default) or NVDEC. |  |
| `face_engine/media/demuxer.py` | 4988 | 111 | Lossless track extraction before processing, and the final remux after it. |  |
| `face_engine/media/encoder.py` | 9741 | 223 | NVENC encoding of GPU frames through an ffmpeg pipe. |  |
| `face_engine/media/ffmpeg_pipe.py` | 25538 | 536 | Encode BGR frames to browser-ready H.264/AAC MP4 through an ffmpeg pipe, and verify it. |  |
| `face_engine/media/ipc_pool.py` | 25188 | 613 | Zero-copy shared-memory frame transport between processes. |  |
| `face_engine/media/tools.py` | 2543 | 68 | Locate ffmpeg/ffprobe and run ffprobe, without depending on the caller's PATH. |  |
| `face_engine/media/worker_pool.py` | 10867 | 253 | Segment-parallel rendering: GOP-aligned chunks, one process per chunk, seamless concat. |  |

### face_engine/models (2 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/models/__init__.py` | 27 | 1 | package marker |  |
| `face_engine/models/zoo.py` | 26532 | 461 | The model zoo: every model the engine can load, declared once. |  |

### face_engine/pipeline (8 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/pipeline/__init__.py` | 3601 | 101 | re-exports pipeline API (detector, tracker, aligner, masker, stereo, provenance) |  |
| `face_engine/pipeline/aligner.py` | 40003 | 846 | 5-point similarity alignment, templates arcface_112/arcface_128/ffhq_512/arcface_112_v1, CUDA Umeyama, ProfileGuardedAligner (min_ratio 0.15), paste-b |  |
| `face_engine/pipeline/detector.py` | 59943 | 1242 | Face detection: SCRFD and YOLOFace behind one interface. |  |
| `face_engine/pipeline/hybrid_masker.py` | 11045 | 266 | Hybrid Video Masker combining SAM 2 video memory tracking and XSeg-3 occlusion. |  |
| `face_engine/pipeline/masker.py` | 28913 | 637 | Tri-layer composite face mask: box x occlusion (XSeg) x semantic regions (BiSeNet). |  |
| `face_engine/pipeline/provenance.py` | 12891 | 370 | Local Cryptographic Provenance Signer & Imperceptible Frequency-Domain Watermarking. |  |
| `face_engine/pipeline/stereo.py` | 15639 | 391 | Stage 21 â€” Stereoscopic 3D (SBS/TB) Epipolar Disparity Lock & VR Viewport Engine. |  |
| `face_engine/pipeline/tracker.py` | 46339 | 1012 | Detection stride: full detection every N frames, optical-flow tracking between. |  |

### face_engine/processors (6 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/processors/__init__.py` | 2250 | 42 | Face processors: swapping, restoration, expression restoration, colour. |  |
| `face_engine/processors/color.py` | 5386 | 128 | Colour stabilisation: tie a swapped / restored face's colour back to the target. |  |
| `face_engine/processors/enhancer.py` | 21055 | 438 | Generative face restoration: GPEN-BFR 512/1024/2048 and RestoreFormer++. |  |
| `face_engine/processors/expression.py` | 27930 | 558 | Expression restoration with LivePortrait: put the target's expression back on the swap. |  |
| `face_engine/processors/modern_processors.py` | 9921 | 253 | Modernized concrete processors adhering to BaseProcessor. |  |
| `face_engine/processors/swapper.py` | 33715 | 720 | Identity transfer: HyperSwap 1a/1b/1c and inswapper-128, with Pixel Boost. |  |

### face_engine/server (7 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/server/__init__.py` | 60 | 1 | package marker |  |
| `face_engine/server/__main__.py` | 1483 | 34 | Run the server: ``python -m face_engine.server [--host H] [--port P] [--workspace DIR]``. |  |
| `face_engine/server/api.py` | 10598 | 229 | FastAPI application: project, detection, pipeline, media and outputs. |  |
| `face_engine/server/preview.py` | 10454 | 235 | POST/GET /api/preview/frame: FrameCache (GOP-filled LRU) + PreviewService single-frame render |  |
| `face_engine/server/processing.py` | 21075 | 438 | Per-frame GPU processing shared by the render job and the live preview. |  |
| `face_engine/server/state.py` | 20385 | 469 | Server state: workspace, project (sources/target/people/assignments), render jobs. |  |
| `face_engine/server/telemetry.py` | 8684 | 204 | GpuSampler (NVML else nvidia-smi), GuardedSampler, TelemetryHub for /ws/telemetry |  |

### face_engine/tests (27 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/tests/__init__.py` | 0 | 0 | package marker (empty) |  |
| `face_engine/tests/bench_presets.py` | 4078 | 90 | Measure the server's presets: preview latency and full render fps. |  |
| `face_engine/tests/bench_stage3.py` | 9650 | 201 | Stage 3 benchmark: single-face sequential vs batched (B=4) throughput. |  |
| `face_engine/tests/bench_streams.py` | 6918 | 162 | A/B: the server's sequential render vs the Stage 7 stream pipeline. |  |
| `face_engine/tests/eval_detection_stride.py` | 5663 | 132 | Measure detection stride on a real clip: landmark accuracy and time per frame. |  |
| `face_engine/tests/eval_stage3_precision.py` | 6314 | 139 | Measure FP16 vs FP32 for the batched swap / restore models on real faces. |  |
| `face_engine/tests/media_fixtures.py` | 2644 | 55 | Synthetic test videos generated with ffmpeg's lavfi sources (no private footage). |  |
| `face_engine/tests/media_workers.py` | 2327 | 77 | Top-level callables for FramePipeline tests (spawned processes import them by name). |  |
| `face_engine/tests/test_angle_resilience.py` | 19808 | 438 | Angle resilience: AngleResilientSCRFD, RobustByteTracker, ProfileGuardedAligner. |  |
| `face_engine/tests/test_boosted.py` | 7832 | 165 | The boosted pipeline: RetinaFace R50, XSeg feathering, dual-stream processor, engines. |  |
| `face_engine/tests/test_cuda_streams.py` | 10843 | 259 | Stage 7: the triple-buffered decode / inference / encode stream pipeline. |  |
| `face_engine/tests/test_detector_gpu.py` | 9432 | 204 | GPUSCRFDDetector (tight canvas, TensorRT FP16) and TemporalFaceTracker. |  |
| `face_engine/tests/test_gpu_pipeline.py` | 20773 | 434 | Stage 2, CUDA-resident: detection stride, kornia warps and the GPU masker. |  |
| `face_engine/tests/test_guardrails.py` | 11029 | 237 | Stage 8 guardrails: degenerate geometry, zero-face pass-through, pose policy, A/V sync. |  |
| `face_engine/tests/test_launcher.py` | 9346 | 198 | Stage 8: ``face_engine/run.py`` (launcher) and ``face_engine/benchmark.py`` logic. |  |
| `face_engine/tests/test_media.py` | 16012 | 337 | Stage 4: ingestion, shared-memory pipeline, ffmpeg output, HTTP 206 streaming. |  |
| `face_engine/tests/test_modern_stages.py` | 3908 | 108 | Tests for modernized Stage 1, Stage 2, and Stage 3 components. |  |
| `face_engine/tests/test_pipeline.py` | 23159 | 498 | Stage 2 vision pipeline: detection, alignment, composite masking. |  |
| `face_engine/tests/test_processors.py` | 20709 | 418 | Stage 3: swapping, restoration, expression restoration. |  |
| `face_engine/tests/test_runtime.py` | 23901 | 495 | Runtime tests for face_engine: provider grant, device buffers, caching, registry. |  |
| `face_engine/tests/test_server.py` | 17382 | 349 | Stage 5 backend: REST, WebSocket telemetry, preview, background render, stop, 206 + CORS. |  |
| `face_engine/tests/test_stage3_gpu.py` | 18305 | 369 | Stage 3, CUDA-resident: batched swapping, Pixel Boost, restoration, colour, expression. |  |
| `face_engine/tests/test_trt_compiler.py` | 11088 | 212 | Stage 6: AOT TensorRT compilation, verification, lookup and use. |  |
| `face_engine/tests/test_ultra_quality.py` | 23538 | 483 | Ultra Restore: frequency split, scale routing, region weights, LAB lock, TensorRT GPEN. |  |
| `face_engine/tests/test_video_hw.py` | 13731 | 273 | Stage 4 hardware I/O: NVDEC decoder, NVENC writer, demux/remux, segment worker pool. |  |
| `face_engine/tests/test_web_ui_e2e.py` | 9745 | 186 | The built web UI, driven in a real headless Chromium against the real server. |  |
| `face_engine/tests/verify_video_pipeline.py` | 9334 | 198 | Measure total pipeline FPS (decode + inference + encode) on a video, e.g. 1080p. |  |

### face_engine/utils (4 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `face_engine/utils/__init__.py` | 25 | 1 | package marker |  |
| `face_engine/utils/downloads.py` | 7691 | 186 | Verified, resumable model downloads. |  |
| `face_engine/utils/gridsample5d.py` | 14221 | 325 | Rewrite 5-D GridSample nodes (LivePortrait warping_spade) into ops TensorRT/CUDA can build. |  |
| `face_engine/utils/onnx_batch.py` | 20264 | 433 | Derived ONNX variants: a dynamic batch dimension, and FP16 weights. |  |

### portable (6 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `portable/.gitattributes` | 39 | 2 | LF for shell scripts |  |
| `portable/.gitignore` | 28 | 3 | ignore runtime/wheels/vendor |  |
| `portable/README.md` | 3263 | 71 | portable runtime docs: options, offline bundle, env |  |
| `portable/bootstrap.py` | 19794 | 455 | Portable runtime bootstrap for Roop Ultimate (stdlib only). |  |
| `portable/run.bat` | 2419 | 61 | Windows zero-install launcher (uv + CPython venv) |  |
| `portable/run.sh` | 2133 | 60 | Linux/macOS zero-install launcher |  |

### scripts (10 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `scripts/benchmarks/benchmark_pipeline.py` | 21771 | 496 | Automated Benchmarking Harness for roop-ultimate on one target clip (default ``k1.mp4``, resolved at runtim | get_peak_vram_mb,ssim_boundary_variance,BenchmarkSuite,main |
| `scripts/benchmarks/run_stage0_benchmark.py` | 11533 | 283 | CLI Runner for Stage 0 Reproducible Performance + Quality Benchmark. |  |
| `scripts/clean.js` | 5983 | 149 | Lives in scripts/: a Pinokio script's working directory is its own folder, |  |
| `scripts/cleanup.py` | 15839 | 390 | Disk cleanup for the Roop Ultimate launcher (driven by scripts/clean.js). |  |
| `scripts/fix_tensorrt.js` | 1973 | 65 | Lives in scripts/: a Pinokio script's working directory is its own folder, |  |
| `scripts/profile_pipeline.py` | 17916 | 440 | Standalone Benchmark & Profiling Script for roop-ultimate. |  |
| `scripts/verification/check_api_mock_parity.py` | 22372 | 431 | Fast static contract check: backend routes and /ws/telemetry keys vs the mock server and the pill model. |  |
| `scripts/verification/verify_integration.py` | 2830 | 60 | Stage 19 integration verifier: what happens when a second face or an object crosses the target. |  |
| `scripts/verification/verify_pipeline.py` | 7676 | 157 | Lightweight verification script to validate the browser-compliant video pipeline. |  |
| `scripts/verify_roop_keep.py` | 24496 | 504 | Production verification runner for the local roop-keep corpus. |  |

### tests (56 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `tests/benchmark_pipeline.py` | 36951 | 841 | Automated End-to-End Integration and Benchmarking Suite. |  |
| `tests/benchmark_results.json` | 2199 | 88 | stored result of tests/benchmark_pipeline.py |  |
| `tests/probe_fused_manual_mask.py` | 5850 | 110 | What do the keep-original matte and the low-band mix cost the fused kernel, and what would the in-kernel in | med_ms,main |
| `tests/probe_fused_postprocess.py` | 6026 | 131 | The fused Stage 3 post-processing kernel against what it replaces. |  |
| `tests/probe_l2_persistence.py` | 5551 | 115 | Does a persisting L2 window keep the ROI working set resident when a full-frame decode ring floods the cach | _read_kernel,one_arm,main |
| `tests/test_angles.py` | 16627 | 356 | Unit tests for Omnidirectional Angle Rectification & 3D Profile-Aware Weighted Umeyama. |  |
| `tests/test_benchmark_runner.py` | 16788 | 400 | Session 2 Verification: 60-frame dry-run benchmark with 1 face selected. |  |
| `tests/test_benchmark_telemetry.py` | 6032 | 174 | Standalone verification for benchmark telemetry and profile persistence. |  |
| `tests/test_benchmark_video_harness.py` | 9236 | 242 | Verification test suite for Session 2 benchmark video harness and workload runner. |  |
| `tests/test_build_trt_engines.py` | 16518 | 329 | tools/build_trt_engines.py with fake TRT sessions: engine cache, profiles, rejection (19 tests) | _Response,_FakeSessionOptions,_FakeGraphOptimizationLevel,_FakeSession,_FakeOrt |
| `tests/test_core_import_surface.py` | 6721 | 153 | Every name the repository imports from `roop.core` must exist on `roop.core`. |  |
| `tests/test_core_module_not_shadowed.py` | 1241 | 28 | asserts no core/ package shadows core.py; exports survive | test_no_core_package_shadows_core_module,test_core_module_exports_survive |
| `tests/test_enhancers_with_occlusion_and_aspect_ratios.py` | 13056 | 297 | Integration test: All face enhancers with non-standard video aspect ratios, |  |
| `tests/test_fused_postprocess.py` | 14245 | 275 | app/core/fused_postprocess.py: the one-kernel Stage 3 post-processing must equal the unfused maths it repla | _case,_lab,_close,TestMathAgainstOpenCV,TestKernelEqualsReference |
| `tests/test_fused_postprocess_fallback.py` | 7580 | 138 | app/core/fused_postprocess.py without a GPU: the keep-original matte, the dispatcher and the torch fallback | _case,_box,TestManualKeepMask,TestLowAndDetailWeight,TestDispatcherAndFallback |
| `tests/test_gpen256_quality_and_bugs.py` | 7351 | 154 | GPEN-256 enhancer stabilizer must not cross-contaminate multi-face selections (6 tests) |  |
| `tests/test_gpu_bindings.py` | 13015 | 302 | GPU binding smoke test: DLL registration, ONNX Runtime providers, torch CUDA. |  |
| `tests/test_interacting_faces_and_occlusions.py` | 7342 | 159 | Unit tests for interacting faces, foreground occlusions, and non-target demarcation. |  |
| `tests/test_l2_persistence.py` | 4028 | 94 | app/core/l2_persistence.py: the carve-out API, the arena, and that nothing is left set behind. |  |
| `tests/test_lod_and_gaze.py` | 6481 | 149 | Self-verification for Adaptive LOD model routing and LivePortrait neural gaze retargeting. |  |
| `tests/test_nvof_pipeline.py` | 5507 | 118 | Self-verification of the three-tier temporal motion-vector pipeline. |  |
| `tests/test_occlusion_interaction.py` | 2753 | 73 | Stage 19: a second face or an object crossing the target, through the real pipeline. |  |
| `tests/test_performance_regression.py` | 31673 | 679 | Stage 5 performance / fidelity regression suite (real production renders). |  |
| `tests/test_prebuild_engines.py` | 6128 | 154 | tools/prebuild_engines.py: the pure parts (the build itself needs a GPU). |  |
| `tests/test_realityux_production_space.py` | 11209 | 244 | RealityUX in the shapes production hands it. |  |
| `tests/test_recognition_pipeline.py` | 25767 | 496 | Recognition pipeline: correctness checks and a benchmark harness. |  |
| `tests/test_relighting_and_grain.py` | 2837 | 66 | Self-verification for SH relighting and scene-matched sensor grain. |  |
| `tests/test_retinaface_closeup.py` | 15052 | 305 | Regression tests for RetinaFace R50 close-up detection, multi-scale pyramid, |  |
| `tests/test_stage0_benchmark.py` | 7766 | 215 | Unit and regression tests for Stage 0 Benchmark Harness. |  |
| `tests/test_stage10_pipeline_concurrency.py` | 13130 | 313 | Unit and integration tests for Stage 10 â€” CPU/GPU Pipeline Concurrency. |  |
| `tests/test_stage11_vram_ram_management.py` | 11776 | 314 | Tests for Stage 11 â€” VRAM/RAM Management Architecture and Runtime Limits. |  |
| `tests/test_stage12_video_io_pipeline.py` | 6669 | 183 | Tests for Stage 12 â€” Video I/O Pipeline Optimization and Bottleneck Analysis. |  |
| `tests/test_stage13_inference_runtime.py` | 19639 | 440 | Tests for Stage 13 â€” Inference Runtime Optimization. |  |
| `tests/test_stage14_advanced_settings.py` | 12719 | 307 | Tests for Stage 14 â€” Advanced Settings Redesign. |  |
| `tests/test_stage15_runtime_scheduler.py` | 19498 | 459 | Tests for Stage 15 â€” Unified CPU/RAM/GPU Runtime Scheduler. |  |
| `tests/test_stage16_quality_profiles.py` | 24301 | 630 | Tests for Stage 16 â€” Hardware-Adaptive Quality Profiles. |  |
| `tests/test_stage17_adversarial_audit.py` | 30018 | 675 | Stage 17 â€” Final Adversarial Audit: test suite. |  |
| `tests/test_stage22_nle_vfx_provenance.py` | 13612 | 315 | Stage22: SMPTE timecode, NLE interchange, VFX multipass, cryptographic provenance |  |
| `tests/test_stage27_preflight_linter_flamegraph_workstation.py` | 6255 | 145 | Stage 27 Verification: Pre-Flight Render Linter, Bottleneck Flamegraph, and Workspace Layouts. |  |
| `tests/test_stage2_model_lifecycle.py` | 10313 | 295 | Tests for Stage 2: Model Lifecycle and Runtime Architecture. |  |
| `tests/test_stage3_adaptive_detector.py` | 12484 | 311 | Tests for Stage 3: Adaptive SCRFD Detector Optimization and Validation Suite. |  |
| `tests/test_stage4_geometric_alignment.py` | 10858 | 296 | Unit and Integration Tests for Stage 4 â€” Face Alignment and Geometric Stability Engine. |  |
| `tests/test_stage5_hyperswap_audit.py` | 7423 | 192 | Unit and Integration Tests for Stage 5 â€” HyperSwap Quality and Performance Audit. |  |
| `tests/test_stage6_restore_ultra.py` | 14168 | 310 | Unit and integration tests for Stage 6 â€” Restore Ultra Quality and Throughput Optimizer. |  |
| `tests/test_stage7_mouth_occlusion_governor.py` | 19786 | 501 | Unit and integration test suite for Stage 7: Surgical Mouth Dynamics, |  |
| `tests/test_stage7_realityux_governor.py` | 12916 | 336 | Comprehensive test suite for RealityUX Mask Engine Stage 7 Integration: |  |
| `tests/test_stage7_ultramax_hook.py` | 9127 | 188 | Stage 7 in the UltraMax enhancer hook: the mouth must be restored WHERE THE MOUTH IS. |  |
| `tests/test_stage7_xseg3.py` | 15290 | 386 | Tests for Stage 7 â€” XSeg 3 Mask Quality and Performance Optimization. |  |
| `tests/test_stage8_compositing.py` | 14239 | 288 | Tests for Stage 8 â€” Compositing Quality Engine. |  |
| `tests/test_stage9_temporal_tracking.py` | 16856 | 388 | Tests for Stage 9 â€” Temporal Face Tracking and Flicker Control. |  |
| `tests/test_stereo_epipolar_disparity.py` | 13777 | 312 | Stage 21: Stereoscopic 3D (SBS/TB) Epipolar Disparity Lock & VR Viewport Tests. |  |
| `tests/test_suite_layout.py` | 3247 | 74 | The two test suites must never share a module basename. |  |
| `tests/test_trt_graph_surgery.py` | 5266 | 111 | fake TensorRT network tests for graph surgery | _FakeLayer,_FakeNetwork,_FakeTrt,GraphSurgeryTests |
| `tests/test_trt_precision_surgeon.py` | 13805 | 288 | Unit and integration tests for app.core.trt_precision_surgeon. |  |
| `tests/test_trt_quant.py` | 21876 | 393 | roop.trt_quant: tier selection, stratified calibration, cache validation, Q/DQ. |  |
| `tests/test_ultramax_e2e.py` | 27970 | 659 | Stage 5 End-to-End Validation Suite for UltraMax Enhancer. |  |

### tools (69 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `tools/_synthetic_inputs.py` | 2266 | 43 | Shared declaration for the benchmark tools whose inputs are generated rather than recorded. |  |
| `tools/ab_tracking_prepass.py` | 5416 | 143 | Run the REAL tracking pre-pass over a clip, with shot awareness on and off, |  |
| `tools/acceptance_real_clip.py` | 35109 | 619 | Evidence-driven acceptance run for the discovered target_person clip. |  |
| `tools/acceptance_stage15.py` | 36642 | 581 | Stage 15 real-file acceptance: <MEDIA_DIR>/target_clip.mp4 x the user's person_a.fsz. |  |
| `tools/analyze_telemetry.py` | 2243 | 40 | Analyze Telemetry CSV and emit summary table. |  |
| `tools/apply_browser_video_updates.py` | 30648 | 723 | Apply browser-compliant video rendering flags across roop-ultimate. |  |
| `tools/apply_capturer_seeking_fix.py` | 5003 | 110 | Add frame is not None check and temporary scrub keyframe extraction fallback to capturer.py. |  |
| `tools/apply_ffmpeg_robustness.py` | 19776 | 490 | Apply FFmpeg robustness fixes: |  |
| `tools/apply_handles_and_fps_fixes.py` | 15868 | 400 | Apply OpenCV lifecycle hardening (try/finally release) and fractional FPS preservation. |  |
| `tools/apply_react_enhancements.py` | 8092 | 181 | Apply React UI enhancements: OutputVideoPlayer, base64 data prefix normalization, and ObjectURL lifecycle. | update_object_urls,update_frame_decoder,update_faceswap_player,update_processing_player,update_extras_player |
| `tools/apply_server_enhancements.py` | 18747 | 419 | Apply server static files, HTTP 206 byte-range streaming, web URLs, and CORS updates. |  |
| `tools/audit_system.py` | 56499 | 1124 | Production audit of the face_engine GPU render: hardware, memory, detection, restore, throughput. |  |
| `tools/bench_codeformer_batch.py` | 8247 | 160 | Does batching CodeFormer fp16 (UltraMax's network) make it cheaper per face?  (Stage 3 contract probe.) |  |
| `tools/bench_mask_gpu_morphology.py` | 5241 | 120 | cv2 vs torch-CUDA for the mask feather chain: speed and exactness (Stage 7 probe). |  |
| `tools/bench_restorer_batch.py` | 18289 | 401 | Does batching a 512 restorer make it cheaper per face?  (Stage 6 contract probe.) |  |
| `tools/benchmark_adaptive_detector.py` | 15072 | 370 | Stage 3 Benchmark: SCRFD Face Detector Optimization. |  |
| `tools/benchmark_alignment_stability.py` | 15693 | 388 | Stage 4 Benchmark: Face Alignment and Geometric Stability. |  |
| `tools/benchmark_compositing.py` | 22347 | 481 | Stage 8 â€” Compositing Quality Engine Benchmark Harness. |  |
| `tools/benchmark_hyperswap_audit.py` | 11606 | 198 | Stage 5 benchmark: HyperSwap / InSwapper / RealSwap quality and performance audit - rebuilt on tools/qualit | availability,source_cache_microbenchmark,table,main |
| `tools/benchmark_restore_ultra.py` | 17748 | 411 | Stage 6 â€” Restore Ultra Quality and Throughput Benchmark Harness. |  |
| `tools/benchmark_swapper.py` | 16523 | 341 | Swap-model inference benchmark: latency, VRAM and ACCURACY per execution path. |  |
| `tools/benchmark_temporal_tracking.py` | 24558 | 571 | Stage 9 â€” Temporal Face Tracking and Flicker Control Benchmark Harness. |  |
| `tools/benchmark_xseg3.py` | 19805 | 402 | Stage 7 â€” XSeg 3 Mask Quality and Performance Benchmark Harness. |  |
| `tools/build_calibration_set.py` | 11960 | 258 | Harvest the INT8/FP8 calibration set for the swap network from real renders. |  |
| `tools/build_trt_engines.py` | 29184 | 663 | Download the offline model set and pre-build TensorRT engine caches. |  |
| `tools/compile_engines.py` | 4801 | 102 | Compile face_engine's ONNX models into TensorRT engines ahead of time. |  |
| `tools/convert_dynamic_shapes.py` | 22410 | 473 | Give static batch-1 ONNX face models a dynamic batch axis, and PROVE it works. |  |
| `tools/convert_fp16.py` | 8160 | 145 | Convert the three auxiliary ONNX models (xseg, 2d106det, 1k3d68) to half precision, with per-strategy FP32 | out_path,_io,_force_fp32_io,convert,main |
| `tools/cut_threshold_probe.py` | 5486 | 145 | Find where the real hard cuts are in a clip, and what the shipped |  |
| `tools/derive_eye_ear_order.py` | 7301 | 199 | Derive a correct 6-point EAR ordering for the InsightFace 106 eye contours. |  |
| `tools/diagnose_trt.py` | 3849 | 93 | surface the LoadLibrary/version-mismatch error TensorRT EP swallows (93 lines) |  |
| `tools/download_scrfd.py` | 1621 | 41 | Download and SHA256-verify SCRFD-10G (``scrfd_10g_bnkps.onnx``) into ``<repo>/.cache/models``. |  |
| `tools/fit_identity_directions.py` | 13861 | 317 | Fit the ArcFace attribute directions shipped in roop/assets/identity_directions.npz. |  |
| `tools/gen_hud_fixture.py` | 4123 | 93 | Regenerate react-ui/.render-check/pipeline-hud.fixture.json from the REAL telemetry writers. |  |
| `tools/gen_jitter_fixture.py` | 6476 | 154 | Regenerate app/tests/data/synthetic_face_jitter.npz and print the smoother comparison. |  |
| `tools/gen_settings.py` | 7162 | 168 | Render the generated settings files from app/settings.py. |  |
| `tools/gen_update_manifest.py` | 10618 | 231 | Render update_manifest.json from the committed dependency/runtime contract. |  |
| `tools/live_monitor.py` | 2208 | 69 | Live monitor for the running roop-ultimate job. |  |
| `tools/measure_canary_cosine.py` | 5245 | 118 | Scratch: cosine / SSIM of the production TRT engines vs a CPU FP32 reference, plus ms/call. |  |
| `tools/measure_gpen_canary_controls.py` | 2262 | 39 | cosine of GPEN canary controls (39 lines, no docstring) | cos |
| `tools/phase14-after-render.ps1` | 1100 | 34 | PowerShell post-render check for phase 14 |  |
| `tools/prebuild_engines.py` | 12453 | 264 | Pre-build the TensorRT engines the APP will actually load. |  |
| `tools/probe_eye_indices.py` | 6007 | 160 | Are the eye landmark index groups in temporal_expression correct, and in |  |
| `tools/probe_landmark_indices.py` | 5476 | 153 | Which indices of insightface's 106-point landmark set are the mouth? |  |
| `tools/probe_mouth_indices.py` | 6784 | 179 | Do the mouth landmark index pairs in temporal_expression actually measure |  |
| `tools/probe_telemetry_latency.py` | 7842 | 180 | WebSocket telemetry latency and backend memory across real renders, through the live API. |  |
| `tools/profile_cv2_mask_ops.py` | 4036 | 107 | Which cv2 blur / erode / dilate calls does a real render make, on what, at what cost? |  |
| `tools/provider_parity_probe.py` | 8361 | 132 | Small provider-change arm used by the actual acceptance report. |  |
| `tools/quality_harness.py` | 65260 | 1138 | Quality harness: how far is a candidate configuration from a full-FP32 reference render, measured on real f | box_iou,detection_recall,cosine,_luma,ssim_map |
| `tools/repair_venv_paths.py` | 11446 | 291 | Repair a virtual environment that has been MOVED to a new folder. |  |
| `tools/restorer_composite_eval.py` | 6933 | 134 | What do the restorer's FP16 / island variants cost on the FINAL composited output, not on the 512 crop? |  |
| `tools/restorer_fp16_ranking.py` | 9926 | 231 | Which layers of the 512 restorer (RestoreFormer++, the network under Restore Ultra) cost the FP16 path its | _init_app,module_of,load_model,inject,session |
| `tools/restorer_native_engine.py` | 10974 | 197 | Build native TensorRT engines of the 512 restorer with chosen FP32 islands inside an FP16 network. |  |
| `tools/run_bench_arm.py` | 4754 | 126 | Bench Arm Runner with Telemetry Sidecar. |  |
| `tools/scene_cut_audit.py` | 6033 | 153 | Measure scene cuts in the source clip and score the tracking pre-pass |  |
| `tools/telemetry_sidecar.py` | 5466 | 121 | Standalone Telemetry Sidecar for Roop Sustained Throughput Profiling. |  |
| `tools/terminal_monitor.py` | 15402 | 353 | Terminal Progress Monitor for Roop Ultimate. |  |
| `tools/test_backend_server_endpoints.py` | 5169 | 107 | Test and verify backend server static serving, HTTP 206 range requests, web URLs, and CORS. |  |
| `tools/test_ffmpeg_robustness.py` | 5136 | 112 | Test FFmpeg odd dimension padding, silent video protection, and error reporting. |  |
| `tools/test_fractional_fps.py` | 3912 | 95 | Test fractional FPS extraction and OpenCV handle cleanup. |  |
| `tools/test_preview_endpoints_and_seeking.py` | 3297 | 83 | Test preview endpoints, accurate seeking, and fallback extraction. |  |
| `tools/trace_pipeline_frame.py` | 24894 | 566 | MicroInstrumentation per-frame stage trace + ASCII flamegraph (feeds trace_frame_analysis.json) |  |
| `tools/verify_audit_improvements.py` | 9396 | 177 | Verification script proving that the audit fixes improved runtime behavior |  |
| `tools/verify_browser_video_compliance.py` | 11258 | 241 | Comprehensive test script to verify browser-compliant video rendering. |  |
| `tools/verify_cut_detector.py` | 3479 | 103 | Run the SHIPPED StreamingStabilizationHistory over a real clip and report |  |
| `tools/verify_expression_measure.py` | 6090 | 151 | Does the SHIPPED measure_expression actually track mouth opening? |  |
| `tools/verify_eye_ear_symmetry.py` | 4593 | 127 | Confirm a symmetric, correct EAR ordering for both eyes. |  |
| `tools/verify_model_lifecycle.py` | 6181 | 163 | Stage 2 Verification: Model Lifecycle and Runtime Architecture. |  |
| `tools/verify_react_components.py` | 2939 | 72 | Automated verification script testing the React UI preview and video player logic. |  |

### web_ui (24 files)

| file | bytes | lines | purpose (from header) | key symbols |
|---|---:|---:|---|---|
| `web_ui/.gitignore` | 46 | 4 | ignore node_modules/dist |  |
| `web_ui/README.md` | 1964 | 33 | web_ui: face_engine control UI (not the app UI) |  |
| `web_ui/index.html` | 547 | 13 | Vite entry HTML |  |
| `web_ui/package-lock.json` | 132576 | 3839 | npm lockfile |  |
| `web_ui/package.json` | 840 | 33 | face-engine-ui: React 18.3.1, Vite 6.3.6, Tailwind 4.1.14, vitest 3.2.4 |  |
| `web_ui/src/App.tsx` | 9333 | 261 | app shell: project load -> face assignment -> params -> live preview/render |  |
| `web_ui/src/components/DualCanvasPlayer.tsx` | 11608 | 349 | source/result dual-canvas scrubber player |  |
| `web_ui/src/components/ErrorBoundary.tsx` | 1362 | 46 | React error boundary |  |
| `web_ui/src/components/FaceSelectorGrid.tsx` | 6036 | 133 | detected-face grid + person assignment |  |
| `web_ui/src/components/ParameterSliders.tsx` | 10803 | 327 | RenderParams controls / presets |  |
| `web_ui/src/components/ProjectLoader.tsx` | 1943 | 52 | source/target upload form |  |
| `web_ui/src/components/TelemetryHUD.tsx` | 8315 | 215 | GPU/job telemetry HUD |  |
| `web_ui/src/components/components.test.tsx` | 13986 | 338 | vitest component tests (338 lines) |  |
| `web_ui/src/hooks/usePreview.test.tsx` | 2287 | 59 | vitest for usePreview |  |
| `web_ui/src/hooks/usePreview.ts` | 3606 | 111 | debounced live preview POST /api/preview/frame |  |
| `web_ui/src/hooks/useTelemetry.ts` | 2694 | 89 | WS /ws/telemetry subscriber, backoff 1s->10s |  |
| `web_ui/src/index.css` | 127 | 6 | tailwind import (6 lines) |  |
| `web_ui/src/lib/api.test.ts` | 2375 | 52 | vitest for api client |  |
| `web_ui/src/lib/api.ts` | 4462 | 141 | typed REST client + /ws/telemetry URL |  |
| `web_ui/src/lib/types.ts` | 5160 | 186 | wire types mirroring server RenderParams/Job/Telemetry |  |
| `web_ui/src/main.tsx` | 450 | 16 | React root mount |  |
| `web_ui/src/test-setup.ts` | 44 | 1 | jest-dom setup (1 line) |  |
| `web_ui/tsconfig.json` | 560 | 19 | TS config |  |
| `web_ui/vite.config.ts` | 879 | 27 | Vite config proxying to FastAPI 127.0.0.1:8765 |  |

## 5. Test and tool group notes (what is covered)

- **`tests/` (56 files, root integration suite)**: Stage-numbered suites 2-27 (`test_stage*`), benchmark harness tests, TRT engine build/precision-surgeon/quant tests (`test_build_trt_engines` 19 tests, `test_prebuild_engines` 18, `test_trt_quant`, `test_trt_precision_surgeon`), fused post-process kernel (`test_fused_postprocess` 23 + fallback 10), RetinaFace closeup (`test_retinaface_closeup`), angle tests (`test_angles` 11: omnidirectional rectification + profile-aware weighted Umeyama on the app side), occlusion interaction, stereo epipolar (`test_stereo_epipolar_disparity`), GPU bindings (10). Marked GPU tests skipped in CI (`ci.yml:100-101` `-m "not gpu"`).
- **`face_engine/tests/` (27 files)**: real GPU tests (`test_pipeline` 498 lines, `test_stage3_gpu`, `test_cuda_streams`, `test_video_hw`, `test_trt_compiler`) plus `bench_*`/`eval_*`/`verify_*` scripts and `test_web_ui_e2e` (headless Chromium against real server). `test_modern_stages.py` (4 tests) covers the "modernised" files `core/base_processor.py`, `core/zero_copy_engine.py`, `processors/modern_processors.py`, `pipeline/hybrid_masker.py` (thin).
- **`tools/` (69 top-level files)**: (a) benchmark harnesses per stage (`benchmark_*`, `bench_*`, `quality_harness.py` 1138 lines: SSIM/PSNR/mask-IoU/ArcFace cosine vs FP32 reference); (b) TRT build/convert (`build_trt_engines`, `compile_engines`, `prebuild_engines`, `convert_fp16`, `convert_dynamic_shapes`, `restorer_native_engine`, `diagnose_trt`); (c) landmark/eye/mouth index probes (`probe_*`, `derive_eye_ear_order`, `verify_eye_ear_symmetry`); (d) cut-detector probes (`cut_threshold_probe`, `scene_cut_audit`, `verify_cut_detector`, `ab_tracking_prepass`); (e) acceptance runs through live API (`acceptance_stage15`, `acceptance_real_clip`); (f) telemetry/terminal (`terminal_monitor.py` 353 lines, `live_monitor`, `telemetry_sidecar`, `probe_telemetry_latency`, `analyze_telemetry`); (g) generators (`gen_settings`, `gen_update_manifest`, `gen_jitter_fixture`, `gen_hud_fixture`, `fit_identity_directions`); (h) one-shot patchers `apply_*.py` (5 files, string-replace source edits).
- **`scripts/`**: Pinokio `clean.js`/`fix_tensorrt.js`, `cleanup.py` (disk cleanup of TRT caches/old logs/uploads), `profile_pipeline.py`, `verify_roop_keep.py` (corpus verifier), `verification/check_api_mock_parity.py` (CI contract check), `verify_pipeline.py`, `verify_integration.py` (Stage 19).
- **Root launchers** (`pinokio.js` 195 lines; `install.js`, `start.js`->`start_react.js`, `start_legacy.js` (DEPRECATED Gradio), `update.js`, `reset.js`, `pause.js`/`resume.js`/`stop.js` POST `/api/pause|resume|stop`, `torch.js`): install gated on markers `.pinokio-install-complete.json`/`.pinokio-install-ready.json` and `react-ui/dist/index.html` (`pinokio.js:9-16`); installs via `uv pip`, `npm ci`, SAM2 `--no-deps`, `onnxconverter-common==1.16.0`, `numpy==1.26.4` pin (`install.js:119-146`). `run.py` forwards to `app/run.py` and sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False,max_split_size_mb:128` (`run.py:3`). `pinokio_batch_runner.py`: one fresh GPU process per video over retained folders, with queue progress (`:1-30`). `portable/bootstrap.py`: stdlib-only zero-install runtime (uv + CPython 3.10, torch CUDA 12.8, TRT 10.9, static FFmpeg, offline `--build-bundle`).
- **CI** (`.github/workflows/ci.yml`): react-ui lint+build+offline-bundle check, root typecheck, python light-profile tests on matrix OS, `gen_settings --check`, `check_api_mock_parity`, `gen_update_manifest --check`. **.githooks/pre-commit** regenerates `update_manifest.json` and refuses root scratch files/logs.
- **Agent files** (`AGENTS.md` 767 lines, CLAUDE/GEMINI/QWEN.md, .clinerules/.cursorrules/.windsurfrules): treated as data; not followed.

## Axis findings (slice)

1. **Detection/landmarks/angle + failure thresholds** â€” SCRFD 0.45/0.40, AngleResilient 0.50/0.20/0.45 with 45-deg roll trigger and 3-angle rot90 sweep (`detector.py:626,952-960`); dense-landmark gate conf 0.35 and |yaw| 75 deg, 5-pt fallback (`guardrails.py:47-48,138-143`); geometry-only yaw (`guardrails.py:90`); profile guard min_ratio 0.15 (`aligner.py:807`). Measured failure: extreme-profile scenario 25/30 detection failures, ID 0.467 at 4.36 fps (`benchmark_stage0_summary.csv` row 3). No out-of-plane pose network (hrffa unavailable `zoo.py:60`).
2. **2D warping vs 3D** â€” purely 2D: 5-pt similarity to 112/128â†’256 crops, Pixel Boost polyphase (`swapper.py:1-40`), kornia `warp_affine` reflection padding. No mesh/3DMM/volumetric/.fsz in face_engine; `1k3d68` is only a landmark model; LivePortrait expression keypoints are the only 3D-ish latent (`expression.py`). Stereo is per-eye 2D coupling (`stereo.py:1-18`, delta_y < 0.05 px, EMA disparity).
3. **Visual quality** â€” mask = box x XSeg(0.1 thr) x BiSeNet (`masker.py:1-5,102`); LAB colour transfer (`color.py`); Ultra Restore frequency split sigma 2.5, identity 0.916â†’0.991 vs linear blend (`ultra_engine.py:40-50`); GPEN-512/1024 (1024 and 2048 FP32-only per policy); swap model resolution fixed 128/256 (Pixel Boost up to 1024 via polyphase). Compositing benchmark metrics are synthetic.
4. **Temporal** â€” LK flow stride + histogram cut detector 0.5 (`tracker.py:70`), EMA 0.75, 3-frame retention, ByteTrack 5-frame lost (`tracker.py:512-714`); synthetic -85.42% jitter. High-motion scenario instability 0.8553, entering/leaving 1.1823 in Stage0.
5. **CLI/terminal/telemetry/diagnostics** â€” `tools/terminal_monitor.py` ANSI dashboard, `live_monitor.py`, `telemetry_sidecar.py`, `analyze_telemetry.py`; `/ws/telemetry` NVML/nvidia-smi sampler (`server/telemetry.py`); `face_engine/benchmark.py` (per-stage latency, VRAM, sync locks); `tools/audit_system.py` (1124 lines production audit); `tools/diagnose_trt.py`; `tools/trace_pipeline_frame.py` flamegraph.
6. **Compute** â€” TRTâ†’CUDAâ†’CPU default with strict-fail option (`config.py:49,130`); AOT engines with sidecars; FP32 locks for swap nets, XSeg, landmarks, GPEN-1024+ (`PRECISION_POLICY.md`, `trt_compiler.py:269`); INT8/FP8 rejected on Ada; 90% torch VRAM cap (`ultra_pipeline.py`); VRAM saturates ~12.1 GB in Stage0 (4070); RTX 3060 validation pending everywhere; NVDEC disabled in-process (`cuda_streams.py:8`).
7. **Job queue/checkpoint/video IO/FFmpeg** â€” single job, thread + cancel event (`state.py:113-126`); GOP-segment worker pool spawn (`worker_pool.py`); libx264 crf18 +faststart; NVENC; lossless demux/remux; A/V sync check; no persisted queue/checkpoint in face_engine; batch runner `pinokio_batch_runner.py` (fresh process per video); pause/resume only via app API.
8. **Workstation UI** â€” `web_ui/` React 18 + Vite 6 + Tailwind 4: project loader, face grid, parameter sliders, DualCanvasPlayer, live preview hook, telemetry HUD (24 files; `web_ui/README.md`). Separate from app `react-ui/` (slice other). Pinokio menu launcher `pinokio.js` + `portable/`.

## Gaps / stubs / defects found

- **No 3D volumetric / faceset path in this repo slice**; the reference is a 2D swapper stack; `.fsz` is only consumed by the app API in `tools/acceptance_stage15.py`.
- **Library-only modules, no production caller** in non-test face_engine: `AngleResilientSCRFD`, `RobustByteTracker`, `ProfileGuardedAligner`, `choose_alignment_points`/`estimate_yaw_5pt`, `EpipolarStereoCoupler`/stereo, `HybridVideoMasker`, `provenance`, `UltraRestorer`, `zero_copy_engine`, `modern_processors` (grep evidence, section 1.2). The server renders with `GPUSCRFDDetector` + Strided/Temporal trackers.
- Model zoo gaps: `hrffa` and `alphaface_256` registered unavailable (`zoo.py:60,186`), yet `web_ui/src/lib/types.ts` and `RenderParams` still expose `alphaface_256` as a selectable swapper.
- Yaw estimator is geometry-only and saturates on collapsed landmarks (`guardrails.py:97`); profile guard changes scale only (`aligner.py:727-760` docstring).
- SCRFD ByteTrack `match_iou_high` default 0.20 vs ByteTrack's 0.8-as-distance reference: documented, not tested against real occlusion/crossing footage (Stage9 crossing results synthetic, naive == robust 100%).
- Benchmarks: Stage 3/4/8/9 baselines are synthetic; Stage0 scenarios 8-13 run at 0.33-1.2 fps (Multiple Faces Interacting 0.33 fps, ID 0.4238) â€” quality/perf floor for hard cases; Stage 5 file documented as previously INVALID (typed-in quality values).
- Baseline pending: RTX 3060 validation (OPTIMIZATION_PLAN.md CURRENT GATE, PRECISION_POLICY INT8 tier, CUDA policy), `benchmark_pipeline.py` default clip missing (`PERFORMANCE_BASELINE.md` not-run list).
- `tools/apply_*.py` (5) are one-shot string-replace patchers of other source files (`apply_browser_video_updates.py` 723 lines, etc.) kept in tree: drift/hazard if re-run.
- Single in-memory job in `face_engine/server/state.py`: no queue, no checkpoint/resume, no persistence across restart.
- Hard-coded TRT profile in `zero_copy_engine.py` (min 256 / max 512 shapes, `:1-20`) â€” baseline-only vs the AOT `trt_compiler` path.
- 4K scenario throughput 0.64 fps and 1080p 0.5 fps in Stage0 vs 56 fps claim in presets: different config (Stage0 full-quality app pipeline vs face_engine TRT presets); not directly comparable [INFERENCE].
- Three agent-rule stub files and AGENTS.md at root carry directives; ignored per audit rules.

## Coverage

Slice files in inventory: 309 | files accounted for: 309 | omitted: none
(verified: `awk` over `roop_ultimate_files.tsv` for prefixes face_engine/ 88, tools/ 69, tests/ 56, web_ui/ 24, benchmarks/ 12, scripts/ 10, portable/ 6, .github/ 1, .githooks/ 1, root 42; table rows = 309.)
