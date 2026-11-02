# Stage 06 — Local model hub and engine architecture

## Technical implementation plan

1. Keep `models.registry.ModelRegistry` and `ModelCatalog` as the artifact authority. Supplement the preserved Stage 03/16 manifests with immutable, reviewed Stage 06 LoRA records; do not alter the preserved discovery baseline.
2. Add a read-only hub projection that exposes artifact identity, rights, hardware/dependency evidence, benchmark state and inference-path evidence as separate fields. Check the three production model digests when a user opens or refreshes the hub. Do not load inference sessions during API startup.
3. Describe the existing `ReferencePipeline` as the only production engine. List H3 and Qwen workflows as unavailable experiments until their full local execution and output contracts exist. Preserve FaceSet and existing job fingerprints while adding a defaulted engine_id dispatch field.
4. Add per-model download jobs with explicit license acknowledgement, progress, cancellation and resumable partial bytes. Only use immutable manifest URLs and publish an artifact after its registered size and SHA-256 pass.
5. Replace the active Settings model tab that used the obsolete registry and fake TensorRT build route with a view of the canonical model/engine catalog.
6. Add API, downloader, UI and regression coverage. Run the full Python suite, real local inference, production frontend build, and the read-only legacy integrity/isolation audit before calling the stage complete.

## Findings and architecture decisions

### Existing processing path

The video job path already uses `ModelCatalog`, `ReferencePipeline`, `FaceTracker`, `IdentitySwapper`, compositor and the existing audio/video muxer. It has per-track source assignment, frame processing, restoration hooks and source-audio preservation. The Stage 06 hub presents these pieces as one production ONNX engine; it does not replace detector, recognizer, swapper, tracker, compositor, or FaceSet storage.

The old `models/registry.py` at the application package root and the old Settings model tab were not authoritative. The Settings tab called `/models/compile_trt`, which wrote the literal bytes `TRT_ENGINE_CACHE_DATA_FOR_STAGE13` and returned `COMPILED`. The route now fails closed as unavailable and does not write an engine. The legacy batch-download route now requires explicit model IDs and uses the canonical catalog.

### Face alignment

The [face-alignment project](https://github.com/1adrianb/face-alignment) provides 2D/3D FAN landmarks and a separate face detector. Roop 3D already has dense landmarks and geometric pose estimation in its production path. The reference adds PyTorch and its model/runtime contract but offers no measured improvement over the current dense landmarks, pose, tracking or reconstruction. No adapter is added until a comparable real-image evaluation shows a benefit.

### InsightFace models

The production path remains bound to its currently registered SCRFD detector, ArcFace recognizer and INSwapper model. The hub hash-checks their exact local files and marks inference available only for the digests with existing real-inference coverage. The repository's code license is not treated as a pretrained-weight license: the [InsightFace model-zoo terms](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md) say the supplied pretrained models are for non-commercial research, and the [INSwapper notice](https://github.com/deepinsight/insightface/blob/master/python-package/docs/model_zoo.md) requires separate licensing contact. The locally observed model records still say `UNKNOWN` for exact weight provenance/license; the hub preserves that fact rather than substituting a code license. Official packs such as `buffalo_l`, `buffalo_m`, `buffalo_s` and `antelopev2` bundle alternative SCRFD/RetinaFace detection, ArcFace recognition and alignment components, but the weights inherit the research-only restriction and this stage adds no competing pack without a controlled quality/performance comparison.

Other InsightFace packs and recognizers are not activated. The project has no comparative image-quality, identity-preservation or hardware-specific benchmark evidence for a replacement.

### Deep-Live-Cam reference

[Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam) is AGPL-3.0; no source code is copied and its webcam implementation is not integrated. Its offline image/video workflow is useful as a feature checklist: explicit source/target pairing, multi-face mapping, frame iteration, restoration, encoding, and audio restoration from the source. Roop 3D already has local mapping and frame-job mechanisms, restoration hooks, and source-audio muxing. The license and implementation remain separate from the standalone codebase.

### MiniMax H3 Ref2VA

The [adapter card](https://huggingface.co/UntMods/FaceSwap_MiniMaxH3_REF2VA) identifies the adapter as Apache-2.0 and exposes the exact 65,623,904-byte safetensors artifact at immutable revision `b2a5823ca64bc78d91725fbfcc576095bfceb764`, SHA-256 `1e032cf519cc143f434e67516d8ad0aacf4c6e146315b2dcc3b1c2800470326d`. It is a LoRA for [MiniMax H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), not a standalone ONNX face swapper. The inspected shared graph uses `CRT_MiniMaxH3USConfig`, `CRT_MiniMaxH3USModelsPipe`, `CRT_MiniMaxH3UnifiedSampler`, `VHS_LoadVideo` and `VHS_VideoCombine`. It loads a 24-fps video with a 192-frame cap and writes H.264 MP4 at 24 fps. The Comfy-Org workflow references an INT8-convrot H3 Ref2VA diffusion file (~20.97 GB), a Qwen3-VL 32B NVFP4/AWQ text encoder (~15.69 GB), H3 video/audio VAE files and a separate H3 Turbo LoRA. The [official ComfyUI guide](https://blog.comfy.org/p/minimax-h3-day-0-support-in-comfyui) describes a 33B base and about 42.5 GB total model footprint with offloading. That is not a verified minimum VRAM figure and does not establish compatibility with either the RTX 3060 6 GB or RTX 4070 12 GB profile. The base model has its own MiniMax H3 Community License, independent of the Apache adapter.

The existing [ComfyUI bridge](../bridges/comfyui_bridge.py) submits local workflows and fetches PNG/JPEG images. It does not upload local video inputs or retrieve video outputs. The H3 workflow generates new video at 24 fps and includes its own generated audio; it does not satisfy the job pipeline's source-timing/audio-preserving replacement contract. Therefore the adapter is registered and can be explicitly downloaded, but the H3 engine remains `UNVERIFIED` and unavailable for production jobs. Implementing this as a runnable experimental video engine requires a separately reviewed local video input/output bridge and real tests with the exact base weights and nodes.

### BFS Qwen-Image face-swap LoRA

The [BFS card and files](https://huggingface.co/maximsobolev275/BFS-Qwen21-Best-Face-Swap-Fix/tree/2260bd560b8182b59a65238aed503ab5619b1d45) expose three LoRA variants. Their adapter repository says MIT; the base is [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1), whose current [Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) permits non-commercial purposes only unless a separate commercial license is obtained. The adapter and base rights are shown independently in the hub.

Qwen-Image is a large still-image editor. It may be useful for manually reviewed still/keyframe correction or identity consistency evaluation, but there is no Roop 3D Qwen workflow adapter or multi-frame consistency integration. It is not selected as a 25-FPS swapper and is not currently selectable for production jobs. VRAM minimum/recommendation and precision remain unknown until actual local hardware tests.

### Kling and LiveFaceSwap

The [Kling face-swap page](https://kling.ai/feature/face-swap) demonstrates multiple identity references and cross-angle/lighting consistency as quality criteria. Its proprietary cloud service is not an engine dependency and no internal model access is claimed. The [LiveFaceSwap page](https://livefaceswap.ai/) demonstrates reference selection and preview controls; the Settings hub uses ordinary local model/engine cards, state labels and explicit selection/download controls without reverse engineering its internals.

## Hub state and download semantics

Every registry entry exposes name/version/family/engine/purpose/source, weight filename/size/SHA, license, min/recommended VRAM, dependencies, precision, providers, benchmark state, actual adapter availability and one of the required model states. Unknown VRAM, precision, provenance and performance remain null or `NOT_RUN`; unknown values are not inferred from neighboring models.

Downloads only start after a per-model action. The reviewed Hugging Face URLs use immutable revisions; the downloader follows at most the observed `huggingface.co` to `us.aws.cdn.hf.co` redirect and rejects any other target. Cancellation leaves a resumable partial file and publishes no unverified final artifact. The downloader never extracts archives or imports model code. API startup and normal processing do not contact external model hosts.

The experimental LoRA cards can be downloaded with their adapter license acknowledgement, but downloading an adapter does not download or license its base model and does not make the engine inference-ready. The engine cards report capability and availability. The hub selection is captured as engine_id in each new job. EngineRegistry resolves it to the existing configured reference runner (including its optimized stream path), and the worker uses that runner for preparation, execution and cleanup. Old jobs default to reference_onnx; the default engine field is omitted from identity_payload to preserve existing idempotency keys and checkpoint fingerprints. Unknown and unavailable engines fail explicitly before inference. ReferencePipeline remains the only production engine; H3/Qwen are not selectable.

## Known limitations and stage gates

- No face-alignment integration: no measured benefit over existing dense landmarks.
- No replacement InsightFace model: license/provenance and comparative output evidence are incomplete.
- H3 is listed but not runnable through the current image-only ComfyUI bridge; base model/node installation and 6/12 GB profile compatibility are unverified.
- BFS is listed as a still/keyframe experiment; its Qwen base has separate non-commercial terms and no local adapter is wired.
- Kling remains a quality benchmark; no cloud calls or media uploads are made.
- The hub records benchmark state as `NOT_RUN`; it makes no performance claims.
- Engine selection and worker dispatch are implemented, but only the existing reference adapter is runnable. Additional application-owned adapters must implement the same local FaceSet/track job contract and be explicitly registered; H3/Qwen remain gated until their exact local dependencies and real inference outputs are verified. No alternative model or GPU performance claim is made.

## Final verification (2026-11-03)

- Full backend suite: 919 passed, 10 skipped, 18 subtests passed in 640.21 seconds. Existing Starlette/httpx deprecation warning remains. This suite includes the existing real local reference-model inference tests; no new H3/Qwen inference is claimed.
- Frontend: production typecheck/lint/Vite build passed (81 existing lint warnings, zero errors); 303 UI tests passed across 36 files.
- Earlier full run had one Windows temporary-directory cleanup failure (WinError 145) in Stage 00. Its complete test file passed on rerun, and the final full suite passed on unchanged source.
- Read-only legacy audit: 2,645 historical and 68,390 comprehensive protected hashes matched, zero violations. 1,566 permitted runtime files were classified separately; no source/model baseline was replaced.
- Independent read-only GPT-6.1 Sol high review: no new P1/P2 findings. Selection reaches worker dispatch; unknown/gated engines fail without fallback. Future adapters require verified models and their own preview integration.
- git diff --check passed. No model weights, media, FaceSets, embeddings, environments or generated frontend build files are included in the commit.

## Changed files

- `app/roop_3d/docs/STAGE06_MODEL_HUB.md`
- `app/roop_3d/engine/registry.py`
- `app/roop_3d/jobs/schema.py`
- `app/roop_3d/jobs/worker.py`
- `app/roop_3d/models/downloader.py`
- `app/roop_3d/models/downloads.py`
- `app/roop_3d/models/hub.py`
- `app/roop_3d/models/registry.py`
- `app/roop_3d/models/stage6_models.json`
- `app/roop_3d/server/app.py`
- `app/roop_3d/server/live_preview.py`
- `app/roop_3d/server/routes.py`
- `app/roop_3d/server/workstation_preview.py`
- `app/roop_3d/tests/test_model_registry.py`
- `app/roop_3d/tests/test_stage03_api.py`
- `app/roop_3d/tests/test_stage03_downloader.py`
- `app/roop_3d/tests/test_stage06_engine_selection.py`
- `app/roop_3d/tests/test_stage06_model_hub.py`
- `app/roop_3d_ui/src/App.tsx`
- `app/roop_3d_ui/src/components/modelhub/ModelHubPanel.tsx`
- `app/roop_3d_ui/src/contracts.ts`
- `app/roop_3d_ui/src/livePreview.ts`
- `app/roop_3d_ui/src/screens/BatchMatrixScreen.jsx`
- `app/roop_3d_ui/src/screens/SettingsScreen.jsx`
- `app/roop_3d_ui/src/stage13ModelHub.test.tsx`
- `app/roop_3d_ui/src/stores/engineSelection.ts`
