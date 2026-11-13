# UI-01 — Roop Ultimate UI and Control Inventory

**Reference:** [`rishabh4496/roop-ultimate`](https://github.com/rishabh4496/roop-ultimate), `main` at `71ebb65c5fa5f11a72e343e8fc4f285bf591b555` (verified against GitHub's `main` commit endpoint during this audit). **Read-only.** No reference files were changed.

## Audit method and limits

This inventory combines the existing repository-level source inventory in [`01_ROOP_ULTIMATE_COMPLETE_AUDIT.md`](01_ROOP_ULTIMATE_COMPLETE_AUDIT.md) (which accounts for 1,422 files and 377 `react-ui/` files at the recorded revision), direct reads of upstream `app/settings.py` and `react-ui/src/components/settingsCatalog.js`, and source references recorded for the UI routes, controls and render paths. Existing per-file inventory identifies all frontend source files and their responsibilities; this document groups repeated controls by their owner component rather than duplicating 377 file rows. This is a static source audit, not a live-browser walkthrough or hardware benchmark.

The requested `rea` GitHub-source tool was not exposed in this session. The public repository was read through GitHub's raw-file endpoint and the current commit API instead. No claim is made that every upstream JSX line was re-opened in this run. Settings keys below are the full generated `UI_SETTINGS` registry, the authoritative catalog for Settings controls; live Face Swap controls are additionally mapped to their source modules below. Exact defaults, legal ranges, option enums, and environment mappings are maintained in upstream `app/settings.py` / generated `app/settings.schema.json`, and runtime values are further affected by user configuration and environment precedence. This report does not reconstruct those values from UI labels.

## Screens, work areas, and controls

The React UI has nine hash routes: **Home, Face Swap, Batch Matrix, Processing, Face Manager, Editor/Extras, Outputs/Gallery, History, Settings**. Face Swap and Processing are eager-loaded; other screens are lazy-loaded. The shell adds a persistent navigation/header, project controls, pipeline state and global command palette.

| Work area | Visible controls and boxes verified in source inventory | Primary modules |
|---|---|---|
| Home | Quick-launch source/target drop zones; resumable projects/checkpoints; current job and project access | `Home.jsx`, `home/QuickLaunch.jsx`, `home/ResumableCheckpoints.jsx`, `ProjectsPanel.jsx` |
| Face Swap source and identity | Source image/FaceSet library; `.fsz` import/export and video-to-FaceSet extraction; pose-coverage radar; target-face selection, per-person source mapping, identity pin/lock/exclusion roles; face similarity/identity blending, multi-source latent blend, profile/preset selection | `FaceSwap.jsx`, `FacesetLibrary.jsx`, `PersonGroups.jsx`, `IdentityBlender.jsx`, `facemgr/{PoseRadar,MorphLab}.jsx`, `VideoFramesetExtractor.jsx`, `faceMapping.js`, `faceRoles.js` |
| Face Swap preview and canvas tools | Original/result split or diagonal wipe; shared zoom/pan and loupe; compare grids (2-up/4-up/9-up); overlay/landmarks/pose HUD; mask brush and Bezier roto; parser-region toggles; box/face masks and linked padding; A/B reference and pop-out monitor; optional stereo VR180/VR360 viewport | `InteractivePreview.jsx`, `PreviewCanvas.jsx`, `CompareSlider.jsx`, `CompareGrid.jsx`, `QuadInspectionGrid.jsx`, `ParameterSweepModal.jsx`, `RotoLayer.jsx`, `ParserRegions.jsx`, `StereoVRViewport.jsx`, `ABReferencePanel.jsx`, `BroadcastScopesDock.jsx` |
| Face Swap processing controls | Swapper selection; restorer selection and stage enable/bypass; color transfer/grading; temporal mode; detection/identity/tracking; expression/eye/mouth behavior; mask/occlusion; lighting/harmonization; profiles and numeric slider tracker; frame/segment overrides and keyframes | `FaceSwap.jsx`, `PipelineControls.jsx`, `EnhancerPipelineStack.jsx`, `TemporalModeControl.jsx`, `Settings.jsx`, `EnhancerSettings.tsx`, `ColorGradingPanel.jsx`, `timeline/OverrideInspector.jsx`, `timeline/IdentityKeyframeInspector.jsx` |
| Timeline and editorial | Multi-track frame/timecode ruler, minimap, thumbnails, scene cuts, face presence/identity lanes, render-state strip, audio waveform, in/out trim, segment overrides, identity keyframes and curve interpolation, J/K/L shuttle | `Timeline.jsx`, `timeline/*`, `SegmentBar.jsx`, `SegmentMatrix.jsx`, `shuttle.js` |
| Face Manager and batch | `.fsz` pose radar, grouped identities and 2–5-input morph/mixer; video frame extraction; one-to-many/grouped/matrix/recipe batch matrix; segment jobs and contact-sheet/parameter sweeps | `FaceManager.jsx`, `facemgr/*`, `BatchSwap.jsx`, `batchMatrix.js`, `segments/*`, `CompareGrid.jsx` |
| Processing and diagnostics | Job progress, pause/resume/cancel; job queue/recovery/checkpoints; pipeline stages and TensorRT compile/canary state; stage timing/flamegraph; models in-use; terminal/log severity filters; runtime report (14 sections); quality report and preflight | `Processing.jsx`, `QueuePanel.jsx`, `ProcessingDock.jsx`, `ProcessingTerminal.jsx`, `RuntimeSections.jsx`, `DiagnosticsPanel.jsx`, `FrameFlamegraph.jsx`, `RunModelsPanel.jsx`, `CompileBanner.jsx`, `TrtCachePanel.jsx`, `PreFlightLinter.jsx` |
| Outputs and history | Before/after output player; finished-render comparison; output and project/run history; checkpoint-vs-current settings diff | `Gallery.jsx`, `OutputCompare.jsx`, `OutputVideoPlayer.jsx`, `RunHistory.jsx`, `history/CheckpointDiff.jsx` |
| Settings and shell | Searchable, changed-only Settings page; named sections; command palette/settings focus; themes; shortcuts/remapping; model hub/installer; environment and diagnostics views | `Settings.jsx`, `settingsCatalog.js`, `settingsTabs.js`, `CommandPalette.jsx`, `ThemeGallery.jsx`, `ThemeStudio.jsx`, model/diagnostic panels |

### Exact Settings catalog (all upstream `UI_SETTINGS` keys)

`app/settings.py:UI_SETTINGS` is the source of truth for labels and sections; `react-ui/src/components/settingsCatalog.js` is generated from it. The Settings screen exposes these keys, grouped by the source's section order:

| Section | Keys |
|---|---|
| Server | `server_share`, `clear_output`, `server_name`, `server_port`, `output_template`, `faceset_library_path` |
| Performance | `provider`, `trt_precision`, `swap_quantization`, `force_cpu`, `auto_thread_selection`, `face_detector_threshold`, `face_detector_nms`, `detector_scale_pyramid`, `max_threads`, `memory_limit` |
| 3-stage pipeline | `swap_model`, `selected_enhancer`, `auto_bypass_restorer_native_swap`, `prepass_detect_interval`, `temporal_smart_skip` |
| Advanced performance | `perf_trt_pool`, `trt_builder_optimization_level`, `trt_auxiliary_streams`, `trt_cuda_graph`, `cpu_opencv_threads`, `perf_detmask_pool`, `perf_detector_pool`, `perf_expr_pool`, `perf_encoder_preset`, `perf_nvdec`, `perf_batch_swap`, `perf_profile`, `cpu_ort_intra_threads`, `cpu_ort_inter_threads`, `cpu_ffmpeg_threads`, `perf_ort_arena_strategy`, `perf_cudnn_conv_algo`, `perf_gpu_mem_limit`, `vram_safety_margin_gb`, `perf_batch_max`, `perf_nvenc_preset`, `perf_gpu_affine`, `perf_pinned_buffers`, `temporal_step`, `enhance_min_face_px` |
| Identity & tracking | `recognizer`, `identity_confidence_threshold`, `face_demarcate`, `track_stitch`, `verify_swap`, `upright_remeasure`, `prepass_landmark_filter`, `process_priority` |
| Output | `output_image_format`, `output_video_format`, `output_video_codec`, `video_quality`, `hdr_pipeline`, `hdr_source_transfer`, `hdr_source_primaries`, `hdr_output_codec`, `use_os_temp_folder`, `output_show_video`, `synthetic_label`, `synthetic_watermark`, `synthetic_watermark_text`, `vfx_multi_pass`, `vfx_export_matte`, `vfx_export_crops`, `enable_provenance_signer`, `embed_provenance_watermark`, `provenance_disclosure` |
| Lighting & Harmonization | `light_harmonizer`, `light_harmonizer_key_intensity`, `light_harmonizer_ambient_bias`, `light_harmonizer_eye_specular`, `light_harmonizer_shadow_occlusion`, `specular_highlight_recovery`, `ambient_shadow_match`, `subdermal_warmth_match` |
| Surgical mouth & occlusion | `restore_original_mouth`, `mouth_open_threshold`, `enable_xseg_occlusion`, `occlusion_erosion` |

This is the complete generated panel key list at the audited commit. It is not a list of every field accepted by every backend endpoint: `ENV_SETTINGS` and runtime configuration include additional deployment/internal knobs, and some fields are intentionally not exposed as ordinary panel controls.

## Preview and rendering path

- Still previews use the local HTTP API (`POST /api/preview` and target-frame routes); the request is debounced/coalesced and aborts/supersedes older work. Preview may perform detection/face overlay and optional live swap. It is not equivalent to a queued final render.
- Playback uses a frame WebSocket (`/ws/frames`) with binary frame transport and a decoder worker/OffscreenCanvas path (`FastCanvasPlayer.jsx`, `fastPlayer.worker.js`, `frameSocket.js`, `frameDecoder.worker.ts`). Frame rendering uses an uncontrolled canvas rather than React image `src` state.
- `useRenderLite` mitigates redundant post-processing work; it does not prove that all sliders bypass detection, nor does it establish GPU-resident stage caching or a sub-16-ms preview bound.
- The preview, compare, grading/scopes and stereo paths use multiple rendering strategies: WebGL2, Canvas, worker decoding and ordinary React state. The upstream audit records a 3D LUT grading path and an equirectangular stereo viewer, but **no 3D face-mesh/volumetric identity viewport**.
- No measured 60–120 FPS guarantee or <16-ms inference/preview SLA was found in the audited sources. Existing frame-budget/e2e tests are measurement code, not a universal performance guarantee.

## Nonexistent or differently named requested controls

The user-requested TensorRT precision/provider settings partly overlap the real UI catalog. A generic 3DMM `alpha`/`beta` mesh editor, 9-band face-mesh relighting viewport, the proposed One-Euro/Kalman/optical-flow slider set, a universal 68/106-point 3D pose editor, or a 3D self-occlusion mesh panel must not be inferred from the presence of similarly named algorithms or diagnostics. Where a backend setting exists, its exact legal values/defaults come from `app/settings.py` and its schema—not a fabricated UI default. The actual `use_3d_recon` control requests a pose-matched source crop; it is not a 3D mesh viewport.

## Source pointers

- `app/settings.py` (`UI_SETTINGS`, `ENV_SETTINGS`, `LIVE_ENV_SETTINGS`)
- `app/settings.schema.json` (generated full field schema/defaults/ranges)
- [Generated settings schema at the audited commit](https://raw.githubusercontent.com/rishabh4496/roop-ultimate/71ebb65c5fa5f11a72e343e8fc4f285bf591b555/app/settings.schema.json) — exact defaults, field types, derived fields, UI labels, and environment declarations; numeric ranges/options are specified where present.
- `react-ui/src/components/settingsCatalog.js`, `Settings.jsx`, `settingsTabs.js`
- `react-ui/src/components/FaceSwap.jsx`, `Processing.jsx`, `faceswap/Timeline.jsx`, `faceswap/FastCanvasPlayer.jsx`, `faceswap/fastPlayer.worker.js`, `faceswap/frameDecoder.worker.ts`, `faceswap/transport/frameSocket.js`
- [`01_ROOP_ULTIMATE_COMPLETE_AUDIT.md`](01_ROOP_ULTIMATE_COMPLETE_AUDIT.md), RU_C (route/capability map and all-file frontend inventory)
- [`react-ui/src/components/faceswap/defaults.js`](https://github.com/rishabh4496/roop-ultimate/blob/71ebb65c5fa5f11a72e343e8fc4f285bf591b555/react-ui/src/components/faceswap/defaults.js) — full Face Swap reset-default map. The actual render controls are in `FaceSwap.jsx` and the referenced Face Swap components; defaults in this file are not by themselves a render-time configuration.
