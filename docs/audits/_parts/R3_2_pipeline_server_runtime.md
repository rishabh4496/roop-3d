# R3_2 â€” roop-3d slice 2: pipeline, server & runtime (code audit)

Scope: `app/roop_3d/{compositor,color,enhancers,pipeline,media,io,execution,core,async_pipeline,models,orchestrator,server,jobs,queue,reliability,telemetry,security,installer,bridges,editing,evaluation,tools,docs}`, top-level `app/roop_3d/*` not owned by slices 1/3 (pose_matrix, angle_tracker, relighting_3d, temporal*, faceset_* excluded as slice 1/3 by assignment), `app/roop/`, `app/core/`, `app/diagnostics/`, `app/tests/`, `app/api.py`, `app/serve.py`. Static reading only; no tests/builds run. Line counts via AST/`count('\n')`; symbol:line from `ast` top-level defs. `[INFERENCE]` = not verified by execution.

Status legend: real = functional code; stub = intentional placeholder; baseline = only a conventional/simple implementation; config/manifest/doc = non-code.

## 1. Executive map

- **Entry**: `app/serve.py:1-28` (UTF-8/unbuffered HUD env, `core/trt_bootstrap.py` DLL registration, `core/dll_injection.setup_environment`, then `roop_3d.server.__main__.main`) and `app/api.py:41-62` (`create_unified_app` -> `server/app.py:create_app`, module-level `app`).
- **Server**: FastAPI + uvicorn (`ws="websockets-sansio"`, `ws_max_size=16384`, `ws_max_queue=16`, `server/__main__.py:117`); 202 decorated REST endpoints in `server/*.py` (grep `@router.get|post|put|delete|patch`; workstation.py 30, routes.py 49, queue_routes.py 17, unified_state.py 13) + 5 WS routes (`server/app.py:322-326`, `server/routes.py:728-729`).
- **Render core**: `compositor/pipeline.py` (`FrameCompositor`, 1297 lines) is the real compositor; `pipeline/photorealistic_swap_core.py` + `pipeline/video_job_runner.py` is the Stage-17 FFmpeg-piped e2e runner; `optimized_processor.StreamingProcessor` is the 3-stage streaming coordinator; `job_queue.SegmentedRenderRunner` + `queue_manager.QueueManager` + `jobs/*` are the persistence/worker layer.
- **Compute**: `trt_session_builder.py` (strict lazy TRT, FP16/FP32/FP8-gated), `execution/trt_mixed_precision.py` (FP32-pinned islands, provider failover), `core/ort_session.py` (TRT-first pooled sessions + IO binding), `execution/zero_copy_pipeline.py`, `async_pipeline/*` (pinned pools, per-stage CUDA streams).

## 2. Per-module tables (every file in slice)

### `app/roop_3d/compositor/` (23 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/compositor/__init__.py` | 19 | Local, full-frame face replacement compositing primitives. |  | real | app/roop_3d/compositor/__init__.py:1 |
| `app/roop_3d/compositor/boundary_fusion.py` | 149 | Finite-safe crop warping and feather, multi-band or Poisson boundary fusion. | `BlendMode`:12, `warp_crop`:18, `_bounds`:41, `_pyramid_blend`:50, `fuse_crop`:77, `fuse_warped`:93 | real | app/roop_3d/compositor/boundary_fusion.py:12 |
| `app/roop_3d/compositor/color_match.py` | 511 | Masked LAB statistics and histogram color transfer. | `ColorStats`:13, `_validate_pair`:30, `_lab`:39, `_ycbcr`:46, `estimate_lab_stats`:53, `apply_lab_stats`:80 | real | app/roop_3d/compositor/color_match.py:13 |
| `app/roop_3d/compositor/detail_temporal.py` | 437 | Causal, confidence-gated stabilization of restoration-only high-frequency detail. | `_DetailState`:19, `TemporalDetailStabilizer`:29 | real | app/roop_3d/compositor/detail_temporal.py:19 |
| `app/roop_3d/compositor/face_regions.py` | 116 | Named face regions from the local BiSeNet 19-class parser. | `FaceRegions`:44, `build_regions`:71, `teeth_estimate`:82, `restoration_protection`:102 | real | app/roop_3d/compositor/face_regions.py:44 |
| `app/roop_3d/compositor/hairline.py` | 95 | Forehead / hairline boundary offset for the face matte. | `hairline_weight`:18, `offset_hairline`:27, `refine_hairline_matte`:43 | real | app/roop_3d/compositor/hairline.py:18 |
| `app/roop_3d/compositor/lighting.py` | 200 | Luminance-chrominance decoupled spherical-harmonic relighting. | `_basis`:10, `relight_spherical_harmonics`:17, `project_submental_contact_shadow`:93 | real | app/roop_3d/compositor/lighting.py:10 |
| `app/roop_3d/compositor/masks.py` | 563 | Semantic face masks with a deterministic landmark-hull fallback. | `extract_forbidden_mask`:41, `compute_dual_bisenet_forbidden_mask`:55, `turn_ratio`:94, `chin_line_clamp`:111, `fill_mask_holes`:143, `confident_forbidden`:170 | real | app/roop_3d/compositor/masks.py:41 |
| `app/roop_3d/compositor/matte.py` | 304 | Edge-aware matte refinement and edge decontamination (pure NumPy/OpenCV, no models). | `_box`:20, `guided_filter`:25, `fast_guided_filter`:41, `refine_occluder`:99, `decontaminate_edges`:125, `border_fade`:167 | real | app/roop_3d/compositor/matte.py:20 |
| `app/roop_3d/compositor/mouth.py` | 62 | Lower-face masks and mouth-only fusion for optional audio-driven edits. | `lower_face_soft_mask`:13, `composite_mouth_region`:40 | real | app/roop_3d/compositor/mouth.py:13 |
| `app/roop_3d/compositor/occlusion.py` | 49 | Foreground protection helpers for glasses, hair, hands and other occluders. | `semantic_occluder_mask`:11, `subtract_occluders`:19, `combine_occluders`:39 | real | app/roop_3d/compositor/occlusion.py:11 |
| `app/roop_3d/compositor/occlusion_analysis.py` | 259 | Unified face + occlusion analysis for one target face crop. | `FaceAnalysis`:35, `_interior_weight`:50, `OcclusionAnalyzer`:65 | real | app/roop_3d/compositor/occlusion_analysis.py:35 |
| `app/roop_3d/compositor/occlusion_model.py` | 93 | Local "visible face" networks: XSeg and the FaceFusion face occluder. | `VisibleFaceNet`:24 | real | app/roop_3d/compositor/occlusion_model.py:24 |
| `app/roop_3d/compositor/occlusion_tracker.py` | 178 | Optical-flow-aware temporal stabilisation of per-face occluder mattes. | `OcclusionState`:27, `_Track`:36, `OcclusionTracker`:43 | real | app/roop_3d/compositor/occlusion_tracker.py:27 |
| `app/roop_3d/compositor/photometric_identity.py` | 58 | Measured identity constraint for aggressive photometric edits. | `IdentityConstraint`:15, `constrain_identity`:25 | real | app/roop_3d/compositor/photometric_identity.py:15 |
| `app/roop_3d/compositor/pipeline.py` | 1297 | Orchestrate per-face parsing, restoration, harmonization and full-frame fusion. | `protect_matte`:42, `RobustOptions`:74, `CompositeOptions`:104, `_FaceContext`:230, `FrameCompositor`:248 | real | app/roop_3d/compositor/pipeline.py:42 |
| `app/roop_3d/compositor/quality.py` | 281 | Fidelity helpers for the robust compositor: illumination, sharpness/motion blur, fine texture. | `_gray01`:23, `_normalised_blur`:27, `local_illumination_match`:38, `shading_gain_match`:68, `_motion_kernel`:110, `_gradient_energy`:124 | real | app/roop_3d/compositor/quality.py:23 |
| `app/roop_3d/compositor/restoration.py` | 277 | Optional ROI restoration, blended only through the face alpha. | `blend_restored_roi`:10, `GpenRestorer`:44, `CodeFormerRestorer`:127, `UltraMaxPipelineRestorer`:185, `GfpganRestorer`:247 | real | app/roop_3d/compositor/restoration.py:10 |
| `app/roop_3d/compositor/robust_fusion.py` | 132 | Frame-space fusion for the robust compositor path. | `RobustFusionOptions`:24, `FusionReport`:37, `_roi`:45, `fuse_robust`:56 | real | app/roop_3d/compositor/robust_fusion.py:24 |
| `app/roop_3d/compositor/sensor_physics.py` | 295 | Target-referred optical blur, defocus and sensor grain for the final face composite. | `_resize`:15, `_gaussian_kernel`:22, `_gaussian`:29, `_normalised_gaussian`:41, `_masked_energy`:46, `_decode_transfer`:54 | real | app/roop_3d/compositor/sensor_physics.py:15 |
| `app/roop_3d/compositor/strand_matte.py` | 146 | Directional strand matte: thin hair strands and facial hair over the face, from the target's own pixels. | `StrandMatteOptions`:28, `scaled_for_face`:41, `_smoothstep`:51, `_normalised_blur`:56, `gabor_bank`:65, `strand_evidence`:77 | real | app/roop_3d/compositor/strand_matte.py:28 |
| `app/roop_3d/compositor/stream_restoration.py` | 600 | Independent restorer sessions, explicit training alignment and native-size bypass. | `RestorerSpec`:24, `gpen_spec`:37, `codeformer_spec`:41, `gfpgan_spec`:45, `ultramax_spec`:50, `load_ultramax_spec`:55 | real | app/roop_3d/compositor/stream_restoration.py:24 |
| `app/roop_3d/compositor/temporal.py` | 172 | Bounded per-track One-Euro filtering for affine, mask and color state. | `_alpha`:15, `OneEuroFilter`:20, `_TrackState`:52, `TemporalResult`:65, `TemporalSmoother`:71 | real | app/roop_3d/compositor/temporal.py:15 |

### `app/roop_3d/color/` (7 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/color/__init__.py` | 7 | Pure colour science for Roop 3D: code points, transfer functions, gamut, YCbCr and HDR tone mapping. |  | real | app/roop_3d/color/__init__.py:1 |
| `app/roop_3d/color/codes.py` | 40 | ITU-T H.273 / FFmpeg colour code points, with names and explicit "unspecified" handling. | `is_unspecified`:27, `name_of`:35 | real | app/roop_3d/color/codes.py:27 |
| `app/roop_3d/color/description.py` | 308 | Colour descriptions of media: what the file says, what is assumed, and what that means. | `PixelFormatTraits`:25, `pixel_format_traits`:37, `HdrStaticMetadata`:68, `parse_mastering_display`:105, `parse_content_light`:130, `merge_hdr_metadata`:137 | real | app/roop_3d/color/description.py:25 |
| `app/roop_3d/color/gamut.py` | 117 | RGB primaries, gamut conversion matrices and YCbCr coefficients. | `_xy_to_xyz`:40, `rgb_to_xyz_matrix`:45, `conversion_matrix`:58, `luma_coefficients`:65, `matrix_weights`:70, `quantisation`:76 | real | app/roop_3d/color/gamut.py:40 |
| `app/roop_3d/color/hdr.py` | 128 | Deterministic HDR (PQ / HLG, wide gamut) -> SDR (BT.709, BT.1886 display) conversion. | `ToneMapParams`:34, `bt2390_eetf`:50, `HdrToSdr`:74 | real | app/roop_3d/color/hdr.py:34 |
| `app/roop_3d/color/ocio.py` | 113 | Optional OpenColorIO adapter. Explicit opt-in: nothing in the default render path imports this module. | `_import`:22, `is_available`:30, `capability`:34, `OcioTransform`:45, `cross_validate`:85 | optional opt-in adapter, not on default path (ocio.py:1) | app/roop_3d/color/ocio.py:22 |
| `app/roop_3d/color/transfer.py` | 161 | Transfer functions (EOTF / OETF) with their published constants. | `_f`:31, `pq_eotf`:35, `pq_inverse_eotf`:43, `hlg_oetf`:49, `hlg_inverse_oetf`:57, `hlg_system_gamma`:63 | real | app/roop_3d/color/transfer.py:31 |

### `app/roop_3d/enhancers/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/enhancers/__init__.py` | 17 | Restorer and Enhancer implementations (UltraMax, GPEN Realistic, Dynamic CodeFormer, RestoreFormer++). |  | real | app/roop_3d/enhancers/__init__.py:1 |
| `app/roop_3d/enhancers/codeformer.py` | 107 | Dynamic CodeFormer & RestoreFormer++ with dynamic fidelity-weight binding. | `_blend`:16, `DynamicCodeFormerRestorer`:22, `RestoreFormerPlusPlusRestorer`:68 | real | app/roop_3d/enhancers/codeformer.py:16 |
| `app/roop_3d/enhancers/gpen_realistic.py` | 88 | GPEN Realistic: static batch-1 FP32-safe restoration with luma-only recolouring. | `luma_only_delta_recolour`:16, `GPENRealisticRestorer`:35 | real | app/roop_3d/enhancers/gpen_realistic.py:16 |
| `app/roop_3d/enhancers/styleganex_age.py` | 71 | StyleGANEX age inference; latent direction is not calibrated chronological age. | `AgeCalibration`:12, `StyleGANEXAge`:36 | real | app/roop_3d/enhancers/styleganex_age.py:12 |
| `app/roop_3d/enhancers/ultramax.py` | 161 | UltraMax Restorer: zero-copy CUDA IOBinding, explicit-only small-face bypass & 3-level Laplacian fusion. | `laplacian_pyramid_fusion`:15, `to_model_input`:61, `from_model_output`:67, `require_session`:73, `UltraMaxRestorer`:80 | real | app/roop_3d/enhancers/ultramax.py:15 |

### `app/roop_3d/pipeline/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/pipeline/__init__.py` | 22 | Stage 17: photorealistic 3D swap core and the FFmpeg-piped end-to-end video job runner. | `__getattr__`:14 | real | app/roop_3d/pipeline/__init__.py:14 |
| `app/roop_3d/pipeline/image_ops.py` | 365 | Pure image / geometry primitives behind :mod:`roop_3d.pipeline.photorealistic_swap_core`. | `HeadFrame`:25, `head_frame`:35, `forehead_arc`:60, `head_polygon`:76, `polygon_mask`:82, `similarity_fit`:94 | real | app/roop_3d/pipeline/image_ops.py:25 |
| `app/roop_3d/pipeline/photorealistic_swap_core.py` | 619 | Photorealistic 3D swap core: craniofacial geometry + rectified neural swap + quotient-image relight + Laplacian composite. | `SwapCoreOptions`:49, `FacePacket`:76, `FaceReport`:94, `_Geometry`:111, `_Layer`:133, `HairDonorReplacer`:140 | real | app/roop_3d/pipeline/photorealistic_swap_core.py:49 |
| `app/roop_3d/pipeline/session_adapter.py` | 207 | Per-frame glue between a prepared ``ReferencePipeline`` session and :class:`Photorealistic3DSwapCore`. | `FrameReport`:35, `SourceViews`:44, `SessionFrameProcessor`:82 | real | app/roop_3d/pipeline/session_adapter.py:35 |
| `app/roop_3d/pipeline/video_job_runner.py` | 513 | End-to-end video job runner: FFmpeg decode pipe -> photorealistic swap per frame -> FFmpeg encode pipe -> audio remux. | `VideoInfo`:56, `_rational`:76, `probe_video`:84, `FFmpegDecoder`:123, `ColourManagedWriter`:188, `_remux`:212 | real | app/roop_3d/pipeline/video_job_runner.py:56 |

### `app/roop_3d/media/` (25 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/media/__init__.py` | 2 | Local media package boundary. |  | real | app/roop_3d/media/__init__.py:1 |
| `app/roop_3d/media/actors.py` | 211 | Stage 23: multi-person target clustering. | `Actor`:28, `ActorReport`:46, `ActorRouting`:53, `ActorMappingResult`:60, `_area`:66, `_union_seconds`:71 | real | app/roop_3d/media/actors.py:28 |
| `app/roop_3d/media/audio.py` | 151 | Explicit all-track audio planning and packet/sample-accurate local muxing. | `transcode_layout`:30, `AudioPlan`:41, `plan_audio`:49, `audio_packets`:82 | real | app/roop_3d/media/audio.py:30 |
| `app/roop_3d/media/av_sync.py` | 181 | Audio integrity and synchronisation evidence for a produced file. | `_open_audio_source`:32, `payload_digest`:39, `decode_window`:54, `LagMeasurement`:111, `measure_lag`:119, `TrackSync`:149 | real | app/roop_3d/media/av_sync.py:32 |
| `app/roop_3d/media/capabilities.py` | 197 | What the media layer can really do on this machine, and the evidence for it. | `available_encoders`:26, `media_capabilities`:37, `ocio_capability`:116, `_clip`:123, `selftest_profile`:162 | real | app/roop_3d/media/capabilities.py:26 |
| `app/roop_3d/media/convert.py` | 91 | 8-bit BGR -> encoder frames with an explicit YCbCr matrix and range. | `RgbFrameConverter`:23, `clone_frame`:68, `promote_planes`:81 | real | app/roop_3d/media/convert.py:23 |
| `app/roop_3d/media/detector.py` | 188 | Selectable SCRFD, RetinaFace-10G and YOLOFace decoders, loaded on inspection. | `Detector`:18, `iou`:182 | real | app/roop_3d/media/detector.py:18 |
| `app/roop_3d/media/export_profiles.py` | 115 | Declarative export presets. A profile states what it *can* carry; it never promises more. | `ExportProfile`:20, `get_profile`:99, `legacy_profile`:108 | real | app/roop_3d/media/export_profiles.py:20 |
| `app/roop_3d/media/face_analysis.py` | 191 | Pose-aware frame face analysis (Stage 12): adaptive detection + embedding + dense landmarks for every face of a frame. | `AnalyzedFace`:36, `five_point_plausible`:51, `agrees_with_106`:70, `thumbnail_b64`:80, `FaceAnalyzer`:96, `face_to_dict`:168 | real | app/roop_3d/media/face_analysis.py:36 |
| `app/roop_3d/media/fidelity.py` | 93 | Does the finished file decode to what the encoder was given? | `FidelitySampler`:21, `_stats`:38, `compare_frames`:48, `luma_plane`:71, `compare_luma`:79 | real | app/roop_3d/media/fidelity.py:21 |
| `app/roop_3d/media/finishing.py` | 158 | Local display-referred finishing. Data passes never enter this pipeline. | `GrainOptions`:16, `FinishingOptions`:24, `ExportOptions`:36, `CubeLut`:64, `parse_cube`:85, `Finisher`:124 | real | app/roop_3d/media/finishing.py:16 |
| `app/roop_3d/media/frame_map.py` | 67 | Streaming CFR sample-and-hold selection, including every drop and duplicate. | `TimedFrame`:12, `FrameMapEntry`:22, `FrameMap`:32 | real | app/roop_3d/media/frame_map.py:12 |
| `app/roop_3d/media/frame_service.py` | 400 | Scrub-speed frame access: persistent decoders, a decoded-frame cache and per-frame face detection. | `_lru_put`:50, `MediaInfo`:60, `FrameResult`:77, `_Entry`:87, `VideoFrameService`:111 | real | app/roop_3d/media/frame_service.py:50 |
| `app/roop_3d/media/nv_writer.py` | 267 | Stage 9 hardware video writer: NVENC with automatic CPU fallback, optional .cube LUT bake, and source-track mux. | `probe_nvenc`:31, `NVHardwareVideoWriter`:47, `mux_source_audio`:222, `_has_copy_streams`:251, `_has_audio`:260 | real | app/roop_3d/media/nv_writer.py:31 |
| `app/roop_3d/media/probe.py` | 195 | Local-only media probing with exact timing and explicit sampling evidence. | `open_media`:25, `AudioInfo`:45, `SubtitleInfo`:65, `VideoInfo`:79, `ProbeResult`:118, `probe`:128 | real | app/roop_3d/media/probe.py:25 |
| `app/roop_3d/media/provenance.py` | 94 | Hash-bound render recipes and optional, local cryptographic authentication. | `canonical`:15, `capabilities`:19, `require_mode`:25, `recipe_claim`:30, `authenticate_manifest`:39, `verify_manifest`:45 | real | app/roop_3d/media/provenance.py:15 |
| `app/roop_3d/media/schema.py` | 139 | Finite, bounded contracts for local target video inspection and previews. | `MediaRequest`:9, `InspectRequest`:13, `AudioTrack`:26, `SubtitleTrack`:36, `MediaProbe`:44, `BoundingBox`:78 | real | app/roop_3d/media/schema.py:9 |
| `app/roop_3d/media/sequence.py` | 233 | Numbered image sequences, as input and as an export profile. | `SequenceInfo`:44, `_open_image`:73, `_decode_image`:82, `to_bgr8`:96, `discover_sequence`:111, `sequence_color`:143 | real | app/roop_3d/media/sequence.py:44 |
| `app/roop_3d/media/service.py` | 546 | Bounded local PyAV decoding; remote protocols and external references disabled. | `MediaService`:40 | real | app/roop_3d/media/service.py:40 |
| `app/roop_3d/media/subtitles.py` | 90 | Subtitle stream preservation: packet copy where the output container can hold the codec, otherwise an | `SubtitlePlan`:30, `plan_subtitles`:39, `subtitle_packets`:61 | real | app/roop_3d/media/subtitles.py:30 |
| `app/roop_3d/media/timestamps.py` | 55 | Integer tick arithmetic. Rounding is nearest, with ties away from zero. | `round_ticks`:10, `rescale`:16, `Timestamp`:25, `TimestampEngine`:41 | real | app/roop_3d/media/timestamps.py:10 |
| `app/roop_3d/media/validation.py` | 223 | Validation of media files: inputs (``validate_media``) and finished outputs (``validate_output``). | `Issue`:24, `validate_media`:33, `_deep`:102, `OutputExpectation`:142, `validate_output`:157 | real | app/roop_3d/media/validation.py:24 |
| `app/roop_3d/media/vfx.py` | 263 | Unfinished, full-frame 16-bit compositing passes and lossless FFV1 previews. | `geometry_passes`:24, `PassCollector`:73, `publish_passes`:115, `discard_sequences`:242 | real | app/roop_3d/media/vfx.py:24 |
| `app/roop_3d/media/video_reader.py` | 270 | Pull-based upright RGB/BGR decode. No background workers or frame queue. | `check_cancel`:29, `describe_frame_color`:34, `rotate_upright`:56, `upright`:68, `DecodedFrame`:82, `VideoReader`:92 | real | app/roop_3d/media/video_reader.py:29 |
| `app/roop_3d/media/video_writer.py` | 621 | Validated local encode, all-track audio / subtitle mux, and atomic output publication. | `_encoder`:68, `_Facts`:95, `_output_format`:110, `_full_range`:125, `encode_video`:129 | real | app/roop_3d/media/video_writer.py:68 |

### `app/roop_3d/io/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/io/__init__.py` | 20 | Stage 12 interchange: binary glTF serialization and the optional, GPL-isolated Blender bridge. | `__getattr__`:14 | real | app/roop_3d/io/__init__.py:14 |
| `app/roop_3d/io/blender_bridge.py` | 255 | Stage 12: optional, clean-room bridge to an external Blender 4.x. | `BlenderBridgeError`:40, `_is_loopback`:44, `BlenderMCPBridge`:64 | real | app/roop_3d/io/blender_bridge.py:40 |
| `app/roop_3d/io/glb_export.py` | 524 | Stage 12: a dependency-free binary glTF 2.0 (``.glb``) serializer for head meshes, textures, shape keys and head tracks. | `_fail`:52, `_png_chunk`:57, `encode_png`:61, `png_size`:86, `_PngCache`:94, `MorphTarget`:123 | real | app/roop_3d/io/glb_export.py:52 |
| `app/roop_3d/io/proxy_and_edl.py` | 336 | Local video proxies, filmstrips, CMX 3600 EDL interchange and render ETA. | `_rate`:18, `timecode_to_frame`:36, `frame_to_timecode`:62, `EDLEvent`:89, `EDLDocument`:102, `parse_edl`:111 | real | app/roop_3d/io/proxy_and_edl.py:18 |

### `app/roop_3d/execution/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/execution/__init__.py` | 44 | GPU execution helpers for mixed precision inference and bounded frame processing. |  | real | app/roop_3d/execution/__init__.py:1 |
| `app/roop_3d/execution/trt_mixed_precision.py` | 804 | TensorRT mixed precision sessions with graph-level FP32 pinning and provider failover. | `OptimizationProfiles`:41, `_format_shapes`:59, `_family_shape`:64, `_input_specs`:93, `derive_optimization_profiles`:113, `_digest_file`:168 | real | app/roop_3d/execution/trt_mixed_precision.py:41 |
| `app/roop_3d/execution/worker_pools.py` | 353 | CPU decode/preprocess and asynchronous post-encode worker pools. | `_cpu_workers`:24, `DeviceFrameLease`:32, `_TransferSlot`:61, `DecodePreprocessPool`:66, `GopChunk`:201, `_concat_quote`:219 | real | app/roop_3d/execution/worker_pools.py:24 |
| `app/roop_3d/execution/zero_copy_pipeline.py` | 396 | Reusable device buffers and ORT I/O binding for frame-resident GPU stages. | `GPUExecutionFallback`:29, `OutputBufferSpec`:34, `_torch_module`:46, `_numpy_dtype`:53, `_output_spec`:68, `_tree_is_cuda`:76 | real | app/roop_3d/execution/zero_copy_pipeline.py:29 |

### `app/roop_3d/core/` (17 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/core/__init__.py` | 2 | Runtime settings and local system contracts. |  | real | app/roop_3d/core/__init__.py:1 |
| `app/roop_3d/core/animation.py` | 113 | Frame-indexed parameter curves and explicit per-track role intervals. | `CurveKey`:16, `ParameterCurves`:24, `_compile`:51, `evaluate_curve`:65, `FaceSegment`:92 | real | app/roop_3d/core/animation.py:16 |
| `app/roop_3d/core/dll_injection.py` | 87 | Automatic CUDA / cuDNN / TensorRT DLL injection and directory bootstrap for Windows. | `setup_environment`:10 | real | app/roop_3d/core/dll_injection.py:10 |
| `app/roop_3d/core/errors.py` | 6 | Safe domain errors; never serialize exception inputs or credentials. | `DomainError`:2 | real | app/roop_3d/core/errors.py:2 |
| `app/roop_3d/core/face_editor.py` | 42 | Validated, per-target portrait edits shared by preview and export. | `FaceEditor`:9 | real | app/roop_3d/core/face_editor.py:9 |
| `app/roop_3d/core/hardware.py` | 449 | Read-only hardware probes: provider advertisement is not execution proof. | `GPU`:26, `Hardware`:37, `resolve_hardware_profile`:98, `effective_vram_budget`:112, `resolve_accelerator`:123, `tensor_core_precisions`:136 | real | app/roop_3d/core/hardware.py:26 |
| `app/roop_3d/core/image_bytes.py` | 37 | Inspect PNG/JPEG dimensions before allowing native image expansion. | `decode_face_image`:7 | real | app/roop_3d/core/image_bytes.py:7 |
| `app/roop_3d/core/integrity.py` | 22 | Honest health status based on the most recent isolation gate result. | `isolation_status`:9 | real | app/roop_3d/core/integrity.py:9 |
| `app/roop_3d/core/model_downloader.py` | 493 | First-Use Automatic Model Downloader & Integrity Verifier for Roop 3D. | `_resolve_project_root`:34, `ModelSpec`:40, `get_model_spec`:213, `get_model_search_paths`:219, `verify_file_hash`:252, `find_model_path`:269 | real | app/roop_3d/core/model_downloader.py:34 |
| `app/roop_3d/core/ort_session.py` | 753 | TensorRT-first ONNX Runtime sessions with per-model dynamic profiles. | `_provider_name`:21, `_string_provider_options`:25, `_trt_mixed_options`:30, `_image_size`:42, `_static_graph_inputs`:67, `_trt_excluded_ops`:76 | real | app/roop_3d/core/ort_session.py:21 |
| `app/roop_3d/core/pose.py` | 241 | Unified head-pose structure, tuple sanitizer and the single 5-point pose estimator. | `Pose3D`:38, `PoseCell`:55, `pose_cell`:78, `_finite`:86, `as_pose3d`:96, `as_legacy_triplet`:132 | real | app/roop_3d/core/pose.py:38 |
| `app/roop_3d/core/redaction.py` | 18 | Defense in depth for structured diagnostics; request bodies are never logged. | `redact`:8 | real | app/roop_3d/core/redaction.py:8 |
| `app/roop_3d/core/settings.py` | 167 | Authoritative v2 catalog. Secrets belong in environment/OS stores only. | `Contract`:15, `runtime_path`:20, `Settings`:37, `migrate_config`:128, `ConfigStore`:139 | real | app/roop_3d/core/settings.py:15 |
| `app/roop_3d/core/stage_timing.py` | 67 | Measured per-frame stage timings for the live bottleneck flamegraph. | `validate_timings`:21, `StageTimings`:35 | real | app/roop_3d/core/stage_timing.py:21 |
| `app/roop_3d/core/storage.py` | 86 | Owned-path, durable atomic JSON storage and cross-process serialization. | `owned`:23, `atomic_bytes`:27, `atomic_json`:49, `exclusive`:56 | real | app/roop_3d/core/storage.py:23 |
| `app/roop_3d/core/transfer_ledger.py` | 246 | Measured host<->device copies at the ONNX Runtime boundary. | `_enabled`:29, `TransferLedger`:33 | real | app/roop_3d/core/transfer_ledger.py:29 |
| `app/roop_3d/core/workstation.py` | 165 | Bounded workstation parameters shared by previews and durable render requests. | `WorkstationParameters`:12 | real | app/roop_3d/core/workstation.py:12 |

### `app/roop_3d/async_pipeline/` (8 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/async_pipeline/__init__.py` | 11 | Asynchronous multi-stream frame pipeline, binary live-preview transport and hardware telemetry. |  | real | app/roop_3d/async_pipeline/__init__.py:1 |
| `app/roop_3d/async_pipeline/buffers.py` | 393 | Bounded hand-off queues and a CUDA pinned-host frame pool. | `PipelineCancelled`:28, `_Closed`:32, `BoundedQueue`:43, `PinnedFramePool`:130, `CudaFrameLease`:212, `CudaFrameRing`:285 | real | app/roop_3d/async_pipeline/buffers.py:28 |
| `app/roop_3d/async_pipeline/config.py` | 27 | Switches for the asynchronous pipeline.  Environment variables, so no settings-schema or UI-contract churn. | `async_pipeline_enabled`:14, `pinned_enabled`:18, `async_queue_depth`:22 | real | app/roop_3d/async_pipeline/config.py:14 |
| `app/roop_3d/async_pipeline/cuda_video.py` | 147 | Explicit device-resident frame path over the shared multi-stream scheduler. | `_frame_and_index`:15, `CudaFrameSource`:27, `_assert_device_pixels`:65, `CudaFramePipeline`:86 | real | app/roop_3d/async_pipeline/cuda_video.py:15 |
| `app/roop_3d/async_pipeline/hwmon.py` | 234 | Live GPU / CPU / PCIe utilisation without optional dependencies. | `_NvmlUtilization`:21, `_NvmlMemory`:25, `Nvml`:29, `_smi_sample`:97, `_FileTime`:112, `_cpu_times`:116 | real | app/roop_3d/async_pipeline/hwmon.py:21 |
| `app/roop_3d/async_pipeline/live.py` | 342 | Zero-copy-ish live egress: GPU JPEG encoding on a background thread and a 16-byte binary frame header. | `_units`:46, `pack_header`:50, `unpack_header`:57, `JpegEncoder`:68, `Subscription`:148, `LiveFrameHub`:180 | real | app/roop_3d/async_pipeline/live.py:46 |
| `app/roop_3d/async_pipeline/reference.py` | 138 | The four per-frame stages of the reference render, shared by the job runner and the benchmark. | `ReferenceStages`:26 | real | app/roop_3d/async_pipeline/reference.py:26 |
| `app/roop_3d/async_pipeline/stages.py` | 466 | Asynchronous multi-stage worker graph with per-stage CUDA streams and event hand-off. | `_torch`:37, `Item`:46, `adopt`:72, `StageStats`:89, `Stage`:103, `AsyncPipeline`:133 | real | app/roop_3d/async_pipeline/stages.py:37 |

### `app/roop_3d/models/` (31 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/models/__init__.py` | 2 | Authoritative model identities, read-only local reuse and gated HTTPS downloads. |  | real | app/roop_3d/models/__init__.py:1 |
| `app/roop_3d/models/accelerated_runtime.py` | 322 | Parity-gated provider selection around the preserved Stage 06 model contract. | `run_checked`:31, `AcceleratorFactory`:59, `AcceleratedModel`:239 | real | app/roop_3d/models/accelerated_runtime.py:31 |
| `app/roop_3d/models/benchmark.py` | 42 | Synchronous session timing including output transfer; bounded raw samples. | `BenchmarkResult`:11, `benchmark`:21, `summarize`:35 | real | app/roop_3d/models/benchmark.py:11 |
| `app/roop_3d/models/catalog.py` | 576 | Compatibility summaries backed by the authoritative Stage 03 registry. | `render_dependencies`:27, `ModelCatalog`:61 | real | app/roop_3d/models/catalog.py:27 |
| `app/roop_3d/models/diagnostics.py` | 30 | Small public provider diagnostics; media and golden tensors are never included. | `ProviderFailure`:11, `ModelDiagnostic`:17, `JobDiagnostics`:28 | real | app/roop_3d/models/diagnostics.py:11 |
| `app/roop_3d/models/downloader.py` | 317 | HTTPS-only, bounded, resumable downloads of reviewed immutable artifacts. | `tls_context`:40, `validate_url`:57, `validate_hf_redirect`:72, `validator`:91, `DownloadResult`:107, `Downloader`:113 | real | app/roop_3d/models/downloader.py:40 |
| `app/roop_3d/models/downloads.py` | 145 | Explicit, cancellable downloads for immutable registered model artifacts. | `DownloadJob`:15, `ModelDownloadManager`:34 | real | app/roop_3d/models/downloads.py:15 |
| `app/roop_3d/models/dynamic_shapes.py` | 246 | Conservative batch graph surgery, admitted only after real ORT execution. | `DynamicShapeError`:16, `StaticBatchRequired`:20, `static_batch_reason`:24, `runtime_inputs`:30, `input_shapes`:35, `sample_inputs`:50 | real | app/roop_3d/models/dynamic_shapes.py:16 |
| `app/roop_3d/models/emap_cache.py` | 54 | Small, hash-bound projection cache; NumPy object arrays are never accepted. | `cached_emap`:19 | real | app/roop_3d/models/emap_cache.py:19 |
| `app/roop_3d/models/engine_cache.py` | 233 | Hardware-keyed metadata only; this module does not compile or load engines. | `ShapeRange`:25, `EngineKey`:38, `EngineMetadata`:53, `BundleFile`:61, `EngineBundle`:67, `EngineCache`:78 | real | app/roop_3d/models/engine_cache.py:25 |
| `app/roop_3d/models/export_validation.py` | 140 | Deterministic numerical gates; no candidate becomes active on shape alone. | `Tolerance`:17, `OutputMetric`:36, `ParityReport`:49, `fixture_digest`:60, `_images`:74, `structural_similarity`:80 | real | app/roop_3d/models/export_validation.py:17 |
| `app/roop_3d/models/golden.py` | 146 | Private fixed golden tensors captured from genuine Stage 06 CPU inference. | `ModelGolden`:22, `frozen_copy`:36, `GoldenBank`:47 | real | app/roop_3d/models/golden.py:22 |
| `app/roop_3d/models/governance.py` | 101 | Fail-closed compatibility and lifecycle assessment for registered artifacts. | `ModelCompatibilityScanner`:16 | real | app/roop_3d/models/governance.py:16 |
| `app/roop_3d/models/governance_reviews.json` | 35 | { |  | manifest | app/roop_3d/models/governance_reviews.json:1 |
| `app/roop_3d/models/grid_sample.py` | 167 | Lower trilinear volumetric sampling to two standard 4-D GridSample nodes. | `lower_grid_sample_3d`:18, `convert_grid_sample`:141 | real | app/roop_3d/models/grid_sample.py:18 |
| `app/roop_3d/models/hub.py` | 374 | Read-only presentation of registered models and compatible engine adapters. | `ModelHubService`:31 | real | app/roop_3d/models/hub.py:31 |
| `app/roop_3d/models/local_provider.py` | 143 | Read-only access to preserved locators and configured local search roots. | `LocalWeightProvider`:19 | real | app/roop_3d/models/local_provider.py:19 |
| `app/roop_3d/models/local_trust.py` | 169 | Trust for weight files the user physically placed in ``app/models``. | `normalize`:37, `family_key`:41, `TrustedFile`:54, `LocalTrust`:61 | real | app/roop_3d/models/local_trust.py:37 |
| `app/roop_3d/models/makehuman_sources.json` | 530 | { |  | manifest | app/roop_3d/models/makehuman_sources.json:1 |
| `app/roop_3d/models/model_manifest.json` | 10185 | { |  | manifest | app/roop_3d/models/model_manifest.json:1 |
| `app/roop_3d/models/native_libraries.py` | 82 | Register packaged CUDA/TRT libraries before importing ONNX Runtime. | `packaged_library_directories`:18, `register_native_libraries`:31, `quiet_onnxruntime`:64 | real | app/roop_3d/models/native_libraries.py:18 |
| `app/roop_3d/models/owned_files.py` | 44 | Project-owned model outputs, with no link or hardlink traversal. | `owned_path`:10, `cache_directory`:25, `model_source`:31 | real | app/roop_3d/models/owned_files.py:10 |
| `app/roop_3d/models/precision_policy.py` | 69 | Model-specific precision decisions. FP16 is a candidate until measured parity. | `PrecisionPolicy`:13, `policy_for`:39, `require_no_custom_ops`:66 | real | app/roop_3d/models/precision_policy.py:13 |
| `app/roop_3d/models/registry.py` | 387 | Authoritative artifact identities; byte, graph and rights evidence stay separate. | `filename`:27, `Locator`:34, `ModelRecord`:44, `Manifest`:142, `_with_supported_candidates`:156, `_apply_verified_sources`:200 | real | app/roop_3d/models/registry.py:27 |
| `app/roop_3d/models/serialization.py` | 30 | Restricted model deserialization helpers. No pickle fallback is permitted. | `load_restricted_torch_weights`:10 | real | app/roop_3d/models/serialization.py:10 |
| `app/roop_3d/models/stage16.py` | 232 | Lazy, hash-checked Stage 16 artifacts and provider execution evidence. | `prepare_generator`:29, `resolve_artifact`:82, `readiness_report`:94, `CheckedSession`:131 | real | app/roop_3d/models/stage16.py:29 |
| `app/roop_3d/models/stage16_models.json` | 767 | { |  | manifest | app/roop_3d/models/stage16_models.json:1 |
| `app/roop_3d/models/stage6_models.json` | 339 | { |  | manifest | app/roop_3d/models/stage6_models.json:1 |
| `app/roop_3d/models/trt_builder.py` | 254 | Offline TensorRT EP compilation with eviction, workspace caps and atomic cache. | `TrtAvailability`:34, `configure_native_paths`:41, `trt_availability`:57, `workspace_budget`:81, `full_precision_build`:92, `session_options`:105 | real | app/roop_3d/models/trt_builder.py:34 |
| `app/roop_3d/models/verified_sources.json` | 693 | { |  | manifest | app/roop_3d/models/verified_sources.json:1 |
| `app/roop_3d/models/verifier.py` | 66 | Bounded SHA-256 verification without deserialization or file mutation. | `verified_stream`:22, `verify_file`:63 | real | app/roop_3d/models/verifier.py:22 |

### `app/roop_3d/orchestrator/` (12 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/orchestrator/__init__.py` | 13 | Stage 14: capability-based multi-engine orchestrator (planning and reporting; no new execution path). |  | real | app/roop_3d/orchestrator/__init__.py:1 |
| `app/roop_3d/orchestrator/compile.py` | 85 | Compile a Pipeline plan onto the existing execution path; never a second runner. | `CompiledPlan`:25, `compile_plan`:41 | real | app/roop_3d/orchestrator/compile.py:25 |
| `app/roop_3d/orchestrator/engines_manifest.json` | 1095 | { |  | manifest | app/roop_3d/orchestrator/engines_manifest.json:1 |
| `app/roop_3d/orchestrator/evidence.py` | 73 | Evidence store and the exact-match rule that decides whether a measurement applies to a plan. | `EvidenceMatch`:22, `EvidenceStore`:33 | real | app/roop_3d/orchestrator/evidence.py:22 |
| `app/roop_3d/orchestrator/plan.py` | 153 | Pipeline plans: one Engine per category, with the structural checks every plan must pass. | `PlanEngine`:15, `Plan`:30, `plan_engine`:50, `make_plan`:55, `_derive`:59, `structural_reasons`:69 | real | app/roop_3d/orchestrator/plan.py:15 |
| `app/roop_3d/orchestrator/registry.py` | 291 | Capability registry: declarative Engine data validated fail-closed against the governed Model registry. | `adapter_present`:53, `Issue`:73, `OrchestratorRegistryError`:82, `job_runner_ids`:89, `legacy_residency_sets`:94, `_schema_issues`:99 | real | app/roop_3d/orchestrator/registry.py:53 |
| `app/roop_3d/orchestrator/render_profiles.json` | 141 | { |  | manifest | app/roop_3d/orchestrator/render_profiles.json:1 |
| `app/roop_3d/orchestrator/report.py` | 88 | Explainable decision report: structured JSON with a deterministic digest, plus a rendered text view. | `_admission`:18, `_selected`:27, `_decision`:41, `build_report`:59, `render_text`:70 | real | app/roop_3d/orchestrator/report.py:18 |
| `app/roop_3d/orchestrator/scene.py` | 61 | Scene segments and traits from media-inspection output (the signals it really produces). | `SceneSegment`:11, `whole_target_segment`:29, `_trait_for`:34, `segments_from_tracks`:45 | real | app/roop_3d/orchestrator/scene.py:11 |
| `app/roop_3d/orchestrator/scene_rules.json` | 22 | { |  | manifest | app/roop_3d/orchestrator/scene_rules.json:1 |
| `app/roop_3d/orchestrator/schema.py` | 249 | Closed vocabularies and frozen contracts for the Stage 14 capability registry. | `ReasonCode`:54, `ModelRef`:79, `Resolution`:84, `Fallback`:95, `EngineEntry`:110, `EngineManifest`:143 | real | app/roop_3d/orchestrator/schema.py:54 |
| `app/roop_3d/orchestrator/selector.py` | 384 | Deterministic Engine selection: Manual, Balanced and Quality. | `ManualPick`:30, `MediaFacts`:38, `SelectionRequest`:48, `PlanEval`:60, `Decision`:73, `Selection`:92 | real | app/roop_3d/orchestrator/selector.py:30 |

### `app/roop_3d/server/` (39 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/server/__init__.py` | 2 | Loopback HTTP service. |  | real | app/roop_3d/server/__init__.py:1 |
| `app/roop_3d/server/__main__.py` | 136 | Reserve the exact loopback socket before Uvicorn starts; never kill a port owner. | `PortCollisionError`:37, `reserve_socket`:41, `lan_addresses`:57, `parse_arguments`:65, `JsonFormatter`:73, `ReadyServer`:85 | real | app/roop_3d/server/__main__.py:37 |
| `app/roop_3d/server/actor_routes.py` | 52 | Stage 23: multi-person target actors ("Person #1" ... "Person #N") for one inspection. | `ActorAssignments`:24, `_inspection`:29, `list_actors`:39, `map_actors`:45 | real | app/roop_3d/server/actor_routes.py:24 |
| `app/roop_3d/server/api.py` | 69 | Production same-origin workstation serving with API-safe SPA fallback. | `ThreadBenchmarkRequest`:15, `benchmark_threads`:20, `benchmark_status`:31, `WorkstationFiles`:35, `mount_workstation`:54 | real | app/roop_3d/server/api.py:15 |
| `app/roop_3d/server/app.py` | 337 | Minimal local service; no model, diffusion or restricted-weight imports. | `release_media_models`:45, `_universal_model_manager`:56, `create_app`:74 | real | app/roop_3d/server/app.py:45 |
| `app/roop_3d/server/blender_routes.py` | 55 | Stage 12 routes for the optional external-Blender workflow (``roop_3d.io.blender_bridge``). | `blender_exporter_script`:24, `blender_status`:30, `send_to_blender`:38 | real | app/roop_3d/server/blender_routes.py:24 |
| `app/roop_3d/server/contracts.py` | 122 | REST response contracts exported in the local OpenAPI document. | `ErrorDetail`:12, `ErrorEnvelope`:18, `IsolationStatus`:22, `Health`:30, `ModelSummary`:40, `Capabilities`:76 | real | app/roop_3d/server/contracts.py:12 |
| `app/roop_3d/server/diagnostics.py` | 1188 | 14-section diagnostics report, pre-flight render linter, and synthetic thread benchmark runner. | `PreFlightLintRequest`:28, `BenchmarkRequest`:39, `_verify_dir_health`:60, `_check_dll_alignment`:80, `_get_node_version`:105, `purge_idle_vram`:119 | real | app/roop_3d/server/diagnostics.py:28 |
| `app/roop_3d/server/engine_gate.py` | 148 | Mutual exclusion between live-preview inference and model-lifecycle owners. | `EngineBusy`:32, `EngineGate`:40 | real | app/roop_3d/server/engine_gate.py:32 |
| `app/roop_3d/server/face_bank.py` | 217 | Durable target-person bank. Discovery ranks survive updates and explicit merges. | `DiscoverBank`:21, `EditBank`:25, `FaceBank`:33, `read_bank`:206, `discover_bank`:210, `edit_bank`:215 | real | app/roop_3d/server/face_bank.py:21 |
| `app/roop_3d/server/face_extract.py` | 487 | `POST /api/faces/extract_from_frame`: the end-to-end "Use Face From Frame" pipeline. | `ExtractRequest`:52, `_unit`:70, `_slug`:78, `FaceExtractService`:82, `extract_from_frame`:352, `target_clusters`:357 | real | app/roop_3d/server/face_extract.py:52 |
| `app/roop_3d/server/faceset_3d.py` | 155 | Automatic in-memory 3D reconstruction and exact sphere coverage API. | `Reconstruct3DRequest`:25, `_read_archive`:44, `_from_images`:72, `_reconstruct`:95, `reconstruct_3d`:127, `faceset_3d_state`:132 | real | app/roop_3d/server/faceset_3d.py:25 |
| `app/roop_3d/server/faceset_builder.py` | 131 | Media builder jobs, archive listing, and curation publication. | `ExtractMedia`:10, `SaveFaceset`:23, `SynthesizeFaceset`:31, `SynthesizePhotos`:38, `synthesize_photos`:44, `synthesize`:50 | real | app/roop_3d/server/faceset_builder.py:10 |
| `app/roop_3d/server/finishing.py` | 187 | Owned local finishing, artifact verification and NLE interchange planning. | `FinishRequest`:28, `FinishPreviewRequest`:33, `finishing_capabilities`:38, `finishing_preview`:45, `finishing_render`:72, `finished_artifact`:79 | real | app/roop_3d/server/finishing.py:28 |
| `app/roop_3d/server/identity_assets.py` | 530 | Stage 7 workstation API for the identity panel: per-view FaceSet thumbnails and the Stage 3 identity asset. | `BakeAtlasRequest`:61, `_archive`:65, `load_document`:78, `_settings`:85, `_store`:90, `_coverage_summary`:96 | real | app/roop_3d/server/identity_assets.py:61 |
| `app/roop_3d/server/identity_lock.py` | 74 | Target-identity API: contact sheet from an identity inspection, and hash-bound locks. | `store`:18, `contact_sheet`:24, `create_lock`:46, `list_locks`:52, `read_lock`:58, `edit_lock`:65 | real | app/roop_3d/server/identity_lock.py:18 |
| `app/roop_3d/server/installer_routes.py` | 94 | REST API endpoints for environment preflight, offline models, updates, and recovery. | `ScanRequest`:23, `ImportBundleRequest`:27, `RollbackRequest`:31, `get_preflight`:36, `get_offline_health`:42, `scan_local_directory`:49 | real | app/roop_3d/server/installer_routes.py:23 |
| `app/roop_3d/server/live_preview.py` | 864 | Production live-preview endpoints: cache-first, non-blocking, mutex-protected. | `PreviewCancelled`:65, `MaskStroke`:70, `OcclusionSettings`:83, `LivePreviewPayload`:88, `CacheEntry`:219, `PreviewCache`:262 | real | app/roop_3d/server/live_preview.py:65 |
| `app/roop_3d/server/live_stream.py` | 149 | Binary live-frame WebSocket and pipeline saturation telemetry. | `make_hub`:36, `_stage_ms`:41, `pipeline_metrics`:49, `preview_stream`:62 | real | app/roop_3d/server/live_stream.py:36 |
| `app/roop_3d/server/live_studio.py` | 192 | Local camera studio. Host-frame egress is measured and never advertised as zero-copy. | `StudioConfig`:24, `LiveStudio`:38, `status`:180, `start`:185, `stop`:190 | real | app/roop_3d/server/live_studio.py:24 |
| `app/roop_3d/server/media_frames.py` | 82 | Timeline scrubbing endpoints: cheap media info, one frame (optionally with its faces), and a filmstrip. | `_meta`:29, `media_info`:41, `media_frame`:47, `media_filmstrip`:78 | real | app/roop_3d/server/media_frames.py:29 |
| `app/roop_3d/server/middleware.py` | 158 | Bounded ASGI intake, explicit origins, and JSON request logging. | `ApiAliasMiddleware`:21, `IntakeMiddleware`:39 | real | app/roop_3d/server/middleware.py:21 |
| `app/roop_3d/server/model_manifest_routes.py` | 100 | REST surface of the universal model download manager (`roop_3d.audit.local_models`). | `ManifestDownloadRequest`:30, `VerifyRepairRequest`:39, `manager_of`:47, `as_domain`:54, `start_manifest_download`:58, `models_manifest`:67 | real | app/roop_3d/server/model_manifest_routes.py:30 |
| `app/roop_3d/server/model_runtime.py` | 35 | Build the same local TensorRT sessions used by preview and render. | `compile_model`:10 | real | app/roop_3d/server/model_runtime.py:10 |
| `app/roop_3d/server/orchestrator.py` | 125 | Stage 14 dry-run API: build a Pipeline plan and its decision report without starting a job. | `PickBody`:27, `TrackBody`:34, `MediaBody`:43, `PlanBody`:48, `_Pose`:58, `_Track`:63 | real | app/roop_3d/server/orchestrator.py:27 |
| `app/roop_3d/server/pipeline_status.py` | 168 | Measured three-stage stream telemetry and preserved reference-job status. | `_idle`:20, `TelemetrySnapshot`:24, `PipelineTracker`:29 | real | app/roop_3d/server/pipeline_status.py:20 |
| `app/roop_3d/server/professional_media.py` | 68 | Stage 15 media API: what the media layer can do here, the export presets, and input validation. | `ValidateBody`:24, `EDLImportBody`:29, `capabilities`:35, `export_profiles`:40, `validate`:53, `import_edl`:59 | real | app/roop_3d/server/professional_media.py:24 |
| `app/roop_3d/server/queue_routes.py` | 225 | REST endpoints for the 10-state backend-owned persistent queue and 4-strategy Batch Matrix. | `SubmitJobRequest`:24, `ReorderRequest`:30, `RetrySliceRequest`:34, `ProgressUpdateRequest`:38, `BatchGenerateRequest`:46, `BatchSubmitRequest`:60 | real | app/roop_3d/server/queue_routes.py:24 |
| `app/roop_3d/server/reliability_routes.py` | 69 | Stage 18 REST surface: local, redacted diagnostics plus the crash-consistency audit. | `RepairRequest`:25, `_auditor`:31, `diagnostics_report`:41, `live_resources`:53, `audit`:59, `repair`:65 | real | app/roop_3d/server/reliability_routes.py:25 |
| `app/roop_3d/server/routes.py` | 730 | Local REST contracts; catalog adapters never import legacy application code. | `facesets`:26, `config`:44, `save_config`:49, `capabilities`:60, `BatchModelDownloadRequest`:121, `ModelVerifyRequest`:128 | real | app/roop_3d/server/routes.py:26 |
| `app/roop_3d/server/runtime_report_routes.py` | 18 | `GET /api/v1/system/runtime-report`: the 14-section structured runtime report as JSON plus copy-ready plain text. | `runtime_report`:14 | real | app/roop_3d/server/runtime_report_routes.py:14 |
| `app/roop_3d/server/security.py` | 56 | Opt-in network sharing (`--server_share`): per-launch bearer token for remote peers. | `ShareMode`:20, `is_loopback_peer`:47 | real | app/roop_3d/server/security.py:20 |
| `app/roop_3d/server/studio_environment_routes.py` | 65 | Read-only route for the five built-in CC0 studio lighting environments (``rendering/assets/stage10_cc0_probes.npz``). | `_thumbnail`:30, `_catalog`:42, `studio_environments`:63 | real | app/roop_3d/server/studio_environment_routes.py:30 |
| `app/roop_3d/server/target_lock.py` | 376 | Target identity locking from any frame (Stage 12). | `store`:35, `ref_of`:42, `annotate`:62, `_pick`:87, `FaceSelector`:100, `LockFromFrame`:114 | real | app/roop_3d/server/target_lock.py:35 |
| `app/roop_3d/server/unified_state.py` | 245 | Unified State Store and state router for Roop 3D. | `RenderMode`:14, `RenderModePayload`:54, `SourceFacePayload`:58, `ActiveProcessorsPayload`:65, `UnifiedStateStore`:69, `get_state`:173 | real | app/roop_3d/server/unified_state.py:14 |
| `app/roop_3d/server/video_pipeline_routes.py` | 99 | Stage 17 routes: one-shot upload / import and the verified job download. | `_store`:34, `_unpublish`:59, `upload_media`:71, `import_faceset`:78, `download_job_video`:85 | real | app/roop_3d/server/video_pipeline_routes.py:34 |
| `app/roop_3d/server/workstation.py` | 747 | Local workstation assets, projects, and honest reconstruction inspection. | `UploadRequest`:34, `upload_path`:39, `create_upload`:46, `append_upload`:63, `upload_chunk`:81, `coverage`:85 | real | app/roop_3d/server/workstation.py:34 |
| `app/roop_3d/server/workstation_preview.py` | 501 | On-demand, bounded local inference previews; no sessions or weights at API boot. | `SweepAxis`:42, `PreviewRequest`:55, `encode`:80, `face_readouts`:91, `collect_composite`:104, `preview_preroll`:154 | real | app/roop_3d/server/workstation_preview.py:42 |
| `app/roop_3d/server/ws.py` | 319 | Bounded process-local event fanout; workers publish after durable commits. | `EventBase`:18, `JobStateEvent`:24, `JobProgressEvent`:32, `TelemetryEvent`:38, `PipelineStatusEvent`:48, `TrackingTelemetryEvent`:72 | real | app/roop_3d/server/ws.py:18 |

### `app/roop_3d/jobs/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/jobs/__init__.py` | 2 | Job execution package boundary. |  | real | app/roop_3d/jobs/__init__.py:1 |
| `app/roop_3d/jobs/project_store.py` | 346 | Versioned owned .r3dp manifests; raw media, embeddings and secrets stay outside. | `settings_digest`:21, `ViewportPreferences`:25, `ProjectPreview`:36, `ProjectRender`:43, `ProjectDraft`:50, `ProjectInput`:58 | real | app/roop_3d/jobs/project_store.py:21 |
| `app/roop_3d/jobs/schema.py` | 238 | Persistent job contracts and the complete legal transition graph. | `now`:23, `JobState`:27, `require_transition`:52, `MaskKeyframe`:57, `JobRequest`:73, `JobProgress`:155 | real | app/roop_3d/jobs/schema.py:23 |
| `app/roop_3d/jobs/store.py` | 262 | Transactional on-disk state, persistent idempotency and worker acknowledgments. | `IdempotencyEntry`:19, `JobDatabase`:24, `JobStore`:30 | real | app/roop_3d/jobs/store.py:19 |
| `app/roop_3d/jobs/worker.py` | 377 | Persistent threaded job worker with pause, cancellation and OOM retry. | `_hud`:32, `JobWorker`:40 | real | app/roop_3d/jobs/worker.py:32 |

### `app/roop_3d/queue/` (3 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/queue/__init__.py` | 6 | Stage 23 queue package: the 4-strategy Batch Matrix engine and the 10-state job lifecycle. |  | real | app/roop_3d/queue/__init__.py:1 |
| `app/roop_3d/queue/batch_matrix.py` | 615 | Four-strategy combinatorial Batch Matrix engine for roop-3d (#/batch). | `BatchStrategy`:24, `sanitize_filename`:149, `format_output_name`:155, `RecipeConfig`:191, `TargetDescriptor`:199, `JobManifest`:208 | real | app/roop_3d/queue/batch_matrix.py:24 |
| `app/roop_3d/queue/lifecycle.py` | 71 | The 10 user-facing job lifecycle states (Stage 23) as a lossless projection of the persisted queue state. | `LifecycleState`:28, `lifecycle_of`:62 | real | app/roop_3d/queue/lifecycle.py:28 |

### `app/roop_3d/reliability/` (13 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/reliability/__init__.py` | 2 | Stage 18: long-duration reliability engineering (leak detection, crash recovery, local diagnostics). |  | real | app/roop_3d/reliability/__init__.py:1 |
| `app/roop_3d/reliability/classify.py` | 96 | Turn an unexpected exception into a stable, actionable, *non-silent* job failure. | `Classified`:18, `_chain`:31, `classify`:39, `classify_domain`:85 | real | app/roop_3d/reliability/classify.py:18 |
| `app/roop_3d/reliability/crash_child.py` | 33 | Child process for the hard-kill scenario: renders a job slowly until the parent terminates it. | `main`:19 | test helper | app/roop_3d/reliability/crash_child.py:19 |
| `app/roop_3d/reliability/disk.py` | 109 | Disk-space budgeting for long renders. | `is_disk_full`:29, `SpaceEstimate`:48, `estimate`:62, `free_bytes`:70, `require`:77, `tree_bytes`:88 | real | app/roop_3d/reliability/disk.py:29 |
| `app/roop_3d/reliability/growth.py` | 203 | Bounded-growth analysis: decide whether a resource series is a plateau (healthy) or a leak. | `theil_sen`:23, `positive_fraction`:33, `_thin`:42, `MetricBound`:51, `BoundedProfile`:64, `MetricVerdict`:79 | real | app/roop_3d/reliability/growth.py:23 |
| `app/roop_3d/reliability/harness.py` | 275 | Test and soak harness: synthetic long videos, a stub inference stage and a ready-to-run job rig. | `make_video`:41, `faceset_document`:67, `StubReader`:80, `StubSession`:86, `Rig`:127, `fixture_catalog`:173 | test/soak rig (stub inference stage by design) | app/roop_3d/reliability/harness.py:41 |
| `app/roop_3d/reliability/ledger.py` | 154 | Lifecycle ledger: every long-lived native resource is opened and closed through a counted token. | `Token`:19, `_Stat`:36, `ResourceLedger`:43 | real | app/roop_3d/reliability/ledger.py:19 |
| `app/roop_3d/reliability/monitor.py` | 153 | Always-on, bounded resource monitor: peaks, growth verdicts, stage timings and a sanitized warning buffer. | `WarningBuffer`:21, `ResourceMonitor`:44 | real | app/roop_3d/reliability/monitor.py:21 |
| `app/roop_3d/reliability/probe.py` | 367 | Dependency-free process, GPU and runtime probes used by the Stage 18 reliability tooling. | `process_create_time`:112, `process_alive`:124, `kill_process`:138, `child_pids`:152, `descendant_pids`:173, `ProcessMemory`:199 | real | app/roop_3d/reliability/probe.py:112 |
| `app/roop_3d/reliability/processes.py` | 235 | Orphan-process prevention for the FFmpeg children Roop 3D spawns. | `bind_children_to_process_lifetime`:36, `spawn_kwargs`:89, `ProcessRegistry`:103, `terminate`:209, `terminate_tree`:226 | real | app/roop_3d/reliability/processes.py:36 |
| `app/roop_3d/reliability/recovery.py` | 499 | Crash-consistency audit and repair for jobs, checkpoints, outputs and the queue. | `Finding`:56, `AuditReport`:70, `ReliabilityAuditor`:93, `JobStore_relative`:486, `reconcile_on_startup`:490 | real | app/roop_3d/reliability/recovery.py:56 |
| `app/roop_3d/reliability/report.py` | 289 | Local, redacted diagnostic report. | `_short`:54, `Redactor`:58, `_hardware`:122, `_libraries`:142, `_models`:158, `_jobs`:179 | real | app/roop_3d/reliability/report.py:54 |
| `app/roop_3d/reliability/soak.py` | 358 | Soak scenarios and CLI: drive the real worker/checkpoint/decode/encode path for a long time and judge the resource series. | `Sampler`:40, `_child_names`:79, `_invariants`:85, `_finish`:101, `_fresh_root`:112, `scenario_long_video`:119 | real | app/roop_3d/reliability/soak.py:40 |

### `app/roop_3d/telemetry/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/telemetry/__init__.py` | 5 | Terminal and diagnostic telemetry: the Pinokio HUD, the 14-section runtime report and the TensorRT build heartbeat. |  | real | app/roop_3d/telemetry/__init__.py:1 |
| `app/roop_3d/telemetry/pinokio_terminal.py` | 666 | Pinokio terminal HUD: UTF-8/unbuffered enforcement, one live in-place line, milestone logs and the boot/job report. | `enforce_terminal_environment`:28, `child_environment`:47, `_enable_windows_vt`:52, `defuse`:78, `paint`:84, `visible_len`:89 | real | app/roop_3d/telemetry/pinokio_terminal.py:28 |
| `app/roop_3d/telemetry/runtime_report.py` | 511 | The 14-section structured runtime report (terminal at boot / job start, ``GET /api/v1/system/runtime-report``, UI drawer). | `Section`:51, `show`:67, `tilde`:79, `mib`:88, `_Context`:92, `_available`:126 | real | app/roop_3d/telemetry/runtime_report.py:51 |
| `app/roop_3d/telemetry/trt_heartbeat.py` | 199 | Heartbeat while TensorRT builds an engine, so neither the terminal nor the React UI looks frozen. | `precision_label`:30, `format_shape`:38, `heartbeat_line`:54, `_default_vram`:61, `TerminalHandle`:71, `TrtCompilationHeartbeat`:75 | real | app/roop_3d/telemetry/trt_heartbeat.py:30 |
| `app/roop_3d/telemetry/trt_heartbeat_sidecar.py` | 92 | Out-of-process TensorRT build heartbeat: the only way to keep a terminal line alive during a cold engine build. | `_vram`:27, `run`:37, `main`:71 | real | app/roop_3d/telemetry/trt_heartbeat_sidecar.py:27 |

### `app/roop_3d/security/` (6 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/security/__init__.py` | 49 | Roop 3D security, privacy, and synthetic media protection module. |  | real | app/roop_3d/security/__init__.py:1 |
| `app/roop_3d/security/artifact_cleaner.py` | 135 | Safe deletion and secure lifecycle management of temporary media and biometric artifacts. | `SafeArtifactCleaner`:15 | real | app/roop_3d/security/artifact_cleaner.py:15 |
| `app/roop_3d/security/content_validator.py` | 150 | Strict file type, content magic bytes, and path safety validation. | `validate_safe_filename`:30, `validate_safe_path`:40, `validate_image_header`:63, `validate_video_container`:94, `validate_faceset_archive_bytes`:144 | real | app/roop_3d/security/content_validator.py:30 |
| `app/roop_3d/security/log_sanitizer.py` | 82 | Log sanitization, secret redaction, and biometric privacy filters for application logging. | `sanitize_log_message`:9, `SecurityRedactionFilter`:53, `install_log_sanitizer`:68 | real | app/roop_3d/security/log_sanitizer.py:9 |
| `app/roop_3d/security/privacy.py` | 146 | Privacy enforcement, biometric permission validation, and network isolation guards. | `ConsentRecord`:16, `TargetFootagePermission`:26, `ConsentVerifier`:37 | real | app/roop_3d/security/privacy.py:16 |
| `app/roop_3d/security/synthetic_media.py` | 163 | Synthetic media disclosure, cryptographic provenance, identity honesty, and safety safeguards. | `SyntheticMediaDisclosure`:25, `AuditRecord`:36, `IdentityHonestyPolicy`:49, `SafetyGuard`:89, `build_provenance_manifest`:116 | real | app/roop_3d/security/synthetic_media.py:25 |

### `app/roop_3d/installer/` (6 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/installer/__init__.py` | 11 | Roop 3D Offline Installer and Environment Manager package. |  | real | app/roop_3d/installer/__init__.py:1 |
| `app/roop_3d/installer/model_manager.py` | 269 | Offline model bundle importer and integrity verifier for Roop 3D. | `validate_archive_entry`:32, `check_archive_safety`:57, `ModelManager`:67, `format_missing_model_error`:242 | real | app/roop_3d/installer/model_manager.py:32 |
| `app/roop_3d/installer/offline_cache.py` | 159 | Offline dependency and health verification for Roop 3D. | `check_offline_dependency_cache`:39, `check_offline_readiness`:73 | real | app/roop_3d/installer/offline_cache.py:39 |
| `app/roop_3d/installer/preflight.py` | 542 | Environment preflight checker for Roop 3D. | `PreflightCheckResult`:39, `check_windows_version`:58, `check_python_version`:98, `check_node_version`:140, `check_cpu`:166, `check_system_ram`:205 | real | app/roop_3d/installer/preflight.py:39 |
| `app/roop_3d/installer/update_manager.py` | 234 | Update inspection, dependency change detection, configuration backup, and safe rollback. | `_git`:25, `check_update_status`:32, `create_configuration_backup`:123, `rollback_to_backup`:179 | real | app/roop_3d/installer/update_manager.py:25 |
| `app/roop_3d/installer/verification.py` | 178 | Installation verification and interrupted installation recovery for Roop 3D. | `verify_installation`:46, `recover_interrupted_installation`:122 | real | app/roop_3d/installer/verification.py:46 |

### `app/roop_3d/bridges/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/bridges/__init__.py` | 2 | External integration package boundary. |  | real | app/roop_3d/bridges/__init__.py:1 |
| `app/roop_3d/bridges/cloud_stub.py` | 24 | Disabled remote adapter. Contains no URLs, credentials, HTTP client, or uploads. | `CloudStub`:6 | stub (intentionally disabled; no URLs/HTTP client) | app/roop_3d/bridges/cloud_stub.py:6 |
| `app/roop_3d/bridges/comfyui_bridge.py` | 442 | Bounded loopback-only ComfyUI API bridge; no model imports or media writes. | `LoopbackConnect`:33, `BridgeConfig`:40, `PromptHandle`:75, `OutputRef`:90, `BridgeEvent`:105, `decode_image`:114 | real | app/roop_3d/bridges/comfyui_bridge.py:33 |
| `app/roop_3d/bridges/provider_protocol.py` | 175 | Typed remote boundary with per-job signed consent and safe diagnostics. | `safe_filename`:21, `MediaDescriptor`:30, `RemoteJob`:38, `ProviderCapabilities`:60, `ProviderStatus`:67, `ProviderProtocol`:72 | real protocol; no concrete remote provider in slice (only CloudStub) | app/roop_3d/bridges/provider_protocol.py:21 |

### `app/roop_3d/editing/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/editing/__init__.py` | 25 | Pre-render editing of the tracked target performance: emotion offsets and audio-locked lip/jaw motion. |  | real | app/roop_3d/editing/__init__.py:1 |
| `app/roop_3d/editing/expression_audio_director.py` | 385 | ExpressionAndAudioDirector: user emotion offsets in psi-space, and an audio lock for lip/jaw motion under occlusion. | `EmotionBasis`:48, `EmotionOffsets`:86, `DirectorSettings`:96, `AudioViseme`:113, `AudioVisemeTrack`:120, `_smooth`:201 | real | app/roop_3d/editing/expression_audio_director.py:48 |

### `app/roop_3d/evaluation/` (8 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/evaluation/__init__.py` | 73 | Roop 3D Objective AI Visual Quality Evaluation Framework (Stage 11). |  | real | app/roop_3d/evaluation/__init__.py:1 |
| `app/roop_3d/evaluation/benchmark_db.py` | 158 | Historical Benchmark Database and Model Provenance Store (Stage 11). | `ModelProvenance`:25, `BenchmarkRecord`:34, `BenchmarkDatabase`:53 | real | app/roop_3d/evaluation/benchmark_db.py:25 |
| `app/roop_3d/evaluation/blind_review.py` | 248 | Blinded A/B Human-Reviewed Comparison System (Stage 11). | `HumanReviewScore`:29, `BlindComparisonPair`:46, `BlindReviewSummary`:60, `BlindedReviewSystem`:83, `BlindComparisonSession`:244 | real | app/roop_3d/evaluation/blind_review.py:29 |
| `app/roop_3d/evaluation/dataset.py` | 479 | Controlled, consented golden-video dataset and scenario generator (Stage 11). | `SceneCategory`:42, `GoldenFrame`:57, `GoldenScene`:74, `_generate_synthetic_canvas`:121, `_draw_synthetic_face`:138, `generate_synthetic_scene`:279 | real | app/roop_3d/evaluation/dataset.py:42 |
| `app/roop_3d/evaluation/harness.py` | 259 | Evaluation Harness for Comparative Model and Pipeline Benchmarking (Stage 11). | `EngineProfile`:31, `EngineOutput`:41, `HarnessRunResult`:52, `EvaluationHarness`:63 | real | app/roop_3d/evaluation/harness.py:31 |
| `app/roop_3d/evaluation/metrics.py` | 444 | Objective Visual Quality and Fidelity Metrics Suite (Stage 11). | `compute_psnr`:32, `compute_ssim`:48, `compute_perceptual_distance`:85, `compute_cosine_similarity`:104, `FrameMetricResult`:116, `SceneMetricSummary`:134 | real | app/roop_3d/evaluation/metrics.py:32 |
| `app/roop_3d/evaluation/regression.py` | 253 | Regression Detection and Baseline Improvement Gate (Stage 11). | `AcceptanceVerdict`:27, `RegressionCheckResult`:34, `CandidateEvaluationReport`:47, `RegressionDetector`:64 | real (baseline = regression baseline, not baseline-quality impl) | app/roop_3d/evaluation/regression.py:27 |
| `app/roop_3d/evaluation/reports.py` | 243 | Quality Report and Markdown Export Generator (Stage 11). | `QualityReportGenerator`:25 | real | app/roop_3d/evaluation/reports.py:25 |

### `app/roop_3d/tools/` (20 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/tools/__init__.py` | 2 | Roop 3D repository maintenance commands. |  | real | app/roop_3d/tools/__init__.py:1 |
| `app/roop_3d/tools/annotate_onnx_shapes.py` | 78 | Write `<name>.static.onnx`: the exported encoder with its output shapes stated statically. | `annotate`:25, `main`:68 | real | app/roop_3d/tools/annotate_onnx_shapes.py:25 |
| `app/roop_3d/tools/benchmark_async_pipeline.py` | 205 | Defect #3 saturation benchmark: serial vs asynchronous multi-stream pipeline, real models, real hardware counters. | `build_session`:36, `run`:46, `main`:143 | real | app/roop_3d/tools/benchmark_async_pipeline.py:36 |
| `app/roop_3d/tools/benchmark_compositor.py` | 96 | Hardware-gated synthetic compositor campaign, never neural inference FPS. | `benchmark_compositor`:12 | real | app/roop_3d/tools/benchmark_compositor.py:12 |
| `app/roop_3d/tools/build_universal_manifest.py` | 342 | Maintainer tool: (re)generate ``roop_3d/audit/universal_model_manifest.json``. | `lock_digest`:78, `asset_entries`:83, `normalize_license`:179, `get_json`:189, `sha256_url`:194, `index_hf`:208 | real | app/roop_3d/tools/build_universal_manifest.py:78 |
| `app/roop_3d/tools/cache_cleanup.py` | 62 | Dry-run report for stale temporary files in the package-owned cache only. | `CacheCandidate`:14, `find_candidates`:20, `main`:46 | real | app/roop_3d/tools/cache_cleanup.py:14 |
| `app/roop_3d/tools/check_topology_assets.py` | 84 | Report which morphable-topology assets are present and say exactly how to get the missing ones. | `check`:37, `render`:54, `main`:72 | real | app/roop_3d/tools/check_topology_assets.py:37 |
| `app/roop_3d/tools/dense_face_bundle.py` | 278 | Local native bundle tooling. From app/: python -m roop_3d.tools.dense_face_bundle --help. | `_json`:32, `_asset`:40, `_embedded_onnx`:49, `_cpu_session`:114, `validate_bundle`:125, `assemble`:150 | real | app/roop_3d/tools/dense_face_bundle.py:32 |
| `app/roop_3d/tools/dense_face_downloads.py` | 69 | Exact public upstream bytes used by the native dense-face export path. | `fetch_public`:35 | real | app/roop_3d/tools/dense_face_downloads.py:35 |
| `app/roop_3d/tools/git_guard.py` | 111 | Enforce the independent Roop 3D Git tree and dedicated remote identity. | `BoundaryError`:10, `allowed_path`:17, `git`:32, `check_remote_url`:35, `check_project`:41, `check_names`:51 | real | app/roop_3d/tools/git_guard.py:10 |
| `app/roop_3d/tools/gpu_saturation.py` | 138 | Measure whether the runtime path saturates the GPU: TensorRT FP16 + ``GpuIoBinding`` (device-resident tensors) + micro-batching. | `available`:24, `_model`:45, `measure`:64, `_measure`:76, `main`:122 | real | app/roop_3d/tools/gpu_saturation.py:24 |
| `app/roop_3d/tools/install_git_guard.py` | 21 | Install hooks only in the independent Roop 3D repository. | `install`:6, `main`:10 | real | app/roop_3d/tools/install_git_guard.py:6 |
| `app/roop_3d/tools/occlusion_acceptance.py` | 448 | Stage 5 acceptance harness: standard vs robust compositor on REAL local video. | `Scenario`:45, `_video`:69, `read_clip`:76, `find_scene_cuts`:89, `find_scene_cut`:110, `Observed`:116 | real | app/roop_3d/tools/occlusion_acceptance.py:45 |
| `app/roop_3d/tools/prepare_dense_face.py` | 348 | Prepare genuine upstream networks and model assets for the dense face runtime. | `sha256`:21, `source_module`:29, `export_network`:48, `export_component`:95, `prepare_bfm`:154, `prepare_flame_anatomy`:196 | real | app/roop_3d/tools/prepare_dense_face.py:21 |
| `app/roop_3d/tools/prepare_relighting_case.py` | 74 | Prepare a real-photo ArcFace relighting fixture from an owned dense FaceSet. | `prepare`:22, `main`:64 | real | app/roop_3d/tools/prepare_relighting_case.py:22 |
| `app/roop_3d/tools/register_bfm_flame.py` | 216 | Register 3DDFA-V3 BFM-35709 and FLAME-2023-Open surfaces and derive BFM anatomical masks. | `_sha`:27, `closest_on_triangles`:35, `similarity`:74, `register`:85, `main`:207 | real | app/roop_3d/tools/register_bfm_flame.py:27 |
| `app/roop_3d/tools/temporal_stability.py` | 131 | Reproducible 120-frame synthetic benchmark for temporal detail stabilization. | `_highpass`:20, `benchmark_temporal_stability`:25, `main`:114 | real | app/roop_3d/tools/temporal_stability.py:20 |
| `app/roop_3d/tools/verify_e2e_workstation.py` | 214 | Stage 7 end-to-end gate: the built single-origin workstation plus the remediation pipeline contracts. | `scratch`:30, `Report`:46, `check_built_bundle`:68, `check_single_origin_server`:87, `check_strict_trt_profile_binding`:122, `check_three_stage_render_and_timings`:145 | real | app/roop_3d/tools/verify_e2e_workstation.py:30 |
| `app/roop_3d/tools/verify_stream_hardware.py` | 119 | Opt-in native synthetic acceptance; no production neural fidelity claims. | `packet_digest`:22, `verify_hardware`:32 | real | app/roop_3d/tools/verify_stream_hardware.py:22 |
| `app/roop_3d/tools/verify_stream_integration.py` | 609 | Real synthetic-clip gate; no production model/hardware claim is inferred. | `_ring_child`:28, `verify_ring`:40, `create_clip`:63, `verify_geometry`:95, `verify_emap`:117, `verify_fp32_precision_rewrite`:134 | real | app/roop_3d/tools/verify_stream_integration.py:28 |

### `app/roop_3d/docs/` (24 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/docs/ASYNC_PIPELINE.md` | 154 | # Asynchronous multi-stream render pipeline and binary live transport (Defect #3) |  | doc | app/roop_3d/docs/ASYNC_PIPELINE.md:1 |
| `app/roop_3d/docs/FINAL_RELEASE_CERTIFICATION.md` | 333 | # Stage 20: Final Roop 3D Release Certification Audit Report |  | doc | app/roop_3d/docs/FINAL_RELEASE_CERTIFICATION.md:1 |
| `app/roop_3d/docs/FINAL_RELEASE_QUALIFICATION.md` | 258 | # Stage 10: Roop 3D Adversarial Validation and Release Audit |  | doc | app/roop_3d/docs/FINAL_RELEASE_QUALIFICATION.md:1 |
| `app/roop_3d/docs/HEAD_TRACKING_STAGE4.md` | 101 | # Head and gaze tracking (Stage 4, tracking series) |  | doc | app/roop_3d/docs/HEAD_TRACKING_STAGE4.md:1 |
| `app/roop_3d/docs/IDENTITY_ASSET_STAGE3.md` | 191 | # Stage 3: Personalized 3D Source Identity Reconstruction |  | doc | app/roop_3d/docs/IDENTITY_ASSET_STAGE3.md:1 |
| `app/roop_3d/docs/INFERENCE_PROFILING_STAGE2.md` | 123 | # Stage 2: Roop 3D Inference Profiling and Performance Engineering |  | doc | app/roop_3d/docs/INFERENCE_PROFILING_STAGE2.md:1 |
| `app/roop_3d/docs/OCCLUSION_STAGE5.md` | 227 | # Stage 5: Advanced Occlusion and High-Fidelity Compositor |  | doc | app/roop_3d/docs/OCCLUSION_STAGE5.md:1 |
| `app/roop_3d/docs/OFFLINE_3D_TARGET_FACE_PLAN.md` | 123 | # Offline 3D-Aware Target Face Replacement â€” Implementation Plan |  | doc | app/roop_3d/docs/OFFLINE_3D_TARGET_FACE_PLAN.md:1 |
| `app/roop_3d/docs/PHOTOREAL_VIDEO_STAGE17.md` | 119 | # Stage 17 - photorealistic 3D swap core and the FFmpeg end-to-end video job |  | doc | app/roop_3d/docs/PHOTOREAL_VIDEO_STAGE17.md:1 |
| `app/roop_3d/docs/REENACTMENT_STAGE4.md` | 148 | # Stage 4: Temporal 3D Face Reenactment Engine |  | doc | app/roop_3d/docs/REENACTMENT_STAGE4.md:1 |
| `app/roop_3d/docs/RELEASE_CHECKLIST.md` | 156 | ï»¿# Roop 3D: Release Checklist & Operational Runbook |  | doc | app/roop_3d/docs/RELEASE_CHECKLIST.md:1 |
| `app/roop_3d/docs/RENDER_AND_PREVIEW_PERFORMANCE.md` | 96 | # Render and preview performance (Nov 12 2026) |  | doc | app/roop_3d/docs/RENDER_AND_PREVIEW_PERFORMANCE.md:1 |
| `app/roop_3d/docs/ROOP3D_MASTER_TECHNICAL_AUDIT.md` | 438 | # Roop 3D: Master Technical Re-Audit & Comprehensive Implementation Plan |  | doc | app/roop_3d/docs/ROOP3D_MASTER_TECHNICAL_AUDIT.md:1 |
| `app/roop_3d/docs/SEAM_EXPRESSION_STAGE6.md` | 134 | # Stage 6 - boundary seams and continuous expression direction |  | doc | app/roop_3d/docs/SEAM_EXPRESSION_STAGE6.md:1 |
| `app/roop_3d/docs/SILHOUETTE_REPLACEMENT_STAGE6B.md` | 61 | # Stage 6B - silhouette replacement (overhang inpaint, perimeter cage, swap scope) |  | doc | app/roop_3d/docs/SILHOUETTE_REPLACEMENT_STAGE6B.md:1 |
| `app/roop_3d/docs/STAGE06_MODEL_HUB.md` | 105 | # Stage 06 â€” Local model hub and engine architecture |  | doc | app/roop_3d/docs/STAGE06_MODEL_HUB.md:1 |
| `app/roop_3d/docs/STAGE07_WORKSTATION_UI.md` | 170 | # Stage 7: Professional workstation UI (Studio layout) |  | doc | app/roop_3d/docs/STAGE07_WORKSTATION_UI.md:1 |
| `app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md` | 257 | # Stage 8: Complete offline video workflow |  | doc | app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md:1 |
| `app/roop_3d/docs/STAGE09_OPTIMIZATION.md` | 121 | # Stage 9: Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization |  | doc | app/roop_3d/docs/STAGE09_OPTIMIZATION.md:1 |
| `app/roop_3d/docs/STAGE11_QUALITY_FRAMEWORK.md` | 140 | # Stage 11: Roop 3D AI Visual Quality Evaluation Framework |  | doc | app/roop_3d/docs/STAGE11_QUALITY_FRAMEWORK.md:1 |
| `app/roop_3d/docs/STAGE12_GLB_BLENDER_IO.md` | 68 | # Stage 12: binary glTF export and the optional Blender bridge |  | doc | app/roop_3d/docs/STAGE12_GLB_BLENDER_IO.md:1 |
| `app/roop_3d/docs/STAGE12_MODEL_GOVERNANCE.md` | 100 | # Stage 12 â€” Model governance and secure model management |  | doc | app/roop_3d/docs/STAGE12_MODEL_GOVERNANCE.md:1 |
| `app/roop_3d/docs/STAGE13_SECURITY_THREAT_MODEL.md` | 265 | # Stage 13 â€” Roop 3D Security, Privacy and Synthetic Media Threat Model |  | doc | app/roop_3d/docs/STAGE13_SECURITY_THREAT_MODEL.md:1 |
| `app/roop_3d/docs/UNIVERSAL_MODEL_DOWNLOADER.md` | 99 | # Universal model downloader (Stage 11, model-distribution series) |  | doc | app/roop_3d/docs/UNIVERSAL_MODEL_DOWNLOADER.md:1 |

### `app/roop/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop/__init__.py` | 5 | Shim module forwarding to roop_3d. |  | shim | app/roop/__init__.py:1 |
| `app/roop/core/__init__.py` | 2 | Shared runtime primitives for the standalone Roop application. |  | shim | app/roop/core/__init__.py:1 |
| `app/roop/core/model_downloader.py` | 3 | Model downloader re-export for roop.core. |  | shim | app/roop/core/model_downloader.py:1 |
| `app/roop/core/ort_session.py` | 3 |  |  | shim | app/roop/core/ort_session.py:1 |
| `app/roop/relighting_3d.py` | 3 | Forwarding module for Stage 17 3D Relighting. |  | shim | app/roop/relighting_3d.py:1 |

### `app/core/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/core/__init__.py` | 2 | Runtime bootstrap helpers for the standalone app. |  | real | app/core/__init__.py:1 |
| `app/core/trt_bootstrap.py` | 205 | Register packaged and system CUDA/TensorRT library directories before ORT loads. | `native_library_directories`:22, `bootstrap_native_library_paths`:130, `_preload_cuda_tensorrt_shared_libraries`:160, `expected_nvinfer_filename`:188 | real | app/core/trt_bootstrap.py:22 |

### `app/diagnostics/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/diagnostics/__init__.py` | 29 | Diagnostics and telemetry package for pipeline auditing. |  | real | app/diagnostics/__init__.py:1 |
| `app/diagnostics/pipeline_audit.py` | 384 | Unified Pipeline Diagnostic & Telemetry Framework (Non-breaking). | `AuditError`:22, `StageQualityDegradationError`:26, `ExecutionProviderFallbackError`:30, `EnhancerBypassError`:34, `AuditConfig`:39, `TensorShapeRecord`:57 | real | app/diagnostics/pipeline_audit.py:22 |

### `app/tests/` (8 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/tests/pytest.ini` | 4 | [pytest] |  | config | app/tests/pytest.ini:1 |
| `app/tests/test_3d_relighting.py` | 8 | Requested Stage 17 gate path; implementation lives in the owned test package. |  | forwarder (8 lines) | app/tests/test_3d_relighting.py:1 |
| `app/tests/test_flame_and_liveportrait.py` | 117 | Mandatory Stage 16 acceptance. Missing hardware/assets FAIL, never skip. | `test_generator_warping_cuda_and_tensorrt_native`:25, `test_hififace_256_with_exact_arcface_converter`:48, `test_independent_eyeball_gaze_on_rendered_swap`:64 | real | app/tests/test_flame_and_liveportrait.py:25 |
| `app/tests/test_heterogeneous_scheduler.py` | 8 | Requested Stage 15 gate path; implementation lives in the owned test package. |  | forwarder (8 lines) | app/tests/test_heterogeneous_scheduler.py:1 |
| `app/tests/test_nv_cvcuda_pipeline.py` | 8 | Requested Stage 14 gate path; implementation lives in the owned test package. |  | forwarder (8 lines) | app/tests/test_nv_cvcuda_pipeline.py:1 |
| `app/tests/test_pipeline_audit.py` | 143 | Unit tests for pipeline_audit diagnostic module. | `test_audit_config_defaults`:14, `test_stage3_downsampling_detection_non_debug`:22, `test_stage3_downsampling_fails_loud_in_debug`:46, `test_enhancer_bypass_detection`:61, `test_enhancer_bypass_fails_loud_in_debug`:97, `test_execution_provider_fallback_detection`:110 | real | app/tests/test_pipeline_audit.py:14 |
| `app/tests/test_stage27_certification.py` | 182 | Final certification: UltraMax native sharpness, full-head 3D replacement from .fsz, GPU saturation, offline install, UI coverage. | `run_evidence`:31, `test_a_ultramax_native_resolution_detail_survives_the_handoff`:55, `test_a_measured_sharpness_gain_is_reported_not_just_asserted`:60, `test_b_full_head_geometry_is_replaced_from_the_fsz_not_just_the_inner_face`:85, `gpu_path`:96, `best_of`:104 | real | app/tests/test_stage27_certification.py:31 |
| `app/tests/test_standalone_install.py` | 261 | Stage 14 Hardening & Standalone Packaging Verification Suite. | `_git`:58, `test_app_state_directories_are_real_directories_not_links`:65, `test_app_state_directories_are_gitignored_and_untracked`:80, `test_dedicated_virtual_environment`:85, `_external_hosts`:91, `test_production_ui_source_references_zero_external_hosts_or_cdns`:95 | real | app/tests/test_standalone_install.py:58 |

### `app/roop_3d/ (top-level)` (40 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/.gitattributes` | 3 | # Keep portable Git hook interpreters executable after Windows checkouts. |  | config | app/roop_3d/.gitattributes:1 |
| `app/roop_3d/CONTRIBUTING.md` | 6 | # Standalone development |  | doc | app/roop_3d/CONTRIBUTING.md:1 |
| `app/roop_3d/README.md` | 53 | # Roop 3D Python package |  | doc | app/roop_3d/README.md:1 |
| `app/roop_3d/__init__.py` | 8 | Roop 3D: isolated local video-processing application. |  | real | app/roop_3d/__init__.py:1 |
| `app/roop_3d/batch_matrix.py` | 10 | Compatibility alias: the Batch Matrix engine now lives in :mod:`roop_3d.queue.batch_matrix` (Stage 23). |  | compat alias -> queue/batch_matrix.py | app/roop_3d/batch_matrix.py:1 |
| `app/roop_3d/cvcuda_ops.py` | 131 | Optional CV-CUDA preprocessing, with a device-only Torch reference path. | `FaceOps`:13 | real | app/roop_3d/cvcuda_ops.py:13 |
| `app/roop_3d/expression_restorer.py` | 265 | Stage 2.5: current-frame expression residuals with identity/pose locked. | `eye_aspect_ratio`:17, `PortraitGeometry`:27, `edit_motion`:55, `close_apertures`:118, `ExpressionRestorer`:152 | real | app/roop_3d/expression_restorer.py:17 |
| `app/roop_3d/flame_3d_engine.py` | 216 | FLAME coefficient extraction with explicit camera, mesh and gaze evidence. | `finite_array`:16, `EyeGaze`:25, `FlameParameters`:47, `blend_shape`:67, `FlameEncoder`:75, `MeshEstimator`:79 | real | app/roop_3d/flame_3d_engine.py:16 |
| `app/roop_3d/gpu_compositor.py` | 459 | Optional Torch CUDA compositor. Images are NCHW RGB float32 in [0, 1]. | `_torch`:14, `warp`:19, `invert_affine`:25, `gaussian`:36, `morphology`:41, `face_feather`:53 | real | app/roop_3d/gpu_compositor.py:14 |
| `app/roop_3d/gpu_transforms.py` | 89 | CUDA-resident image transforms shared by the streaming inference stages. | `warp_affine_gpu`:11, `normalize_model_input`:36, `model_output_to_rgb01`:52, `gaussian_feather_mask`:68 | real | app/roop_3d/gpu_transforms.py:11 |
| `app/roop_3d/hardware_streamer.py` | 519 | Bounded host-frame transport and explicit FFmpeg NVDEC/NVENC boundaries. | `FrameMetadata`:27, `SharedMemoryFrameRing`:35, `_binary`:145, `_process_environment`:152, `stream_color_supported`:170, `NvdecFrameSource`:189 | real | app/roop_3d/hardware_streamer.py:27 |
| `app/roop_3d/heterogeneous_scheduler.py` | 284 | Heterogeneous Intel + NVIDIA Hybrid Execution Scheduler. | `VideoCodecMultiplexer`:56, `HeterogeneousScheduler`:135 | real code, optional/hardware-gated [INFERENCE] | app/roop_3d/heterogeneous_scheduler.py:56 |
| `app/roop_3d/hires_handoff.py` | 374 | High-resolution Stage 2 -> Stage 3 handoff. | `_affine`:31, `invert_affine_np`:38, `hires_scale`:47, `scale_matrix_output`:54, `scale_matrix_input`:62, `mapped_bounds`:72 | real | app/roop_3d/hires_handoff.py:31 |
| `app/roop_3d/job_queue.py` | 440 | Stage 9: checkpointed, segment-safe render jobs on top of the persistent ``QueueManager``. | `SegmentEntry`:47, `JobCheckpoint`:65, `_default_provider`:82, `_sha256`:90, `count_video_packets`:98, `iter_source_frames`:105 | real | app/roop_3d/job_queue.py:47 |
| `app/roop_3d/liveportrait_trt.py` | 156 | Modular LivePortrait retargeting using explicitly prepared checked sessions. | `GazeRetargeter`:18, `OnnxGazeRetargeter`:22, `LivePortraitNames`:41, `LivePortraitTRT`:53 | real | app/roop_3d/liveportrait_trt.py:18 |
| `app/roop_3d/masking_engine.py` | 238 | 19-Class Region Masking + DFL XSeg + Text-Prompted CLIPSeg Masking Engine. | `MaskingEngine`:47 | real | app/roop_3d/masking_engine.py:47 |
| `app/roop_3d/mesh_raycaster.py` | 211 | Intel Embree / SIMD 3D Mesh Self-Occlusion & Raycast Z-Buffer. | `VisibilityResult`:30, `MeshRaycaster`:42 | real, optional Embree/SIMD [INFERENCE] | app/roop_3d/mesh_raycaster.py:30 |
| `app/roop_3d/model_registry.py` | 731 | Unified Declarative Model Registry & Resilient Auto-Downloader. | `ModelManifestEntry`:41, `UnifiedModelRegistry`:503, `ResilientModelDownloader`:600 | real | app/roop_3d/model_registry.py:41 |
| `app/roop_3d/nv_video_codec.py` | 211 | Lazy NVIDIA codec adapters. Pixel buffers stay on CUDA; packets/metadata do not. | `GPUBackendUnavailable`:17, `optional_module`:21, `configure_gpu_runtime`:28, `cuda_tensor`:34, `DeviceFrame`:48, `motion_summary`:59 | real | app/roop_3d/nv_video_codec.py:17 |
| `app/roop_3d/openvino_coprocessor.py` | 389 | Intel OpenVINO Asynchronous Co-Processor for heterogeneous inference offloading. | `probe_cpu_simd_features`:38, `get_openvino_device_info`:67, `OpenVINOSessionAdapter`:119, `AuxiliaryLookaheadWorker`:189, `OpenVINOCoprocessor`:300 | real code, optional/hardware-gated [INFERENCE: Intel iGPU/NPU path unverified here] | app/roop_3d/openvino_coprocessor.py:38 |
| `app/roop_3d/optical_flow_tracker.py` | 152 | Device flow propagation with explicit direction, visibility and scene resets. | `_grid`:5, `_sample`:12, `OpticalFlowTracker`:19, `RAFTSmallFlow`:74, `NvidiaHardwareFlow`:97, `select_flow`:143 | real | app/roop_3d/optical_flow_tracker.py:5 |
| `app/roop_3d/optimized_prepass.py` | 195 | Sparse candidates and dense verification; checkpoints contain numbers only. | `LockedFace`:24, `PrepassResult`:32, `merge_ranges`:53, `SparsePrepass`:64 | real | app/roop_3d/optimized_prepass.py:24 |
| `app/roop_3d/optimized_processor.py` | 388 | Three-stage streaming coordinator and production job-runner dispatch. | `StreamingProcessor`:27, `make_stream_runner`:226 | real | app/roop_3d/optimized_processor.py:27 |
| `app/roop_3d/optimized_trt_engine.py` | 95 | Explicitly requested strict TRT inference; constructor never loads a model. | `OptimizedTRTEngine`:13 | real | app/roop_3d/optimized_trt_engine.py:13 |
| `app/roop_3d/py.typed` | 1 |  |  | config | app/roop_3d/py.typed:1 |
| `app/roop_3d/pyproject.toml` | 48 | [build-system] |  | config | app/roop_3d/pyproject.toml:1 |
| `app/roop_3d/queue_manager.py` | 719 | Backend-owned 10-state persistent job queue with ACID-safe SQLite/JSON storage. | `CanonicalJobState`:47, `normalize_canonical_state`:139, `QueueSlice`:178, `QueueJobProgress`:193, `QueueJob`:203, `_ClosingConnection`:225 | real | app/roop_3d/queue_manager.py:47 |
| `app/roop_3d/render_policy.py` | 81 | Explicit stream policy and per-thread CUDA primary-context activation. | `RenderPolicy`:13, `CudaWorkerContext`:44 | real | app/roop_3d/render_policy.py:13 |
| `app/roop_3d/requirements-trt.lock` | 558 | # This file was autogenerated by uv via the following command: |  | config | app/roop_3d/requirements-trt.lock:1 |
| `app/roop_3d/requirements.lock` | 837 | # This file was autogenerated by uv via the following command: |  | config | app/roop_3d/requirements.lock:1 |
| `app/roop_3d/requirements.txt` | 6 | # NVIDIA runtime: ORT and conversion tools, plus pinned TensorRT CUDA 12: |  | config | app/roop_3d/requirements.txt:1 |
| `app/roop_3d/run_benchmarks.py` | 336 | Stage 15: Hardware Benchmark Protocol. | `get_current_vram_mib`:47, `get_current_ram_mib`:59, `percentile`:67, `run_benchmark`:74 | real | app/roop_3d/run_benchmarks.py:47 |
| `app/roop_3d/segment_dispatcher.py` | 409 | GOP-aligned stream-copy video segment splitting and lossless concat demuxer. | `find_ffmpeg`:29, `find_ffprobe`:42, `SegmentSlice`:55, `detect_keyframe_timestamps`:70, `_detect_keyframes_ffprobe`:112, `compute_gop_aligned_cuts`:139 | real | app/roop_3d/segment_dispatcher.py:29 |
| `app/roop_3d/stream_checkpoint.py` | 76 | Stage 9: ``job_checkpoint.json`` for the streaming render path (``optimized_processor.StreamingProcessor``). | `StreamSegmentCheckpoint`:20 | real | app/roop_3d/stream_checkpoint.py:20 |
| `app/roop_3d/supported_models.json` | 34 | { |  | manifest | app/roop_3d/supported_models.json:1 |
| `app/roop_3d/thread_benchmark.py` | 114 | Explicit, bounded local thread/pool benchmark; never runs at API startup. | `ThreadBenchmark`:12 | real | app/roop_3d/thread_benchmark.py:12 |
| `app/roop_3d/trt_fp8_builder.py` | 83 | Explicit E4M3FN Q/DQ admission, never an implicit FP8 builder flag. | `model_name`:14, `validate_swapper_fp32`:21, `validate_fp8_graph`:44 | validator only; FP8 needs separate calibrated Q/DQ ONNX (no builder) | app/roop_3d/trt_fp8_builder.py:14 |
| `app/roop_3d/trt_session_builder.py` | 387 | Strict, lazy TensorRT sessions with explicit profiles and executed canaries. | `StaticBatchError`:33, `TensorRTBuildError`:37, `precision_for`:41, `BatchProfile`:55, `validate_profile`:65, `auto_shape_profiles`:75 | real | app/roop_3d/trt_session_builder.py:33 |
| `app/roop_3d/vectorized_pipeline.py` | 350 | Batched identity transfer and GPU-resident affine/composite ONNX operators. | `crop_matrix`:19, `feather_mask`:25, `composite_cpu`:31, `CpuBatchPipeline`:58, `affine_graph`:109, `normalized_matrix`:171 | real | app/roop_3d/vectorized_pipeline.py:19 |
| `app/roop_3d/vfx_export.py` | 813 | Stage 22: NLE Timeline Interchange, Standalone VFX Matte Pass Exports, and Cryptographic Provenance Metadata. | `combine_alpha_matte`:28, `export_alpha_matte`:85, `extract_foreground_pass`:99, `VfxPassCollector`:137, `TimelineClip`:212, `TimelineMarker`:224 | real | app/roop_3d/vfx_export.py:28 |

### `app/ (entrypoints)` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/api.py` | 62 | Unified FastAPI backend entrypoint for Roop 3D. | `create_unified_app`:41 | real | app/api.py:41 |
| `app/serve.py` | 28 | Standalone entrypoint; the import path contains only this app's source. |  | real | app/serve.py:1 |

### 2.x Manifest / JSON summaries (not dumped)

| file | content summary |
|---|---|
| `models/model_manifest.json` | schema_version + 101 model entries (role, expected_sha256, byte_size, I/O contract). Roles: detector, recognizer, face_parser, occlusion_mask (face_occluder), face_restorer (CodeFormer v0.1, GFPGANv1.4, GPEN-BFR-512/1024, codeformer.fp16), identity_swapper (inswapper_128, hyperswap_1a/1b/1c_256, crossface_hififace), liveportrait_* (appearance/motion/warp/stitch), flame_geometry, lip_sync (Wav2Lip), diffusion_base/adapter (SDXL-Turbo, FLUX.1-Schnell, InstantID, PuLID); also 1k3d68/2d106det incl. `*_trt_precision_surgeon_v2_fp16.onnx` variants, `.batch.onnx` static-batch variants |
| `models/stage6_models.json` | 6 entries (SS_FaceSwap_MiniMax / bfs_* safetensors, nomos8k_sc_x4, RealESRGAN_x4plus_anime_6B) |
| `models/stage16_models.json` | 16 entries: LivePortrait generator/stitcher/eye+lip retargeter/motion+feature extractors, hififace_unofficial_256 + arcface_converter, bisenet_resnet_18, xseg_1/xseg_2, styleganex_age, swin2_sr_x4, real_web_photo_x4, SMIRK_em1, emoca_flame |
| `models/verified_sources.json` | 84 hash-keyed verified upstream sources |
| `models/governance_reviews.json` | 2 reviews + local_use_acknowledgement (license scope) |
| `models/makehuman_sources.json` | pinned MakeHuman commit, 130 file hashes (base.obj + targets), license note |
| `supported_models.json` | 28 swap/restore/detect/mask/upscale ids incl. inswapper_128(+fp16), hyperswap_1a/b/c_256, realswap_256/512, simswap_256/512_unofficial, ghost_256_unet_1-3, blendswap_256, uniface_256, ultramax, gpen_bfr_512/1024, codeformer, gfpgan_v14, restoreformer_pp, scrfd, retinaface, yoloface, xseg, bisenet, realesrgan_x2/x4, 4x_ultrasharp |
| `orchestrator/engines_manifest.json` | 21 declarative engines (video_decoding, face_detection, face_recognition, dense_landmarks, identity_swapping, reconstruction_3d, expression_transfer, restoration x5, ...), fields: adapter_id, models, resolution, precisions, providers, hardware_tiers, declared_peak_vram_mib, residency, exclusive_with, fallback |
| `orchestrator/render_profiles.json` | 2 render profiles (Balanced / Quality selector inputs) |
| `orchestrator/scene_rules.json` | `large_yaw_degrees: 30.0`; trait_map frontal / large_yaw; `low_confidence: withhold`; `occlusion: not_assessed` |

## 3. Focus-area findings (with evidence)

### 3.1 Blending
- Three real modes: `BlendMode.FEATHER / MULTI_BAND / POISSON` (`compositor/boundary_fusion.py:12-15`). MULTI_BAND = true Laplacian-pyramid blend via `cv2.pyrDown/pyrUp`, levels clamped to `floor(log2(min(h,w)))-1` (`boundary_fusion.py:50-74`); POISSON = `cv2.seamlessClone(NORMAL_CLONE)` on a `>0.05` binary mask with feather fallback on `cv2.error` (`:128-137`). Default is FEATHER, `pyramid_levels=4` (`compositor/pipeline.py:107,131`).
- Linear-light blending optional (`linear_light=False` default, `blend_transfer` in `color/transfer.py:28` = srgb/bt1886/gamma22/gamma28/bt709_oetf/linear); boundary_fusion converts via LUT before `_pyramid_blend` (`boundary_fusion.py:125`).
- Stage-17 core uses its own `ops.laplacian_pyramid_blend(photo, plate, alpha, levels=4, limit_overshoot=True)` (`pipeline/photorealistic_swap_core.py:236`, `pipeline/image_ops.py:250`), feather = 0.02 x face width (`:64`).
- Frequency separation: UltraMax restorer fuses via 3-level Laplacian pyramid: finest `detail_levels` bands 100% restorer, mid bands 50/50 swap/restorer, base from swap (`enhancers/ultramax.py:15-58`). `compositor/quality.py` has `transfer_fine_texture`, `match_sharpness`, `local_illumination_match`; `skin_texture_retention=0.25` re-injects target pore residual after UltraMax (`pipeline.py:128`).
- Colour: LAB mean/std + YCbCr + masked histogram + OKLab/OKLab-Wasserstein (`compositor/color_match.py:53-198`; `color_method` whitelist `pipeline.py:~205`), default `"lab"`, `color_strength=0.65`. SH-based relighting `compositor/lighting.py:17`, contact shadow (0..0.15) / SSS warmth options (`pipeline.py:144-149`). Sensor-physics finish: target-referred blur, defocus, shutter 180 deg, grain 100 (`sensor_physics.py:264`, `pipeline.py:~137`), CUDA path when torch.cuda available else CPU fallback (`:276-282`).

### 3.2 Masks
- **BiSeNet 19-class** parser: include classes (1,2,3,4,5,10,11,12,13), strict-forbidden (6,7,8,9,14,15,16,17,18) (`compositor/masks.py:25-26`); dual-BiSeNet forbidden mask (`masks.py:55`); jaw/neck clamp `apply_submental_jaw_cutoff` (`:204`); landmark-hull deterministic fallback (header `:1`). Class labels were *not* recorded in registry -> mapping is "conventional candidate" (`masks.py:3-6`) [defect-ish].
- **XSeg / face_occluder** 256x256 NHWC visible-face nets: `SUPPORTED_NETS = (xseg_3.onnx, xseg.onnx, face_occluder.onnx)` (`compositor/occlusion_model.py:20-21`); default `xseg_3.onnx`, missing model -> reported capability state, "nothing is simulated" (`:1-12`). (`xseg_3.onnx` is not in `model_manifest.json`/`stage16_models.json`, which carry `xseg_1/xseg_2` + `face_occluder.onnx` â€” [INFERENCE] resolved via `ModelCatalog`; verify.)
- Matte refinement: guided filter / fast guided filter (`matte.py:25,41`), edge decontamination (`:125`), 12px boundary-ribbon trimap matting (`:188-238`, opt-in `boundary_ribbon_matting=False`), hairline offset (`hairline.py:27`), directional strand matte with 12-orientation Gabor bank (`strand_matte.py:65`, opt-in `strand_matte=False`, `pipeline.py:83-86`).
- `masking_engine.py:47` `MaskingEngine`: "19-Class + DFL XSeg + text-prompted CLIPSeg" (header :1); 8 `baseline` mentions â€” [INFERENCE] part of this is optional/lazy.
- Mask seam watchdog `MaskSeamWatchdog` (`stream_restoration.py:97`) with registered-correlation scene reset.

### 3.3 Optical-flow mask stabilisation / temporal (axis 4, slice-2 part)
- Occluder matte: `OcclusionTracker` warps previous matte with dense **DIS** flow (`cv2.DISOpticalFlow_create(PRESET_MEDIUM)`, `compositor/occlusion_tracker.py:52`), history 0.7, `max_hold=4`, `hold_decay=0.8`, `flow_side=96`, `cut_reliability=0.18`, `cut_correlation=0.6`, `max_tracks=16` (`:44-45`); history only affects uncertain pixels (~0.5), confident detections pass instantly (docstring `:1-12`); `occlusion_release=0.75` (`pipeline.py:81`).
- Restorer-detail residual: `TemporalDetailStabilizer` flow-warps enhancement residual; NVIDIA OF 2.0 (CUDA) when available, DIS CPU fallback; `temporal_alpha=.25`, `flow_side=96`, `max_side=512`, `cut_correlation=.35`, `min_flow_confidence=.12`, 32 MiB cache (`detail_temporal.py:40-45`). Warps use `F.grid_sample(bilinear, align_corners=True)` (`:150-189`).
- Geometry/colour smoothing: `TemporalSmoother` One-Euro for affine, mask, colour: `min_cutoff=1.0`, `beta=0.035` (`compositor/temporal.py:75`; filter default beta 0.025 `:23`), `max_mask_side=512`, 32 MiB cap.
- Flow trackers: `optical_flow_tracker.py` `RAFTSmallFlow` (injected runner, `:74`), `NvidiaHardwareFlow` (`:97`), `select_flow` (`:143`).
- Scene-cut: compositor consumes a `scene_cut` flag everywhere (`pipeline.py:344,645,701,813,932`) and resets histories; the *producer* is `engine/detection/face_tracker.py:156 mark_scene_cut` (slice 1) and `tools/occlusion_acceptance.py:find_scene_cuts` (offline harness only). No detector inside slice-2 runtime path.

### 3.4 Restorers
- `compositor/stream_restoration.py:24-52`: `RestorerSpec` with explicit 5-pt template, `size in (128,256,512,1024)`, normalization `[-1,1]`/`[0,1]`; specs: GPEN `gpen_bfr_512.onnx`, CodeFormer `CodeFormerv0.1.onnx`, GFPGAN `GFPGANv1.4.onnx`, UltraMax = `codeformer.fp16.onnx` FFHQ-512 RGB[-1,1] (`:50`; overridable by `roop_3d/.runtime/restorers/ultramax.json`, evidence-required `:55-67`). `restore_in_crop_space` (`:70`), `OnnxFaceRestorer` (`:337`), `DecoupledRestorer` (`:429`).
- `enhancers/`: `DynamicCodeFormerRestorer` (fidelity `w` bound per frame, 512 px, CUBIC resize, `cv2.detailEnhance` fallback `codeformer.py:44-63`), `RestoreFormerPlusPlusRestorer`, `GPENRealisticRestorer` (static batch-1 FP32, luma-only recolour, `gpen_realistic.py:16`), `UltraMaxRestorer` (CUDA IOBinding, explicit-only small-face bypass `ultramax_small_face_mode="off"`, `ultramax_min_native_pixels=320`), `StyleGANEXAge` (latent direction, explicitly *not* calibrated age, `styleganex_age.py:1`).
- Legacy ROI restorers `compositor/restoration.py` (`GpenRestorer:44, CodeFormerRestorer:127, UltraMaxPipelineRestorer:185, GfpganRestorer:247`) blended only through face alpha (`blend_restored_roi:10`). Resolution policy: `roi_resolution_limit` 0 or 128..2048 (`pipeline.py:181-183`); restorer up-resizes <512 with CUBIC else AREA (`restoration.py:99`).
- `RealESRGAN`/`4x_ultrasharp`/`swin2_sr` appear only in manifests (`supported_models.json`, `stage16_models.json`); no slice-2 code wires a frame-level upscaler other than restorers [INFERENCE].

### 3.5 Sub-pixel warping
- CPU: `cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT)` for crop and alpha (`boundary_fusion.py:34-37`). No Lanczos/INTER_CUBIC warp, no half-pixel-centre handling beyond cv2 default [baseline]. Cubic is used only for resizes (`pipeline.py:436,509,531`; `restoration.py:87`).
- GPU: `gpu_compositor.warp` -> `gpu_transforms.warp_affine_gpu` (`gpu_compositor.py:19-22`, `gpu_transforms.py:11`); closed-form affine inverse on-stream (`gpu_compositor.py:25-33`). `hires_handoff.py` rescales matrices between low-res swap and hi-res frame (`:47-104`). `vectorized_pipeline.affine_graph` expresses warp as ONNX GridSample graph (`:109`); `models/grid_sample.py:18` lowers 3-D trilinear GridSample to two 4-D ones.
- Boundary support dilated; "pyramid reconstruction and Poisson gradients can spread beyond nonzero alpha" -> write footprint clamped (`boundary_fusion.py:140-141`).

### 3.6 CLI / headless / JSON logs / TUI / diagnostics (axis 5)
- **No Rich/TUI**: no `rich` import anywhere in slice (grep). Terminal UX = custom ANSI HUD `telemetry/pinokio_terminal.py` (CLEAR_LINE `\r\x1b[K` `:68`; in-place live line + milestone logs; `enforce_terminal_environment:28`; `NO_COLOR`/`FORCE_COLOR` honoured `:377`; `ROOP3D_HUD=0` silences `:391`; `ROOP3D_TRT_SIDECAR` `:594`).
- **CLI flags**: only one real flag, `--server_share/--server-share` (`server/__main__.py:67`) + env `ROOP3D_SERVER_SHARE`. Everything else is env (`ROOP3D_PORT`, `ROOP3D_HOST`, `ROOP3D_ASYNC_PIPELINE`, `ROOP3D_PIPELINE_QUEUE` 1..64 default 8 `async_pipeline/config.py:22-26`, `ROOP3D_PINNED_FRAMES`, `ROOP3D_CUDA_GRAPH`, `ROOP_TRT_FP16`, `ROOP_TRT_FP8`, `ROOP_SWAP_FP16`, `ROOP_STRICT_NVDEC`, `ROOP_OPT_STRICT_TRT`, `ROOP_AUDIT_DEBUG`, `ROOP_GPU_COMPOSITOR`, `ROOP3D_ORT_LOG_LEVEL`, `ROOP3D_PREVIEW_SESSION_POOL`, ...). No headless one-shot render CLI: `tools/*` are maintenance/benchmark/gates (e.g. `python -m roop_3d.tools.benchmark_async_pipeline`, `reliability/soak.py` has its own CLI) â€” a render is only reachable via REST `/api/v1/queue` or `pipeline.video_job_runner.EndToEndVideoJobRunner` as library.
- **JSON logs**: `JsonFormatter` (`server/__main__.py:73-82`) emits `{"level","logger","message"}` or redacted structured dicts; startup events `{"event":"ready"}` and `{"event":"share"}` (`:85-100`); errors printed as `{"error":{"code":...}}` (`:112`). `core/redaction.py:8`, `security/log_sanitizer.py:53` redact.
- **Diagnostics reports** (two coexisting 14-section reports, different content):
  1. `telemetry/runtime_report.py:29-44` (GET `/api/v1/system/runtime-report`, `server/runtime_report_routes.py:14`; also printed at boot/job start by HUD): Host/OS/Venv, CPU cores & worker pools, GPU arch & compute capability, EP chain & mixed precision, TRT engine cache, zero-copy OrtValue ring, VRAM budget & OOM thresholds, model governance & SHA-256, 3D head topology, full-craniofacial swap lock, physical camera & SH relighting, GOP slicer & audio remux, offline isolation audit, loopback server & deep links. Per-section `OK/INFO/WARN/UNAVAILABLE/UNKNOWN`; a failing collector degrades to UNKNOWN (`:461-467`).
  2. `server/diagnostics.py:937-1184` `generate_14_section_report` (GET `/api/v1/system/diagnostics-report`, `routes.py:699-700`): Python & Env, Host HW, ORT providers, TRT cache & quantisation, FFmpeg media & codec matrix, model registry & SHA, detection & ArcFace alignment, 3D mesh & LivePortrait, Identity transfer & inswapper-128 graph, Compositor/XSeg/alpha seams, Restoration adapters (GPEN-512/CodeFormer), VRAM governor, Temporal landmark smoothing, Strict offline verification.
  3. `reliability/report.py:222-275 build_report/save_report` (redacted local diagnostic; `/api/v1/...reliability`), `app/diagnostics/pipeline_audit.py` (tensor-shape tracker, EP auditor, `CopyCounter`; errors `StageQualityDegradationError`, `ExecutionProviderFallbackError`, `EnhancerBypassError`; `ROOP_AUDIT_DEBUG` fail-loud).
  4. Pre-flight linter `POST /system/preflight-lint` (`server/diagnostics.py:621 run_preflight_lint`, checks: yaw coverage vs .fsz `:455`, TRT engines `:516`, UltraMax 4K `:542`, output size `:563`, disk `:586`); synthetic benchmark `POST /system/benchmark` (`:868`).
- TRT build heartbeat: 1 Hz `trt_compilation_progress` WS frame + out-of-process sidecar (`telemetry/trt_heartbeat.py:75`, `trt_heartbeat_sidecar.py:37`).

### 3.7 Compute: TRT / FP16 / INT8 / IO binding / pinned / VRAM (axis 6)
- Precision policy `trt_session_builder.precision_for` (`:41-52`): FP32 for `inswapper_128`, `motion_extractor`, `gpen*`; `ROOP_TRT_FP16=0` -> FP32; `ROOP_TRT_FP8=1` + heavy restorers -> FP8 else FP16. **INT8 explicitly off**: `"trt_int8_enable": False` (`:236`). FP8 requires separate calibrated E4M3FN Q/DQ ONNX + reference parity callback (`:170-190,329-330`; `trt_fp8_builder.py:21-44`).
- ORT TRT options (`trt_session_builder.py:229-246`): engine cache + prefix, timing cache, `builder_optimization_level=5`, `auxiliary_streams=2`, `cuda_graph` only for static inputs and auto-disabled on partition error (`:285-292`), `layer_norm_fp32_fallback=True`, workspace <=1024 MiB (6gb) / 2048 (12gb) (`:111-114`), VRAM budget minus 768 MiB reserve (`:89-100`), batch profile (1,4,8) opt 4 (`:55`). Note `core/ort_session.py:38` uses `builder_optimization_level "3"` â€” two TRT configs with different levels.
- Mixed precision: graph-level FP32 pinning of `RodriguesRotation, SphericalHarmonics, LayerNormalization, InstanceNormalization, Pow, ReduceMean, Softmax`; FP16 on Conv/ConvTranspose/Gemm/MatMul; policy `fp16-fp32pinned-v1`; failover TRT->CUDA->CPU replaying inputs (`execution/trt_mixed_precision.py:25-35`, `:552-711`).
- IO binding: `core/ort_session.py:679-743` (`GpuIoBinding`: bind_input by device ptr, `bind_ortvalue_*`, preallocated outputs); `execution/zero_copy_pipeline.py` `ZeroCopyGPUExecutionRing` (reusable device buffers); `tools/gpu_saturation.py` measures FP16+IOBinding+micro-batch; `app/tests/test_stage27_certification.py:110` asserts >85% GPU use with micro-batched IOBinding.
- Pinned memory: `async_pipeline/buffers.py` `PinnedFramePool` (`torch.empty(pin_memory=True)`, `:130-156`), `CudaFrameRing:285`, bounded queue default 8 (`:25`, ~200 MB of 4K BGR); `AsyncPipeline` one thread+CUDA stream per stage, event hand-off, no device sync (`stages.py:1-20`); `hardware_streamer.SharedMemoryFrameRing:35` for process ring.
- VRAM governor (code lives in `engine/vram_governor.py`, slice 1): consumed here by `core/ort_session.py:99-169` (pool VRAM measured via NVML before/after each pooled session, `reclaimable_vram_mib`), `telemetry/runtime_report.py:299-321` (per-stream admission reservation, watchdog pressure/near-OOM thresholds), `jobs/worker.py:340` `RecoverablePipelineOOM` -> checkpoint then one CPU-profile retry (report text `:310`), `server/diagnostics.py:119 purge_idle_vram`, `compositor roi_resolution_limit` pressure cap.
- Hardware probes `core/hardware.py` (`effective_vram_budget:112`, `tensor_core_precisions:136`, `tensorrt_version:159`); `core/transfer_ledger.py` counts host<->device copies at the ORT boundary.

### 3.8 Job queue, checkpoint, GOP slicing, NVENC, remux, WS (axis 7)
- **Real job-state enums** (three coexisting vocabularies â€” see defects):
  - `jobs/schema.py:27` `JobState`: queued, preparing, running, pausing, paused, cancelling, cancelled, succeeded, failed, interrupted (10) with explicit `TRANSITIONS` graph `:39-48` (terminal: cancelled, succeeded, failed) enforced by `require_transition:51`.
  - `queue_manager.py:47` `CanonicalJobState`: QUEUED, PREFLIGHT, ANALYZING, PRESCANNING, PREPASSING, RENDERING, MUXING, PAUSED, COMPLETED, FAILED, RETRYING, CANCELLED (12) + `CANONICAL_TRANSITIONS` (FAILED->QUEUED/RETRYING; COMPLETED/CANCELLED terminal).
  - `queue/lifecycle.py:28` `LifecycleState` (10 UI states): queued, preparing, compiling_trt, tracking_3d, rendering, paused, muxing_audio, completed, failed, cancelled; pure projection `lifecycle_of` (`:62`), `compiling_trt` inferred from stage-text markers (`:48`).
- **Persistence/atomicity**: `core/storage.py:27-41` `atomic_bytes/atomic_json` = temp file + `fsync` + `os.replace` + dir fsync, `exclusive` cross-process lock. `jobs/store.py` transactional on-disk state with idempotency + worker acks; `queue_manager.py` "ACID-safe SQLite/JSON" (header). `stream_checkpoint.py:72-75` saves `job_checkpoint.json` via `atomic_json`; segments recorded with sha256 + frame count (`:60-65`) and resume re-verifies `sha256 == _sha256(segment) and count_video_packets == expected_frames` (`:47`). `job_queue.py:364,395` writes `.part` then `os.replace` -> published only when complete. `JobCheckpoint` schema v1: segments list with PENDING|COMPLETED|FAILED, inputs fingerprint, frame_progress, execution_provider, encoder (`job_queue.py:47-85`). `video_writer.py:594` `os.replace(...)` atomic publish; `reliability/recovery.py` `ReliabilityAuditor` + `reconcile_on_startup:490` repairs jobs/checkpoints/outputs.
- **GOP-aligned slicing**: `segment_dispatcher.py` detects keyframes via PyAV (`:70-109`, 50 ms dedupe) with ffprobe fallback, `compute_gop_aligned_cuts` (`:139`, default `max_segment_duration=15.0` `:376`; min 0.5 s from previous cut), `-c copy` split (`:199-217`), `concat_lossless` demuxer + original audio remux with AAC fallback if copy fails (`:262-330`). Exposed `POST /segments/split|concat` (`queue_routes.py:202,214`). Parallel GOP-chunk path: `execution/worker_pools.py` `GopChunk:201`, `stitch_gop_chunks:226`, `encode_nvenc_chunk:259`, `PostEncodeWorkerPool:283`.
- **NVENC**: `media/nv_writer.py:25-26,100-103` h264_nvenc/hevc_nvenc `-preset p5 -rc vbr -cq <q> -b:v 0`, auto-fallback libx264/libx265 (`-preset medium -crf`), frame replay (`_REPLAY_FRAMES`, `:168,194`), `probe_nvenc:31`; `media/video_writer.py:68-91` hardware=auto picks NVENC for 8-bit 4:2:0 only (`:277`); `execution/worker_pools.py:271-272` lossless NVENC `-tune lossless -rc constqp -qp 0` chunks; `hardware_streamer.py:245` NVDEC `-hwaccel cuda -hwaccel_output_format cuda`, `:366-386` NVHardwareVideoWriter. No AV1 NVENC. Export profiles: h264/hevc 8-bit, ProRes 422 HQ 10-bit (`prores_ks` profile 3), FFV1 mkv, image sequences (`export_profiles.py:60-92`).
- **Audio/subtitle remux**: `media/audio.py:19-22,75` per-container compatible-codec sets, copy if codec+packet boundaries align else AAC/PCM; all tracks; `media/subtitles.py:23-25` packet copy (mp4/mov: mov_text; mkv: subrip/ass/ssa/webvtt/dvd/pgs/dvb) else explicit recorded drop; `media/av_sync.py:119 measure_lag` verifies sync. BUT the Stage-17 runner drops subtitles (`-sn`), metadata (`-map_metadata -1`), refuses HDR (`HDR_INPUT_UNSUPPORTED`) and remuxes only `1:a:0` (`pipeline/video_job_runner.py:12-14,128,135,217-224`) â€” narrower than `media/video_writer`.
- **WebSocket routes**: `/ws/telemetry`, `/ws/events`, `/ws/queue` -> `event_stream`; `/ws/v1/pipeline/status` -> `pipeline_status_stream` (carries `model_download_progress` and `trt_compilation_progress` frames too); `/ws/v1/preview/stream` -> `preview_stream` (binary: 16-byte header + JPEG, JSON control) (`server/app.py:322-326`, `live_stream.py:3`). Event types (`server/ws.py`): JobState, JobProgress, Telemetry, PipelineStatus, TrackingTelemetry, StreamGap, StreamReady, Queue, ModelDownloadProgress, TrtCompilationProgress. `IntakeMiddleware` (`middleware.py:39`) enforces bounded body, explicit origins (`127.0.0.1:5174`), repeated-header rejection, `ROOP3D_MAX_BODY_BYTES`. LAN share: bearer token per launch (`security.py:20`).
- **Queue REST**: `/queue` list/submit/get, `/{id}/cancel|pause|resume|progress|duplicate|retry|join`, `/queue/reorder`, `/batch/recipes|generate|submit`, `/queue-lifecycle`, `/segments/split|concat` (`queue_routes.py:81-214`). Batch Matrix: 4 strategies (`queue/batch_matrix.py:24 BatchStrategy`, `BatchMatrixEngine:262`).

### 3.9 Workstation UI (axis 8; server side only, UI is another slice)
- Same-origin SPA fallback `server/api.py:35-54 mount_workstation`; built-bundle gate `tools/verify_e2e_workstation.py:68`; workstation REST `server/workstation.py` (30 routes, chunked upload `:46-91`), preview `workstation_preview.py` (on-demand bounded previews, `PreviewService:162`), live preview cache/mutex `live_preview.py:219-306`, timeline scrub `media_frames.py`, Face bank / target lock / identity assets / blender / studio environments, installer routes (preflight, offline models, update, rollback `installer_routes.py:36-65`), unified state (`unified_state.py:14 RenderMode`).

## Axis findings (slice)

1. **Detection/landmarks/angle**: slice-2 hosts `media/detector.py:18` (SCRFD, RetinaFace-10G, YOLOFace selectable decoders), `media/face_analysis.py:51 five_point_plausible`, `:70 agrees_with_106` (5-pt vs 106-pt cross-check), `core/pose.py:173 estimate_pose_5pt`; pre-flight yaw coverage vs .fsz (`server/diagnostics.py:455`); orchestrator `scene_rules.json large_yaw_degrees=30`, `low_confidence: withhold`. Failure thresholds: `ROOP3D_AUTO_3D_YAW/PITCH`, `ROOP3D_3D_MAX_ROLL` envs (grep, values live in slice 1).
2. **2D vs 3D/.fsz**: `server/faceset_3d.py:127 reconstruct_3d`, `identity_assets.py` atlas bake, `pipeline/photorealistic_swap_core.py:188` craniofacial geometry + rectified swap + quotient-image relight; `swap_scope` full_craniofacial/adaptive_skull/inner_face (`compositor/pipeline.py:~165`); `io/glb_export.py` binary glTF, `mesh_raycaster.py` self-occlusion; `models/grid_sample.py` volumetric sampling.
3. **Visual quality**: Laplacian/Poisson/feather, BiSeNet+XSeg, guided-filter matting, LAB/OKLab colour, SH relight, sensor physics, GPEN/CodeFormer/GFPGAN/UltraMax restorers (sections 3.1-3.4); `evaluation/metrics.py` PSNR/SSIM/perceptual/cosine, regression gate, blind A/B (`regression.py:64`, `blind_review.py:83`). Colour-science package `color/` (H.273 codes, PQ/HLG, BT.2390 EETF, HdrToSdr `hdr.py:74`) â€” only used for output tagging/media; HDR still rejected in Stage-17 runner.
4. **Temporal**: One-Euro (affine/mask/colour), DIS-flow occluder tracker, flow-warped detail residual (3.3); scene_cut flag plumbed but detector external.
5. **CLI/telemetry**: custom ANSI HUD, JSON logs, two 14-section reports + reliability report + pre-flight linter (3.6); no Rich, no `--json`/`--verbose`/`--headless` flags.
6. **Compute**: FP16 default, FP32 for swapper/gpen/motion-extractor, FP8 gated, INT8 off, IO binding, pinned pools, CUDA graphs for static shapes, VRAM admission with OOM->CPU retry (3.7).
7. **Queue/IO**: 10-state persisted `JobState`, 12-state canonical queue states, atomic checkpoint, GOP stream-copy slicing, NVENC w/ CPU fallback, all-track audio/subtitle remux in media layer (3.8).
8. **Workstation UI (server)**: 202 REST routes + 5 WS, SPA mount, uploads, previews, live JPEG stream (3.9).

## Gaps / stubs / defects found

1. **State vocabulary drift** â€” three enums: `JobState` (10, `jobs/schema.py:27`), `CanonicalJobState` (12 members, `queue_manager.py:47`) whose `normalize_canonical_state` docstring still says "one of the 10 canonical states" (`:139-140`), and 10-state `LifecycleState` (`queue/lifecycle.py:28`). `jobs/*` (JobWorker) and `queue_manager`/`job_queue` are parallel stacks; `server/app.py:86-100` instantiates BOTH `JobStore`+`JobWorker` (jobs/*) and `QueueManager` (queue_manager.py), so two state machines run side by side.
2. **No Rich TUI, no headless render CLI, no `--json`/`--log-level` flags**: only `--server_share` (`server/__main__.py:67`). Render only via REST/library.
3. **INT8 absent** (`trt_session_builder.py:236 trt_int8_enable False`); FP8 needs a separately calibrated artifact that is not shipped (no FP8 entries in manifests).
4. **Inconsistent TRT config**: `core/ort_session.py:38` optimization level 3 vs `trt_session_builder.py:242` level 5; two builders (`trt_builder.py:133 TensorRTBuilder`, `trt_session_builder.py:103`, `optimized_trt_engine.py`, `liveportrait_trt.py`) coexist.
5. **Sub-pixel warping is baseline**: bilinear `cv2.warpAffine` only (`boundary_fusion.py:34`); no higher-order resampling or sub-pixel landmark refinement in warp path.
6. **Stage-17 e2e runner is narrower than the media layer**: drops subtitles/metadata, refuses HDR, first audio track only (`video_job_runner.py:12-14,217`). Four writer stacks: `media/video_writer.py`, `media/nv_writer.py` (class `NVHardwareVideoWriter:47`), `hardware_streamer.py` (another `NVHardwareVideoWriter:359`), `video_job_runner.ColourManagedWriter:188` â€” duplicate class name, divergent behaviour.
7. **BiSeNet class mapping unverified** ("conventional 19-class mapping as a candidate", `masks.py:3-6`); default XSeg name `xseg_3.onnx` has no manifest entry (manifests list xseg_1/xseg_2 + face_occluder).
8. **Scene-cut detection not in the render path of slice 2**: compositor needs an externally supplied flag; `find_scene_cuts` only in `tools/occlusion_acceptance.py:89`.
9. **GOP slicing is stream-copy, not frame-accurate**: cuts clamp to keyframes (>=0.5 s apart, default 15 s segments); chunk encode path (`worker_pools.encode_nvenc_chunk`) is lossless-NVENC (`-qp 0`) intermediates [INFERENCE: large temp files].
10. **Intentional stub**: `bridges/cloud_stub.py` (disabled remote adapter); `bridges/provider_protocol.py` has no concrete provider. `reliability/harness.py` uses stub inference stage (test only).
11. **Hardware-gated / unverified here**: `openvino_coprocessor.py`, `heterogeneous_scheduler.py`, `cvcuda_ops.py`, `mesh_raycaster.py` (Embree), NVIDIA OF 2.0 paths in `detail_temporal.py`/`optical_flow_tracker.py` â€” fall back to CPU/DIS; none executed in this audit.
12. **`app/tests/` is thin**: 7 py files / 720 lines; three (`test_3d_relighting.py`, `test_heterogeneous_scheduler.py`, `test_nv_cvcuda_pipeline.py`) are 8-line forwarders to the real suite in `app/roop_3d/tests` (slice 3). `test_flame_and_liveportrait.py` and `test_stage27_certification.py` FAIL (not skip) when hardware/assets are missing (header `:1`).
13. **Shim**: `app/roop/*` only re-exports `roop_3d` (`roop/__init__.py`, `core/ort_session.py`); no independent code.

Slice files in inventory: 335 | files accounted for: 335 | omitted: none
