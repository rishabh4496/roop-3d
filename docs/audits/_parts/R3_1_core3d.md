# R3_1 â€” roop-3d core 3D slice (reconstruction / rendering / engine / identity / faceset / tracking / swappers / top-level pose+temporal+faceset modules)

Paths are relative to `app/roop_3d/`. Status: **REAL** = implemented algorithm; **HEUR** = heuristic/approximation labelled as such in code; **BASE** = baseline/fallback/stub; **INFRA** = non-3D plumbing (listed for coverage). Evidence = file:line where I read it; everything else is from module docstring + public symbol list (AST scan) and flagged [INFERENCE] if it describes behaviour I did not read.

## 0. Headline verdicts (what is true 3D vs baseline)

| Question | Verdict | Evidence |
|---|---|---|
| PnP used? | **Yes, in 6 independent places**, all OpenCV: `cv2.solvePnP` ITERATIVE + LM refine on canonical 68-pt model with dynamic focal (18â€“135 mm bounds); `solvePnPRansac(EPNP,100 it)`+`RefineLM` for dense head; `SQPNP` for MakeHuman and PoseBlend anchors | reconstruction/tps_fallback.py:175-243; dynamic_camera.py:146-215; full_head.py:412-430 (`solve_dense_pose`); makehuman_head.py:150; pose_blend.py:248-251; flame_fitter.py:264 |
| PnP flip hazard acknowledged | faceset_3d falls back to stored measured pose on degeneracy ("measured pitch -177Â° on frontal") | reconstruction/faceset_3d.py:25-31 |
| Per-frame video pose in live swap path | Not PnP: `FaceObservation.pose` comes from 5-pt `core.pose.estimate_pose_5pt` ("not calibrated 3D reconstruction") | engine/detection/dense_landmarks.py:102-106 |
| z-buffer self-occlusion | **Yes, real** in 3 renderers: exact CPU triangle z-buffer w/ barycentric attrs (`vertex_raster.py:34-77`), GPU ModernGL hardware depth test (`dense_correspondence.py:104-105`, `full_head.py`, `physical_buffers.py:5`), UV-texel visibility by `fillConvexPoly` depth map with `DEPTH_TOLERANCE_M=0.008`, `GRAZING_MIN_COS=0.25` (uv_synthesizer.py:373-412, 509) |
| `mesh_raycaster.py` ("Embree/BVH, <0.8 ms") | **BASE/misleading** â€” no Embree, no BVH, no ray cast. If `|yaw|>45` it zeroes a hard-coded landmark index list (68-pt: 13-16,44-47 / 0-3,36-38,41) or vertices with `x>0.025 & z<0.04`; otherwise all visible; z-buffer is vertex splats (5Ã—5 footprint <200 verts). `use_simd`/`bvh_enabled` flags are stored and never read. Only consumer: heterogeneous_scheduler.py + its test | mesh_raycaster.py:42-52, 127-150, 163-200; grep consumers |
| SH relighting | **Real 2nd-order (9-coeff) SH**, Ramamoorthiâ€“Hanrahan constants, regularised LS fit; linear-light fit with 0.5 albedo prior in reconstruction/photometry.py:1-30; per-frame tracker SH `MeshSHLightingEstimator` (tracking/lighting.py:53). `relighting_3d.NeuralRelighter3D` name is misleading: no neural net â€” SH shading + luminance-ratio cast-shadow mask + Blinn-Phong (k=0.25, shininess 24) | relighting_3d.py:19-24,126-150,287-345 |
| UV albedo fusion | **Real**: visibility-weighted multi-view UV synth, confidence-gated (`OBSERVED_MIN_WEIGHT=0.02`), multiband seam blend, albedo prior fill (skin Lab gamut gate, â‰¥200 skin texels), intrinsic 4-level pyramid | uv_synthesizer.py:123,373-552; uv_completion.py:29,155,270; albedo_prior.py:32-36,82-88; intrinsic_texture.py:26 |
| Pose bins | Four incompatible grids (see Â§4). Faceset coverage 12Ã—8 bins (yaw Â±90 / pitch Â±60 clamp); target-lock 5Ã—3 (yaw Â±75, 30Â° wide; pitch Â±45); Stage-0/13 3Ã—3 yaw edges Â±25 / pitch Â±15, anchors yaw Â±75, pitch Â±35; legacy PoseSelector 3Ã—3 Â±15/Â±12 | faceset_3d.py:20,55-58; target_lock.py:24-28; pose_matrix.py:24-28; pose_selector.py:17-34 |
| Kalman / One-Euro / SLERP | **All three exist.** One-Euro + constant-velocity Kalman (no-overshoot) per-channel; SO(3) One-Euro using quaternion SLERP; Kalman box in target-lock; embedding Slerp/FrÃ©chet mean for identity blend | tracking/filters.py:1-62,170; temporal.py:16,56,139-177,251; temporal_stabilizer.py:5; identity/target_lock.py:31; faceset/identity_blend.py:21-93 |
| Scene-cut detection | **Yes, 3 sites** â€” 32Ã—32 thumbnail diff (>0.45 cut; >0.18 AND Bhattacharyya>0.7 cut) in FaceTracker (face_tracker.py:180-185) and `ShotBoundaryDetector` (video_tracker.py:189-211, settings 71-76); plus kinematic cuts: face-centre jump >2.5 IOD or rotation >2700 Â°/s within dt â‰¤0.2 s (video_tracker.py:343) | cited |
| ArcFace re-ID | **Yes**: TrackPolicy cosine ladder strong 0.55 / match 0.35 / continuity 0.25 / weak 0.15, reacquire 0.5 (+margin 0.08), ambiguity margin 0.06, uncertain_limit 45, lost_limit 64, retention 1800 frames; target-lock MIN_COSINE 0.42, STRONG 0.55 | track_policy.py:46-62; target_lock.py:12-15; face_tracker.py:205,387-391 |
| ByteTrack | **Absent.** Only token match is a docstring mention in angle_tracker.py (grep). Association = Hungarian (`optimal_assignment`, face_tracker.py:46) on `0.8*cos+0.2*overlap` / `0.6*sim+0.25*shared+0.15*layout` + IoU â‰¥0.1 gating + constant-velocity predicted box; **no low-confidence second association pass** | face_tracker.py:46,206,320,333,391 |
| Dense-3D engage thresholds | Automatic switch from 2D swap to dense 3D when `|yaw|>30` or `|pitch|>25` (env `ROOP3D_AUTO_3D_YAW/PITCH`, 180 disables); roll >40Â° hands back to 2D (`ROOP3D_3D_MAX_ROLL`); profile far-cheek mask trim fades in over |yaw| 30â†’40Â°; swap presence fades from 65Â° to 82Â° off-axis angle `acos(cos yaw cos pitch)` with 2.5Â° hysteresis, hold 2 frames, ramp 3, forget 30 | volumetric/session.py:121-132,1029-1050; presence.py:17-22,45-58 |
| Tracker/reenactment yaw limits | Video tracker `max_abs_yaw_deg=85` (reject if fit >85; pre-reject 5-pt >100); expression only to 60Â°; ReenactmentSettings `max_abs_yaw_deg=80`, `expression_max_yaw_deg=60` | video_tracker.py:79-80,399-409; reenactment.py:70-71 |
| True volumetric swap model? | The *identity* is still transferred by a 2D network (INSwapper via `IdentitySwapper`; HifiFace-256 wrapper states exported runtime inputs are RGB+embedding, not FLAME) â€” 3D only conditions pose/geometry/texture (PoseBlendProjector TPS/perspective-mesh, ReenactmentEngine, neural UltraMax correspondence) | volumetric/identity_swapper.py:1-116; swappers/hififace.py:3-5; session.py:33-34,722 |

## 1. File-level inventory

### 1.1 `reconstruction/` (52 files)
| File | Purpose / key symbols | Status |
|---|---|---|
| __init__.py | Re-exports geometry/TPS/PnP/FLAME/UV APIs ("2.5D TPS fallback" in docstring, :2) | INFRA |
| ARTICULATED_MODELS.md / ARTICULATED_MODELS_LICENSE | Licence/provenance of articulated-head assets (CC0-style notice) | doc |
| assets/makehuman_head_v1.npz (386 KB) + .npz.json | Packed MakeHuman CC0 head arrays + digest sidecar | asset |
| geometry_types.py | `CameraIntrinsics`, `Pose3D`, `Mesh3D`, OpenCVâ†”OpenGL camera, Rodrigues/quat/Euler converters, UVâ†”pixel | REAL |
| tps_fallback.py | `get_canonical_3d_landmarks`, `solve_pose_pnp` (solvePnP + dynamic focal), `build_canonical_head_mesh` (Delaunay), `ThinPlateSplineWarper`, `warp_delaunay_piecewise_affine` â€” "2.5D" | PnP REAL / warp BASE |
| tps_warp.py | TPS helper shared by frontalization/expression (`border_anchors`, `warp_points_to`) | BASE |
| dynamic_camera.py | `DynamicCameraSolver`: focal bounds 18â€“135 mm eq., PnP + `solvePnPRefineLM`, SQPNP/EPNP fallback | REAL |
| frontalization.py | `frontalize_crop`: 3D-pose frontalization via TPS before swap | BASE (TPS) |
| expression_tps.py | TPS expression warp on 106 landmarks, pinned silhouette, `layout_is_plausible` | BASE (2D warp) |
| pose_blend.py | `PoseBlendProjector`: estimated 3D proxy, SQPNP+LM, z-buffer raster, TPS/perspective-mesh correction; `MAX_SIZE=512`; cos(yaw) from PnP rotation | HEUR (proxy mesh) |
| jaw_fit.py | Per-frame jaw width ratio fit; `MIN_FORESHORTENING=0.75` (~41Â° yaw); 5 PnP anchors fixed | HEUR |
| multiview_fit.py | Bayesian (MAP) multi-view landmarkâ†’identity shape fit; Ïƒ_obs 2 mm, prior Ïƒ xy 12 mm / z 15 mm, pose Ïƒ 8Â°, Huber Î´=3Ïƒ; self-occlusion soft weights by normals; `estimate_pose_from_landmarks` | REAL |
| rbf_deform.py | Wendland RBF + Laplacian-regularised template deformation | REAL |
| faceset_3d.py | `observe_poses` (12Ã—8 bins), `reconstruct_faceset`; canonical fallback labelled `canonical_template_fallback` | REAL + BASE fallback (:102-116,203-220) |
| faceset_quality.py | `analyze_faceset` per-view quality + coverage report; placeholder-view detection | REAL |
| identity_asset.py | `build_identity_asset` (schema `roop.identity_asset.v1`, algo `identity-mv-5`), weak-perspective MAP BA, z-buffer visibility texture, GLB export; template canonical|makehuman | REAL |
| mesh_io.py | Atomic OBJ/GLB export + loaders | REAL |
| mesh_bridge.py | Barycentric surface bridge between topologies (Umeyama, closest-point, UV rasterise) | REAL |
| topology.py | `MorphableTopologyRegistry` (FLAME-open, canonical, MakeHuman) | REAL |
| morphable_backbone.py | Three-tier versioned export `MorphableBackbone`; forbids zero-filled substitutes/MakeHuman fallback (:6) | REAL |
| native_backbone.py | 3DDFA-V3/SMIRK/HRN ONNX encoders, graph contract + asset verification | REAL (needs local weights) |
| native_face_geometry.py | NumPy decoders (deep3d crop, BFM rotation/camera, FLAME rig, UV map) | REAL |
| backbone_status.py | Which backbones are usable on this install (UI dropdown) | INFRA |
| flame_fitter.py | Licence-gated FLAME; Huber/Geman-McClure; `SyntheticFLAMEModel` for tests/fallback | REAL gated / BASE synthetic |
| makehuman_build.py / makehuman_head.py / makehuman_landmarks.py / makehuman_template.py | Open CC0 FLAME alternative: build, `fit_multiview`, 68-pt correspondence, template adapter | REAL |
| anatomical_pack.py | Procedural anatomical sub-meshes (eyes, teeth, cavity AO, soft-tissue spring filter); `_analytic_fallback_head` offline (:747,798) | HEUR/BASE fallback |
| subdivision_and_masks.py | Craniofacial vertex groups, feathered region masks, silhouette subdivision | REAL |
| displaced_surface.py | UV displacement surface refinement | REAL |
| dense_correspondence.py | `ReferenceSurfaceRasterizer` + `GPUSurfaceRasterizer` (ModernGL z-buffer), `correspondence_field` | REAL |
| full_head.py | `GPUHeadRasterizer`, `FullHeadEngine`, `solve_dense_pose` (EPNP RANSAC 100 it, conf .999, reproj â‰¥2 px/4% span, â‰¥6 pts, reject err >10% span), `complete_diffuse_albedo`; "no software raster fallback" | REAL (GPU only) |
| photometry.py | Linear-light SH fit (`fit_sh`), `decompose_albedo`, `fit_target_surface`, `extend_neck` | REAL |
| photometric_gpu.py | OpenGL photometric relight pass at enhancer ROI res (`relight_head`) | REAL |
| albedo_prior.py | Skin-tone prior fill for unseen texels | REAL |
| uv_synthesizer.py | Multi-view UV synth with z-buffer visibility | REAL |
| uv_completion.py | Multiband blend, nearest-texel and `FullHeadUVCompleter` | REAL |
| view_selection.py | `ViewSelector`: hysteresis (MARGIN .035, SWITCH_FRAMES 3, RAMP_STEP .25, SMOOTHING .35, FADE_FLOOR .05, GIMBAL_PITCH 75Â°, MAX_GAP .25 s) | REAL |
| view_texture_bank.py | 9 acquisition buckets + two-view interpolation on view sphere | REAL |
| swap_coverage.py | Swap coverage = rasterised surface âˆª landmark hull with feather | REAL |
| reenactment.py | `ReenactmentEngine` Stage 4: rig from identity asset, per-track `NeutralCalibrator`, One-Euro settings (pose 1.5/.03, expr 6/1, anchor 3/.05), hard-fail reasons POSE_OUT_OF_SUPPORTED_RANGE/POSE_JUMP_EXCEEDS_PHYSICAL_RATE (:536-568), UV-warp/hybrid render, `FastUVFallbackRenderer` on hybrid failure (:699-724) | REAL + BASE fallback |
| neural_face.py | `NeuralFaceEngine`: multi-view correspondence conditioning + neural synthesis; unobserved texels â†’ mirror else median skin tone (:204,397-400) | REAL (model-dependent) |
| face_detail.py | Iris/oral aperture constraints (`extract_detail`) | REAL |
| lbs.py | `ArticulatedHeadLayer` differentiable LBS | REAL |
| semantic_shape.py / shape_keys_plus.py | Semantic sliders; asymmetric shape-key manager | REAL |
| monocular_regressors.py | ArcFaceâ†’metric-shape adapter, disentangled/asymmetric encoders, lip-phoneme regressor, ONNX export (trainable heads; [INFERENCE] untrained unless weights supplied) | REAL arch / weights [INFERENCE] |
| arkit_bridge.py | ARKit-52 â†” 3DMM expression bridge | REAL |
| gpu_background.py | GPU jump-flood background propagation into silhouette ribbon | REAL |

### 1.2 `rendering/` (20 files)
| File | Purpose | Status |
|---|---|---|
| __init__.py | Re-exports renderers | INFRA |
| RENDERING_MODELS.md / RENDERING_MODELS_LICENSE | Provenance of CC0 lighting probes | doc |
| assets/stage10_cc0_probes.npz (350 KB) + .json | CC0 environment probe bundle | asset |
| vertex_raster.py | Exact CPU z-buffer with barycentric attr interpolation (depth, normals, swap-mask weights); `orthographic_pixels` | REAL (Python-loop raster; perf [INFERENCE]) |
| physical_buffers.py | GPU buffers (depth/normal/albedo/UV), SH basis+irradiance `shade_lambertian_sh`, differentiable buffers | REAL |
| physical_camera.py | `PhysicalCameraMatcher`, `DominantKeyLightAligner`, `TemporalSpatialSwapDenoiser`, micro-pore normal synth, CC0 probe loader | REAL/HEUR |
| hybrid_renderer.py | Mesh + boundary Gaussian splats (eyebrow/eyelash/beard/inner_mouth only, :19,492), `FastUVFallbackRenderer` low-VRAM path (:553,629-650) | REAL + BASE fallback |
| intrinsic_texture.py | Intrinsic UV cleanup, 4-level pyramid | REAL |
| uv_baker.py | PBR atlas bake | REAL |
| uv_gaussian_map.py | UV micro-detail planes + continuous face-size LoD | REAL |
| color_and_flow.py | `.cube` 3D LUT, histogram match, tangent-aligned hair shading | REAL |
| skin_calibration.py | Linear-light skin tone calibration, gray-world fallback (:119) | REAL |
| swap_region_3d.py | Full craniofacial 3D swap mask + Laplacian jaw adaptation (`JawTarget`) | REAL |
| perimeter_cage.py | Stage 6B hair/ear/neck cage warp | REAL (2D field on perimeter) |
| silhouette_inpaint.py / silhouette_replacement.py | Silhouette delta inpaint, scope plan, donor history | REAL |
| strand_occlusion.py | 4-cue occlusion (depth z-test, â€¦) + strand matte | REAL |
| super_res_tiles.py | VRAM-aware overlapped tiling of head restoration | REAL |

### 1.3 `engine/` (35 files)
| File | Purpose | Status |
|---|---|---|
| __init__.py, detection/__init__.py, diffusion/__init__.py, volumetric/__init__.py, audio_sync/__init__.py | package boundaries | INFRA |
| volumetric/session.py (64 KB) | `ReferencePipeline`: trackâ†’INSwapperâ†’optional 3D reenactment; auto-3D limits 30/25, roll 40; `profile_aware_mask` | REAL |
| volumetric/identity_swapper.py | INSwapper ONNX with emap (`embedding_map`) | REAL (2D net) |
| volumetric/swapper_engine.py | Architecture-aware adapter for registered swappers | REAL |
| volumetric/reference_adapter.py | LivePortrait NumPy/ONNX adapter (relative motion, stitching, 3D warp) | REAL |
| volumetric/preprocess.py | Similarity alignment, `roll_compensated_aligned_crop`, `landmark_crop` | REAL |
| volumetric/presence.py | `PoseFade` 65â†’82Â°, hysteresis 2.5Â°; `SwapPresence` hold/ramp | REAL |
| volumetric/angle_diagnostics.py | Per-frame angle/jitter/mask/ring-luma/sharpness JSONL metrics; `yaw_bucket` | REAL (diagnostics, axis 5) |
| detection/face_tracker.py | `FaceTracker`: Hungarian + ArcFace gallery + scene cut + Kalman-like velocity | REAL (no ByteTrack) |
| detection/track_policy.py | `identity_lock_v1` state machine TRACKED/UNCERTAIN/LOST/REACQUIRED/SCENE_CUT/EXPIRED | REAL |
| detection/dense_landmarks.py | 106/203 pt ONNX + 478 mesh (optional `face_landmarker.task`); `geometric_pose` = 5-pt legacy estimate | REAL / HEUR pose |
| diffusion/base.py, flux_backend.py, sdxl_backend.py, identity_adapter.py, tile_manager.py | Optional lazy diffusion VFX backends, memory gates, cosine-window ROI tiling; family-locked adapters | opt-in; not on default swap path |
| audio_sync/audio_alignment.py, audio_features.py, lip_sync_adapter.py, track_selection.py | VFRâ†’sample windows, log-mel, Wav2Lip ONNX, track pick; unavailable model preserves swapped crop | REAL gated |
| pipeline_runner.py (73 KB) | `MasterPipelineRunner`: checkpoint npz, windowed frames, pause/cancel/OOM | REAL (axis 7) |
| runtime.py | `OrtModel`, ORT session facade, TRTâ†’CUDAâ†’CPU fallback reason, `disable_cpu_ep_fallback` | REAL (axis 6) |
| adaptive_runtime.py | Stage 9 adaptive CPU/GPU profiles, `CudaIoBinder` | REAL |
| active_models.py, registry.py, resources.py, scheduler.py, vram_governor.py, vram_watchdog.py | active-model tracking, engine dispatch, fence/ownership, admission, VRAM lease/OOM stepdown | REAL |
| edit_plan.py | Role intervals / `FrameEditPlan` | REAL |
| stage16_pipeline.py | Optional Stage-16 crop pipeline (`ShapeConditioner`) | opt-in |

### 1.4 `identity/` (5)
__init__.py (INFRA) Â· guard.py (`IdentityGuard`/`IdentityRunReport`: render-time ArcFace check against locked signature) Â· lock.py (`LockStore`, hash-bound video+inspection+trackâ†’FaceSet) Â· report.py (contact-sheet/report collector; biometric templates local only) Â· target_lock.py (`TargetIdentityLock`, `FaceTrackRegistry`, `KalmanBox` 8-state CV, 5Ã—3 pose banks, MIN_COSINE 0.42, BUCKET_QUALITY 100) â€” all REAL.

### 1.5 `faceset/` (8)
__init__.py Â· faceset_reader.py (reader, legacy normalizer, V2 export; mean+medoid 512-D ArcFace, pose-bin clustering) Â· identity_blend.py (Linear/Slerp FrÃ©chet mean + tangent dials) Â· pose_matrix.py (Stage 0 3Ã—3 matrix; `GUARD_MAX_COSINE_DISTANCE=0.80`, `_FILL_PENALTY=0.5`, flag `ROOP_STAGE0`) Â· pose_selector.py (`classify_pose_bin`: yaw Â±15Â°, pitch Â±12Â° â†’ 3Ã—3 names) Â· recognizer.py (ArcFace ONNX; 5-pt `estimateAffinePartial2D` LMEDS to reference) Â· schemas.py (immutable V2 manifest) Â· validator.py (zip-bomb/path security). REAL; `.fsz` = zip FaceSet archive validated by `ArchiveValidator` [INFERENCE on ext name; reader exports V2 archive].

### 1.6 `tracking/` (9)
__init__.py Â· filters.py (One-Euroâ†’no-overshoot Kalman, `DEFAULT_SPECS` e.g. translation 1.5/0.6/8, expression 6/1/15, gaze 1.5/1.5/10 coast 0.15 s, lighting 0.4/0/1 coast 1 s; `ROTATION_SPEC` 1.5/0.03/8 reset 1400Â°/s; GAP_RESET 0.5 s) Â· gaze.py (iris geometry from mesh478, vergence) Â· identity_lock.py (per-shot weighted-median shape descriptor) Â· lighting.py (per-frame SH fit on generic head) Â· runner.py (calibrate/track video, `HeadTrackingTap`) Â· scene_adaptation.py (time-boxed actor calibration; accepted only if held-out residual â‰¤0.98Ã—generic) Â· telemetry.py (throttled JSON frames, non-blocking; axis 5) Â· video_tracker.py (`VideoHeadTracker3D`, FLAME-lineage, `ShotBoundaryDetector`, 50 psi dims, jaw lever 0.085 m, JAW_MAX 0.6 rad, POSE_PRIOR 45Â°). REAL.

### 1.7 `swappers/` (4)
__init__.py Â· swapper_base.py (`SWAPPER_CONFIGS`: inswapper_128 (128 px, emap), hyperswap_1a/1b/1c_256, simswap_unofficial_512, ghost_1_256, â€¦; `SwapperFactory`) Â· hififace.py (HifiFace-256 + ArcFace converter; no FLAME inputs) Â· pixel_boost.py (sub-patch tiling, raised-cosine). 2D swappers = BASELINE identity transfer.

### 1.8 Top-level `*.py` (39)
| File | Purpose | Status |
|---|---|---|
| __init__.py | package marker | INFRA |
| angle_tracker.py | 3-stage offline multi-angle scan; `pose_match_threshold` (same-angle cosine 0.40, profile continuity floor 0.22 when IoU>0.55 and \|yaw\|â‰¥65Â°, linear in pose delta/90Â°); `MAX_BRIDGE_GAP=8`; `optical_flow_landmarks_3d` LK 21Ã—21, 3 pyr, z kept | REAL |
| pose_matrix.py | Stage 13 frozen 3Ã—3 source-pose identities (yaw edges Â±25, pitch Â±15, anchors Â±75/Â±35, optional 96-bin sphere, roll comp âˆ’180..180) | REAL |
| relighting_3d.py | SH + normals + Blinn-Phong "NeuralRelighter3D" (not neural) | REAL SH / HEUR |
| temporal.py | `OneEuroFilter`, `TrackTemporalStabilizer` (Kalman+OneEuro+SO(3) SLERP), `CoupledTrackSmoother`, `TemporalMaskEMA` | REAL |
| temporal_stabilizer.py | Older `OneEuroFilter`, `PoseHysteresis`, `TemporalStabilizer` (duplicate of temporal.py concerns) | REAL / duplicate |
| mesh_raycaster.py | see Â§0 | BASE (misnamed) |
| faceset_builder.py / faceset_loader.py / faceset_synthesizer.py | curation to Stage-7 archive; loader with PnP-pose per view (`aligned_crop_512`, `LandmarkEngine`); single-photo synthesis mesh/LivePortrait/UltraMax (`angle_uv`) | REAL; synthesized views flagged `is_synthetic` |
| flame_3d_engine.py | FLAME coefficient extraction ONNX/MediaPipe478, gaze | REAL gated |
| expression_restorer.py | Stage 2.5 expression residual (EAR, aperture) | REAL |
| liveportrait_trt.py | LivePortrait retargeting via checked sessions | REAL |
| gpu_compositor.py / gpu_transforms.py / cvcuda_ops.py | Torch-CUDA warp/feather/Lab colour transfer/laplacian blend; CV-CUDA ops | REAL (axis 3/6) |
| hires_handoff.py | Stage2â†’3 hi-res warp, YCbCr moment/histogram skin match | REAL |
| masking_engine.py | 19-class + XSeg + CLIPSeg masking | REAL gated |
| optical_flow_tracker.py | RAFT-small / NVIDIA HW flow with scene resets | REAL |
| optimized_prepass.py / optimized_processor.py / optimized_trt_engine.py | sparse prepass, 3-stage streaming, strict TRT | REAL (axis 6/7) |
| trt_session_builder.py / trt_fp8_builder.py | strict TRT sessions, FP8 E4M3 Q/DQ validation | REAL |
| vectorized_pipeline.py | CPU/GPU batch identity+composite | REAL |
| hardware_streamer.py / nv_video_codec.py | NVDEC/NVENC FFmpeg, shared-memory ring | REAL (axis 7) |
| heterogeneous_scheduler.py / openvino_coprocessor.py | Intel+NVIDIA hybrid scheduling; OpenVINO co-processor (consumes mesh_raycaster) | REAL/[INFERENCE] benefit |
| queue_manager.py / job_queue.py / stream_checkpoint.py / segment_dispatcher.py / batch_matrix.py | 10-state SQLite queue; checkpointed segment render; GOP-aligned split/concat; batch_matrix = alias to `roop_3d.queue.batch_matrix` | REAL |
| model_registry.py | Declarative model manifest + resilient downloader | INFRA |
| render_policy.py / thread_benchmark.py / run_benchmarks.py | CUDA ctx/stream policy, benchmarks | INFRA |
| vfx_export.py | Alpha/foreground passes, FCPXML/XMEML/EDL, provenance manifest | REAL |

## 2. Axis findings (slice)
1. **Detection/landmarks/angle**: live angle source is 5-pt geometric pose (engine/detection/dense_landmarks.py:102-106); true PnP only in offline/asset code and tracker (video_tracker `solve_view_pose` with pose prior 45Â°, :59,404). Thresholds: auto-3D 30Â°/25Â° (session.py:1050), roll 40Â° (:1029), tracker yaw 85Â° (video_tracker.py:79), reenactment 80Â° (reenactment.py:70), swap fade 65â€“82Â° (presence.py:17-18), far-cheek trim 30â€“40Â° (session.py:121), jaw fit 0.75 foreshortening (jaw_fit.py:31), Raycaster far-cheek 45Â° (mesh_raycaster.py:127). Profile limit of identity gate relaxes at |yaw|â‰¥65Â° (angle_tracker.py:41-47).
2. **2D vs 3D**: Hybrid â€” INSwapper/HifiFace/hyperswap remain 2D identity nets; 3D contributes mesh-fit asset (multiview MAP, MakeHuman/canonical/FLAME), z-buffered UV re-render, TPS/perspective pose-blend. TPS "2.5D" is explicit fallback (tps_fallback.py:1,351). `.fsz`/FaceSet: reader/validator/V2 export in faceset/; coverage 12Ã—8 bins (faceset_3d.py:20).
3. **Quality**: SH relight (photometry.py, relighting_3d.py:126), UV albedo prior (albedo_prior.py), multiband blend (uv_completion.py:29), strand-aware occlusion (strand_occlusion.py:115), tile super-res (super_res_tiles.py:76), LUT/histogram (color_and_flow.py), skin calibration linear light (skin_calibration.py:169). Gaussian splats limited to 4 boundary regions (hybrid_renderer.py:19,492).
4. **Temporal**: One-Euro+no-overshoot Kalman (filters.py:1-62), SO(3) SLERP (temporal.py:139-177; filters.py:170), view-selection hysteresis (view_selection.py:19-25), PoseFade hysteresis (presence.py), scene cut thumbnail 32Ã—32 (face_tracker.py:180-185; video_tracker.py:189-211), kinematic cuts (video_tracker.py:343), TrackPolicy SCENE_CUT park/reacquire (face_tracker.py:156-168). No ByteTrack.
5. **CLI/telemetry/diagnostics**: tracking/telemetry.py (throttled JSON), volumetric/angle_diagnostics.py (per-frame JSONL: jitter, ring-luma, sharpness, yaw_bucket), identity/report.py, `ActiveModel` switch-failure registry (engine/active_models.py).
6. **Compute**: ORT TRTâ†’CUDAâ†’CPU with recorded fallback_reason (engine/runtime.py:105,218), strict TRT builder with canaries (trt_session_builder.py:103,385), FP8 Q/DQ validation (trt_fp8_builder.py:44), VRAM governor/watchdog (engine/vram_governor.py:50; vram_watchdog.py:33), SharedExclusiveFence (resources.py:14), GPU rasterizer is OpenGL (ModernGL) with no software fallback (full_head.py:1).
7. **Queue/IO**: queue_manager.py 10-state SQLite; pipeline_runner.py checkpoint npz + pause/cancel/OOM recovery; segment_dispatcher.py GOP-aligned lossless concat; hardware_streamer.py NVDEC/NVENC; audio_alignment.py rational VFRâ†’sample windows.
8. **Workstation UI**: only consumers inside slice: reconstruction/backbone_status.py (backbone dropdown), swappers registry; `relighting_3d` also used by server/workstation_preview.py:141 (outside slice).

## 3. Gaps / stubs / defects found
- **mesh_raycaster.py is not a raycaster**: docstring promises Embree/BVH exact per-pixel occlusion in <0.8 ms; implementation is a yaw>45Â° hard-coded index/threshold mask; `use_simd`, `bvh_enabled` unused; reported `execution_time_ms` is measured not guaranteed (mesh_raycaster.py:1-8,127-150). Production swap path does not use it (consumers: heterogeneous_scheduler + test only). Real self-occlusion lives in vertex_raster/GPU rasterisers.
- **`NeuralRelighter3D` has no neural component** (relighting_3d.py:287-345); naming overclaims.
- **Live per-frame pose is 5-pt geometric**, not PnP/mesh-fit (dense_landmarks.py:102-106), so the 30Â°/25Â° auto-3D gate and PoseFade operate on a coarse estimate; PnP flip ambiguity documented in faceset_3d.py:31.
- **Pose grids are inconsistent** across four modules (12Ã—8, 5Ã—3, 3Ã—3 Â±25/Â±15, 3Ã—3 Â±15/Â±12) â€” risk of mismatched bin semantics (faceset_3d.py:20; target_lock.py:24; pose_matrix.py:24-27; pose_selector.py:19-31). `YAW_BINS,PITCH_BINS=12,8` duplicated in top-level pose_matrix.py:28.
- **Duplicate temporal implementations**: temporal.py vs temporal_stabilizer.py both define `OneEuroFilter`; tracking/filters.py wraps temporal.py; reenactment.py has a third wrapper (reenactment.py:145).
- **No ByteTrack / no low-score second pass**; association is single-pass Hungarian with IoU gating â‰¥0.1 (face_tracker.py:46,333).
- **Hybrid/neural fallbacks silently degrade**: reenactment falls to `FastUVFallbackRenderer` on hybrid failure (reenactment.py:699-724); neural_face fills unobserved texels with mirror/median skin (neural_face.py:397-400); faceset_3d falls to `canonical_symmetric_fallback` covering face only, not skull (faceset_3d.py:203-220); anatomical_pack has analytic fallback head (:747-798); flame_fitter `SyntheticFLAMEModel` (:164).
- **TPS/2.5D warping remains in the live path** (frontalization, expression_tps, pose_blend) â€” baseline, not volumetric.
- **vertex_raster.py is a Python per-triangle loop** (rendering/vertex_raster.py:50-77) [INFERENCE: slow for dense meshes; used where GPU raster not required].
- Identity swap quality is bounded by 128 px INSwapper unless hyperswap/PixelBoost enabled (swapper_base.py SWAPPER_CONFIGS).
- monocular_regressors.py heads are untrained architectures unless weights provided [INFERENCE].
- Task statement said reconstruction=50 / rendering=18 files; inventory has 52 / 20 (the extra are `.npz` + `.npz.json` assets), all accounted for above.

Slice files in inventory: 172 | files accounted for: 172 | omitted: none
(inventory counts: reconstruction 52 + rendering 20 + engine 35 + identity 5 + faceset 8 + tracking 9 + swappers 4 = 133 in subdirs; top-level `app/roop_3d/*.py` 39; verified via `grep -cP "\tapp/roop_3d/(...)/"` and `grep -cP "\tapp/roop_3d/[^/]+\.py$"` against roop3d_files.tsv)
