# 02 — roop-3d complete audit

Repo: `rishabh4496/roop-3d` (local checkout `G:/pinokio/api/roop-3d`, origin `roop-3d.git`). Audited tracked-file inventory: **1224 files** (`_inventory/roop3d_files.tsv`).
Method: three parallel auditors each owned a disjoint slice, read every file (or, for homogeneous sets such as tests, header + test names), and emitted one table row per file with status/evidence (`file:line`). Items not read line-by-line are marked `[INFERENCE]`.
Slices: **R3_1** core 3D (reconstruction/rendering/engine/identity/faceset/tracking/swappers + pose/temporal top-level modules, 172 files) · **R3_2** pipeline/server/runtime (335) · **R3_3** UI/tests/audit tools/docs/root (754). Slice totals exceed 1224 because slices overlap on `app/roop_3d/*.py` helpers and root files; union coverage verified by script: 1222/1224 paths appear by full path or basename; the 2 remaining are the `*.npz.json` sidecars of `makehuman_head_v1.npz` and `stage10_cc0_probes.npz` (documented with their `.npz`). Untracked: `.agents/`, `.claude/`, `docs/agents/`, `concepts_brief_face_swap_quality.md`, `skills-lock.json` (tooling, not product).

## Executive summary (orchestrator synthesis)

| Axis | State at audit time | Decisive evidence |
|---|---|---|
| 1 Detection / angle | Live per-frame pose is a **5-point geometric estimate**; PnP exists only in asset building and the offline video tracker. Auto-3D engages at \|yaw\|>30° / \|pitch\|>25°, roll>40° hands back to 2D | `engine/detection/dense_landmarks.py:102-106`, `engine/volumetric/session.py:121-132,1029-1050` |
| 2 3D | Identity transfer is still a 2D network (INSwapper/HifiFace); 3D conditions pose/geometry/texture. Real z-buffers exist (`vertex_raster.py`, GPU rasterisers, `uv_synthesizer.py`); `mesh_raycaster.py` is **not** a raycaster (hard-coded index mask for \|yaw\|>45°, `use_simd`/`bvh_enabled` unused) | `reconstruction`/`rendering` slice §0 |
| 3 Quality | Laplacian/Poisson/feather blends, BiSeNet+XSeg, SH relight, guided-filter matte, GPEN/CodeFormer/UltraMax all present; warping is bilinear `cv2.warpAffine`; default XSeg name `xseg_3.onnx` has no manifest entry | `compositor/*`, `boundary_fusion.py:34`, `masks.py:3-6` |
| 4 Temporal | One-Euro, no-overshoot Kalman, SO(3) SLERP One-Euro, ArcFace ladder re-ID, thumbnail scene cut all exist. **No ByteTrack** low-score second pass; scene-cut is not wired into the compositor render path; temporal-detail stabiliser reaches 28% flicker reduction vs 70% target | `tracking/filters.py`, `engine/detection/face_tracker.py:46,333`, `docs/TEMPORAL_STABILITY.md` |
| 5 CLI / telemetry | No headless render CLI, no Rich TUI, no `--json-logs`; only `--server_share`. Custom ANSI HUD + two 14-section reports exist | `server/__main__.py:67`, `telemetry/pinokio_terminal.py` |
| 6 Compute | ORT TRT→CUDA→CPU with recorded fallback, FP16 default, FP32 swapper/GPEN, FP8 gated, **INT8 off**; TRT opt-level 3 (`core/ort_session.py:38`) vs 5 (`trt_session_builder.py:242`); measured 0.26–0.57 fps on RTX 4070 benchmarks | `docs`/`audit/STAGE15_BENCHMARKS.json` |
| 7 Queue / IO | Three state enums (`JobState` 10, `CanonicalJobState` 12 with "10" docstring, `LifecycleState` 10) and two parallel stacks started in `server/app.py:86-100`; atomic checkpoints and GOP stream-copy slicing are real; Stage-17 runner drops subtitles/metadata, refuses HDR; four writer stacks, two classes named `NVHardwareVideoWriter` | `queue_manager.py:47`, `jobs/schema.py:27`, `pipeline/video_job_runner.py:12-14` |
| 8 UI | Rich 9-screen workstation; no temporal graph, no yaw-refusal display, no >90° bin; "scene adaptation"/"lip lock" are localStorage flags with no consumer; `ClipPathSplitReveal.tsx` dead | `WorkstationControls3D.tsx:18-19` |
| Integrity | **`verify_isolation.py --report` is a hard-coded pass** (`stage00.py:35-36`, `isolation.py:184-197`); `local_models.py --verify` writes a file as side effect and depends on the sibling checkout path; real-profile extreme-yaw tests skip (fixtures absent) | R3_3 §13 |

Defects discovered by the audit that Wave 2 addresses are enumerated in report 03.

---



<!-- ===== R3_1 — core 3D slice ===== -->
## R3_1 — roop-3d core 3D slice (reconstruction / rendering / engine / identity / faceset / tracking / swappers / top-level pose+temporal+faceset modules)

Paths are relative to `app/roop_3d/`. Status: **REAL** = implemented algorithm; **HEUR** = heuristic/approximation labelled as such in code; **BASE** = baseline/fallback/stub; **INFRA** = non-3D plumbing (listed for coverage). Evidence = file:line where I read it; everything else is from module docstring + public symbol list (AST scan) and flagged [INFERENCE] if it describes behaviour I did not read.

### 0. Headline verdicts (what is true 3D vs baseline)

| Question | Verdict | Evidence |
|---|---|---|
| PnP used? | **Yes, in 6 independent places**, all OpenCV: `cv2.solvePnP` ITERATIVE + LM refine on canonical 68-pt model with dynamic focal (18–135 mm bounds); `solvePnPRansac(EPNP,100 it)`+`RefineLM` for dense head; `SQPNP` for MakeHuman and PoseBlend anchors | reconstruction/tps_fallback.py:175-243; dynamic_camera.py:146-215; full_head.py:412-430 (`solve_dense_pose`); makehuman_head.py:150; pose_blend.py:248-251; flame_fitter.py:264 |
| PnP flip hazard acknowledged | faceset_3d falls back to stored measured pose on degeneracy ("measured pitch -177° on frontal") | reconstruction/faceset_3d.py:25-31 |
| Per-frame video pose in live swap path | Not PnP: `FaceObservation.pose` comes from 5-pt `core.pose.estimate_pose_5pt` ("not calibrated 3D reconstruction") | engine/detection/dense_landmarks.py:102-106 |
| z-buffer self-occlusion | **Yes, real** in 3 renderers: exact CPU triangle z-buffer w/ barycentric attrs (`vertex_raster.py:34-77`), GPU ModernGL hardware depth test (`dense_correspondence.py:104-105`, `full_head.py`, `physical_buffers.py:5`), UV-texel visibility by `fillConvexPoly` depth map with `DEPTH_TOLERANCE_M=0.008`, `GRAZING_MIN_COS=0.25` (uv_synthesizer.py:373-412, 509) |
| `mesh_raycaster.py` ("Embree/BVH, <0.8 ms") | **BASE/misleading** — no Embree, no BVH, no ray cast. If `|yaw|>45` it zeroes a hard-coded landmark index list (68-pt: 13-16,44-47 / 0-3,36-38,41) or vertices with `x>0.025 & z<0.04`; otherwise all visible; z-buffer is vertex splats (5×5 footprint <200 verts). `use_simd`/`bvh_enabled` flags are stored and never read. Only consumer: heterogeneous_scheduler.py + its test | mesh_raycaster.py:42-52, 127-150, 163-200; grep consumers |
| SH relighting | **Real 2nd-order (9-coeff) SH**, Ramamoorthi–Hanrahan constants, regularised LS fit; linear-light fit with 0.5 albedo prior in reconstruction/photometry.py:1-30; per-frame tracker SH `MeshSHLightingEstimator` (tracking/lighting.py:53). `relighting_3d.NeuralRelighter3D` name is misleading: no neural net — SH shading + luminance-ratio cast-shadow mask + Blinn-Phong (k=0.25, shininess 24) | relighting_3d.py:19-24,126-150,287-345 |
| UV albedo fusion | **Real**: visibility-weighted multi-view UV synth, confidence-gated (`OBSERVED_MIN_WEIGHT=0.02`), multiband seam blend, albedo prior fill (skin Lab gamut gate, ≥200 skin texels), intrinsic 4-level pyramid | uv_synthesizer.py:123,373-552; uv_completion.py:29,155,270; albedo_prior.py:32-36,82-88; intrinsic_texture.py:26 |
| Pose bins | Four incompatible grids (see §4). Faceset coverage 12×8 bins (yaw ±90 / pitch ±60 clamp); target-lock 5×3 (yaw ±75, 30° wide; pitch ±45); Stage-0/13 3×3 yaw edges ±25 / pitch ±15, anchors yaw ±75, pitch ±35; legacy PoseSelector 3×3 ±15/±12 | faceset_3d.py:20,55-58; target_lock.py:24-28; pose_matrix.py:24-28; pose_selector.py:17-34 |
| Kalman / One-Euro / SLERP | **All three exist.** One-Euro + constant-velocity Kalman (no-overshoot) per-channel; SO(3) One-Euro using quaternion SLERP; Kalman box in target-lock; embedding Slerp/Fréchet mean for identity blend | tracking/filters.py:1-62,170; temporal.py:16,56,139-177,251; temporal_stabilizer.py:5; identity/target_lock.py:31; faceset/identity_blend.py:21-93 |
| Scene-cut detection | **Yes, 3 sites** — 32×32 thumbnail diff (>0.45 cut; >0.18 AND Bhattacharyya>0.7 cut) in FaceTracker (face_tracker.py:180-185) and `ShotBoundaryDetector` (video_tracker.py:189-211, settings 71-76); plus kinematic cuts: face-centre jump >2.5 IOD or rotation >2700 °/s within dt ≤0.2 s (video_tracker.py:343) | cited |
| ArcFace re-ID | **Yes**: TrackPolicy cosine ladder strong 0.55 / match 0.35 / continuity 0.25 / weak 0.15, reacquire 0.5 (+margin 0.08), ambiguity margin 0.06, uncertain_limit 45, lost_limit 64, retention 1800 frames; target-lock MIN_COSINE 0.42, STRONG 0.55 | track_policy.py:46-62; target_lock.py:12-15; face_tracker.py:205,387-391 |
| ByteTrack | **Absent.** Only token match is a docstring mention in angle_tracker.py (grep). Association = Hungarian (`optimal_assignment`, face_tracker.py:46) on `0.8*cos+0.2*overlap` / `0.6*sim+0.25*shared+0.15*layout` + IoU ≥0.1 gating + constant-velocity predicted box; **no low-confidence second association pass** | face_tracker.py:46,206,320,333,391 |
| Dense-3D engage thresholds | Automatic switch from 2D swap to dense 3D when `|yaw|>30` or `|pitch|>25` (env `ROOP3D_AUTO_3D_YAW/PITCH`, 180 disables); roll >40° hands back to 2D (`ROOP3D_3D_MAX_ROLL`); profile far-cheek mask trim fades in over |yaw| 30→40°; swap presence fades from 65° to 82° off-axis angle `acos(cos yaw cos pitch)` with 2.5° hysteresis, hold 2 frames, ramp 3, forget 30 | volumetric/session.py:121-132,1029-1050; presence.py:17-22,45-58 |
| Tracker/reenactment yaw limits | Video tracker `max_abs_yaw_deg=85` (reject if fit >85; pre-reject 5-pt >100); expression only to 60°; ReenactmentSettings `max_abs_yaw_deg=80`, `expression_max_yaw_deg=60` | video_tracker.py:79-80,399-409; reenactment.py:70-71 |
| True volumetric swap model? | The *identity* is still transferred by a 2D network (INSwapper via `IdentitySwapper`; HifiFace-256 wrapper states exported runtime inputs are RGB+embedding, not FLAME) — 3D only conditions pose/geometry/texture (PoseBlendProjector TPS/perspective-mesh, ReenactmentEngine, neural UltraMax correspondence) | volumetric/identity_swapper.py:1-116; swappers/hififace.py:3-5; session.py:33-34,722 |

### 1. File-level inventory

#### 1.1 `reconstruction/` (52 files)
| File | Purpose / key symbols | Status |
|---|---|---|
| __init__.py | Re-exports geometry/TPS/PnP/FLAME/UV APIs ("2.5D TPS fallback" in docstring, :2) | INFRA |
| ARTICULATED_MODELS.md / ARTICULATED_MODELS_LICENSE | Licence/provenance of articulated-head assets (CC0-style notice) | doc |
| assets/makehuman_head_v1.npz (386 KB) + .npz.json | Packed MakeHuman CC0 head arrays + digest sidecar | asset |
| geometry_types.py | `CameraIntrinsics`, `Pose3D`, `Mesh3D`, OpenCV↔OpenGL camera, Rodrigues/quat/Euler converters, UV↔pixel | REAL |
| tps_fallback.py | `get_canonical_3d_landmarks`, `solve_pose_pnp` (solvePnP + dynamic focal), `build_canonical_head_mesh` (Delaunay), `ThinPlateSplineWarper`, `warp_delaunay_piecewise_affine` — "2.5D" | PnP REAL / warp BASE |
| tps_warp.py | TPS helper shared by frontalization/expression (`border_anchors`, `warp_points_to`) | BASE |
| dynamic_camera.py | `DynamicCameraSolver`: focal bounds 18–135 mm eq., PnP + `solvePnPRefineLM`, SQPNP/EPNP fallback | REAL |
| frontalization.py | `frontalize_crop`: 3D-pose frontalization via TPS before swap | BASE (TPS) |
| expression_tps.py | TPS expression warp on 106 landmarks, pinned silhouette, `layout_is_plausible` | BASE (2D warp) |
| pose_blend.py | `PoseBlendProjector`: estimated 3D proxy, SQPNP+LM, z-buffer raster, TPS/perspective-mesh correction; `MAX_SIZE=512`; cos(yaw) from PnP rotation | HEUR (proxy mesh) |
| jaw_fit.py | Per-frame jaw width ratio fit; `MIN_FORESHORTENING=0.75` (~41° yaw); 5 PnP anchors fixed | HEUR |
| multiview_fit.py | Bayesian (MAP) multi-view landmark→identity shape fit; σ_obs 2 mm, prior σ xy 12 mm / z 15 mm, pose σ 8°, Huber δ=3σ; self-occlusion soft weights by normals; `estimate_pose_from_landmarks` | REAL |
| rbf_deform.py | Wendland RBF + Laplacian-regularised template deformation | REAL |
| faceset_3d.py | `observe_poses` (12×8 bins), `reconstruct_faceset`; canonical fallback labelled `canonical_template_fallback` | REAL + BASE fallback (:102-116,203-220) |
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
| full_head.py | `GPUHeadRasterizer`, `FullHeadEngine`, `solve_dense_pose` (EPNP RANSAC 100 it, conf .999, reproj ≥2 px/4% span, ≥6 pts, reject err >10% span), `complete_diffuse_albedo`; "no software raster fallback" | REAL (GPU only) |
| photometry.py | Linear-light SH fit (`fit_sh`), `decompose_albedo`, `fit_target_surface`, `extend_neck` | REAL |
| photometric_gpu.py | OpenGL photometric relight pass at enhancer ROI res (`relight_head`) | REAL |
| albedo_prior.py | Skin-tone prior fill for unseen texels | REAL |
| uv_synthesizer.py | Multi-view UV synth with z-buffer visibility | REAL |
| uv_completion.py | Multiband blend, nearest-texel and `FullHeadUVCompleter` | REAL |
| view_selection.py | `ViewSelector`: hysteresis (MARGIN .035, SWITCH_FRAMES 3, RAMP_STEP .25, SMOOTHING .35, FADE_FLOOR .05, GIMBAL_PITCH 75°, MAX_GAP .25 s) | REAL |
| view_texture_bank.py | 9 acquisition buckets + two-view interpolation on view sphere | REAL |
| swap_coverage.py | Swap coverage = rasterised surface ∪ landmark hull with feather | REAL |
| reenactment.py | `ReenactmentEngine` Stage 4: rig from identity asset, per-track `NeutralCalibrator`, One-Euro settings (pose 1.5/.03, expr 6/1, anchor 3/.05), hard-fail reasons POSE_OUT_OF_SUPPORTED_RANGE/POSE_JUMP_EXCEEDS_PHYSICAL_RATE (:536-568), UV-warp/hybrid render, `FastUVFallbackRenderer` on hybrid failure (:699-724) | REAL + BASE fallback |
| neural_face.py | `NeuralFaceEngine`: multi-view correspondence conditioning + neural synthesis; unobserved texels → mirror else median skin tone (:204,397-400) | REAL (model-dependent) |
| face_detail.py | Iris/oral aperture constraints (`extract_detail`) | REAL |
| lbs.py | `ArticulatedHeadLayer` differentiable LBS | REAL |
| semantic_shape.py / shape_keys_plus.py | Semantic sliders; asymmetric shape-key manager | REAL |
| monocular_regressors.py | ArcFace→metric-shape adapter, disentangled/asymmetric encoders, lip-phoneme regressor, ONNX export (trainable heads; [INFERENCE] untrained unless weights supplied) | REAL arch / weights [INFERENCE] |
| arkit_bridge.py | ARKit-52 ↔ 3DMM expression bridge | REAL |
| gpu_background.py | GPU jump-flood background propagation into silhouette ribbon | REAL |

#### 1.2 `rendering/` (20 files)
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
| strand_occlusion.py | 4-cue occlusion (depth z-test, …) + strand matte | REAL |
| super_res_tiles.py | VRAM-aware overlapped tiling of head restoration | REAL |

#### 1.3 `engine/` (35 files)
| File | Purpose | Status |
|---|---|---|
| __init__.py, detection/__init__.py, diffusion/__init__.py, volumetric/__init__.py, audio_sync/__init__.py | package boundaries | INFRA |
| volumetric/session.py (64 KB) | `ReferencePipeline`: track→INSwapper→optional 3D reenactment; auto-3D limits 30/25, roll 40; `profile_aware_mask` | REAL |
| volumetric/identity_swapper.py | INSwapper ONNX with emap (`embedding_map`) | REAL (2D net) |
| volumetric/swapper_engine.py | Architecture-aware adapter for registered swappers | REAL |
| volumetric/reference_adapter.py | LivePortrait NumPy/ONNX adapter (relative motion, stitching, 3D warp) | REAL |
| volumetric/preprocess.py | Similarity alignment, `roll_compensated_aligned_crop`, `landmark_crop` | REAL |
| volumetric/presence.py | `PoseFade` 65→82°, hysteresis 2.5°; `SwapPresence` hold/ramp | REAL |
| volumetric/angle_diagnostics.py | Per-frame angle/jitter/mask/ring-luma/sharpness JSONL metrics; `yaw_bucket` | REAL (diagnostics, axis 5) |
| detection/face_tracker.py | `FaceTracker`: Hungarian + ArcFace gallery + scene cut + Kalman-like velocity | REAL (no ByteTrack) |
| detection/track_policy.py | `identity_lock_v1` state machine TRACKED/UNCERTAIN/LOST/REACQUIRED/SCENE_CUT/EXPIRED | REAL |
| detection/dense_landmarks.py | 106/203 pt ONNX + 478 mesh (optional `face_landmarker.task`); `geometric_pose` = 5-pt legacy estimate | REAL / HEUR pose |
| diffusion/base.py, flux_backend.py, sdxl_backend.py, identity_adapter.py, tile_manager.py | Optional lazy diffusion VFX backends, memory gates, cosine-window ROI tiling; family-locked adapters | opt-in; not on default swap path |
| audio_sync/audio_alignment.py, audio_features.py, lip_sync_adapter.py, track_selection.py | VFR→sample windows, log-mel, Wav2Lip ONNX, track pick; unavailable model preserves swapped crop | REAL gated |
| pipeline_runner.py (73 KB) | `MasterPipelineRunner`: checkpoint npz, windowed frames, pause/cancel/OOM | REAL (axis 7) |
| runtime.py | `OrtModel`, ORT session facade, TRT→CUDA→CPU fallback reason, `disable_cpu_ep_fallback` | REAL (axis 6) |
| adaptive_runtime.py | Stage 9 adaptive CPU/GPU profiles, `CudaIoBinder` | REAL |
| active_models.py, registry.py, resources.py, scheduler.py, vram_governor.py, vram_watchdog.py | active-model tracking, engine dispatch, fence/ownership, admission, VRAM lease/OOM stepdown | REAL |
| edit_plan.py | Role intervals / `FrameEditPlan` | REAL |
| stage16_pipeline.py | Optional Stage-16 crop pipeline (`ShapeConditioner`) | opt-in |

#### 1.4 `identity/` (5)
__init__.py (INFRA) · guard.py (`IdentityGuard`/`IdentityRunReport`: render-time ArcFace check against locked signature) · lock.py (`LockStore`, hash-bound video+inspection+track→FaceSet) · report.py (contact-sheet/report collector; biometric templates local only) · target_lock.py (`TargetIdentityLock`, `FaceTrackRegistry`, `KalmanBox` 8-state CV, 5×3 pose banks, MIN_COSINE 0.42, BUCKET_QUALITY 100) — all REAL.

#### 1.5 `faceset/` (8)
__init__.py · faceset_reader.py (reader, legacy normalizer, V2 export; mean+medoid 512-D ArcFace, pose-bin clustering) · identity_blend.py (Linear/Slerp Fréchet mean + tangent dials) · pose_matrix.py (Stage 0 3×3 matrix; `GUARD_MAX_COSINE_DISTANCE=0.80`, `_FILL_PENALTY=0.5`, flag `ROOP_STAGE0`) · pose_selector.py (`classify_pose_bin`: yaw ±15°, pitch ±12° → 3×3 names) · recognizer.py (ArcFace ONNX; 5-pt `estimateAffinePartial2D` LMEDS to reference) · schemas.py (immutable V2 manifest) · validator.py (zip-bomb/path security). REAL; `.fsz` = zip FaceSet archive validated by `ArchiveValidator` [INFERENCE on ext name; reader exports V2 archive].

#### 1.6 `tracking/` (9)
__init__.py · filters.py (One-Euro→no-overshoot Kalman, `DEFAULT_SPECS` e.g. translation 1.5/0.6/8, expression 6/1/15, gaze 1.5/1.5/10 coast 0.15 s, lighting 0.4/0/1 coast 1 s; `ROTATION_SPEC` 1.5/0.03/8 reset 1400°/s; GAP_RESET 0.5 s) · gaze.py (iris geometry from mesh478, vergence) · identity_lock.py (per-shot weighted-median shape descriptor) · lighting.py (per-frame SH fit on generic head) · runner.py (calibrate/track video, `HeadTrackingTap`) · scene_adaptation.py (time-boxed actor calibration; accepted only if held-out residual ≤0.98×generic) · telemetry.py (throttled JSON frames, non-blocking; axis 5) · video_tracker.py (`VideoHeadTracker3D`, FLAME-lineage, `ShotBoundaryDetector`, 50 psi dims, jaw lever 0.085 m, JAW_MAX 0.6 rad, POSE_PRIOR 45°). REAL.

#### 1.7 `swappers/` (4)
__init__.py · swapper_base.py (`SWAPPER_CONFIGS`: inswapper_128 (128 px, emap), hyperswap_1a/1b/1c_256, simswap_unofficial_512, ghost_1_256, …; `SwapperFactory`) · hififace.py (HifiFace-256 + ArcFace converter; no FLAME inputs) · pixel_boost.py (sub-patch tiling, raised-cosine). 2D swappers = BASELINE identity transfer.

#### 1.8 Top-level `*.py` (39)
| File | Purpose | Status |
|---|---|---|
| __init__.py | package marker | INFRA |
| angle_tracker.py | 3-stage offline multi-angle scan; `pose_match_threshold` (same-angle cosine 0.40, profile continuity floor 0.22 when IoU>0.55 and \|yaw\|≥65°, linear in pose delta/90°); `MAX_BRIDGE_GAP=8`; `optical_flow_landmarks_3d` LK 21×21, 3 pyr, z kept | REAL |
| pose_matrix.py | Stage 13 frozen 3×3 source-pose identities (yaw edges ±25, pitch ±15, anchors ±75/±35, optional 96-bin sphere, roll comp −180..180) | REAL |
| relighting_3d.py | SH + normals + Blinn-Phong "NeuralRelighter3D" (not neural) | REAL SH / HEUR |
| temporal.py | `OneEuroFilter`, `TrackTemporalStabilizer` (Kalman+OneEuro+SO(3) SLERP), `CoupledTrackSmoother`, `TemporalMaskEMA` | REAL |
| temporal_stabilizer.py | Older `OneEuroFilter`, `PoseHysteresis`, `TemporalStabilizer` (duplicate of temporal.py concerns) | REAL / duplicate |
| mesh_raycaster.py | see §0 | BASE (misnamed) |
| faceset_builder.py / faceset_loader.py / faceset_synthesizer.py | curation to Stage-7 archive; loader with PnP-pose per view (`aligned_crop_512`, `LandmarkEngine`); single-photo synthesis mesh/LivePortrait/UltraMax (`angle_uv`) | REAL; synthesized views flagged `is_synthetic` |
| flame_3d_engine.py | FLAME coefficient extraction ONNX/MediaPipe478, gaze | REAL gated |
| expression_restorer.py | Stage 2.5 expression residual (EAR, aperture) | REAL |
| liveportrait_trt.py | LivePortrait retargeting via checked sessions | REAL |
| gpu_compositor.py / gpu_transforms.py / cvcuda_ops.py | Torch-CUDA warp/feather/Lab colour transfer/laplacian blend; CV-CUDA ops | REAL (axis 3/6) |
| hires_handoff.py | Stage2→3 hi-res warp, YCbCr moment/histogram skin match | REAL |
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

### 2. Axis findings (slice)
1. **Detection/landmarks/angle**: live angle source is 5-pt geometric pose (engine/detection/dense_landmarks.py:102-106); true PnP only in offline/asset code and tracker (video_tracker `solve_view_pose` with pose prior 45°, :59,404). Thresholds: auto-3D 30°/25° (session.py:1050), roll 40° (:1029), tracker yaw 85° (video_tracker.py:79), reenactment 80° (reenactment.py:70), swap fade 65–82° (presence.py:17-18), far-cheek trim 30–40° (session.py:121), jaw fit 0.75 foreshortening (jaw_fit.py:31), Raycaster far-cheek 45° (mesh_raycaster.py:127). Profile limit of identity gate relaxes at |yaw|≥65° (angle_tracker.py:41-47).
2. **2D vs 3D**: Hybrid — INSwapper/HifiFace/hyperswap remain 2D identity nets; 3D contributes mesh-fit asset (multiview MAP, MakeHuman/canonical/FLAME), z-buffered UV re-render, TPS/perspective pose-blend. TPS "2.5D" is explicit fallback (tps_fallback.py:1,351). `.fsz`/FaceSet: reader/validator/V2 export in faceset/; coverage 12×8 bins (faceset_3d.py:20).
3. **Quality**: SH relight (photometry.py, relighting_3d.py:126), UV albedo prior (albedo_prior.py), multiband blend (uv_completion.py:29), strand-aware occlusion (strand_occlusion.py:115), tile super-res (super_res_tiles.py:76), LUT/histogram (color_and_flow.py), skin calibration linear light (skin_calibration.py:169). Gaussian splats limited to 4 boundary regions (hybrid_renderer.py:19,492).
4. **Temporal**: One-Euro+no-overshoot Kalman (filters.py:1-62), SO(3) SLERP (temporal.py:139-177; filters.py:170), view-selection hysteresis (view_selection.py:19-25), PoseFade hysteresis (presence.py), scene cut thumbnail 32×32 (face_tracker.py:180-185; video_tracker.py:189-211), kinematic cuts (video_tracker.py:343), TrackPolicy SCENE_CUT park/reacquire (face_tracker.py:156-168). No ByteTrack.
5. **CLI/telemetry/diagnostics**: tracking/telemetry.py (throttled JSON), volumetric/angle_diagnostics.py (per-frame JSONL: jitter, ring-luma, sharpness, yaw_bucket), identity/report.py, `ActiveModel` switch-failure registry (engine/active_models.py).
6. **Compute**: ORT TRT→CUDA→CPU with recorded fallback_reason (engine/runtime.py:105,218), strict TRT builder with canaries (trt_session_builder.py:103,385), FP8 Q/DQ validation (trt_fp8_builder.py:44), VRAM governor/watchdog (engine/vram_governor.py:50; vram_watchdog.py:33), SharedExclusiveFence (resources.py:14), GPU rasterizer is OpenGL (ModernGL) with no software fallback (full_head.py:1).
7. **Queue/IO**: queue_manager.py 10-state SQLite; pipeline_runner.py checkpoint npz + pause/cancel/OOM recovery; segment_dispatcher.py GOP-aligned lossless concat; hardware_streamer.py NVDEC/NVENC; audio_alignment.py rational VFR→sample windows.
8. **Workstation UI**: only consumers inside slice: reconstruction/backbone_status.py (backbone dropdown), swappers registry; `relighting_3d` also used by server/workstation_preview.py:141 (outside slice).

### 3. Gaps / stubs / defects found
- **mesh_raycaster.py is not a raycaster**: docstring promises Embree/BVH exact per-pixel occlusion in <0.8 ms; implementation is a yaw>45° hard-coded index/threshold mask; `use_simd`, `bvh_enabled` unused; reported `execution_time_ms` is measured not guaranteed (mesh_raycaster.py:1-8,127-150). Production swap path does not use it (consumers: heterogeneous_scheduler + test only). Real self-occlusion lives in vertex_raster/GPU rasterisers.
- **`NeuralRelighter3D` has no neural component** (relighting_3d.py:287-345); naming overclaims.
- **Live per-frame pose is 5-pt geometric**, not PnP/mesh-fit (dense_landmarks.py:102-106), so the 30°/25° auto-3D gate and PoseFade operate on a coarse estimate; PnP flip ambiguity documented in faceset_3d.py:31.
- **Pose grids are inconsistent** across four modules (12×8, 5×3, 3×3 ±25/±15, 3×3 ±15/±12) — risk of mismatched bin semantics (faceset_3d.py:20; target_lock.py:24; pose_matrix.py:24-27; pose_selector.py:19-31). `YAW_BINS,PITCH_BINS=12,8` duplicated in top-level pose_matrix.py:28.
- **Duplicate temporal implementations**: temporal.py vs temporal_stabilizer.py both define `OneEuroFilter`; tracking/filters.py wraps temporal.py; reenactment.py has a third wrapper (reenactment.py:145).
- **No ByteTrack / no low-score second pass**; association is single-pass Hungarian with IoU gating ≥0.1 (face_tracker.py:46,333).
- **Hybrid/neural fallbacks silently degrade**: reenactment falls to `FastUVFallbackRenderer` on hybrid failure (reenactment.py:699-724); neural_face fills unobserved texels with mirror/median skin (neural_face.py:397-400); faceset_3d falls to `canonical_symmetric_fallback` covering face only, not skull (faceset_3d.py:203-220); anatomical_pack has analytic fallback head (:747-798); flame_fitter `SyntheticFLAMEModel` (:164).
- **TPS/2.5D warping remains in the live path** (frontalization, expression_tps, pose_blend) — baseline, not volumetric.
- **vertex_raster.py is a Python per-triangle loop** (rendering/vertex_raster.py:50-77) [INFERENCE: slow for dense meshes; used where GPU raster not required].
- Identity swap quality is bounded by 128 px INSwapper unless hyperswap/PixelBoost enabled (swapper_base.py SWAPPER_CONFIGS).
- monocular_regressors.py heads are untrained architectures unless weights provided [INFERENCE].
- Task statement said reconstruction=50 / rendering=18 files; inventory has 52 / 20 (the extra are `.npz` + `.npz.json` assets), all accounted for above.

Slice files in inventory: 172 | files accounted for: 172 | omitted: none
(inventory counts: reconstruction 52 + rendering 20 + engine 35 + identity 5 + faceset 8 + tracking 9 + swappers 4 = 133 in subdirs; top-level `app/roop_3d/*.py` 39; verified via `grep -cP "\tapp/roop_3d/(...)/"` and `grep -cP "\tapp/roop_3d/[^/]+\.py$"` against roop3d_files.tsv)


<!-- ===== R3_2 — pipeline, server, runtime slice ===== -->
## R3_2 — roop-3d slice 2: pipeline, server & runtime (code audit)

Scope: `app/roop_3d/{compositor,color,enhancers,pipeline,media,io,execution,core,async_pipeline,models,orchestrator,server,jobs,queue,reliability,telemetry,security,installer,bridges,editing,evaluation,tools,docs}`, top-level `app/roop_3d/*` not owned by slices 1/3 (pose_matrix, angle_tracker, relighting_3d, temporal*, faceset_* excluded as slice 1/3 by assignment), `app/roop/`, `app/core/`, `app/diagnostics/`, `app/tests/`, `app/api.py`, `app/serve.py`. Static reading only; no tests/builds run. Line counts via AST/`count('\n')`; symbol:line from `ast` top-level defs. `[INFERENCE]` = not verified by execution.

Status legend: real = functional code; stub = intentional placeholder; baseline = only a conventional/simple implementation; config/manifest/doc = non-code.

### 1. Executive map

- **Entry**: `app/serve.py:1-28` (UTF-8/unbuffered HUD env, `core/trt_bootstrap.py` DLL registration, `core/dll_injection.setup_environment`, then `roop_3d.server.__main__.main`) and `app/api.py:41-62` (`create_unified_app` -> `server/app.py:create_app`, module-level `app`).
- **Server**: FastAPI + uvicorn (`ws="websockets-sansio"`, `ws_max_size=16384`, `ws_max_queue=16`, `server/__main__.py:117`); 202 decorated REST endpoints in `server/*.py` (grep `@router.get|post|put|delete|patch`; workstation.py 30, routes.py 49, queue_routes.py 17, unified_state.py 13) + 5 WS routes (`server/app.py:322-326`, `server/routes.py:728-729`).
- **Render core**: `compositor/pipeline.py` (`FrameCompositor`, 1297 lines) is the real compositor; `pipeline/photorealistic_swap_core.py` + `pipeline/video_job_runner.py` is the Stage-17 FFmpeg-piped e2e runner; `optimized_processor.StreamingProcessor` is the 3-stage streaming coordinator; `job_queue.SegmentedRenderRunner` + `queue_manager.QueueManager` + `jobs/*` are the persistence/worker layer.
- **Compute**: `trt_session_builder.py` (strict lazy TRT, FP16/FP32/FP8-gated), `execution/trt_mixed_precision.py` (FP32-pinned islands, provider failover), `core/ort_session.py` (TRT-first pooled sessions + IO binding), `execution/zero_copy_pipeline.py`, `async_pipeline/*` (pinned pools, per-stage CUDA streams).

### 2. Per-module tables (every file in slice)

#### `app/roop_3d/compositor/` (23 files)

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

#### `app/roop_3d/color/` (7 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/color/__init__.py` | 7 | Pure colour science for Roop 3D: code points, transfer functions, gamut, YCbCr and HDR tone mapping. |  | real | app/roop_3d/color/__init__.py:1 |
| `app/roop_3d/color/codes.py` | 40 | ITU-T H.273 / FFmpeg colour code points, with names and explicit "unspecified" handling. | `is_unspecified`:27, `name_of`:35 | real | app/roop_3d/color/codes.py:27 |
| `app/roop_3d/color/description.py` | 308 | Colour descriptions of media: what the file says, what is assumed, and what that means. | `PixelFormatTraits`:25, `pixel_format_traits`:37, `HdrStaticMetadata`:68, `parse_mastering_display`:105, `parse_content_light`:130, `merge_hdr_metadata`:137 | real | app/roop_3d/color/description.py:25 |
| `app/roop_3d/color/gamut.py` | 117 | RGB primaries, gamut conversion matrices and YCbCr coefficients. | `_xy_to_xyz`:40, `rgb_to_xyz_matrix`:45, `conversion_matrix`:58, `luma_coefficients`:65, `matrix_weights`:70, `quantisation`:76 | real | app/roop_3d/color/gamut.py:40 |
| `app/roop_3d/color/hdr.py` | 128 | Deterministic HDR (PQ / HLG, wide gamut) -> SDR (BT.709, BT.1886 display) conversion. | `ToneMapParams`:34, `bt2390_eetf`:50, `HdrToSdr`:74 | real | app/roop_3d/color/hdr.py:34 |
| `app/roop_3d/color/ocio.py` | 113 | Optional OpenColorIO adapter. Explicit opt-in: nothing in the default render path imports this module. | `_import`:22, `is_available`:30, `capability`:34, `OcioTransform`:45, `cross_validate`:85 | optional opt-in adapter, not on default path (ocio.py:1) | app/roop_3d/color/ocio.py:22 |
| `app/roop_3d/color/transfer.py` | 161 | Transfer functions (EOTF / OETF) with their published constants. | `_f`:31, `pq_eotf`:35, `pq_inverse_eotf`:43, `hlg_oetf`:49, `hlg_inverse_oetf`:57, `hlg_system_gamma`:63 | real | app/roop_3d/color/transfer.py:31 |

#### `app/roop_3d/enhancers/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/enhancers/__init__.py` | 17 | Restorer and Enhancer implementations (UltraMax, GPEN Realistic, Dynamic CodeFormer, RestoreFormer++). |  | real | app/roop_3d/enhancers/__init__.py:1 |
| `app/roop_3d/enhancers/codeformer.py` | 107 | Dynamic CodeFormer & RestoreFormer++ with dynamic fidelity-weight binding. | `_blend`:16, `DynamicCodeFormerRestorer`:22, `RestoreFormerPlusPlusRestorer`:68 | real | app/roop_3d/enhancers/codeformer.py:16 |
| `app/roop_3d/enhancers/gpen_realistic.py` | 88 | GPEN Realistic: static batch-1 FP32-safe restoration with luma-only recolouring. | `luma_only_delta_recolour`:16, `GPENRealisticRestorer`:35 | real | app/roop_3d/enhancers/gpen_realistic.py:16 |
| `app/roop_3d/enhancers/styleganex_age.py` | 71 | StyleGANEX age inference; latent direction is not calibrated chronological age. | `AgeCalibration`:12, `StyleGANEXAge`:36 | real | app/roop_3d/enhancers/styleganex_age.py:12 |
| `app/roop_3d/enhancers/ultramax.py` | 161 | UltraMax Restorer: zero-copy CUDA IOBinding, explicit-only small-face bypass & 3-level Laplacian fusion. | `laplacian_pyramid_fusion`:15, `to_model_input`:61, `from_model_output`:67, `require_session`:73, `UltraMaxRestorer`:80 | real | app/roop_3d/enhancers/ultramax.py:15 |

#### `app/roop_3d/pipeline/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/pipeline/__init__.py` | 22 | Stage 17: photorealistic 3D swap core and the FFmpeg-piped end-to-end video job runner. | `__getattr__`:14 | real | app/roop_3d/pipeline/__init__.py:14 |
| `app/roop_3d/pipeline/image_ops.py` | 365 | Pure image / geometry primitives behind :mod:`roop_3d.pipeline.photorealistic_swap_core`. | `HeadFrame`:25, `head_frame`:35, `forehead_arc`:60, `head_polygon`:76, `polygon_mask`:82, `similarity_fit`:94 | real | app/roop_3d/pipeline/image_ops.py:25 |
| `app/roop_3d/pipeline/photorealistic_swap_core.py` | 619 | Photorealistic 3D swap core: craniofacial geometry + rectified neural swap + quotient-image relight + Laplacian composite. | `SwapCoreOptions`:49, `FacePacket`:76, `FaceReport`:94, `_Geometry`:111, `_Layer`:133, `HairDonorReplacer`:140 | real | app/roop_3d/pipeline/photorealistic_swap_core.py:49 |
| `app/roop_3d/pipeline/session_adapter.py` | 207 | Per-frame glue between a prepared ``ReferencePipeline`` session and :class:`Photorealistic3DSwapCore`. | `FrameReport`:35, `SourceViews`:44, `SessionFrameProcessor`:82 | real | app/roop_3d/pipeline/session_adapter.py:35 |
| `app/roop_3d/pipeline/video_job_runner.py` | 513 | End-to-end video job runner: FFmpeg decode pipe -> photorealistic swap per frame -> FFmpeg encode pipe -> audio remux. | `VideoInfo`:56, `_rational`:76, `probe_video`:84, `FFmpegDecoder`:123, `ColourManagedWriter`:188, `_remux`:212 | real | app/roop_3d/pipeline/video_job_runner.py:56 |

#### `app/roop_3d/media/` (25 files)

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

#### `app/roop_3d/io/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/io/__init__.py` | 20 | Stage 12 interchange: binary glTF serialization and the optional, GPL-isolated Blender bridge. | `__getattr__`:14 | real | app/roop_3d/io/__init__.py:14 |
| `app/roop_3d/io/blender_bridge.py` | 255 | Stage 12: optional, clean-room bridge to an external Blender 4.x. | `BlenderBridgeError`:40, `_is_loopback`:44, `BlenderMCPBridge`:64 | real | app/roop_3d/io/blender_bridge.py:40 |
| `app/roop_3d/io/glb_export.py` | 524 | Stage 12: a dependency-free binary glTF 2.0 (``.glb``) serializer for head meshes, textures, shape keys and head tracks. | `_fail`:52, `_png_chunk`:57, `encode_png`:61, `png_size`:86, `_PngCache`:94, `MorphTarget`:123 | real | app/roop_3d/io/glb_export.py:52 |
| `app/roop_3d/io/proxy_and_edl.py` | 336 | Local video proxies, filmstrips, CMX 3600 EDL interchange and render ETA. | `_rate`:18, `timecode_to_frame`:36, `frame_to_timecode`:62, `EDLEvent`:89, `EDLDocument`:102, `parse_edl`:111 | real | app/roop_3d/io/proxy_and_edl.py:18 |

#### `app/roop_3d/execution/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/execution/__init__.py` | 44 | GPU execution helpers for mixed precision inference and bounded frame processing. |  | real | app/roop_3d/execution/__init__.py:1 |
| `app/roop_3d/execution/trt_mixed_precision.py` | 804 | TensorRT mixed precision sessions with graph-level FP32 pinning and provider failover. | `OptimizationProfiles`:41, `_format_shapes`:59, `_family_shape`:64, `_input_specs`:93, `derive_optimization_profiles`:113, `_digest_file`:168 | real | app/roop_3d/execution/trt_mixed_precision.py:41 |
| `app/roop_3d/execution/worker_pools.py` | 353 | CPU decode/preprocess and asynchronous post-encode worker pools. | `_cpu_workers`:24, `DeviceFrameLease`:32, `_TransferSlot`:61, `DecodePreprocessPool`:66, `GopChunk`:201, `_concat_quote`:219 | real | app/roop_3d/execution/worker_pools.py:24 |
| `app/roop_3d/execution/zero_copy_pipeline.py` | 396 | Reusable device buffers and ORT I/O binding for frame-resident GPU stages. | `GPUExecutionFallback`:29, `OutputBufferSpec`:34, `_torch_module`:46, `_numpy_dtype`:53, `_output_spec`:68, `_tree_is_cuda`:76 | real | app/roop_3d/execution/zero_copy_pipeline.py:29 |

#### `app/roop_3d/core/` (17 files)

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

#### `app/roop_3d/async_pipeline/` (8 files)

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

#### `app/roop_3d/models/` (31 files)

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

#### `app/roop_3d/orchestrator/` (12 files)

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

#### `app/roop_3d/server/` (39 files)

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

#### `app/roop_3d/jobs/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/jobs/__init__.py` | 2 | Job execution package boundary. |  | real | app/roop_3d/jobs/__init__.py:1 |
| `app/roop_3d/jobs/project_store.py` | 346 | Versioned owned .r3dp manifests; raw media, embeddings and secrets stay outside. | `settings_digest`:21, `ViewportPreferences`:25, `ProjectPreview`:36, `ProjectRender`:43, `ProjectDraft`:50, `ProjectInput`:58 | real | app/roop_3d/jobs/project_store.py:21 |
| `app/roop_3d/jobs/schema.py` | 238 | Persistent job contracts and the complete legal transition graph. | `now`:23, `JobState`:27, `require_transition`:52, `MaskKeyframe`:57, `JobRequest`:73, `JobProgress`:155 | real | app/roop_3d/jobs/schema.py:23 |
| `app/roop_3d/jobs/store.py` | 262 | Transactional on-disk state, persistent idempotency and worker acknowledgments. | `IdempotencyEntry`:19, `JobDatabase`:24, `JobStore`:30 | real | app/roop_3d/jobs/store.py:19 |
| `app/roop_3d/jobs/worker.py` | 377 | Persistent threaded job worker with pause, cancellation and OOM retry. | `_hud`:32, `JobWorker`:40 | real | app/roop_3d/jobs/worker.py:32 |

#### `app/roop_3d/queue/` (3 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/queue/__init__.py` | 6 | Stage 23 queue package: the 4-strategy Batch Matrix engine and the 10-state job lifecycle. |  | real | app/roop_3d/queue/__init__.py:1 |
| `app/roop_3d/queue/batch_matrix.py` | 615 | Four-strategy combinatorial Batch Matrix engine for roop-3d (#/batch). | `BatchStrategy`:24, `sanitize_filename`:149, `format_output_name`:155, `RecipeConfig`:191, `TargetDescriptor`:199, `JobManifest`:208 | real | app/roop_3d/queue/batch_matrix.py:24 |
| `app/roop_3d/queue/lifecycle.py` | 71 | The 10 user-facing job lifecycle states (Stage 23) as a lossless projection of the persisted queue state. | `LifecycleState`:28, `lifecycle_of`:62 | real | app/roop_3d/queue/lifecycle.py:28 |

#### `app/roop_3d/reliability/` (13 files)

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

#### `app/roop_3d/telemetry/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/telemetry/__init__.py` | 5 | Terminal and diagnostic telemetry: the Pinokio HUD, the 14-section runtime report and the TensorRT build heartbeat. |  | real | app/roop_3d/telemetry/__init__.py:1 |
| `app/roop_3d/telemetry/pinokio_terminal.py` | 666 | Pinokio terminal HUD: UTF-8/unbuffered enforcement, one live in-place line, milestone logs and the boot/job report. | `enforce_terminal_environment`:28, `child_environment`:47, `_enable_windows_vt`:52, `defuse`:78, `paint`:84, `visible_len`:89 | real | app/roop_3d/telemetry/pinokio_terminal.py:28 |
| `app/roop_3d/telemetry/runtime_report.py` | 511 | The 14-section structured runtime report (terminal at boot / job start, ``GET /api/v1/system/runtime-report``, UI drawer). | `Section`:51, `show`:67, `tilde`:79, `mib`:88, `_Context`:92, `_available`:126 | real | app/roop_3d/telemetry/runtime_report.py:51 |
| `app/roop_3d/telemetry/trt_heartbeat.py` | 199 | Heartbeat while TensorRT builds an engine, so neither the terminal nor the React UI looks frozen. | `precision_label`:30, `format_shape`:38, `heartbeat_line`:54, `_default_vram`:61, `TerminalHandle`:71, `TrtCompilationHeartbeat`:75 | real | app/roop_3d/telemetry/trt_heartbeat.py:30 |
| `app/roop_3d/telemetry/trt_heartbeat_sidecar.py` | 92 | Out-of-process TensorRT build heartbeat: the only way to keep a terminal line alive during a cold engine build. | `_vram`:27, `run`:37, `main`:71 | real | app/roop_3d/telemetry/trt_heartbeat_sidecar.py:27 |

#### `app/roop_3d/security/` (6 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/security/__init__.py` | 49 | Roop 3D security, privacy, and synthetic media protection module. |  | real | app/roop_3d/security/__init__.py:1 |
| `app/roop_3d/security/artifact_cleaner.py` | 135 | Safe deletion and secure lifecycle management of temporary media and biometric artifacts. | `SafeArtifactCleaner`:15 | real | app/roop_3d/security/artifact_cleaner.py:15 |
| `app/roop_3d/security/content_validator.py` | 150 | Strict file type, content magic bytes, and path safety validation. | `validate_safe_filename`:30, `validate_safe_path`:40, `validate_image_header`:63, `validate_video_container`:94, `validate_faceset_archive_bytes`:144 | real | app/roop_3d/security/content_validator.py:30 |
| `app/roop_3d/security/log_sanitizer.py` | 82 | Log sanitization, secret redaction, and biometric privacy filters for application logging. | `sanitize_log_message`:9, `SecurityRedactionFilter`:53, `install_log_sanitizer`:68 | real | app/roop_3d/security/log_sanitizer.py:9 |
| `app/roop_3d/security/privacy.py` | 146 | Privacy enforcement, biometric permission validation, and network isolation guards. | `ConsentRecord`:16, `TargetFootagePermission`:26, `ConsentVerifier`:37 | real | app/roop_3d/security/privacy.py:16 |
| `app/roop_3d/security/synthetic_media.py` | 163 | Synthetic media disclosure, cryptographic provenance, identity honesty, and safety safeguards. | `SyntheticMediaDisclosure`:25, `AuditRecord`:36, `IdentityHonestyPolicy`:49, `SafetyGuard`:89, `build_provenance_manifest`:116 | real | app/roop_3d/security/synthetic_media.py:25 |

#### `app/roop_3d/installer/` (6 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/installer/__init__.py` | 11 | Roop 3D Offline Installer and Environment Manager package. |  | real | app/roop_3d/installer/__init__.py:1 |
| `app/roop_3d/installer/model_manager.py` | 269 | Offline model bundle importer and integrity verifier for Roop 3D. | `validate_archive_entry`:32, `check_archive_safety`:57, `ModelManager`:67, `format_missing_model_error`:242 | real | app/roop_3d/installer/model_manager.py:32 |
| `app/roop_3d/installer/offline_cache.py` | 159 | Offline dependency and health verification for Roop 3D. | `check_offline_dependency_cache`:39, `check_offline_readiness`:73 | real | app/roop_3d/installer/offline_cache.py:39 |
| `app/roop_3d/installer/preflight.py` | 542 | Environment preflight checker for Roop 3D. | `PreflightCheckResult`:39, `check_windows_version`:58, `check_python_version`:98, `check_node_version`:140, `check_cpu`:166, `check_system_ram`:205 | real | app/roop_3d/installer/preflight.py:39 |
| `app/roop_3d/installer/update_manager.py` | 234 | Update inspection, dependency change detection, configuration backup, and safe rollback. | `_git`:25, `check_update_status`:32, `create_configuration_backup`:123, `rollback_to_backup`:179 | real | app/roop_3d/installer/update_manager.py:25 |
| `app/roop_3d/installer/verification.py` | 178 | Installation verification and interrupted installation recovery for Roop 3D. | `verify_installation`:46, `recover_interrupted_installation`:122 | real | app/roop_3d/installer/verification.py:46 |

#### `app/roop_3d/bridges/` (4 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/bridges/__init__.py` | 2 | External integration package boundary. |  | real | app/roop_3d/bridges/__init__.py:1 |
| `app/roop_3d/bridges/cloud_stub.py` | 24 | Disabled remote adapter. Contains no URLs, credentials, HTTP client, or uploads. | `CloudStub`:6 | stub (intentionally disabled; no URLs/HTTP client) | app/roop_3d/bridges/cloud_stub.py:6 |
| `app/roop_3d/bridges/comfyui_bridge.py` | 442 | Bounded loopback-only ComfyUI API bridge; no model imports or media writes. | `LoopbackConnect`:33, `BridgeConfig`:40, `PromptHandle`:75, `OutputRef`:90, `BridgeEvent`:105, `decode_image`:114 | real | app/roop_3d/bridges/comfyui_bridge.py:33 |
| `app/roop_3d/bridges/provider_protocol.py` | 175 | Typed remote boundary with per-job signed consent and safe diagnostics. | `safe_filename`:21, `MediaDescriptor`:30, `RemoteJob`:38, `ProviderCapabilities`:60, `ProviderStatus`:67, `ProviderProtocol`:72 | real protocol; no concrete remote provider in slice (only CloudStub) | app/roop_3d/bridges/provider_protocol.py:21 |

#### `app/roop_3d/editing/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/editing/__init__.py` | 25 | Pre-render editing of the tracked target performance: emotion offsets and audio-locked lip/jaw motion. |  | real | app/roop_3d/editing/__init__.py:1 |
| `app/roop_3d/editing/expression_audio_director.py` | 385 | ExpressionAndAudioDirector: user emotion offsets in psi-space, and an audio lock for lip/jaw motion under occlusion. | `EmotionBasis`:48, `EmotionOffsets`:86, `DirectorSettings`:96, `AudioViseme`:113, `AudioVisemeTrack`:120, `_smooth`:201 | real | app/roop_3d/editing/expression_audio_director.py:48 |

#### `app/roop_3d/evaluation/` (8 files)

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

#### `app/roop_3d/tools/` (20 files)

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

#### `app/roop_3d/docs/` (24 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop_3d/docs/ASYNC_PIPELINE.md` | 154 | # Asynchronous multi-stream render pipeline and binary live transport (Defect #3) |  | doc | app/roop_3d/docs/ASYNC_PIPELINE.md:1 |
| `app/roop_3d/docs/FINAL_RELEASE_CERTIFICATION.md` | 333 | # Stage 20: Final Roop 3D Release Certification Audit Report |  | doc | app/roop_3d/docs/FINAL_RELEASE_CERTIFICATION.md:1 |
| `app/roop_3d/docs/FINAL_RELEASE_QUALIFICATION.md` | 258 | # Stage 10: Roop 3D Adversarial Validation and Release Audit |  | doc | app/roop_3d/docs/FINAL_RELEASE_QUALIFICATION.md:1 |
| `app/roop_3d/docs/HEAD_TRACKING_STAGE4.md` | 101 | # Head and gaze tracking (Stage 4, tracking series) |  | doc | app/roop_3d/docs/HEAD_TRACKING_STAGE4.md:1 |
| `app/roop_3d/docs/IDENTITY_ASSET_STAGE3.md` | 191 | # Stage 3: Personalized 3D Source Identity Reconstruction |  | doc | app/roop_3d/docs/IDENTITY_ASSET_STAGE3.md:1 |
| `app/roop_3d/docs/INFERENCE_PROFILING_STAGE2.md` | 123 | # Stage 2: Roop 3D Inference Profiling and Performance Engineering |  | doc | app/roop_3d/docs/INFERENCE_PROFILING_STAGE2.md:1 |
| `app/roop_3d/docs/OCCLUSION_STAGE5.md` | 227 | # Stage 5: Advanced Occlusion and High-Fidelity Compositor |  | doc | app/roop_3d/docs/OCCLUSION_STAGE5.md:1 |
| `app/roop_3d/docs/OFFLINE_3D_TARGET_FACE_PLAN.md` | 123 | # Offline 3D-Aware Target Face Replacement — Implementation Plan |  | doc | app/roop_3d/docs/OFFLINE_3D_TARGET_FACE_PLAN.md:1 |
| `app/roop_3d/docs/PHOTOREAL_VIDEO_STAGE17.md` | 119 | # Stage 17 - photorealistic 3D swap core and the FFmpeg end-to-end video job |  | doc | app/roop_3d/docs/PHOTOREAL_VIDEO_STAGE17.md:1 |
| `app/roop_3d/docs/REENACTMENT_STAGE4.md` | 148 | # Stage 4: Temporal 3D Face Reenactment Engine |  | doc | app/roop_3d/docs/REENACTMENT_STAGE4.md:1 |
| `app/roop_3d/docs/RELEASE_CHECKLIST.md` | 156 | ﻿# Roop 3D: Release Checklist & Operational Runbook |  | doc | app/roop_3d/docs/RELEASE_CHECKLIST.md:1 |
| `app/roop_3d/docs/RENDER_AND_PREVIEW_PERFORMANCE.md` | 96 | # Render and preview performance (Nov 12 2026) |  | doc | app/roop_3d/docs/RENDER_AND_PREVIEW_PERFORMANCE.md:1 |
| `app/roop_3d/docs/ROOP3D_MASTER_TECHNICAL_AUDIT.md` | 438 | # Roop 3D: Master Technical Re-Audit & Comprehensive Implementation Plan |  | doc | app/roop_3d/docs/ROOP3D_MASTER_TECHNICAL_AUDIT.md:1 |
| `app/roop_3d/docs/SEAM_EXPRESSION_STAGE6.md` | 134 | # Stage 6 - boundary seams and continuous expression direction |  | doc | app/roop_3d/docs/SEAM_EXPRESSION_STAGE6.md:1 |
| `app/roop_3d/docs/SILHOUETTE_REPLACEMENT_STAGE6B.md` | 61 | # Stage 6B - silhouette replacement (overhang inpaint, perimeter cage, swap scope) |  | doc | app/roop_3d/docs/SILHOUETTE_REPLACEMENT_STAGE6B.md:1 |
| `app/roop_3d/docs/STAGE06_MODEL_HUB.md` | 105 | # Stage 06 — Local model hub and engine architecture |  | doc | app/roop_3d/docs/STAGE06_MODEL_HUB.md:1 |
| `app/roop_3d/docs/STAGE07_WORKSTATION_UI.md` | 170 | # Stage 7: Professional workstation UI (Studio layout) |  | doc | app/roop_3d/docs/STAGE07_WORKSTATION_UI.md:1 |
| `app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md` | 257 | # Stage 8: Complete offline video workflow |  | doc | app/roop_3d/docs/STAGE08_OFFLINE_WORKFLOW.md:1 |
| `app/roop_3d/docs/STAGE09_OPTIMIZATION.md` | 121 | # Stage 9: Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization |  | doc | app/roop_3d/docs/STAGE09_OPTIMIZATION.md:1 |
| `app/roop_3d/docs/STAGE11_QUALITY_FRAMEWORK.md` | 140 | # Stage 11: Roop 3D AI Visual Quality Evaluation Framework |  | doc | app/roop_3d/docs/STAGE11_QUALITY_FRAMEWORK.md:1 |
| `app/roop_3d/docs/STAGE12_GLB_BLENDER_IO.md` | 68 | # Stage 12: binary glTF export and the optional Blender bridge |  | doc | app/roop_3d/docs/STAGE12_GLB_BLENDER_IO.md:1 |
| `app/roop_3d/docs/STAGE12_MODEL_GOVERNANCE.md` | 100 | # Stage 12 — Model governance and secure model management |  | doc | app/roop_3d/docs/STAGE12_MODEL_GOVERNANCE.md:1 |
| `app/roop_3d/docs/STAGE13_SECURITY_THREAT_MODEL.md` | 265 | # Stage 13 — Roop 3D Security, Privacy and Synthetic Media Threat Model |  | doc | app/roop_3d/docs/STAGE13_SECURITY_THREAT_MODEL.md:1 |
| `app/roop_3d/docs/UNIVERSAL_MODEL_DOWNLOADER.md` | 99 | # Universal model downloader (Stage 11, model-distribution series) |  | doc | app/roop_3d/docs/UNIVERSAL_MODEL_DOWNLOADER.md:1 |

#### `app/roop/` (5 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/roop/__init__.py` | 5 | Shim module forwarding to roop_3d. |  | shim | app/roop/__init__.py:1 |
| `app/roop/core/__init__.py` | 2 | Shared runtime primitives for the standalone Roop application. |  | shim | app/roop/core/__init__.py:1 |
| `app/roop/core/model_downloader.py` | 3 | Model downloader re-export for roop.core. |  | shim | app/roop/core/model_downloader.py:1 |
| `app/roop/core/ort_session.py` | 3 |  |  | shim | app/roop/core/ort_session.py:1 |
| `app/roop/relighting_3d.py` | 3 | Forwarding module for Stage 17 3D Relighting. |  | shim | app/roop/relighting_3d.py:1 |

#### `app/core/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/core/__init__.py` | 2 | Runtime bootstrap helpers for the standalone app. |  | real | app/core/__init__.py:1 |
| `app/core/trt_bootstrap.py` | 205 | Register packaged and system CUDA/TensorRT library directories before ORT loads. | `native_library_directories`:22, `bootstrap_native_library_paths`:130, `_preload_cuda_tensorrt_shared_libraries`:160, `expected_nvinfer_filename`:188 | real | app/core/trt_bootstrap.py:22 |

#### `app/diagnostics/` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/diagnostics/__init__.py` | 29 | Diagnostics and telemetry package for pipeline auditing. |  | real | app/diagnostics/__init__.py:1 |
| `app/diagnostics/pipeline_audit.py` | 384 | Unified Pipeline Diagnostic & Telemetry Framework (Non-breaking). | `AuditError`:22, `StageQualityDegradationError`:26, `ExecutionProviderFallbackError`:30, `EnhancerBypassError`:34, `AuditConfig`:39, `TensorShapeRecord`:57 | real | app/diagnostics/pipeline_audit.py:22 |

#### `app/tests/` (8 files)

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

#### `app/roop_3d/ (top-level)` (40 files)

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

#### `app/ (entrypoints)` (2 files)

| file | lines | purpose (module docstring / first line) | key symbols (name:line) | status | evidence |
|---|---|---|---|---|---|
| `app/api.py` | 62 | Unified FastAPI backend entrypoint for Roop 3D. | `create_unified_app`:41 | real | app/api.py:41 |
| `app/serve.py` | 28 | Standalone entrypoint; the import path contains only this app's source. |  | real | app/serve.py:1 |

#### 2.x Manifest / JSON summaries (not dumped)

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

### 3. Focus-area findings (with evidence)

#### 3.1 Blending
- Three real modes: `BlendMode.FEATHER / MULTI_BAND / POISSON` (`compositor/boundary_fusion.py:12-15`). MULTI_BAND = true Laplacian-pyramid blend via `cv2.pyrDown/pyrUp`, levels clamped to `floor(log2(min(h,w)))-1` (`boundary_fusion.py:50-74`); POISSON = `cv2.seamlessClone(NORMAL_CLONE)` on a `>0.05` binary mask with feather fallback on `cv2.error` (`:128-137`). Default is FEATHER, `pyramid_levels=4` (`compositor/pipeline.py:107,131`).
- Linear-light blending optional (`linear_light=False` default, `blend_transfer` in `color/transfer.py:28` = srgb/bt1886/gamma22/gamma28/bt709_oetf/linear); boundary_fusion converts via LUT before `_pyramid_blend` (`boundary_fusion.py:125`).
- Stage-17 core uses its own `ops.laplacian_pyramid_blend(photo, plate, alpha, levels=4, limit_overshoot=True)` (`pipeline/photorealistic_swap_core.py:236`, `pipeline/image_ops.py:250`), feather = 0.02 x face width (`:64`).
- Frequency separation: UltraMax restorer fuses via 3-level Laplacian pyramid: finest `detail_levels` bands 100% restorer, mid bands 50/50 swap/restorer, base from swap (`enhancers/ultramax.py:15-58`). `compositor/quality.py` has `transfer_fine_texture`, `match_sharpness`, `local_illumination_match`; `skin_texture_retention=0.25` re-injects target pore residual after UltraMax (`pipeline.py:128`).
- Colour: LAB mean/std + YCbCr + masked histogram + OKLab/OKLab-Wasserstein (`compositor/color_match.py:53-198`; `color_method` whitelist `pipeline.py:~205`), default `"lab"`, `color_strength=0.65`. SH-based relighting `compositor/lighting.py:17`, contact shadow (0..0.15) / SSS warmth options (`pipeline.py:144-149`). Sensor-physics finish: target-referred blur, defocus, shutter 180 deg, grain 100 (`sensor_physics.py:264`, `pipeline.py:~137`), CUDA path when torch.cuda available else CPU fallback (`:276-282`).

#### 3.2 Masks
- **BiSeNet 19-class** parser: include classes (1,2,3,4,5,10,11,12,13), strict-forbidden (6,7,8,9,14,15,16,17,18) (`compositor/masks.py:25-26`); dual-BiSeNet forbidden mask (`masks.py:55`); jaw/neck clamp `apply_submental_jaw_cutoff` (`:204`); landmark-hull deterministic fallback (header `:1`). Class labels were *not* recorded in registry -> mapping is "conventional candidate" (`masks.py:3-6`) [defect-ish].
- **XSeg / face_occluder** 256x256 NHWC visible-face nets: `SUPPORTED_NETS = (xseg_3.onnx, xseg.onnx, face_occluder.onnx)` (`compositor/occlusion_model.py:20-21`); default `xseg_3.onnx`, missing model -> reported capability state, "nothing is simulated" (`:1-12`). (`xseg_3.onnx` is not in `model_manifest.json`/`stage16_models.json`, which carry `xseg_1/xseg_2` + `face_occluder.onnx` — [INFERENCE] resolved via `ModelCatalog`; verify.)
- Matte refinement: guided filter / fast guided filter (`matte.py:25,41`), edge decontamination (`:125`), 12px boundary-ribbon trimap matting (`:188-238`, opt-in `boundary_ribbon_matting=False`), hairline offset (`hairline.py:27`), directional strand matte with 12-orientation Gabor bank (`strand_matte.py:65`, opt-in `strand_matte=False`, `pipeline.py:83-86`).
- `masking_engine.py:47` `MaskingEngine`: "19-Class + DFL XSeg + text-prompted CLIPSeg" (header :1); 8 `baseline` mentions — [INFERENCE] part of this is optional/lazy.
- Mask seam watchdog `MaskSeamWatchdog` (`stream_restoration.py:97`) with registered-correlation scene reset.

#### 3.3 Optical-flow mask stabilisation / temporal (axis 4, slice-2 part)
- Occluder matte: `OcclusionTracker` warps previous matte with dense **DIS** flow (`cv2.DISOpticalFlow_create(PRESET_MEDIUM)`, `compositor/occlusion_tracker.py:52`), history 0.7, `max_hold=4`, `hold_decay=0.8`, `flow_side=96`, `cut_reliability=0.18`, `cut_correlation=0.6`, `max_tracks=16` (`:44-45`); history only affects uncertain pixels (~0.5), confident detections pass instantly (docstring `:1-12`); `occlusion_release=0.75` (`pipeline.py:81`).
- Restorer-detail residual: `TemporalDetailStabilizer` flow-warps enhancement residual; NVIDIA OF 2.0 (CUDA) when available, DIS CPU fallback; `temporal_alpha=.25`, `flow_side=96`, `max_side=512`, `cut_correlation=.35`, `min_flow_confidence=.12`, 32 MiB cache (`detail_temporal.py:40-45`). Warps use `F.grid_sample(bilinear, align_corners=True)` (`:150-189`).
- Geometry/colour smoothing: `TemporalSmoother` One-Euro for affine, mask, colour: `min_cutoff=1.0`, `beta=0.035` (`compositor/temporal.py:75`; filter default beta 0.025 `:23`), `max_mask_side=512`, 32 MiB cap.
- Flow trackers: `optical_flow_tracker.py` `RAFTSmallFlow` (injected runner, `:74`), `NvidiaHardwareFlow` (`:97`), `select_flow` (`:143`).
- Scene-cut: compositor consumes a `scene_cut` flag everywhere (`pipeline.py:344,645,701,813,932`) and resets histories; the *producer* is `engine/detection/face_tracker.py:156 mark_scene_cut` (slice 1) and `tools/occlusion_acceptance.py:find_scene_cuts` (offline harness only). No detector inside slice-2 runtime path.

#### 3.4 Restorers
- `compositor/stream_restoration.py:24-52`: `RestorerSpec` with explicit 5-pt template, `size in (128,256,512,1024)`, normalization `[-1,1]`/`[0,1]`; specs: GPEN `gpen_bfr_512.onnx`, CodeFormer `CodeFormerv0.1.onnx`, GFPGAN `GFPGANv1.4.onnx`, UltraMax = `codeformer.fp16.onnx` FFHQ-512 RGB[-1,1] (`:50`; overridable by `roop_3d/.runtime/restorers/ultramax.json`, evidence-required `:55-67`). `restore_in_crop_space` (`:70`), `OnnxFaceRestorer` (`:337`), `DecoupledRestorer` (`:429`).
- `enhancers/`: `DynamicCodeFormerRestorer` (fidelity `w` bound per frame, 512 px, CUBIC resize, `cv2.detailEnhance` fallback `codeformer.py:44-63`), `RestoreFormerPlusPlusRestorer`, `GPENRealisticRestorer` (static batch-1 FP32, luma-only recolour, `gpen_realistic.py:16`), `UltraMaxRestorer` (CUDA IOBinding, explicit-only small-face bypass `ultramax_small_face_mode="off"`, `ultramax_min_native_pixels=320`), `StyleGANEXAge` (latent direction, explicitly *not* calibrated age, `styleganex_age.py:1`).
- Legacy ROI restorers `compositor/restoration.py` (`GpenRestorer:44, CodeFormerRestorer:127, UltraMaxPipelineRestorer:185, GfpganRestorer:247`) blended only through face alpha (`blend_restored_roi:10`). Resolution policy: `roi_resolution_limit` 0 or 128..2048 (`pipeline.py:181-183`); restorer up-resizes <512 with CUBIC else AREA (`restoration.py:99`).
- `RealESRGAN`/`4x_ultrasharp`/`swin2_sr` appear only in manifests (`supported_models.json`, `stage16_models.json`); no slice-2 code wires a frame-level upscaler other than restorers [INFERENCE].

#### 3.5 Sub-pixel warping
- CPU: `cv2.warpAffine(..., INTER_LINEAR, BORDER_CONSTANT)` for crop and alpha (`boundary_fusion.py:34-37`). No Lanczos/INTER_CUBIC warp, no half-pixel-centre handling beyond cv2 default [baseline]. Cubic is used only for resizes (`pipeline.py:436,509,531`; `restoration.py:87`).
- GPU: `gpu_compositor.warp` -> `gpu_transforms.warp_affine_gpu` (`gpu_compositor.py:19-22`, `gpu_transforms.py:11`); closed-form affine inverse on-stream (`gpu_compositor.py:25-33`). `hires_handoff.py` rescales matrices between low-res swap and hi-res frame (`:47-104`). `vectorized_pipeline.affine_graph` expresses warp as ONNX GridSample graph (`:109`); `models/grid_sample.py:18` lowers 3-D trilinear GridSample to two 4-D ones.
- Boundary support dilated; "pyramid reconstruction and Poisson gradients can spread beyond nonzero alpha" -> write footprint clamped (`boundary_fusion.py:140-141`).

#### 3.6 CLI / headless / JSON logs / TUI / diagnostics (axis 5)
- **No Rich/TUI**: no `rich` import anywhere in slice (grep). Terminal UX = custom ANSI HUD `telemetry/pinokio_terminal.py` (CLEAR_LINE `\r\x1b[K` `:68`; in-place live line + milestone logs; `enforce_terminal_environment:28`; `NO_COLOR`/`FORCE_COLOR` honoured `:377`; `ROOP3D_HUD=0` silences `:391`; `ROOP3D_TRT_SIDECAR` `:594`).
- **CLI flags**: only one real flag, `--server_share/--server-share` (`server/__main__.py:67`) + env `ROOP3D_SERVER_SHARE`. Everything else is env (`ROOP3D_PORT`, `ROOP3D_HOST`, `ROOP3D_ASYNC_PIPELINE`, `ROOP3D_PIPELINE_QUEUE` 1..64 default 8 `async_pipeline/config.py:22-26`, `ROOP3D_PINNED_FRAMES`, `ROOP3D_CUDA_GRAPH`, `ROOP_TRT_FP16`, `ROOP_TRT_FP8`, `ROOP_SWAP_FP16`, `ROOP_STRICT_NVDEC`, `ROOP_OPT_STRICT_TRT`, `ROOP_AUDIT_DEBUG`, `ROOP_GPU_COMPOSITOR`, `ROOP3D_ORT_LOG_LEVEL`, `ROOP3D_PREVIEW_SESSION_POOL`, ...). No headless one-shot render CLI: `tools/*` are maintenance/benchmark/gates (e.g. `python -m roop_3d.tools.benchmark_async_pipeline`, `reliability/soak.py` has its own CLI) — a render is only reachable via REST `/api/v1/queue` or `pipeline.video_job_runner.EndToEndVideoJobRunner` as library.
- **JSON logs**: `JsonFormatter` (`server/__main__.py:73-82`) emits `{"level","logger","message"}` or redacted structured dicts; startup events `{"event":"ready"}` and `{"event":"share"}` (`:85-100`); errors printed as `{"error":{"code":...}}` (`:112`). `core/redaction.py:8`, `security/log_sanitizer.py:53` redact.
- **Diagnostics reports** (two coexisting 14-section reports, different content):
  1. `telemetry/runtime_report.py:29-44` (GET `/api/v1/system/runtime-report`, `server/runtime_report_routes.py:14`; also printed at boot/job start by HUD): Host/OS/Venv, CPU cores & worker pools, GPU arch & compute capability, EP chain & mixed precision, TRT engine cache, zero-copy OrtValue ring, VRAM budget & OOM thresholds, model governance & SHA-256, 3D head topology, full-craniofacial swap lock, physical camera & SH relighting, GOP slicer & audio remux, offline isolation audit, loopback server & deep links. Per-section `OK/INFO/WARN/UNAVAILABLE/UNKNOWN`; a failing collector degrades to UNKNOWN (`:461-467`).
  2. `server/diagnostics.py:937-1184` `generate_14_section_report` (GET `/api/v1/system/diagnostics-report`, `routes.py:699-700`): Python & Env, Host HW, ORT providers, TRT cache & quantisation, FFmpeg media & codec matrix, model registry & SHA, detection & ArcFace alignment, 3D mesh & LivePortrait, Identity transfer & inswapper-128 graph, Compositor/XSeg/alpha seams, Restoration adapters (GPEN-512/CodeFormer), VRAM governor, Temporal landmark smoothing, Strict offline verification.
  3. `reliability/report.py:222-275 build_report/save_report` (redacted local diagnostic; `/api/v1/...reliability`), `app/diagnostics/pipeline_audit.py` (tensor-shape tracker, EP auditor, `CopyCounter`; errors `StageQualityDegradationError`, `ExecutionProviderFallbackError`, `EnhancerBypassError`; `ROOP_AUDIT_DEBUG` fail-loud).
  4. Pre-flight linter `POST /system/preflight-lint` (`server/diagnostics.py:621 run_preflight_lint`, checks: yaw coverage vs .fsz `:455`, TRT engines `:516`, UltraMax 4K `:542`, output size `:563`, disk `:586`); synthetic benchmark `POST /system/benchmark` (`:868`).
- TRT build heartbeat: 1 Hz `trt_compilation_progress` WS frame + out-of-process sidecar (`telemetry/trt_heartbeat.py:75`, `trt_heartbeat_sidecar.py:37`).

#### 3.7 Compute: TRT / FP16 / INT8 / IO binding / pinned / VRAM (axis 6)
- Precision policy `trt_session_builder.precision_for` (`:41-52`): FP32 for `inswapper_128`, `motion_extractor`, `gpen*`; `ROOP_TRT_FP16=0` -> FP32; `ROOP_TRT_FP8=1` + heavy restorers -> FP8 else FP16. **INT8 explicitly off**: `"trt_int8_enable": False` (`:236`). FP8 requires separate calibrated E4M3FN Q/DQ ONNX + reference parity callback (`:170-190,329-330`; `trt_fp8_builder.py:21-44`).
- ORT TRT options (`trt_session_builder.py:229-246`): engine cache + prefix, timing cache, `builder_optimization_level=5`, `auxiliary_streams=2`, `cuda_graph` only for static inputs and auto-disabled on partition error (`:285-292`), `layer_norm_fp32_fallback=True`, workspace <=1024 MiB (6gb) / 2048 (12gb) (`:111-114`), VRAM budget minus 768 MiB reserve (`:89-100`), batch profile (1,4,8) opt 4 (`:55`). Note `core/ort_session.py:38` uses `builder_optimization_level "3"` — two TRT configs with different levels.
- Mixed precision: graph-level FP32 pinning of `RodriguesRotation, SphericalHarmonics, LayerNormalization, InstanceNormalization, Pow, ReduceMean, Softmax`; FP16 on Conv/ConvTranspose/Gemm/MatMul; policy `fp16-fp32pinned-v1`; failover TRT->CUDA->CPU replaying inputs (`execution/trt_mixed_precision.py:25-35`, `:552-711`).
- IO binding: `core/ort_session.py:679-743` (`GpuIoBinding`: bind_input by device ptr, `bind_ortvalue_*`, preallocated outputs); `execution/zero_copy_pipeline.py` `ZeroCopyGPUExecutionRing` (reusable device buffers); `tools/gpu_saturation.py` measures FP16+IOBinding+micro-batch; `app/tests/test_stage27_certification.py:110` asserts >85% GPU use with micro-batched IOBinding.
- Pinned memory: `async_pipeline/buffers.py` `PinnedFramePool` (`torch.empty(pin_memory=True)`, `:130-156`), `CudaFrameRing:285`, bounded queue default 8 (`:25`, ~200 MB of 4K BGR); `AsyncPipeline` one thread+CUDA stream per stage, event hand-off, no device sync (`stages.py:1-20`); `hardware_streamer.SharedMemoryFrameRing:35` for process ring.
- VRAM governor (code lives in `engine/vram_governor.py`, slice 1): consumed here by `core/ort_session.py:99-169` (pool VRAM measured via NVML before/after each pooled session, `reclaimable_vram_mib`), `telemetry/runtime_report.py:299-321` (per-stream admission reservation, watchdog pressure/near-OOM thresholds), `jobs/worker.py:340` `RecoverablePipelineOOM` -> checkpoint then one CPU-profile retry (report text `:310`), `server/diagnostics.py:119 purge_idle_vram`, `compositor roi_resolution_limit` pressure cap.
- Hardware probes `core/hardware.py` (`effective_vram_budget:112`, `tensor_core_precisions:136`, `tensorrt_version:159`); `core/transfer_ledger.py` counts host<->device copies at the ORT boundary.

#### 3.8 Job queue, checkpoint, GOP slicing, NVENC, remux, WS (axis 7)
- **Real job-state enums** (three coexisting vocabularies — see defects):
  - `jobs/schema.py:27` `JobState`: queued, preparing, running, pausing, paused, cancelling, cancelled, succeeded, failed, interrupted (10) with explicit `TRANSITIONS` graph `:39-48` (terminal: cancelled, succeeded, failed) enforced by `require_transition:51`.
  - `queue_manager.py:47` `CanonicalJobState`: QUEUED, PREFLIGHT, ANALYZING, PRESCANNING, PREPASSING, RENDERING, MUXING, PAUSED, COMPLETED, FAILED, RETRYING, CANCELLED (12) + `CANONICAL_TRANSITIONS` (FAILED->QUEUED/RETRYING; COMPLETED/CANCELLED terminal).
  - `queue/lifecycle.py:28` `LifecycleState` (10 UI states): queued, preparing, compiling_trt, tracking_3d, rendering, paused, muxing_audio, completed, failed, cancelled; pure projection `lifecycle_of` (`:62`), `compiling_trt` inferred from stage-text markers (`:48`).
- **Persistence/atomicity**: `core/storage.py:27-41` `atomic_bytes/atomic_json` = temp file + `fsync` + `os.replace` + dir fsync, `exclusive` cross-process lock. `jobs/store.py` transactional on-disk state with idempotency + worker acks; `queue_manager.py` "ACID-safe SQLite/JSON" (header). `stream_checkpoint.py:72-75` saves `job_checkpoint.json` via `atomic_json`; segments recorded with sha256 + frame count (`:60-65`) and resume re-verifies `sha256 == _sha256(segment) and count_video_packets == expected_frames` (`:47`). `job_queue.py:364,395` writes `.part` then `os.replace` -> published only when complete. `JobCheckpoint` schema v1: segments list with PENDING|COMPLETED|FAILED, inputs fingerprint, frame_progress, execution_provider, encoder (`job_queue.py:47-85`). `video_writer.py:594` `os.replace(...)` atomic publish; `reliability/recovery.py` `ReliabilityAuditor` + `reconcile_on_startup:490` repairs jobs/checkpoints/outputs.
- **GOP-aligned slicing**: `segment_dispatcher.py` detects keyframes via PyAV (`:70-109`, 50 ms dedupe) with ffprobe fallback, `compute_gop_aligned_cuts` (`:139`, default `max_segment_duration=15.0` `:376`; min 0.5 s from previous cut), `-c copy` split (`:199-217`), `concat_lossless` demuxer + original audio remux with AAC fallback if copy fails (`:262-330`). Exposed `POST /segments/split|concat` (`queue_routes.py:202,214`). Parallel GOP-chunk path: `execution/worker_pools.py` `GopChunk:201`, `stitch_gop_chunks:226`, `encode_nvenc_chunk:259`, `PostEncodeWorkerPool:283`.
- **NVENC**: `media/nv_writer.py:25-26,100-103` h264_nvenc/hevc_nvenc `-preset p5 -rc vbr -cq <q> -b:v 0`, auto-fallback libx264/libx265 (`-preset medium -crf`), frame replay (`_REPLAY_FRAMES`, `:168,194`), `probe_nvenc:31`; `media/video_writer.py:68-91` hardware=auto picks NVENC for 8-bit 4:2:0 only (`:277`); `execution/worker_pools.py:271-272` lossless NVENC `-tune lossless -rc constqp -qp 0` chunks; `hardware_streamer.py:245` NVDEC `-hwaccel cuda -hwaccel_output_format cuda`, `:366-386` NVHardwareVideoWriter. No AV1 NVENC. Export profiles: h264/hevc 8-bit, ProRes 422 HQ 10-bit (`prores_ks` profile 3), FFV1 mkv, image sequences (`export_profiles.py:60-92`).
- **Audio/subtitle remux**: `media/audio.py:19-22,75` per-container compatible-codec sets, copy if codec+packet boundaries align else AAC/PCM; all tracks; `media/subtitles.py:23-25` packet copy (mp4/mov: mov_text; mkv: subrip/ass/ssa/webvtt/dvd/pgs/dvb) else explicit recorded drop; `media/av_sync.py:119 measure_lag` verifies sync. BUT the Stage-17 runner drops subtitles (`-sn`), metadata (`-map_metadata -1`), refuses HDR (`HDR_INPUT_UNSUPPORTED`) and remuxes only `1:a:0` (`pipeline/video_job_runner.py:12-14,128,135,217-224`) — narrower than `media/video_writer`.
- **WebSocket routes**: `/ws/telemetry`, `/ws/events`, `/ws/queue` -> `event_stream`; `/ws/v1/pipeline/status` -> `pipeline_status_stream` (carries `model_download_progress` and `trt_compilation_progress` frames too); `/ws/v1/preview/stream` -> `preview_stream` (binary: 16-byte header + JPEG, JSON control) (`server/app.py:322-326`, `live_stream.py:3`). Event types (`server/ws.py`): JobState, JobProgress, Telemetry, PipelineStatus, TrackingTelemetry, StreamGap, StreamReady, Queue, ModelDownloadProgress, TrtCompilationProgress. `IntakeMiddleware` (`middleware.py:39`) enforces bounded body, explicit origins (`127.0.0.1:5174`), repeated-header rejection, `ROOP3D_MAX_BODY_BYTES`. LAN share: bearer token per launch (`security.py:20`).
- **Queue REST**: `/queue` list/submit/get, `/{id}/cancel|pause|resume|progress|duplicate|retry|join`, `/queue/reorder`, `/batch/recipes|generate|submit`, `/queue-lifecycle`, `/segments/split|concat` (`queue_routes.py:81-214`). Batch Matrix: 4 strategies (`queue/batch_matrix.py:24 BatchStrategy`, `BatchMatrixEngine:262`).

#### 3.9 Workstation UI (axis 8; server side only, UI is another slice)
- Same-origin SPA fallback `server/api.py:35-54 mount_workstation`; built-bundle gate `tools/verify_e2e_workstation.py:68`; workstation REST `server/workstation.py` (30 routes, chunked upload `:46-91`), preview `workstation_preview.py` (on-demand bounded previews, `PreviewService:162`), live preview cache/mutex `live_preview.py:219-306`, timeline scrub `media_frames.py`, Face bank / target lock / identity assets / blender / studio environments, installer routes (preflight, offline models, update, rollback `installer_routes.py:36-65`), unified state (`unified_state.py:14 RenderMode`).

### Axis findings (slice)

1. **Detection/landmarks/angle**: slice-2 hosts `media/detector.py:18` (SCRFD, RetinaFace-10G, YOLOFace selectable decoders), `media/face_analysis.py:51 five_point_plausible`, `:70 agrees_with_106` (5-pt vs 106-pt cross-check), `core/pose.py:173 estimate_pose_5pt`; pre-flight yaw coverage vs .fsz (`server/diagnostics.py:455`); orchestrator `scene_rules.json large_yaw_degrees=30`, `low_confidence: withhold`. Failure thresholds: `ROOP3D_AUTO_3D_YAW/PITCH`, `ROOP3D_3D_MAX_ROLL` envs (grep, values live in slice 1).
2. **2D vs 3D/.fsz**: `server/faceset_3d.py:127 reconstruct_3d`, `identity_assets.py` atlas bake, `pipeline/photorealistic_swap_core.py:188` craniofacial geometry + rectified swap + quotient-image relight; `swap_scope` full_craniofacial/adaptive_skull/inner_face (`compositor/pipeline.py:~165`); `io/glb_export.py` binary glTF, `mesh_raycaster.py` self-occlusion; `models/grid_sample.py` volumetric sampling.
3. **Visual quality**: Laplacian/Poisson/feather, BiSeNet+XSeg, guided-filter matting, LAB/OKLab colour, SH relight, sensor physics, GPEN/CodeFormer/GFPGAN/UltraMax restorers (sections 3.1-3.4); `evaluation/metrics.py` PSNR/SSIM/perceptual/cosine, regression gate, blind A/B (`regression.py:64`, `blind_review.py:83`). Colour-science package `color/` (H.273 codes, PQ/HLG, BT.2390 EETF, HdrToSdr `hdr.py:74`) — only used for output tagging/media; HDR still rejected in Stage-17 runner.
4. **Temporal**: One-Euro (affine/mask/colour), DIS-flow occluder tracker, flow-warped detail residual (3.3); scene_cut flag plumbed but detector external.
5. **CLI/telemetry**: custom ANSI HUD, JSON logs, two 14-section reports + reliability report + pre-flight linter (3.6); no Rich, no `--json`/`--verbose`/`--headless` flags.
6. **Compute**: FP16 default, FP32 for swapper/gpen/motion-extractor, FP8 gated, INT8 off, IO binding, pinned pools, CUDA graphs for static shapes, VRAM admission with OOM->CPU retry (3.7).
7. **Queue/IO**: 10-state persisted `JobState`, 12-state canonical queue states, atomic checkpoint, GOP stream-copy slicing, NVENC w/ CPU fallback, all-track audio/subtitle remux in media layer (3.8).
8. **Workstation UI (server)**: 202 REST routes + 5 WS, SPA mount, uploads, previews, live JPEG stream (3.9).

### Gaps / stubs / defects found

1. **State vocabulary drift** — three enums: `JobState` (10, `jobs/schema.py:27`), `CanonicalJobState` (12 members, `queue_manager.py:47`) whose `normalize_canonical_state` docstring still says "one of the 10 canonical states" (`:139-140`), and 10-state `LifecycleState` (`queue/lifecycle.py:28`). `jobs/*` (JobWorker) and `queue_manager`/`job_queue` are parallel stacks; `server/app.py:86-100` instantiates BOTH `JobStore`+`JobWorker` (jobs/*) and `QueueManager` (queue_manager.py), so two state machines run side by side.
2. **No Rich TUI, no headless render CLI, no `--json`/`--log-level` flags**: only `--server_share` (`server/__main__.py:67`). Render only via REST/library.
3. **INT8 absent** (`trt_session_builder.py:236 trt_int8_enable False`); FP8 needs a separately calibrated artifact that is not shipped (no FP8 entries in manifests).
4. **Inconsistent TRT config**: `core/ort_session.py:38` optimization level 3 vs `trt_session_builder.py:242` level 5; two builders (`trt_builder.py:133 TensorRTBuilder`, `trt_session_builder.py:103`, `optimized_trt_engine.py`, `liveportrait_trt.py`) coexist.
5. **Sub-pixel warping is baseline**: bilinear `cv2.warpAffine` only (`boundary_fusion.py:34`); no higher-order resampling or sub-pixel landmark refinement in warp path.
6. **Stage-17 e2e runner is narrower than the media layer**: drops subtitles/metadata, refuses HDR, first audio track only (`video_job_runner.py:12-14,217`). Four writer stacks: `media/video_writer.py`, `media/nv_writer.py` (class `NVHardwareVideoWriter:47`), `hardware_streamer.py` (another `NVHardwareVideoWriter:359`), `video_job_runner.ColourManagedWriter:188` — duplicate class name, divergent behaviour.
7. **BiSeNet class mapping unverified** ("conventional 19-class mapping as a candidate", `masks.py:3-6`); default XSeg name `xseg_3.onnx` has no manifest entry (manifests list xseg_1/xseg_2 + face_occluder).
8. **Scene-cut detection not in the render path of slice 2**: compositor needs an externally supplied flag; `find_scene_cuts` only in `tools/occlusion_acceptance.py:89`.
9. **GOP slicing is stream-copy, not frame-accurate**: cuts clamp to keyframes (>=0.5 s apart, default 15 s segments); chunk encode path (`worker_pools.encode_nvenc_chunk`) is lossless-NVENC (`-qp 0`) intermediates [INFERENCE: large temp files].
10. **Intentional stub**: `bridges/cloud_stub.py` (disabled remote adapter); `bridges/provider_protocol.py` has no concrete provider. `reliability/harness.py` uses stub inference stage (test only).
11. **Hardware-gated / unverified here**: `openvino_coprocessor.py`, `heterogeneous_scheduler.py`, `cvcuda_ops.py`, `mesh_raycaster.py` (Embree), NVIDIA OF 2.0 paths in `detail_temporal.py`/`optical_flow_tracker.py` — fall back to CPU/DIS; none executed in this audit.
12. **`app/tests/` is thin**: 7 py files / 720 lines; three (`test_3d_relighting.py`, `test_heterogeneous_scheduler.py`, `test_nv_cvcuda_pipeline.py`) are 8-line forwarders to the real suite in `app/roop_3d/tests` (slice 3). `test_flame_and_liveportrait.py` and `test_stage27_certification.py` FAIL (not skip) when hardware/assets are missing (header `:1`).
13. **Shim**: `app/roop/*` only re-exports `roop_3d` (`roop/__init__.py`, `core/ort_session.py`); no independent code.

Slice files in inventory: 335 | files accounted for: 335 | omitted: none


<!-- ===== R3_3 — UI, tests, audit tooling, docs, root slice ===== -->
## R3 slice 3 — roop-3d: UI, tests, audit scripts, docs, root files

Slice: `app/roop_3d_ui/` (359 tracked files; node_modules/dist/test-results are not tracked), `app/roop_3d/tests/` (202), `app/roop_3d/audit/` (49), repo-root `docs/` (98), root-level and other tracked files (46). Total **754** tracked files. All paths relative to `G:/pinokio/api/roop-3d`. Evidence is `file:line` read from the working tree on 2026-11-13; counts of tests are static (`def test_` / `it(`), nothing was executed. Anything not read from code/docs is tagged [INFERENCE].

### 1. Headline findings (slice)

1. **`verify_isolation.py [--report]` verifies nothing.** It delegates to `stage00.main()`, which hard-codes `historical = {"ok": True, "checked": 2645 ...}` and `comprehensive = {... "checked": 2645}` (`app/roop_3d/audit/stage00.py:35-36`); `isolation.verify()` returns `ok: True, violations: []` unconditionally (`isolation.py:184-197`). `--legacy-only` and `--snapshot` are parsed and ignored (`stage00.py:26-27`); `verify_policy`/`immutable_manifest` are imported (`stage00.py:16`) but never called. Exit is always 0 unless an OS error occurs. `--report` only writes `roop_3d/audit/stage00_verification.json` (`stage00.py:47-48`).
2. **`local_models.py --verify` is real** (hash + size per registered model, FLAME-2023-Open pin, CC0 pack rebuild/digest, universal-manifest corruption check) — see §4.
3. **The UI has no temporal-stability graph, no per-frame jitter plot and no yaw-failure/limit read-out.** Temporal smoothing is one slider mapped to `temporal_min_cutoff` (`WorkstationControls3D.tsx:33-46,186-201`); "speech-lip lock" and "scene adaptation" toggles are localStorage-only preferences with no engine consumer (`WorkstationControls3D.tsx:18-19,240`), although the backend has a 16-frame/3 s scene-adaptation pass (`tests/test_head_tracking_stream.py` header). Scene cuts are shown only as markers on `TrackLanes` (`TrackLanes.tsx:78-80`).
4. **Extreme-yaw real-media tests are skipped by design:** `test_extreme_pose_3d_swap.py::test_real_target_frames` needs `tests/fixtures/extreme_pose_real/*.fsz`, which are absent from the tracked tree (no non-`.py` file under `app/roop_3d/tests/`; `docs/EXTREME_POSE_3D.md:108-113` says 4 tests skip; "no real-profile pass" claimed). Synthetic yaw ±85° is covered (`test_extreme_pose_3d_swap.py:73,94,119,268`). No tracked `conftest.py` in `tests/`.
5. **Measured throughput is far below real-time and still labelled releasable:** `audit/STAGE15_BENCHMARKS.json` (RTX 4070, CUDA EP, 512×512, 15 frames) → `completed_output_fps = 0.26`, mean detect_track 391.9 ms, retarget 501.6 ms, composite 522.4 ms, peak VRAM 8645 MiB; `STAGE15_VALIDATION.json` still says `"READY FOR RELEASE"`. `STAGE09_BENCHMARKS.json`: 0.57 fps, warm frame p50 1056 ms.

### 2. UI (`app/roop_3d_ui`) — what exists, what it consumes

Stack: React 19.3.1, three 0.187 + @react-three/fiber 9.8 / drei 10.7, framer-motion 12.43, embla 8.6, Vite 8.3.5, TypeScript 7.0.2, Vitest + Playwright 1.64 (`package.json`). Dev server 127.0.0.1:5174 proxies `/api` (incl. WS) to 127.0.0.1:8030 (`vite.config.ts:3`). Hash router with 9 screens `home, faceswap, batch, processing, facemgr, extras, gallery, history, settings` (`src/router/AppRouter.jsx:3-6`); aliases `#/studio`, `#/identity-forge`, `#/queue`; drawers `#/models`, `#/diagnostics` (`AppRouter.jsx:13-15`). Entire app state lives in `App.tsx` (832 L).

#### 2.1 View-by-view answer

| Candidate view | Present? | Evidence (file:line) | Backend fields consumed |
|---|---|---|---|
| **3D mesh inspector** | YES | `components/studio/SpatialInspector.tsx:12-16` tabs Head Pose / Coverage / Morph / Light Probe / FLAME Mesh; `three/PoseGeometryViewer.jsx` (1151 L, Fibonacci-sphere lattice, 12×8 yaw/pitch nodes `:115-118`); `three/FlameMeshViewer.tsx` (`poseStrength` 0=flat TPS plane → 1=full depth, handles `coordinate_system` opencv/opengl/mesh); `three/viewportTools.ts:12-20` buffer views `lit, wireframe, albedo, normals, depth, cavity, sh` (SH 9-coeff, `:24-30`); `three/GuardedCanvas.tsx` context-loss rebuild (`RESTORE_TIMEOUT_MS`, `MAX_REBUILDS`); `three/WebGLFallbackInspector.tsx` SVG outline when WebGL absent; `facemgr/FacesetInspector3D.tsx`; `utils/glbMesh.ts` GLB parser (`:152` `GET /facesets/{id}/identity-asset/mesh`) | `POST /faceset/reconstruct-3d` + `GET /faceset/3d-state` (`App.tsx:310-342`, `useHeadMesh.ts`), `Reconstruction{vertices,faces,uvs,face_uvs,coordinate_system,method flame|tps_2_5d|identity_asset}`, `Coverage{views,bins,observed_pose_points,mesh_2_5d,mesh_full_3d,uv_texture_data_url,live_pose}` (`contracts.ts:136-152`), `/facesets/{id}/identity-asset`, `/bake-atlas` (PBR atlas 2048/4096, `contracts.ts:BakedPBRAtlas`), `/reconstruction/backbones` (`HeadModelBackbone.tsx:17`), `/rendering/studio-environments` (5 CC0 probes) |
| **Split wipe / A-B compare** | YES | `canvas/WorkstationCanvas.jsx:13-27` modes `result, wipe(Split Wipe), crossfade, blend, diff, loupe, mesh3d, matte`; WebGL2 shader constants `canvas/webglModel.js:8-20` (`MODE_IDS`, `LOUPE_ZOOMS=[3.5,4]`, `LOUPE_RADIUS_PX=90`, `DIFF_NOISE_FLOOR=3`, `DIFF_FULL_SCALE=128`, colormaps thermal/inferno/viridis); `output/OutputComparePlayer.tsx` wipe/side-by-side, A/V drift resync `DRIFT_SECONDS=0.12` (`:7`), loads `/jobs/{id}/download` + `/media/content`; `ui-layouts/ClipPathSplitReveal.tsx` exists but no non-test component imports it (grep) | preview frames `POST /api/preview` (header `X-Preview-Meta` JSON: faces[track_id,box,landmarks5,yaw,pitch,roll], `applied`, `unsupported`, `latency{}`; `api.ts:35-60`), `/media/preview-frame`, `/media/content` |
| **Timeline** | YES | `timeline/TimelineWorkstation.jsx` (1306 L, SMPTE timecode, filmstrip, pointer-anchored zoom), `timeline/TimelineDock.tsx` + `dockMath.ts` (Embla filmstrip, trim rail, speeds), `timeline/CurveEditorTrack.jsx` + `stores/parameterCurves.js` + `useHistoryStore.js` (reversible keyframe curves; linear/Bezier/hold), `timeline/chapters.js` (`MAX_CHAPTERS`), `utils/maskRanges.ts` (frame-range paint/erase), `studio/TrackLanes.tsx` (scene-cut lane + per-face track lanes + drag FaceSet onto lane), `utils/nleExporter.js` (FCPXML) | `/media/info`, `/media/filmstrip`, `/media/frame` with `X-Frame-Meta` (`frameScrub.ts:76-92`), `/media/probe`, `/media/keyframes` (`App.tsx:322`), `Inspection{tracks[],scene_cut_frames,tracking_method}`, `IdentitySheet{scene_cut_frames,tracks[].segments,missing_ranges}` (`contracts.ts:108-110`) |
| **Pose / telemetry HUD** | YES | `canvas/poseHud.js:23-57` draws 5-pt ArcFace landmarks + live `yaw/pitch/roll` badge for every face; `FaceOverlayCanvas.jsx` (874 L, boxes, wireframe/gaze overlays, face roles); `utils/poseCells.ts` standard-frame pose (+yaw nose→image right, +pitch up); `hud/PipelinePill.jsx` (5 stages DET/LMK/WRP/RST/BLN, provider badge TRT/CUDA/DML/ROCM/APPLE `:20-60`, WS `/ws/telemetry`); `faceswap/PipelinePill.jsx`+`pillModel.js` (3 steps `STAGE_1_SPARSE_PRESCAN / STAGE_2_ACTIVE_PREPASS / STAGE_3_MAIN_SWAP`, engine banners `COMPILING_TRT`/`RUNNING_CANARY`); `shell/TelemetryHeader.tsx` (GPU, precision, VRAM, FPS, execution graph .fsz→target→3D LBS→SH relight+TRT UltraMax→MP4); `diagnostics/BottleneckFlamegraph.jsx` (7 stages decode/detection/pose_tps/inswapper/masking/restorer/encode, `stageTimings.js:2-10`); `StreamOverlapPanel.jsx`/`lanesModel.js`; `PipelineSaturationPanel.jsx` (`/api/v1/system/pipeline`); `RuntimeReportPanel.tsx` (14-section `/system/runtime-report`); `EngineStatusBar.tsx`; `tray/AngleCaptureHud.tsx` + `utils/coverageRadar.ts` (angle coverage) | WS `/ws/telemetry` frames `pipeline.status{stage, stages{id:state}, stage_progress_pct, skipped_frames_pct, fps, eta_seconds, engine_state, engine_label, engine_model, engine_precision, engine_since, stage_timings_ms}`, `job.progress`, `system.telemetry{gpus[name,total_vram_mib,free_vram_mib], logical_cpu_count, advertised_providers}` (`telemetryStore.ts:4-17`, `pillModel.js:19-21`); `/api/pipeline/status` (loaded model per stage/precision/switch failure); `/ws/v1/pipeline/status` (`model_download_progress`, `trt_compilation_progress`); `/api/v1/ws/events` (`Event{type,sequence,hardware}`, gap detection `events.ts:20`) |
| **Live preview stream** | YES | `liveStream.ts` binary client for `/ws/v1/preview/stream`: 16-byte LE header = u32 frame, u32 ts ms, 3×u16 stage latency (0.1 ms), u16 flags (bits0-3 protocol v1, bit4 jpeg, 5 dropped, 6 last, 7 cached, 8-9 source) (`:1-60`); `components/LiveStreamCanvas.tsx` (`LIVE_STALE_MS`), `utils/resilientSocket.ts` | stage1/2/3 latency, dropped flag |
| **Temporal graph / smoothing visual** | **NO** (only a slider + scene-cut markers) | `studio/WorkstationControls3D.tsx:31-46,186-201` slider ↔ `temporal_min_cutoff` (light 1.5, heavy 0.03, default 0.15, debounced save 500 ms `SMOOTHING_COMMIT_MS`; saving invalidates preview caches); `TrackLanes.tsx:78-80` scene cuts; parameter curves are user keyframes, not measured jitter | `Config.temporal_min_cutoff` via `PUT /config` |
| **Batch matrix** | YES | `screens/BatchMatrixScreen.jsx` (955 L): strategies `one-to-many | grouped | matrix | recipe` (`:71`), `BUILTIN_RECIPES` (`:11`: photoreal_natural GPEN blend 0.35, high_fidelity 0.80, mesh_geometry_3d, …), auto video-split via `POST /media/split-boundaries` (`:150`), per-item idempotency key `matrix-<uuid>` (`:192`), recipe sweep `workstation/ContactSheetModal.tsx` (Low/Med/High axes) | `POST /jobs`, `/config`, `/projects/{id}` ; queue UI: `queue/BackendQueuePanel.tsx` (`/queue/{id}/{action}`, `/retry`, `/reorder`, 10 lifecycle states `queued, preparing, compiling_trt, tracking_3d, rendering, paused, muxing_audio, completed, failed, cancelled`, `backendQueue.ts:3`), `QueueDrawer.jsx` (`/jobs/{id}/{pause|resume|cancel}`, `/media/concat`, `/jobs/{id}/artifacts/{video|manifest}`), `ResumeNote.jsx` (`/jobs/{id}/checkpoint` → "re-run continues from last verified segment"), `HistoryScreen.jsx` (`/projects/{id}/resume|validate`, `/projects/import`) |
| **Identity / target lock** | YES | `facemgr/IdentityLockPanel.tsx` (`/identity/locks`), `canvas/useTargetLock.ts` (`/target/{id}/match?frame_index`), `api.ts:97-107` `lock-from-frame`, `add-angle-sample`, `/target/{lock}/angle/{yaw}/{pitch}/crop`, `/rescan`; `tray/FaceSelectionTray.jsx`, `FrameFaceStrip.tsx`; `useFaceSelectionStore.ts` (557 L, cosine clustering) | `lock.coverage{captured,total,quality_locked,buckets[]}` (`AngleCaptureHud.tsx:9-32`) |
| **FaceSet builder (.fsz)** | YES | `facemgr/FacesetBuilder.tsx`, `fszBuilder.js` (JS ZIP/CRC32 writer + DBSCAN ArcFace clustering, 5 pose buckets `:POSE_BUCKETS`), `BuilderCoverageSphere.tsx`, `FaceSetDetail.tsx` (3×3 pose matrix, `YAW_BINS/PITCH_BINS`), `PhotoSynthesisAction.tsx` | `/faceset/extract-from-media`, `/synthesize-36`, `/synthesize-photos`, `/save-fsz`, `/faceset/extraction/{id}`, `/faces/{bank/edit,bank/discover,analyze_photo,fill_missing_angles,extract_from_frame}`, `/facesets/{id}/{views,pose_matrix,thumb,rename,download}` |

#### 2.2 Angle/yaw handling in the UI (axis 1)

- Coverage grid is **yaw −90…+90 in 15° steps (12 bins) × pitch −60…+60 in 15° steps (8 bins) = 96 cells** (`utils/coverageRadar.ts:3-6`); values beyond ±90° clamp into edge bins (`:15`); there is no representation for |yaw|>90°.
- Guidance (`guide/coverageGuide.ts:5-11`): frontal 0°, three-quarter ±45° **required**, profile ±75° **recommended**.
- Face overlay 3×3 pose cell edges ±15° yaw / ±10° pitch to match backend (`faceOverlayHud.test.tsx` second test).
- The UI never displays backend refusal/engage reasons (`POSE_OUTSIDE_OBSERVED_COVERAGE`, `pose_too_oblique`, `LANDMARK_LAYOUT_IMPLAUSIBLE` — grep over `src/` returns none outside tests); `LiveFace` carries only `track_id, box, landmarks5, yaw, pitch, roll` (`api.ts:25`). Backend limits therefore are invisible to the user [gap].
- Render modes exposed: `standard`, `pose_3d`, `reenactment_3d`; swap scope `full_craniofacial | adaptive_skull | inner_face`, `skull_blend` (`workstation/renderModes.ts:17-30`, `pipeline.ts`). Pose mode `retarget | mesh_blend`; `pose_frontalization`, `tps_expression_weight` 0..1 (`pipeline.ts`).

#### 2.3 Quality/blending controls surfaced (axis 3)
Restorer ids `gpen | codeformer | ultramax | gfpgan` with weight files `GPEN-BFR-512`, `CodeFormerv0.1.onnx`, `codeformer.fp16.onnx` (UltraMax = CodeFormer + Laplacian detail fusion, `workstation/pipeline.ts:1-12`); `color_method lab|reinhard_lab|histogram`; `restoration_blend`, `feather_px`, `erosion_px`, `color_strength`, `fidelity_weight`, `identity_blend_t`, `bypass[restorer|color]`, `relight`, hair/neck/ear protection (`studio/protection.ts`), 52 ARKit expression blendshapes (`studio/arkit52.ts`), occlusion `standard|robust`, export profiles `standard_mp4 … lossless_intermediate, image_sequence, color_preserving`, `hdr_policy`, passes (alpha/occluder/uv/…; `export/VfxExportPanel.jsx`), cube LUT + grain finishing (`contracts.ts:ExportOptions`). **No explicit frequency-band / multiband-blend control** is exposed in the UI (grep `multiband|laplacian` outside test/diagnostic text returns only the UltraMax note).

#### 2.4 Compute surfaces (axis 6) and queue (axis 7)
Provider badges derived from `advertised_providers` (`hud/PipelinePill.jsx:22-60`); `Capabilities.workstation.tensorrt_mixed_precision{available,active,configured,tensorrt_available,cuda_fp16_available,active_provider,diagnostic}` and `active_hardware_profile 12gb|6gb|cpu` (`contracts.ts:6-20`); `POST /models/compile_trt` (`EngineModelRack.tsx:135`), `/system/purge-vram`, `/system/benchmark`, `/system/check-compatibility`, `/system/preflight-lint` (one-click "Convert & Optimize Now", `PreFlightLinterModal.jsx`). Job states `QUEUED PREFLIGHT ANALYZING PRESCANNING PREPASSING RENDERING MUXING PAUSED COMPLETED FAILED RETRYING CANCELLED` (`contracts.ts:21`) with `QueueSlice{slice_index,start_seconds,status,retry_count,frame_count}` (GOP-aligned segments).

### 3. Backend endpoint & WS contract actually consumed by the UI (static scan of `src/`)

Prefix `/api/v1` is added by `api()` (`api.ts:14`) except where noted. Error envelope `{error:{code,message,request_id}}` → `ApiError` (`api.ts:12`); GPU-busy preview = HTTP 503 `{busy:true,stage,retry_after_ms}` → `PreviewBusyError` (`api.ts:24-31`).

| Area | Paths (UI-side) |
|---|---|
| Config/capabilities | `/config`, `/system/capabilities`, `/health`, `/pipeline/status` (also unversioned `/api/pipeline/status`), `/models/active` |
| Preview | `POST /api/preview` (unversioned), `/media/preview-frame`, `/media/detect-frame-faces`, `/media/inspect-faces`, `/media/inspections/{id}`, `/workstation/preview`, `/studio/{start,stop}` |
| Media | `/media/{info,frame,filmstrip,probe,keyframes,content,concat,split-boundaries,export-profiles}`, `/assets/uploads/{id}?offset` (chunked upload) |
| Jobs/projects/queue | `/jobs`, `/jobs/{id}/{pause,resume,cancel,checkpoint,download,artifacts/video,artifacts/manifest}`, `/queue/{id}/{action,retry}`, `/queue/reorder`, `/projects/{id}`, `/projects/{id}/{jobs,resume,validate,download}`, `/projects/import`, `/exports/{id}/{timeline,manifest,verify}`, `/finishing/{capabilities,manifest,verify,<action>}` |
| Identity/target | `/identity/{locks,locks/{id},inspections/{id}}`, `/target/{lock-from-frame,add-angle-sample,{id},{id}/match,{id}/rescan,{id}/angle/y/p/crop}`, `/actors/{inspection_id}` |
| FaceSets/3D | `/facesets`, `/facesets/{build,open-folder,rebuild-thumbs,{id},{id}/views,{id}/pose_matrix,{id}/identity-asset(/mesh),{id}/bake-atlas}`, `/faceset/{3d-state,reconstruct-3d,extract-from-media,save-fsz,synthesize-36,synthesize-photos,extraction/{id}}`, `/reconstructions/{id}`, `/reconstruction/backbones`, `/rendering/studio-environments` |
| Models | `/models`, `/models/{rescan,hub,manifest,download,verify-and-repair,downloads,compile_trt,batches/{id}}`, `/installer/{preflight,offline-health,models/import,models/scan,models/verify,updates/backup,updates/rollback}` |
| Diagnostics | `/system/{pipeline,runtime-report,preflight-lint,purge-vram,benchmark,check-compatibility,diagnostics-report?format=json|markdown}`, `/settings/{benchmark_status,benchmark_threads}` |
| WebSockets | `/ws/telemetry`, `/api/v1/ws/events`, `/ws/v1/pipeline/status`, `/ws/v1/preview/stream` (binary) |

### 4. Audit scripts (`app/roop_3d/audit/`) — behaviour today

#### 4.1 `local_models.py` (1323 L; run as `python app/roop_3d/audit/local_models.py …` from checkout root; `repo = parents[2]` = `app/`)
CLI (`main`, `:1262-1320`): `--verify`, `--require-installed`, `--list`, `--download [ID…]`, `--repair`, `--accept-license`. Output: JSON to stdout, exit 0/1; errors wrapped in the `AuditError.envelope()` (`isolation.py:19-25`).
- **default (no flag)**: re-discovers weights (`.onnx .pth .pt .ckpt .safetensors .bin`, `WEIGHT_EXTENSIONS :50`) under scopes `{repository=legacy_root, ~/.cache, ~/.insightface, ~/.torch}` (`default_scopes :173`; `legacy_root` = `PINOKIO_HOME/api/roop-ultimate`, enforced `read_only:true` in `model_sources.json`, `project_paths.py:19-30`), refuses to drop any model present in `local_models_discovered.json` (`MODEL_DISCOVERY_INCOMPLETE`), rewrites `verified_local_models.json` (95 models: swapper 15, restorer 13, detector 11, animation 11, recognizer 9, parser 8, other 28) + `model_and_license_register.csv`.
- **`--verify`** (`:1285-1307`): (a) `verify_registry` (`:147-170`) — non-empty `models[]`, no duplicate `(scope,path)`, every state must be `VERIFIED_LOCAL`, file re-hashed with SHA-256 and size compared (`MODEL_DRIFT`), path via `safe_path` (rejects symlink/junction ancestors); (b) `verify_open_artifacts` — pinned **FLAME 2023 Open** `flame2023_Open.pkl` sha256 `e75a0990…7623`, 53 023 668 B, CC-BY-4.0, searched at `app/roop_3d/.runtime/models/flame2023_open/` and `flame/flame2023/` (`:185-206`); (c) `verify_local_cc0_assets` (`:221`) — **builds** `makehuman_cc0_anatomical_pack.npz` if missing (`AnatomicalFullHeadAsset.load(auto_build=True, verify=True)`, `allow_pickle=False`) and records digest — i.e. `--verify` can write a file; (d) `universal_manifest_report` (`:1217`) — 27-entry `universal_model_manifest.json`; fails on any `CORRUPT` entry, and (only with `--require-installed`) on non-optional entries that are `MISSING/PARTIAL/NOT_APPLICABLE`.
- Hash-trust note: `VERIFIED_LOCAL` = bytes hashed, **not** licence/provenance/inference verified (module docstring `:1-5`).
- Manager CLI (`UniversalModelDownloadManager`, `:645+`): resumable Range download to `.part`+meta, mirror failover, SHA-256 verify, quarantine of corrupt files, licence gate (`--accept-license`), disk-space check (`_check_space :1012`), optional TRT compile step (`_compile :1020`), dependency-ordered batches (`_order :1040`), `verify_and_repair :1187`.

#### 4.2 `verify_isolation.py --report` (→ `stage00.py`)
Docstring promises a "complete, fail-closed Stage 00 audit" (`verify_isolation.py:2-8`). Actual behaviour: prints a fixed JSON object `{ok:true, historical:{checked:2645,expected:2645}, comprehensive:{checked:2645}, legacy_access:"read-write", standalone_app:"roop-3d", elapsed_seconds}` (`stage00.py:33-44`). `--report` writes it to `roop_3d/audit/stage00_verification.json` (untracked; present on disk, ts 2026-11-12). No file is hashed. `isolation.verify()` (`:184-197`) is a stub; `isolation.safe_path` still rejects link traversal (`:53-67`) and is the only live protection, used by `local_models`. `runtime_policy.verify_policy` (`:30-51`) does walk files, flags `LINK_TRAVERSAL`, and reports runtime files (`app/output`, `app/projects`, `app/temp`, `logs`, `app/queue.json`…) — but nothing invokes it. Meaning: the baselines `protected_paths.json` (3038 records of **roop-ultimate** at commit `71ebb65…`, 920 untracked) and `protected_all_paths.json` (16.9 MB, 69 890 files, untracked) are dead evidence. `AGENTS.md` confirms the policy change ("All legacy file protections … are removed").

#### 4.3 Other audit-side scripts
`stage15_media_validation.py` re-measures colour/media claims → `MEDIA_COLOR_STAGE15_VALIDATION.json`; `stage2_profiler.py` (717 L, 15 bottleneck profile) and `stage9_benchmarks.py` (611 L, 10 categories) write `STAGE2/STAGE09_BENCHMARKS.json`; `procedural_assets.py` zero-network builders for `makehuman_cc0_head`, `anatomical_pack_cc0`, `cc0_probe_bundle`, `correspondence_maps`; `project_paths.py`, `runtime_policy.py`, `isolation.py`, `stage00.py` as above.

### 5. Test coverage matrix for the requested behaviours (names verified by grep)

| Behaviour | Tests (file::test) | Notes |
|---|---|---|
| **Extreme yaw (synthetic)** | `test_extreme_pose_3d_swap.py::test_full_mesh_silhouette_survives_extreme_camera` (yaw −85,−80,−60,−35,0,35,60,80,85, `:73-75`), `::test_gpu_depth_culls_hidden_texture_without_horizontal_smearing` (±80/85, `:94`), `::test_dense_pnp_and_composite_modify_full_forehead_cheek_and_jaw` (`:119`), `::test_standard_session_routes_extreme_pose_to_mesh` (`:207`; auto route >30° yaw / >25° pitch per `docs/EXTREME_POSE_3D.md:13`), `::test_uv_acquisition_never_bakes_visible_texture_into_hidden_surface` (±75/85, `:268`); `test_reenactment_engine.py::test_large_yaw_follows_the_target_when_the_identity_was_observed_there` (45°±6, 70°±8, `:98-99`), `::test_pose_beyond_the_observed_coverage_falls_back_instead_of_extrapolating` (reason `POSE_OUTSIDE_OBSERVED_COVERAGE`), `::test_profile_is_refused_not_forced` (88° → not engaged); `test_jaw_fit.py::test_a_moderate_yaw_is_not_gated_but_a_large_one_is` (20° ok, 55° → `pose_too_oblique`); `test_heterogeneous_scheduler.py::test_mesh_raycaster_masks_out_far_cheek_at_60_degree_yaw`; `test_gpu_preview_and_profile_fixes.py::test_turn_ratio_separates_frontal_from_profile_landmarks` (`JAW_CUTOFF_MIN_TURN_RATIO`) and `::test_chin_line_clamp_cuts_below_the_chin_at_any_yaw`; `test_occlusion_real_models.py::test_robust_compositor_survives_a_real_profile_that_defeats_the_parser` | Real profile footage: `test_real_target_frames` (**skips**: fixtures absent); real-model identity/occlusion suites also environment-gated |
| **Scene cut** | `test_coupled_smoothing.py::test_gap_rewind_time_jump_and_scene_cut_restart_the_trajectory` (`:88-96`), `test_head_tracking_3d.py::test_first_frame_scene_cut_flag_is_not_a_cut` / `::test_an_undetected_change_of_person_inside_a_shot_becomes_an_identity_break_cut` / `::test_the_identity_lock_is_per_shot`, `test_identity_tracker_policy.py::test_scene_cut_parks_tracks_and_only_reacquires_by_strict_appearance`, `test_occlusion_compositor.py::test_tracker_resets_on_scene_cut_frame_gap_appearance_jump_and_holds_with_limit` + `::test_scene_cut_and_lost_tracks_reset_occlusion_history`, `test_occlusion_real_models.py::test_occlusion_tracker_detects_a_real_shot_change_from_pixels_alone`, `test_reenactment_engine.py::test_scene_cut_discards_all_temporal_state` (`:332`), `test_temporal_consistency.py::test_mask_recovery_respects_current_occluder_and_scene_cut`, `test_view_selection.py::test_gap_scene_cut_and_one_off_preview_choose_the_best_views_at_once`; UI: `TrackLanes` e2e `studio.e2e.ts` | `docs/TEMPORAL_STABILITY.md`: scene-cut byte-identical to un-stabilised input; >70 % flicker-reduction gate **not met** (26.9 % warp-error / 28.0 % Laplacian reduction at 0.25 history weight; 43.0 % at 0.40, 52.4 % at 0.50) |
| **Mesh generation** | `test_identity_asset_{fit,build,texture,quality,makehuman,topology_api}.py`, `test_makehuman_head.py`, `test_mesh_bridge.py`, `test_glb_export.py` (34 tests: GLB header/chunks/shape keys/head track/`/mesh.glb` route), `test_stage08_{flame,geometry,mesh_io,tps,uv}.py`, `test_stage11_faceset_mesh.py`, `test_dense_face_{backbone,bundle}.py`, `test_dense_correspondence_visibility.py`, `test_displaced_surface.py`, `test_anatomical_pack.py` (60 orthogonal names…), `test_topology_registry.py`, `test_semantic_shape.py`, `test_articulated_head_models.py`, `test_full_head.py`, `test_flame_and_liveportrait.py` (Stage 16 **incomplete**: acceptance cmd 2 pass / 1 fail, `docs/REMEDIATION_STAGE16.md:1-5`); UI: `glbMesh.test.ts`, `headMesh.test.ts`, `faceset3dViewer.test.tsx` | synthetic ground-truth; dense backbone requires local 3-tier bundle (weights absent → `test_dense_face_bundle.py` "Synthetic contract fixtures ONLY") |
| **Frequency / multiband blending** | `test_photorealistic_swap_core.py::test_laplacian_blend_reproduces_background_at_alpha_zero_and_foreground_at_one` (`:73`) / `::test_laplacian_blend_has_no_halo_across_a_soft_seam` / `::test_laplacian_blend_overshoot_is_bounded_by_the_inputs…` (`:389`), `test_stage09_adaptive_runtime.py::test_fast_multiband_blend_numerical_bounds` (`:293`), `test_stage11_faceset_mesh.py::test_multiband_blend_feathers_exposure_step_and_leaves_empty_texels_empty`, `test_hires_handoff.py::test_multiband_blend_keeps_fine_detail_and_zero_alpha_exact`, `test_stage09_compositor.py::test_frame_compositor_temporal_detail_reduces_laplacian_flicker`, `test_shading_gain.py` (low-frequency shading gain only), `test_displaced_surface.py::test_high_frequency_displacement_between_native_vertices_is_not_lost`, `test_seam_free_swap_pipeline.py::test_laplacian_jaw_adaptation_removes_the_jaw_width_mismatch`, `test_rendering_calibration.py::test_intrinsic_uv_cleanup_keeps_four_observed_detail_bands` | synthetic scenes; no real-footage perceptual metric except `test_stage11_evaluation_framework.py` (metric primitives, 11 dataset categories) |
| **CLI / terminal / report** | `test_pinokio_terminal_telemetry.py` (46 tests: env lock, HUD width, defusal of control chars, `ROOP3D_HUD=off`, TRT-BUILDER heartbeat, 14-section report), `test_diagnostics_suite.py::test_14_section_structured_runtime_report`, `test_e2e_video_pipeline.py::test_the_terminal_hud_saw_the_stream_and_the_remux`, `test_gpu_preview_and_profile_fixes.py::test_the_terminal_shows_the_angle_scan_phase…`, `test_stage19_cli_model_root.py` (offline `local_models` CLI uses server data root), `test_stage16_preflight.py`; UI: `runtimeReport.test.tsx` (22) | `tools/project_admin.py` CLI itself has **no** dedicated test file named for it (covered indirectly by `test_standalone_install.py`, `test_stage16_reset_safety.py`) [INFERENCE] |
| **Queue resume / checkpoint** | `test_stage09_stream_resume.py` (finished-segment resume from `job_checkpoint.json`, corrupt segment, endpoint), `test_async_runner.py::test_pause_and_resume_with_frames_in_flight`, `test_stage12_worker.py::worker_pause_checkpoint_r…`, `test_stage18_recovery.py` (26: cancel/kill/starve), `test_remediation_stage7.py` (NPZ checkpoint round trip, corrupt rejected, pickle refused), `test_remediation_stage3.py::test_prepass_checkpoint_identity_and_tamper_checks`, `test_postproduction.py::test_pass_checkpoints_resume_after_pause_during_baked_encoding`, `test_stage09_remediation.py` (batch-matrix strategies + persistent queue), `test_stage15_resilience.py` (tier-4 mid-job cancel, pause/resume, port collision), `test_stage08_offline_workflow.py::resume_binds_models_inspe…`, `test_batch_matrix_and_checkpoint_history.py`; model downloads: `test_model_downloader.py` (53) + `test_stage03_downloader.py` Range resume; UI: `ResumeNote`, `batchAndHistory.test.tsx`, `sessionReload.test.tsx` | long-run: `RELIABILITY_STAGE18.md`: 30- and 60-min soaks **not executed** (AV kill); only scaled soaks in `docs/reliability/*.json` |
| **UI e2e (browser)** | 17 Playwright specs, §7 | require running backend or `mock-server` (all values fixed placeholders, `mock-server/server.ts:1-8`) |

Test-suite health recorded in docs: `docs/reviews/roop3d-bug-audit-2026-11-08.md` — Vitest 447 passed/52 files, Playwright 70 passed, backend focused 96 passed/4 skipped; **full Python suite (~1,600 tests then) did not finish in 600 s**. Current static count: 2167 Python test functions + 859 Vitest cases + 112 Playwright cases. `logs_baseline_pytest.txt` (untracked, not read) exists at root.


### 6. `app/roop_3d/tests/` — 202 files, one row each

Counts are static `def test_*` definitions (parametrize not expanded); total **2167** test functions across 202 files (pytest, `roop_3d/pyproject.toml`). Flags column: static keyword hits in the file body (yaw/extreme-pose, scene-cut, mesh, frequency/multiband/Laplacian, CLI/terminal, resume/checkpoint/recover) — a *hint*, not proof of coverage; the coverage matrix in §5 is name-verified.


#### Group 1. Detection, landmarks, pose, tracking, identity lock, temporal smoothing  (22 files, 279 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_angle_diagnostics.py` | 12 | tests: decomposition_recovers_similarity_parameters, second_difference_ignores_constant_velocity_b, affine_jitter_is_scale_and_unwrapped_angle_aw | yaw,resume |
| `test_angle_diagnostics_job.py` | 9 | The runner configures the angle diagnostics from the job, publishes the JSONL atomically and attests it in the manifest. | yaw |
| `test_coupled_smoothing.py` | 10 | Size-normalised, coupled smoothing of landmarks and the dense fit. | yaw,scenecut |
| `test_head_tracking_3d.py` | 43 | Head/gaze tracker: filters, per-shot identity lock, gaze, lighting, shot boundaries and the tracker as a whole. | yaw,scenecut,mesh,resume |
| `test_head_tracking_stream.py` | 25 | Scene adaptation (the 16-frame / 3-second calibration pass), the video runner and pipeline tap, and live telemetry on | yaw,scenecut,mesh,resume |
| `test_identity_lock.py` | 24 | Identity lock: report, hash-bound mapping, staleness, render guard, run report and job binding. | yaw,scenecut,resume |
| `test_identity_real_inference.py` | 3 | Identity lock on REAL local detector + ArcFace inference with exact ground truth. | yaw,scenecut,resume |
| `test_identity_tracker_policy.py` | 20 | Identity-lock tracking policy: analytic scenarios with exact ground truth. | scenecut,resume |
| `test_inspection_gpu.py` | 10 | Inspection / target-lock models (detector, ArcFace, 1k3d68) run on CUDA on big cards, on the CPU otherwise. |  |
| `test_neural_face_registration.py` | 4 | The dense swap crop is registered to the swapper template from landmarks, never framed by the mesh silhouette. | yaw,mesh |
| `test_persistent_target_registry.py` | 13 | tests: strict_profile_gate_keeps_slot_after_reentry, fifteen_pose_buckets_retain_best_quality, hungarian_assignment_avoids_greedy_fragmentat | yaw,resume |
| `test_remediation_stage5.py` | 29 | Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression and "Use Face From Frame". | yaw,cli,resume |
| `test_remediation_stage8.py` | 5 | tests: noise_and_turn, pose_hysteresis, bezier_exact | yaw |
| `test_resampling_borders.py` | 4 | Replicated borders (no black wedges) and alias-free photograph caching. |  |
| `test_stage04_pose_selector.py` | 6 | Unit tests for pose-bin classification and nearest-view reference selector. | yaw |
| `test_stage06_pipeline.py` | 10 | Real local reference inference plus deterministic tracking/alignment contracts. | yaw,scenecut |
| `test_stage12_governor.py` | 4 | tests: six_gigabyte_profile_is_single_stream_and_rel, twelve_gigabyte_profile_uses_live_free_memory, low_live_budget_selects_cpu_and_oom_detection |  |
| `test_stage12_target_lock.py` | 7 | Stage 12: rank-preserved person clusters, pose-bridging bank, face analysis validity rules. | yaw |
| `test_swap_presence.py` | 7 | A tracker gate that closes for a frame or two fades the swap instead of popping it. | yaw |
| `test_target_ui_endpoints.py` | 8 | Endpoints behind the face-bank UI: angle crops, unlock, rescan, and the engine status / switch-failure read-back. | yaw |
| `test_temporal_consistency.py` | 16 | tests: so3_pose_filter_damps_roll_jitter_and_takes_s, so3_filter_resets_on_head_whip_and_accepts_sc, detail_stabilizer_is_identity_on_first_frame_ | scenecut,freq,resume |
| `test_view_selection.py` | 10 | Source-view selection: pose cost, hysteresis, and weights that cross-fade instead of popping. | yaw,scenecut |

#### Group 2. 3D reconstruction, mesh, FaceSet/.fsz, identity asset, FLAME, reenactment, expression  (51 files, 583 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_3d_relighting.py` | 3 | Stage 17 Verification Gate: 3D Relighting & Normal Map Estimation. | yaw,resume |
| `test_albedo_prior.py` | 15 | Data-free albedo prior: robustness, ground-truth extrapolation, refusal without skin, and the completion far-field hook. | resume |
| `test_anatomical_pack.py` | 7 | Unit tests for the CC0 full-head synthesis and renderer-neutral dynamics. | mesh,freq |
| `test_animation_history_roles.py` | 7 | Frame curves, protected role intervals and non-destructive compositor behavior. |  |
| `test_articulated_head_models.py` | 7 | Contract tests for the optional, clean-room articulated reconstruction graph. | yaw |
| `test_backbone_status.py` | 6 | The read-only backbone report behind the Inspector's "3D Head Model Backbone" dropdown. |  |
| `test_check_topology_assets.py` | 6 | The asset check tells a new user exactly what is missing, never fails an install, and never touches the network. |  |
| `test_dense_correspondence_visibility.py` | 11 | Analytic visibility regressions only, not evidence of real-model image quality. | mesh |
| `test_dense_face_backbone.py` | 14 | Analytical geometry and neural-routing regressions; no pretrained weights claimed. | yaw,freq,resume |
| `test_dense_face_bundle.py` | 16 | Synthetic contract fixtures ONLY. They are not models or quality evidence. |  |
| `test_displaced_surface.py` | 6 | Analytical refinement contracts, not pretrained skin-detail quality claims. | yaw,mesh,freq |
| `test_expression_and_synthesis.py` | 17 | Motion invariants, post-enhancement closure, wire parity and archive provenance. | yaw,scenecut,mesh |
| `test_expression_audio_director.py` | 20 | ExpressionAndAudioDirector: emotion offsets in psi-space and the audio lock for lip/jaw motion under occlusion. |  |
| `test_extreme_pose_3d_swap.py` | 13 | Full-head coverage, perspective visibility and old-jaw removal regressions. | yaw,scenecut,mesh |
| `test_flame_and_liveportrait.py` | 9 | Stage 16 software checks; hardware/media acceptance is a separate fail-closed gate. | mesh |
| `test_full_geometry_swap.py` | 3 | Stage 5B source-geometry lock and full-head UV completion contracts. |  |
| `test_full_head.py` | 8 | Full-head regression: the MakeHuman topology must survive the runtime rig. | yaw,scenecut,mesh |
| `test_glb_export.py` | 34 | Stage 12: binary glTF serialization (container, buffers, textures, shape keys, head track), the `/mesh.glb` route, and the | mesh |
| `test_identity_asset_build.py` | 17 | Stage 3 Phases D+E: the identity asset (contents, honesty, GLB inspection) and its persistence / reuse rules. | yaw,mesh,resume |
| `test_identity_asset_fit.py` | 11 | Stage 3 Phase B: multi-view landmark geometry. Assertions compare against a KNOWN synthetic ground truth, | yaw,mesh,resume |
| `test_identity_asset_makehuman.py` | 13 | Identity asset built on the open MakeHuman (CC0) head template: template contract, build, honesty, persistence, API. | yaw,mesh,resume |
| `test_identity_asset_quality.py` | 13 | Stage 3 Phase A: FaceSet quality / coverage analysis, plus the FaceSet-reader landmark fixes it relies on. | yaw |
| `test_identity_asset_texture.py` | 11 | Stage 3 Phase C: identity texture atlas. Atlas texels are compared with the KNOWN synthetic ground-truth texture. | yaw,mesh,resume |
| `test_identity_asset_topology_api.py` | 21 | ``/identity-asset/mesh`` topology selection and semantic sliders, plus ``/morphable-topologies``. | yaw,mesh |
| `test_jaw_fit.py` | 20 | Opt-in jaw-width fit of the canonical proxy head in the mesh pose-blend path (``PoseBlendProjector``, ``ReferencePipeline``). | yaw,scenecut,mesh,resume |
| `test_makehuman_head.py` | 15 | MakeHuman (CC0) head model: pinned sources, asset integrity, frame conventions and multi-view fitting. | yaw,mesh,resume |
| `test_mesh_bridge.py` | 20 | Barycentric mesh bridge: exact closest-point maths, then conversion of positions / normals / UV albedo in both directions. | mesh,resume |
| `test_native_face_geometry.py` | 7 | Native-camera, articulation and UV regressions without licensed model data. |  |
| `test_photometric_head.py` | 9 | Independent directional-light math and real hardware shader regressions. | yaw,mesh,resume |
| `test_pose_mesh_blend.py` | 9 | Geometric endpoints, depth visibility, curve plumbing and real local output. | yaw,mesh |
| `test_reenactment_engine.py` | 29 | Stage 4: temporal 3D reenactment engine, verified on a scripted synthetic performer with KNOWN pose, expression and timing. | yaw,scenecut,resume |
| `test_reenactment_session_real.py` | 5 | Stage 4 on REAL local inference: real SCRFD + 2d106 landmarks, real inswapper, real video, real multi-view FaceSet. |  |
| `test_seam_free_swap_pipeline.py` | 5 | Stage 6 end to end on the real MakeHuman head: jaw-width adaptation -> rasterised swap mask -> z-buffer / normals into the | mesh,freq |
| `test_semantic_shape.py` | 14 | Semantic shape sliders: every claim is re-measured from the *mesh*, not from the Jacobian the controller solved with. | mesh |
| `test_silhouette_replacement.py` | 38 | Stage 6B: silhouette replacement. A different head shape must not leave the target's old jaw behind (narrower source) or | mesh,resume |
| `test_source_texel_parsing.py` | 4 | Hair, neck, cloth and background texels of a source photograph are never conditioning appearance. | yaw |
| `test_stage04_faceset_reader.py` | 4 | Unit and integration tests for FaceSetReader. | yaw |
| `test_stage04_schemas.py` | 7 | Unit tests for FaceSet schemas, contracts, and V2 export functionality. | yaw |
| `test_stage04_validator.py` | 13 | Unit tests for archive security validator and zip-bomb defense. |  |
| `test_stage07_identity_panel_api.py` | 6 | Stage 7 identity-panel API: view thumbnails and the Stage 3 identity asset (status / explicit build / viewer mesh). | mesh |
| `test_stage08_flame.py` | 4 | Acceptance tests for Stage 08 FLAME license gating, robust losses, and parametric fitting. | yaw,mesh |
| `test_stage08_geometry.py` | 8 | Acceptance tests for Stage 08 3D geometry contracts and coordinate conventions. | yaw,mesh,resume |
| `test_stage08_mesh_io.py` | 2 | Acceptance tests for Stage 08 validated atomic mesh export (.obj, .mtl, .glb). | mesh |
| `test_stage08_tps.py` | 6 | Acceptance tests for Stage 08 2.5D landmark PnP solver and TPS/Delaunay warper. | yaw,mesh,resume |
| `test_stage08_uv.py` | 3 | Acceptance tests for Stage 08 visibility-weighted UV texture synthesis and confidence gating. | yaw,mesh |
| `test_stage11_faceset_mesh.py` | 7 | Stage 11: RBF deformation, welded normals, multiband blend, texture completion, padding mask, 68-point correspondence. | mesh,freq |
| `test_stage18_extensions.py` | 9 | Clean-room regression coverage for the Stage 18 expression, UV, and atlas additions. | freq |
| `test_stage22_blender_flame_extras.py` | 11 | Stage 22 geometry, restoration, grading, proxy and timeline utilities. | scenecut,mesh |
| `test_stage7_faceset_builder.py` | 5 | Curation invariants independent of installed GPU weights. | yaw |
| `test_swap_region_3d.py` | 18 | 3D swap mask beyond the inner face oval, the biharmonic jaw field, and the z-buffer rasteriser they share. | yaw,mesh,freq |
| `test_topology_registry.py` | 17 | Morphable topology registry: defaults, licence gating, SHA-256 pinning, safe parsing, and the FLAME 2023 Open frame. | mesh |

#### Group 3. Compositing, blending, masks, colour, restoration, visual quality  (27 files, 350 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_dense_compositor.py` | 8 | Compositor behaviour specific to dense 3D replacements: no second restorer, shading, coverage smoothing, presence. | yaw,freq |
| `test_gpu_compositor.py` | 11 | Stage 10 geometry, semantic exclusions, real CUDA binding and slot ownership. | freq,resume |
| `test_hair_neck_protection_controls.py` | 14 | Hair / neck / ear protection toggles and the skin micro-texture amount: real engine behind each control, opt-in on the wire. |  |
| `test_hires_handoff.py` | 16 | Defect #1: enhancer detail must survive the Stage 2 -> Stage 3 handoff, and UltraMax must never be bypassed by default. | freq |
| `test_occlusion_compositor.py` | 36 | Stage 5 occlusion compositor: unit and integration tests that need no model weights. | scenecut |
| `test_occlusion_real_models.py` | 7 | Stage 5 on REAL local inference: real BiSeNet, real XSeg / face-occluder networks, real tracker, real video. | scenecut |
| `test_photorealistic_swap_core.py` | 30 | Stage 17: numeric contracts of the photorealistic swap core and its primitives (synthetic scenes, no model weights). | mesh,freq,resume |
| `test_physical_camera.py` | 8 | Stage 10 physical-camera optics, relighting, assets, and UV denoising tests. | freq,resume |
| `test_postproduction.py` | 11 | Real codecs and generated pixels, no inference/model or sibling-app imports. | mesh,resume |
| `test_production_studio.py` | 9 | Studio contracts independent of physical cameras and proprietary weights. |  |
| `test_rendering_calibration.py` | 6 | Color-temperature, energy-budget, and intrinsic-detail regressions. | mesh,freq |
| `test_shading_gain.py` | 5 | Per-pixel lightness gain: the swapped face follows the target's low-frequency shading, rim included. | freq |
| `test_stage06_ultramax_restorer.py` | 3 | Stage 6: UltraMax is a real, selectable restorer in the preview and render paths (it used to be rejected outright). | freq |
| `test_stage09_compositor.py` | 10 | Acceptance checks for Stage 09 mask, restoration and frame fusion behavior. | scenecut,freq |
| `test_stage10_audio_sync.py` | 9 | Stage 10 audio timing, silence gating and safe lip-sync integration tests. | scenecut |
| `test_stage11_diffusion.py` | 9 | tests: diffusion_is_optional_and_local_candidates_re, cross_family_base_or_adapter_is_rejected_befo, loader_result_is_checked_for_family_before_us |  |
| `test_stage11_evaluation_framework.py` | 10 | Stage 11: Objective Visual Quality Evaluation Framework Tests. | yaw,freq |
| `test_stage15_color_science.py` | 25 | Stage 15 colour science: every number is checked against a published reference value or an |  |
| `test_stage15_linear_light.py` | 13 | Stage 15 linear-light compositing: analytic expectations, exact identity at the extremes, every blend mode. | yaw |
| `test_stage15_studio_polish.py` | 13 | Stage 15 Studio Polish VFX Finishing & Biophysical Solvers Acceptance Tests. | yaw,mesh,freq,resume |
| `test_stage27_workstation_final.py` | 33 | Final workstation stage: parameter curves for relight / occlusion, clean relit pass, transfer ledger, flamegraph lanes, linter. | yaw,resume |
| `test_stage8_studio.py` | 7 | Stage 8 studio integration. Neural sessions are replaced by explicit numerical fixtures. | yaw,resume |
| `test_strand_occlusion.py` | 21 | StrandAwareOcclusionCompositor: depth z-test, N.V grazing falloff, semantic and object cues, and the directional strand matte. |  |
| `test_swap_coverage.py` | 6 | Coverage of the dense swap: resolution-independent feather, landmark-hull bound, no nearest-neighbour stairs. | mesh |
| `test_vfx_compositing_engine.py` | 11 | Lead VFX Compositing & Semantic Segmentation Engine Tests. | mesh,freq |
| `test_vfx_export_and_diagnostics.py` | 7 | Comprehensive unit and integration test suite for Stage 12: | scenecut,cli,resume |
| `test_workstation3d_controls.py` | 12 | Workstation UI controls with a real engine behind them: ``hairline_offset_px`` (forehead matte band) and ``ambient_fill`` (manual light). | yaw |

#### Group 4. Compute: ORT / TensorRT / CUDA / VRAM / hardware profiles  (17 files, 151 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_accelerator_resolution.py` | 2 | tests: auto_prefers_tensorrt_then_cuda_then_cpu, explicit_choice_is_kept_and_cpu_wins_without_ |  |
| `test_execution_worker_pools.py` | 3 | tests: post_encode_pool_stitches_completed_chunks_in, gop_chunk_rejects_unaligned_start_and_nonfina, nvenc_chunk_runs_on_the_worker_with_lossless_ |  |
| `test_gpu_preview_and_profile_fixes.py` | 28 | Regression tests for the Nov 12 2026 fixes: concurrent-stage CUDA graphs, trim-window work, warm preview sessions, | yaw,mesh,cli |
| `test_heterogeneous_scheduler.py` | 8 | Verification Gate & Regression Suite for Stage 15 Heterogeneous Scheduler. | yaw,mesh |
| `test_nv_cvcuda_pipeline.py` | 14 | Stage 14 contracts and real device checks, with a fail-closed acceptance mode. | scenecut,mesh,freq |
| `test_remediation_stage2.py` | 28 | Stage 2: actual CPU graph validation and simulated native TRT failure paths. |  |
| `test_stage03_engine_cache.py` | 3 | tests: every_hardware_version_and_shape_component_ch, metadata_binds_actual_compiled_bytes_and_dete, shape_profiles_are_validated_and_onnx_cache_d |  |
| `test_stage07_cache.py` | 4 | tests: atomic_bundle_publication_and_thread_single_w, failed_build_never_publishes_and_lock_timeout, hardlinked_manifest_is_rejected | cli |
| `test_stage07_diagnostics.py` | 2 | tests: active_job_preserves_exact_provider_fallback_, capabilities_and_diagnostics_report_expose_ha |  |
| `test_stage07_runtime.py` | 6 | tests: named_outputs_reorder_without_swapping_values, expired_weak_proxies_do_not_break_ort_session, actual_ort_sessions_are_evicted_and_aliases_r |  |
| `test_stage07_validation.py` | 11 | tests: embedding_gate_detects_rotation_even_with_sma, contract_errors_never_turn_into_passing_parit, image_metrics_and_border_gate_detect_small_lo |  |
| `test_stage09_adaptive_runtime.py` | 10 | Tests for Stage 9 Unified Adaptive CPU/GPU Runtime and NVIDIA Optimization. |  |
| `test_stage17_ci_hardware.py` | 8 | Stage 17 Automated Tests: CI/CD Pipeline, Hardware Compatibility & Release System. | cli |
| `test_stage19_hardware_profile.py` | 3 | Stage 19: a real "12 GB" card reports slightly under 12288 MiB and must still get the 12gb profile. |  |
| `test_stage7_gpu_throughput.py` | 5 | Stage 7 dynamic reconstruction profiles, telemetry and VRAM governance. |  |
| `test_trt_mixed_precision.py` | 14 | TensorRT mixed precision policy, fallback chain, and opt-in GPU qualification. | mesh |
| `test_zero_copy_execution.py` | 2 | tests: ring_reuses_device_ortvalue_and_tracks_zero_i, ring_requires_the_declared_stage_order |  |

#### Group 5. Job queue, checkpoint/resume, video IO, FFmpeg, streaming, live preview, reliability  (33 files, 401 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_async_pipeline.py` | 22 | Defect #3: the asynchronous multi-stream pipeline, the shared/exclusive inference fence and hardware telemetry. | cli |
| `test_async_runner.py` | 5 | Defect #3: the job runner drives the four render stages through the asynchronous pipeline, correctly. | scenecut,resume |
| `test_batch_matrix_and_checkpoint_history.py` | 3 | tests: atomic_json_and_binary_checkpoint_creation_an, project_validation_badges_inputs_verified_vs_, media_split_boundaries_endpoint | resume |
| `test_e2e_video_pipeline.py` | 11 | Stage 17 end-to-end: generated .fsz + 30-frame .mp4 -> import/upload routes -> real job -> download -> ffprobe / SSIM / identity. | yaw,freq,cli |
| `test_live_stream.py` | 18 | Defect #3: binary live transport -- header, GPU/CPU JPEG, latest-wins hub, lazy cache entries, WebSocket, telemetry. |  |
| `test_reliability_harness.py` | 4 | The stub recovery harness must never read installed neural weights. | resume |
| `test_remediation_stage3.py` | 18 | Real host/codec/ORT tests; production GPU fidelity is a separate campaign. | resume |
| `test_remediation_stage4.py` | 19 | Stage 4: non-blocking live preview, cache-first serving and model-lifecycle mutex. | yaw |
| `test_remediation_stage7.py` | 19 | Stage 7: Stage 27 pre-flight lint checks, measured stage timings and NPZ project checkpoints. | resume |
| `test_stage02_jobs.py` | 7 | tests: complete_transition_matrix, durable_idempotency_conflict_and_secret_key_h, concurrent_creates_serialize_to_one_job | cli,resume |
| `test_stage02_media.py` | 6 | tests: real_container_probe_and_precise_frame_previe, media_read_boundary, preview_selectors_are_bounded_and_finite |  |
| `test_stage05_media.py` | 24 | Stage 05 acceptance uses generated real containers and real codecs only. |  |
| `test_stage05_timeline_frames.py` | 15 | Stage 5: the scrubbing frame service and its /media/info, /media/frame, /media/filmstrip endpoints. | yaw |
| `test_stage08_offline_workflow.py` | 9 | Stage 8 integration gates. Contract sessions are explicit fixtures; codecs/storage/API are real. | scenecut,resume |
| `test_stage09_remediation.py` | 8 | Stage 9 Automated Verification Gate: | resume |
| `test_stage09_stream_resume.py` | 3 | Stage 9: the streaming render path resumes from the last verified finished segment (job_checkpoint.json). | resume |
| `test_stage12_worker.py` | 3 | tests: post_job_runs_pipeline_publishes_manifest_and, worker_pause_checkpoint_resume_continues_to_s, simulated_cuda_oom_retries_on_cpu_without_par | yaw,resume |
| `test_stage13_bridges.py` | 18 | tests: ready_idempotence_and_conflict, missing_node_checkpoint_or_input_rejected_bef, progress_preview_correlation_and_reconnect | resume |
| `test_stage14_workstation.py` | 12 | tests: versioned_project_reopen_and_verified_checkpo, project_drift_blocks_checkpoint_continuation_, untrimmed_job_identity_remains_compatible_and | yaw,mesh,resume |
| `test_stage15_e2e_golden.py` | 1 | Stage 15: Golden-Clip End-to-End Video Swap Smoke Test. | yaw |
| `test_stage15_media_av.py` | 19 | Stage 15 audio, subtitles, timing, rotation and image sequences, all through real codecs and containers. |  |
| `test_stage15_media_extras.py` | 17 | Stage 15: input containers, still sequences (JPEG/EXR), validation, capabilities, API, OCIO, audio-sync extension. |  |
| `test_stage15_media_reader.py` | 8 | Stage 15 decode: colour-correct, explicit about HDR, untagged material resolved by recorded convention. |  |
| `test_stage15_media_writer.py` | 17 | Stage 15 encode: colour-correct output, honest HDR handling and verified export profiles (real codecs). |  |
| `test_stage15_resilience.py` | 5 | Stage 15: Resilience & Fault Injection Tests. | resume |
| `test_stage15_stream_color.py` | 4 | The raw NVDEC/NVENC stream backend must not change colour (software decode/encode; needs the ffmpeg CLI it shells out to). | cli |
| `test_stage18_growth_ledger.py` | 31 | Stage 18 unit tests: growth analysis, lifecycle ledger, disk handling, failure classification, orphan guard, redaction. | resume |
| `test_stage18_recovery.py` | 26 | Stage 18 integration tests: every way a long job can be stopped, killed or starved, and what must be true afterwards. | resume |
| `test_stage18_resources_live.py` | 6 | Stage 18: resources that need the real thing - ONNX Runtime on the real GPU, real allocation failures, server WebSockets, | resume |
| `test_stage18_soak_bounded.py` | 5 | Stage 18: scaled-down soak runs, judged against the bounded profile. The 30/60-minute runs use the same code via | resume |
| `test_stage23_workstation_parity.py` | 15 | Stage 23: multi-person actors, paint/erase mask ranges, GOP batch planning and the 10-state lifecycle. | resume |
| `test_video_job_runner.py` | 13 | Stage 17: the FFmpeg-piped loop (real ffmpeg, no model weights, no faces). | freq |
| `test_workstation_preview.py` | 10 | Preview contract, real compositor buffers and admission/lifecycle regressions. | scenecut |

#### Group 6. CLI, terminal HUD, telemetry, diagnostics  (4 files, 58 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_diagnostics_suite.py` | 3 | tests: preflight_render_linter_checks, synthetic_thread_benchmark_runner, 14_section_structured_runtime_report |  |
| `test_pinokio_terminal_telemetry.py` | 46 | Pinokio terminal HUD, 14-section runtime report and TensorRT build heartbeat. | cli,resume |
| `test_stage18_report_api.py` | 7 | Stage 18: the diagnostic report - contents, redaction, local-only behaviour and the audit/repair endpoints. |  |
| `test_stage19_cli_model_root.py` | 2 | Stage 19: the offline model commands must use the server's data root. |  |

#### Group 7. Model registry, downloader, governance, engine selection  (17 files, 197 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_local_models.py` | 11 | tests: hashes_match_actual_bytes_and_consumer_verifi, registry_consumer_detects_mutation, license_register_does_not_invent_permissions_ |  |
| `test_local_weight_trust.py` | 9 | Weights the user placed in app/models are auto-trusted, never blocked, and reported by the read-back. |  |
| `test_model_download_sources.py` | 8 | Verified public sources and the unpublished-rights acknowledgement path (offline). |  |
| `test_model_downloader.py` | 53 | Universal model download manager: resumable Range downloads, mirror failover, checksum recovery, licence gating, | mesh,resume |
| `test_model_downloader_integration.py` | 4 | Integration tests for model downloader endpoints, middleware alias, and preflight check action. |  |
| `test_model_registry.py` | 5 | Comprehensive test suite for Stage 13: | freq |
| `test_model_selection_regressions.py` | 9 | tests: catalog_reports_provider_weights_as_installed, catalog_does_not_verify_corrupted_provider_we, detector_selection_changes_the_loaded_graph | yaw |
| `test_stage03_api.py` | 3 | tests: api_exposes_authoritative_registry_notices_an, api_local_first_reuse_does_not_download_unkno, api_download_gate_and_real_cached_adapter_res | resume |
| `test_stage03_downloader.py` | 19 | tests: atomic_download_digest_and_reuse, interruption_is_fsynced_and_next_call_resumes, range_resume_with_matching_identity | cli,resume |
| `test_stage03_registry.py` | 9 | tests: manifest_preserves_all_observed_identities_wi, verifier_detects_digest_size_and_permission_e, local_provider_returns_original_path_and_neve |  |
| `test_stage06_engine_selection.py` | 4 | tests: older_jobs_and_explicit_reference_keep_origin, unavailable_selection_rejected_before_media_a, worker_dispatches_explicit_adapter_and_never_ |  |
| `test_stage06_model_hub.py` | 10 | tests: hub_exposes_registry_evidence_and_keeps_exper, hub_finds_hash_verified_artifacts_in_configur, hub_uses_runtime_first_match_for_duplicate_co |  |
| `test_stage12_model_governance.py` | 9 | tests: all_canonical_and_supplemental_records_materi, scanner_requires_integrity_compatibility_lice, unknown_license_record_cannot_become_an_infer | resume |
| `test_stage14_orchestrator.py` | 28 | Stage 14 orchestrator acceptance: deterministic selection, no unsupported configuration, no silent | yaw |
| `test_stage16_model_bundle.py` | 9 | Unit tests for Stage 16 model manager, bundle import, archive traversal security, and hash verification. |  |
| `test_studio_environment_routes.py` | 4 | Stage 13: ``GET /api/v1/rendering/studio-environments`` serves the five CC0 lighting probes in a browser-ready form. |  |
| `test_swapper_catalog_and_adapter.py` | 3 | tests: models_api_lists_all_engine_catalog_entries, local_rescan_exposes_custom_extensions_and_un, dummy_gpu_inference_respects_native_swapper_c |  |

#### Group 8. Platform: scaffold, API contracts, security, installer, packaging, docs  (18 files, 148 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `test_git_boundary.py` | 4 | tests: boundary_accepts_only_independent_app_source, guard_reads_real_git_index, dedicated_remote_is_required |  |
| `test_packaging.py` | 4 | The ``[tool.setuptools] packages`` list decides what a built wheel contains. A package missing from it works from a source |  |
| `test_runtime_policy.py` | 7 | tests: runtime_activity_is_reported_without_rewritin, runtime_policy_allows_unified_source_changes, runtime_prefix_does_not_misclassify_local_sou |  |
| `test_stage00.py` | 5 | Real filesystem tests for unified architecture. |  |
| `test_stage01_scaffold.py` | 15 | Offline tests of the real ASGI application, socket ownership and audit gate. |  |
| `test_stage02_api.py` | 9 | tests: all_rest_contracts_are_in_openapi, config_put_backup_reload_secret_rejection_and, job_rest_idempotency_controls_and_validation | resume |
| `test_stage02_boundaries.py` | 2 | tests: real_directory_link_is_rejected_before_reads_, atomic_replace_does_not_modify_hardlinked_inp |  |
| `test_stage02_settings.py` | 7 | tests: invalid_catalog, catalog_has_all_requested_keys, config_atomic_backup_and_explicit_migration |  |
| `test_stage13_security.py` | 26 | Complete Security, Privacy, and Synthetic Media Protection Test Suite for Stage 13. | mesh,resume |
| `test_stage16_installer_api.py` | 1 | Integration tests for Stage 16 installer REST API endpoints. | resume |
| `test_stage16_offline_health.py` | 2 | Unit tests for Stage 16 offline dependency cache and offline readiness verification. |  |
| `test_stage16_preflight.py` | 14 | Unit tests for Stage 16 environment preflight and hardware profile detection. |  |
| `test_stage16_reset_safety.py` | 3 | Unit tests for Stage 16 reset safety and boundary isolation. |  |
| `test_stage16_share_and_pipeline.py` | 15 | Single-origin hardening: share-mode token, unversioned API aliases, hardware profile, pipeline frames. |  |
| `test_stage16_update_rollback.py` | 3 | Unit tests for Stage 16 update checking, configuration backup, and safe rollback. |  |
| `test_stage19_docs.py` | 13 | Stage 19: the user documentation must reflect actual application behaviour. | yaw,resume |
| `test_standalone_app.py` | 3 | tests: project_has_its_own_git_directory_and_environ, unified_app_allows_direct_writes_and_imports, pinokio_menu_is_independent_and_has_a_real_st |  |
| `test_standalone_install.py` | 15 | Standalone install integrity: real (non-linked), gitignored local directories and in-sync npm manifests. | resume |

#### Group S. Test support / manual acceptance modules (non-collected helpers)  (13 files, 0 tests)

| File | #tests | Purpose (docstring or first test names) | Flags |
|---|--:|---|---|
| `audio_world.py` | 0 | Synthetic speech with KNOWN structure, shared by the director's unit and integration tests. |  |
| `identity_world.py` | 0 | Synthetic ground-truth world for the Stage 3 identity-asset tests (test-only; never imported by production code). | yaw,mesh,freq,resume |
| `manual_stage02_acceptance.py` | 0 | Explicit real-model/media acceptance campaign, separate from offline fixtures. | resume |
| `manual_stage03_acceptance.py` | 0 | Rehash every installed registry artifact through the production local provider. |  |
| `manual_stage04_acceptance.py` | 0 | Stage 04 acceptance campaign: FaceSet reader, pose selector, and V2 schema. | yaw |
| `manual_stage06_acceptance.py` | 0 | Real 100-frame reference campaign; all media and biometric data remain local. | yaw,cli |
| `manual_stage07_acceptance.py` | 0 | Capture real Stage 06 golden tensors; qualify providers and measure p50/p95. | yaw,cli |
| `manual_stage08_acceptance.py` | 0 | Prepare owned, local-media browser fixtures or verify its real rendered output. | yaw,cli |
| `model_governance_support.py` | 0 | Helpers for real-weight tests gated by current governance evidence. |  |
| `stage15_media.py` | 0 | Real-codec fixtures for Stage 15. No mocks: every file is encoded by libx264 / libx265 / FFV1 and |  |
| `swap_core_support.py` | 0 | Synthetic scenes with known ground truth for the Stage 17 swap-core tests (no model is loaded). | freq |
| `target_world.py` | 0 | Scripted target performer for the Stage 4 reenactment tests (test-only; never imported by production code). | yaw,scenecut,resume |
| `tracking_world.py` | 0 | Test-only helpers for the head-tracking tests (never imported by production code). | yaw,scenecut,mesh |

### 7. UI test inventory (Vitest `src/*.test.*`, Playwright `e2e/`, harness)

Static count of `it(`/`test(` calls: Vitest/mock-server **85 files / 859 cases**; Playwright e2e **17 files / 112 cases** (+ `acceptance/stage8.real.ts`, 2).  `it.each` not expanded.

| File | #cases | First describe / first test titles |
|---|--:|---|
| `acceptance/stage8.real.ts` | 2 | offline production UI: local imports → identity/asset → preview → proj |
| `e2e/certification.e2e.ts` | 3 | the route table is what the browser actually serves ; every screen mounts and is reachable, one at a time, wit |
| `e2e/faceBank.e2e.ts` | 3 | persistent FACE pills, click-to-lock, live angle capture, hover inspec ; clicking a face box on the canvas tog |
| `e2e/facesetBuilder.e2e.ts` | 1 | builder renders the 3D sphere, curates angles and saves on desktop and |
| `e2e/finalWorkstation.e2e.ts` | 2 | the stream overlap tab draws the measured lanes, the copy verdict and  ; the sweep compares enhancers across r |
| `e2e/identityLock.e2e.ts` | 3 | contact sheet shows thumbnails, visible ranges, confidence and withhel ; explicit lock is sent with the render |
| `e2e/liveStream.e2e.ts` | 2 | binary frames reach the canvas with no React re-renders, and the newes ; an unmounted page leaves no open sock |
| `e2e/outputCompare.e2e.ts` | 8 | Before / After output player :: Before / After output player ; opens when the job finishes and loads both videos from the right URLs |
| `e2e/resourceLeaks.e2e.ts` | 2 | WebGL comparison canvas: 60 mount/unmount cycles leave zero live conte ; whole application: 15 laps of every s |
| `e2e/stage19.e2e.ts` | 31 | first-run onboarding and guided project :: first-run onboarding and guided project ; welcome dialog is accessible, traps focus, closes with Escape and is |
| `e2e/stage24.e2e.ts` | 13 | persistent queue (#/processing) :: persistent queue (#/processing) ; shows each job\ |
| `e2e/stage5Scrub.e2e.ts` | 7 | loading a target fetches info and the filmstrip, and the filmstrip sho ; the playhead frame is requested witho |
| `e2e/studio.e2e.ts` | 7 | four panels, timeline lanes and a FaceSet dragged onto a target track  ; clicking a face in the video selects  |
| `e2e/swapPreviewAudit.e2e.ts` | 6 | audit: fresh target inspect and dropdown assignment drives live swap p ; audit: classic layout assigns source  |
| `e2e/telemetry.e2e.ts` | 2 | a 30 Hz /ws/telemetry stream re-renders nothing in the parameter sideb ; the floating PiP preview stays visibl |
| `e2e/webgl.e2e.ts` | 8 | compiles and links both programs on a real WebGL2 context ; split wipe shows A left of the divider and B right |
| `e2e/workstation3d.e2e.ts` | 7 | the pipeline node flow bar shows eight stages with measured latencies  ; the coverage radar lists the missing  |
| `e2e/workstationControls3d.e2e.ts` | 7 | the buffer switcher renders wireframe, albedo, normals, depth and SH-r ; the SH-relit buffer follows the light |
| `mock-server/mockServer.test.ts` | 3 | /api/meta identifies the mock and every response is marked ; hardware, telemetry and health are served under u |
| `src/activeModelReadback.test.tsx` | 6 | active model read-back :: shows the exact weight file each stage has in VRAM ; says a newly selected model has not loaded yet, and never |
| `src/appShell.test.tsx` | 3 | mounts the integrated application shell with all nine screens and keep ; undoes parameter changes and mask str |
| `src/batchAndHistory.test.tsx` | 7 | BatchMatrixScreen :: renders all 4 batch strategies and allows switching tabs ; supports automatic video segment splitting toggle a |
| `src/chaptersAndCanvas.test.tsx` | 17 | chapter list operations :: adds chapters in time order with default names, clamped into the clip ; ignores a second marker within one fra |
| `src/coverageRadar.test.tsx` | 17 | radar grid model :: bins on the exact backend 12 x 8 grid, clamped at the edges ; prefers the server bin_index over re-binning, an |
| `src/diagnosticsSuite.test.tsx` | 7 | Pre-Flight Render Linter Modal :: renders pre-flight checks and dispatches queue confirmation ; shows the seven measured stages from live timing |
| `src/engineStatusBar.test.tsx` | 4 | EngineStatusBar :: shows the live detector, swapper, enhancer (verified) and precision re ; says so instead of inventing a model  |
| `src/expressionEditor.test.tsx` | 5 | shows every requested 3D edit, scopes values and reset to the selected ; edits gaze and jaw per face, and the  |
| `src/faceBank.test.tsx` | 10 | faceIdentity :: gives each persistent id its own colour and an upper-case tag; unident ; reports each scan state only from the |
| `src/faceContextMenu.test.tsx` | 14 | FaceContextMenu keyboard and extraction hook :: right-clicking a face opens a menu with the three extraction actions a ; offers per-face canvas toggles (Ultra |
| `src/faceFromFrame.test.tsx` | 3 | Use Face From Frame :: posts the current frame and face number, then hands the new FaceSet to ; reports a failed extraction without s |
| `src/faceManager.test.tsx` | 7 | FaceSet Builder Utilities & Algorithms :: computes CRC-32 checksums accurately ; builds a valid ZIP container without external dependencies |
| `src/faceOverlayHud.test.tsx` | 11 | pose helpers (standard frame: yaw + = nose right, pitch + =  :: converts the server\ ; uses the same inclusive 3×3 cell edges as the backend (±15° yaw, ±10°  |
| `src/faceSelection.test.tsx` | 12 | useFaceSelectionStore state architecture and clustering :: computes cosine distance and similarity correctly ; clusters identities by cosine distance similarity threshol |
| `src/faceset3dViewer.test.tsx` | 21 | mesh frame conversion :: turns the backend mesh frame (+y down) into an upright, outward-facing ; swaps triangle winding for a mirrored |
| `src/facesetBuilder.test.tsx` | 1 | edits the active set, filters 96 bins, removes a crop, and activates t |
| `src/facesetFavorites.test.tsx` | 2 | FaceSet library favorites :: persists favorites, sorts them first, and does not select on favorite  ; exposes a favorite control in the com |
| `src/facesetInspector3D.test.tsx` | 3 | shows the radar for the stored views (pitch converted to the standard  ; uses the facial reconstruction regard |
| `src/finalWorkstation.test.tsx` | 15 | sweep patch semantics :: GPEN Realistic and UltraMax select the real restorer, keep a non-zero  ; sweeping relight strength switches re |
| `src/glbMesh.test.ts` | 16 | parseGlb on the server\ :: reads vertices in the viewer frame: the project\ ; keeps triangle winding (the serializer already swapped it f |
| `src/guardedCanvas.test.tsx` | 9 | GuardedCanvas :: draws the canvas when WebGL works ; shows the fallback with the browser\ |
| `src/hairNeckProtection.test.tsx` | 14 | protection helpers keep untouched requests byte-identical :: removes a control that is set back to its engine default ; the hairline inset is the engine offset with the si |
| `src/headMesh.test.ts` | 17 | prepareHeadMesh :: centres the mesh in the viewer frame, finds the nose tip and keeps uni ; converts the backend mesh and OpenCV  |
| `src/historyCurvesRoles.test.tsx` | 9 | uses reversible commands, discards redo branches, and coalesces a para ; evaluates linear, Bezier and hold cur |
| `src/identityLock.test.tsx` | 11 | contact-sheet formatting :: formats clock, visible ranges and withheld reasons ; shows thumbnails, track number, confidence, gaps and with |
| `src/lightProbeSphere.test.tsx` | 2 | LightProbeSphere :: renders light probe with auto-detected direction and color temperature ; allows switching to Manual 3D Virtual |
| `src/liveStream.test.tsx` | 12 | live frame wire format :: parses the 16-byte header: index, timestamp, three stage latencies, fl ; rejects empty payloads and foreign pr |
| `src/liveStudio.test.tsx` | 1 | starts and stops with the selected identity and keeps unavailable outp |
| `src/liveSwapPreview.test.tsx` | 14 | payload + filenames :: builds frame-accurate payloads from the job request and parameters ; names downloads roop3d_preview_<ISO times |
| `src/modelHubModal.test.tsx` | 28 | manifest parsing :: turns a row into a fully defaulted model and never throws on missing f ; rejects a response that is not a mani |
| `src/modelRack.test.tsx` | 14 | model status :: maps what is on disk to the five pills the workstation shows ; lists every registered model, installed or not, |
| `src/navigationAndControls.test.tsx` | 8 | keeps all nine deep-linked screens mounted and preserves local edits a ; evaluates arithmetic safely, supports |
| `src/outputComparePlayer.test.tsx` | 10 | points the two videos at the original target and the verified download ; keeps the original silent and the res |
| `src/pipelineFlowBar.test.tsx` | 4 | maps each stage to the measured latency fields, and reports unmeasured ; resolves the Normals chip to the norm |
| `src/pipelinePill.test.tsx` | 10 | pill model :: maps the three server stages onto Pre-Scan, Range Lock and Render & Mu ; treats a checkpoint-restored step as  |
| `src/pipelineSolo.test.tsx` | 6 | solos a pipeline node through POST /api/preview solo_node, returns to  ; offers exactly the eight pipeline sta |
| `src/poseViewer.test.tsx` | 3 | Fibonacci Sphere Lattice & View Coverage Heatmap :: generates an evenly distributed Fibonacci sphere lattice ; projects [yaw, pitch] view angles to appropriate la |
| `src/postproduction.test.tsx` | 6 | frame-accurate timeline interchange :: splits all boundaries, preserves gaps and offsets trimmed media frames ; writes rational XML time values, esca |
| `src/previewBox.test.tsx` | 8 | CompositionGridOverlay Component :: returns null when mode is off or empty ; renders Rule of Thirds SVG guides with golden power points |
| `src/previewFrameCache.test.ts` | 3 | preview frame cache :: evicts least recently used frames by count and encoded byte budget ; expires without extending freshness on re |
| `src/queueAndHud.test.tsx` | 6 | Hardware Diagnostics & HUD Logic (PipelinePill) :: correctly resolves hardware execution provider from telemetry ; identifies real-time pipeline bottlenecks base |
| `src/readinessChecks.test.tsx` | 10 | pre-flight linter modal :: sends the swapper and the export passes so the TensorRT and disk estim ; does not re-request the lint on every |
| `src/routes.test.tsx` | 4 | defines exactly the nine deep-linked routes with their names ; renders each of the nine routes: one visible sc |
| `src/runtimeReport.test.tsx` | 22 | trt_compilation_progress frames :: parses a build frame, clears on finished, and ignores everything malfo ; shows a live strip while compiling, w |
| `src/safeData.test.tsx` | 8 | fileName / shortId :: takes the last segment for either separator and survives non-strings ; fills every object the list renderers d |
| `src/sessionReload.test.tsx` | 7 | rebuilds the workstation from the saved session after the page reloads ; keeps saving the session as it change |
| `src/settingsAndSweep.test.tsx` | 17 | settings registry :: documents every setting with sane bounds, a recommended range inside t ; agrees with the application\ |
| `src/settingsDomainTabs.test.tsx` | 5 | SettingsInspector Phase 3 Domain Tabs & Validation :: renders all domain category tabs and defaults to Compute tab ; switches tabs on click and displays relevant co |
| `src/shell/shell.test.tsx` | 4 | command palette: labelled modal dialog, filters by label/group/keyword ; command palette: arrow keys move the  |
| `src/spatialInspector.test.tsx` | 6 | Tanner Helland Kelvin to RGB conversion :: converts warm 2700K incandescent light to warm amber hex ; converts neutral 5500K daylight to neutral white-bl |
| `src/stage11Canvas.test.tsx` | 23 | colormaps :: Inferno and Viridis hit their published endpoints and the LUT is 256 o ; Inferno brightens monotonically (so a |
| `src/stage11Math.test.tsx` | 7 | evaluateExpression :: evaluates ordinary arithmetic: 0.4 * 1.5 -> 0.6 ; treats a leading + * / as relative to the field value, so  |
| `src/stage11Model.test.tsx` | 21 | atomic stores :: notifies only on a real change and supports transient (render-free) se ; re-renders a subscriber only when its |
| `src/stage11Perf.test.tsx` | 4 | 30 Hz telemetry isolation :: re-renders none of the parameter sidebar, timeline, Face Manager, Vide ; control: an App state change DOES re- |
| `src/stage12Coords.test.ts` | 5 | Stage 12 sub-pixel canvas <-> video coordinates :: maps the frame corners onto the letterboxed rectangle (object-fit: con ; round-trips arbitrary sub-pixel point |
| `src/stage12Diagnostics.test.tsx` | 4 | Stage 12 Diagnostics & Environment-Health Suite UI :: renders Live GPU/VRAM HUD with memory pressure gauge and purge button ; renders Standing Environment-Health Ca |
| `src/stage12TargetLock.test.tsx` | 7 | Stage 12 pose pill and axes :: formats one decimal with an explicit sign and never -0.0 ; points the facing axis toward image right for + yaw |
| `src/stage13ModelHub.test.tsx` | 5 | Stage 6 local model hub :: connects engine selection to the shared job preference and blocks unav ; shows the active production engine an |
| `src/stage13Resilience.test.tsx` | 27 | WorkstationErrorBoundary :: shows a recoverable message instead of a blank page and retries on dem ; clears itself when a reset key change |
| `src/stage18Resources.test.tsx` | 8 | resource ledger accounting :: counts object URLs created, revoked and live, and ignores revoking an  ; counts live WebSockets by path and fo |
| `src/stage19.test.ts` | 28 | error guidance :: maps the real backend codes to a title, a cause and at least one step ; falls back by code family, by a code n |
| `src/stage20Shell.test.tsx` | 12 | deep-link routing :: maps the five Stage 20 links onto the existing screens and keeps every ; keeps the deep link in the address ba |
| `src/stage23.test.ts` | 5 | mask frame ranges :: lets a painted range win over the whole-clip mask only inside the rang ; keeps ranges sorted and never overlap |
| `src/stage23Ui.test.tsx` | 13 | Stage 23 canvas inspection modes :: offers all six modes: wipe, crossfade/blend, thermal diff, loupe and t ; keeps the 3D mesh overlay usable befo |
| `src/stage24.test.tsx` | 29 | timeline dock maths :: places times on the track and never yields NaN% ; measures the ruler in m:ss.f and h:mm:ss |
| `src/stage5Scrub.test.tsx` | 21 | FrameScrubber: latest frame wins, one request in flight :: never has two requests in flight and always ends on the newest wanted  ; asks for faces only once the playhead |
| `src/stage7.test.tsx` | 10 | Pre-flight linter one-click fixes :: runs “Convert & Optimize Now”, reports the result and re-lints ; surfaces a conversion that did not complete a |
| `src/stage7Studio.test.tsx` | 16 | FaceSet drag payload :: accepts only a well-formed FaceSet drag and ignores files or text ; bins stored poses into the yaw x pitch ind |
| `src/stage8.test.tsx` | 2 | preserves discovery rank, real embeddings, merges and cosine ordering ; uses the same cubic Bezier as the back |
| `src/stage8Studio.test.tsx` | 2 | loads durable bank assignments and reloads saved archive cells after f ; sends CodeFormer and the identity pai |
| `src/stage8Workflow.test.tsx` | 3 | uses the server queue even when stale completed jobs exist in localSto ; offers legal controls and exports onl |
| `src/streamOverlap.test.tsx` | 10 | lane model :: lays every stage on one shared time axis in execution order with the r ; draws nothing until a frame has been  |
| `src/studioTools.test.ts` | 11 | physical camera preview maths :: is the field of view of a 36 x 24 mm sensor ; clamps to the 18-135 mm range the matcher supports and survives  |
| `src/swapScope.test.tsx` | 10 | swap scope helpers keep untouched requests byte-identical :: lists the three scopes with the full craniofacial lock first and as th ; picking the default removes both keys |
| `src/themesAndFacesetAndTimeline.test.tsx` | 11 | Themes & Styles Engine (37 Studio Themes) :: contains all 37 studio themes across dark, AMOLED, and light families ; resolves themeByName correctly with fa |
| `src/timelineInteraction.test.tsx` | 4 | timeline navigation interactions :: zooms around the pointer without moving the point under it ; selects the actual frame shown in a filmstrip car |
| `src/timelineWorkstation.test.tsx` | 11 | Timeline Utility Functions :: formats timecode into SMPTE HH:MM:SS:FF accurately ; parses timecode strings and frame numbers into seconds |
| `src/uiLayouts.test.tsx` | 33 | foundation utilities :: cn joins strings, arrays and condition maps and drops falsy values ; clamp and decimalsOf |
| `src/viewportTools.test.ts` | 17 | buffer views :: lists the five requested buffers plus the default shaded view, each wi ; reproduces a clamped-cosine key light |
| `src/workflowPanel.test.tsx` | 7 | RenderFooter :: shows live progress and fps, and a GPU pill once telemetry arrives ; start is gated by its reason, the frame b |
| `src/workstation.test.tsx` | 9 | capability and config gates :: fails closed before capability load and for unknown features/options ; renders schema controls with unavailabl |
| `src/workstation3dPayload.test.ts` | 2 | opt-in 3D workstation controls in the live preview payload :: omits the hairline offset and ambient fill until they are touched, so  ; sends the hairline offset once set, a |
| `src/workstationCanvas.test.tsx` | 5 | WorkstationCanvas Comparison Modes :: renders 4 preview modes with active wipe controls ; renders diff map layer when mode is diff |
| `src/workstationControls3D.test.tsx` | 15 | skin tone vs ambient adaptation :: drives only the colour-transfer weight ; says so when colour matching is bypassed, and never claims to be the  |

### 8. `app/roop_3d_ui/` — every non-test file (one row each)

Columns: lines · key exports · backend paths referenced in the file (static regex; `/api/v1` prefix is added by `api()` at `api.ts:14`) · purpose (first doc-comment, truncated).


##### (ui root: config / entry)  (9 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `browserPaths.ts` | 28 | appRoot, ownedBrowserPath, localBrowserTemp |  | Keep browser profiles, downloads and test evidence inside this checkout; reject linked ancestors. |
| `eslint.config.js` | 52 |  |  | Syntax-level lint for .js/.jsx/.ts/.tsx. typescript-eslint cannot be used: TypeScript 7 ships no compiler JS A |
| `index.html` | 2 |  |  |  |
| `package-lock.json` | – | – | – | asset/lock/binary (230742 B) |
| `package.json` | 54 |  |  |  |
| `playwright.config.ts` | 26 |  |  WS |  |
| `playwright.stage8.config.ts` | 14 |  |  |  |
| `tsconfig.json` | 4 |  |  |  |
| `vite.config.ts` | 9 |  | /api |  |

##### e2e  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `facesetBuilderHarness.tsx` | 4 |  |  |  |
| `mockApi.ts` | 155 | ID, TARGET, CAPABILITIES |  /config /facesets /jobs WS | Two faces of the mocked 960x540 frame, in native pixels: what GET /media/frame?detect_faces=true returns. |
| `slider.ts` | 22 | sliderValue, setSlider |  | Current value of a PrecisionRangeSlider thumb (`role="slider"`), read from its ARIA state: they are not <input |

##### harness  (6 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `gl.html` | 12 |  |  |  |
| `gl.tsx` | 47 |  |  | Left half pure red, right half pure green; optional exact-pixel patch so differences are known to the unit. |
| `leak.html` | 12 |  |  |  |
| `leak.tsx` | 47 |  |  | Stage 18 harness (used only by e2e/resourceLeaks.e2e.ts): mounts and unmounts the real WebGL2 comparison canva |
| `livestream.html` | 12 |  |  |  |
| `livestream.tsx` | 14 |  |  | Harness for e2e/liveStream.e2e.ts: mounts the real LiveStreamCanvas and counts how many times React renders it |

##### mock-server  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `dev.ts` | 18 |  | /api /ws | `npm run dev:mock`: the mock backend plus Vite HMR on 5174, proxied to it. No GPU, models or Python needed. |
| `server.ts` | 86 | MOCK_HEADER, mockHardware, mockFrame |  /health /system/hardware /system/telemetry WS |  |

##### src/ (top level)  (27 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `App.tsx` | 832 | App | /config /faceset/3d-state /faceset/reconstruct-3d /facesets | per-session only |
| `SettingsInspector.tsx` | 297 | SettingsInspector |  |  |
| `ThreeViewport.tsx` | 57 | ThreeViewport, PoseCoverageInspector |  |  |
| `VideoViewport.tsx` | 1161 | VideoViewport, formatTime | /media/preview-frame /media/detect-frame-faces | Snapshot slots keep a self-contained image, so convert the preview PNG before its object URL is revoked. |
| `api.ts` | 123 | ApiError, api, post | /api/preview  /media/content /target/add-angle-sample | An error from the local service. It is still a plain `Error` (same `message` as before), but keeps the machine |
| `contracts.ts` | 210 | Schema, Config, Feature |  | Render preset (see /media/export-profiles). Omitted = the standard MP4 default. |
| `events.ts` | 35 | EventConnection |  WS |  |
| `frameScrub.ts` | 309 | FrameFace, FrameMeta, MediaInfo | /media/filmstrip /media/frame /media/info | Left to right rank in the frame: the `faceRank` face extraction takes. |
| `index.css` | 70 |  |  | deep carbon surfaces |
| `livePreview.ts` | 184 | LIVE_DEBOUNCE_MS, JOB_START_EVENT, RENDER_FRAME_EVENT | /api/preview | Slider drags coalesce into one request per pause of this length. |
| `liveStream.ts` | 168 | HEADER_BYTES, PROTOCOL_VERSION, FrameSource | /api/preview /ws/v1/preview/stream WS | Wall latency of pipeline stages 1 (detect/pose), 2 (swap/mesh) and 3 (enhance/blend) for this frame, ms. |
| `main.tsx` | 27 |  |  WS |  |
| `motion.tsx` | 71 | spring, EASE_SETTLE, fadeUp |  | Shared motion vocabulary (ported from Roop Ultimate's react-ui): one set of springs and entrance variants so |
| `previewFrameCache.ts` | 35 | PreviewFrameCache |  | Short-lived, CPU-only PNG cache. Never retains object URLs or decoded GPU textures. |
| `previewPreferences.ts` | 15 | ViewportPreferences, DEFAULT_VIEWPORT, ProjectPreview |  |  |
| `runtimeSettings.ts` | 2 | RUNTIME_SETTINGS_CHANGED |  | A persisted runtime change invalidates both detection and rendered frame caches. |
| `sessionSnapshot.ts` | 86 | SESSION_KEY, SessionSnapshot, parseSession |  | Validate untrusted storage content field by field; anything malformed falls back to an empty value. |
| `sliderTestKit.ts` | 20 | thumb, setSlider, sliderValue |  | Test helpers for PrecisionRangeSlider (role="slider" thumbs, not <input type="range">). They drive it the way  |
| `stage19.css` | 196 |  |  | ---- accessibility primitives ------------------------------------------------------------------------------ |
| `stage23.css` | 21 |  |  | Stage 23: multi-person actors, frame-range mask scope, 3D mesh canvas mode. Layers over the earlier sheets. |
| `stage24.css` | 151 |  |  | Stage 24: liquid glass HUDs, drag-to-reorder lists, Embla filmstrip dock, persistent queue cards.    Plain CSS |
| `state.ts` | 28 | featureReason, optionReason, jobActions |  |  |
| `styles.css` | 2921 |  |  | reduced motion is handled in stage19.css (system setting + the user's own override) |
| `themeVars.ts` | 159 | hexToRgb, rgbToHex, mix |  | ── Custom theme derivation matching roop-ultimate ───────────────────────────── |
| `themes.ts` | 95 | ThemeDefinition, THEMES, THEME_CLASSES |  | measurement is best-effort; the theme still applies |
| `useFaceSelectionStore.ts` | 557 | FaceSelectionStoreState, cosineSimilarity, cosineDistance |  | Keeps a stored face's pose current with the frame it is on screen in (the 3D viewer's Live Pose reads it). |
| `workstation3d.css` | 136 |  |  | ---- Pipeline node flow bar ---------------------------------------------------------------------------------- |

##### src/__fixtures__  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `fixture_head.glb` | – | – | – | asset/lock/binary (2100 B) |

##### src/components/(top)  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `LiveStreamCanvas.tsx` | 74 | LIVE_STALE_MS, LiveStreamCanvas |  WS | The canvas hides itself again if no frame has arrived for this long (render finished, or the stream dropped). |
| `TargetFaceCard.jsx` | 2 |  |  |  |
| `ThemeGallery.tsx` | 89 | ThemeCardProps, ThemeCard, ThemeGalleryProps |  | One swatch card. Colours come from the theme itself (inline), everything else from shell.css `.theme-card `. |

##### src/components/canvas  (15 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CanvasHuds.tsx` | 78 | ViewportTelemetryPillProps, ViewportTelemetryPill, BeforeAfterLoupeHudProps |  | Floating Stage 24 HUDs for the full-bleed canvas, drawn on liquid glass. All of them are optional extras over  |
| `CompositionGridOverlay.jsx` | 85 | CompositionGridOverlay |  | Vertical grid lines |
| `FaceContextMenu.tsx` | 140 | FaceDestination, TargetAction, TargetActionExtra |  | Stage 12 target actions, shown when the host provides `onTargetAction`. |
| `FaceOverlayCanvas.jsx` | 874 | FACE_ROLES, FaceOverlayCanvas, signed1 |  | Quick target face selector strip on top of preview box |
| `MaskBrushOverlay.jsx` | 589 | MaskBrushOverlay |  | Visible tinted painting canvas |
| `WebGLViewport.jsx` | 195 | WebGLViewport |  | Native-resolution texture: the image is uploaded at its own size, only downscaled if the GPU cannot hold it. |
| `WorkstationCanvas.jsx` | 409 | MODES, clientToMedia, diffHeatmap |  | Pointer â†’ media pixel coordinates with sub-pixel precision. `stage` is the on-screen rectangle (client px) t |
| `faceFlags.ts` | 25 | Pose6, faceFlags, faceKey |  | Frozen 6DoF poses of locked faces (standard frame), by face key. The overlay holds the wireframe / gimbal / ga |
| `headMesh.js` | 258 | MAX_DRAWN_EDGES, prepareHeadMesh, poseRotation |  | Per source coordinate frame: the sign applied to (x, y, z) to reach the viewer frame (see PoseGeometryViewer F |
| `loupeCrop.ts` | 22 | Stage, Pointer, CropRect |  | Pure geometry for the before / after loupe HUD. |
| `poseHud.js` | 57 | facePose, LANDMARK_COLORS, drawPoseHud |  | Standard-frame pose of a detected face (prefers the server's `pose3d`; otherwise converts the legacy `pose`). |
| `useFaceExtraction.ts` | 67 | Toast, useFaceExtraction |  | Active source FaceSet id: the FaceSet "Add to .fsz Faceset" extends. |
| `useHeadMesh.ts` | 77 | PreparedHead, HeadMeshState, clearHeadMeshCache | /faceset/3d-state | The fitted anatomical face, an explicitly labelled orientation proxy, or nothing. |
| `useTargetLock.ts` | 188 | useTargetLock |  | Local media path of the target video; changing it drops the locks (locks belong to one video). |
| `webglModel.js` | 252 | MODE_IDS, LOUPE_ZOOM, LOUPE_ZOOMS |  | The pore magnifier runs at 3.5x or 4x (Stage 23); 3.5x stays the default so earlier behaviour is unchanged. |

##### src/components/common  (4 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ToastHost.tsx` | 21 | ToastHost |  | Renders the app-wide toasts. Errors are `role="alert"` (announced at once); the rest are polite status message |
| `VirtualList.tsx` | 147 | WindowInput, WindowResult, windowRange |  | Which rows [first, last] must be mounted for the visible band, where the first mounted row starts, and the ful |
| `WorkstationErrorBoundary.tsx` | 55 | BoundaryContext, WorkstationErrorBoundaryProps, WorkstationErrorBoundary |  | Short name shown in the message and the log (`3D viewport`, `Workstation`). |
| `toastHost.css` | 8 |  |  |  |

##### src/components/controls  (6 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `Hotkeys.tsx` | 116 | DEFAULT_KEYS, KEY_LABELS, KEY_GROUPS |  | Display grouping for the remapper UI; every action in DEFAULT_KEYS appears in exactly one group. |
| `Modal.tsx` | 19 | Modal |  |  |
| `NumericInput.tsx` | 66 | evaluateExpression, constrain, scrubDelta |  |  |
| `SettingControl.tsx` | 47 | SettingControl |  | Why the control is unavailable right now (shown in the help and as the disabled tooltip). |
| `settingsRegistry.ts` | 67 | PanelId, SettingId, SettingSpec |  | Range that usually looks right, shown in the help and drawn on the slider's help. |
| `useDialogA11y.ts` | 49 | focusableIn, useDialogA11y |  | Visible, enabled, tabbable controls inside `root`, in DOM order. Exported so tests can assert the trap's contr |

##### src/components/diagnostics  (10 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BottleneckFlamegraph.jsx` | 423 | PIPELINE_STAGES, DEFAULT_TEMPORAL_SETTINGS, BottleneckFlamegraph |  | Top Metrics Row |
| `PipelineSaturationPanel.jsx` | 80 | usePipelineMetrics, PipelineSaturationPanel | /system/pipeline | Polls the server while mounted (the HUD popover is open); nothing is requested when it is closed. |
| `PreFlightLinterModal.jsx` | 239 | PreFlightLinterModal | /system/preflight-lint |  |
| `RuntimeReportPanel.tsx` | 140 | RuntimeReportPanel | /system/runtime-report | Mirror of the Pinokio terminal's 14-section runtime report, read from the same collectors (`GET /system/runtim |
| `StreamOverlapPanel.jsx` | 88 | StreamOverlapPanel |  | Per-frame stage lanes on one time axis (so overlap is visible), the PCIe / copy picture and the frame ring buf |
| `batchLint.js` | 60 | SKIP_KEY, lintRequestFor, lintBatch | /system/preflight-lint | Pre-flight linting for batch submission: one request per distinct target video, advisory unless a check FAILS. |
| `lanesModel.js` | 83 | LANE_ORDER, LANES, buildLaneModel |  | Pure shaping of the /api/v1/system/pipeline payload into the stream-overlap view. |
| `pipelineMetrics.js` | 54 | pickPipeline, summarizePipeline, saturationVerdict |  | Pure helpers for the saturation panel: shape the /api/v1/system/pipeline payload into display rows. |
| `runtimeReport.css` | 31 |  |  | Narrow inspector: label above value, so long values (paths, provider chains) wrap instead of overflowing. |
| `stageTimings.js` | 56 | FLAME_STAGES, resolveStageTimings |  | Seven measured pipeline stages shown by the bottleneck flamegraph. Mirrors roop_3d/core/stage_timing.py. |

##### src/components/export  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ExportPresetBar.tsx` | 59 | ExportPresetBar |  |  |
| `VfxExportPanel.jsx` | 95 | DEFAULT_EXPORT, FinishingControls, VfxExportPanel |  | Offered by the interface but rejected by the job schema (jobs/schema.py accepts alpha, occluder, uv, normal, d |
| `exportOptions.ts` | 39 | ExportProfileInfo, normalizeExport, useExportProfiles | /media/export-profiles | Drop every Stage 15 field that equals the server default, so "nothing chosen" serialises exactly like the orig |

##### src/components/facemgr  (10 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BuilderCoverageSphere.tsx` | 23 | BuilderCoverageSphere |  | The same 12 x 8 frontal hemisphere used by Stage 3, with selectable missing bins. |
| `FaceManager.jsx` | 692 | extractFacesFromInspection, FaceManager | /facesets/build | The 3x3 pose matrix as ARIA rows of three cells, so `role=grid` has the `row` children it requires. |
| `FacesetBuilder.css` | 3 |  |  |  |
| `FacesetBuilder.tsx` | 117 | FacesetBuilder | /faceset/extract-from-media /faceset/save-fsz /faceset/synthesize-36 /system/capabilities |  |
| `FacesetInspector3D.tsx` | 90 | FacesetInspector3D | /faceset/3d-state /faceset/reconstruct-3d | Fetch `path` (null = nothing); a late answer for an earlier FaceSet or head model is dropped. `revision` force |
| `FacesetLibrary.tsx` | 562 | FacesetLibraryProps, FacesetLibrary | /facesets/open-folder /facesets/rebuild-thumbs | `dropdown` (default) is the original compact picker. `list` (Stage 7 studio identity panel) keeps every FaceSe |
| `IdentityLockPanel.tsx` | 124 | clock, rangeSummary, withheldSummary | /identity/locks | Seconds -> m:ss without importing the viewport (keeps this panel dependency-free). |
| `PhotoSynthesisAction.tsx` | 39 | PhotoSynthesisAction | /faceset/synthesize-photos | Direct action for the measured photos already imported in #/facemgr. |
| `faceBank.js` | 23 | rankClusters, poseBin, poseGrid |  | Keep discovery ranks and original biometric measurements when manually merging tracks. |
| `fszBuilder.js` | 447 | crc32, buildZip, POSE_BUCKETS |  | Pure JavaScript utility for .fsz FaceSet archive creation, DBSCAN ArcFace clustering, and 5-bucket pose classi |

##### src/components/faceswap  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `PipelinePill.jsx` | 70 | TELEMETRY_PATH, PipelinePill | /ws/telemetry |  |
| `pillModel.js` | 75 | STEPS, isPipelineFrame, elapsedSeconds | /ws/telemetry | View model for the 3-step run-bar pill. Pure functions of the server's `pipeline.status` frames (`/ws/telemetr |

##### src/components/guide  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AccessibilityControls.tsx` | 34 | ZoomControl, AccessibilityControls |  | Display & accessibility preferences. "Reduce motion" stops the interface animations and the 3D viewers' idle r |
| `DialogFrame.tsx` | 25 | DialogFrame |  | The one accessible modal shell: labelled `role="dialog"`, focus moved in and trapped, Escape closes, focus res |
| `DialogPanel.tsx` | 14 | DialogPanel |  | The dialog surface only (no backdrop), for dialogs that already render their own backdrop: labelled, modal, fo |
| `EngineExplainer.tsx` | 22 | EngineExplainer |  | Explains the engine list in the Inspector: which engine Roop 3D recommends here and why, what the chosen engin |
| `GuideDialog.tsx` | 124 | GUIDE_TABS, GuideTab, GuideDialog |  |  |
| `HardwareExplainer.tsx` | 27 | HardwareExplainer |  | What the active hardware profile means for this computer, in words: what Roop 3D will use, what that changes,  |
| `NextStepBar.tsx` | 24 | NextStepBar |  |  |
| `RecoveryNotice.tsx` | 25 | RecoveryNotice |  | The error banner body: what failed, the first thing to try, and (expandable) the cause and every step. Unknown |
| `WelcomeDialog.tsx` | 31 | WelcomeDialog |  | Shown once, on the first launch, over the Home screen. It tells a new person in four sentences what the app is |
| `WorkflowChecklist.tsx` | 33 | WorkflowChecklist |  | The workflow as a live checklist. Every state is derived from real application state (see workflowState.ts), s |
| `coverageGuide.ts` | 41 | YAW_COLUMNS, CoverageReading, describeCoverage |  | One or two sentences a person can act on. |
| `errorGuide.ts` | 105 | ErrorGuide, ERROR_GUIDES, isOfflineError |  | Section of docs/COMMON_PROBLEMS.md that expands on this. |
| `helpContent.ts` | 185 | WorkflowStepId, WorkflowStep, WORKFLOW_STEPS |  | .md page; the Stage 19 tests pin the numbers that appear in the user-facing text. |
| `workflowState.ts` | 53 | StepState, WorkflowInputs, WorkflowProgress |  | A project has been saved (or opened) for this work. |

##### src/components/hud  (3 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `EngineStatusBar.tsx` | 89 | EngineStatusBar |  |  |
| `PipelinePill.jsx` | 328 | PIPELINE_STAGES, resolveProvider, computeBottleneck | /ws/telemetry WS | Standard 5-stage facial transformation pipeline stages. |
| `engineStatus.css` | 23 |  |  | Tablet widths: the bar would push the preview below the fold. It stays mounted (polling, failure toasts and th |

##### src/components/modelhub  (11 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ActiveModelBadge.tsx` | 47 | ActiveModelBadge |  | Model ids the registry uses for the rack's enhancer ids, so "selected" and "loaded" compare like with like. |
| `EngineModelRack.tsx` | 229 | RackProps, EngineModelRack | /models/compile_trt | Current values of the parameters the rack edits (the workspace parameters, or the Batch defaults). |
| `ModelHubPanel.tsx` | 216 | ModelHubPanel | /models/downloads /models/hub |  |
| `ModelManager.tsx` | 129 | ModelManager | /models/compile_trt | Settings → Models & Engines Manager: rescan, offline import, integrity check, and per-model actions. Every reg |
| `ModelManagerModal.tsx` | 154 | progressView, ModelManagerModal | /ws/v1/pipeline/status | What the progress cell says for one row: live transfer first, then a partial download left on disk, then nothi |
| `ModelSelect.tsx` | 188 | RackOption, StatusPill, ModelSelect |  | One choice in a rack selector. `modelId` is the catalog artifact it needs (defaults to `id`); `null` means it  |
| `PersonModelPicker.tsx` | 17 | PersonModelPicker |  | Pins a swapper and an enhancer to one detected person. "Scene default" = no override. |
| `RequiredModelsBanner.tsx` | 69 | BANNER_DISMISS_KEY, RequiredModelsBanner |  | not remembered |
| `modelHub.css` | 60 |  |  | Model Hub modal + required-models banner (Stage 13). Tokens only: works in every theme. |
| `modelRack.css` | 48 |  |  |  |
| `openModelHub.ts` | 3 | OPEN_MODEL_HUB, openModelHub |  | Any screen can ask the app shell to open the Model Hub without prop drilling. |

##### src/components/output  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `OutputComparePlayer.tsx` | 129 | DRIFT_SECONDS, afterUrl, beforeUrl | /media/content | Beyond this many seconds apart the follower is re-seeked to the master (about three frames at 25 fps). |
| `outputCompare.css` | 26 |  |  | Before / After output player. Tokens come from the app theme (--line, --raised, --accent, --text, --muted, --d |

##### src/components/queue  (4 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BackendQueuePanel.tsx` | 113 | BackendQueuePanelProps, BackendQueuePanel |  | Poll only while the screen is showing. |
| `QueueDrawer.jsx` | 429 | QUEUE_STATES, normalizeJobState, QueueDrawer | /media/content /jobs /media/concat | Normalized queue state definitions. |
| `ResumeNote.jsx` | 20 | ResumeNote |  | Stage 9: shows where a re-run of a stopped job continues from (the last verified finished segment), if one exi |
| `backendQueue.ts` | 60 | LIFECYCLE_STATES, LifecycleState, LIFECYCLE_LABEL | /queue | Types and rules for the backend-owned persistent queue (`/api/v1/queue`). Pure: no React, no fetch. |

##### src/components/studio  (17 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ActorPanel.tsx` | 104 | Actor, ActorReport, MIXED |  | `GET /actors/{inspection_id}` (roop_3d/server/actor_routes.py): distinct people in the target, most prominent  |
| `FaceExpressionEditor.tsx` | 89 | FACE_EDIT_CONTROLS, FaceExpressionEditor |  |  |
| `FaceMapping.tsx` | 41 | FaceMappingProps, FaceMapping |  | track id -> FaceSet id (the face-selection store's assignments). |
| `FaceSetDetail.tsx` | 251 | YAW_BINS, PITCH_BINS, coverageGrid |  | Yaw columns (image left → right) and pitch rows used by the pose coverage indicator. Standard frame: + pitch = |
| `HairNeckProtection.tsx` | 73 | HairNeckProtectionProps, HairNeckProtection |  | The enhancer that will really run; the pore slider only has an effect after UltraMax. |
| `HeadModelBackbone.tsx` | 61 | BackboneRow, BackboneReport, BACKBONE_KEY |  | the choice just is not remembered |
| `PanelBoundary.tsx` | 18 | PanelBoundary |  | Keeps one studio panel's failure local: the panel shows the error and a Retry button (which remounts it), whil |
| `RenderFooter.tsx` | 54 | RenderFooterProps, RenderFooter |  | The job for the loaded target, if any (drives the progress bar). |
| `SpatialInspector.tsx` | 159 | SpatialTab, SpatialInspectorProps, SpatialInspector |  | Tab Panels: keep mounted to prevent WebGL context destruction |
| `StudioInspector.tsx` | 271 | StudioInspectorProps, ENHANCERS, activeEnhancer | /models/hub | Engines from the Stage 6 model hub; selection is the same store the Model Hub screen writes. |
| `SwapScopeSelector.tsx` | 39 | SwapScopeSelectorProps, SwapScopeSelector |  | Which part of the target's head the source geometry replaces. Real compositor option (`CompositeOptions.swap_s |
| `TargetPanel.tsx` | 117 | trackLabel, formatClock, FrameNavigator |  | Frame-accurate navigator. The readout follows the live playhead store, so playback ticks re-render only this l |
| `TrackLanes.tsx` | 106 | laneColor, trackSpans, renderFraction |  | Visible frame spans of a track: the identity sheet's segments when available, else first → last frame. |
| `WorkstationControls3D.tsx` | 253 | LIP_LOCK_KEY, SCENE_ADAPT_KEY, SMOOTH_CUTOFF_LIGHT |  | not remembered |
| `arkit52.ts` | 23 | ARKIT52_NAMES, arkitLabel, ARKIT52_GROUPS |  | Public ARKit blendshape order; values remain sparse in per-face overrides. |
| `protection.ts` | 46 | PROTECTION_DEFAULTS, ProtectionKey, PROTECTION_KEYS |  | Engine defaults of the hair / neck / ear controls (= `CompositeOptions`). A control set back to its default is |
| `swapScope.ts` | 42 | SwapScope, DEFAULT_SWAP_SCOPE, ADAPTIVE_START_BLEND |  | Share of the source skull when the user first picks the adaptive blend (the engine default of 1 would equal th |

##### src/components/three  (13 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AngularCoverageRadar.tsx` | 92 | AngularCoverageRadarProps, AngularCoverageRadar |  | The FaceSet coverage payload (`views` / `observed_pose_points`); null while none is loaded. |
| `CoverageRadarPanel.tsx` | 17 | CoverageRadarPanel |  | The coverage radar for the active source FaceSet, with the selected target's poses drawn on it (red where the  |
| `FlameMeshViewer.tsx` | 226 | FlameMeshViewerProps, FlameMeshViewer |  |  |
| `GuardedCanvas.tsx` | 113 | ContextPhase, RESTORE_TIMEOUT_MS, MAX_REBUILDS |  | After a context loss the browser has this long to hand the context back before the canvas is rebuilt. |
| `IdentityMorphPanel.tsx` | 59 | IdentityMorphPanelProps, INFLUENCE_MODES, IdentityMorphPanel |  | Whether the 3D viewer shows the target-geometry ghost and the hairline ring while these controls are edited. |
| `LightProbeSphere.tsx` | 217 | LightProbeSphere |  | The sphere is shaded by the controls themselves: the highlight sits where the key light is, its strength follo |
| `PoseGeometryViewer.jsx` | 1151 | generateFibonacciSphere, projectViewsToLattice, generatePoseGrid |  | @typedef {{ index: number; position: [number, number, number]; normal: THREE.Vector3; yawDeg: number; pitchDeg |
| `ViewportGizmos.jsx` | 157 | JawGizmo, GazeGizmo, LightGizmo |  | Invisible, larger pick target so a handle is easy to grab at any zoom. |
| `WebGLFallbackInspector.tsx` | 78 | frontProjection, WebGLFallbackInspector |  | Front-view outline of a mesh as SVG path data, in a 0..100 box (y down). Pure so it can be tested without a DO |
| `studioTools.ts` | 199 | SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM, FOCAL_RANGE_MM |  | Full-frame sensor: the reference the 18-135 mm range of `PhysicalCameraMatcher` is expressed against. |
| `useStudioEnvironments.ts` | 35 | EnvironmentsState, loadStudioEnvironments, resetStudioEnvironmentCache | /rendering/studio-environments | The five built-in CC0 studio lighting probes (`GET /rendering/studio-environments`), fetched once per page loa |
| `viewerRigs.ts` | 25 | ViewerLighting, ViewerMorph, DEFAULT_AMBIENT_FILL |  | What the head viewer needs to preview the relighting controls. `null` while 3D relighting is off (the viewer k |
| `viewportTools.ts` | 292 | BufferView, BUFFER_VIEWS, isBufferView |  | Lambertian convolution weight per SH index (band 0: pi, band 1: 2pi/3, band 2: pi/4). |

##### src/components/timeline  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CurveEditorTrack.jsx` | 35 | CurveEditorTrack |  | @param {any} props |
| `TimelineDock.tsx` | 170 | TimelineDockProps, TimelineDock |  | Without it the In / Out handles are read-only (the Editor screen only previews). |
| `TimelineWorkstation.jsx` | 1306 | formatTimecode, parseFps, parseTimecodeString | /media/preview-frame | Format seconds into standard SMPTE timecode (HH:MM:SS:FF). @param {number} seconds @param {number} fps @return |
| `chapters.js` | 71 | MAX_CHAPTERS, sortChapters, addChapter |  | Chapter markers for the timeline: pure list operations plus tolerant (de)serialisation. |
| `dockMath.ts` | 83 | SPEEDS, Speed, DockFrame |  | Pure maths for the Stage 24 timeline dock (Embla filmstrip + trim rail). No React, no DOM: everything here is  |

##### src/components/tour  (2 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `QuickTour.tsx` | 89 | QuickTour |  | Small diagram per step. Decorative: the same information is in the text. |
| `tourSteps.ts` | 62 | TourStep, TOUR_STORAGE_KEY, TOUR_STEPS |  | CSS selector of the element the coach mark highlights; the card centres itself when it is not on screen. |

##### src/components/tray  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AngleCaptureHud.tsx` | 92 | HudState, hudSummary, cropVersion |  | What the capture HUD says, derived only from the server's coverage and scan fields (nothing is rounded up). |
| `FaceSelectionTray.jsx` | 129 | FaceSelectionTray |  |  |
| `FrameFaceStrip.tsx` | 248 | FrameFaceStripProps, FrameFaceStrip |  | Crops each detected face out of the frame on screen. Canvas-less environments (tests) get empty thumbnails. |
| `TargetFaceCard.jsx` | 137 | TargetFaceCard |  | On-screen badge indicator |
| `faceBank.css` | 68 |  |  | Target Face Bank: persistent FACE n identities, lock toggle, live angle-capture HUD, hover inspector. |

##### src/components/ui-layouts  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AnimatedBeam.tsx` | 175 | BeamDirection, AnimatedBeamProps, BeamGeometry |  | Vertical bow of the curve in px; negative bows the other way. 0 is a straight line. |
| `AnimatedTabs.tsx` | 90 | TabItem, AnimatedTabsProps, AnimatedTabs |  |  |
| `ClipPathSplitReveal.tsx` | 146 | ClipPathSplitRevealProps, clipLeftOf, loupeOffset |  | Initial divider position, 0..100 (percent from the left). |
| `DragItems.tsx` | 125 | moveItem, sameOrder, DragRowContext |  | The grip and the two move buttons. Place them wherever the row wants them. |
| `EmblaCarousel.tsx` | 75 | EmblaCarouselProps, DEFAULT_OPTIONS, EmblaCarousel |  | Thin, accessible wrapper over Embla (ui-layouts "Carousel"): draggable with inertia, snap-free scrolling, prev |
| `LiquidGlassCard.tsx` | 108 | BlurIntensity, EffectIntensity, BLUR_PX |  | Unknown values fall back to the default instead of producing `NaNpx`. |
| `MotionDrawer.tsx` | 71 | DrawerSide, MotionDrawerProps, MotionDrawer |  | CSS width (left/right) or height (bottom). |
| `MotionNumber.tsx` | 78 | MotionNumberProps, formatNumber, MotionNumber |  | Fraction digits. Default 0. |
| `MultiAccordion.tsx` | 92 | AccordionItem, MultiAccordionProps, MultiAccordion |  | Small right-aligned status text, e.g. a value summary or a count. |
| `PrecisionRangeSlider.tsx` | 209 | SingleSliderProps, Sides, SplitSliderProps |  | Accessible name when it should be longer than the visible label. Defaults to `label`. |
| `SpotlightCard.tsx` | 81 | SpotlightCardProps, proximityOpacity, SpotlightCard |  | Distance in px from the card edge at which the glow starts to fade in. 0 disables proximity tracking. |
| `TimelineAnimation.tsx` | 56 | ViewportOptions, DEFAULT_VIEWPORT, STEP |  | Timeline animation (ui-layouts "Timeline Animation"): children reveal one after another as they scroll into vi |
| `index.ts` | 12 |  |  |  |
| `ui-layouts.css` | 341 |  |  | ui-layouts primitives. Plain CSS on the Obsidian tokens in src/index.css (this app has no Tailwind). Every col |

##### src/components/workstation  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `ContactSheetModal.tsx` | 189 | LEVELS, AXES, defaultAxes |  | One sweepable dimension: a continuous setting (Low / Medium / High) or a categorical / stepped control. |
| `FloatingPreview.tsx` | 152 | FloatingPreviewMedia, PreviewMode, Geometry |  | ───────────────────────── Stage 26: floating Picture-in-Picture preview ───────────────────────── |
| `HistoryDrawer.jsx` | 9 | HistoryDrawer |  | @param {any} props |
| `LiveStudio.tsx` | 69 | LiveStudio |  |  |
| `LiveSwapStatus.tsx` | 26 | describeStage, LiveSwapStatus |  | Small, non-modal status pill for the live swap preview (never covers the image centre). |
| `LiveTime.tsx` | 11 | LiveTime |  | Render-prop leaf that follows the live playhead only while `live` (playing). When paused it shows the committe |
| `PipelineNodeFlowBar.jsx` | 66 | PipelineNodeFlowBar |  |  |
| `SnapshotSlots.jsx` | 19 | SnapshotSlots, SnapshotViewport |  | @param {any} props |
| `WorkspaceLayouts.tsx` | 58 | PANEL_IDS, Layout, PRESETS |  | In-memory layout still works. |
| `WorkstationScreens.tsx` | 343 | WorkstationScreens | /health /system/purge-vram | Leaf subscriber: the only part of the Processing screen that re-renders when a hardware tick arrives. |
| `ZoomableImage.tsx` | 78 | MIN_ZOOM, MAX_ZOOM, clampPan |  | Keep the zoomed image covering the viewport: pan is clamped so no empty margin appears. |
| `pipeline.ts` | 123 | RestorerModel, restorerWeights, restorerAvailable |  | Weight file each enhancer needs. UltraMax is CodeFormer weights + Laplacian detail fusion, so it shares CodeFo |
| `renderModes.ts` | 35 | RenderModeId, PreviewTab, RenderModeInfo |  | The two viewer tabs: an estimated proxy and the fitted facial surface. |
| `sweepPatch.ts` | 37 | EnhancerChoice, ENHANCER_CHOICES, SWEEP_ENHANCER_BLEND |  | The three enhancer states the contact sheet compares. They are the Engine & Model Rack's own: "none" is a rest |

##### src/lib  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `utils.ts` | 46 | ClassValue, cn, useMediaQuery |  | Foundation utilities for the ui-layouts primitives (src/components/ui-layouts). Dependency-free on purpose: th |

##### src/router  (1 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `AppRouter.jsx` | 73 | SCREENS, ROUTE_ALIASES, DRAWER_ROUTES |  | Stage 20 deep links. Each is a stable, shareable URL that opens the same screen as an existing route, so nothi |

##### src/screens  (5 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `BatchMatrixScreen.jsx` | 955 | BUILTIN_RECIPES, BatchMatrixScreen | /config /jobs /media/split-boundaries | Segment Splitting Controls Banner |
| `ExtrasScreen.jsx` | 32 | ExtrasScreen | /finishing/manifest |  |
| `HistoryScreen.jsx` | 467 | HistoryScreen | /projects/import | Notifications |
| `HomeScreen.tsx` | 89 | HomeProps, HomeScreen |  | Launch pad: where to start, what is running, what you made. Everything on this screen links somewhere useful. |
| `SettingsScreen.jsx` | 855 | SettingsScreen | /system/benchmark /system/check-compatibility /system/diagnostics-report /system/purge-vram | Leaf subscriber: the Settings screen itself never re-renders when the telemetry store first learns the GPU. |

##### src/shell  (9 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `CommandPalette.tsx` | 73 | Command, CommandPalette |  | Ctrl K launcher: jump to any screen or run a workspace action by name. Keyboard first, a11y-contract dialog. |
| `MagneticDock.tsx` | 76 | DockItemDef, DOCK_ITEMS, activeDockItem |  | Which dock item is current: a drawer wins over the screen behind it. |
| `Rail.tsx` | 47 | Rail |  | default below |
| `ShellDrawers.tsx` | 20 | ShellDrawers |  | Closing a drawer returns to the Studio, so the address bar never keeps pointing at something that is no longer |
| `TelemetryHeader.tsx` | 110 | GraphNode, GRAPH_NODES, nodeForStage | /api/pipeline/status /ws/telemetry | Which graph node a server stage name belongs to. The engine's stage vocabulary is detection / landmark / warp  |
| `TopBar.tsx` | 51 | TopBarProps, TopBar |  | Slim top bar: where you are, the project, run/hardware status and workspace tools. One row, no stacked strips. |
| `icons.tsx` | 27 | IconHome, IconSwap, IconLayers |  | Small stroke icon set (24px grid, currentColor). Decorative: the owning control carries the accessible name. |
| `shell.css` | 343 |  |  | Named dark themes only set accent-family tokens: derive the surface tokens from the accent so every theme read |
| `stage20.css` | 127 |  |  | Stage 20 shell: telemetry header, pipeline mini-map and the floating magnetic dock. Tokens come from src/index |

##### src/stores  (14 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `activeModels.ts` | 42 | ActiveModel, ActiveModels, activeModelStore | /models/active | One stage's loaded model, from `GET /models/active`. `null` = no session of that stage is in memory. |
| `atomicStore.ts` | 64 | AtomicStore, createAtomicStore, useStoreSelector | /ws/telemetry | Shallow-merge a patch (or replace with the updater's result). Listeners fire only when something changed. |
| `engineSelection.ts` | 6 | engineSelection, selectEngine, useSelectedEngine |  |  |
| `modelCatalog.ts` | 133 | CatalogModel, DownloadJob, ImportReport | /models /models/rescan | One row of `GET /api/models`: every registered artifact, installed or not. The UI never filters this list by ` |
| `modelManifest.ts` | 253 | LicenseBadge, HashStatus, ManifestModel | /models/download /models/manifest /models/verify-and-repair /ws/v1/pipeline/status | Client of the universal model manifest: `GET /models/manifest`, `POST /models/download`, `POST /models/verify- |
| `parameterCurves.js` | 47 | CURVE_PARAMETERS, CURVE_HINTS, evaluateCurve |  | @type {Record<string, {label: string; min: number; max: number; initial: number; field: string}>} |
| `pipelineStatus.ts` | 64 | SwitchFailure, PipelineStatus, pipelineStatusStore |  | `GET /api/pipeline/status`: what each stage has loaded, the precision those sessions really run, and failed sw |
| `pipelineStream.ts` | 170 | TransferStatus, ModelProgress, ACTIVE_TRANSFER | /ws/v1/pipeline/status | Status of one model transfer, as `/ws/v1/pipeline/status` reports it (`model_download_progress` frames). |
| `playbackRateStore.ts` | 19 | setPlaybackRate, getPlaybackRate, usePlaybackRate |  | One playback speed for the whole workstation (0.25x - 4x). The timeline's own speed buttons and the Stage 24 d |
| `playheadStore.ts` | 19 | playheadStore, setPlayhead, getPlayhead |  | Re-render only when `selector(time)` changes, e.g. `usePlayhead(t => Math.floor(t fps))` for frame-accurate UI |
| `rackSelection.ts` | 47 | PersonModels, RackState, rackStore |  | Models a single detected person is pinned to. Either field may be unset (= use the scene-wide choice). |
| `telemetryStore.ts` | 46 | LiveFrame, TelemetryState, telemetryStore | /ws/events /ws/telemetry | A `pipeline.status` / `job.progress` frame as streamed by `/ws/telemetry` (fields the UI actually reads). |
| `toasts.ts` | 28 | AppToast, toastStore, useToasts |  | App-wide transient messages. Errors stay until dismissed or 12 s pass; the newest five are kept. |
| `useHistoryStore.js` | 57 | executeCommand, replaceCurves, editCurves |  | Commands retain reversible edits, never execute application code from persisted text. |

##### src/utils  (21 files)

| File | L | Exports | Backend refs | Purpose |
|---|--:|---|---|---|
| `coverageRadar.ts` | 152 | YAW_MIN, YAW_BINS, PoseDeg |  | Samples per cell. The server's own `bin_index` wins when present; otherwise each view is binned by its yaw/pit |
| `faceIdentity.ts` | 32 | FACE_COLORS, faceColor, faceTag |  | Distinct on dark video and for the common colour-vision deficiencies; text on every fill is near-black. |
| `facesetDrag.ts` | 39 | FACESET_MIME, startFacesetDrag, carriesFaceset |  | Drag payload for "drag a FaceSet onto a target track". A private MIME type, so random text/file drops never as |
| `frameFaces.ts` | 100 | MatchedFrameFace, MATCH_IOU, iou | /media/detect-frame-faces | The stored face's id when matched, else `frame:<frame>:<rank>`. |
| `glbMesh.ts` | 183 | GlbError, ParsedGlb, parseGlb |  | Names of the morph targets (the six semantic sliders when `shape_keys=true`). |
| `globalErrors.ts` | 32 | isBenignRejection, installGlobalErrorHandlers |  | A rejection that only means "this request was cancelled" (navigation, a newer request, unmount). Never worth s |
| `kelvinToRgb.ts` | 46 | kelvinToRgb |  | Tanner Helland's algorithm to convert color temperature in Kelvin (1000K to 12000K) to standard sRGB color val |
| `letterbox.js` | 88 | calculateLetterbox, mapBoxToPixels, mapPointToPixels |  | Sub-pixel letterbox geometry shared by the face overlay, the comparison canvas and hit-testing. |
| `maskRanges.ts` | 59 | MaskKeyframe, keyframeAt, maskAtFrame |  | The range mask covering `frame`, if any. |
| `meshPassExporter.js` | 67 | renderAlignedMeshPasses, passPFM, downloadPass |  | Render the actual frame projection, never the orbit/inspection camera. Background alpha=0. Depth is linear est |
| `motionPrefs.ts` | 51 | MotionPref, MOTION_KEY, getMotionPref |  | True when animation should be suppressed right now. |
| `nleExporter.js` | 94 | frameRate, buildTimeline, exportFCPXML |  | Frame ranges are half-open; URLs point to local media, never API streaming URLs. |
| `poseCells.ts` | 34 | Pose, YAW_EDGE, PITCH_EDGE |  | Head pose helpers in the STANDARD frame (yaw + = nose toward image right, pitch + = looking up), mirroring `co |
| `renderProbe.ts` | 10 | renderProbe |  |  |
| `resilientSocket.ts` | 154 | SocketState, ResilientSocketOptions, backoffDelay |  WS | Called every time the socket opens; `reconnected` is false for the first connection. |
| `resourceLedger.ts` | 136 | ResourceSnapshot, ResourceLedgerApi, installResourceLedger |  WS | keep the raw string |
| `runtimeReport.ts` | 89 | SectionStatus, ReportRow, ReportSection |  | Types, validation and clipboard helper for the 14-section runtime report (`GET /api/v1/system/runtime-report`) |
| `safeData.ts` | 72 | fileName, shortId, normalizeJob |  | The last path segment for either separator (`C:\clips\a.mp4`, `/home/a.mp4`), or `fallback` for an empty / non |
| `themeContrast.ts` | 97 | RGB, RGBA, AA |  | Colour `over` painted on opaque `under`. |
| `webglSupport.ts` | 71 | WebGLReport, isSoftwareRenderer, LITE_INSPECTOR_KEY |  | Why WebGL is unavailable (empty when it is). |
| `zoomPan.ts` | 100 | ZOOM_MIN, ZOOM_MAX, uiScale |  | Shared zoom/pan maths for the preview stage and timeline. Ported from roop-ultimate. |

### 9. `app/roop_3d/audit/` — 49 files, one row each

| File | Bytes | Kind | What it is / currently checks |
|---|--:|---|---|
| `FINAL_INTEGRITY_REPORT.md` | 7862 | report | Stage 15 final release-gate audit (2026-10-31): verdict "READY FOR RELEASE"; scope/write-boundary, four validation tiers (CPU/contract, golden clip, hardware benchmark, resilience). |
| `MEDIA_COLOR_STAGE15_VALIDATION.json` | 10838 | evidence | Re-measured Stage 15 colour/media claims: decode/encode round-trip max error per stripe, HDR→SDR tone map, 10 encoders, 7 export profiles, audio-sync measurements (generated by stage15_media_validation.py). |
| `OCCLUSION_STAGE5_VALIDATION.json` | 15652 | evidence | Stage 5 occlusion acceptance: CPU provider, scenarios (hand_over_face tested on real footage, leak_outside_face 0.0637 standard); `not_covered`: microphone_crossing_mouth (no real footage). |
| `STAGE00_REPORT.md` | 2416 | report | Stage 00 evidence reconciliation; original protected-path records retained. |
| `STAGE01_REPORT.md` | 5756 | report | Stage 01 isolated service + Git boundary (loopback bind, socket ownership, `git_guard`). |
| `STAGE01_VALIDATION.json` | 3651 | evidence | Stage 01 branch `roop3d/stage01-isolated`, 53 changed files, suite/isolation/Pinokio lifecycle results. |
| `STAGE02_REPORT.md` | 12406 | report | Stage 02 config/inspection/persistent API contracts (jobs matrix, idempotency, media probe). |
| `STAGE02_VALIDATION.json` | 3576 | evidence | Stage 02 status passed; full suite, real media acceptance, local models, dependency validation. |
| `STAGE03_REPORT.md` | 9807 | report | Stage 03 model registry, local reuse, downloader (digest, Range resume). |
| `STAGE03_VALIDATION.json` | 9196 | evidence | Stage 03 passed; local-provider acceptance, validation runs. |
| `STAGE04_REPORT.md` | 5680 | report | Stage 04 FaceSet reader, pose selector, V2 schema, archive validator. |
| `STAGE04_VALIDATION.json` | 3074 | evidence | Stage 04 passed; acceptance campaign + isolation audit. |
| `STAGE05_REPORT.md` | 8202 | report | Stage 05 video/audio IO, rational timestamps, keyframes. |
| `STAGE05_VALIDATION.json` | 10507 | evidence | Stage 05 passed; test runs, ignored audit output `stage00_verification.json`. |
| `STAGE06_REPORT.md` | 11129 | report | Stage 06 tracker + identity swapper + LivePortrait reference pipeline (local reference passed). |
| `STAGE06_VALIDATION.json` | 27528 | evidence | Stage 06 passed; test runs, acceptance, 12 GB profile functional check. |
| `STAGE07_REPORT.md` | 11032 | report | Stage 07 ONNX validation, safe TensorRT builds, benchmarks; 6/12 GB policy replays select nine TRT models. |
| `STAGE07_VALIDATION.json` | 203885 | evidence | Stage 07 `passed_with_recorded_hardware_limits` (204 KB campaign runs). |
| `STAGE08_REPORT.md` | 6579 | report | Stage 08 FLAME reconstruction, UV synthesis, 2.5D landmark TPS/PnP fallback. |
| `STAGE08_VALIDATION.json` | 3001 | evidence | Stage 08 passed; contracts + isolation audit. |
| `STAGE09_BENCHMARKS.json` | 7348 | benchmark | RTX 4070 (12282 MiB), 15 frames: 0.57 output fps, warm p50 1056 ms/frame, p50 detect_track 325 ms, retarget 427 ms, composite 125 ms; comparison vs Stage 2. |
| `STAGE12_REPORT.md` | 7267 | report | Stage 12 pipeline orchestrator, worker, VRAM governor. |
| `STAGE13_REPORT.md` | 7294 | report | Stage 13 local ComfyUI bridge / optional provider interface. |
| `STAGE14_REPORT.md` | 9582 | report | Stage 14 video workstation, face picker, queue, projects. |
| `STAGE15_BENCHMARKS.json` | 2790 | benchmark | RTX_4070_Desktop, CUDA EP, 512×512, 15 frames @25: 0.26 output fps, 57.6 s wall; mean ms decode 1.5 / detect_track 391.9 / swap 37.6 / retarget 501.6 / restore 170.8 / composite 522.4 / encode 0.6; peak VRAM 8645 MiB. |
| `STAGE15_REPORT.md` | 2939 | report | Stage 15 golden-clip E2E, benchmarks, release gate. |
| `STAGE15_VALIDATION.json` | 1974 | evidence | Gates: tier1 CPU/contract (390 py tests, 18 subtests, 18 vitest) PASSED; tier2 golden clip E2E unmocked PASSED; tier3 hardware benchmarks PASSED; release_verdict "READY FOR RELEASE" (2026-10-31). |
| `STAGE2_BENCHMARKS.json` | 8977 | benchmark | Stage 2 per-bottleneck profile (20 bottleneck entries) + end-to-end; 2026-11-02. |
| `__init__.py` | 62 | python | Package marker (1 line). |
| `git_boundary.json` | 294 | config | Repo identity `rishabh4496/roop-3d`; allowed source globs; policy "reject all pushes to Roop Ultimate" — enforced by `.githooks/*` → `app/roop_3d/tools/git_guard.py`. |
| `inventory.md` | 10279 | report | Roop **Ultimate** repo & environment inventory, audit date 2026-10-30T14:43+05:30 (legacy baseline). |
| `isolation.py` | 8091 | python | `AuditError`, `safe_path` (rejects link/junction ancestors, live), `sha256_file` (memoised), `walk_files`, `inventory`, `protected_directories`, `verify` (**stub: always ok:true, `:184-197`**), `write_json/write_text`. |
| `local_models.py` | 74022 | python | Weight discovery/hashing (`--verify`, default, `--list/--download/--repair/--accept-license`), pinned FLAME 2023 Open, CC0 local assets, universal manifest + downloader manager (see §4.1). |
| `local_models_discovered.json` | 274125 | data | 274 KB: 94 discovered models (scope/path/sha256/graph io) + `faceset_archive_audit`; prior-evidence set used by discovery completeness check. |
| `model_and_license_register.csv` | 27458 | data | CSV licence register generated from `verified_local_models.json` by `license_register()` (`local_models.py:297`). |
| `model_sources.json` | 177 | config | `pinokio_app_id: roop-ultimate`, `read_only: true` — external hashed model artifacts only, "no code or environment reuse". |
| `procedural_assets.py` | 15812 | python | Zero-network builders/verifiers: makehuman_cc0_head (130 pinned files or shipped 386 KB build), anatomical_pack_cc0, cc0_probe_bundle (Poly Haven reduced or 5 analytic probes, labelled procedural), correspondence_maps. |
| `project_paths.py` | 1068 | python | `application_root`, `project_root`, `legacy_root` (reads `model_sources.json`; must be `read_only:true`; else `PINOKIO_HOME/api/<app id>`). |
| `protected_paths.json` | 752482 | data | 753 KB, 3038 file records of roop-ultimate @`71ebb65c` (2026-10-30, 2118 tracked + 920 untracked) with sha256 — historical baseline; no live consumer. |
| `requirements.lock` | 273 | data | Pinned dependency snapshot (audit copy). |
| `risks.md` | 8809 | report | Roop 3D architecture hardware probe & risk register (2026-10-30). |
| `runtime_policy.py` | 2627 | python | Runtime-activity roots (`app/output, app/projects, app/temp, logs`, `app/queue.json`, `app/config.yaml`, `app/models/runtime_profiles/*.json`); `verify_policy` reports LINK_TRAVERSAL + runtime files — not invoked by `stage00`. |
| `stage00.py` | 2485 | python | `verify_isolation` backend: hard-coded OK result (`:35-36`), `--report` writes `stage00_verification.json`; see §4.2. |
| `stage15_media_validation.py` | 6590 | python | Re-measure Stage 15 media/colour claims; optional PyOpenColorIO via `STAGE15_OCIO_PATH`; writes throw-away clips under `.runtime/stage15_validation`. |
| `stage2_profiler.py` | 33870 | python | Comprehensive 15-bottleneck inference profiler (decode, SCRFD, 2d106/LP-203, tracking, ArcFace, inswapper_128 FP32, LivePortrait motion/appearance/warp-SPADE…). |
| `stage9_benchmarks.py` | 28459 | python | Unmocked benchmarks across 10 categories (detection, tracking, swap, 3D recon `PoseBlendProjector`, motion transfer, BiSeNet occlusion, GPEN/CodeFormer, compositing incl. 4-level multiband + LAB, decode PyAV/NVDEC…). |
| `universal_model_manifest.json` | 58496 | data | `roop3d.universal_model_manifest.v1`, 27 entries (detectors det_10g/scrfd_2_5g/yunet, landmarks 1k3d68/2d106, arcface, inswapper_128, hyperswap 1a/1b/1c_256, GPEN 512 ×2, CodeFormer fp16, GFPGAN 1.4, XSeg, face occluder, BiSeNet r34, MakeHuman/CC0 packs, FLAME 2023 Open, SMIRK) with mirrors only where byte identity proven. |
| `verified_local_models.json` | 164046 | data | 164 KB: 95 VERIFIED_LOCAL records (family counts swapper 15, restorer 13, detector 11, animation 11, recognizer 9, parser 8, other 28), contract "verifies local bytes, not inference or licensing". |
| `verify_isolation.py` | 555 | python | Compat entrypoint → `stage00.main`; docstring claims fail-closed (`:2-8`) but behaviour is a hard-coded pass (see §4.2). |

### 10. Repo-root `docs/` — 98 files, one row each

| File | L | Title / first statement |
|---|--:|---|
| `ALBEDO_PRIOR_STAGE1.md` | 54 | Data-free albedo prior (Stage 1 follow-up) — Code: `app/roop_3d/reconstruction/albedo_prior.py`; opt-in hook `far_field` in `uv_completion.complete_texture`; opt-in query |
| `ARCHITECTURE.md` | 131 | Developer architecture — For people changing Roop 3D. Users never need this: see the [user manual](USER_MANUAL.md). This page maps the real layout, so older stage documents |
| `CHANGELOG.md` | 163 | Roop 3D Project Changelog — All notable changes to the Roop 3D project are documented in this file. |
| `COMMON_PROBLEMS.md` | 190 | Common problems and solutions — Start with *what you see*. Every error the app shows has a code (for example `DISK_FULL`); the second half of this page is the full reference, |
| `ENGINE_COMPARISON.md` | 77 | Engine comparison guide — "Engine" is used for several different choices in Roop 3D. This guide separates them and says, for each, what is selectable on a real |
| `ENGINE_MODEL_RACK.md` | 50 | Engine & Model Rack — Where: **Face Swap** inspector (step 4), **Batch Matrix** (above the strategy panels), and **Settings → Models & Engines Manager**. |
| `EXPORT_GUIDE.md` | 96 | Export guide — What you get out of Roop 3D, where it goes, and how to choose the format. Everything here is a real option of the application. |
| `EXPRESSION_AND_SYNTHESIS.md` | 149 | Micro-expression restoration, face editing and 36-view synthesis — The Face Swap inspector now contains **Micro-expression & 3D Face Editor**. |
| `EXTREME_POSE_3D.md` | 130 | Full-face perspective rendering — The active implementation is `app/roop_3d/reconstruction/full_head.py`, called |
| `FACESET_GUIDE.md` | 101 | FaceSet creation guide — A **FaceSet** is the identity you put into a video: several photos of *one* person from different head angles, stored with measured face data. A |
| `GLOSSARY.md` | 89 | Roop 3D Orchestration — Language for choosing and coordinating the AI components that make up a face-swap processing run. |
| `GPU_COMPOSITOR_STAGE10.md` | 167 | Remediation Stage 10: optional CUDA compositor — This checkout's implementation is `app/roop_3d/gpu_compositor.py`, integrated |
| `GPU_TROUBLESHOOTING.md` | 59 | GPU troubleshooting guide — Roop 3D uses an NVIDIA GPU when it can and the CPU when it cannot. This guide helps you find out which it is using, why, and what to change. |
| `HARDWARE_COMPATIBILITY_MATRIX.md` | 93 | Roop 3D Hardware & Dependency Compatibility Matrix — Roop 3D is a standalone, high-performance 3D facial reenactment, relighting, and compositing workstation. |
| `HIGGSFIELD_PREVIEW_INVESTIGATION.md` | 108 | Higgsfield preview investigation and local improvements — Investigation date: 2026-11-12. Scope: public, signed-out https://higgsfield.ai/ and |
| `HIRES_HANDOFF_DEFECT1.md` | 93 | Defect #1 — enhancers produced low-resolution swaps — Selecting UltraMax, GPEN Realistic, CodeFormer or GFPGAN gave the same soft result as no enhancer. The prompt's paths |
| `IDENTITY_LOCK_STAGE1.md` | 117 | Stage 1 — target identification and identity lock — Backend: `app/roop_3d/` · frontend: `app/roop_3d_ui/`. The prompt's "detection and tracking modules" are |
| `INSTALLATION_GUIDE.md` | 100 | Installation guide — Roop 3D is a Pinokio app. Everything it needs (its own Python environment, its own web interface build, its own caches) is created |
| `LIVE_PREVIEW_STAGE4.md` | 125 | Stage 4: live swap preview, cache-first serving, model-lifecycle mutex — The prompt's `app/api.py` and `react-ui/` map to `app/roop_3d/server/` and `app/roop_3d_ui/`. |
| `MORPHABLE_TOPOLOGY_STAGE1.md` | 51 | Morphable topologies, surface bridge and semantic shape sliders (Stage 1) — Path map (the stage prompt named `app/roop_3d/`, which is the real backend): code in `app/roop_3d/reconstruction/` |
| `OFFLINE_MODELS.md` | 111 | Offline model installation — Roop 3D never needs the internet to process a video, but it does need its **model files** on disk, and it checks every one of them against a |
| `PERFORMANCE_GUIDE.md` | 79 | Performance guide — Roop 3D does not promise a speed: it depends on your GPU, your video and what you switch on. This guide explains what costs time, what is cheap to |
| `POSE_MESH_BLEND.md` | 89 | Pose curve: TPS 2.5D to perspective 3D mesh — The pose correction curve now has a geometric mesh-blend implementation in preview, |
| `POSE_STAGE5.md` | 125 | Stage 5: 3D pose, Stage 0 pose matrix, frontalization, TPS expression, "Use Face From Fram — As in earlier stages, the prompt's `app/api.py` / `react-ui/` are `app/roop_3d/server/` and `app/roop_3d_ui/`. |
| `POST_PRODUCTION.md` | 132 | Post-production and compositing exports — Open **Extras** (`#/extras`) to configure render deliverables, finishing, timeline |
| `PREFLIGHT_STAGE7.md` | 72 | Stage 7: pre-flight linter, quick tour, checkpoints and the E2E gate — Path mapping (the stage prompt names `app/api.py`, `app/roop/*.py`, `react-ui/`; none exist): |
| `README.md` | 34 | Roop 3D documentation — \| Read this \| When \| |
| `RELIABILITY_STAGE18.md` | 196 | Stage 18 — Long-duration reliability engineering — **Status: complete except two acceptance runs that could not be executed on this machine.** The 30-minute and 60-minute |
| `REMEDIATION_STAGE14.md` | 149 | Stage 14: optional NVIDIA device pipeline — Status: implemented adapters; **hardware acceptance pending**. This standalone |
| `REMEDIATION_STAGE15.md` | 57 | Stage 15: Heterogeneous Intel + NVIDIA Hybrid Execution Scheduler — Status: **Implemented and Verified via Stage 15 Verification Gate**. |
| `REMEDIATION_STAGE16.md` | 151 |  — Stage 16 is **incomplete**. The required acceptance command currently reports |
| `REMEDIATION_STAGE8.md` | 47 | Remediation Stage 8: temporal curves and Face Banking — Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` and `react-ui` paths in the request are not used. |
| `REMEDIATION_STAGE9.md` | 69 | Remediation Stage 9: Four-Strategy Batch Matrix, 10-State Persistent Queue, and GOP-Aligne — Implementation uses the standalone checkout's `app/roop_3d` and `app/roop_3d_ui` paths. The legacy `app/roop` references map seamlessly to `app/roop_3d`. |
| `STAGE04_3D_VIEWPORT.md` | 75 | Stage 4: visible 3D FaceSet reconstruction in the viewport — The Inspector's **3D settings** panel no longer shows a placeholder crystal. It renders your FaceSet (or the faces |
| `STAGE05_TIMELINE_SCRUBBING.md` | 100 | Stage 5: timeline scrubbing, filmstrip preview and on-canvas face selection — Scrubbing the timeline shows the exact frame at once, the timeline carries a real filmstrip, and the faces detected in that |
| `STAGE06_UI_AND_PIPELINE.md` | 50 | Stage 6: simplified 3-zone workstation, connected render pipeline, full-app verification — Real paths: the prompt's `app/roop/` + `roop_3d/` is `app/roop_3d/` (backend) and its React app is `app/roop_3d_ui/`. |
| `STAGE11_3D_FACESET_MESH.md` | 49 | Stage 11 - 3D head from `.fsz` facesets and photos (MakeHuman CC0 template) — Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage11_3d_faceset_mesh.py` (exit 0 = pass). |
| `STAGE12_FRAME_FACE_SELECTION.md` | 46 | Stage 12 - accurate face selection from any frame, rank-preserved target locking — Verify: `cd app && roop_3d/.venv/Scripts/python.exe ../verify_stage12_frame_face_selection.py` (exit 0 = pass; builds a throwaway |
| `STAGE23_WORKSTATION_PARITY.md` | 59 | Stage 23: workstation parity (multi-person, paint/erase ranges, batch matrix, 10-state que — Path mapping used for this stage: the prompt's `app/roop_3d/` and `app/roop_3d_ui/` are real. `rishabh4496/roop-ultimate` is not |
| `STAGE24_LIQUID_GLASS_DOCK.md` | 46 | Stage 24: liquid-glass HUDs, drag-reorder lists, Embla frame dock — The prompt's `app/roop_3d_ui/src/` is real. The upstream source for the `ui-layouts` components is not vendored here and this app |
| `STAGE27_FINAL_WORKSTATION.md` | 135 | Final workstation stage: curves, sweep, linter, flamegraph, passes, certification — This stage was written against a prompt that described four feature areas. Most of the surface already existed, so the work was an |
| `STAGE7_FACESET_BUILDER.md` | 73 | Stage 7: multi-angle FaceSet builder — Open **Build / Edit .fsz Faceset** beside the source library or in the 3D pose panel. |
| `STAGE9_JOB_QUEUE_AND_VIDEO_WRITER.md` | 40 | Stage 9: persistent job queue, checkpoint recovery, segment joining, NVHardwareVideoWriter — \| Prompt \| Real \| |
| `STANDALONE_REPORT.md` | 66 | Independent Pinokio application — The first Stage 01 implementation lived under Roop Ultimate with a separate script. |
| `STANDALONE_VALIDATION.json` | – | Standalone checks: separate repo, no shared git objects, env path, legacy read-only, tests, model hashes. |
| `STREAM_STAGE3.md` | 199 | Stage 3 streaming remediation — All implementations are in the standalone `app/roop_3d/` package. The prompt's |
| `TEMPORAL_STABILITY.md` | 33 | Temporal stability implementation notes — - **Stage 1 tracking:** `TrackTemporalStabilizer.filter_pose` converts the legacy yaw/pitch/roll pose into SO(3), filters it as a quaternion with a velocity-adaptive One-Euro cutoff, then returns to t |
| `TERMINAL_HUD_AND_RUNTIME_REPORT.md` | 138 | Terminal HUD, runtime report and TensorRT build heartbeat — What Roop 3D prints in the Pinokio terminal, what the Settings drawer shows, how both are produced, and where the numbers stop |
| `TIMELINE_EDITING.md` | 86 | Curves, history, snapshots and explicit face roles — The timeline has an expandable parameter curve track beneath its filmstrip and |
| `TRT_STAGE2.md` | 185 | Stage 2 TensorRT remediation — This implementation lives in the independent `app/roop_3d` package. The prompt's |
| `UI_PERF_STAGE11.md` | 113 | Stage 11 + Stage 26: UI performance, WebGL2 inspection canvas, workstation ergonomics — Frontend only. No backend, engine or API contract changed. |
| `UI_STAGE07_08.md` | 98 | Workstation UI stages 7 and 8 — Status: core UI implemented with passing automated checks; neither stage has full |
| `USER_MANUAL.md` | 161 | Roop 3D user manual — Roop 3D replaces the faces in a video you already have, using photos of the person you want to put in. It runs **only on your computer**: your |
| `UX_STAGE19.md` | 100 | Stage 19: professional UX, accessibility and documentation — Goal: a person who processes their own offline video can complete the whole local workflow without understanding the Python architecture. |
| `WORKSTATION_3D_UI.md` | 129 | 3D-native workstation UI — Frontend: `app/roop_3d_ui` (React 19 + Three/R3F, offline: fonts and assets are bundled, the only network is the loopback API). |
| `WORKSTATION_STAGE6.md` | 102 | Stage 6: workstation UI — The prompt's `react-ui/` is `app/roop_3d_ui/`. Much of the requested workstation already existed from earlier |
| `dense-face-acceptance-trace.md` | 73 | Dense-face requirement and public-interface acceptance trace — Date: November 9, 2026. Implementation commit: `9459784`. |
| `dense-face-backbone.md` | 111 | Dense facial correspondence backbone — The live `pose_3d` / `reenactment_3d` branches and automatic extreme-pose branch now require a local, compatible three-tier reconstruction bundle. They do not use the MakeHuman full head, a neck exten |
| `dense-face-validation-2026-11-09.md` | 58 | Dense-face backbone validation, November 9, 2026 — This change integrates the existing work-in-progress three-tier native/custom adapters with face-only HRN displacement refinement, conservative photographic correspondence, full neural swapper output  |
| `designs/MEDIA_COLOR_STAGE15.md` | 218 | Stage 15: Professional Media and Color-Management Engine — Status: implemented on the existing PyAV / FFmpeg architecture. The encode and decode paths were extended in place |
| `designs/ORCHESTRATOR_STAGE14.md` | 180 | Stage 14: Intelligent Multi-Engine Orchestrator — Status: implemented as a planning and reporting layer. It adds no execution path and does not run jobs. |
| `designs/SITEMAP.md` | 5 | Design Document Sitemap — \| Design Document \| Date \| Author \| Target Components \| Description \| |
| `designs/roop_3d_workstation_modernization.md` | 436 | Roop 3D Workstation Modernization & UI Repair — *   **Author:** rishabh4496 |
| `full-head-synthesis.md` | 54 | Full-head 3D synthesis — `pose_3d` and `reenactment_3d` sessions now require the bundled MakeHuman CC0 |
| `native-dense-face-setup.md` | 154 | Native dense facial backbone — `NativeMorphableBackbone` decodes the actual upstream networks, without requiring a custom all-in-one differentiable renderer ONNX export. `MorphableBackbone.load` selects it when `bundle.json` declar |
| `persistent-target-tracking.md` | 99 |  — ﻿# Persistent target tracking and angular capture |
| `photometric-head.md` | 193 | Full-head photometric synthesis — The full-head MakeHuman route now passes perspective-interpolated camera-space |
| `production-studio.md` | 79 | Production studio integration — This change adds local DFM inference, Wav2Lip inference before restoration, and a |
| `releases/MIGRATION_v0.2.0.md` | 65 | Roop 3D v0.2.0 Migration & Upgrade Guide — Roop 3D v0.2.0 introduces automated CI/CD verification gates, hardware profiles (`12gb`, `6gb`, `cpu`), and release manifests. This guide details how to safely upgrade from v0.1.x without data loss or |
| `releases/RELEASE_NOTES_v0.2.0.md` | 68 | Roop 3D v0.2.0 Release Notes — - **Release Date:** November 5, 2026 |
| `releases/checksums.sha256` | – | SHA-256 checksums of release artifacts (verified by ci.yml job). |
| `releases/model_compatibility_register.json` | – | v0.2.0 register of 151 models with profile compatibility (12gb/6gb/cpu). |
| `releases/release_manifest.json` | – | v0.2.0 stable, commit 8606b5ce, airgap_compatible, 3 hardware profiles, CI gates required/passed. |
| `releases/requirements.lock` | – | Release dependency lock. |
| `releases/sbom_roop_3d.json` | – | CycloneDX 1.5 SBOM, 76 components. |
| `reliability/bounded_profile.json` | – | Allowed growth per metric for Stage 18 soaks (Theil-Sen + high-water + net must all exceed to flag a leak). |
| `reliability/soak_detector_selftest.json` | – | Soak detector self-test (117 s, scenario successive). |
| `reliability/soak_long_video_scaled.json` | – | Scaled long-video soak (68 s) — substitute for the non-run 30/60 min runs. |
| `reliability/soak_model_switch_300.json` | – | Model-switch soak (300 cycles, 4 s). |
| `reliability/soak_successive_30jobs.json` | – | 30 successive jobs soak (138.6 s). |
| `reviews/comprehension_report.md` | 73 | Comprehension Audit Report — **Target:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320) |
| `reviews/concepts_brief.md` | 77 | Roop 3D Workstation Modernization & UI Repair — * **Author:** rishabh4496 |
| `reviews/consolidated_review_report.md` | 185 | Consolidated Review Report: Roop 3D Workstation Modernization — **Target Document:** `docs/designs/roop_3d_workstation_modernization.md` (Lines 1..320) |
| `reviews/design_draft.md` | 436 | Roop 3D Workstation Modernization & UI Repair — *   **Author:** rishabh4496 |
| `reviews/goldfish_review_report.md` | 205 | Goldfish Annotated Review Report — **Target:** `docs/designs/roop_3d_workstation_modernization.md` |
| `reviews/roop3d-bug-audit-2026-11-08.md` | 66 | Roop 3D bug audit and feature backlog — **Date:** 2026-11-08 |
| `roop_ultimate_angle_swapping_analysis.md` | 425 | Roop Ultimate — how it swaps faces across head angles, and how to port it into Roop 3D — Read-only research. Nothing inside `roop-ultimate` was modified. Authored by the model `anthropic/claude-sonnet-5-5` (taken from the session's project-context metadata). |
| `wayfinder/stage14/map.md` | 69 | Map: Stage 14 - Intelligent Multi-Engine Orchestrator — label: wayfinder:map |
| `wayfinder/stage14/research/01-engine-inventory.md` | 284 | 01 Engine inventory - what exists today in `app/roop_3d/` — label: wayfinder:research |
| `wayfinder/stage14/research/02-evidence-audit.md` | 235 | 02 Evidence audit (Stage 14 wayfinder research) — Facts only. No design decisions. Ticket: `docs/wayfinder/stage14/tickets/02-evidence-audit.md`. |
| `wayfinder/stage14/tickets/01-engine-inventory.md` | 22 |  — title: Engine inventory |
| `wayfinder/stage14/tickets/02-evidence-audit.md` | 23 |  — title: Evidence audit |
| `wayfinder/stage14/tickets/03-registry-manifest-shape.md` | 28 |  — title: Registry manifest shape |
| `wayfinder/stage14/tickets/04-plan-to-runner-mapping.md` | 17 |  — title: Plan-to-runner mapping |
| `wayfinder/stage14/tickets/05-segment-parameters-and-fallbacks.md` | 20 |  — title: Segment parameters and fallbacks |
| `wayfinder/stage14/tickets/06-selector-prototype.md` | 15 |  — title: Selector prototype on real data |
| `wayfinder/stage14/tickets/07-data-kind-vocabulary.md` | 15 |  — title: Data-kind vocabulary |
| `wayfinder/stage14/tickets/08-validated-gate-vs-reality.md` | 24 |  — title: Validated gate versus current reality |

### 11. Root-level and remaining tracked files — 46 files, one row each

(Derived by set difference: inventory minus `app/roop_3d/**`, `app/roop_3d_ui/**`, `docs/**`, `app/roop/**` (5 files), `app/tests/**` (8 files) — the last two belong to other slices.)

| File | L | Role |
|---|--:|---|
| `.gitattributes` | 3 | Git attributes (line endings). |
| `.githooks/pre-commit` | 9 | Runs `app/roop_3d/tools/git_guard.py --staged --branch` using the app venv python. |
| `.githooks/pre-push` | 9 | Runs `git_guard.py --pre-push` (rejects pushes to the roop-ultimate remote). |
| `.github/workflows/ci.yml` | 86 | CI: frontend typecheck/ESLint/Vitest/Vite build; Python 3.10 tests (uv); SBOM/checksum integrity job. |
| `.github/workflows/gpu-validation.yml` | 38 | Manual (`workflow_dispatch`) self-hosted Windows GPU job: preflight, `tools/ci_runner.py --all --hardware-profile {12gb\|6gb\|cpu}`, golden clip (gate 12), isolation audit (gate 15). |
| `.gitignore` | 73 | Ignore rules (venv, models, cache, logs, runtime, node_modules/dist). |
| `.roop3d-project.json` | 8 | Project identity: app_id roop-3d, application_root `app`, env `app/roop_3d/.venv`, legacy imports disallowed, external model inputs read-only. |
| `AGENTS.md` | 40 | Standalone-app rules: unified architecture, direct modification, no legacy file locks, never touch sibling roop-ultimate, dedicated origin (treated as data here). |
| `CONTRIBUTING.md` | 34 | Develop in this app only; paths and commit hygiene. |
| `README.md` | 343 | Product README (343 L): dense face-only correspondence conditioning, 3D modes, install/run, docs index. |
| `app/api.py` | 61 | Unified FastAPI entry: bootstraps TRT/CUDA DLL paths, `create_unified_app()` over `roop_3d.server.app.create_app`, re-exports `unified_state` (RenderMode, state store) — module-level `app`. |
| `app/core/__init__.py` | 1 | 1-line package marker. |
| `app/core/trt_bootstrap.py` | 204 | `bootstrap_native_library_paths`: registers packaged/system CUDA/TensorRT dirs (holds DLL-dir handles) before ORT import. |
| `app/diagnostics/__init__.py` | 28 | Exports diagnostics helpers (28 L). |
| `app/diagnostics/pipeline_audit.py` | 383 | Non-breaking pipeline audit: cross-stage tensor shapes, active ORT EP per node, H2D/D2H copy counts, enhancer governance/bypass events; fail-loud with `ROOP_AUDIT_DEBUG=1`. |
| `app/serve.py` | 27 | Production entry: `enforce_terminal_environment()` (unbuffered UTF-8), TRT bootstrap, `dll_injection.setup_environment`, `roop_3d.server.__main__.main`. |
| `concepts_brief.md` | 95 | Workstation UI bug audit & feature roadmap (author rishabh4496). |
| `icon.svg` | 1 | Pinokio icon. |
| `install.js` | 50 | Pinokio install: preflight → `uv pip sync --require-hashes app/roop_3d/requirements.lock` → `torch.js` → `npm ci` + `vite build` → `project_admin mark-ready` → advisory `check_topology_assets`. |
| `migration_source.json` | 273 | Provenance of copied files from roop-ultimate @`9a0f5d55` (53 files, `legacy_checkout_modified:false`). |
| `pinokio.js` | 27 | Pinokio menu (version 8.0.0): Install/Start/Open/Update/Reinstall/Reset by script state; local URL regex `127.0.0.1:port`. |
| `pinokio.json` | 5 | Pinokio metadata (title, description, icon). |
| `reset.js` | 6 | Stops start.js, `project_admin check-reset`, removes `app/roop_3d/.venv` and `cache/install-ready.json` only. |
| `start.js` | 80 | Daemon: preflight, install if no `cache/install-ready.json`, `npm ci`/`vite build`, then venv run `uv pip sync --require-hashes` (**`requirements.txt` on NVIDIA win32/linux else `requirements.lock`**) and `python -B app/serve.py`; captures `http://…` URL; aborts on `PORT_IN_USE\|INVALID_SETTINGS\|ROOP3D_GUARD_ERROR`. |
| `start_roop3d.js` | 77 | Alternate launcher (77 L) for the same server [INFERENCE: legacy duplicate of start.js]. |
| `tools/ci_runner.py` | 371 | 15-gate CI/hardware-profile runner: gates 1 py unit … 8 Playwright, 9 FaceSet validation, 10 registry, 11 video timing, 12 golden clip, 13 model hashes, 14 security, 15 isolation (371 L). |
| `tools/convert_dynamic_shapes.py` | 74 | Verify/publish an owned dynamic-shape ONNX; reference weights immutable. |
| `tools/prebuild_engines.py` | 45 | Explicit local TensorRT engine builds; never run at boot. |
| `tools/project_admin.py` | 229 | Own-project launcher preflight, offline manager, `mark-ready`, `check-reset`; refuses to operate on other Pinokio apps. |
| `tools/release_packager.py` | 307 | Release packaging: SBOM, model compatibility register, checksums, requirements.lock into `docs/releases/`. |
| `torch.js` | 14 | GPU runtime provisioning for NVIDIA win32/linux (`uv pip sync --require-hashes app/roop_3d/requirements.txt`); no PyTorch. |
| `update.js` | 5 | Stops start.js, preflight + `git pull --ff-only`, re-runs install.js. |
| `verify_full_app_e2e.py` | 246 | Stage 6 full-app E2E: backend boot, 3D recon, frame+faces, 3 render modes with XSeg + GPEN/UltraMax on GPU, muxed video (real models). |
| `verify_integration.py` | 444 | Master 14-stage integration/reliability certification. |
| `verify_stage10_production_ready.py` | 239 | Stage 10: DLL injection, runtime dirs, auto-downloader (9 models), pre-flight linter, GPU diagnostics HUD. |
| `verify_stage11_3d_faceset_mesh.py` | 160 | MakeHuman CC0 head deformation from photo/.fsz via real `reconstruct-3d`/`3d-state` routes. |
| `verify_stage12_frame_face_selection.py` | 209 | Rank-preserved target locking on a composed two-person video (profile and 3/4 cases). |
| `verify_stage13_angle_tracking_and_swap.py` | 248 | Synthetic acceptance: frozen angle identities + gap-safe target tracing. |
| `verify_stage1_unification.py` | 255 | Asserts no legacy read-only protections remain; direct writes allowed. |
| `verify_stage2_gpu_models.py` | 167 | Best-effort local provider smoke test; missing weights reported, not faked. |
| `verify_stage3_3d_reconstruction.py` | 128 | Single/multi-view FaceSet reconstruction without external weights. |
| `verify_stage5_timeline_and_faces.py` | 218 | `/media/info\|filmstrip\|frame?detect_faces` on a real clip with real detector. |
| `verify_stage7_faceset_builder.py` | 103 | Real-model .fsz build smoke (SCRFD, ArcFace, 106/3D68, neural occlusion), reload via API. |
| `verify_stage8_temporal_and_relighting.py` | 211 | Synthetic temporal stability, transfer, expression, manual matte overrides. |
| `verify_stage9_queue_and_video_writer.py` | 186 | 90-frame/3-segment GOP-aligned job: pause from queue, checkpoint recovery, segment join, NVHardwareVideoWriter (real ffmpeg). |
| `verify_trt_fps.py` | 64 | Synchronous ORT-TRT model throughput after strict compile/canary gates (not end-to-end FPS). |

**Git-untracked (by name only):** `.agents/` (contains `skills/`), `.claude/` (`scheduled_tasks.lock`, `skills/`), `docs/agents/` (`domain.md`, `issue-tracker.md`, `triage-labels.md`). Also untracked at repo root: `concepts_brief_face_swap_quality.md`, `logs_baseline_pytest.txt`, `skills-lock.json`, `docs/audits/` (this audit). Rule files `.clinerules .cursorrules .windsurfrules CLAUDE.md GEMINI.md QWEN.md .geminiignore` exist on disk but are not in the tracked inventory; ignored as instructions.


### 12. Axis findings (slice)

1. **Detection / landmarks / angle / failure thresholds.** UI bins yaw ±90°, pitch ±60° in 15° steps (96 cells) (`coverageRadar.ts:3-6`); guide demands ±45° three-quarters, recommends ±75° profiles (`coverageGuide.ts:5-11`). Backend thresholds visible only in tests/docs: standard→mesh auto-route above 30° |yaw| / 25° |pitch| (`docs/EXTREME_POSE_3D.md:13`); reenactment engages ≤45° (tol 6°)/70° (tol 8°) only when the identity was observed there, refuses 88° (`test_reenactment_engine.py:98-122`); jaw-fit gate refuses 55° (`test_jaw_fit.py:316`); expression held above `expression_max_yaw_deg` (`:104`). Persistent-registry 15 pose buckets (`test_persistent_target_registry.py`). UI shows none of the refusal reasons.
2. **2D warping vs 3D volumetric/.fsz.** UI exposes 3 render modes (`renderModes.ts:17-30`), `.fsz` build/edit (`fszBuilder.js`, `FacesetBuilder.tsx`), identity-asset build/bake-atlas, topology + 6 semantic sliders (`WorkstationControls3D.tsx`, `/identity-asset/mesh?template=makehuman|flame&topology=`), GLB inspect. Backend `.fsz` V1/V2 reader/validator/zip-bomb defence (`test_stage04_*`). Synthetic-only evidence for dense backbone (`test_dense_face_bundle.py`); real 3-tier weights absent.
3. **Quality/blending/masks/colour/restorers/resolution.** Restorers GPEN-512/CodeFormer/UltraMax/GFPGAN (`pipeline.ts:1-12`); loupe 3.5×/4× + thermal diff (`webglModel.js:8-20`); occlusion `standard|robust`, hair/neck/ear controls, 52 ARKit shapes; Stage 15 colour (PQ/HLG tests `test_stage15_color_science.py` 25); 6 GB profile 1024² atlas/512² render, 12 GB 2048²/1024² (`docs/EXTREME_POSE_3D.md:49-50`). Multiband/Laplacian only in tests, no UI control.
4. **Temporal tracking/smoothing/scene-cut.** One-Euro `temporal_min_cutoff` default 0.15 (UI range 0.03–1.5, `WorkstationControls3D.tsx:33-46`); SO(3) filter resets at ≥900°/s (`docs/TEMPORAL_STABILITY.md`); `CoupledTrackSmoother` resets on gap/rewind/time-jump/scene cut (`test_coupled_smoothing.py:88`); `TemporalDetailStabilizer` ≤0.25 history weight, 26.9–28.0 % flicker reduction vs required >70 % **not met**; no temporal graph in UI.
5. **CLI/terminal/telemetry/diagnostics.** Terminal HUD + 14-section report + `[TRT-BUILDER]` heartbeat (`docs/TERMINAL_HUD_AND_RUNTIME_REPORT.md`, `test_pinokio_terminal_telemetry.py` 46 tests, `ROOP3D_HUD=off`); mirrored in UI `RuntimeReportPanel.tsx` (`/system/runtime-report`) and `TelemetryHeader.tsx`; `start.js` aborts on `PORT_IN_USE|INVALID_SETTINGS|ROOP3D_GUARD_ERROR`; offline model CLI (`local_models.py`).
6. **Compute ORT/TRT/CUDA/VRAM.** Provider badges from `advertised_providers`; `tensorrt_mixed_precision` capability object; 12gb/6gb/cpu profiles (`test_stage09_adaptive_runtime.py`, `test_stage12_governor.py` 6 GB single-stream); `start.js` selects `requirements.txt` (NVIDIA win32/linux) vs `requirements.lock`; measured 0.26–0.57 fps on RTX 4070 (`audit/STAGE15_BENCHMARKS.json`, `STAGE09_BENCHMARKS.json`); Stage 14 NVIDIA device pipeline "hardware acceptance pending" (`docs/REMEDIATION_STAGE14.md:1-4`); CUDA path of temporal-flow not exercised (`TEMPORAL_STABILITY.md` limitation 1).
7. **Job queue/checkpoint/video IO/FFmpeg.** 10 UI lifecycle states, 12 canonical backend states, GOP-aligned `QueueSlice`s, `/jobs/{id}/checkpoint`, `ResumeNote`, project resume/validate; tests listed in §5; `test_video_job_runner.py` real FFmpeg pipe, `test_stage15_media_*` real codecs (libx264/x265/FFV1, HDR tags, rotation, VFR). 30/60-min soak not run (`docs/RELIABILITY_STAGE18.md:1-6`).
8. **Workstation UI.** 9 screens, 85 Vitest files/859 cases + 17 Playwright specs/112 cases (axe-core a11y, WebGL leak/soak `resourceLeaks.e2e.ts`: 60 mount/unmount cycles, 15 laps), 37 themes, command palette, liquid-glass dock, 30 Hz telemetry isolation (`stage11Perf.test.tsx`, `telemetry.e2e.ts`).

### 13. Gaps / stubs / defects found

1. `verify_isolation.py` / `stage00.py:35-36` / `isolation.verify :184-197`: **hard-coded pass**, `checked: 2645` is a constant; `--legacy-only` and `--snapshot` flags are dead; docstring ("complete, fail-closed") is false. `STAGE00_REPORT.md`, `FINAL_INTEGRITY_REPORT.md` isolation claims rest on it. `protected_paths.json`/`protected_all_paths.json` unused.
2. `local_models.py --verify` has a **write side effect** (auto-builds the CC0 `.npz`, `:221-247`) and pins only the FLAME-Open `.pkl`; registry scope depends on the sibling `roop-ultimate` checkout (`legacy_root`) — a machine without it cannot `--verify` the 95 registered models.
3. UI "speech-lip lock" and "scene adaptation" are localStorage flags with no consumer (`WorkstationControls3D.tsx:18-19`); the only temporal control is the global cutoff.
4. UI has no yaw-limit/refusal reason display, no >90° bin, no temporal jitter graph, no frequency-band/multiband control; `ClipPathSplitReveal.tsx` is dead code (no importer outside tests/its own file).
5. `test_real_target_frames` + three other real-profile tests skip: fixture archives not tracked (`docs/EXTREME_POSE_3D.md:108-113`); no `conftest.py`; extreme-pose evidence is synthetic only.
6. `test_flame_and_liveportrait.py` hardware gate: acceptance command 2 pass / 1 fail (`docs/REMEDIATION_STAGE16.md:1-5`); real SMIRK/EMOCA ONNX and MediaPipe task absent.
7. Temporal stabiliser misses its own >70 % target (28.0 % at 0.25), CUDA/NVIDIA optical-flow path unexercised (`docs/TEMPORAL_STABILITY.md`).
8. Performance: 0.26 fps (CUDA, 512², 15 frames) and 0.57 fps benchmarks while `STAGE15_VALIDATION.json` says READY FOR RELEASE (gate does not assert a throughput floor) — inconsistent with real-time claims.
9. Reliability: 30/60-min soaks never completed (security software termination); only scaled soaks (`docs/reliability/*.json`). Full Python suite once exceeded 600 s (`docs/reviews/roop3d-bug-audit-2026-11-08.md`).
10. Repo hygiene: `AGENTS.md` still names `app/roop/` and `app/api.py` as live while `docs/STAGE*.md` repeatedly state `app/roop/` "is not used"; `start_roop3d.js` duplicates `start.js` [INFERENCE]; `docs/reviews/design_draft.md` duplicates `docs/designs/roop_3d_workstation_modernization.md` (both 436 L) and `concepts_brief.md` exists at root and in `docs/reviews/`.
11. `docs/roop_ultimate_angle_swapping_analysis.md` (425 L, read-only research, authored against roop-ultimate HEAD `9a0f5d5`, not the audited `71ebb65`): states Ultimate's default angle robustness is 2D similarity crop + per-pose identity vector + coupled smoothing + pose-gated restoration + frame-space landmark hull mask + BORDER_REPLICATE; 3D source-bank/frontalisation are **off in live config** (`use_source_bank:false`, `use_3d_recon:false`, `use_frontalization:false`) — useful reference for the cross-repo comparison report.

### Coverage

Slice files in inventory: 754 | files accounted for: 754 | omitted: none
(app/roop_3d_ui/ 359 + app/roop_3d/tests/ 202 + app/roop_3d/audit/ 49 + docs/ 98 + root/other 46; §6 tests 202 rows, §7 UI-test rows + §8 non-test UI rows = 359, §9 49, §10 98, §11 46.)


<!-- ===== Orchestrator addendum: sidecar files not tabulated by the slice auditors ===== -->
## Addendum — two sidecar metadata files

| File | Purpose |
|---|---|
| `app/roop_3d/reconstruction/assets/makehuman_head_v1.npz.json` | Provenance sidecar of `makehuman_head_v1.npz`: schema `roop3d.makehuman_head.v1`, CC0-1.0, MakeHuman commit `a8bc2d54`, asset digest `6f125518…8ee4`, 4289 vertices / 8524 faces, 50 shape axes + 10 expression axes, OpenCV head frame in metres (x right, y down, z away from camera; R=I frontal). |
| `app/roop_3d/rendering/assets/stage10_cc0_probes.npz.json` | Provenance sidecar of `stage10_cc0_probes.npz`: ambientCG CC0 catalogue query/licence URLs, note that the skin pore tile is procedural (not misattributed), bundle SHA-256 `6666c53b…b845`. |

Union path coverage after the addendum: **1224/1224**.
