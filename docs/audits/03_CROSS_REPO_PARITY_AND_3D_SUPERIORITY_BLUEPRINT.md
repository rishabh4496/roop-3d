# 03 — Cross-repo parity & 3D-superiority blueprint

Inputs: report 01 (roop-ultimate `71ebb65`, 2118 files) and report 02 (roop-3d, 1224 files). Everything below is anchored to a file the auditors read. "Wave 2" items are what this mission implements; "Out of scope" items are stated honestly rather than faked.

Execution note: the requested per-agent model/effort bindings (`gpt-6-luna`, `gpt-6.1-sol`) are not selectable from this harness; the swarm ran on the harness default model with the role split preserved (auditor / 3D / temporal+quality / ops+QA / orchestrator+UI).

## 1. What roop-3d already beats roop-ultimate on (do not regress)

| Capability | roop-3d | roop-ultimate |
|---|---|---|
| Mesh-fit identity asset, UV albedo fusion, z-buffer UV visibility | real (`reconstruction/uv_synthesizer.py:373-552`, `rendering/vertex_raster.py`) | none (`fit_source`/`render_from_coefficients` stubs) |
| SH-9 illumination fit | real (`reconstruction/photometry.py`, `tracking/lighting.py:53`) | none |
| Live engagement of dense 3D at \|yaw\|>30° / \|pitch\|>25° | `engine/volumetric/session.py:121-132,1029-1050` | `use_3d_recon` off, no-op for inswapper family |
| 3D viewport, pose radar, identity inspector | `components/three/*`, `facemgr/FacesetInspector3D.tsx` | none |
| Atomic checkpoint + GOP stream-copy, 10-state persisted queue | `queue_manager.py`, `segment_dispatcher.py` | `queue.json`, sha256 segments |

## 2. Parity / superiority gap matrix

Severity: **H** = visible artifact or false claim, **M** = capability absent vs spec, **L** = hygiene.

| # | Axis | Gap (evidence) | Sev | Wave 2 action | Owner / files | Acceptance |
|---|---|---|---|---|---|---|
| G1 | 1 Angle | Live pose = 5-pt geometric (`engine/detection/dense_landmarks.py:102-106`); PnP only offline | H | `reconstruction/pose_solver.py`: dense-landmark PnP (RANSAC → LM refine), metric 6DoF `R∈SO(3), t∈ℝ³`, flip-ambiguity resolution, reprojection RMS; `dense_landmarks.py` uses it when ≥68 landmarks, records `pose_source` | 3D agent | synthetic ±75° yaw grid recovered ≤1.5° |
| G2 | 1/2 Bins | Four incompatible pose-bin grids (`faceset_3d.py:20`, `target_lock.py:24`, `pose_matrix.py:24-27`, `pose_selector.py:19-31`) | M | `reconstruction/pose_bins.py`: one spherical grid (frontal, ±15, ±30, ±45, ±60, ±75+ yaw × pitch bands), great-circle reference weighting; `.fsz` coverage/views expose `spherical_bin` | 3D agent | bin of (yaw,pitch) deterministic; weights sum 1 and peak on nearest bin |
| G3 | 2 3D | `mesh_raycaster.py` claims Embree/BVH; is a hard-coded index mask (`:42-52,127-200`) | H | Replace with vectorised depth-buffered visibility (back-face cull + z-test with tolerance), per-vertex visibility + soft image mask; delete dead `use_simd`/`bvh_enabled`; update `heterogeneous_scheduler.py` consumer | 3D agent | at ±75° yaw far-cheek vertices occluded, near-side visible; frontal ≥ 0.95 visible |
| G4 | 2 3D | Identity-asset endpoint does not expose disentangled α/β/pose/albedo | M | `identity_asset.py` / `server/identity_assets.py`: `parametric` block {identity shape α, expression β, camera pose R,t, albedo UV summary} | 3D agent | JSON contract test |
| G5 | 3 Quality | `NeuralRelighter3D` has no neural net (`relighting_3d.py:287-345`); SH + Blinn-Phong is real | L | Rename-free: document + expose SH coefficients in telemetry; no model claim | 3D agent | docstring corrected |
| G6 | 3 Quality | Warp = bilinear `cv2.warpAffine` (`boundary_fusion.py:34`) | M | `pipeline/subpixel.py`: float-matrix crop via `cv2.remap` with Lanczos4/cubic, no integer bbox rounding, affine+perspective | Temporal agent | sub-pixel shift round-trip error < integer-crop baseline |
| G7 | 3 Quality | Laplacian blend exists; no high-frequency retention; no local skin-tone transfer | M | `compositor/frequency_blend.py`: Laplacian-pyramid split; low bands from swap/restorer, high bands from target (`retain_hf` ∈[0,1]) gated by mask; masked Lab Reinhard transfer | Temporal agent | HF energy of result within tolerance of target at `retain_hf=1`; identity low-freq preserved |
| G8 | 3 Quality | Default XSeg `xseg_3.onnx` has no manifest entry (`masks.py`, manifests) | M | Resolve default to a manifest-known model or fail with explicit code | Temporal agent | test: default resolves |
| G9 | 4 Temporal | No ByteTrack low-score second pass (`face_tracker.py:46,333`); ultimate has an unused one | M | Two-pass association (high ≥ τ_h, then low-score dets vs unmatched tracks, IoU only), keep ArcFace ladder for re-ID | Temporal agent | occluded-detection continuity test |
| G10 | 4 Temporal | Scene cut thumbnail-only and **not wired into compositor render path** (R3_2 gap 8) | H | `tracking/scene_cut.py`: HSV-hist Bhattacharyya + SSIM drop + pose-velocity spike; flash vs cut discrimination; fan-out reset to filters, tracker, mask-flow | Temporal agent | synthetic cut and flash cases |
| G11 | 4 Temporal | Mask edge flicker; temporal detail stabiliser 28% vs 70% target (`docs/TEMPORAL_STABILITY.md`) | H | `compositor/mask_flow.py`: dense flow t-1→t warps previous alpha, confidence-weighted blend, reset on cut; wired into compositor matte | Temporal agent | moving-edge synthetic: temporal alpha variance ↓ vs unstabilised |
| G12 | 4 Temporal | Duplicate `OneEuroFilter` implementations (`temporal.py`, `temporal_stabilizer.py`, `reenactment.py:145`) | L | Single implementation | Temporal agent | one definition |
| G13 | 5 CLI | No headless CLI, no Rich TUI, no `--json-logs` (`server/__main__.py:67`) | H | `roop_3d/cli.py` (`render`, `--headless`, `--json-logs`, `--diagnostics`), `telemetry/tui.py` Rich dashboard (stage bars, FPS p50/p95/p99, provider, VRAM, CPU/RAM, ETA); plain fallback when Rich absent (`tui` extra) | Ops agent | CLI tests via subprocess/in-proc |
| G14 | 5 Diagnostics | Two 14-section reports exist (`telemetry/runtime_report.py`, `server/diagnostics.py`) | L | CLI `--diagnostics` renders the existing 14 sections + model SHA256 + 3D asset integrity; no third report | Ops agent | exactly 14 sections |
| G15 | 6 Compute | TRT opt-level 3 vs 5 (`core/ort_session.py:38` vs `trt_session_builder.py:242`); INT8 off | M | one shared constant; INT8 as explicit opt-in that requires a calibration cache and otherwise returns a named refusal (reference repo rejected INT8 on Ada) | Ops agent | test: refusal reason, level consistent |
| G16 | 7 Queue | Three state enums, two parallel stacks (`jobs/schema.py:27`, `queue_manager.py:47`, `queue/lifecycle.py:28`; `server/app.py:86-100`) | M | Published mapping canonical→10 spec states (queued, validating, slicing, tracking, swapping, restoring, merging, remuxing, completed, failed/cancelled); fix "10" docstring; resume from atomic checkpoint test | Ops agent | mapping total over all enum members |
| G17 | 7 IO | Stage-17 runner drops subtitles/metadata, first audio track only (`video_job_runner.py:12-14,217`) | M | map all audio+subtitle streams, copy metadata/chapters | Ops agent | ffprobe-level test on a synthetic container |
| G18 | Integrity | `verify_isolation.py --report` is hard-coded pass (`stage00.py:35-36`, `isolation.py:184-197`) | **H** | Real checks: no import/sys.path into sibling checkout, no sibling paths in write targets, venv inside checkout, git origin identity, no tracked artefacts | Ops agent | fails on injected violation |
| G19 | Integrity | `local_models.py --verify` writes files, depends on sibling checkout | M | `--verify` strictly read-only; missing sibling → reported, not crash | Ops agent | no file mutated |
| G20 | 8 UI | No live 6DoF/temporal-stability HUD, no refusal-reason display, no >90° bin, dead `ClipPathSplitReveal.tsx`, no-op `scene adaptation`/`lip lock` toggles. **Correction after orchestrator code read:** the 3D inspector already ships lit/wireframe/albedo/normals/depth/cavity/SH buffer views (`components/three/PoseGeometryViewer.jsx:979`, `viewportTools.ts BUFFER_VIEWS`), so no inspector rebuild is needed | M | Orchestrator: `stores/trackingStream.ts` + `components/hud/TrackingTelemetryPanel.tsx` (yaw/pitch/roll, quaternion, translation, pose source, spherical bin, reprojection RMS, self-occlusion, stability/jitter graph with cut+flash markers) mounted in the pipeline pill; `TrackingTelemetryEvent` extended (additive, optional fields) | Orchestrator | Vitest + `tsc --noEmit` + eslint |

## 3. Mathematical / pipeline specification

**3.1 6DoF solve.** Pixel `x_i ≃ K(R X_i + t)`, with `X_i` dense/68-landmark points of the fitted head (metres, OpenCV frame: x right, y down, z away). `K` from an assumed 35 mm-equivalent focal unless supplied. Initial `(R,t)` by `solvePnPRansac` (SQPNP/EPnP, reprojection threshold 3 px, confidence 0.99); refine on inliers by `solvePnPRefineLM`; require `t_z>0`; resolve flip ambiguity by evaluating the mirrored-depth hypothesis and a temporal prior (previous `R`): pick the lower of `reproj_rms + λ·angle(R, R_prev)`. Euler `(yaw,pitch,roll)` from `R = R_y R_x R_z`; `q = (w,x,y,z)` canonical sign (`w ≥ 0`).

**3.2 Spherical bins.** View direction `d = R·(0,0,−1)`; yaw/pitch bin centres on the lattice `yaw ∈ {0,±15,±30,±45,±60,±75,±90}`, pitch `{−30,−15,0,15,30}`. Reference weight `w_k ∝ exp(−θ_k²/2σ²)`, `θ_k` = great-circle angle between view direction and bin `k`'s centre, `σ=18°`, top-3 renormalised.

**3.3 Self-occlusion.** Project vertices `u = K(RX+t)`; rasterise depth into a `H×W` float buffer (nearest-z via `np.minimum.at` splat with triangle-centroid supplement); vertex visible iff `z_v ≤ z_buf(u_v)+ε` (`ε=8 mm`) **and** face-normal·view `> cos 80°`. Output: per-vertex bool + soft image mask (convex-fill of visible triangles, 3 px feather).

**3.4 SH relight.** `L(n)=Σ_{l≤2} c_lm Y_lm(n)` (9 RGB coeffs) fitted by regularised LS to target-face luminance given albedo prior (existing `photometry.py`); relit swap `I = A·L(n_target) + k_s·spec`; Blinn-Phong highlight matched to target by luminance-ratio.

**3.3b Temporal filters.** One-Euro per channel: `α = 1/(1+τ/Te)`, `τ=1/(2π f_c)`, `f_c = f_min + β|ẋ̂|`. Rotation: `q_t = slerp(q_{t−1}, q_meas, α(ω))` with `ω` = angular speed (°/s) as the adaptive-cutoff driver; hard reset ≥900°/s or on cut.

**3.5 Scene cut / flash.** Per frame: `H` = Bhattacharyya distance of 8×8×8 HSV histograms; `S` = 1 − SSIM on 64×64 luma; `V` = face pose-velocity. Cut iff `(H>0.45 ∧ S>0.35) ∨ (H>0.30 ∧ S>0.30 ∧ V>spike)`. Flash iff global luminance jumps >Δ for ≤2 frames and returns (`H` high, `S` low, structure-preserved) → flush filters but keep tracks.

**3.6 Mask flow.** `α̃_t = (1−c)·α_t + c·warp(α_{t−1}, F_{t−1→t})`, `c = c_max·conf`, conf from forward-backward flow consistency `exp(−‖F+F⁻¹∘F‖²/σ²)`; `c_max=0.6`; `c=0` after cut.

**3.7 Frequency-separated blend.** Gaussian/Laplacian pyramid (5 levels). `L_out^k = m_k·L_swap^k + (1−m_k)·L_tgt^k` for `k ≥ k_hf` takes target HF with weight `retain_hf`; base level gets masked Lab Reinhard transfer `(x−μ_s)·σ_t/σ_s+μ_t` computed inside the mask.

## 4. Telemetry contract (additive v2 fields on `tracking.telemetry` frames, `/ws/v1/pipeline/status`)

The v2 event remains a flat JSON object to preserve the existing WebSocket event shape. `schema_version: 2` identifies this
additive contract; every new measurement is optional and omitted/null when unavailable. It uses the existing calibrated weak-
perspective camera where available; it does not imply metric translation unless the upstream `FaceObservation.pose_6dof` exists.
The optional `HeadTrackingTap` adapter forwards its actual PnP mapping to the event; the standalone video tracker does not
derive metric depth or dense-PnP provenance by itself. The render job runner does not currently invoke this tap.

```
schema_version: 2
yaw_deg, pitch_deg, roll_deg: number | null
quat_wxyz: [w,x,y,z] | null                # derived from measured rotation matrix, normalized, w >= 0
translation_px: [tx,ty] | null              # measured weak-perspective image translation
pose_scale_px_per_m: number | null          # model scale; not metric depth
translation_m, reproj_rms_px: number | null # forwarded only from a measured upstream PnP result
pose_source, spherical_bin: string | null   # forwarded only from the real face observation
jitter_px, one_euro_cutoff_hz: null          # unavailable from this producer
stability_score, mask_flow_confidence, self_occluded_fraction: null
shot_cut: bool                              # observed tracker cut flag
scene_flash: null                           # this producer does not distinguish flashes
shot_cuts: integer                          # cumulative cut counter
```

UI treats every v2 measurement as optional and remains compatible with older frames.

## 5. Out of scope (stated, not faked)

* A neural network that *swaps identity natively from 3D parameters* needs trained weights not present locally; identity transfer stays on INSwapper/HifiFace-class nets, with 3D conditioning pose/geometry/texture/visibility (report 02 §0).
* Real TensorRT INT8 calibration and RTX 3060 validation need hardware runs; Wave 2 adds the guarded opt-in path only.
* Real-footage extreme-yaw fixtures are absent; Wave 2 evidence is deterministic synthetic geometry.
* 30/60-minute soaks (R3_3 gap 9) are not re-run.
