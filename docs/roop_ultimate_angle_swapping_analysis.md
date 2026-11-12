# Roop Ultimate — how it swaps faces across head angles, and how to port it into Roop 3D

Read-only research. Nothing inside `roop-ultimate` was modified. Authored by the model `anthropic/claude-sonnet-5-5` (taken from the session's project-context metadata).

Source studied: local checkout `G:\pinokio\api\roop-ultimate` at HEAD `9a0f5d5` ("feat(roop3d): isolate Stage 01 service and guard legacy paths"). Roop 3D side: `G:\pinokio\api\roop-3d` at HEAD `d2a5609`.

Conventions
- Paths in section 1-3 are relative to `roop-ultimate/app/` unless noted (`roop/...` = `app/roop/...`). Roop 3D paths are relative to `roop-3d/app/roop_3d/`.
- **[C]** = confirmed by reading the cited code. **[I]** = inference (not directly proven by code or by a run). Nothing here was executed; no number below is a measurement I made. Numbers quoted from Ultimate's CHANGELOG/comments are marked "(Ultimate's own claim)".
- Pose convention in Ultimate (`roop/angle_portfolio.py:9-22`, `roop/face_util.py:3012-3014`): `+yaw` = nose toward the viewer's right, `+pitch` = nose tipped DOWN, roll = `atan2(r1y, r2y)`.

---

## 0. TL;DR — what "angle swapping" actually is in Ultimate

Ultimate does **not** re-render the source face in 3D at the target angle in its default embedding-swapper path (hyperswap/inswapper family). Angle robustness comes from six cheap, classical mechanisms stacked together [C]:

1. **One canonical, angle-independent crop.** The target face is always warped by a *2D similarity* fit of the 5 detector keypoints to a fixed model template (`estimate_norm`, `roop/face_util.py:3477-3512`), so eyes/nose/mouth land at the same crop pixels at every yaw/pitch/roll. Roll is removed exactly by the similarity; no shear is possible. The swap net itself (hyperswap, 256 px, `roop/processors/FaceSwapInsightFace.py:148-157`) handles pose from the target crop.
2. **Pose-dependent identity vector, not pose-dependent pixels.** The source identity (512-D ArcFace vector) is *selected/blended per target pose* — pose-cell centroid, inverse-distance top-3 SLERP, or a profile/frontal mix for |yaw|>35° — never pixel-blended across source views (`roop/source_portfolio.py:310-336`, `roop/faceset_v2.py:1264-1384`, `roop/stage0_preflight.py:106-121`).
3. **Temporal stabilisation of everything that defines geometry**, coupled: 5-pt kps and 106-pt landmarks share one velocity-adaptive EMA factor (`roop/temporal_smoother.py:149-386`); crop texture (HF band, flow-warped), enhancer crop and mask each have their own per-track temporal filter.
4. **Pose-gated corrections**: plate eye/mouth restoration fades out between 25° and 38° off-axis (`roop/procmgr_masking.py:2000-2002`, `2096-2099`); the far (foreshortened) eye of a restorer output is pulled 25 % back toward the swap on |yaw|>35° (`roop/source_portfolio.py:339-364`).
5. **A mask that is pose-aware by construction in frame space**: crop-inscribed ellipse ∩ convex hull of the *smoothed* 106 landmarks (+ forehead extension along the head's own up-axis), eroded by ¼ of the feather radius and Gaussian-feathered (`roop/procmgr_masking.py:629-681`, `1037-1125`).
6. **Out-of-frame / border handling that never makes black**: `BORDER_REPLICATE` on the forward crop and on the paste-back warp (`roop/face_util.py:3517-3542`, `roop/procmgr_masking.py:799-834`).

Optional extras that exist but are **off in the live config** (`app/config.yaml`): pose-aware source bank (`use_source_bank: false`, line 243), 3D source-crop shear/flip (`use_3d_recon: false`, line 240; also gated to image-source swappers only), target frontalisation (`use_frontalization: false`, line 241). `app/default_config.yaml` turns the first two on (lines 161, 164) — i.e. the repo ships two different defaults [C].

Dead / not in the live path [C]: `roop/geometric_alignment.py` (weighted-Umeyama solver, 3-point profile anchors, `OneEuroFilter1D`, `TemporalGeometryStabilizer`) has **no non-test caller** (grep over `app/` excluding tests); `core/fused_postprocess.py` states "Not wired into the render" (docstring, line 45-47); `post_swap.py` is only AI upscaling / frame interpolation after the render (`post_swap.py:1-60`), unrelated to angles. `canonicalize_face_alignment` explicitly rejects the ear/nose/chin profile template because it "displaced the eyes and mouth, leaving doubled features" (`roop/face_analyser.py:695-707`); the 3-point profile solvers (`face_analyser.py:386-490`, `493-542`, `640-692`) are reachable only from the legacy `roop/processors/frame/face_swapper.py` (I did not verify whether that module is loaded by the live app) [C for the call graph, I for liveness].

---

## 1. Per-frame pipeline, detection → paste-back (with citations)

Live default per `app/default_config.yaml`/`config.yaml`: detector `retinaface_r50` (`default_config.yaml:20`), swapper `hyperswap` (131), `temporal_detection: true` (144), `subsample_upscale: 256px` (130), `color_transfer_mode: lct` (12), restorer `Restore Ultra` (106), `face_mask_blend: 25` (35), `stabilize_*` on (114-129).

### 1.0 Pre-pass (temporal detection) — runs before frame 0 is swapped
- `ProcessMgr.process_frame` (`roop/ProcessMgr.py:3438`) → `swap_faces` (3880). With `_temporal_mode`, faces for each frame are *read from the pre-pass cache* (`ProcessMgr.py:3946-3959`) instead of detected in the hot loop.
- Pre-pass `procmgr_tracking._build_temporal_faces` (`roop/procmgr_tracking.py:3091`): detect → Hungarian association (`roop/association_cost.py:139-171`, cost below) → gap-fill by interpolation (≤10 frames) → Kalman coasting → **landmark smoothing** (`procmgr_tracking.py:3310-3423`; section 2.5) → in-plane roll resolution (3433) → pose-aware source decisions with hysteresis (3452-3665; section 2.3).
- Track association cost (relevant because angle changes break identity matching; `association_cost.py:139-171`):

  ```
  C = l_id · r(c) · (1 − cos_sim) + l_iou · (1 − GIoU) + l_kpt · mean_i‖k_i − k'_i‖ / √(bbox_area) + l_pose · mean(|Δyaw|,|Δpitch|,|Δroll|) / 90
  l_id=1, l_iou=1, l_kpt=0.5, l_pose=0.25;  r(c)=1 for c≤0.30, →0.25 at c≥0.60 (linear)
  ```
  (`CostWeights`, `association_cost.py:99-119`). Note the file docstring says the legacy cost `(1−IoU)+cosine` is what `_consume` "has always used"; `composite_cost` is the Stage-16 candidate. Which one is live I did not trace [I].

### 1.1 Keypoint refinement, pose, alignment
| Step | What | Where |
|---|---|---|
| a | If 68-pt landmarks exist, replace the 5 detector kps by `[mean(lm[36:42]), mean(lm[42:48]), lm[30], lm[48], lm[54]]` ("68-point model is more stable than the 5-point regressor at angles") | `roop/face_util.py:676-703` [C] |
| b | Smoothed kps from the pre-pass replace raw kps (and in the sequential path `_apply_stab` → `KpsStabilizer`+`AdaptiveLandmarkSmoother`) | `ProcessMgr.py:3831-3859`, `3798-3829` |
| c | Alignment: `canonicalize_face_alignment(plate, face, subsample_size, template, dst=crop_buffer)` → `M = estimate_norm(kps, size, mode)`, crop = `warpAffine(plate, M, (S,S), INTER_CUBIC, BORDER_REPLICATE)` | `ProcessMgr.py:5731-5739`, `roop/face_analyser.py:695-735`, `roop/face_util.py:3477-3542` |
| d | Pose for gating: `solve_pose_5pt(kps)` (weak-perspective, §2.1) → `_adaptive_yaw/_pitch/_roll` | `ProcessMgr.py:5803-5809` |
| e | Head angles with the jaw removed (`solve_pose_jaw_5pt`) are used only by mouth/eye fades, because an open mouth reads as pitch (measured by Ultimate: closed 0.0° → −28° at full opening) | `ProcessMgr.py:5814-5843`, `roop/face_util.py:3203` |

### 1.2 Identity selection
- `use_source_bank` path: `fs.select_pose_aware_reference(...)` with hysteresis (`ProcessMgr.py:5920-5980`; math in §2.3). Not active in the live `config.yaml`.
- Embedding swappers: `_stage0.pose_conditioned_input` → `pose_embedding_for_target` (V2 pose cell / SLERP) (`ProcessMgr.py:6128-6139`, `roop/stage0_preflight.py:106-121`, `roop/processors/FaceSwapInsightFace.py:22-82`). Stage-0 freezes the answer for all 3×3×3 (yaw,pitch,roll) cells before frame 0 (`stage0_preflight.py:18-38`); it declines whenever a per-frame bank/portfolio/quality engine could re-select.
- If the source carries an `angle_portfolio` (built by `POST /api/angle-scan/source-portfolio`, `app/routes_angle_scan.py:586-603`): `source_portfolio.route` replaces the vector (`ProcessMgr.py:6142-6166`, §2.2).
- Optional: 3D source-crop shear/flip (image-source swappers only; `ProcessMgr.py:6028-6118`, `roop/face_3d_recon.py:172-220`); target frontalisation before the swap (`ProcessMgr.py:6187-6217`; off).

### 1.3 Swap, colour, mask
1. Swap net runs on the 256² crop (`ProcessMgr.py:6299-6420`; pixel-boost tiling `implode/explode`, default 1 tile for hyperswap@256).
2. Colour transfer of the swapped crop toward the **aligned target crop** (`ProcessMgr.py:6454-6461` → `procmgr_color.py:331-396`, §2.6).
3. Mask processor(s) on the aligned crop → `img_mask` (1 = keep plate, 0 = swapped); `_recover_undersized_mask` floor; optional guided filter; per-track `MaskStabilizer`; ownership/occlusion; then **single** crop-space composite `fake = target·(1−m) + plate_crop·m` (`ProcessMgr.py:6496-6549`, `roop/procmgr_masking.py:1281-1742`, `1744-1782`).

### 1.4 Restorer (sharpness), crop-space temporal, post colour
- Restorer on the swap crop (256 → 512). Optional template re-alignment (`enhancer_align`, off): `A = M_e · M⁻¹`; the ring outside the swap crop is filled from the plate (not replicated edge), cover mask feathered with Gaussian σ=2 (`ProcessMgr.py:6621-6673`).
- `protect_restorer_output` on dark targets, Restore-Ultra frequency recombine, **far-eye damping** on |yaw|>35° (`ProcessMgr.py:6738-6763`).
- `EnhancerStabilizer` (6776-6782), `HighFrequencyFlowStabilizer` (6797-6804), expression restore, detail transfer (6855-6864), post-enhance colour match (6877-6890), light harmoniser (6893-6906), merger post-ops (6916-6924), manual masks, **temporal identity low-band blend** (7039-7064), identity detail (7074-7122), temporal quality QC (7128-7217).

### 1.5 Paste-back
- `fake_frame` resized to 512 (`INTER_CUBIC`, `ProcessMgr.py:7219-7224`), `paste_upscale(fake, enhanced, M, frame, scale_factor, mask_offsets, face_landmarks=lm106, face_kps=kps, ...)` (`ProcessMgr.py:7283-7288`, `roop/procmgr_masking.py:629-919`): math in §2.4.
- Post-composite in frame space: mouth restore (7332-7348), eye restore (7355-7367), expression/gaze fidelity, lip-sync, teeth/oral repair, jaw reshape, verify-swap gate (7548-7578). All mouth/eye overlays fade by yaw/pitch (§2.7).

---

## 2. The angle logic — exact algorithms and equations

### 2.1 Pose from 5 keypoints (weak perspective, closed form) [C]
`roop/face_util.py:2818-2871, 2923-3017`.

Reference head `X ∈ R^{5×3}` = `_REF3D_68` points {mean(36:42), mean(42:48), 30, 48, 54}, centred and y-flipped to image axes (`_REF5_IMAGE`). With `x ∈ R^{5×2}` the observed kps centred on their centroid:

```
A  = pinv(X) · x                         (3×2;  precomputed _REF5_PINV; 0.79 µs vs 6.5 µs lstsq)
a0 = A[:,0] , a1 = A[:,1]                (rows of s·R[:2])
n1 = ‖a0‖ ;  r1 = a0 / n1
r2 = a1 − (a1·r1) r1 ;  n2 = ‖r2‖ ;  r2 /= n2          (Gram–Schmidt)
r3 = r1 × r2
pitch = asin(clip(−r3_y, −1, 1))
yaw   = −atan2(r3_x, r3_z)
roll  = atan2(r1_y, r2_y)
scale = (n1 + n2)/2
```
Ultimate's claim: error <0.05° over yaw 0–90 × pitch ±40 × roll ±30 on synthetic projections (docstring `face_util.py:2948-2951`).
`solve_pose_jaw_5pt` (`face_util.py:3203`) adds a 6th DOF `j` (mouth corners slide along the head's own eye→mouth axis), `X(j) = X + j·J`, `J` = both corners shifted by `((m_L+m_R)/2 − (e_L+e_R)/2)`, clamped `j ∈ [−0.35, 0.90]`, 5 Gauss–Newton steps, two-basin scan step 0.1 (constants `face_util.py:2919-2920, 3034, 3055`).

Off-axis angle (single number for "how turned"): `offaxis = acos(cos(yaw)·cos(pitch))` (`face_util.py:3366-3373`; also `pose_source_selector._off_axis`, line 86).

Stage-1/2 scan pose (target LUT) uses a **perspective** variant: EPnP on the same 5-pt reference + one LM refinement, camera `f = max(w,h)`, then the rotation is re-expressed relative to the line of sight `R ← R_ray(t)ᵀ R` before `_decompose_projection` (`roop/pose_quality.py:131-168`). Frontal-equivalent IOD = `f · |ref eye dist| / t_z` (167) so profiles are not rejected as "too small".

Neutral-pitch correction (anatomy offset of 5-pt pitch): `neutral = median{pitch_i : |yaw_i| ≤ 25°}` per person/track; bins use `pitch_rel = pitch − neutral` (`roop/angle_portfolio.py:139-159`; observed: a level passport photo solves to 19° "up").

### 2.2 Source-view selection #1 — the "angle portfolio" (identity vector routing) [C]
Two halves, deliberately separate people (`roop/source_portfolio.py:1-33`): the **source** person gets a portfolio; the **target** person gets only a per-frame pose LUT. The *target* portfolio (`roop/angle_portfolio.py`) is for reference-capture/UI only and "must NOT be fed to the swapper" (`source_portfolio.py:16-19`).

**Bin lattice** (`angle_portfolio.py:58-101, 109-136`; `p = −pitch_rel` = up-positive):

| Bin | yaw range | p range |
|---|---|---|
| 0 Frontal | [−10, 10] | [−10, 10] |
| 1 / 2 Quarter L / R | [−25, −10) / (10, 25] | [−10, 10] |
| 3 / 4 Half-profile L / R | [−45, −25) / (25, 45] | [−15, 15] |
| 5 / 6 Profile L / R | < −45 / > 45 | any |
| 7 / 8 Up / Down | [−20, 20] | > 15 / < −15 |
Gaps exist (e.g. yaw 0 at p=12) → `angle_bin` returns `None`.

**Distance to a bin** (closed-rectangle distance): `d = hypot(max(y0−y,0,y−y1), max(p0−p,0,p−p1))` (`angle_portfolio.py:162-168`).

**Target-portfolio selection** (`angle_portfolio.py:203-277`): best per bin = `max over (composite_score, id_similarity, −frame_idx)`; empty bins borrow the nearest unused candidate within `MAX_FILL_DEG = 10°` (global closest-pairs-first); user overrides win. Quality `Q = 0.35·sharp + 0.20·norm_iod + 0.15·illum + 0.30·id − 0.01·|roll|` (renormalised when id missing; `pose_quality.py:238-268`; `sharp = Laplacian var on a 160² gray / 350`, `W_*` at line 91). Fused reference embedding = L2-normalised score-weighted mean of the bin 0/1/2 embeddings (`angle_portfolio.py:280-309`).

**Source portfolio build** (`source_portfolio.py:130-192`): per face `pose = solve_pose_5pt(kps)`; drop eyes-closed (EAR<0.20, both eyes) / no pose / between-bins; per bin keep max `score = det · (0.5 + 0.5·clip((‖e‖−14)/(26−14), 0, 1))`; fused vector:

```
e_fused = unit( Σ_{b∈{0,1,2}} w_b · e_b / Σ w_b ),   w_b = max(score_b, 1e-3),  e_b unit
```

**Routing, per target face per frame** (`source_portfolio.py:310-336`):

```
(yaw, pitch_up) = LUT[frame] if bbox-IoU ≥ 0.3 (or nearest scanned frame within one stride, IoU ≥ 0.15) else solve_pose_5pt(kps)  (~11 µs)
if |yaw| > 35° and source has the matching profile bin (5 if yaw<0, 6 if yaw>0):
        e = unit( 0.7·e_profile + 0.3·e_fused )
else:   e = e_fused
```
Constants `PROFILE_YAW_DEG=35, PROFILE_WEIGHT=0.7` (`source_portfolio.py:46-47`). The vector keeps its shape, so the TensorRT binding is unchanged. **Observation [C]:** the 35° switch has no hysteresis or ramp in `route`; its stability relies on the already-smoothed pre-pass kps. A target hovering at |yaw|≈35° can flip between e_fused and the 0.7/0.3 mix every frame [I: not measured].

**Far-eye damping** (`source_portfolio.py:278-288, 339-364`): far eye = the one nearer the nose in x: `0 if |x0−x2| < |x1−x2| else 1`. Pull enhanced toward pre-enhance inside a feathered ellipse:

```
rx = max(4, 0.45·max(iod, 0.18·W)),  ry = 0.6·rx,  mask = GaussianBlur(ellipse, σ = max(1, 0.35·rx)),  mask /= max(mask)
a  = 0.25 · mask ;   out = enh·(1−a) + pre·a            (FAR_EYE_DAMPING = 0.25)
```

### 2.3 Source-view selection #2 — pose-aware bank (V2 `.fsz`) and pose-cell vectors [C]
**Scoring** (`roop/pose_source_selector.py:306-334`) — lower is better:

```
dy = min(2, |Δyaw|/55),  dp = min(2, |Δpitch|/42),  dr = min(2, wrap180|Δroll|/55)
pose_cost  = 0.52·dy + 0.28·dp + 0.20·dr                       (missing source pose: 1.25)
pose_w     = 0.50 + 0.18·target_pose_confidence
total = pose_w·pose_cost + 0.18·(1−Q_src) + 0.10·min(1,expr_d) + 0.08·min(1,light_d) + 0.09·min(1,geom_d) + 0.05·scale_d
Q_src = 0.65·quality + 0.35·identity_conf ;   scale_d = min(1, |ln(ts/ss)|/ln 2)
expr_d  = mean min(2,|Δ|/s_k), s=(0.55,0.55,0.75,0.42) over (eyeL, eyeR, mouth-open, smile) ;  light_d over luminance/temperature/skin BGR
```
Target pose confidence `0.52·det + 0.28·lm_conf + 0.12·available + 0.08·size_conf` (then `0.9·c + 0.1·clip(face/(0.08·frame_min))`) (`pose_source_selector.py:220-231`).

**Hysteresis** (`pose_source_selector.py:347-396`): keep `previous` if `score[prev] ≤ best + 0.035` (`ROOP_POSE_SOURCE_SWITCH_MARGIN`). V1 banks: nearest by squared (yaw,pitch) distance with a 15° margin (225 deg², `roop/FaceSet.py:287-303`) plus a 5-frame candidate hold in the pre-pass (`procmgr_tracking.py:3643-3660`). Temporal identity (opt-in) adds `switch_frames=3`, `transition_frames=4` (`roop/temporal_identity.py:150-156`) and blends only the **low band** of the output with the previous frame (`blend_output`, 567-640) with retained weight `output_strength(0.35)·(1−min(1,2.5·motion))·(0.35+0.65·(1−conf))`, or `1−transition_alpha` during a source hand-off [C for form, I for default-enabled state; "Phase 6 layer is opt-in" `procmgr_tracking.py:3455-3457`].

**`needs_3d` triggers** (`pose_source_selector.py:372-396`): inverted (`|roll|≥135`); pose confidence <0.45; `pose_cost > 0.42`; between two bank yaws and `pose_cost > 0.12`; off-axis ≥72° and `pose_cost > 0.16`; `geom_d > 0.42`; `perspective_risk > 0.72`.

**Pose cells** (`faceset_v2.py:89-90, 276-300`): yaw edges ±25°, pitch edges ±15° → 3×3 cell; `pose_bin_embedding(..., fallback=False)` is deliberately all-or-nothing — "borrowing a neighbouring pitch cell … reintroduced the wrong-pose identity vector on profiles" (`FaceSet.py:168-181`).

**Continuous vector: inverse-distance SLERP over the 3 nearest source poses** (`faceset_v2.py:1264-1384`, used first by `pose_embedding_for_target`, `FaceSwapInsightFace.py:51-72`):

```
d_i  = hypot(yaw_i − yaw_t, pitch_i − pitch_t)           (exact match if d<1e-4)
w_i  = (1/(d_i + 1e-5)) / Σ_{top3} (1/(d_j + 1e-5))
v01  = slerp(v0, v1, w1/(w0+w1)) ;   v = slerp(v01, v2, w2)    ;  v ← v/‖v‖
```
Pose used here = `_adaptive_yaw/_adaptive_pitch` (the same `solve_pose_5pt` of smoothed kps; `ProcessMgr.py:5803-5809`, `FaceSwapInsightFace.py:51`).

### 2.4 Alignment, inverse, mask, paste-back equations [C]

**Similarity fit (template alignment).** `estimate_norm` (`face_util.py:3477-3512`) forces `skimage.transform.SimilarityTransform` ("AffineTransform … stretched foreheads on angled/close-up faces", comment 3480-3485) and fits the 5 kps to the template points `dst = swap_template_points(size, mode)` (3439-3474). Templates (normalised, ×size; `face_util.py:2711-2753`): `arcface` (insightface 112 base; for sizes %128 it scales `size/128` and shifts x by `8·ratio`; %512: `32·ratio`), `arcface_112_v1/v2`, `mtcnn_512`, `ffhq_512`. The swapper's template is used (hyperswap = `arcface`, `FaceSwapInsightFace.py:157`). Non-square-pixel video: fit in display space `x·SAR`, then map back `M_pixel[:,0] *= SAR` (3497-3509).
skimage's similarity estimate is the Umeyama least-squares solution [I: library behaviour, not re-read here]:

```
μx, μy centroids;  Σ = (1/n) Σ (y_i−μy)(x_i−μx)ᵀ = U S Vᵀ;  D = diag(1, sign(det U det Vᵀ))
R = U D Vᵀ ;  s = tr(S D)/σx² ;  t = μy − s R μx ;  M = [sR | t]  minimises Σ‖s R x_i + t − y_i‖²
```
(The repo's own dead `solve_similarity` in `geometric_alignment.py:441-487` is this exact weighted form with the same SVD; it is **not** called.) Roop 3D's `preprocess.similarity` solves the same objective as a 4-unknown linear LS (`a,b,tx,ty`) — equivalent [C for Roop 3D code, I for equivalence beyond degenerate cases].

**Crop:** `crop = warpAffine(img, M, (S,S), INTER_CUBIC, BORDER_REPLICATE)` (`face_util.py:3540-3542`). Comment: with the old zero border a face at the frame's left edge produced 22,449/65,536 black px (34 %) in the crop; replicate is "in-distribution" for the swap net (3519-3535).

**Inverse for paste:** `M_scale = M · scale_factor` (whole matrix incl. translation; `scale_factor` = enhancer upscale or 512/crop), `IM = cv2.invertAffineTransform(M_scale)` ⇒ `A⁻¹, −A⁻¹t` (`procmgr_masking.py:630-631`). Optional `output_face_scale`: `A' = sA, b' = s·b + (1−s)·f_c` with `f_c = IM·centre` (`_scale_paste`, 997-1035).

**Matte** (`paste_upscale`, `procmgr_masking.py:641-682`):

```
matte0   = ellipse in crop space, centre ((l+r)/2,(t+b)/2), semi-axes ((r−l)/2,(b−t)/2), l,r,t,b from mask_offsets[0..3] (default 0 ⇒ the inscribed circle of the 512 crop)
matte1   = warpAffine(matte0, IM, frame_size, INTER_LINEAR); outer 1-px frame border := 0
hull     = convexHull( lm106 ∪ 3 forehead points ),  forehead_s = top_brow_s + int(0.6·face_h)  along u = unit(eye_mid − mouth_mid)  (procmgr_masking.py:430-510)
           [ top_brow_s = max_u over lm[33:53];  face_h = top_brow_s − min_u(all);  forehead points at across-axis extents of the band along > top_brow_s − int(0.15·face_h) ]
lm_mask  = fillConvexPoly(hull), dilated by ellipse k = 2·e+1,  e = max(1, int(√(face_h·face_w)·blend/400))   (create_landmark_mask, 1127-1180)
matte2   = min(matte1, lm_mask)
```
`blend = mask_offsets[4] = face_mask_blend` (config 25; fallback `[0,0,0,0,20,10]`, `ProcessMgr.py:7225-7227`; `mask_offsets` built by `_mask_offsets_from_cfg`, `app/source_gallery.py:36-48`).

**Feather** (`blur_area` Gaussian mode, 1094-1125):

```
m = GaussianBlur(matte2, 3×3)
mask_size = int(√(mask_h·mask_w)) over pixels >127
blend_px  = max(1, int(mask_size·blend/200))        # e.g. 400 px face, blend 25 ⇒ 50 px
erosion   = max(1, blend_px // 4)                   # ellipse kernel (2·erosion+1)  — was blend_px//2: "profile face lost 15–20 px … a second nose, pale under-eye band" (1112-1119)
alpha     = GaussianBlur( erode(m), k = 2·blend_px+1 )        # OpenCV σ = 0.3·(blend_px−1) + 0.8 when σ=0
```
`mask_edge_mode: distance` alternative (not default): distance-transform + sigmoid ramp, constant width everywhere, `margin = adaptive_margin_px(√(bw·bh)) · blend/12` (1060-1082, `temporal_smoother.py:588-600`, `626-700`).

**Order matters** (comments 706-762): non-target/overlap trimming, pose-visibility trim, swap-net mask (`swap_model_mask_strength`, default 0) are applied **after** the feather so the feather cannot spread the swap back over trimmed hair, and so `blur_area`'s extent-based radius is not shrunk.

**Composite:**

```
out_ROI = α · warp(upsk_face, IM, INTER_CUBIC, BORDER_REPLICATE) + (1−α) · target_ROI       (float32 over the tight bbox of α>0.001, ±2/3 px)   [lines 779-911]
```
`blend_ratio` (default 1) blends the un-enhanced and enhanced faces only when <0.999 (802-834). Optional photometric steps inside the ROI (rim-only low-band illumination match `boundary_illumination_strength`, default 0; sensor-grain match) and multiband compositing when the temporal compositor is enabled (837-897).

**Crop-space composite inside `process_mask`:** `fake = fake·(1−m) + plate_crop·m` with `m` = 1 where the plate must be kept (`_composite_mask`, 1744-1782).

**Mask safety nets** [C]:
- Floor ellipse (pose-specific by construction): `axis_x = 1.3·eye_dist`, `axis_y = 2.0·|mouth_mid − eye_mid|`, centre at their midpoint, blurred with `k = max(3, (min(h,w)//16)|1)`. If the engine's swap area `<0.5·floor_area`: `mask = min(mask, 1 − floor)` (`procmgr_masking.py:188-256`, `ROOP_MASK_RECOVER_TRIGGER=0.5`, line 185).
- XSeg guided filter (He et al.), radius 4, ε=1e-3 on the aligned crop (opt-in, 280-303).
- Occluder-edge blur sized so the ramp is ~`ROOP_OCCLUDER_EDGE_PX=5` frame px regardless of face size: `radius = clip(round(5/frame_per_mask/2 − 0.5), 0, 8)` (306-359).
- The aligned crop (similarity) is *not* distorted (measured by Ultimate: anisotropy 1.000000000, shear 2.8e-14°, comment 1395-1402), so mask models run on the aligned crop; the "unwarped box" non-frontal path is off by default (`ROOP_NONFRONTAL_MASK=0`, `procmgr_masking.py:1422-1441`).

### 2.5 Temporal filters — equations and constants [C]
**(a) Pre-pass landmark smoother — default kind `adaptive`** (`roop/prepass_policy.py:52, 59-64`; selected at `procmgr_tracking.py:3341-3361` when `stabilize_landmarks` (config true) and `stabilize_method != kalman`). `AdaptiveLandmarkSmoother` (`temporal_smoother.py:149-386`):

```
extent   = max(‖kps0 − kps1‖, ptp_x(kps), ptp_y(kps), 1)          # not IOD alone: IOD → 0 on a profile (307-318)
v        = mean_i ‖kps_i(t) − kps^s_i(t−1)‖ / extent
β(v)     = β_slow + (β_fast − β_slow)·smoothstep((v − lo)/(hi − lo)),  lo = 0.02−0.01, hi = 0.02+0.01,  β_slow = 0.35, β_fast = 0.90
kps^s(t) = β·kps(t) + (1−β)·kps^s(t−1)
lm106^s(t)= β·lm106(t) + (1−β)·lm106^s(t−1)          # SAME β for kps and dense landmarks (the crop and its own outline cannot drift apart)
```
Guards: only if the previous state is exactly `frame−1` for this track; reset on shot cut; drop track after 8 missing frames. Reported by Ultimate (changelog 586): stationary jitter std 1.0 px → <0.6×, lag <3 px at 15 px/frame [Ultimate's claim].
Alternatives: `zero_phase` (same EMA forward+backward averaged, `temporal_smoother.py:446-585`), `one_euro` vectorised zero-phase (`roop/one_euro_vec.py`):

```
speed_t = mean_i ‖k_i(t+1) − k_i(t−1)‖/2 · fps / extent        (central difference of RAW kps)
dspeed  = zero-phase EMA of speed at d_cutoff = 3 Hz
fc_t    = min_cutoff + beta · max(dspeed, 0)                   min_cutoff = 0.6 Hz, beta = 25, fps default 24   (one_euro_vec.py:67-70)
α_t     = 1 / (1 + fps/(2π·fc_t)) ;   y = EMA_α forward, then EMA_α backward  (Hillis–Steele parallel scan)
```
One shared speed/α for kps, lm106 and lm68. Ultimate's own end-to-end result: "the effect is small" (changelog 384) and `adaptive` remains the default.
**(b) Sequential path** (`ProcessMgr._apply_stab`): `KpsStabilizer` — `OneEuroFilter` with `α = r/(r+1), r = 2π·fc·t_e` (`one_euro.py:32-35, 273-303`), `fc = min_cutoff + beta·|dx̂|` per coordinate, `min_cutoff 0.1, beta 0.1` in `default_config.yaml:114,129` (0.05 / 0.05 in `config.yaml:182,192`).
**(c) Enhancer crop** (`one_euro.py:534-616`): `base = 0.4(1−strength)+0.02` (strength 0.6 ⇒ 0.18); `motion = (centroid shift / t_e)/size`; `fc = base + 8·motion`; `α = max(0.15, r/(r+1))`; forced `α ≥ 0.5` every 30 frames (prevents progressive blur); `out = α·crop + (1−α)·prev`; per-track, keyed by `track_id`, 8-frame max gap.
**(d) HF flow stabiliser** (`temporal_smoother.py:782-870`): `hf = crop − blur(crop, σ=1.0)`; `hf_out = (1−w)·hf_t + w·warp(hf_{t−1}, flow)` with `w=0.15` (default_config `stabilize_hf_texture_weight: 0.15`), flow on a 64² grid, forward/backward-cycle check 1.5 px, photometric check 0.10, reset if mean abs residual >18, then a 3-frame range-weighted temporal bilateral (σ_range=10). Only the HF band is blended.
**(e) Mask** (`MaskStabilizer(EnhancerStabilizer)` with `motion_beta=0`, `ProcessMgr.py:1197-1251`): constant cutoff from `stabilize_mask_strength` (0.6). Its `apply` (`one_euro.py:711`) was not read, so the per-frame α is [I].
**(f) Target appearance EMA** (`appearance_conditioning.py:161-215`): `α = 0.30 + 0.35·motion`, `× (0.65 + 0.35·conf)`, raised to ≥0.75 when the exposure tier changes (so a cut does not smear a warm/cool ghost).
**(g) Coasting / gap fill:** Kalman coasting of detector dropouts, linear interpolation across ≤10 frames, shot-cut resets (`procmgr_tracking.py` ~2361, 2886-2911; `ProcessMgr.py:3914-3937`). Negative result recorded by Ultimate: refusing to swap synthetic/coasted faces "turned the anti-flicker machinery into a flicker source" (`ProcessMgr.py:5608-5615`).

### 2.6 Colour matching [C]
`apply_color_transfer(swap_crop, aligned_target_crop)` (`procmgr_color.py:331-396`), mode `lct` by default (LAB covariance whitening/re-colouring; fixes casts a per-channel scale cannot, 335-339 — body not re-read). The legacy `rct` form (398-418):

```
lab_s, lab_t = BGR→LAB (float)
std_scale = σ_t / max(σ_s, 1)  ;  std_scale_a, std_scale_b clipped to [0.80, 1.20]      (L unconstrained)
out = (lab_s − μ_s)·std_scale + μ_t → clip → BGR
```
B&W guard: skipped when mean|B−G|<5 and mean|G−R|<5. Second pass after the restorer (`color_match_after_enhance`, strength ×0.5, `ProcessMgr.py:6877-6890`). Target-conditioned appearance (measured on the aligned target crop only, 128² down-sample; `appearance_conditioning.py:87-139`) classifies `NORMAL/DARK/VERY_DARK` (p50 ≤0.16 or mean ≤0.19&p90≤0.48 ⇒ VERY_DARK; p50 ≤0.33 … ⇒ DARK) and blends the restorer output back toward the swap by `{1, 0.58, 0.20}` (`protect_restorer_output`, 218-227). The fused GPU kernel (`core/fused_postprocess.py`) implements the same Reinhard (chroma ratio [0.8,1.2]), a frequency-split blend `face = clamp(lw·blur(CT(swap)) + (1−lw)·blur(restored) + w·(restored − blur(restored))) + grain`, and `α = blur(xseg)·(1−keep)`, but is not wired into the render.

### 2.7 Pose-gated fades [C]
Mouth/eye plate restoration (`procmgr_masking.py:1997-2002`, `2095-2099`):

```
θ = max(|yaw|, |pitch|)  (jaw-free angles from solve_pose_jaw_5pt)
if θ > 25°:  mask *= clip((38° − θ)/13°, 0, 1)         → 100 % at 25°, 0 % at ≥38°
```
Reason (comment): "past ~25° the far eye is foreshortened … pasting the plate there doubles the socket".

Enhancer eligibility by pose (adaptive enhancer only, `roop/adaptive_enhancer.py:87-103, 143-145, 172-229`): `pose = clip(1 − max(|yaw|,|pitch|)/65, 0, 1)`; quality `Q = 0.24 res + 0.24 sharp + 0.14 pose + 0.14 illum + 0.12 conf + 0.12 temporal`; `pose < 0.18` ⇒ **no enhancer** ("extreme-angle-geometry-first"); path hysteresis ±0.06 around the profile cut (0.63/0.68/0.72/0.76). (`adaptive_detector.py` only flags `|yaw| ≥ 45°` as profile for telemetry/recall, lines 51, 636-642.)

### 2.8 3D source-crop correction (not the default path)
Only for image-source swappers (`BlendSwap/UniFace`), `ProcessMgr.py:6037-6118`. Plan (`face_3d_recon.py:172-220`): flip only if `|src_yaw|≥20 ∧ |tgt_yaw|≥20 ∧ opposite signs ∧ |Δyaw|≥45`; `yaw_shear = sign(Δ)·clip((|Δ|−15)/45, 0, 1)·0.20`; `pitch_shear = sign(Δp)·clip((|Δp|−15)/45,0,1)·0.07` (= `min(0.2·0.35, 0.08)`); safe iff `|yaw_shear|≤0.20 ∧ |pitch_shear|≤0.08`; shear matrices rejected if singular values leave [0.74, 1.35] (line 160). Embedding swappers are deliberately *not* fed sheared/flipped crops ("only DEGRADES identity", `ProcessMgr.py:6029-6036`).

---

## 3. Why the result is stable (no shake), sharp, and has no black edges / halos

### 3.1 Stable
1. **Geometry comes from smoothed landmarks only** and the *same* smoothed landmarks drive crop (`M`), hull (matte) and mask crop — one β for both sets (§2.5a). The crop is keyed to the landmarks, not to a regressed 3D fit, so the swap net sees a template-registered face at every frame [C].
2. **Zero-lag option** (zero-phase/vec 1-euro) because the pre-pass owns whole tracks [C]; default is the causal adaptive EMA [C].
3. **Contiguity guards everywhere** (`state.frame_index == frame−1`), because swap workers are round-robin; filters that cannot apply "DECLINE and say so" (`summary_line`) rather than blend frames N apart (`temporal_smoother.py:53-68`) [C].
4. **Decisions that can flip are made in one ordered pre-pass thread** (source index, roll latch, pose annotation), then replayed: "keeping these decisions here avoids a frame-order race at a source-pose boundary" (`procmgr_tracking.py:3452-3457`) [C].
5. **Hysteresis on every discrete choice**: source index (0.035 margin / 15° / 5-frame hold), enhancer path (±0.06), temporal-identity switch (3 frames) + 4-frame low-band transition, mask router latch (`NonFrontalRouter`) [C].
6. **No hard thresholds in continuous gates**: β uses a smoothstep ramp ("a latch over a wrong metric still flickers", `temporal_smoother.py:170-180`), mouth/eye fades are linear over 13°, appearance EMA is ramped [C].
7. **Crop-space temporal filters operate in a registered frame** (the canonical crop), so they can blend without ghosting; they are skipped when autorotate rebinds the crop (`ProcessMgr.py:6776-6804`) [C].

### 3.2 Sharp
1. **One forward and one inverse resampling** of the face (cubic, both with replicate border) — no intermediate down-scale; enhancer re-alignment (two extra warps) is opt-in "because two extra resampling passes" (`ProcessMgr.py:6614-6620`) [C].
2. Swap at 256 px native (hyperswap) → restorer 512 → paste with `M·scale_factor` so the paste resolution follows the restorer's output; Restore-Ultra keeps the swap's low band and adds `w×` the restorer's high band, σ = 2.0·(W/512) (`roop/enhance_blend.py:47, 62-102`) [C].
3. Skip the restorer when the face is too small for a 512 restoration to survive the paste-back (`_enhance_gate`, `ProcessMgr.py:6584-6596`) [C]; skip it entirely at extreme angles in the adaptive enhancer (§2.7) [C].
4. Optional target-detail injection (`detail_transfer_strength`), identity-detail map, HF stabiliser (6855-6864, 7074-7122, 6797-6804) [C]. Whether each is on in the live config was not individually checked beyond `stabilize_hf_texture` [C for that flag].
5. Plate-sharpness gate in the merger (`procmgr_merger.py:117-160`) caps added sharpening to the plate's own HF energy [C for existence, I for tuning].

### 3.3 No black edges / halos
- Black wedges: `BORDER_REPLICATE` on crop warp and paste warp (§2.4). Occluded/out-of-frame ring for the enhancer is filled from the *plate*, not replicated pixels (`ProcessMgr.py:6639-6658`) [C].
- Gray/hairy rim: matte = circle ∩ hull-of-landmarks, forehead extended along the head axis (so a tilted head's hull does not swallow background on one side), hull dilated by ~`√(face_h·face_w)·blend/400`.
- Halo: erosion `blend_px//4` then Gaussian feather sized from the matte's own extent; trims applied after the feather; floor ellipse prevents a too-small mask from leaving a ring of target skin; optional rim-only illumination match removes the brightness step "that survives a perfect alpha ramp" (`procmgr_masking.py:837-858`, default strength 0) [C].
- The mask lives in the aligned crop (face fills it) rather than on a boxed tight crop where the segmenter sees an 8 %-of-crop face (measured in comments 1195-1199) [C].
- Occluder edge width in *frame* pixels, not crop pixels (otherwise a close-up produced a 42 px glow, comment 306-333) [C].

### 3.4 Known weak spots in Ultimate (so we do not port them)
- Portfolio profile switch has no hysteresis (§2.2) [C/I].
- Two default config files disagree (`use_3d_recon`, `use_source_bank`) [C].
- Dead alignment/stabiliser module `geometric_alignment.py` still documented as "Stage 4" [C].
- Own end-to-end results say zero-phase smoothing's effect is small and mask-fusion reversed end-to-end (changelog 384, 467-469) — diminishing returns after the coupled adaptive EMA [Ultimate's claim].

---

## 4. Mapping: Ultimate technique → Roop 3D counterpart / gap

Roop 3D `reenactment_3d` call chain [C]: `ReferencePipeline.infer_frame` (`engine/volumetric/session.py:583`) → tracker (`engine/detection/face_tracker.py` + `TrackTemporalStabilizer`) → `_apply_angle_trajectories` (offline scan, `angle_tracker.py`) → `NeuralFaceEngine.process` (`reconstruction/neural_face.py:228`) when `face_mode != "standard"` and the dense engine is available (`session.py:662-678`) → `IdentitySwapper.synthesize_correspondence` → `UltraMaxPipelineRestorer.restore_hires` → `Replacement(image, inverse, coverage)` → compositor (`compositor/pipeline.py`, temporal `compositor/temporal.py`). The `_reenact`/`ReenactmentEngine` branch (`session.py:726-730`) is only reached when the dense path falls back to the standard swap.

| # | Ultimate technique (cite) | Roop 3D counterpart (cite) | Status |
|---|---|---|---|
| 1 | Template similarity alignment of the 5 kps at every angle (`face_util.py:3477-3512`) | 2D swap path: `aligned_crop`/`roll_compensated_aligned_crop` with `similarity()` LS (`engine/volumetric/preprocess.py:24-82`, used at `identity_swapper.py:270-272`). **Dense/neural path does not use it**: crop = bbox of projected visible mesh, `_mesh_crop_affine` (`neural_face.py:206-221, 257`) | **GAP in the 3D path** |
| 2 | Crop border = `BORDER_REPLICATE`, cubic (`face_util.py:3540-3542`) | `aligned_crop`: `INTER_LINEAR`, **`BORDER_CONSTANT`** (`preprocess.py:53`); backbone fit warps `BORDER_CONSTANT` (`morphable_backbone.py:232-235, 258-259`); compositor `_crop_to_frame` uses `REFLECT_101` (`compositor/pipeline.py:301-304`) | **GAP** (black wedge for edge faces / strong roll) |
| 3 | Refine kps from 68-pt landmarks (`face_util.py:676-703`) | `face.landmarks68` present; `_bank_landmarks` fallback (`session.py:324-334`). No "replace kps by 68-derived centres" step found | Partial / unverified |
| 4 | Coupled smoothing of kps+dense under ONE velocity-normalised β (`temporal_smoother.py:149-386`) | Per-channel independent One-Euro+Kalman on `bbox, landmarks5/106/203/68, shape/expression coefs, pose` with raw-pixel/second derivatives (`face_tracker.py:253-286`, `temporal.py:190-217`; defaults `min_cutoff .15, beta .04, d_cutoff 1`, `core/settings.py:38-40`); offline bidirectional One-Euro (`angle_tracker.py:545-583`) | Partial: smoothing exists, **not coupled, not size-normalised** |
| 5 | Zero-phase whole-track filter (`one_euro_vec.py`) | `AngleTrajectoryTracer` forward+backward average (`angle_tracker.py:545-554`) | **Present** |
| 6 | Smooth the *fit* used by the swap | `NeuralFaceEngine.process` calls `backbone.fit(image, landmarks5)` every frame and uses its rotation/translation/scale/expression raw (`neural_face.py:232, 246-258`; `morphable_backbone.py:231-271`); no filter on those (grep of `reconstruction/` found filters only in `reenactment.py`) | **GAP** |
| 7 | Source-view selection with pose cost + hysteresis (`pose_source_selector.py:306-396`) | Identity vector: frozen bilinear/spherical pose matrix, 3×3 cells or 96 sphere bins, smooth (`pose_matrix.py:24-28, 325-345`; `identity_swapper.py:183-223`). **Source photographs** for the dense warp: re-sorted every frame by `‖R_src − R_tgt‖_F` then `−quality`, top-2 (6 GB) / top-4 (12 GB), weights `conf·quality·exp(−2‖ΔR‖_F)` (`neural_face.py:237-238, 268-274`) with **no hysteresis**. `session.py:351` sets `swapper.pose_deadband`/`_hysteresis` but nothing in `IdentitySwapper` reads them (grep) | **GAP for photo views**; identity-vector side is already ported |
| 8 | Blending across views in vector space (SLERP top-3; 0.7/0.3 profile mix; 4-frame low-band transition) | Pose matrix does spherical bilinear mean (`pose_matrix.py:432-440`) = same idea. Photo views blended in *pixel* space by confidence-weighted average (`neural_face.py:274, 290`) — view switch is a pixel pop | Partial |
| 9 | Pose-gated plate eye/mouth fade 25→38° (`procmgr_masking.py:2000, 2096`) | No yaw/pitch gating found in `compositor/mouth.py, pipeline.py, face_regions.py` (grep hits are only `manual_light_*` and comments). `profile_aware_mask` cuts the far cheek for |yaw|>35° (`session.py:113-137, 731-734`) | Partial / different mechanism |
| 10 | Far-eye damping of restorer output (`source_portfolio.py:339-364`) | None found | **GAP** |
| 11 | Mask = ellipse ∩ landmark hull + ¼-radius erosion + extent-sized Gaussian feather; floor ellipse (`procmgr_masking.py:629-681, 1094-1125, 188-256`) | Coverage = rasterised target front-face surface, feathered by `clip(distance/max(0.01·W,1),0,1)` (≈5 px @ 512) (`neural_face.py:316-318`); compositor adds XSeg/parser, hairline, strand matte, jaw clamp, boundary fusion (`compositor/pipeline.py`, `masks.py`, `matte.py`, `boundary_fusion.py`); `profile_aware_mask` (`session.py:113-137`) | Different, heavier; **edge width is the gap** |
| 12 | Crop-space temporal filter on enhancer HF band, flow-warped (`temporal_smoother.py:782-1220`) | `TemporalDetailStabilizer` (`compositor/detail_temporal.py`, `pipeline.py:267, 374, 647`) | **Present** |
| 13 | Temporal filter on paste affine/mask/colour | `compositor/temporal.py:121-171` filters `inverse` (all 6 raw entries, One-Euro `min_cutoff 1, beta .035`), alpha, colour stats | Present, but see §5 hazard H3 |
| 14 | Colour transfer LAB/LCT + post-enhance re-match + dark-target protection (`procmgr_color.py`, `appearance_conditioning.py`) | `compositor/color_match.py` (LAB/YCbCr/hist/OKLab sliced Wasserstein), `lighting.py`, `photometric_identity.py`, `quality.py:190` linear-light blend | **Present (richer)** |
| 15 | Hysteresis/transition on discrete choices | `identity/lock.py`, `identity/target_lock.py`, track policy (`face_tracker.py` `_assign_policy`), 9→96 bin identity matrix | Present for identity; **absent for photo-view choice** |
| 16 | Dense target pose → swap net conditioning crop | Swapper input is the *source-photo warp rendered in the target camera* (`neural_face.py:274-303`) fed to hyperswap-class net, then UltraMax; keeps source geometry + target expression (`_source_geometry_in_target_pose`, 181-203) | Roop 3D's own (stronger) 3D geometry — keep |
| 17 | Neutral-pitch offset per person (`angle_portfolio.py:139-159`) | None found for photo-view ranking | Gap (small) |
| 18 | Diagnostics: swap audit, per-face `[ANGLE]` log, route stats (`ProcessMgr.py:3876-3878`, `procmgr_masking.py:1451-1491`) | `last_source_view_indices`, `last_fields` on `NeuralFaceEngine` (`neural_face.py:98-100, 284-286`); `[Stage13]` log lines (`identity_swapper.py:200-211`) | Partial |

---

## 5. Where Roop 3D's `reenactment_3d` path is likely to shake / halo / soften [I unless noted]
All four are read from code; none was reproduced by running a clip.

- **H1 — crop framing depends on the visible-mesh silhouette** [C for the code, I for the visual effect]. `_mesh_crop_affine` scales to `size·(1−2·0.12)/max(extent of projected visible vertices)` and centres the bbox at `(0.5·size, 0.52·size)` (`neural_face.py:206-221`). The extent/centre change with yaw (silhouette narrows, bbox centre shifts), with jaw opening (height grows → whole face zooms out), and with any frame-to-frame change of the CNN-regressed rotation/scale. Ultimate instead pins the 5 landmarks to fixed template pixels, which is also what the swap net was trained on. Expected symptom: face "breathes"/slides in the crop as the head turns or the mouth opens, plus out-of-distribution feature positions for the swap net.
- **H2 — raw per-frame CNN fit drives geometry, view choice and coverage** [C]. `process` fits the target every frame and uses it unfiltered for correspondence, visibility, view ranking and coverage (`neural_face.py:232-258, 266-274`); the tracker's filters act on landmarks/Euler pose, not on the fit (`face_tracker.py:253-286`).
- **H3 — mismatch between smoothed paste affine and unsmoothed crop content** [I]. `compositor/temporal.py:145-161` One-Euro-filters `replacement.inverse`, while the pixels in the crop were synthesised under the *unfiltered* camera of that frame; any residual high-frequency in the fit then appears as swimming relative to the plate. It also filters the six affine entries as independent raw numbers (rotation/scale/translation mixed units) — Ultimate warns that per-element filters with element-wise derivative cutoffs "settle at different rates" (`temporal_smoother.py:18-38`).
- **H4 — hard top-k view set + pixel blending of photos** [C code, I effect]: when two source views swap rank, `weight` of the dropped view goes from `w>0` to `0` in one frame (`neural_face.py:237-238, 268-274, 290`). Colour/texture pop at the turn; the 4th-ranked view appearing is the same.
- **H5 — black/edge artefacts**: `aligned_crop` and the backbone's two input warps use `BORDER_CONSTANT` (`preprocess.py:53`, `morphable_backbone.py:232-235, 258-259`) — black wedge for faces at the frame edge (Ultimate measured 34 % of the crop; its fix: replicate) [C for code]. The dense-path edge feather is 1 % of the width (`neural_face.py:317-318`) [C], vs Ultimate's extent-proportional `blend_px = size·blend/200` (≈ 12.5 % of the face size at blend 25) [C].
- **H6 — source photo cache sampled with `INTER_LINEAR` after a 128→512/1024 warp from original** (`neural_face.py:147-150`) [C]: acceptable on up-sampling, aliasing on down-sampling; plus `warpAffine` default border zero — covered by `observed` mask [C].

---

## 6. Port plan — ordered, minimal, keeps Roop 3D's 3D geometry

Principle: keep the mesh-derived correspondence (source-photo warp in the target camera) and swap/restore stack; adopt Ultimate's **registration, temporal coupling, view hysteresis, border/feather and diagnostics**. Each step is independently shippable and measurable with the diagnostics of step 0.

### Step 0 — Diagnostics first (so every later step has a number)
New module `roop_3d/engine/volumetric/angle_diagnostics.py` (pure functions + a small recorder), called from `NeuralFaceEngine.process` (add `self.last_diag: dict`, same pattern as `last_fields`, `neural_face.py:98-100, 284-286`) and from the compositor result stage (`compositor/pipeline.py` around line 843-851 where `effective_alpha`/`fuse_inverse` are known). Per track, per frame, append to a bounded ring + JSONL next to the job's existing telemetry:
1. **Jitter (px, IOD-normalised)**:
   - `lm5_jit = ‖L_t − 2L_{t−1} + L_{t−2}‖_rms / IOD` for tracker-output `landmarks5` (second difference removes constant-velocity motion; Ultimate's changelog uses stationary std for the same purpose).
   - Crop-affine jitter: decompose `crop_matrix` into `(s, θ, tx, ty)` and log `|Δ²s|/s`, `|Δ²θ|`, `|Δ²t|/ (s·IOD)`.
   - Content jitter: ORB-free, cheap: phase-correlation shift (cv2.phaseCorrelate) between `crop_t` and `crop_{t−1}` of the **final enhanced crop** after removing the predicted shift from the crop-matrix delta → residual px.
2. **Sharpness vs yaw**: Laplacian variance on a 160×160 gray resize of the face ROI (`_sharpness` formula, `adaptive_enhancer.py:65-74`: `var/350` clamped) of (a) final composited frame ROI, (b) the plate ROI at the same box; log ratio `a/b` bucketed by `|yaw|` in 10° bins (use the tracker yaw so bins are comparable with Ultimate's).
3. **Mask-edge luma** (black-edge/halo detector): from final `effective_alpha` build rings `R_in = {0.5<α<0.98}` and `R_out = {0.02<α<0.5}`; log `mean_luma(R_out of composite)`, `min_luma`, count of px with `luma<16` where `plate_luma>40`, and `|mean_luma(R_in) − mean_luma(plate at R_in)|`.
4. **Source-view selection log**: per frame `[(view_index, yaw, pitch, raw_score, smoothed_weight)]`, a `switch` flag (set of selected indices changed), `dwell` frames, and a running count of switches per 100 frames. Existing `last_source_view_indices` (`neural_face.py:286`) is the seed.
5. Add `[ANGLE]` console line gated by env `ROOP3D_DEBUG_ANGLE=1`, analogous to `ROOP_DEBUG_ANGLE` (`procmgr_masking.py:1451-1491`).

### Step 1 — Register the dense crop to the swapper template (replaces `_mesh_crop_affine`)
In `NeuralFaceEngine.process` (`neural_face.py:228-323`):
- Compute `crop_matrix = similarity(face.landmarks5, T·size)` where `T` = `FFHQ_REFERENCE_5PTS_512/512` or `ARCFACE_REFERENCE_5PTS` chosen exactly like `identity_swapper.py:272` (`model.spec["alignment_template"]`), using the **tracker-filtered** `face.landmarks5` (already coupled-filtered after step 2). Reuse `preprocess.similarity` (`preprocess.py:24-37`) — it is the same LS objective as Ultimate's `estimate_norm`.
- Derive `mesh_affine = crop_matrix ∘ source_to_fit⁻¹` (3×3 compose; `source_to_fit` is already built at line 258) so `k = mesh_affine @ target.intrinsics` (line 262) and all downstream rasterisation/correspondence stay unchanged. Keep `_mesh_crop_affine` only as a fallback when `landmarks5` are non-finite.
- `AlignedCrop(...)`/`inverse` (lines 260-261, 319-322) then come from `crop_matrix`. Keep `roll`: `similarity` removes roll exactly; no extra rotation needed.
Why: removes H1; the swap net and UltraMax (`restore_in_crop_space`, which already expects FFHQ-5pt landmarks, `restoration.py:162-173`) see registered features at every yaw.
Acceptance: `crop-affine jitter` (diag 1) drops; `sharpness ratio` at |yaw| 30–60° does not regress; no change in `valid` coverage area ±3 %.

### Step 2 — Coupled, size-normalised smoothing of the geometry that defines the crop and mask
2a. Add `CoupledTrackSmoother` in `roop_3d/temporal.py` (next to `TrackTemporalStabilizer`) implementing Ultimate's equations exactly (§2.5a): `extent = max(IOD, ptp_x, ptp_y, 1)`, `v = mean‖Δ‖/extent`, `β = 0.35 + 0.55·smoothstep((v−0.01)/0.02)`, one β applied to `landmarks5`, `landmarks106`, `landmarks68`, `landmarks203` and `box` of the same track; frame-contiguity + scene-cut reset (reuse `TrackTemporalStabilizer.forget_track` semantics, `face_tracker.py:257-266`). Wire in `FaceTracker._stabilize` (`face_tracker.py:253-286`) as the default for those five channels; keep `filter_pose` (SO(3)) for rotation and the Kalman/One-Euro path for coefficients. Settings `temporal_*` (`core/settings.py:38-41`) gain `landmark_beta_slow/fast/velocity` (defaults .35/.90/.02).
2b. In the offline angle scan, keep the zero-phase pass (`angle_tracker._smooth_values`) but divide speed by face extent and use one shared α for all landmark sets (as `one_euro_vec.smooth_runs`, `one_euro_vec.py:204-254`: `min_cutoff 0.6, beta 25, d_cutoff 3`), because per-coordinate raw-px cutoffs (`.15 + .04·|dx̂|`, px/s) are resolution-dependent [I].
2c. **Filter the dense fit** before it is used: in `process`, run `rotation` through `TrackTemporalStabilizer.filter_rotation_matrix` (`temporal.py:152`), and `translation`, `scale`, `expression` through `.filter(track_id, channel, value, timestamp)` (`temporal.py:190-217`), with timestamps from `session._frame_time()` (`session.py:266`); reset on `scene_cut`. For offline renders, preferred: a **pre-pass** that caches the raw fits per frame (rotation, translation, scale, expression) and zero-phase filters them (same form as 2b) before pass-2 — this mirrors Ultimate's "replay from ordered pre-pass" design (`procmgr_tracking.py:3452-3457`). Apply the filtered values in `_source_geometry_in_target_pose(source_identity, target_fit)` (`neural_face.py:181-203`) by passing a lightweight `FilteredFit` shim (same attributes).
Acceptance: `lm5_jit`, `Δ²t`, content-jitter reduce; "lag" test: constant-velocity pan of 15 px/frame keeps residual < 3 px (Ultimate's own acceptance; changelog 586).

### Step 3 — View selection with cost, hysteresis and temporal weights (photo views)
In `NeuralFaceEngine.process`, replace lines 237-238 and 268-274 with a per-track `ViewSelector` (new class in `neural_face.py`, state keyed by `track_id`, reset on scene cut):
- Cost per source view (Ultimate's pose cost, §2.3) using Euler angles from `view.yaw/pitch/roll` (already stored on views; used in `prepare`, line 125) vs the **filtered** target pose: `pose_cost = 0.52·min(2,|Δyaw|/55) + 0.28·min(2,|Δpitch|/42) + 0.20·min(2, wrap|Δroll|/55)`, `total = pose_cost·(0.50+0.18·conf) + 0.18·(1−quality)`. Keep `‖R_src−R_tgt‖_F` as a tiebreak if Euler is unreliable near gimbal (|pitch|>75°).
- Top-k as today (2 / 4), but a view enters/leaves only if `cost_new + 0.035 < cost_old` for ≥3 consecutive frames (`switch_frames=3`, `transition_frames=4`, `temporal_identity.py:150-156`); neutral-pitch correction per FaceSet (median pitch of views with |yaw|≤25°, `angle_portfolio.py:139-159`) applied to view pitches.
- Temporal weights: keep `weight_i = field.confidence·quality·exp(−2‖ΔR‖_F)` (line 272-273) but multiply by a per-view ramp `ramp_i ∈ [0,1]` that rises by `1/4` per frame on entry and falls by `1/4` on exit (`transition_frames=4`), so the pixel blend `color/support` (line 290) crossfades instead of popping; EMA `α = 0.35` (`geometry_alpha`) on `ramp·weight` before accumulation. The mirrored fallback (lines 294-300) stays.
- Remove the dead `pose_deadband/_hysteresis` assignments (`session.py:351-352, 628-630`) or implement them here with `settings.pose_hysteresis_degrees` (2.5°) as the Euler dead-band on the *identity-vector* side; do not leave both.
Acceptance: switches per 100 frames on a continuous 0→60° yaw sweep ≤ 3 and no frame-to-frame change of any view weight >0.25 (diag 4).

### Step 4 — Borders and edge feather (halo / black edge)
- `preprocess.aligned_crop` (`preprocess.py:53`): `INTER_CUBIC`, `borderMode=BORDER_REPLICATE`. Same for `roll_compensated_aligned_crop` inner call (it already uses `REFLECT_101` for the pre-rotation, line 77 — fine). `MorphableBackbone.fit` warps (`morphable_backbone.py:232-235, 258-259`): `BORDER_REPLICATE` (do **not** add `borderValue`; these crops feed a CNN that, like Ultimate's swapper, was trained on full crops). Source photograph cache: `INTER_CUBIC` when up-scaling, `INTER_AREA` when down-scaling (`neural_face.py:147-150`).
- Replace the dense edge ramp (`neural_face.py:316-318`) with Ultimate's recipe computed in crop pixels of the swapper crop: `mask_size = √(h·w)` of `coverage>0.5`; `blend_px = max(1, int(mask_size·25/200))` (make `25` the existing `face_mask_blend`-equivalent compositor option); `coverage = GaussianBlur(erode(coverage, ellipse(2·max(1,blend_px//4)+1)), k=2·blend_px+1)`; keep `min(·, hull_of_landmarks106_∪_forehead)` (`landmark_hull`, `procmgr_masking.py:430-510`) as an extra `∧` so a mesh rasterisation overshoot never paints background. Option B (nearer Roop 3D's existing distance-transform code): `alpha = sigmoid-ramp(dist/margin)` with `margin = lerp(8, 18 px, clip((extent−lo)/(hi−lo)))` (`adaptive_margin_px`, `temporal_smoother.py:588-600`, `soft_distance_matte`, 626-700). Constants `MARGIN_MIN_PX/MAX/SCALE_LO/HI` are defined at the top of `temporal_smoother.py` (not re-read here; copy from there).
- Apply trims **after** the feather (Ultimate's ordering rule, `procmgr_masking.py:706-762`): `profile_aware_mask` (`session.py:731-734`) currently multiplies coverage before the compositor's own feathering — confirm in `compositor/pipeline.py:760-773, 843-844` that the ramp is not re-derived from a shrunken extent.
Acceptance: `mask-edge luma` diag: zero pixels with `luma<16` in `R_out` where plate luma >40; `|mean_luma(R_in) − plate|` within 3 levels at |yaw|≤45°.

### Step 5 — Pose gates and restorer behaviour at angle
- Port `damp_far_eye` (`source_portfolio.py:339-364`, 25 % toward the pre-enhance crop) into `NeuralFaceEngine.process` right after `self.enhancer.restore_hires(...)` (line 309): `pre = neural`, `kps_in_enh = crop_points·(enhanced_w/size)` (already computed at 308/322), `eye = far_eye_index(face.landmarks5)` when `|face.pose[0]| > 35`.
- Port the 25→38° fade `clip((38−θ)/13, 0, 1)` for any plate-sourced eye/mouth overlay in `compositor/mouth.py` (`composite_mouth_region`, line 40), with `θ = max(|yaw|,|pitch|)` from the **jaw-free** solve (port `solve_pose_jaw_5pt`, `face_util.py:3203`, or reuse the tracker yaw if it is already jaw-independent — not verified). Skip restoration at `pose < 0.18` (`adaptive_enhancer.py:194-195`) only if diag 2 shows UltraMax over-sharpening artefacts above ~75° yaw.
- Optional (only if H3 is confirmed by diag): make `compositor/temporal.py` filter a *decomposed* similarity `(s, θ, tx, ty)` (as Ultimate's dead `filter_matrix` does, `geometric_alignment.py:259-312`: angle unwrapping, per-component accel kick) instead of the 6 raw entries (`temporal.py:145-161`), or drop affine filtering entirely once step 2 filters the fit upstream.

### Step 6 — Hysteresis on any pose-threshold branch in the port
Any new `|yaw| > 35°` style switch must use a latch (enter at 37°, exit at 33°) or a smoothstep ramp; do not copy `route`'s bare threshold (`source_portfolio.py:326-327`). Existing Roop 3D branches that need the same: `auto_3d_limits()` extreme test (`session.py:653-655`), `profile_aware_mask` (`session.py:731`, |yaw|>35 with a hard `abs(yaw) <= 35.0` early return, line 117), `dense_3d_max_roll` (657).

### Acceptance metrics (proposed targets; none measured — set baselines first with diag on the current build)
Run each on ≥3 clips including a 0→75° yaw sweep, a pitch ±30° nod, a talking frontal head, and an edge-of-frame face.

| Metric | How (diag) | Target |
|---|---|---|
| Landmark jitter | `lm5_jit` on still segments | ≤ 0.6× baseline (Ultimate's claim for its smoother) |
| Tracking lag | 15 px/frame pan, filtered vs raw | < 3 px (Ultimate's claim) |
| Crop-affine 2nd-difference | `|Δ²s|/s`, `|Δ²t|/(s·IOD)` | ≤ 0.5× baseline; no scale pulse correlated with mouth-open (corr(`s`, MAR) < 0.2) |
| Content jitter | residual phase-correlation px of final crop | ≤ 0.5× baseline on still frames |
| View switches | per 100 frames, sweep | ≤ 3; max per-frame weight Δ ≤ 0.25 |
| Sharpness vs yaw | `Lap-var(final)/Lap-var(plate)` per 10° bin | ≥ 0.8 for |yaw|≤45°, ≥ 0.6 for 45–75°; adjacent-bin drop ≤ 0.2 (no cliff) |
| Black edge | px with `luma<16` in `R_out` while plate luma >40 | 0 on all clips |
| Edge luma step | `|mean_luma(R_in) − plate|` | ≤ 3 levels at |yaw|≤45° |
| Identity | cosine(embed(final face ROI), source fused vector) at yaw bins | not worse than baseline by > 0.02 |
| Latency | ms/frame of `NeuralFaceEngine.process` | ≤ baseline + 5 % (steps 1–4 add one 2×3 solve, one 3-tap filter, one Gaussian on a ≤512² mask) |

---

## 7. Evidence log (what I read, in order of importance)
Ultimate: `roop/ProcessMgr.py` (3438-3880, 5583-7580), `roop/face_util.py` (676-703, 2711-3017, 3366-3542), `roop/face_analyser.py` (333-735), `roop/geometric_alignment.py` (all), `roop/angle_portfolio.py` (all), `roop/source_portfolio.py` (all), `roop/pose_source_selector.py` (all), `roop/FaceSet.py` (123-393), `roop/faceset_v2.py` (276-300, 1264-1414), `roop/stage0_preflight.py` (1-140), `roop/processors/FaceSwapInsightFace.py` (22-82, 145-160), `roop/procmgr_masking.py` (59-72, 188-513, 629-1182, 1183-1743, 1990-2112), `roop/procmgr_color.py` (331-418), `roop/temporal_smoother.py` (1-91, 149-386, 437-600, 782-870), `roop/one_euro.py` (28-36, 258-351, 534-616), `roop/one_euro_vec.py`, `roop/prepass_policy.py`, `roop/procmgr_tracking.py` (3300-3670), `roop/appearance_conditioning.py`, `roop/association_cost.py`, `roop/pose_quality.py` (1-300), `roop/adaptive_enhancer.py` (65-232), `roop/adaptive_detector.py` (header/config), `roop/enhance_blend.py` (62-102), `roop/face_3d_recon.py` (160-270), `roop/temporal_identity.py` (140-175, 567-640), `app/core/fused_postprocess.py` (header), `app/post_swap.py` (header/index), `app/routes_angle_scan.py` (575-615), `app/default_config.yaml`, `app/config.yaml`, `docs/POSE_AWARE_SOURCE_SELECTION.md`, `docs/CHANGELOG.md` (grep).
Not read in depth (so claims about them are [I] or omitted): `procmgr_color._color_transfer_lct`, `MaskStabilizer.apply`, `temporal_compositing.py`, `face_frontalize.py`, `nonfrontal.py` router internals, `mask_fusion.py`, `docs/PHASE6_*.md`, `docs/development/*`, the `identity_directions.npz` / `identity_render_validation.json` files (only that `roop/identity_algebra.py` consumes them for Identity-Blender dials and that `identity_render_validation.json` marks `age/gender/expression` dials unwritable per changelog line 340; unrelated to head-angle handling).
Roop 3D: `reconstruction/neural_face.py` (all), `engine/volumetric/{session.py (96-140, 341-353, 583-745), preprocess.py, identity_swapper.py}`, `engine/detection/face_tracker.py` (253-286), `temporal.py` (1-220), `compositor/temporal.py` (1-171), `compositor/pipeline.py` (grep), `angle_tracker.py` (1-60, 240-262, 491-583), `pose_matrix.py` (grep), `reconstruction/{morphable_backbone.py fit, dense_correspondence.py 176-260}`, `compositor/restoration.py` (restore_hires), `core/settings.py` (38-41).
