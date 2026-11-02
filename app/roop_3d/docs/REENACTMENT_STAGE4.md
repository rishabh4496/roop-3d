# Stage 4: Temporal 3D Face Reenactment Engine

**Document Path:** `app/roop_3d/docs/REENACTMENT_STAGE4.md`
**Scope:** drive the Stage 3 source identity asset with the target video's tracked motion, behind an explicit render mode,
with a deterministic fallback to the existing proven path.

> **Read this first.** This is a *parametric, hand-built* reenactment model: the source identity mesh (shape) and atlas
> (texture) are posed by the target's measured head pose and deformed by the target's measured expression deltas. It is
> not FLAME and not a learned model, and it renders the **facial region only**. Where its geometry cannot be trusted it
> steps aside; it never presents a guess as a measurement.

## 1. Architecture

```
SOURCE  FaceSet ──Stage 3──► IdentityAsset ──► IdentityRig   (rest mesh, atlas, per-texel confidence, anatomical weight fields,
                                                              observed yaw/pitch range)
TARGET  frame ──tracker (SCRFD + 2d106) ──► landmarks5, landmarks106, timestamp, scene_cut, focus
                                                        │
                       ReenactmentEngine.process (per track, deterministic)
   1 validate  2 camera vs SOURCE anchors  3 expression deltas  4 One-Euro smoothing  5 deform rig  6 render
   7 anchor correspondence + snap  8 measured confidence (min of components) + hysteresis + cross-fade
                                                        │
              engaged ──► image + coverage  (cross-faded with the proven crop while ramping)
              not engaged ─► the existing identity-swap crop, untouched, with `fallback_reason`
```

### Modes (`render_mode`, default `standard`)

| Mode | What runs | Fallback |
| :--- | :--- | :--- |
| `standard` | the existing identity swap, byte-for-byte unchanged | n/a |
| `pose_3d` | the swap crop projected by `PoseBlendProjector` built from the **source identity's own geometry** (`from_identity_asset`; the generic canonical proxy when no asset) | projection rejected by the landmarks ⇒ the swap crop, with `POSE_PROJECTION_REJECTED:<code>` |
| `reenactment_3d` | the engine: source geometry + atlas, target pose and expression | not engaged ⇒ the swap crop, with the reason |

`render_mode` is a `WorkstationParameters` field (omitted from serialization unless set, so stored requests stay valid),
reaches `ReferencePipeline.prepare_session(render_mode=…)` and `infer_frame(render_mode=…)`, and every `Replacement`
records `render_mode`, `fallback_reason` and the engine frame. `capability()["render_modes"]` reports the identity-asset
status per FaceSet. A FaceSet that cannot support an asset keeps the session usable and the proven path.

## 2. The twelve requirements

| # | Requirement | Implementation | Honest scope |
| :-- | :--- | :--- | :--- |
| 1 | Target geometry estimation | 5 landmarks + the block structure of the 106 landmarks; pose solved against the source's 3D anchors | no dense **target** mesh is reconstructed |
| 2 | Per-frame camera pose | `multiview_fit.solve_view_pose`: the Stage 3 MAP residual model with the geometry held at the source anchors (numeric Levenberg-Marquardt, 6 parameters, Huber-reweighted, pose prior) | weak-perspective, ~a few degrees |
| 3 | Yaw/pitch/roll transfer | smoothed rotation, scale, translation → `WeakPerspectiveCamera` | |
| 4 | Expression estimation | 7 inter-ocular-normalised deltas from the track's own neutral (`measure_expression`, `NeutralCalibrator`) | image-space measurements, not FLAME coefficients |
| 5 | Jaw | rotation about an identity-specific hinge; the angle is the **exact** inverse of the lip-drop relation | |
| 6 | Mouth open/close | jaw + lip; the mouth interior is **released to the target** above 0.10 IOD (the atlas has none) | |
| 7 | Eye opening / blink | eyelid vertex displacement; the eye is released to the target from 45 % closure (the atlas has no eyelid skin); per eye (winks work) | |
| 8 | Eyebrows | kernel displacement around the brow vertices | |
| 9 | Cheek | kernel displacement **driven by smile / width** | a heuristic, not independently measured |
| 10 | Smile / other | corner raise + widening (smile, pucker) | |
| 11 | Temporal stabilisation | One-Euro filters (pose, expression, anchors; separate bandwidths), physical-rate rejection of impossible jumps, scene-cut reset, enter/exit hysteresis, cross-fade | |
| 12 | Source↔target correspondence | template vertex groups ↔ 5 anchors + the mouth block centre, with a bounded 2D snap whose residual is measured and reported | **block-level**, never a point-by-point reading of the detector's 106 ordering |

Engine options: **A (FLAME)** not used (blocked, no verified weights). **B (LivePortrait)** unchanged and optional;
`reenactment_3d` runs without it. **C (PnP/TPS fallback)** and **D (perspective-mesh projection)** are the existing
`PoseBlendProjector`, reused (D drives `pose_3d`) and kept as the fallback. **E** no additional learned adapter was added.

## 3. The gate (no fabricated confidence)

`confidence = min` of the **measured** components (never an average, `None` where unmeasured):
`landmarks` (finite, inside the canvas), `layout` (`expression_tps.layout_is_plausible`), `pose_fit` (reprojection
residual / IOD), `pose_support` (observed-texture share of the rendered footprint), **`pose_coverage`**, `alignment`
(eye/mouth anchor residual before snapping), `temporal` (physical pose rate), `sharpness` (focus relative to the track's
own sharp level), `calibration`.

`pose_coverage` is the key honesty rule: the target pose must lie inside the yaw/pitch range of the source views the
asset was fitted from (+10° free, then falling to 0 over 10°). A front-only asset (depth is only a prior) therefore
refuses a 45° target, a ±40° asset refuses 70°, an asset without pitch views refuses ±28° pitch. Hard failures (invalid
landmarks, implausible layout, impossible jump, bad timestamp) withdraw the engine at once; re-entry needs `enter_frames`
good frames; a single bad frame does not drop an engaged engine.

## 4. Verification

**Synthetic ground truth** (`tests/target_world.py`): a scripted performer (a different person from the source) with known
pose, expression and timing, producing the tracker's 5- and 106-point layouts; the identity asset is built from rendered
multi-view images (Stage 3). 31 engine tests; three mutations (coverage gate off, hole release off, smoothing off) each
fail exactly the tests that should.

| Spec scenario | Result (test) |
| :--- | :--- |
| frontal | engages after calibration; pose < 3°, alignment < 0.05 IOD |
| 45° / 70° yaw | follows within 6° / 8° when the asset observed that yaw; otherwise falls back with `POSE_OUTSIDE_OBSERVED_COVERAGE` |
| profile (88°) | refused (`LANDMARK_LAYOUT_IMPLAUSIBLE`), never forced |
| looking up / down | follows ±28° within 8° with pitch views; refuses without them |
| fast head turns | no jump > 20°/frame at 600°/s; bounded lag; an impossible one-frame jump is rejected and does not corrupt the filters |
| mouth opening / talking | correlation > 0.95 / > 0.9 with the script; lip drop equals the measured opening; interior released |
| smiling | corner raise follows the measured smile; cheeks follow |
| blinking | blink > 0.8, eye released, reopened; winks leave the other eye textured |
| extreme expression | parameters clamped; no triangle folds; no strongly stretched triangle outside mouth/eyes/brows; face width unchanged; recovers |
| motion blur | focus component < 0.2 ⇒ engine withdrawn; an unmeasured focus is `None`, not "good" |
| missing landmarks | 7 distinct corruptions each fall back with a reason; re-entry needs several good frames |
| scene cuts | all temporal state discarded; the previous shot's pose does not leak |

| Acceptance criterion | Evidence |
| :--- | :--- |
| no face-shape collapse | fold guard on rigidly moving triangles, displacement cap, IOD invariance, edge-ratio bounds under extreme input |
| no eye/mouth misalignment | anchor residual after snapping < 0.05 IOD (frontal), reported per frame; < 0.1 even for a 22 mm anatomy mismatch |
| no geometry flicker | rendered anchors jitter < 50 % of the (noisy) input; hysteresis holds the state when confidence hovers around the thresholds |
| no identity drift | asset arrays hash-identical after 160 random frames; neutral geometry returns to the rest shape (< 3.5 mm); rendered shape is the source's, not the target's |
| no fabricated confidence | min-of-measured, `None` for unmeasured, `pose_coverage` from the asset's real pose range |
| deterministic fallback | identical inputs ⇒ identical images, coverage, decisions (synthetic and on real video) |

**Real local inference** (`tests/test_reenactment_session_real.py`, 6 tests, skipped if models/media are absent): a real
multi-view FaceSet (read-only legacy photographs) → Stage 3 asset → real SCRFD/2d106 tracker + real inswapper on a local
30 fps video. Measured on frames 600–690 and 1200–1280 of that video:

* first 7 frames: the proven path (calibration); then a 3-frame cross-fade; then full engine frames, confidence 0.76–0.95
  and anchor alignment 0.05–0.07 IOD on frontal footage;
* the actor turning from −31° to −59° is followed with confidence falling 0.84 → 0.56 and alignment loosening to 0.19 IOD;
  when the real 2d106 layout becomes implausible near profile the engine withdraws and the swap path takes over;
* rendered anchors jitter 0.42 px against 2.23 px for the detector's landmarks;
* overhead of the mode ≈ +40 ms/frame on a ~2 s/frame swap pipeline (the engine alone is ~35 ms on CPU).

## 5. What real data taught (and changed in Stage 3)

Real photographs exposed three things the synthetic world could not:

1. **The "5-view" legacy FaceSets contain duplicates, not right-side views** (e.g. two left profiles, two near-identical
   left 3/4 views, correlation 0.986). The Stage 3 analyzer flagged them correctly.
2. **Stage 3's atlas ghosted and seamed on real photos**, because 5-point geometry only loosely registers the views.
   Fixed (`identity-mv-2`): a texel must face the camera within ~75° (was 87°), weights are sharpened (`vis³`), a secondary
   view with < 25 % of the best view's weight is not blended in, and near-profile views are down-weighted by registration
   reliability (full trust to 60°, 10 % at 90°). Under a 3.5 px landmark misregistration the atlas error against the known
   texture drops from 17.1 to 16.5 grey levels; on the real FaceSet the ghosts and the profile smear are largely gone (small patches remain).
3. **The render coverage now follows the atlas's per-texel confidence**, so a weakly observed texel (a seam) lets the
   target's skin through instead of being painted at full strength.

## 6. Known limitations

* **Visual quality on real photos is limited by 5-point geometry.** The identity's eyes, brows, mouth and expression land
  correctly, but fine artifacts remain at the chin seam, the temple and around the eyes; some regions show the target's
  skin tone through low-confidence texels. A 68-point or learned landmark source would be the real fix.
* Facial region only (ears, skull, hair, neck are the target's); the mouth interior and closed eyelids are the target's.
* **Anatomy mismatch is intrinsic.** Five landmarks cannot separate head pitch from a different mouth-to-eye proportion
  (even the generic 5-point estimator reads −8.7° for the *same* person as the source); features are snapped into
  alignment, but the reported pitch carries that bias and `pose_coverage` accounts for it.
* Expression is measured relative to a **per-track neutral** (percentiles over a sliding window), so the first
  `calibration_frames` (8) of every track and every shot always use the proven path.
* Cheek motion is a heuristic of smile/width, not measured.
* Beyond |yaw| 60° the expression is held, not measured; near profile the real 106-point detector is implausible and the
  engine withdraws.
* No UI control was added for `render_mode` (API/parameters only), and no frontend change was made.
* The proven crop is computed on every frame (it is the fallback and the cross-fade partner), so the engine's cost is
  added to the swap pipeline's; skipping the swap on fully engaged frames is a possible optimisation that was not made.
