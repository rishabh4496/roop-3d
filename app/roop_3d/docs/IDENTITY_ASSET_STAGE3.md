# Stage 3: Personalized 3D Source Identity Reconstruction

**Document Path:** `app/roop_3d/docs/IDENTITY_ASSET_STAGE3.md`
**Scope:** turn an authorised multi-view FaceSet into a persistent, identity-specific 3D face asset, and say exactly
what that asset does and does not contain.

> **Read this first.** The asset is *the anthropometric facial template, deformed to the identity-specific 3D landmark
> positions that the supplied views actually constrain*, plus a texture atlas built only from pixels that were
> observed. It is **not** a dense photogrammetric head, it is **not** FLAME, and it never claims complete 3D recovery.
> Every vertex and texel carries a confidence state, and the manifest lists what is missing.

## 1. Path mapping (the stage prompt vs. the repository)

All files named by the prompt exist. The new code lives beside them:

| Prompt item | Real location | Change |
| :--- | :--- | :--- |
| `faceset/faceset_reader.py` | `app/roop_3d/faceset/faceset_reader.py` | fixed a 68-point landmark defect, flags fabricated landmarks, round-trips `landmarks_68` |
| `faceset/schemas.py` | `app/roop_3d/faceset/schemas.py` | additive `FaceView.landmarks_68`, `FaceView.landmarks_trusted` (defaults keep every old caller valid) |
| `faceset/pose_matrix.py`, `pose_selector.py` | unchanged | reused (`view_pose` for the legacy→standard pitch frame) |
| `reconstruction/geometry_types.py` | same | additive `WeakPerspectiveCamera` |
| `reconstruction/uv_synthesizer.py` | same | new camera path (z-buffer, gain compensation, outlier rejection, bounded interpolation); legacy path untouched except for a chunked-sampling fix |
| `reconstruction/mesh_io.py` | same | `export_mesh_glb` gained custom vertex attributes + `extras`; new `read_glb` validator |
| `reconstruction/flame_fitter.py`, `tps_fallback.py` | unchanged | reused (`huber_weights`, `build_canonical_head_mesh`); FLAME is gated, see §4 |
| *(new)* | `reconstruction/faceset_quality.py` | Phase A |
| *(new)* | `reconstruction/multiview_fit.py` | Phase B |
| *(new)* | `reconstruction/identity_asset.py` | Phases D + E |
| *(new)* | `tests/identity_world.py` + `tests/test_identity_asset_*.py` | synthetic ground-truth world and tests |

## 2. What was found in the existing code (evidence, not opinion)

1. **`FaceView.landmarks` is 5 points, and the V2 reader truncated with `lm_raw[:5]`.** For an archive that stores a
   68-point set, the first five entries are *jaw* points, so "eyes / nose / mouth" were wrong. Fixed (the canonical
   eyes, nose tip and mouth corners are derived; the full 68 are kept in `landmarks_68`). The new test fails on the
   original reader and passes on the fix.
2. **The reader fabricates landmarks** (fixed 64 px layout, or fractions of the image, when an archive has no
   geometry or the detector finds nothing). They are now flagged `landmarks_trusted=False`, and the analyzer also
   recognises the constants, so a reconstruction can never treat them as observations.
3. **No dense-landmark or occluder inference adapter exists in the app.** `liveportrait_landmark.onnx`,
   `face_occluder.onnx`, `xseg*.onnx` appear only as registry/audit entries. Their tensor contracts are not verified
   here, so none is wired in. Consequence: real FaceSets give a **5-point** multi-view fit; **68-point** fitting is
   used when the FaceSet carries 68 points. Semantic occlusion is *not measurable* and is reported as such.
4. **FLAME is `blocked`** in the registry (`flame-2020`: no filename, no hash, `manual_acceptance_required`).
5. **`UVSynthesizer` (legacy path)** had no self-occlusion test, no exposure compensation, and inpainted *every*
   unobserved face texel as INTERPOLATED regardless of distance from an observation. With a front-only set that
   labels the whole side of the face INTERPOLATED. Its `cv2.remap` sampling also fails once more than 32 766 texels
   are sampled (atlas sizes above ~128 px); fixed with a chunked sampler shared by both paths.
6. **`export_mesh_glb` writes y-down vertices unchanged** and flips `v`, while `UVSynthesizer` reads `v` as an image
   row. A viewer would show the face upside-down with inverted culling and a flipped texture. The legacy exporter is
   left as is (existing callers); the asset exporter converts the frame explicitly (`to_gltf_frame`) and a round-trip
   test proves orientation and texture registration through the GLB.

## 3. Pipeline

```
FaceSetDocument
  └─ Phase A  analyze_faceset ───────────► FaceSetCoverageReport (usable views, coverage, image requests)
  └─ Phase B  fit_identity_landmarks ────► IdentityShapeFit      (3D landmarks + posterior confidence + hold-out error)
              deform_template ───────────► vertices + per-vertex support
  └─ Phase C  UVSynthesizer (cameras) ───► atlas + OBSERVED / INTERPOLATED / UNSEEN + per-texel confidence
  └─ Phase D  IdentityAsset + manifest ──► GLB (inspectable), arrays.npz, texture.png, manifest.json
  └─ Phase E  IdentityAssetStore ────────► content-addressed reuse under roop_3d/.runtime/identity_assets/
```

### Phase A: FaceSet quality / coverage (`faceset_quality.py`)

Measured per view: pose (standard frame), sharpness (Laplacian over the face box), exposure (mean luma, clipped
fractions), landmark quality (observed? inside the image? eye/nose/mouth topology after de-rolling? stored pose vs. a
pose re-estimated from the landmarks), face size, border truncation, optional supplied occlusion mask, exact and
perceptual duplicates, and identity consistency (single-link clustering of ArcFace embeddings at cosine ≥ 0.30).

* One foreign face → excluded as `IDENTITY_OUTLIER`. Two groups of ≥2 → `blocked_inconsistent_identity`: nothing is
  built from a mixed set and the report says so.
* A perceptual-hash duplicate additionally needs a near-identical pose (a 9×8 hash cannot separate two poses of one
  person on one background; this was a bug found by the tests).
* Coverage: yaw targets −75/−45/0/+45/+75 (±20°, "image left/right" wording to avoid left/right ambiguity), pitch
  −20/0/+20 (±12°). Missing ones become concrete `requests` ("[required] Add a three-quarter portrait with the head
  turned toward the image LEFT …"). `readiness` describes **coverage only**
  (`front_only | partial_coverage | full_coverage | insufficient | blocked_inconsistent_identity`); the level actually
  achieved is reported by the fit.

### Phase B: shape estimation (`multiview_fit.py`)

A MAP bundle adjustment under the scaled-orthographic camera already used by `core.pose.estimate_pose_5pt`:
unknowns are a 3-vector displacement per landmark and yaw/pitch/roll/scale/translation per view; residuals are
Huber-reweighted; priors are Gaussian on the displacement and on the stored pose. The Jacobian is analytic (verified
against finite differences to 5e-7 on fully visible landmarks; the visibility weights are treated as constants). The
Gauss-Newton Hessian gives a **posterior**, so `confidence_z` is measured: with a single frontal view the depth rows of
the Jacobian vanish and depth confidence stays ≈ 0.

Of the seven evaluated items: PnP initialisation was **not** used (the template is defined in the project's
z-toward-viewer frame, the frame of `core.pose` and of every stored FaceView pose; `solvePnP` would introduce a second
pose convention, so the initial poses come from the stored pose instead); robust fitting, shape
coefficients, multi-view consistency (leave-one-view-out), regularisation (priors + bounded Gaussian deformation) are
implemented. Jaw/expression: *measured* per view (§5), not fitted, because no verified parametric expression model is
available. The existing FLAME fitter is reused only for `huber_weights`; its `SyntheticFLAMEModel` is not evidence of a
real identity and is not used.

Inconsistent views (hold-out error above 6 mm **and** more than 2× the median of the other views) are dropped one at a
time, worst first, and the fit is redone.

### Phase C: texture (`UVSynthesizer`, camera path)

Per view and texel: back-face/normal weight (`vis²`), **z-buffer self-occlusion** (the nose cannot paint the cheek),
local relative sharpness, view quality (landmark consistency × exposure × sharpness), image-edge feather,
**silhouette feather** (mask-edge texels would otherwise sample hair/background), optional occlusion mask. Then
per-view per-channel **gain compensation** from overlaps, and **photometric outlier rejection** against the per-texel
median (≥3 samples). Texel states:

* `OBSERVED`: total blend weight ≥ 0.04 after rejection.
* `INTERPOLATED`: unobserved, within `max_interpolation_fraction × atlas` (default 2 %) of an observed texel; low
  confidence (≤ 0.3).
* `UNSEEN`: everything else; colour is exactly zero. Nothing is generated for the back of the head, ears or skull, and
  the mesh does not contain them.

### Phase D: asset contents

`vertices`, `faces` (triangles), `normals`, `uvs`, texture atlas (BGR), `vertex_confidence`/`vertex_state`,
`texel_confidence`/`texel_state`/`texel_view_count`/`texel_source_view`, `landmarks_3d`, `landmark_confidence`,
`landmark_sigma`, `template_landmarks`, `shape_params` (landmark displacement from the template, metres),
`expression_params` (per-view measurements), canonical head pose, source image references (index, SHA-256, used or the
exact exclusion reasons, refined pose, reprojection and hold-out error), model version, model hashes, and an SHA-256
of every array. Vertex state: `UNSEEN` if no observed texel at its UV; `OBSERVED` only if visible **and** both in-plane
and depth confidence reach 0.6; otherwise `INTERPOLATED`.

`head.glb` is a standard glTF 2.0 binary with `_CONFIDENCE` and `_STATE` per-vertex attributes and the hashes/claims/
missing-information in the root `extras`; `read_glb()` parses and validates it (header, chunks, accessor bounds,
indices).

### Phase E: persistence

`IdentityAssetStore` writes `roop_3d/.runtime/identity_assets/<faceset_id>/<build_key[:24]>/{manifest.json,arrays.npz,
texture.png,head.glb}` (atomic directory rename, `safe_path`-confined, FaceSet id validated, 3 newest builds kept).
The build key hashes image bytes, embeddings, landmarks, bboxes, poses, quality, occlusion masks, settings, the
template geometry hash, FLAME status and caller-supplied `model_hashes`. A renamed FaceSet reuses the asset; changing
an image, a landmark, a setting or a model hash rebuilds; arrays that no longer match their manifest hash are
treated as absent. The asset stores no FaceSet path and no source image bytes (only SHA-256 references) and never
modifies the FaceSet.

## 4. FLAME policy

`flame_status()` returns `available: False, used: False` unless the registry record is unblocked **and** has a hash
**and** a locator, and even then reports that this build ships no FLAME shape-fitting adapter. The manifest records the
status, `flame_weights_sha256: null`, and `shape.flame_shape_coefficients: null`. No synthetic FLAME geometry is used
as evidence of identity.

## 5. Capability reporting for absent inputs

| Missing | What the code reports |
| :--- | :--- |
| 68-point landmarks | layout `5pt`; limitation "only 5 landmarks per view"; depth confidence is lower |
| occluder model | `occlusion: … semantic occluder model: unavailable`; `occlusion_fraction=None` unless a mask is supplied |
| FLAME weights / license | `flame.available=false, used=false` with the reason |
| side / profile / pitch views | `requests` + `missing_information` + `coverage.missing` |
| yaw span < 25° | `reconstruction_level = single_view_prior`, depth not triangulated, `depth_triangulated=false` |

## 6. Verification

Real local-model inference was **not** exercised for this stage: no verified multi-view *real photo* FaceSet is
available, and the only landmark source in the app (SCRFD, 5 points) cannot be run meaningfully on synthetic
renders. Verification therefore uses a **synthetic ground-truth world** (`tests/identity_world.py`): a known 3D head and
UV texture rendered with painter's visibility through known cameras into real PNGs, with noisy landmarks and noisy
stored poses. The code under test sees only pixels, landmarks and stored poses; assertions compare against the hidden
truth. Numbers below are from the test environment (CPU, 256 px atlas):

| Scenario | Result |
| :--- | :--- |
| 68-pt, 3–5 views | landmark error 0.4–0.6 mm vs 2.9 mm for the generic template |
| 5-pt, 3–5 views | 0.4–0.9 mm vs 5.2 mm for the template |
| front-only (either layout) | error ≈ template (no gain), depth confidence 0, **0 OBSERVED vertices** |
| two different identities | each reconstruction is ≥ 3× closer to its own truth than to the other's; aspect ratio within 2 % |
| atlas vs ground-truth texture | MAE < 22 grey levels on OBSERVED texels (68-pt, 3 views) |
| single turned view | far-side texels never OBSERVED; UNSEEN texels exactly zero; interpolation bounded |
| z-buffer on/off | single yaw-55 view: atlas error strictly lower with the z-buffer |
| lighting 0.55×–1.15× | gains ordered correctly; brightness error std-dev lower than uncompensated |
| eye occluder (consensus / mask) | rejected samples > 0; eye texels within 25 grey levels of truth |
| corrupted view | flagged and dropped; refit error < 0.4× template error |

## 7. Known limitations (all also in each manifest)

* Facial region only; no ears, skull or back of head. Surface detail between landmarks is template-derived (see
  per-vertex confidence).
* Absolute head size is a prior (not observable without a reference length); aspect ratio is observable.
* Pose precision is that of the 5-point estimator plus refinement (a few degrees).
* Expression is measured (mouth width/opening, brow raise, yaw-corrected, unmeasurable beyond |yaw| 60°), not fitted.
* Lighting is gain-compensated only; shading and speculars are baked into OBSERVED texels.
* Runtime: sub-second to a few seconds per build; 68-point multi-view builds measured between ~1.5 s and ~20 s on this
  32-thread machine. A profile showed the time is spent in `np.linalg.solve` (about 10 ms per small solve, slower than
  expected); the cause was not investigated further. No GPU is used.
* Not validated on real multi-view photographs. No API route or UI panel was added in this stage.
