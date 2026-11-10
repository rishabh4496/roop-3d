# Native dense facial backbone

`NativeMorphableBackbone` decodes the actual upstream networks, without requiring a custom all-in-one differentiable renderer ONNX export. `MorphableBackbone.load` selects it when `bundle.json` declares `roop3d.native_dense_face.v1`. The original custom export schema remains supported.

## Current readiness

Public assets are downloaded under ignored `app/roop_3d/.runtime/models/dense_face/upstream/`. The real networks have been exported under `prepared/`. No checkpoints, source photographs, or downloaded model data are tracked by Git.

| Component | What was actually checked locally |
| --- | --- |
| 3DDFA-V3 MobileNet-V3 | Strict checkpoint state loading; two-tensor PyTorch/ONNX parity, max absolute error `1.383e-5`; inference on an upstream real photograph with measured detector landmarks |
| SMIRK expression encoder | Strict checkpoint state loading; expression-50, jaw-3 and independent eyelid-2 outputs; two-tensor parity, max absolute error `1.103e-6` |
| HRN mid network | Strict checkpoint state loading; native 6-channel 64px input; two-tensor parity, max absolute error `1.941e-4` |
| HRN high network | Strict checkpoint state loading; native 9-channel 256px input; two-tensor parity, max absolute error `9.102e-5` |
| Native reconstruction crop | Pixel-identical to upstream POS/PIL preprocessing on the measured example; retained inverse includes resize pixel-centre offsets |
| Shared BFM topology | Actual HRN and 3DDFA triangles match exactly; recentered neutral shapes match exactly; native HRN UV differences below `3e-8` |
| FLAME anatomical masks | Imported official named regions; 1,860 allowed vertices before triangle pruning; scalp, ears, neck/boundary and eyeballs excluded |

These checks establish export and geometry conventions. They do not establish end-to-end reconstruction or swap quality. A runtime bundle was assembled on 2026-11-10 using the official FLAME 2023 Open and FLAME 2020 `generic_model.pkl` (user-downloaded into the git-ignored `flame/` folder). `convert-expression` passed its gate (max RMS 0.19 mm, p95 0.25 mm per unit coefficient). Eyelid directions are SMIRK's `l_eyelid.npy`/`r_eyelid.npy` (index 0 = left, 1 = right, per SMIRK `FLAME.py`), checked only for location and symmetry on Open geometry. BFM anatomy masks and surface registration come from `tools/register_bfm_flame.py` (similarity ICP + thin-plate refinement; FLAME face to BFM 0.32 mm mean, BFM landmarks within 0.9 mm of the FLAME surface; no landmark embedding, so lip/eyelid correspondence is not validated). The expression/HRN graphs carry static output-shape annotations written by `tools/annotate_onnx_shapes.py` (`*.static.onnx`, outputs identical to the originals). Remaining validation: lip/eyelid registration, jaw/eyelid articulation and appearance on real video. Original outstanding list, now partly addressed:

1. The actual **FLAME 2023 Open** model. Its official download currently presents an account sign-in. The application does not request or store FLAME passwords.
2. The SMIRK checkpoint's original FLAME model, to measure its expression conversion into Open. The public checkpoint was built around FLAME2020; the official FLAME2023-to-Open conversion cannot simply be applied to those coefficients.
3. Validated BFM-to-FLAME and FLAME-to-BFM registered surface correspondences, an aligned HRN UV chart on FLAME, anatomical BFM inclusion/exclusion masks, and eyelid directions validated on Open geometry.
4. Source/target photo and video validation of the assembled chain, including 6 GB and 12 GB runtime measurements. The application keeps geometry networks on CPU to avoid unbudgeted CUDA arenas; this is not a realtime performance claim.

The pipeline intentionally has no MakeHuman/CG output fallback when any required input is missing. Frontal standard neural swaps remain available; the automatic extreme-pose branch also requires the bundle.

## Reproduce public model preparation

From `app/`, run preparation with a Python environment containing the application's dependencies plus the `dense-face-export` extra (`torch==2.7.1`, `torchvision==0.22.1`, `timm==0.9.16`). CPU PyTorch wheels are sufficient. The optional exporter can use a separate environment inside `.runtime/`; runtime inference needs no PyTorch. `pillow==12.3.0` is a runtime dependency for the upstream-compatible crop.

```powershell
python -B -m roop_3d.tools.prepare_dense_face fetch --output roop_3d/.runtime/models/dense_face/upstream

python -B -m roop_3d.tools.prepare_dense_face export --component 3ddfa --source roop_3d/.runtime/models/dense_face/upstream/3ddfa --checkpoint roop_3d/.runtime/models/dense_face/upstream/net_recon_mbnet.pth --backbone mbnetv3 --output roop_3d/.runtime/models/dense_face/prepared/coarse.onnx

python -B -m roop_3d.tools.prepare_dense_face export --component smirk --source roop_3d/.runtime/models/dense_face/upstream/smirk --checkpoint roop_3d/.runtime/models/dense_face/upstream/SMIRK_em1.pt --output roop_3d/.runtime/models/dense_face/prepared/expression.onnx

python -B -m roop_3d.tools.prepare_dense_face export --component hrn --source roop_3d/.runtime/models/dense_face/upstream/hrn --checkpoint roop_3d/.runtime/models/dense_face/upstream/epoch_10.pth --output roop_3d/.runtime/models/dense_face/prepared/hrn
```

`fetch` checks SHA-256 before reusing or installing files and never overwrites different local bytes. Architecture sources are pinned to upstream revisions; checkpoint URLs are pinned by their bytes. The downloaded checkpoint sizes total about 600 MB. Export uses `weights_only=True`, strict state loading and two independent seeded inputs for torch/ORT comparison. Partial graph files are removed on failure; existing exports are not overwritten. `.export.json` reports contain checkpoint/source hashes and measured parity, including HRN's actual learned `alpha`/`beta` scales. There are no random-weight exports.

ResNet50 3DDFA is also supported with `--backbone resnet50` and its corresponding checkpoint. It has not been exported in this checkout. SMIRK exports only its expression encoder, with jaw and eyelids; its generator and weak-perspective pose/shape encoders are not needed by this path.

## Prepare anatomy, expression conversion and geometry

The CLI imports trusted upstream pickle-backed assets once into pickle-free runtime NumPy archives. Never pass an uploaded FaceSet or unknown file to these importers.

```powershell
python -B -m roop_3d.tools.prepare_dense_face flame-anatomy --masks roop_3d/.runtime/models/dense_face/upstream/FLAME_masks.pkl --output roop_3d/.runtime/models/dense_face/prepared/flame_anatomy.npz

python -B -m roop_3d.tools.prepare_dense_face bfm --model PATH/face_model.npy --standard PATH/similarity_Lm3D_all.mat --hrn-uvs PATH/bfm_uvs2.npy --anatomy PATH/bfm_anatomy.npz --output PATH/prepared_bfm

python -B -m roop_3d.tools.prepare_dense_face convert-expression --original PATH/checkpoint_original_flame.pkl --open-model PATH/flame2023_Open.pkl --anatomy PATH/flame_anatomy.npz --output PATH/smirk_to_open.npy

python -B -m roop_3d.tools.prepare_dense_face flame --model PATH/flame2023_Open.pkl --conversion PATH/smirk_to_open.npy --eyelids PATH/open_validated_eyelid_dirs.npy --output PATH/flame_rig.npz
```

Replace `PATH` entries with real paths **inside this checkout**. These commands do not create placeholder models, automatic BFM anatomy masks, or unvalidated surface registrations. `convert-expression` checks identical registered FLAME triangle topology, fits the checkpoint's original 50 expression directions to Open's expression subspace, and records each unit coefficient's anatomical facial RMS and p95 error. It rejects a conversion above 1 mm RMS or 2 mm p95 by default. These are preparation error limits, not evidence of expression recovery on video; neutral geometry, jaw and eyelids still need validation.

Every anatomy archive has binary `[N]` arrays `face_surface`, `scalp_hair`, `ears`, `neck`. BFM part names are the actual upstream order: right eye, left eye, right eyebrow, left eyebrow, nose, upper lip, lower lip, skin. Their union includes all 35,709 native vertices; the eight-part mask alone is **not** a scalp/neck-removal classifier. The BFM importer retains this annotation and requires a separate explicit anatomical surface mask. Active triangles require all three vertices to be allowed, and the resulting surface is physically compacted. FLAME's prepared native vertex count remains 5,023; its active facial surface is smaller.

`flame_rig.npz` contains `template_m [5023,3]`, `expression_dirs_m [5023,3,K]`, `smirk_to_expression [K,50]`, `pose_dirs_m [36,15069]`, `joint_regressor [5,5023]`, `parents [5]`, `skinning_weights [5023,5]`, and anatomical left/right `eyelid_dirs_m [2,5023,3]`. The importer extracts these from Open plus the measured conversion and supplied eyelid directions. Runtime uses full linear blend skinning with pose correctives on the supplied identity anatomy; it does not add a linear jaw displacement in place of articulation.

`registration.npz` requires:

| Array | Contract |
| --- | --- |
| `bfm_to_flame_indices`, `bfm_to_flame_weights` | integer `[5023,3]` BFM vertex IDs and convex barycentric weights, registered by anatomical correspondence |
| `flame_to_bfm_indices`, `flame_to_bfm_weights` | integer `[35709,3]` FLAME vertex IDs and convex barycentric weights |
| `flame_to_bfm_rotation` | proper rotation `[3,3]`, FLAME metric axes into native BFM metric axes |
| `flame_to_bfm_scale` | positive scalar registration scale |
| `flame_to_bfm_translation_m` | translation `[3]` in metres |
| `flame_hrn_uvs` | `[5023,2]`, registered HRN BFM-chart coordinates; UV seams must be handled during registration, not averaged across charts |

BFM identity offsets transfer into FLAME before expression articulation. Expressed FLAME deltas transfer back into BFM for HRN. On the FLAME output branch, HRN mid geometry transfers back onto FLAME and high displacement samples its registered HRN chart. Each final family uses its own face-only native topology. Registration quality must be measured at the lips, eyelids, nostrils and jaw; matching vertex counts or nearest index reshaping is insufficient.

## Native graph/runtime contracts

| Graph | Input | Output |
| --- | --- | --- |
| `coarse` | RGB `[1,3,224,224]`, `[0,1]`, Deep3D POS crop | `coefficients [1,257]` |
| `expression` | RGB `[1,3,224,224]`, `[0,1]` | `expression [1,50]`, `jaw [1,3]`, `eyelids [1,2]` |
| `hrn_mid` | `features [1,6,64,64]`: UV canonical XYZ plus observed photographic RGB `[-1,1]` | `deformation [1,3,64,64]` |
| `hrn_high` | `features [1,9,256,256]`: canonical XYZ, photograph reprojected after mid deformation, upsampled mid map | `displacement [1,1,256,256]` |

Actual 3DDFA checkpoints have **257** coefficients: identity 80, coarse expression 64, albedo 80, Euler 3, SH 27, translation 3. Native encoders output albedo/SH coefficients, but this application does not decode or render them. Rotation and translation use the actual 3DDFA camera equation. The metric gauge is explicitly fixed to 0.1 m per native BFM unit with `s=1`; camera depth is `1m - 0.1*tz`. Monocular reconstruction does not measure absolute scale.

SMIRK cropping uses the coarse fit's projected 68-point facial bounding box with scale 1.4. Upstream SMIRK uses MediaPipe's bounding box; this substitution is explicit in manifest/capability metadata and requires photo validation. Target facial anatomy comes from the target fit. Source identity embeddings condition the neural model separately; source skull geometry does not replace the target bones.

HRN geometry is reconstructed in its native UV coordinate system. XYZ mid deformation uses HRN's actual y-up sampling and learned `alpha`; high displacement uses learned `beta`, converts native units to metres and is applied along outward normals. High-resolution photographic inputs are sampled from the original image, not enlarged from the 224px crop. Both self-occluded and padded-image regions have zero observed UV support. HRN's original de-retouching, face fattening, photometric optimization and shaded visualization are not used to generate the output face. Geometry quality under this coupled expression/detail input distribution has not yet been measured.

UV displacement is sampled on a shared-edge refined facial surface, without adding anatomical regions or filling facial openings. Native BFM/FLAME model vertex counts remain unchanged. `detail_subdivisions` accepts 0, 1 (default) or 2. The 6 GB/CPU profiles cap this at one level and 12 GB allows two levels. Relief below the refined triangle or UV-map resolution is not resolved; neural skin detail is not a guarantee of measured microgeometry. The final path remains real-photo correspondence → selected neural swapper at native resolution/alignment → UltraMax → anatomical facial coverage compositor. Source holes are filled by neural identity synthesis, with the target photograph as input context; they are never reported as observed texture. Completely unsupported source/target poses still fail explicitly. No Lambertian or Phong face render enters the compositor.

Correspondence accepts photographic samples only when every nonzero OpenCV bilinear tap is both observed and depth-visible. Depth checks use the interpolated source triangle plane at each tap, including sloped surfaces and partial occlusions. `last_fields` addresses the bounded aligned source photograph cache; `last_original_fields`, paired with `last_source_view_indices`, exposes the same correspondence in original `.fsz` photograph coordinates. They are diagnostics, not claims of unbounded-resolution texture caching.

## Assemble the bundle

Create a preparation JSON inside the checkout with these fields and genuine asset paths. Place all assets beneath the destination bundle directory, for example under its `prepared/` directory. Paths in the preparation JSON are relative to that JSON. Supplied hashes are checked; omit `sha256` to have the publisher compute it from local bytes. Never use the illustrative hash strings below as real digests.

```json
{
  "schema": "roop3d.native_dense_face.v1",
  "topology": "bfm35709",
  "expression_basis": "smirk_flame50",
  "detail_subdivisions": 1,
  "bfm_unit_m": 0.1,
  "flame_model": "flame2023_Open",
  "smirk_crop": "projected_bfm68_bbox_1.4",
  "mask_provenance": "actual annotation version and anatomical cut validation report",
  "expression_mapping_evidence": "checkpoint-original to Open conversion report and validation",
  "registration_evidence": "anatomical registration version and measured errors",
  "hrn_alpha": 0.5436066389083862,
  "hrn_beta": 1.6597100496292114,
  "assets": {
    "topology": {"file": "prepared_bfm/bfm_topology.npz", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "bfm_topology": {"file": "prepared_bfm/bfm_topology.npz", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "bfm": {"file": "prepared_bfm/bfm.npz", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "flame_rig": {"file": "flame_rig.npz", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "registration": {"file": "registration.npz", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "coarse": {"file": "prepared/coarse.onnx", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "expression": {"file": "prepared/expression.onnx", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "hrn_mid": {"file": "prepared/hrn/hrn_mid.onnx", "sha256": "REPLACE_WITH_REAL_SHA256"},
    "hrn_high": {"file": "prepared/hrn/hrn_high.onnx", "sha256": "REPLACE_WITH_REAL_SHA256"}
  }
}
```

Use `flame2023open` and a genuine Open face-only topology archive for the alternate surface; `bfm_topology` still supplies the BFM HRN surface. The alphas above belong specifically to the downloaded HRN epoch-10 checkpoint. Always use scales recorded by the exporter for the selected checkpoint.

```powershell
python -B -m roop_3d.tools.prepare_dense_face assemble --config PATH/native-preparation.json --output roop_3d/.runtime/models/dense_face/bundle.json
```

Assembly verifies SHA-256, anatomy/rig/registration contracts, refinement budgets and every native ONNX graph contract using CPU-only sessions. It never copies or overwrites assets, creates a GPU context, runs inference, or invents missing registrations. ONNX weights must be embedded, so every weight byte is covered by the manifest digest. It rechecks assets after validation and publishes the manifest atomically without replacing an existing manifest. The legacy `prepare_dense_face assemble` command delegates to this same publisher.

Check local readiness without attempting a render:

```powershell
python -B -m roop_3d.tools.dense_face_bundle status --bundle roop_3d/.runtime/models/dense_face/bundle.json
python -B -m roop_3d.tools.dense_face_bundle assemble --config PATH/native-preparation.json --output roop_3d/.runtime/models/dense_face/bundle.json
```

`status` (also `readiness`) exits 0 only for a structurally valid published bundle, 1 for unavailable/invalid assets, and 2 for invalid CLI arguments. It discovers nested prepared encoder exports and reports ambiguous filenames rather than guessing. Optional `--config` and `--prepared` point at a preparation manifest or export directory. Supplied export reports are inventoried as evidence, not independently certified quality. Neither this status nor provenance strings certify anatomical registration or reconstruction quality. End-to-end readiness requires source fitting, hardware rasterization, neural swapping and UltraMax to succeed on real photographs and video.

See [the dated validation record](dense-face-validation-2026-11-09.md) for the checks actually run and remaining activation blockers.

## Primary upstream references

- [3DDFA-V3 code and asset documentation](https://github.com/wang-zidu/3DDFA-V3/tree/e15385837dc1e051a6cf376b3827f2e279537b29): MIT code, BFM/Exp-PCA data with separate terms; [author-hosted assets](https://huggingface.co/datasets/Zidu-Wang/3DDFA-V3/tree/main/assets).
- [SMIRK](https://github.com/georgeretsi/smirk/tree/c7de404c4389f073906a6db1adabf62efcea3f35): MIT encoder code; the repository documents its original FLAME dependency and public checkpoint.
- [HRN](https://github.com/youngLBW/HRN/tree/58833698fd199fc08574a3cb9a8d8959fc58c213): Apache-2.0 code, separately sourced BFM assets and the author's public checkpoint linked from its assets README.
- [Official FLAME site](https://flame.is.tue.mpg.de/): FLAME 2023 Open is CC-BY-4.0; earlier model releases have different terms. The official site links the [2023-to-Open expression converter](https://github.com/Suraj-Bhor/flame_to_flame_open_converter). Its input model edition matters for SMIRK.
