# Dense facial correspondence backbone

The live `pose_3d` / `reenactment_3d` branches and automatic extreme-pose branch now require a local, compatible three-tier reconstruction bundle. They do not use the MakeHuman full head, a neck extension, diffuse albedo completion, SH lighting, or a high-pass neural residual. Standard frontal neural swapping remains available without reconstruction assets.

The application now has a **native encoder/decoder path**, an upstream checkpoint exporter, pinned public-asset fetching, and the downstream correspondence/neural pipeline. See [native preparation, measured validation and outstanding assets](native-dense-face-setup.md). Real 3DDFA-V3, SMIRK expression and HRN mid/high checkpoints have been exported and checked against PyTorch locally. The full three-tier chain is not activated: the FLAME 2023 Open model, checkpoint-specific expression conversion, anatomical masks and registered cross-topology mappings must be supplied and validated. There is currently no assembled runtime bundle in this checkout, and real-photo three-tier quality has **not** been validated. Missing or incompatible assets produce `DENSE_FACE_ASSETS_MISSING` / `DENSE_FACE_BUNDLE_INVALID`.

## Actual image path

1. Fit each selected real `.fsz` source photograph once. Synthetic source views cannot count as observed appearance.
2. Fit each target frame independently, retaining its coarse facial anatomy, expression and camera. There is no source-skull transplantation in this path.
3. Physically compact the native topology using anatomical face inclusion, eight-part BFM annotations, and explicit scalp/hair, ear and neck exclusions. Keep a triangle only when all three vertices survive. Do not extrude the jaw into the neck.
4. Apply articulated expression geometry and HRN mid-frequency vertex offsets, then deterministically refine the already-pruned facial triangles and sample high-frequency UV displacement along interpolated surface normals. Shared-edge indexing is identical for source and target. Both source and target use this operation. Refinement does not add scalp, ears, neck, or new anatomical regions.
5. Rasterize geometry attributes only. At each target pixel, perspective-interpolate the corresponding **source camera-space 3D point**, project it into the source photograph and check its source depth. Both source and target z-buffers and front-facing triangle tests constrain sampling. Every nonzero bilinear photograph tap must pass visibility and observation support. Its expected depth comes from the corresponding source triangle's inverse-depth plane, so slanted surfaces are not incorrectly compared to the centre's depth. Interpolating projected source pixels would be incorrect under perspective.
6. Blend observed photograph samples across the nearest source views. `last_fields` exposes target-to-source sampling maps in **aligned source photograph cache pixels**. `last_original_fields` exposes the same `W_3d_to_target` maps transformed into original `.fsz` image pixels, and `last_source_view_indices` identifies each source view in matching order. The inverse-sampling convention maps a native neural-target crop pixel to a source image pixel. Out-of-frame and occluded samples have zero confidence and coordinates `(-1, -1)`. Maps are reset before each frame, including failed inference, so stale fields are not reported as current.
7. Condition the selected neural swapper on that photographic warp and the existing pose-selected identity embedding. Use the model's native resolution and alignment: InSwapper 128, HyperSwap 256, RealSwap 512 are handled by the existing architecture adapter. Feed the complete neural output through the existing UltraMax implementation (CodeFormer plus pyramid fusion). The compositor receives only this neural output, bounded by the target's anatomical facial surface. Source visibility controls photographic sampling separately from the replacement mask.

Unobserved source regions use the real target photograph as neural input context, then receive the neural model's full output inside the anatomical face boundary. They are synthesized appearance, not observations; they are never filled by a diffuse shader or stretched atlas. Completely unsupported poses raise `DENSE_FACE_NO_CORRESPONDENCE`. Target parser/occluder masks still apply. Displacement is sampled at refined mesh vertices rather than native vertices alone. Features smaller than a refined triangle or the learned UV-map resolution remain unresolved. Subdivision does not recover missing pores or improve an inaccurate displacement prediction. A trained neural network may synthesize texture, but the integration cannot guarantee recovered pores or identity fidelity.

## Bundle location and manifest

Use `app/roop_3d/.runtime/models/dense_face/bundle.json`. Every asset must be inside this directory tree and this checkout. Digests cover the exact local bytes; no models or media belong in Git. Inference does not download models. The explicit preparation CLI can fetch pinned public checkpoints.

The recommended `roop3d.native_dense_face.v1` bundle uses native encoder outputs with in-application geometry decoding, documented in [native setup](native-dense-face-setup.md). The following `roop3d.dense_face_bundle.v1` interface remains supported for existing custom exports.

Example structure (replace every placeholder with inspected metadata and actual hashes):

```json
{
  "schema": "roop3d.dense_face_bundle.v1",
  "topology": "bfm35709",
  "coordinates": "opencv_metres",
  "uv_origin": "bottom_left",
  "crop": "arcface_128_rgb_0_1",
  "detail_crop_size": 512,
  "expression_basis": "smirk_flame50",
  "detail_topology": "bfm35709",
  "mask_provenance": "upstream revision, annotation filename and anatomical-mask derivation",
  "expression_mapping_evidence": "registered FLAME-to-BFM mapping revision and validation report",
  "assets": {
    "topology": {"file": "topology.npz", "sha256": "<64 lowercase hex characters>"},
    "coarse": {"file": "coarse.onnx", "sha256": "<64 lowercase hex characters>"},
    "expression": {"file": "expression.onnx", "sha256": "<64 lowercase hex characters>"},
    "detail": {"file": "hrn.onnx", "sha256": "<64 lowercase hex characters>"}
  }
}
```

`topology` may instead be `flame2023open` (5023 native vertices). `detail_topology` must match. Converting BFM HRN detail onto FLAME requires a registered UV/surface mapping inside the export. A vertex count or reshaped tensor is not a valid mapping. Select one compatible family per session; graph families are never mixed by index. Use a separate bundle directory and the Python loader's explicit `bundle` argument when validating an alternate family.

`topology.npz` uses NumPy arrays with `allow_pickle=False`:

| Array | Contract |
| --- | --- |
| `vertices_m` | float `[N,3]`, native neutral geometry, OpenCV model axes, metres |
| `faces` | integer `[F,3]`, zero-based, counterclockwise viewed from outside |
| `uvs` | float `[N,2]`, `[0,1]`, bottom-left origin, same chart as detail graph |
| `face_surface` | binary `[N]`, anatomical forehead below hairline through jaw border |
| `scalp_hair`, `ears`, `neck` | three binary `[N]` masks, always excluded |
| `semantic_parts` | binary `[8,N]`, required for BFM, explicit upstream eight-part membership |

Only annotated face vertices survive. The upstream eight-part masks are facial-part annotations, **not** a universal scalp/neck classifier. Supply the explicit anatomical inclusion/exclusions; never guess semantic IDs. Model UV seams must be normalized into per-vertex UVs before bundling.

## Export graph contracts

These are the older **custom application export contracts**, not the native output names of upstream models. Such custom graphs must include verified upstream preprocessing transformations and decoders. The included preparation tool exports the native encoder graphs for `roop3d.native_dense_face.v1` instead.

Every `rgb` input is float32 `[1,3,224,224]` RGB `[0,1]`. The application warps the original photograph directly to 224 pixels using the same canonical frame as `aligned_crop(...,128,template="arcface_112")`, scaled by `224/128`. Exporters must explicitly undo/redo upstream crop conventions inside their wrappers; simply tracing an upstream encoder with its original crop convention is insufficient.

| Graph | Inputs | Required outputs |
| --- | --- | --- |
| Coarse | `rgb` | `identity_vertices_m [1,N,3]`, `rotation [1,3,3]`, `translation_m [1,3]`, `scale [1,1]`, `intrinsics [1,3,3]` |
| Expression | `rgb`, `identity_vertices_m [1,N,3]` | `coefficients [1,50]` for `smirk_flame50` or `[1,52]` for `arkit52`; `expressed_vertices_m [1,N,3]` |
| HRN detail | `rgb`, `rgb_hd [1,3,D,D]`, `vertices_m [1,N,3]` | `mid_offsets_m [1,N,3]`, `displacement_uv_m [1,1,H,W]`, 64–2048 pixels per axis |

`D` is the manifest's `detail_crop_size` (512 or 1024). `rgb_hd` is RGB `[0,1]` sampled directly from the original high-resolution photograph in the same canonical frame. It is not an enlarged 224px encoder crop. Both source and target HRN fits receive this input. The original image still limits recoverable detail.

Camera equation: `p_camera = scale * (p_model @ rotation.T) + translation_m`; `pixel = intrinsics @ p_camera`, followed by division by z. Intrinsics belong to the 224px input crop. Rotation must be SO(3), scale positive, all active vertices in front of the near plane. Native SMIRK's weak-perspective `cam` is **not** metric translation and must not be relabelled as such. An export must provide a calibrated/estimated pinhole conversion and record its assumptions in its validation report.

`identity_vertices_m` is expression-neutral. The expression graph must articulate the supplied identity, including jaw and eyelids, in the selected native topology. SMIRK FLAME coefficients cannot be treated as ARKit-52. An ARKit export requires actual ordered asymmetric blendshapes and coefficients in `[0,1]`; preserve anatomical left/right channels. Older SMIRK coefficients require the official FLAME 2023 Open basis conversion (and validation of the checkpoint's original FLAME version) before Open-model decoding.

HRN outputs must use metres, the same canonical axes and identity/expression frame, and the selected topology's UV chart. Convert HRN's native deformation-map representation into vertex offsets in the export wrapper. No random, zero-filled or image-gradient substitute is accepted as evidence of HRN inference.

## Hardware and validation

Reconstruction export graphs run on CPU with bounded ORT threads. The geometry rasterizer requires hardware OpenGL; it does not silently use a software renderer. The existing swapper/UltraMax provider policy applies. This conservative policy avoids adding three independent CUDA arenas on top of the neural chain; it is **not** a realtime performance claim.

| Profile | Cached real views per identity | Source photograph crop | Views sampled per frame |
| --- | --- | --- | --- |
| CPU / 6 GB | up to 12 | 512 | 2 |
| 12 GB | up to 24 | 1024 | 4 |

Optional `detail_subdivisions` is an integer `0..2` in either bundle schema and defaults to `1` at runtime load. CPU/6 GB profiles cap it at one level; 12 GB permits two when explicitly requested. Each level splits a facial triangle into four, retaining shared edges and the anatomical boundary. Capability reports distinguish `native_vertex_count`, `active_face_vertices`, and `warp_surface_vertices`. These are bounded geometry policies, not measured VRAM or realtime guarantees.

Source view selection balances angular buckets, then quality. Full-HD originals remain in their `.fsz` archive; only aligned image/depth caches are retained in memory. Preparation and failure cleanup release reconstruction resources with the Pinokio-owned session. Target coefficients currently run per frame without temporal coefficient filtering.

Run focused regressions from `app/`:

```powershell
roop_3d/.venv/Scripts/python.exe -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d roop_3d/tests/test_dense_face_backbone.py -q
```

Tests in `test_dense_face_backbone.py`, `test_native_face_geometry.py`, `test_displaced_surface.py`, and `test_dense_correspondence_visibility.py` cover both native topology sizes, irreversible semantic pruning, shared-edge refinement and UV detail missed by native vertices, analytical perspective correspondence, bilinear-footprint source occlusion, GPU-versus-CPU geometry interpolation, original-image warp coordinates, and complete neural-swapper/UltraMax routing with test doubles. They do not establish pretrained reconstruction accuracy, cross-topology registration quality, real-video expression recovery, or end-to-end 6/12 GB performance. Those require the actual bundle plus captured source/target validation sequences.

## Upstream assets and licenses

- [3DDFA-V3](https://github.com/wang-zidu/3DDFA-V3): MIT code; the [asset instructions](https://github.com/wang-zidu/3DDFA-V3/tree/main/assets) cover BFM-35709 and semantic annotations. The code license does not relicense BFM data.
- [HRN](https://github.com/youngLBW/HRN): Apache-2.0 code; [assets](https://github.com/youngLBW/HRN/blob/main/assets/README.md) and their dependencies have their own terms. Native HRN is a multi-stage CUDA/PyTorch pipeline, not the ONNX contract above.
- [SMIRK](https://github.com/georgeretsi/smirk): MIT code; the [native encoder](https://github.com/georgeretsi/smirk/blob/main/src/smirk_encoder.py) returns 50 FLAME expression coefficients plus separate jaw/eyelid parameters, not ARKit-52.
- [FLAME 2023 Open](https://flame.is.tue.mpg.de/): CC-BY-4.0 model, distinct from older research-only releases. The official site links the Open model and expression conversion code. Record attribution and the exact model/version used.

MakeHuman builders and old rendering utilities remain available for existing offline assets and historical regression fixtures. They are no longer the selected live 3D swap branch or the preferred FaceSet preview. Canonical previews explicitly report when the dense bundle is unavailable.
