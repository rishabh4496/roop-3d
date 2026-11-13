# Engine & Model Rack

Where: **Face Swap** inspector (step 4), **Batch Matrix** (above the strategy panels), and **Settings → Models & Engines Manager**.

Every registered model is listed in every selector. Nothing is filtered by `installed` or by registry state; a model the engine cannot run is shown disabled with its reason.

## Status pills

Derived from `GET /api/v1/models` (`modelStatus` in `src/stores/modelCatalog.ts`):

| Pill | Meaning |
| --- | --- |
| Ready (green dot) | on disk and `VERIFIED_LOCAL` |
| TRT FP16 cached (blue bolt) | on disk and a TensorRT engine is cached |
| On disk (amber) | on disk, not yet verified against the registry |
| Click to download (cloud) | not installed; the registry lists a source URL |
| Locate local file (folder) | not installed and no source is registered |
| Blocked | the registry refuses the artifact |

## Choosing a model that is not installed

Selecting it opens an inline panel under the selector (no modal):

* **Download & activate**: only when a source exists. Progress bar, cancel, licence checkbox when the registry requires acceptance. When the file lands the selector activates the model by itself. A refused download shows the server's reason and the locate option remains.
* **Locate local .onnx**: a folder or `.zip` path, posted to `/installer/models/import`. The server installs only files whose SHA-256 and size match a registry record, and the panel reports a mismatch instead of loading the file.

## What each control really does

| Control | Wiring |
| --- | --- |
| Face swapper model | **Live.** Sent as the job's `swapper_model_id` (previews and renders). Persisted in `rackSelection` (`swapper`); `null` follows `Settings.swapper_model_id`. |
| 3D pipeline mode | **Live.** `workstation_parameters.render_mode`. *3D Facial Warp* (`pose_3d`) projects the swap through the source FaceSet's stored identity asset; *3D Expression Transfer* (`reenactment_3d`) renders that asset with the target's measured expression. Both reuse the canonical asset built under the FaceSet's **3D identity asset** (they never build it); without a matching asset Facial Warp uses the generic canonical head proxy and Expression Transfer keeps the plain 2D swap. FullHead synthesis is not offered. |
| Enhancer, blend, fidelity | **Live.** `restorer_model`, `restoration_blend`, `fidelity_weight`. Fidelity exists for CodeFormer and UltraMax only: the engine rejects it for GPEN / GFPGAN, so requests leave it out there. **UltraMax** is the registered `codeformer.fp16` artifact followed by Laplacian detail fusion (not a GPEN / Real-ESRGAN cascade); its readiness is the catalog row `ultramax`, the same check `/system/capabilities` makes. GPEN 1024, RestoreFormer++ and every other registered restorer artifact are listed but disabled: the engine has no adapter for them. |
| Loaded in VRAM | The enhancer is read back by registry hash id, so the loaded weight **file** is matched to the selected restorer (`codeformer.fp16.onnx` = UltraMax, `CodeFormerv0.1.onnx` = CodeFormer). |
| XSeg occluder, face box feather | **Live.** `occlusion_mode`, `feather_px`. The pills show the networks Robust mode really loads (`xseg_3.onnx`, `face_occluder.onnx`); BiSeNet is always on. |
| XSeg occlusion strength, seam blending, pyramid bands | **Live** (Face Swap step 2; Batch rack). `xseg_occlusion_strength` (0..1, Robust occlusion only) → `CompositeOptions.xseg_occlusion_strength` → `OcclusionAnalyzer.analyze`; `blend_mode` (`multi_band` default, `feather`, `poisson`, `frequency_separated`) and `pyramid_levels` (4..6, banded fusions only) → `CompositeOptions.mode` / `.pyramid_levels`. A control the selection cannot act on is disabled with the reason and left out of requests; `frequency_separated` cannot be combined with Robust occlusion (the engine rejects the pair). The Photoreal engine ignores `blend_mode` and follows `pyramid_levels`. |
| Provider / precision | **Live, global.** Saved through `PUT /config` (`preferred_provider`, `precision_policy`). Combinations the engine reports unavailable (from `/system/capabilities`) are disabled with the reason. |
| Rebuild TRT engine | Calls `POST /models/compile_trt`, which currently answers 409 (no verified build adapter). The message is shown as-is. |
| Detector | **Live.** `detector_model_id`, saved globally. Landmarks (5-point detector, 68- and 106-point models) are measured by the engine and are not selectable. |
| Frame upscaler | **Not supported.** The engine has no frame-upscaler stage: Real-ESRGAN / 4x-UltraSharp are listed disabled with that reason, and nothing is remembered. |
| 3D landmarker, cranial silhouette, per-person swapper / enhancer | **Removed.** They had no engine setting and were only remembered. The silhouette is controlled by the swap scope (step 1); a job carries one swapper and one enhancer. |
| Batch | The rack's swapper and any field you touch are merged into every queued job (`rackStore.batchParameters`); untouched recipe values are kept. Recipe parameters are real `WorkstationParameters` fields. |

## Settings → Models & Engines Manager

Rescan `app/models`, import a local folder/`.zip`, verify installed models, and a table of every registered model with per-row actions.

* Preload / Unload to VRAM are shown **disabled**: there is no manual VRAM API; models load on first use.
* **SHA-256 verification cannot be turned off.** There is no "force unblock": the model provider opens only hash-pinned registry locators, and the error guide documents that no setting skips the check. To use a weight that fails it, add it to the registry with its hash and licence review.

## Registry duplicates

The registry can hold several records for one filename. They can be *different byte builds* (for example the engine's pinned `hyperswap_1a_256` has a different SHA-256 from the public `hyperswap_1a_256.onnx`). Selectors list each distinct build once, labelled with its SHA prefix, under "Other registered … artifacts"; those rows are not selectable as swappers because the engine runs only the curated ids.
