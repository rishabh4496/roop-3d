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
| 3D pipeline mode | **Live.** `workstation_parameters.render_mode` (`reenactment_3d` / `pose_3d` + relight / `standard`). |
| Enhancer, blend, fidelity | **Live.** `restorer_model`, `restoration_blend`, `fidelity_weight`. GPEN 1024 and RestoreFormer++ are listed but disabled: the engine has no adapter for them. |
| XSeg occluder, face box feather | **Live.** `occlusion_mode`, `feather_px`. BiSeNet is always on. |
| Provider / precision | **Live, global.** Saved through `PUT /config` (`preferred_provider`, `precision_policy`). Combinations the engine reports unavailable (from `/system/capabilities`) are disabled with the reason. |
| Rebuild TRT engine | Calls `POST /models/compile_trt`, which currently answers 409 (no verified build adapter). The message is shown as-is. |
| Frame upscaler, detector, 3D landmarker, cranial silhouette | **Saved preference.** The render engine has no setting for these yet, so the choice is remembered (localStorage `roop3d-model-rack`) and labelled `SAVED PREFERENCE`. It does not change output. |
| Per-person swapper / enhancer | **Saved preference.** Right-click a face → *Swapper & enhancer for this person…*, or the rack's *Per-person models*. The job schema carries one swapper and one enhancer per job. |
| Batch | The rack's swapper and any field you touch are merged into every queued job (`rackStore.batchParameters`); untouched recipe values are kept. |

## Settings → Models & Engines Manager

Rescan `app/models`, import a local folder/`.zip`, verify installed models, and a table of every registered model with per-row actions.

* Preload / Unload to VRAM are shown **disabled**: there is no manual VRAM API; models load on first use.
* **SHA-256 verification cannot be turned off.** There is no "force unblock": the model provider opens only hash-pinned registry locators, and the error guide documents that no setting skips the check. To use a weight that fails it, add it to the registry with its hash and licence review.

## Registry duplicates

The registry can hold several records for one filename. They can be *different byte builds* (for example the engine's pinned `hyperswap_1a_256` has a different SHA-256 from the public `hyperswap_1a_256.onnx`). Selectors list each distinct build once, labelled with its SHA prefix, under "Other registered … artifacts"; those rows are not selectable as swappers because the engine runs only the curated ids.
