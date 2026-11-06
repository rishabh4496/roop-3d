# Engine comparison guide

"Engine" is used for several different choices in Roop 3D. This guide separates them and says, for each, what is selectable on a real
installation and why something may be unavailable. Anything the interface refuses to let you select shows the reason on hover.

## 1. The swap engine (Inspector → Models & engine)

| Engine | What it is | Status on the build this was checked against (2026-11-05) |
|---|---|---|
| **Roop 3D Reference ONNX** (`reference_onnx`) | Local still-image and **video** face swap with the existing tracking, compositing and audio-preserving mux | **Selectable.** Status shown as *Licence restricted*: usable, but its models' licence records are incomplete (see [Offline models §4](OFFLINE_MODELS.md#4-model-review-why-a-verified-file-can-still-be-refused)) |
| MiniMax H3 Ref2VA (experimental video generation) | Reference-conditioned *generative* video; **not** frame-preserving identity replacement | Not selectable: *Unverified*. The current bridge returns images and has no local video path |
| BFS Qwen-Image-2.1 (experimental still/keyframe edit) | Possible still-image or keyframe correction; **not** a video-rate swapper | Not selectable: *Incompatible*. No adapter or keyframe-consistency integration exists |

The Inspector shows a **Recommended** engine with the reason. The rule is simple and never recommends an unavailable engine: the first
*selectable* engine, preferring one marked ready. Today that is the Reference ONNX engine, because it is the only one that renders your video
frame by frame and keeps your audio and timing. The selection is saved with each job.

## 2. Render mode (Inspector → 3D settings)

| Mode | What it does | Use it when | Needs |
|---|---|---|---|
| **Standard identity swap** | The proven 2D swap path | Faces mostly face the camera; fastest | A FaceSet |
| **2.5D Pose-Aware Swap** (`pose_3d`) | Projects the swap through the source identity's own 3D asset. Falls back to *standard* **per frame** when the geometry is unreliable | People turn their heads; you have three-quarter views | A FaceSet with good coverage ([FaceSet guide](FACESET_GUIDE.md)); the 3D asset is built on first use |
| **Full 3D Volumetric Re-enactment** (`reenactment_3d`) | Source identity geometry driven by the target's expression. Falls back to *standard* per frame when geometry is unreliable | Strong expressions or speech | As above |

The 3D viewer under the selector previews the geometry each mode works from: *Standard* and *2.5D Pose-Aware* open the **Estimated 2.5D** tab, *Full 3D Volumetric Re-enactment* opens **Full 3D Reconstructed**. Fallbacks are per frame and reported, so a 3D mode never leaves a gap. Estimated geometry is an estimate: it is not a scan.

## 3. Restoration model

| Option | Weights | Notes |
|---|---|---|
| Off (default `face_restoration_policy`) | none | Fastest. The swapped face is as sharp as the swapper makes it |
| **GPEN** (`gpen_bfr_512.onnx`) | must be installed and verified | Sharpens and cleans the face. Needs the *stream* render backend and registered weights |
| **CodeFormer** (`CodeFormerv0.1.onnx`) | must be installed and verified | Has a *fidelity* control (low = more restored detail, high = stays true to the input face; default 0.5, recommended 0.4–0.7). **In the current build CodeFormer is not wired into the preview compositor, and the app marks this control inactive: it currently changes nothing in the preview.** Prefer GPEN |
| GFPGAN, UltraMax | — | GFPGAN has no verified adapter; UltraMax needs an explicitly registered model and preprocessing contract |

The Inspector says *Weights verified locally* or *Weights not verified locally: restoration stays off*. Restoration never runs without verified weights.

## 4. Where it runs: hardware profile and provider

| Setting (Settings → Compute) | Values and availability |
|---|---|
| Hardware profile | `auto`, `cpu`, `6gb` (needs a CUDA GPU), `12gb` (needs a CUDA GPU with at least 10 GB of video memory) |
| Preferred provider | `auto` · `CPUExecutionProvider` · `CUDAExecutionProvider` (needs CUDA) · `TensorrtExecutionProvider` (needs the stream backend, an NVIDIA GPU and the packaged TensorRT libraries) |
| Precision policy | `auto`, `fp32`. `fp16` is disabled: the reference session uses observed FP32 graph contracts |
| Render backend | `auto`, `reference`, `stream` |

In short: **TensorRT** is the fastest and needs a one-time engine build for your GPU; **CUDA** needs no build and is the usual fallback; **CPU**
always works and is much slower. Profiles are explained in [GPU troubleshooting](GPU_TROUBLESHOOTING.md#hardware-profiles).

## 5. Other settings that look like engines

| Setting | Values |
|---|---|
| Execution mode | `swap` (default). `volumetric` needs LivePortrait weights; `hybrid` and `diffusion_vfx` have no verified loader and are disabled |
| Lip sync mode | `off`. `wav2lip` has no verified adapter and is disabled |
| Audio policy | `copy` (default) or `transcode`. `drop` is disabled: the workstation preserves source audio |
| Colour transfer | `LAB statistics` or `Stabilized histograms` |

## 6. The pipeline's building blocks (orchestrator registry)

Behind the swap engine, Roop 3D selects components per task from a registry (`GET /api/v1/orchestrator/engines?tier=cpu|6gb|12gb`), each with a
tier, a category and an *available* flag with machine-readable reasons (for example `NOT_GOVERNANCE_READY`). Categories include face detection,
face recognition, dense landmarks, identity swapping, expression transfer, occlusion segmentation, restoration, 3D reconstruction, video
decoding and video encoding. The registry reports honestly what can run on *your* machine; on the build this guide was checked against, only the
video decode/encode, 3D reconstruction (`identity_asset_builder`, `pose_blend_projector`), `reenactment_3d` and `tps_expression` components were
available, and the neural components were `NOT_GOVERNANCE_READY` (the licence/safety review described in [Offline models §4](OFFLINE_MODELS.md)).

## 7. Choosing quickly

| Your situation | Choose |
|---|---|
| First try, front-facing video | Standard identity swap, restoration off |
| People turn their heads | 3D pose, with a FaceSet that covers three-quarter views |
| Faces look soft | Turn on GPEN (needs verified weights and the stream backend); CodeFormer's control is currently inactive |
| Not enough GPU memory | Lower the hardware profile (12gb → 6gb → cpu) |
| Slow previews | Keep *Live swap* on only when you need it; see the [Performance guide](PERFORMANCE_GUIDE.md) |
