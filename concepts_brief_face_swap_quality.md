# Diagnose 3D Face-Processing Artifacts

* **Author:** rishabh4496
* **Date:** 2026-11-12

## Overview

The user reports halos, blur, and face cropping at different angles when swapping faces in a 3D workflow. The requested run uses the Harjot FaceSet and each single-face MP4 in `G:\pinokio\roop-keep\single`, with 3D expression transfer and the UltraMax enhancer. The user wants the first result saved under `G:\pinokio\api\roop-3d\app\roop_3d\outputs\test 1`, with average FPS, output location, and applied fixes reported. Stop after each video for the user's visual review and angle guidance before processing another. The first `angle.mp4` baseline is complete; later diagnosis and implementation status are summarized in Analysis below.

## Analysis

### Identify the actual processing route before changing it

The route is resolved. In the Studio Inspector, **3D Expression Transfer** sets `render_mode: reenactment_3d`; the Engine & Model Rack exposes that same parameter as **3D Full-Head Hybrid**. The default engine ID `reference_onnx` is separate from `render_mode`. The ordinary `reference_onnx` job path carries workstation parameters to the reference renderer; this is not the Stage 17 `photoreal_3d_ffmpeg` route. The selected mode requires a compatible facial model/reconstruction bundle and may fall back to the swap crop when its geometry is not engaged. Therefore, do not call an output a successful 3D expression-transfer result without checking the job manifest and capability evidence for the effective mode and geometry/fallback status.

The request is to leave every other setting at its current/default value while selecting this mode and UltraMax. Source defaults set `restoration_blend` to 0. Selecting UltraMax through the Studio Inspector sets `restorer_model: ultramax` and, when the blend is zero, activates it at 0.7. That is the UI's enhancer-activation behavior, not a separate tuning choice. Existing saved project/live settings were not inspected; do not assume they equal source defaults.

### Input inventory and proposed first file

The supplied folder currently lists 12 MP4 files. `angle.mp4` is 12.7 seconds, 1280Ã—720, 30 fps, H.264/AAC; its preview shows frontal, profile/back-facing, and frontal views. It was proposed as the first file because it is short and angle-focused; its completed baseline is recorded below.

### FaceSet and target coverage evidence

The local `harjot` FaceSet exists with five views. Its existing identity-asset manifest reports yaw coverage of about âˆ’78.8Â° to +78.9Â°, level pitch only (no up/down views), 59.4% of the facial atlas unseen, and no recovered back-of-head. The `angle.mp4` preview includes frontal, profile/back-facing, then frontal views. This mismatch is a coverage risk to check, not a proven cause by itself. Do not assume the separate identity-asset manifest is the exact object used by `reenactment_3d`; that route builds source surfaces from FaceSet views. The completed `angle.mp4` baseline below provides the first output.

### Diagnose symptoms with evidence

Halos, blur, and cropping are symptoms, not diagnoses. Inspect the selected route and compare evidence from the relevant processing stages before selecting a fix. Existing `app/diagnostics/pipeline_audit.py` tracks tensor shapes and can report enhancer bypasses; its UltraMax checks include a small-face crop bypass at 320 pixels or below. This is a diagnostic signal, not proof that the user's clips hit that condition. The Stage 17 documentation's halo cause and correction are unrelated prior evidence from another route/fixture; they do not establish a cause or fix for this selected `reenactment_3d` mode.

Reuse the existing job manifest and pipeline audit before adding new diagnostic machinery. Preserve source videos and prior outputs. Check input resolution, duration, and the configured GPU tier before a full render; one-at-a-time processing does not by itself limit peak memory. Record the effective engine, enhancer, provider and fallback, model identifiers, relevant settings, and stage timings so a later result can be reproduced. Keep any diagnostic frames local and out of ordinary logs.

### First baseline: `angle.mp4`

Job `4aa7c9e87ef140a7be00c3abd52f68ed` completed on the requested `reenactment_3d` route with `reference_onnx` and Harjot. Its manifest confirms pose projection active, identity transfer verified, one identity prepared, and four source views used per frame. UltraMax restoration ran with verified local weights; the job request used `restoration_blend=1.0`. The output is 381/381 frames, 1280Ã—720 at 30 fps. Elapsed time from job creation to completion was 447.285 seconds (0.852 end-to-end fps, including setup and mux). The output was copied to `app/roop_3d/outputs/test 1/angle_baseline.mp4`.

The requested blend differs from the Studio Inspector picker behavior of 0.7 when starting from zero, so this is not a fully matched UI-default run. No fix was applied; the user is reviewing before angle changes or processing the next clip. These data do not establish root causes.

### Output and measurement

The ordinary `reference_onnx` job writes through `Settings.output_root` (default `roop_3d/.runtime/outputs`) using the ordinary renderer's output naming and adjacent manifest. The canonical job output and manifest remain under `.runtime`; the baseline MP4 has been copied and verified at `app/roop_3d/outputs/test 1/angle_baseline.mp4`. `app/roop_3d/outputs/test 1` is outside `.runtime`; the writer rejects locations outside its owned runtime. Do not apply Stage 17's separate `<job_id>/result.mp4` contract to this route. Retain the canonical job output and manifest along with the verified copy; never overwrite an existing result or source video, and tie the exported filename and manifest to the input.

The first job did not retain first-decoded-frame/render-only timing, so 0.852 fps is an end-to-end fallback from job creation through completion, including setup and muxâ€”not pure render FPS. For future runs, preferred measure: successfully encoded output frames divided by wall-clock time from first decoded frame through output verification; report model setup separately. Also report input dimensions, frame rate and duration, decoded/encoded frame counts, hardware/provider/fallback, and available per-stage timings. Do not use the input frame rate as processing FPS. The user's own visual review remains the quality decision; no automated score replaces the stop-and-review gate.

Claude Sonnet 5.5 was used by workers in this session at high, xhigh, and medium levels per session metadata, so it is not unavailable as a tool-session model. The requested Claude Sonnet 5.5 High planning/brainstorm itself was not run by Claude; do not claim that it was. GPT-6 Luna workers ran at medium/low per metadata; no worker was confirmed at xhigh. The GPT-6 Luna xhigh session submitted the first baseline job; local `reference_onnx`/CUDA performed the rendering. One baseline throughput was measured, but no comparative benchmark occurred. The Goldfish review used GPT-6 Luna medium/high workers.

### Status update

The 0.7-blend control run showed the same artifacts as the 1.0 run, so blend level was not the cause. Measured causes: edge jitter from unfiltered per-frame 3D fit and bbox-framed crop with 128px nearest-upsampled coverage; an unreplaced target band; source-photo hair/collar texels admitted; no local shading; and a double UltraMax pass. A tracker confidence >1.0 float32 bug was fixed in `target_lock.py`. Stability techniques from Roop Ultimate have been ported to Roop 3D; a new-build render is pending measurement.

## Goals vs. Non-Goals

* **In-Scope Goals:** Identify the exact active processing route; establish the causes of the reported artifacts from stage-level evidence; apply only a supported, evidenced fix; process one input at a time with Harjot's FaceSet; report the actual output path, fixes, and measured FPS; wait for the user's visual review before continuing.
* **Out-of-Scope Non-Goals:** Batch all files before review; change angle behavior without the user's guidance; assume that the Stage 17 engine is the requested expression-transfer mode; invent causes from symptoms; overwrite source media or previous outputs; silently replace the requested model stages with other models; add a new diagnostic framework before checking existing audit and manifest data.

## Alternatives Weighed

* **Process every MP4 in one batch:** Rejected by the author because the user explicitly requires a review pause after each output.
* **Treat expression-transfer engines, volumetric render modes, and `photoreal_3d_ffmpeg` as one mode:** Rejected; they are separate concepts/routes. The exact Studio Inspector label resolves to `render_mode: reenactment_3d`, also exposed in the Engine & Model Rack under a different label; `reference_onnx` is the separate default engine ID.
* **Redirect the job's internal output directly to `app/roop_3d/outputs/test 1`:** Rejected; this destination is outside the ordinary renderer's owned `.runtime` output root. Keep its canonical output and manifest, then export a verified copy.
* **Substitute different models without notice:** Not authorized. Claude Sonnet 5.5 was used by session workers, but the requested Claude High planning/brainstorm was not run; no comparative benchmark has occurred.
* **Add new diagnostic infrastructure immediately:** Not selected; first reuse the existing job manifest and pipeline audit, then add only what a reproduced gap requires.

## Key Decisions & Staging for Next Steps

1. Use the Studio Inspector's **3D Expression Transfer** control (`render_mode: reenactment_3d`) with the default `reference_onnx` engine unless the user has saved different settings. This mode needs a compatible facial model/reconstruction bundle; verify manifest/capability evidence that geometry engaged rather than falling back to a swap crop.
2. Keep all other settings at their current/default values. UltraMax selection activates a zero restoration blend to 0.7 in the Studio Inspector; this is its UI selection behavior. No saved project/live settings were inspected.
3. Preserve the input MP4s and existing outputs. Establish evidence for the first file before applying a fix; do not process another file yet.
4. Produce one full first-file output after the selected fix, report the exact destination, applied change, frame counts, and the agreed FPS measurement, then stop for the user's review and angle guidance.
5. Keep the ordinary renderer's canonical job output and adjacent manifest under `Settings.output_root` (default `roop_3d/.runtime/outputs`). The baseline MP4 has already been copied and verified at `app/roop_3d/outputs/test 1/angle_baseline.mp4`.
6. Do not claim a Claude plan or comparative benchmark. The GPT-6 Luna xhigh session submitted the baseline job; local `reference_onnx`/CUDA rendered it.
7. `angle.mp4` is the first processed baseline and is awaiting the user's review. Preserve the user-confirmed pause; do not proceed to another file until instructed. Output filename/collision behavior and model fallback remain to be checked.
