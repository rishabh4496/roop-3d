# Stage 17 - photorealistic 3D swap core and the FFmpeg end-to-end video job

Status: built, **not committed**. Engine id `photoreal_3d_ffmpeg` (opt-in per job; the default engine is unchanged).

## Path mapping (the prompt vs this repository)

| The prompt says | What is real |
|---|---|
| `app/roop_3d/pipeline/photorealistic_swap_core.py`, `video_job_runner.py` | created as named (`roop_3d.pipeline` was already listed in `pyproject.toml`) |
| `POST /api/v1/facesets/import`, `POST /api/v1/media/upload` | did not exist; added as one-call aliases of the validated chunked protocol (`/assets/uploads`) |
| `GET /api/v1/jobs/{id}/download` | did not exist (`/jobs/{id}/artifacts/video` did); added, SHA-256 verified, HTTP Range |
| `status === "completed"` | the job state is `succeeded` (`COMPLETED` is only the queue's display state) |
| `roop_3d/tests/test_e2e_video_pipeline.py` | `app/roop_3d/tests/test_e2e_video_pipeline.py` |
| `SilhouetteDeltaInpainter`, `PerimeterCageWarper` | already built in Stage 6B (`rendering/`), but only ever ran for `full_head` replacements, which the live engine disables; the core now drives them from landmark-derived outlines |
| `outputs/<job_id>/result.mp4` | `<settings.output_root>/<job_id>/result.mp4`, i.e. `roop_3d/.runtime/outputs/<job_id>/result.mp4` (the job contract requires paths under `roop_3d/.runtime/`) |
| "GPEN Realistic / UltraMax" | UltraMax (CodeFormer-based, as the repo defines it) by default; `restorer_model` selects GPEN / CodeFormer / GFPGAN |
| the pasted snippet | not copied: it encodes with `-shortest` (can drop the last frame when the audio is a few ms shorter) and the default RGB->YUV matrix (BT.601: shifts colour on HD), and it loses frames' colour tags |

## What it does (per face, `Photorealistic3DSwapCore.process_face`)

The face the viewer sees is always photographic: neural swap pixels + the FaceSet's own photograph + a restorer. A shaded mesh is never
composited. Geometry decides where pixels land; illumination geometry multiplies them by a smooth ratio.

1. **Geometry** - target head outline vs. source-locked outline (`swap_scope`: `full_craniofacial` locks 100 %, `adaptive_skull` blends by
   `skull_blend`, `inner_face` keeps the target outline). `SilhouetteReplacer` fills the overhang with background (temporal donors first,
   multi-scale Navier-Stokes after) and the perimeter cage warps parser-derived hair / ears / neck onto the new outline.
2. **Rectify** - the 128 px swap is warped by a thin-plate field that moves only the jaw contour (eyes / brows / nose / mouth pinned, ROI
   border pinned, fold-free by bisection); the FaceSet photograph nearest in pose fills the forehead / cheeks / jaw the swap cannot reach,
   colour-matched along the seam. Nothing is written outside the locked outline (the feather ramps up inside it).
3. **Relight** - `R_sh = Render_SH(N, L_target) / (Render_SH(N, L_source) + 1e-4)` per channel in linear light, smoothed (it cannot carry
   pore detail) and clipped, applied at `relight_strength`. Then a TRUST-style skin-tone gain: albedo vs. illumination decomposition
   (`UnbiasedSkinToneCalibrator`), chromatic only (luminance is the quotient image's job), bounded to 1/1.15..1.15 and scaled by the
   illuminant estimate's confidence (a guessed illuminant must not move skin tone).
4. **Restore and composite** - restorer on the aligned face ROI (blend 0.85, eyes/lips protect map when available), FaceSet fine detail
   re-injected only where the photographic layer is softer than the photograph and never beyond it, camera focus matched (softens only),
   grain matched (seeded, capped at 4 levels), front-occluder-aware alpha, then a 4-band Laplacian pyramid composite onto the step-1 plate.

`session_adapter.py` supplies the real inputs per frame: detection / tracking / identity gate / `inswapper` from `ReferencePipeline`,
the nearest-pose FaceSet photograph with its 106 landmarks (stored, else measured once), and BiSeNet + XSeg + face-occluder masks from a
full-head 512 px crop (`landmark_crop(..., expansion=2.3)`).

## The video job (`EndToEndVideoJobRunner`)

* Inherits input validation from `MasterPipelineRunner.prepare` (inspection, FaceSet mapping, identity lock, model digests) and replaces
  only the render loop, so the job keeps its durable lifecycle, idempotency, `PinokioTerminalHUD` frame bar and `/ws/v1/pipeline/status`.
* `ffprobe` probe -> `ffmpeg` decode pipe (`rawvideo bgr24`, the source's own colour matrix) -> per-frame swap -> `ffmpeg` encode pipe
  (`h264_nvenc` `-cq 16`, automatic `libx264` `-crf 16` fallback with frame replay, BT.709 limited range with matching tags) -> audio
  stream-copy (`-c:a copy`; AAC re-encode only when the MP4 container refuses the codec, and then reported as `transcode`).
* Writes `result.mp4` + `manifest.json` (what actually ran: encoder and fallback reason, restorer, per-frame mean stage times, faces
  swapped / skipped / withheld, fallbacks tally, model hashes, limits) and verifies the file by packet count before publishing.
* Children are registered with the orphan-process registry; cancel / failure removes partial files.

## Limits (also in every manifest - do not read more into the numbers than these allow)

* **Not resumable**: a paused / interrupted job restarts at frame 0 (no PNG frame store is written).
* **The skull is a 2-D projection**, not a per-frame fitted 3DMM: the FaceSet view's 106 landmarks aligned to the target by a similarity on
  the eyes / nose / mouth, plus a forehead Bezier (the 106 layout stops at the brows; the same heuristic is applied to both heads so its
  bias cancels in the difference). The MakeHuman / FLAME mesh bridge is not wired into this path.
* **SH normals are a silhouette ellipsoid** unless `FacePacket.normals` is supplied (no provider is wired).
* **Inpainting is plausible continuation**, not recovery: measured on synthetic scenes the filled band is within ~5 grey levels of the true
  background; no real-footage evaluation exists.
* SDR only (PQ / HLG -> `HDR_INPUT_UNSUPPORTED`); CFR output at the source's average rate (`vfr_source` is flagged); no source metadata or
  subtitle tracks in the output; odd frame sizes are padded to even (`padded_to_even`).
* One-shot upload routes are bounded by `ROOP3D_MAX_BODY_BYTES` (2 MiB default, 64 MiB max); the middleware rejects larger bodies and that
  security cap was deliberately not loosened. Larger files use `/assets/uploads` (what the UI does).
* Restoration is on by default for this engine (`auto` -> UltraMax even though the saved global policy defaults to `off`); disable it with
  `restorer` in the parameters' `bypass`.

## UI

`components/output/OutputComparePlayer.tsx`: when a job FINISHES in the session (jobs already finished at page load stay quiet) a
"Before / After" dialog loads `/api/v1/media/content?path=<target>` beside `/api/v1/jobs/{id}/download`. The result video is the sync
master (it has the audio): play / pause / seek / rate mirror onto the original, drift beyond 0.12 s is corrected on `timeupdate`.
Wipe divider (keyboard accessible) or side by side; "Download MP4". The engine appears in the Engine rack (`models/hub.py`).

## Bugs found by looking at real output (the numeric tests had passed)

* distance-transform normals have a medial-axis crease -> a visible seam after relight (now an ellipse-fit height field);
* an alpha feather that straddles the silhouette leaves a halo (now ramps up inside it);
* a gray-world illuminant guessed from a head ROI pinned the TRUST gain at its clamp (now chromatic-only, confidence-scaled);
* `tps_flow` had a half-pixel error (coarse grid must be corner-aligned and upsampled with `remap`);
* the saved restoration policy defaulted to `off`, so the engine's restorer silently never ran (found by the manifest assertion);
* the Before / After dialog opened for jobs that were already finished at page load (the baseline was taken on the empty initial list);
* on the real fixture a **green fringe along the hair** and a streak on the forehead: the stock Stage 6B fill excludes hair from its
  donors, so with hair behind the jaw it reached for the far backdrop, and the heuristic forehead outline counted hair as "old skin".
  Now silhouette editing is limited to below the brow line and `HairDonorReplacer` (a subclass; Stage 6B is untouched) lets hair donate;
* **blue colour specks** at the left jaw: located by dumping every stage (none of them had it) to the 4-band composite, whose per-band
  mixing overshoots at hard edges; the blend now bounds each channel by the local min / max of its two inputs (exact at alpha 0 / 1).
  Also, the restorer's influence now fades out over the last 7 % of its aligned crop (restorers are unreliable on border pixels).
  `test_no_colour_fringes_or_specks_were_introduced` guards all three on the real render.

## Measured (this machine, RTX 4070; read these as a floor, not a benchmark)

* CPU provider, 30 frames 512x512 + audio, full chain incl. UltraMax and parsers: ~6.5 min for the whole e2e test (model loads included).
* CUDA provider (not TensorRT), same fixture, run while another heavy job was using the CPU and including cold model load: 30 frames in
  66 s wall (0.46 fps), `h264_nvenc` used with no fallback; mean per frame ms: detection 428, inswapper 547, pose/TPS 159,
  mask/relight/blend 398, restorer 226. ORT warned that some Conv ops fall back from cuDNN (an existing environment matter).
  **TensorRT mixed precision was not exercised**, so no claim is made about it.
* Visual verdict on real faces: photographic, identity moves to the source, hairline and hair preserved. Skin is visibly smoother and a little
  warmer than the original photograph: a 128 px swap plus a restorer cannot reproduce every pore, and the detail guard only bounds it.

## Tests and how to run them

From `app/` (never two pytest sessions at once: `--basetemp` is shared):

```
roop_3d/.venv/Scripts/python.exe -B -m pytest roop_3d/tests/test_photorealistic_swap_core.py roop_3d/tests/test_video_job_runner.py -q   # ~15 s, no weights
roop_3d/.venv/Scripts/python.exe -B -m pytest roop_3d/tests/test_e2e_video_pipeline.py -q                                               # real models, CPU, minutes
```

UI (`app/roop_3d_ui`, Node via `G:\pinokio\bin\miniforge`): `npx vitest run src/outputComparePlayer.test.tsx`,
`npx playwright test e2e/outputCompare.e2e.ts` (real Chrome, real mp4s, real Range requests), `npm run build`.

The e2e generates a `.fsz` (re-exported from a consented fixture under a new id) and a 30-frame 25 fps clip with audio, sends both through
the new routes, runs the engine, downloads the result and checks, with ffprobe, frame count / rate / size / audio (stream-copy byte-equal),
with SSIM that the background is preserved and the face changed, with ArcFace that identity moved from the target to the source, a
fine-detail guard against plastic skin, the HUD text and the WebSocket frames. It also leaves `roop_3d/.runtime/e2e/photoreal/before_after.png`
and `result.mp4` for a human to look at.
