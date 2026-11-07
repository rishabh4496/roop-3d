# Defect #1 — enhancers produced low-resolution swaps

Selecting UltraMax, GPEN Realistic, CodeFormer or GFPGAN gave the same soft result as no enhancer. The prompt's paths
(`app/api.py`, `react-ui/`) do not exist; the real layout is `app/roop_3d/` (see `STAGE-N doc` mapping in the earlier stages).

## Root causes (all verified in code, not assumed)

| # | Where | Defect |
|---|---|---|
| 1 | `compositor/restoration.py` (`restore_aligned`, `GpenRestorer.restore`), `compositor/stream_restoration.py` (`DecoupledRestorer._restore`) | The 512 px enhanced face was warped **back into the 128 px swap crop** and only then composited, so ~15/16 of the enhancer's pixels were discarded before Stage 3. |
| 2 | `vectorized_pipeline.py` (`composite_cpu`, `affine_graph`), `gpu_compositor.py` | The composite stage was hard-wired to 128 px crops (graph, mask, matrix). The resident GPU path capped the enhanced tensor at the 512 crop and used a bilinear warp. |
| 3 | `CompositeOptions`, `WorkstationParameters`, `Settings`, `RenderPolicy`, `DecoupledRestorer` | UltraMax **defaulted to a `<=320 px` bypass** (`bypass` / `ultramax_native_bypass=True`). In the reference compositor the "native size" was the swap *crop's footprint* (`crop x |inverse|`), not the face's destination box, and any face at or under it was returned unenhanced. (On the three real frames measured below the footprint was above 320 px, so the bypass did not trigger there; the code path is real but was not the cause of those numbers.) `server/diagnostics.py` even told users to switch the bypass on. |
| 4 | `engine/pipeline_runner.py`, `server/workstation_preview.py` | `face_restoration_policy` `ultramax` and `gfpgan` were silently remapped to GPEN. GFPGAN had no adapter at all. |
| 5 | `enhancers/{ultramax,gpen_realistic,codeformer}.py` | The input was fed **BGR** into models trained on RGB (channel-order bug) and, with no session, a fake OpenCV filter was returned as if it were an enhancement. |
| 6 | `compositor/pipeline.py::_enhance` | Colour statistics came from the whole face alpha (eyes, brows, lips, hair) rather than the skin class. |

There was no literal catch-all `except` returning the unenhanced tensor; the silent fallbacks were #3, #4 and the fake
filter in #5. Restorer failures are now logged at `CRITICAL` and raised (`RESTORER_FAILED`, `ENHANCER_INFERENCE_FAILED`,
`ENHANCER_OUTPUT_INVALID`, `ENHANCER_SESSION_MISSING`), never swallowed.

## Fix: `roop_3d/hires_handoff.py`

```
S      = enhancer_res / swap_res                        (e.g. 512 / 128 = 4, UltraMax 1024 / 128 = 8)
frame -> hr crop : scale_matrix_output(M_fwd, S)        basis x S, translation x S + (S-1)/2
hr crop -> frame : scale_matrix_input (M_inv, S)        basis / S, translation - A (S-1)/(2S)
```

The `(S-1)/2` term is the pixel-centre correction: with it the re-based matrix reproduces `cv2.resize` registration
(error 0.0022 vs 0.0094 for plain scaling, `test_rebased_matrices_are_centre_aligned_and_invertible`).

* **Warp** — `bicubic_warp` = `grid_sample(mode='bicubic', align_corners=False)` straight into the **frame ROI** (not the
  full frame): `warp_to_frame_roi`. A face drawn much smaller than its enhanced tensor is first area-filtered
  (antialiased bicubic) so the gather does not alias; the CPU twin is `fit_hires_to_frame`.
* **Colour** — `match_skin_colour`: LAB / YCbCr mean-std and LAB / YCbCr weighted-histogram transfer, GPU-resident and
  sync-free, statistics from the skin class only, empty class = exact identity. Computed against the *native* frame pixels
  in the ROI. The reference (CPU) compositor now also derives its LAB statistics from the parser's skin class.
* **Blend** — `multiband_blend`: Laplacian pyramid at ROI resolution. Where alpha is 1 the finest bands are the enhancer's
  (verified: <2 % high-pass deviation); exact zero alpha stays the target byte for byte.
* **Stage 2 handoff** — the swap is only ever *enlarged* (bicubic) for the enhancer and never pasted or shrunk first.

Wired into: `GpuCompositor` (resident CUDA), `DecoupledRestorer` (host + `cuda_adapter`, native resolution out, dynamic
`fidelity_weight` / `enhancer_blend`), `composite_cpu` and the legacy ONNX GridSample graph (`affine_graph(crop_size=)`),
and the reference `FrameCompositor` (standard and robust paths: `restore_hires`).

## Behaviour changes

* UltraMax bypass is **opt-in everywhere** (`ultramax_small_face_mode='off'`, `ultramax_native_bypass=False`). When the
  user enables it, it is judged on the destination box in the target frame (`destination_face_side`).
* `diagnostics` lint `ultramax_crop_bypass` now warns when the bypass is *enabled*.
* GFPGAN v1.4 has its own adapter (`GfpganRestorer`, spec `gfpgan`: FFHQ-512 facexlib alignment, RGB, [-1, 1]) in the
  reference pipeline, previews, live preview and `face_restoration_policy`. Not added to the orchestrator engine manifest.
* UltraMax fusion at hi-res takes the `log2(S)` finest Laplacian bands 100 % from the restorer (they lie above the swap's own
  Nyquist limit); the previous 50/50 mix halved the restorer's detail there.
* New settings: `enhancer_blend`, `restorer_fidelity` (read per frame, no session rebuild).

## Measured (real inference, local video 03de3642, real `inswapper_128`, real SCRFD/BiSeNet, CPU provider)

Laplacian variance of the swapped-face interior relative to the unenhanced swap; before = old default path.

| frame | engine | before | after |
|---|---|---|---|
| 1200 | GPEN Realistic | 2.2x | **6.4x** |
| 1200 | CodeFormer | 1.5x | 3.1x |
| 1200 | GFPGAN | 1.2x | 3.0x |
| 1200 | UltraMax | 1.5x | 2.97x |
| 3000 | GPEN Realistic | 1.9x | **4.0x** |
| 3000 | UltraMax | 1.5x | 2.3x |

**The 3x target is met by GPEN Realistic on every measured frame but not by UltraMax on real footage (2.0-3.0x).**
UltraMax deliberately keeps the swap's colour and low/mid bands; with the compositor's default fidelity (0.5) it lands
at CodeFormer parity. `verify_integration.py` therefore asserts the hard 3x only on fixture engines (plumbing) and
*reports* the real-model gain; `--real-models --require-3x` turns a miss into a failure.

## Verification

* `python verify_integration.py` — adds `enhancer_laplacian_detail_3x` (fixture engines through the production adapters,
  FrameCompositor and handoff; asserts >=3x vs unenhanced, >=2x vs the crop-sized path, distinct outputs, 512 px restorer
  output, no default bypass).
* `python verify_integration.py --real-models [--require-3x]` — same measurement with the installed weights.
* `pytest roop_3d/tests/test_hires_handoff.py` — geometry, bicubic/ROI warp, colour, multiband, CPU compositor
  (standard + robust, 7.7x / 7.9x vs the shrunk path), CUDA adapter geometric exactness at 512 and 1024 with an identity
  model, resident GPU chain with a 1024 restorer, bypass defaults, loud failures.

## Known limits

* `RobustOptions.sharpness_matching` (default on) blends the enhanced face's sharpness toward the target's measured
  sharpness in the robust / full-head path; it can attenuate enhancer detail by design.
* The legacy ONNX GridSample compose graph uses bilinear sampling (ORT CUDA bicubic GridSample not relied on); the resident
  torch compositor is the bicubic path.
* The CPU reference compositor defaults to feather blending; set `CompositeOptions.mode=MULTI_BAND` for pyramid blending
  (the GPU path uses multi-band when `laplacian_blend_weight > 0`).
* The UI restorer picker does not list GFPGAN yet (the API accepts `restorer_model: "gfpgan"`).
