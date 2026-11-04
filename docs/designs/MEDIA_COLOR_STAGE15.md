# Stage 15: Professional Media and Color-Management Engine

Status: implemented on the existing PyAV / FFmpeg architecture. The encode and decode paths were extended in place
(`media/video_reader.py`, `media/video_writer.py`, `media/probe.py`); nothing was replaced.
Code: `app/roop_3d/color/` (new), `app/roop_3d/media/` · API: `app/roop_3d/server/professional_media.py` ·
Evidence: `app/roop_3d/audit/MEDIA_COLOR_STAGE15_VALIDATION.json` (regenerate with `audit/stage15_media_validation.py`).
Vocabulary: `docs/GLOSSARY.md`.

> Numbering note: older documents named `STAGE15` in this repository (`audit/STAGE15_*`, `docs/REMEDIATION_STAGE15.md`,
> `test_stage15_e2e_golden`) belong to an earlier numbering series and are unrelated. Nothing there was changed.

## Path mapping

The prompt named no paths. Media handling lives in `app/roop_3d/media/`; the new pure colour science is a sibling
package `app/roop_3d/color/` (no file or network access); tests are `app/roop_3d/tests/test_stage15_*.py`.
No Pinokio launcher script was touched (this is application code, not launcher work).

## The headline: what was actually wrong

Measured with an independent float oracle (`color/gamut.py::rgb_to_ycbcr`, which shares no code with swscale or
libavfilter) before changing anything:

| Defect | Evidence (8-bit code values, worst stripe) | Fix |
|---|---|---|
| **BT.2020 SDR encoded with the BT.601 matrix but tagged BT.2020.** PyAV's `Colorspace` enum has no BT.2020, so `reformat(dst_colorspace=...)` could not request it. | 33 / 255 on green | RGB to YCbCr through a libavfilter `scale` graph (`out_color_matrix=bt2020`), verified to 1 LSB at 8-bit (`media/convert.py`). |
| **PQ and HLG sources were decoded as if SDR and written as 8-bit `yuv420p` tagged PQ/HLG.** A false HDR claim: 8-bit PQ is not valid HDR10, and blending happened on PQ code values. | n/a (invalid output) | Explicit policy: tone-map to SDR BT.709 (default), refuse, or, for *unmodified* media only, pass through. |
| **Untagged HD decoded with swscale's BT.601 default.** | 23 / 255 | Matrix resolved by documented convention, applied explicitly, and recorded as an assumption. |
| **10-bit sources silently reduced to 8-bit** with nothing recorded. | n/a | Manifest states processing and container bit depth separately. |
| **The default GPU "stream" backend dropped all colour tags and used FFmpeg's untagged BT.601 default** when encoding. | **41 / 255 across the whole frame** on tagged BT.709 HD | Encoder takes the source colour: explicit matrix, range and all four tags. Material the chain was not validated for goes to the reference pipeline. |
| **Transcoding audio with an unlabelled channel layout failed** (`"1 channels"` for PCM in Matroska/AVI/WAV). Found by a new test, not by this stage's brief. | export aborted | Map an unlabelled layout to the standard one for its channel count (`media/audio.py`). |

After the fix: worst-stripe round-trip error is 5 to 13 / 255 for BT.709, BT.601 and BT.2020 in limited and full range (lossy
4:2:0 coding of fully saturated primaries; mean error is about 3, against about 12 for the old BT.2020 bug) with every tag preserved; `audit/MEDIA_COLOR_STAGE15_VALIDATION.json` holds the table.

## Colour model (`color/`)

* `codes.py`: H.273 / FFmpeg code points, and what counts as "unspecified" (0 and 2).
* `description.py`: `ColorDescription` (primaries, transfer, matrix, range, bit depth, chroma, dynamic range, HDR static
  metadata). Every field is either `tagged` or `assumed(<rule>)`; `assumed_fields` is surfaced in the probe, the manifest
  and `POST /api/v1/media/validate`. **Untagged material is never classified as HDR.**
* `WorkingColor` (also in `description.py`) is the contract between decode and encode: what the 8-bit RGB frames handed
  to a transform *mean*, and how the output must be tagged. For SDR it equals the source (the pipeline is
  colour-transparent: no gamut or transfer conversion); for HDR under `tonemap_sdr` it is BT.709 SDR and names the operator.
* `transfer.py`: PQ, HLG (+ BT.2100 OOTF), BT.709/BT.2020 OETF, sRGB, power laws. `gamut.py`: primaries matrices
  **derived from chromaticities**, YCbCr coefficients, a float reference RGB/YCbCr implementation. `hdr.py`: tone mapping.
* Convention for untagged material (matches common players): width > 1024 or height > 576 is BT.709; 625-line SD is
  BT.470BG; 525-line SD is SMPTE 170M; primaries and transfer follow the matrix; range is limited unless the pixel
  format says otherwise (`yuvj*`, RGB).

Validated against published values, not the implementation: PQ(203 cd/m2)=0.5807, PQ(1000)=0.7518, HLG OETF(1/12)=0.5,
the HLG OOTF puts 75 % signal at 203.15 cd/m2 on a 1000-nit display (BT.2408), and the derived BT.2020 to BT.709 matrix
equals BT.2087's. DCI-P3 (code 11) is refused: its white point needs chromatic adaptation that is not guessed.

## HDR

| Capability | State | Detail |
|---|---|---|
| Detect PQ / HLG, read MaxCLL / MaxFALL / mastering display | verified | `media/probe.py`; metadata parsed from frame side data. |
| HDR input, rendered output | **tone-mapped to SDR, labelled as such** | Face models and the compositor are 8-bit display-referred; the manifest says `hdr_output: NOT_PRODUCED_TONE_MAPPED_TO_SDR`. |
| HDR input, *unmodified* media (no transform) | verified passthrough | Planes re-encoded without an RGB round trip. `lossless_intermediate`: Y planes bit-exact. `color_preserving`: HEVC Main10 with MaxCLL and mastering display carried and checked. FFV1 cannot carry those SEI messages; the manifest says so. |
| HDR rendered output | **UNSUPPORTED** | Not offered. `ExportOptions.hdr_policy` accepts `tonemap_sdr` or `reject` only; `passthrough` is refused with a transform. |

Tone-mapping operator (`bt2390_eetf_v1`, deterministic, parameters recorded and hashed in the manifest): decode to absolute
light (PQ, or HLG with the reference OOTF); scale 203 to 100 cd/m2 so HDR reference white lands on SDR white and midtones
keep their exposure (HDR 18 % grey, 36.5 cd/m2, lands on SDR 0.18 exactly); BT.2390 EETF highlight roll-off; BT.2020 to
BT.709 in linear light with clipping; inverse BT.1886 encoding. Source peak comes from the mastering metadata, else 1000
cd/m2 (stated). Diffuse white (203 cd/m2) lands at 0.78 linear (about 90 % signal, the broadcast diffuse-white level) for a
1000-nit source and lower for a brighter master. The highlight compression is per-channel, so very bright saturated
highlights desaturate; that is the documented price of a stateless, reproducible, LUT-fast operator.

## Bit depth

| Stage | State |
|---|---|
| Decode 8 / 10-bit (12 / 16 by the same path) | verified (8, 10) |
| Face detection, swapping, restoration, compositing | **8-bit; high-bit-depth processing is UNSUPPORTED** |
| Encode 8 / 10-bit | verified |
| Unmodified (native) path | planes carry the source depth |

A 10-bit container (`archival_hevc10`, ProRes) removes *encode-stage* quantisation; it does not recover precision the
8-bit pipeline discarded, and the manifest records `processing_bit_depth: 8` next to `output_bit_depth: 10`.

## Linear-light compositing

`WorkstationParameters.linear_light` (default off) with `blend_transfer` = `auto | srgb | bt1886 | gamma22 | gamma28 |
bt709_oetf | linear`. `auto` follows the media's actual transfer (sRGB-tagged to sRGB; BT.709/601/2020 to BT.1886;
linear stays linear), resolved once per job from the reader's working colour and identically in previews.

What existed: a linear-light blend hard-wired to the **sRGB** curve, feather mode only, inside the occlusion-robust path,
not user-selectable. What it is now: the curve is a parameter (default sRGB, so existing output is unchanged), and it
applies to every blend mode: feather mixes in linear light; multi-band builds and merges its pyramids in linear light;
Poisson solves its gradient field on the encoded image (gradient-domain cloning is perceptual) and does the final alpha
mix in linear light. Alpha 0 and 1 return the inputs exactly for every code value and every transfer.
**Not** linear: colour matching, restoration, relighting. Those stay in the encoded domain.

## Export profiles (`media/export_profiles.py`)

| Profile | Container | Video | Notes |
|---|---|---|---|
| `standard_mp4` | `.mp4` | H.264 8-bit 4:2:0, preset fast | The pre-Stage-15 encoder settings; a test asserts the legacy call and this profile emit identical packets. NVENC allowed. |
| `high_quality_mp4` | `.mp4` | H.264 High, slow, CRF 16, fast-start | |
| `archival_hevc10` | `.mkv` | HEVC Main10, CRF 10, slow | Keeps every audio and subtitle stream; HDR passthrough capable. |
| `mov_workflow` | `.mov` | ProRes 422 HQ, 10-bit 4:2:2 | For editors. |
| `lossless_intermediate` | `.mkv` | FFV1 v3 | Rendered frames stored as RGB: decode equals the encoder input exactly (tested). Untouched media in its own planes. |
| `image_sequence` | directory | PNG / TIFF / DPX, 8-bit | `sequence.json` holds exact rational PTS, per-frame SHA-256, colour description; source audio stream-copied to `audio.mka`. |
| `color_preserving` | `.mov` | ProRes 422 HQ / 4444 10-bit; HEVC Main10 for PQ/HLG passthrough | Source primaries, transfer, matrix and range tags kept exactly. |

"Validated" means measured: `GET /api/v1/media/export-profiles?verify=true` encodes and fully validates a clip with each
profile on the running machine (rendered-frames path, unmodified-media path, HDR passthrough where applicable), and the
test-suite does the same. A profile whose encoder is missing reports `UNSUPPORTED` with the reason. There is no EXR
*output* (it is a linear-float format; writing gamma-encoded 8-bit pixels into it would be misleading) and no 16-bit
output (the pipeline is 8-bit).

## Input support and validation

* Containers: MP4, MOV, MKV, WebM. CFR and VFR, exact rational PTS, rotation metadata, any number of audio tracks.
  Subtitle streams are probed (text or bitmap).
* Image sequences: PNG, TIFF, DPX (8/16-bit, reduced to 8 with rounding), JPEG (8-bit, lossy), EXR (scene-linear float,
  sRGB-encoded and reported). A sequence has no clock, so an explicit rational frame rate is required; a numbering gap,
  a mixed series or a size change is an error, never silently shifted timing.
* `validate_media(path, deep=True)` returns structured findings (rotation, VFR, non-zero start, **assumed colour tags**,
  HDR, bit depth, audio/video start offsets, multiple tracks, unsupported subtitles) and, when deep, decodes every frame and
  audio packet (PTS order, declared vs decoded frame count, audio decode).
* `validate_output` (run on every export) decodes the finished file completely and checks: dimensions, pixel format,
  all four colour tags, pixel aspect, display rotation, **every frame's PTS** (to within one source time-base tick), frame
  count, container duration (a truncated tail), audio and subtitle track counts, audio decodability.

## Preserving the picture: the fidelity check

Frame count and PTS can be perfect while the colours are wrong. During encode, up to six frames (first, 8th, 64th, 512th,
4096th, last) are kept; after mux the same frames are decoded from the finished file and compared with what the encoder was
given: per-channel signed bias and the 1st/99th percentile deltas (which a limited/full range slip moves by 16 to 20
code values). Lossless profiles must match exactly. Tolerances are per profile (standard 6 / 14, high-quality 4 / 12, 10-bit profiles 3 / 10) and were set after observing real codec noise on saturated synthetic content. Statistics cannot see a range
error in a frame with no black or white, which is why the tags are verified exactly as well and the conversions are
tested against the oracle.

## Audio and synchronisation

Default: **every audio track is preserved and untouched.** Stream copy when the container allows; otherwise re-encode
(AAC, or PCM with `audio_policy: pcm`). `audio_policy: drop` is refused when audio exists. Per track, in the manifest:

* **Payload identity.** For a copied track, SHA-256 of the compressed packets in source and output must match.
* **Signal lag.** Source and output audio are decoded onto the same absolute timeline (the trim origin subtracted) and
  cross-correlated at 8 kHz (0.125 ms resolution). Measured here: **0.0 ms** for stream copy, AAC (the MP4 edit list
  compensates the encoder delay) and PCM, trimmed or not. Tolerances: 0.5 ms for copies, 2 ms for re-encodes. Silent windows
  are reported as "not measurable", periodic content (a pure tone) as "ambiguous" (the nearest peak to zero is taken); neither
  is reported as success. Since video PTS are verified exactly, a bounded audio lag bounds the A/V offset.
* The measurement has teeth: a test shifts a clip by 100 ms and requires a 100 ms reading and a failed check.

Extension of `engine/audio_sync/`: the audio start time is now the first decoded sample's presentation time (the old code
used `stream.start_time`, usually absent, and silently assumed 0); `choose_track` / `decode_selected` make a speech-driven
feature name its track when a file has several (never "the first"); files are opened through a handle with every protocol
disabled. A test proves video-frame to audio-sample alignment end to end through VFR and a trim.

### Local speech-driven lip-sync correction: investigation

Kept as a separate, explicitly selected feature (`lip_sync_mode`, default `off`); it is **UNSUPPORTED** here and the state is
reported honestly. No verified local weights and tensor contract are registered (`LipSyncAdapter` returns the crop
byte-for-byte rather than fabricating mouth frames), and Wav2Lip's published weights carry a research-only licence that
the model governance gate (Stage 12) would not clear for redistribution. Whatever model eventually fills the slot works on
pixels only: it receives a face crop and a mel window and has no path to any audio track, which a test pins (audio packets and
measured lag are identical whatever the video transform does). Enabling it can therefore never change speech.

## Subtitles

Stream-copied (retimed by the trim origin) where the output container can hold the codec: ASS/SSA in Matroska is verified;
`mov_text` into MP4/MOV is verified where an `ffmpeg` command-line tool exists to build the fixture. Anything else is
reported as `drop` with the reason in the manifest; `subtitles: require` turns that into an error. **This PyAV build cannot
open the text subtitle encoders, so there is no subtitle transcoding** and the capability matrix says so.

## OpenColorIO investigation

Installed OpenColorIO 2.6.0 into an isolated scratch directory (never the project environment) and exercised it offline.

* Built-in configs (`ocio://default`, `cg-config-latest`, `studio-config-latest`) load in milliseconds with no files or network.
* **It is an excellent independent reference.** Maximum absolute difference from the native maths: sRGB EOTF 2.4e-5,
  BT.1886 and gamma 2.2 2.4e-5 (float32 LUT precision), BT.2020 to/from BT.709 matrices 5e-8 and 2e-8, PQ inverse EOTF
  5e-8, HLG OETF with the inverse OOTF 3e-8. The one visible gap, BT.709 OETF at 2.5e-4, is expected: OCIO uses the rounded
  textbook constants (1.099 / 0.018), the native code the exact continuous pair.
* **Decision: optional adapter, not a dependency** (`color/ocio.py`). The native maths covers every conversion the pipeline
  performs and is now cross-validated; OCIO would add a binary dependency to a standalone, hash-locked environment for no
  functional gain. The adapter exists for users who install it and want an ACES or studio-config transform; it accepts only
  built-in `ocio://` configs because a config *file* can reference LUT files by path. The cross-validation test runs when
  `PyOpenColorIO` is importable and is skipped otherwise.

## Things worth your review

Decisions I made that you may want to overrule:

1. **HDR input now defaults to tone-mapping to SDR.** Previously it was silently processed as SDR and written as 8-bit "PQ".
   `hdr_policy: reject` is available per job.
2. **Untagged HD is now decoded as BT.709** (by the convention above) where swscale used BT.601, and the output is explicitly
   tagged. This changes pixel values for untagged HD sources (it corrects them).
3. **Diffuse white maps to about 90 % signal**, not 100 %, for 1000-nit masters. Raising it clips highlights; the parameters
   are in `ToneMapParams`.
4. **The stream backend is stricter.** Jobs whose source is untagged (matrix or range), above 8-bit or not 4:2:0 now use the
   reference pipeline instead of the fast stream path.
5. The sync lag tolerance is 2 ms for re-encodes, taken from measurement (0.0 ms) with room for the 0.125 ms resolution.

## Known limits (not done, and why)

* The workstation UI does not expose the new options (`export.profile`, `hdr_policy`, `linear_light`); they are API-level.
* GPU (NVDEC / NVENC) decode and encode could not be exercised on this machine (the bundled FFmpeg CLI has no NVENC, and
  the encoder smoke test falls back to libx264). The colour logic is the same FFmpeg scaler downstream and is tested in
  software mode.
* A 10-bit VP9 WebM could not be fixtured (this libvpx has no high-bit-depth encoder); the 10-bit decode path is shared with
  other codecs and verified through HEVC.
* Subtitle transcoding; EXR output; 16-bit processing; DCI-P3 (needs chromatic adaptation).
* Dithering when tone-mapping to 8 bits (smooth HDR gradients can band).

## Tests

`app/roop_3d/tests/`: `test_stage15_color_science.py`, `test_stage15_media_reader.py`, `test_stage15_media_writer.py`,
`test_stage15_media_av.py`, `test_stage15_media_extras.py`, `test_stage15_linear_light.py`, `test_stage15_stream_color.py`;
fixtures in `stage15_media.py` build real H.264 / HEVC / VP9 / FFV1 / ProRes / PCM / AAC / Opus / ASS files whose pixels come
from the float oracle. Run from `app/` with `roop_3d/.venv/Scripts/python.exe -m pytest roop_3d/tests/test_stage15_*.py`
(never two pytest sessions at once: the shared `--basetemp` is wiped per session). Diagnose an encode failure with
`ROOP3D_MEDIA_TRACE=1`, which prints the underlying FFmpeg error that the API message deliberately hides.
