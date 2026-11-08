# Post-production and compositing exports

Open **Extras** (`#/extras`) to configure render deliverables, finishing, timeline
interchange, or finish an existing local video. The workstation's **Queue render**
uses the same export configuration. Projects and snapshot restores retain it.
Everything runs inside this application; no legacy modules or APIs are used.

## Compositing deliverables

Select the baked video, any combination of passes, or passes alone. Keep the video
timing policy on **preserve** for pass exports. Lossless FFV1 Matroska videos and
authoritative **16-bit PNG sequences** are published into this app's configured
runtime output directory. Sequence paths, exact source-frame IDs, rational PTS,
SHA-256 hashes, and data conventions are in the downloadable pass manifest.
Successful worker cleanup preserves these sequences.

| Pass | Actual input and encoding |
| --- | --- |
| Face alpha | Final temporal face coverage, after available occluder subtraction; greyscale 0–65535. |
| Occluder / XSeg | Actual semantic parser and manual foreground masks; black when no mask is supplied/detected. This application has no XSeg inference adapter and does not promise detection of every hand, microphone, or accessory. |
| UV warped face | Current projected swap RGB before restoration, grading, and blending, in full-frame coordinates; straight/unpremultiplied RGB with separate face alpha. This is a warped face layer, not a UV-coordinate AOV. |
| Normal | Frame-aligned mesh blend geometry, flat camera-space normals encoded as `(XYZ+1)/2`; RGB axes use OpenCV (+X right, +Y down, +Z away). |
| Depth | Linear camera Z / 2048, background zero; estimated canonical camera units, **not metres**. Out-of-range geometry fails explicitly. |
| Clean relit face | The swapped face after SH relighting, before it is blended with the target: straight RGB, no background or occluders, in full-frame coordinates. Pair it with the alpha pass. Requires 3D Relighting to be on (a request without it is rejected, not exported as black frames). |

Normal/depth sequences require **mesh blend** pose correction. Geometry and camera
are estimated canonical proxies, not a calibrated identity reconstruction.
Sequence rasterization uses the actual per-frame projection and z-buffer; it does
not capture the freely orbiting inspection viewport. Face coverage clips these
sequence passes. Multiple faces follow the compositor's deterministic track order.

For a direct **Three.js** export, select mesh blend mode and use **Export
current-frame Three.js normal & depth**. This fetches the current frame's aligned
mesh and renders into floating-point offscreen targets. PFM files preserve signed
normals and linear Z, with a JSON geometry/frame description. These are raw
geometric surface passes: apply face alpha separately to clip texture coverage.
WebGL floating-point framebuffer support is required. Background is zero, rows are
stored bottom-first, and PFM float bytes are little endian. This exports one frame;
long sequences use the backend rasterizer described above.

LUTs, grain, video-range conversion, and provenance do not modify data passes.
FFV1 is verified pixel-for-pixel against the PNG checkpoints before publication.
Matroska timing quantizes to milliseconds; PNG manifest timestamps preserve the
exact source timing. NLEs that cannot decode FFV1 should relink to the PNG sequence
at the declared frame rate or transcode losslessly in their own finishing tools.

## Timeline interchange

Select a completed render made with export options and download:

- **FCPXML 1.10:** source storyline plus connected swap and alpha lanes.
- **OpenTimelineIO:** three Video tracks, explicit gaps, and media source ranges.
- **CMX 3600:** a ZIP containing V1, V2, and V3 single-track EDLs. EDL cannot
  represent a multi-track stack in one file. NDF is the default; the generator also
  supports 29.97/59.94 drop-frame timecode.

V1 contains source footage, V2 actual swapped frame ranges, and V3 available face
alpha. Every detected scene cut and swapped-range boundary produces a separate
clip. Trimming offsets the swapped/alpha media source ranges without moving the
clips on the full-source timeline. Missing baked or alpha deliverables omit those
clips. All ranges are half-open; no guessed rates or approximate second-based
timecodes are used. The source is decoded to verify CFR before interchange;
variable-rate footage and changed source/render files fail with an explanation.
Conform VFR footage before rendering if you need frame-based interchange.

DaVinci Resolve imports the FCPXML, OTIO or EDL files through *File > Import > Timeline*.

Assign V3 as the V2 track matte in the receiving NLE. Track-matte effects and blend
modes are application-specific and are not synthesized into interchange files.
Local file URIs are escaped; relink when moving outputs to another workstation.
FCPXML/OTIO schema structure, rational edit positions, gaps, trims, and EDL timecodes
are covered by tests. Import acceptance in Final Cut, Resolve, Nuke, or After
Effects has not been performed on this host.

## Finishing

The `.cube` loader accepts bounded 3D LUTs (sizes 2–33, at most 1 MiB), supports
DOMAIN_MIN/MAX, validates exact sample counts and finite values, and uses red-fastest
sample order with trilinear interpolation. 1D, mixed, and malformed LUTs fail.
Preview and encoding run the same CPU transform before PyAV/FFmpeg's selected
software or verified NVENC encoder. No new hardware writer is invented.

The working domain is display-referred RGB. There is no implicit log-to-linear,
HDR, OCIO, or gamut conversion: choose a LUT appropriate for the input. The writer
retains its existing 8-bit YUV420 baked-video path; these controls do not introduce
a high-bit-depth HDR mastering encoder.

Grain is deterministic per source-frame ID and seed, with size, intensity, shadow,
and highlight amplitude controls. It follows the LUT on the final composite,
including restored faces. It is manual grain synthesis, not an automatic sensor
noise estimator. Preview the current pipeline's **final** buffer in the workstation,
or use the standalone finishing preview and bake actions in Extras. Standalone
preview uses the decoded frame's real identity, including for VFR inputs.

## Provenance

Standard output retains the synthetic-media notice. Optional modes are:

- **SHA-256:** embeds a render-recipe hash in media container metadata and writes
  the final file's hash into the render manifest. This checks integrity, not identity.
- **HMAC-SHA256:** additionally authenticates the complete manifest using
  `ROOP3D_PROVENANCE_KEY` (at least 32 UTF-8 bytes), kept only in the environment.
  Verification requires the same secret; this is not a publicly verifiable signature.
- **C2PA:** optional `c2pa-python==0.38.0` SDK and an ES256 signing certificate/key
  in `ROOP3D_C2PA_CERT_PEM` / `ROOP3D_C2PA_KEY_PEM`. Install the `provenance` extra
  only into this application's dedicated environment when needed. Signing embeds
  a real C2PA manifest, then reads it back before normal output verification.
  No timestamp server, cloud signer, remote manifest fetch, or media transfer is
  configured. Certificate trust must be verified independently by recipients.

Missing SDK/credentials disable their UI choices and fail explicit API requests.
Keys and certificates are never included in request schemas, render manifests, or
logs. The final file is hashed **after** embedding/signing; the embedded recipe hash
does not pretend to be a self-referential hash of its own containing file.
The **Verify output integrity** action checks file hashes and, when configured,
HMAC authentication. C2PA recipient trust verification uses a C2PA-aware validator.
The optional SDK and a production certificate were not present during acceptance;
no signed C2PA artifact or certificate trust result is claimed.

## Local endpoints

`GET /api/v1/finishing/capabilities`, `POST /api/v1/finishing/preview`, and
`POST /api/v1/finishing/render` expose standalone finishing. Jobs accept the optional
`export` contract (baked_video, passes, finishing), included in checkpoint identity.
`GET /api/v1/exports/{job_id}/timeline`, `/manifest`, `/manifest?passes=true`, and
`/verify` operate only on completed local jobs. Original request identities without
export options retain their prior fingerprints. No models are loaded to boot these
endpoints and no remote media protocols are enabled.

Format references: [Apple FCPXML](https://developer.apple.com/documentation/professional-video-applications/creating-fcpxml-documents),
[OpenTimelineIO schemas](https://github.com/AcademySoftwareFoundation/OpenTimelineIO/blob/main/docs/tutorials/otio-serialized-schema.md),
[C2PA Python SDK](https://github.com/contentauth/c2pa-python/blob/main/docs/usage.md).
