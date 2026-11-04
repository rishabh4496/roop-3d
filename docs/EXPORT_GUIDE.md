# Export guide

What you get out of Roop 3D, where it goes, and how to choose the format. Everything here is a real option of the application.

## 1. Quick choice

| I want to… | Choose |
|---|---|
| Watch or share the result anywhere | **Standard MP4** (the default) |
| A final copy to publish | **High-quality MP4** |
| A master to keep | **High-bitrate archival (HEVC 10-bit)** |
| Edit it in Premiere, Final Cut or DaVinci Resolve | **MOV workflow (ProRes 422 HQ)** |
| Nothing lost between steps | **Lossless intermediate (FFV1)** |
| One image per frame for a compositing program | **Image sequence** |
| Keep HDR / wide-gamut colour tags for an editor | **Professional colour-preserving** |

Pick the preset in the **Export preset** box next to the trim bar (Face Swap screen) *before* you press **Queue render**. It is saved with the project.

## 2. The presets

| Preset | Container | Bit depth | What it is |
|---|---|---|---|
| Standard MP4 (`standard_mp4`) | `.mp4` | 8-bit | H.264 4:2:0, the widest playback compatibility. The same encoder settings as the original default output. The **only preset allowed to use hardware (NVENC) encoding** |
| High-quality MP4 (`high_quality_mp4`) | `.mp4` | 8-bit | H.264 High profile, slow preset, CRF 16, fast-start MP4 |
| High-bitrate archival (`archival_hevc10`) | `.mkv` | 10-bit | HEVC Main10, CRF 10, slow preset in Matroska; visually transparent; keeps every audio and subtitle stream |
| MOV workflow (`mov_workflow`) | `.mov` | 10-bit | Apple ProRes 422 HQ (4:2:2) for editing and finishing |
| Lossless intermediate (`lossless_intermediate`) | `.mkv` | 8-bit | FFV1 version 3. Rendered frames are stored as RGB so decoding returns the encoder input exactly |
| Image sequence (`image_sequence`) | folder | 8-bit | One numbered PNG, TIFF or DPX per frame plus a sidecar manifest with exact timestamps, colour description and an audio sidecar |
| Professional colour-preserving (`color_preserving`) | `.mov` | 10-bit | ProRes 422 HQ / 4444 keeping the source's colour primaries, transfer, matrix and range tags; untouched PQ/HLG media is carried as HEVC Main10 with its HDR10 static metadata |

> **Honest limits.** Everything Roop 3D produces is **8-bit, display-referred (SDR) RGB**. A 10-bit preset reduces *encoding* loss, but it cannot add
> precision the 8-bit pipeline does not have, and no preset claims "HDR output" for rendered frames. HDR can only pass through *unmodified*, on the
> presets whose codec and pixel format can carry it.

Each preset's live details (encoder, pixel format, whether it is lossless) are served by `GET /api/v1/media/export-profiles`; add `?verify=true` to
encode and validate every preset on your machine.

## 3. More export options (the *More export options* section under the preset)

| Option | Values | Meaning |
|---|---|---|
| If the video is HDR | *Convert to SDR* (default) · *Stop with an error* | Rendered frames are 8-bit SDR, so HDR is tone-mapped or the render is refused (`hdr_policy`: `tonemap_sdr` / `reject`) |
| Subtitles in the source | *Copy them if the container allows* (default) · *Require them (fail if they cannot be kept)* · *Drop them* | `subtitles`: `copy` / `require` / `drop` |
| Image format | PNG (default) · TIFF · DPX | Only for the *Image sequence* preset |

A preset applies to the **baked video**. If you turn off *Final baked video* in the Editor screen (passes only), the preset box is disabled.

## 4. Audio and timing

By default the source **audio is copied** and the source **frame timing is preserved** (Settings → Finishing: *Audio policy* `copy`, *Video timing policy*
`preserve`). *Audio policy* can also be `transcode`; `drop` is not offered because the workstation preserves source audio. If the audio cannot be
carried over you get `AUDIO_STREAM_MISSING`.

## 5. Where the files go

Finished videos are written to `app/roop_3d/.runtime/outputs` as `roop3d_<job id>` plus the preset's extension (a folder for an image sequence), together with
a **manifest** that records hashes and the settings used. They are listed on the **Outputs** screen, and the queue's **Inspect output** plays them with
**Export video** and **Render manifest** download buttons. Only the range between **In** and **Out** is rendered.

## 6. Every render is verified

After encoding, Roop 3D checks the file against the source before publishing it: dimensions, aspect, rotation, duration, frame count, timestamps,
audio/video sync, pixel format, colour tags and (when relevant) HDR metadata, plus a fidelity check against what was handed to the encoder. A failed
check raises an `OUTPUT_…` code (for example `OUTPUT_FIDELITY`, `OUTPUT_FRAME_LOSS`) and the file is not published; try again with a higher-quality
preset. A render never leaves a half-written file behind: on a hard stop the output either does not exist or is complete and hash-checked.

## 7. Editor screen: passes, finishing and interchange

**Compositing passes** (*Render deliverables*): 16-bit PNG sequences plus lossless FFV1 video, with the face RGB unblended and straight alpha supplied separately.

| Pass | Contents |
|---|---|
| Face-canvas alpha matte | Where the swapped face is |
| Occluder / XSeg matte | Foreground occluders from the available parser or manual masks; black where none were detected. *Not* an XSeg inference claim |
| UV warped face | The face in UV space |
| 3D normal map / 3D depth map | Estimated geometry; need a 3D pose mode (**mesh blend**); the checkboxes are disabled otherwise. Depth uses canonical camera units |
| ~~Swapped-face-only foreground (RGBA)~~ | Listed but **not available in this build**: the render service does not produce it, so the box is disabled |

**Finishing pipeline** (needs a baked video): *3D LUT* (`.cube`, up to 1 MiB, 3D size 2–33) → *grain* (size 0.5–8, intensity 0–0.2, shadow and highlight
balance 0–2, seed) → video encoding → optional signing. Data passes keep their original values.

**Provenance** (what is embedded in the file):

| Choice | What it adds | Needs |
|---|---|---|
| Standard synthetic-media notice | The notice only | Nothing |
| Embedded recipe hash + SHA-256 manifest | Integrity check | Nothing |
| Keyed cryptographic manifest (`hmac-sha256`) | Authenticates with your secret | The local key, environment variable `ROOP3D_PROVENANCE_KEY` |
| C2PA signature | A signed content credential | The local C2PA SDK and ES256 credentials (`ROOP3D_C2PA_KEY_PEM`, `ROOP3D_C2PA_CERT_PEM`) |

SHA-256 proves a file was not altered; only the keyed and C2PA choices prove who made it. An image-sequence preset cannot carry embedded provenance.

**Interchange** (from a completed job): Premiere/FCP **XML**, **FCPXML**, **OTIO**, and **EDL** tracks. Relink local media if you move it; assign V3 as the
V2 track matte in your editor.

See [Post-production and acceptance limits](POST_PRODUCTION.md) for the engineering detail, and [Common problems](COMMON_PROBLEMS.md#export) for output errors.
