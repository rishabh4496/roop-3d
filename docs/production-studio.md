# Production studio integration

This change adds local DFM inference, Wav2Lip inference before restoration, and a
host-frame live studio. It does **not** establish 100% phoneme accuracy, a 3D
viseme solver, native CUDA/DirectX zero-copy egress, or sub-35 ms end-to-end latency.
MuseTalk, EDTalk and NDI remain unavailable and are reported explicitly.

## Audio Lip Syncer (Stage 2.8)

Install the `lip-sync` optional dependencies from `app/` with the dedicated runtime:

```powershell
uv pip install --python roop_3d/.venv/Scripts/python.exe -e "./roop_3d[lip-sync]"
```

Place a Wav2Lip ONNX export under `app/models/` with `wav2lip` in its filename.
For `wav2lip_gan_fp16`, the filename must also contain `gan` and graph inputs must
be float16. The graph must expose NCHW image `[1,6,96,96]`, mel `[1,1,80,16]`,
and BGR image output `[1,3,96,96]` in 0..1. Other contracts fail explicitly.
The selected runtime provider controls ONNX Runtime CUDA/TensorRT execution;
standalone serialized `.engine`/`.trt` plans are not loaded by this adapter.

In Settings → Pipeline select `lip_sync_mode`: `wav2lip` or `wav2lip_gan_fp16`.
The default is `off`. An empty `lip_sync_audio_path` reads embedded target audio.
Set `lip_sync_audio_stream` when the target has multiple audio tracks. A custom
`lip_sync_audio_path` must name an owned local WAV/MP3 relative to `app/`; it
conditions the mouth and replaces the export audio. Voiceover time zero maps to
target time zero; trims use the same absolute timeline for video and voiceover.

Features use the training frontend's 16 kHz mono audio, preemphasis, Slaney mel
amplitudes and normalized decibels. Each window is indexed independently by the
decoded frame PTS. Silence does not suppress consonant closures. Mouth pixels
are composited before the selected restorer, including UltraMax. This is pixel
conditioning of the rendered crop, not modification of the 3D mesh or a guarantee
of bilabial correctness. Model quality requires evaluation with real weights.

## DFM identity models

Place `.dfm` files anywhere under `app/models/` and rescan the model catalog.
They appear as selectable swappers in the model rack. Source FaceSet selection
is still required by the tracking/pose pipeline; the DFM graph supplies identity.
`dfm_morph_factor` in Settings → Pipeline defaults to 0.75 and accepts 0..1.

The adapter inspects `in_face:0` (square BGR NHWC) and optional `morph_value:0`
(`[1]`), discovers native resolution, and reads the three ordered DeepFaceLive
outputs: input mask, identity image, identity mask. Both masks constrain the
result. Invalid shapes, nonfinite outputs and unsupported graph inputs fail
explicitly. Local graph validation and existing 6 GB / 12 GB provider budgets
apply. Files retain candidate provenance; loading does not certify model rights.

## Live Studio

Install the `live-studio` optional dependencies with the same environment, select
a source FaceSet, then use **Live Studio Mode** in the `#/faceswap` header.
Choose webcam or screen, device index, and 30/60 FPS. Camera rates are requests
to the device; processing FPS can be lower. Windows supports DirectShow/MSMF
(API `backend` field); Linux supports V4L2. Screen capture requires `mss`.

Optional simultaneous outputs:

- Browser WebGL2 preview over the existing bounded JPEG WebSocket stream.
- `pyvirtualcam`, requiring an installed OBS virtual camera or platform driver.
- Windows Spout2 through `SpoutGL.sendImage`, with sender name **Roop 3D**.

Spout currently uploads host BGR pixels. Browser JPEG decode and texture upload,
and pyvirtualcam delivery, also cross host-memory boundaries. None is advertised
as native GPU zero-copy. The status endpoint reports capture-to-publish p95,
which excludes browser presentation and downstream OBS buffering. It cannot
verify the requested end-to-end latency. No live audio ingestion is included.

The process-local session uses existing scheduler admission and holds an engine
lease until cleanup. Stop releases camera, outputs, models and admission; hiding
the panel leaves the session running. Pinokio retains ownership of the process.
API: `GET /api/v1/studio`, `POST /api/v1/studio/start`, `POST /api/v1/studio/stop`.

Contract references: [DeepFaceLive DFMModel](https://github.com/iperov/DeepFaceLive/blob/master/modelhub/DFLive/DFMModel.py),
[Wav2Lip audio frontend](https://github.com/Rudrabha/Wav2Lip/blob/master/audio.py),
[pyvirtualcam](https://github.com/letmaik/pyvirtualcam),
[SpoutGL image sender](https://github.com/jlai/Python-SpoutGL/blob/main/examples/image/sender.py).
