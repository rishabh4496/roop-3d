# Higgsfield preview investigation and local improvements

Investigation date: 2026-11-12. Scope: public, signed-out https://higgsfield.ai/ and
https://higgsfield.ai/apps/face-swap, plus this Roop 3D checkout. No user media was
uploaded, no generation was purchased, and no Higgsfield service was integrated.

## Evidence and limits

REA 6.8.0 was used through its public CLI because its MCP tools were absent from
the active session. A read-only doctor reported missing Codex registration. No
client settings were changed. `capture-browser-scenario` launched an owned,
headless Chrome, visited the homepage, waited four seconds, navigated to Face
Swap, waited four seconds and captured DOM and network events. REA reports that
its owned browser was terminated on completion.

Evidence ID: `ev_adf1fa894578934ba09f3c1ad0fcbb31dc2336c51a83c1587af18e9ed10aeee0`.
The complete record is in `cache/higgsfield-rea-evidence.json`; inputs are in
`cache/higgsfield-scenario.json`; a compact derived summary is in
`cache/higgsfield-rea-summary.json`. These are ignored local investigation state.
The record, not this summary, is the original REA evidence.

All four recorded steps completed. Collection reached its 1,024 network
transaction budget and omitted later events; several requests were unfinished
when capture ended. Coverage is explicitly incomplete. The observation is not
a browser frame-time benchmark, authenticated editor investigation or backend
model inspection. Network bodies and header values were not requested.

## Why the public preview appears smooth

Observed in the face-swap DOM: the featured demonstration is a native `<video>`
with a poster, `muted`, `playsinline`, `autoplay` and `preload="none"`, loading
`https://static.higgsfield.ai/videofaceswap/showcase-2-v2.mp4`. REA also recorded
media requests to Higgsfield's static media hosts and image requests through its
resizing/WebP delivery URLs. Other example video elements use `preload="none"`.

Inference: native playback of an already generated file separates display work
from neural generation. Posters and deferred media loading can reduce initial
decode/download work. This is a plausible contributor to responsiveness, not a
measured attribution of every interaction or proof of GPU hardware decoding.
No evidence establishes zero-latency live face swapping.

Higgsfield's [public Face Swap page](https://higgsfield.ai/apps/face-swap) describes
uploading a target photo/video and a source face, automatically matching
expressions/lighting and blending the result. It describes facial tracking and
quotes typical generation times of about 30 seconds to two minutes, affected by
queue and subscription, with video usually slower. Those are vendor statements,
not measured performance. Exact networks, weights, identity embeddings, temporal
losses and serving hardware remain unknown. Public frontend inspection cannot
recover proprietary server-side inference.

## Roop 3D's corresponding behavior

The local pipeline detects and tracks faces, selects the mapped source identity
and reference pose, aligns face crops, runs identity transfer, optionally applies
expression/geometry controls, then composites masks, color, restoration and
finishing. See `app/roop_3d/engine/volumetric/identity_swapper.py`,
`app/roop_3d/engine/volumetric/swapper_engine.py` and
`app/roop_3d/server/workstation_preview.py`.

Roop already has binary WebSocket previews, newest-frame decode buffering,
requestAnimationFrame canvas presentation and no per-frame React state updates
in `src/liveStream.ts` and `src/components/LiveStreamCanvas.tsx`. Backend JPEG
encoding and preview fanout are bounded and run outside the render slot. These
features should be retained.

Two verified costs were addressed:

- Normal live requests consumed only the final image, but the backend captured
  intermediate stages at full resolution, synthesized diagnostic normal and
  lighting maps, drew detection/pose overlays and decoded the source thumbnail.
  The live final-image path now composes without diagnostic capture and returns
  before that extra work. Finishing still runs. The workstation debug endpoint
  and solo-buffer previews retain their original behavior.
- Returning to a rendered frame still required a debounced HTTP request, even
  when the backend could serve it from cache. The live hook now retains up to
  12 encoded results within a 24 MiB blob/key budget for 30 seconds. An exact
  payload match is reused immediately. Different settings, masks, identities,
  media or timestamps have different keys. No object URLs or GPU textures are
  retained in the cache. Retry, runtime changes, job start, source-builder save,
  angle-scan completion, disabling preview and unmount invalidate it. External
  in-place file changes without a UI event can remain cached until expiry;
  explicit Retry forces a request. This cache is per mounted preview and uses
  local memory only.

These are original changes applying general preview design principles; they do
not copy Higgsfield application code or install its models.

## Validation and remaining work

- Frontend: 84 test files, 863 tests passed. Includes byte/count eviction,
  expiry, cancelled/oversized result exclusion, exact-payload reuse, settings
  changes and explicit invalidation tests.
- Production frontend build, TypeScript checking and ESLint passed. Vite retains
  a large-chunk warning for the Three.js bundle.
- Backend: 68 passed, 2 skipped across workstation preview, live stream,
  production studio and remediation stage 4 tests, using the dedicated venv.
  A regression compares final pixels and metadata exactly against the debug
  path and fails if diagnostic normal estimation or overlay drawing runs.
- `git diff --check` passed. Existing unrelated UI work was preserved.

No representative-media GPU before/after benchmark was run, so no percentage
speedup or real-time FPS is claimed. Cold preview sessions still construct and
release ONNX models (`engine/runtime.py`, `identity_swapper.py`). Tracking-aware
and explicit frame-index requests can replay from frame zero
(`workstation_preview.py`). A larger follow-up would need measured model-session
reuse and validated tracking checkpoints, with independent memory admission for
6 GB and 12 GB profiles, proper invalidation and job ownership. Those lifecycle
changes were not made in this focused improvement.
