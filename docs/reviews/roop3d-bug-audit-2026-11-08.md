# Roop 3D bug audit and feature backlog

**Date:** 2026-11-08
**Scope:** React workstation, preview and timeline interactions, FaceSet library, API diagnostics, asynchronous processing and cleanup, and Pinokio launchers.

## Summary

This pass fixed reproducible defects found in the exercised UI and backend paths. It covered the React workstation and browser interaction tests, API and pipeline paths, resource cleanup, and launcher entrypoints. It does not prove that every possible bug is gone, especially on every GPU, driver, video format, and Windows installation.

## Confirmed defects fixed

### Workstation UI

- Timeline frame selection now requests the selected frame. Timeline zoom is anchored to the pointer and the timeline supports panning, rather than behaving like a static strip.
- At desktop widths around 1680 pixels, the top-bar layout control was being hidden to make room for other controls. The layout control now remains available and the palette button compacts to free space.
- Selecting an already tracked face could try to create another persistent target lock. Selection now updates the active face without duplicating the lock.
- Target-lock UI and polling could fail on partial or stale scan responses. Missing scan state is now handled safely.
- Live-stream telemetry appeared blank until the stream had been open for a delay. It now renders on the first frame and is then rate limited.
- Corrupted text characters were visible in portions of the canvas UI and have been corrected.
- FaceSet library browsing now supports favorites and improved search and sorting.

### Processing and API

- An input iterator that raised while being created could terminate a worker before it reported failure, leaving the consumer waiting. Iterator setup is now covered by the worker's error path.
- Queue cleanup could skip release callbacks when a cleanup callback raised, and queued items abandoned during cancellation could remain unreleased. Release handling now runs once across those paths.
- Shutdown now stops processing workers before closing model, media-frame, and live-stream resources.
- Live encoder initialization and capability probing could fail outside the encoder worker's error handling. Initialization failures are now isolated and reported through that path.
- Diagnostic reports could crash when a model record had no SHA-256 value. The report now keeps the checksum nullable.
- Legacy API aliases caused duplicate OpenAPI operation IDs. The aliases still work at runtime but are omitted from the generated schema in favor of canonical routes.
- Two tests had expectations that no longer matched the model catalogue and memory-test behavior. Their assertions were aligned with the current contracts and a relative-memory check.

### Pinokio launchers

The launcher entrypoints were reviewed for URL capture and handoff, health polling, process ownership, and reset scope. No confirmed launcher defect requiring a source change was found in this pass.

## Validation

- Frontend unit tests: **447 passed across 52 files**.
- Frontend production build: **passed**, including TypeScript checking and ESLint.
- Browser E2E suite: **70 passed**. This includes accessibility, responsive layout, preview/timeline interactions, FaceSet flows, stream handling, WebGL rendering, and a resource soak.
- Backend focused suites: **96 passed, 4 skipped** across async pipeline, runtime, API diagnostics, and live-stream paths. Additional focused asset and API modules passed separately.
- The complete Python suite did not finish within the 600-second command limit. It collected about 1,600 tests and made progress without producing a final suite result. Therefore, a clean full-backend-suite result is not claimed.
- The browser resource soak showed stable DOM node and listener counts and bounded heap growth. Four sockets remained open at the end for the app's active telemetry, preview, and event connections, which the test treats as expected.

## Suggested feature backlog

These are proposed additions, not claims about current defects. The timeline already supports frame navigation and zoom/pan interactions. FaceSet browsing already has favorites and search.

### Priority 1

1. **Audio waveform and synchronized scrubbing:** show the source clip waveform, align it with video time, and support audio-aware in/out points.
2. **Track quality lane:** show face confidence, occlusion, scene changes, and identity drift along the timeline. Let the editor jump to a flagged frame and correct the assignment.
3. **FaceSet organization:** add user tags and collections, duplicate or near-duplicate review, and a report of missing pose coverage before a render.
4. **Batch preflight:** validate multiple clips and selected FaceSets before queueing, with per-item reasons for warnings or blocking issues.

### Priority 2

5. **Named timeline ranges and notes:** save reusable labeled ranges with notes such as “check profile turn” or “exclude title card,” alongside existing markers and trim controls.
6. **Render history and reproducibility:** keep a local record of input, settings, model versions, hardware profile, and output for each render, with a way to reopen or compare prior settings.
7. **Preview quality controls:** let users choose a fast proxy preview or full-resolution preview and clearly show when a frame is stale or still rendering.

## Remaining risks and follow-up

- The full Python suite needs a longer-running validation job or test partitioning so it can finish within a practical time limit.
- GPU-dependent behavior should still be exercised on both supported memory profiles and real target hardware. The browser tests cannot validate GPU inference quality.
- The React test environment prints warnings for Three.js scene elements rendered through a DOM-only test renderer. The tests pass, but a renderer-aware test harness would make genuine UI warnings easier to spot.
