# Dense-face backbone validation, November 9, 2026

## Scope and activation state

This change integrates the existing work-in-progress three-tier native/custom adapters with face-only HRN displacement refinement, conservative photographic correspondence, full neural swapper output and UltraMax. The live branch no longer selects the MakeHuman full-head renderer. Legacy offline head builders remain available for historical assets, not as a live fallback.

**The real three-tier system is not activated or appearance-validated in this checkout.** Local readiness returned exit 1, `ready: false`, on November 9, 2026. Four genuine encoder exports are present: coarse, expression, HRN mid and HRN high. The required anatomy/topology archives, BFM decoder archive, FLAME Open rig and registered surface-transfer archive are missing. No runtime `bundle.json` was created. Existing export-parity reports are not end-to-end quality evidence.

The outstanding dependencies are the actual FLAME 2023 Open model, the original FLAME basis used by the SMIRK checkpoint, measured conversion and validated eyelid directions, anatomical BFM masks, bidirectional BFM/FLAME registration and a registered HRN UV chart on FLAME. See [native preparation](native-dense-face-setup.md). Code licenses do not automatically license the separately distributed BFM, expression-basis or FLAME model data.

## Observed checks

- Consolidated bounded integration and asset-safety run: **265 passed, 6 skipped**, 82.58 seconds, with `RuntimeWarning` treated as an error. The skips are opt-in real-media acceptance fixtures. One existing FastAPI/Starlette test-client deprecation warning was emitted.
- Final focused native geometry, topology, displacement, correspondence and bundle-publisher regressions, including the last allocation-budget guard: **128 passed**, 9.42 seconds, with `RuntimeWarning` treated as an error. Tests use explicitly synthetic geometry/encoder fixtures and analytic photographs. These overlap the consolidated suite and are not additional independent acceptance cases. They prove contracts and numerical behavior, not reconstruction accuracy.
- Workstation UI: **39 tests passed** across `faceset3dViewer.test.tsx`, `facesetInspector3D.test.tsx` and `stage7Studio.test.tsx`. React Three elements emit expected custom-tag warnings in jsdom.
- `npm run build`: passed TypeScript checking, ESLint and the production Vite build.
- Actual local readiness CLI: correctly discovered nested HRN encoder files and supplied export reports, and enumerated all five missing geometry asset roles plus the absent manifest. No inference or GPU context was needed by this command.
- Full backend suite attempt: reached the 600-second command deadline and exited 124. **The full suite did not pass or complete.** Bounded suites are reported separately, not as a replacement claim.

- Follow-through on actual installed models/media: `test_reenactment_session_real.py -rs` produced **3 passed, 2 skipped in 24.22 seconds**. Standard neural video inference and real missing-bundle refusal passed. The two complete three-tier video cases skipped specifically because the bundle is absent.
- Follow-through on actual production ASGI routes with an existing loaded local `.fsz`: health, models and FaceSet listing returned HTTP 200; reconstruction and cached-state read returned HTTP 200 with `template=canonical`, `limited_multiview`, `backbone.available=false` and an explicit missing-bundle reason. Both runtime profile configurations rejected loading with `DENSE_FACE_ASSETS_MISSING`, status 503. No model/media mocks were used. Startup recovery and queue workers were disabled to preserve existing jobs.

The [requirement-level acceptance trace](dense-face-acceptance-trace.md) records each explicit requirement, the interpretation choices, exact observations and outstanding real-data checks. Component improvements are demonstrated; **improved real swapped-face appearance has not been demonstrated**.

## What the checks cover

| Requirement | Concrete observation |
| --- | --- |
| Native topology and anatomical cuts | Full native BFM/FLAME vertex-count fixtures, binary inclusion/exclusion masks, all-three-vertices triangle retention and physical compaction |
| Expression and camera contracts | Native 257-value coarse camera decoding, explicit 50-expression/jaw/independent-eyelid outputs, FLAME LBS and registered barycentric transfer fixtures |
| Genuine UV detail reaches geometry | A supplied displacement peak between original vertices moves refined vertices; shared edges, winding and facial boundaries remain intact; native topology counts are unchanged |
| Profile-specific geometry budgets | Requested refinement 0..2, CPU/6 GB cap 1 and 12 GB cap 2; no assertion of measured VRAM safety or realtime speed |
| Dense photographic correspondence | Perspective-correct source camera interpolation, target visibility, slant-aware source depth planes and every nonzero OpenCV bilinear footprint tap tested against depth and observed support |
| Neural output ownership | Test doubles verify the full neural image, not a shader or high-pass residual, is passed through the enhancer; legacy live CG routing is not invoked |
| Coordinate diagnostics | Aligned-cache fields and separately labelled original-photograph fields agree through the measured affine; invalid coordinates remain -1 and stale fields clear before processing |
| Safe bundle publication | CPU-only ONNX input/output contracts, embedded weight coverage, digests and tamper failures, local-path ownership, no overwrite on races, no publication after invalid geometry/provenance/contracts |
| UI truthfulness | Dense facial preview and explicitly labelled canonical orientation fallback; no MakeHuman head/neck selection in the preferred viewer |

## Remaining acceptance work

After genuine missing assets are prepared, publish a validated bundle and run real `.fsz` source identities against target photos and video. Measure anatomical cut quality at forehead/jaw/ears, expression asymmetry, lip/eyelid registration, identity retention, appearance seams, temporal stability and HRN detail behavior under this coupled input distribution. Run the actual selected HyperSwap/RealSwap/InSwapper plus UltraMax chain, not test doubles. Profile latency, CPU memory and peak GPU allocation separately on physical 6 GB and 12 GB devices.

The available device is an RTX 4070 with approximately 12 GB VRAM. Analytic OpenGL tests are not evidence for the complete model chain or a physical 6 GB deployment. Sub-triangle and sub-UV-map microgeometry remains unresolved, and recovered skin micro-relief is not guaranteed. No fabricated registrations, synthetic wrinkles, model weights, FaceSet photographs or cache contents are committed.

## Reproduce the bounded checks

From `app/`, using the dedicated environment:

```powershell
roop_3d/.venv/Scripts/python.exe -u -B -m pytest -c roop_3d/pyproject.toml --confcutdir=roop_3d --basetemp=roop_3d/.runtime/pytest-dense-validation -o faulthandler_timeout=90 roop_3d/tests/test_dense_face_bundle.py roop_3d/tests/test_dense_face_backbone.py roop_3d/tests/test_native_face_geometry.py roop_3d/tests/test_displaced_surface.py roop_3d/tests/test_dense_correspondence_visibility.py roop_3d/tests/test_full_head.py roop_3d/tests/test_extreme_pose_3d_swap.py roop_3d/tests/test_reenactment_session_real.py roop_3d/tests/test_stage07_runtime.py roop_3d/tests/test_stage11_faceset_mesh.py roop_3d/tests/test_stage14_workstation.py roop_3d/tests/test_swapper_catalog_and_adapter.py roop_3d/tests/test_model_selection_regressions.py -q --tb=short -W error::RuntimeWarning
```

Use a distinct in-checkout `--basetemp` if tests run concurrently. From `app/roop_3d_ui/`:

```powershell
npm test -- --reporter=dot faceset3dViewer.test.tsx facesetInspector3D.test.tsx stage7Studio.test.tsx
npm run build
```
