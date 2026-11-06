# Stage 7: multi-angle FaceSet builder

Open **Build / Edit .fsz Faceset** beside the source library or in the 3D pose panel.
Drop a video or photo batch, or enter a checkout-local video/image-directory path.
External files use the existing chunked asset upload pipeline. Select a person cluster,
review the coverage sphere/grid, and toggle gallery crops. Selecting an alternative in
a bin replaces that bin's previous choice. Save & Activate publishes the identity and
refreshes the viewport, including when editing the currently selected identity.

Extraction uses the production SCRFD / ArcFace / 106-point adapters, the verified
`1k3d68.onnx` 3D landmark model, and BiSeNet/XSeg occlusion analysis. 3D68 decoding
follows [InsightFace's landmark adapter](https://github.com/deepinsight/insightface/blob/master/python-package/insightface/model_zoo/landmark.py).
The same Stage 3 PnP observer determines both selection bins and reconstruction coverage.
No synthetic angles are counted as observed coverage. Depth and unseen texture retain
Stage 3's existing reconstruction limitations.

Cluster ranks follow first observation (faces ordered left-to-right within a frame).
Cosine matching uses 0.60; accumulating detections never sorts or renumbers identities.
Review clusters when appearance, occlusion, or extreme pose makes recognition ambiguous.
Two quality-ranked alternatives per identity/bin are retained, with at most 16 identities.
Automatic selection requires measured visible-face fraction >= 0.65. Unknown occlusion
is shown for manual review and never silently counted as an unobstructed sample.

One extraction runs at a time, sampling every fifth frame by default, up to 300 frames.
The JSON API allows `frame_stride` 1–300 and `max_frames` 1–3000. There are four in-memory
drafts; older drafts expire when new jobs are created. Restarting the app discards drafts.
GPU sessions use the existing separate 6 GB / 12 GB budgets and inference/build gate.

API:

- `POST /api/v1/faceset/extract-from-media`: JSON `{media_path?, image_paths?, faceset_id?, max_frames?, frame_stride?}`.
  Supply `faceset_id` to load existing angles and optionally add media. Also accepts multipart
  image/video `files`, with optional `faceset_id`. Direct multipart requests obey the configured
  `ROOP3D_MAX_BODY_BYTES` limit (2 MiB by default); use chunked asset uploads for larger batches/videos.
- `GET /api/v1/faceset/extraction/{job_id}`: status, progress, stable clusters, thumbnails and default selected IDs.
- `POST /api/v1/faceset/save-fsz`: `{job_id, cluster_rank, selected_ids, name, faceset_id?}`.
  Only one cluster and one crop per bin can be saved. Return includes the refreshed Stage 3 reconstruction.
- `GET /api/v1/faceset/list`: saved local builder archives and coverage counts.

Archives in `app/facesets/` contain V2 crops and 512-D embeddings, observed 68-point
geometry, a curation tensor entry with measured 3D68 and 106-point landmarks, the 96-bin
observations, and `uv_albedo.png`. Existing renderer/source-library APIs receive identical
archive bytes at their compatibility path. Edits rebuild the texture and invalidate pose
and reconstruction caches. Legacy archives preserve whatever geometry was available.

Run the real-model smoke verifier from the repository root:

```powershell
app/roop_3d/.venv/Scripts/python.exe -B verify_stage7_faceset_builder.py --profile 12gb
app/roop_3d/.venv/Scripts/python.exe -B verify_stage7_faceset_builder.py --profile 6gb
```

An optional positional media path selects a different checkout-local reference. By default,
the verifier scans 12 frames of the existing local sample video, checks identity clusters,
saves an ignored archive in `app/facesets/`, validates its contents, and reloads the identity
through Stage 3. Explicit GPU verification fails if 3D68 execution falls back to CPU.


Validation performed:

- Real local video, both `6gb` and `12gb`: two identities, three observed pose bins;
  3D68 ran on CUDA without CPU fallback. Save, reload, crop removal, and immediate
  reconstruction refresh all passed; verifier exit code 0.
- Five new backend curation tests and four existing archive-reader tests passed,
  including a synthetic 96/96-bin archive capacity check and multipart photo uploads.
- All 415 frontend tests, the production build/typecheck/lint, and a real Chrome
  desktop/mobile builder test passed. Browser screenshots are in the ignored UI test-results directory.
- The optional full backend suite was interrupted after reporting 90% progress with
  failures and then reporting no further progress for several minutes. Six failures
  were separately reproduced in the unchanged legacy runtime-policy tests, whose
  protection expectations conflict with the current standalone repository rules.
  Other broad-suite failures were not diagnosed in this stage. The partial run log
  is `logs/stage7_pytest.log`; this is not a claim that the full backend suite passes.
