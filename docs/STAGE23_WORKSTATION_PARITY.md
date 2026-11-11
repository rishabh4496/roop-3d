# Stage 23: workstation parity (multi-person, paint/erase ranges, batch matrix, 10-state queue)

Path mapping used for this stage: the prompt's `app/roop_3d/` and `app/roop_3d_ui/` are real. `rishabh4496/roop-ultimate` is not
in this checkout, so parity was measured against what roop-3d already had, not copied.

## What already existed (audited, not rebuilt)

| Prompt item | Already in the tree |
|---|---|
| 4 batch strategies | `BatchMatrixEngine` (One-to-Many, Grouped, Per-File Matrix, Recipe Matrix) + `/api/v1/batch/*` |
| Persistent queue | `queue_manager.py` (SQLite, reorder, duplicate, retry slice, clip join, `recover_on_startup`), `test_stage09_remediation.py` restart test |
| 9 deep-linked routes | `router/AppRouter.jsx` `SCREENS`, screens stay mounted across navigation |
| Environment-health card, `UNVERIFIED` update check | `SettingsScreen.jsx` (`compatStatus` defaults to `UNVERIFIED`, offline) |
| Canvas modes | Wipe, Crossfade, Alpha Blend, Diff Map, 3.5x Loupe in `WorkstationCanvas.jsx` |
| Include / Exclude / Erase brush | `MaskBrushOverlay.jsx` |

## What this stage added or fixed

1. **Multi-person actors** (`roop_3d/media/actors.py`, `server/actor_routes.py`, `ActorPanel.tsx`).
   Tracks are grouped into "Person #1 … Person #N", ranked by `screen_time * sqrt(mean_box_area / largest_box_area)`
   (ties: screen time, earliest appearance). Screen time counts each second once even if a person's tracks overlap.
   `POST /api/v1/actors/{inspection_id}/mappings` turns `{actor: faceset}` into one `face_mappings` entry per track, so one job
   can swap several people, each with its own `.fsz` (its own `beta_source` skull lock and atlas). The renderer was not changed:
   `JobRequest.face_mappings` already carried one FaceSet per track.
   * Grouping evidence over HTTP is the tracker's `persistent_id` only. Identity embeddings stay in the local sidecar by design,
     so embedding merge (`cluster_actors(..., embeddings=...)`, never joins two people visible at the same time) is
     library-only today. Without `persistent_id` every track is its own person; position is never used to guess.
2. **Real defect fixed: RGBA brush masks never reached video jobs.** `async_pipeline/reference.py` passed occluders to the
   compositor only when `edit_plan.mask is not None`, but the include/exclude brush writes RGBA, which lives in
   `mask_channels`. Only legacy grayscale masks worked. The gate is now `edit_plan.has_masks`.
3. **Mask frame ranges** (`mask_keyframes` on `JobRequest`, `MaskKeyframe`, `utils/maskRanges.ts`). A painted range overrides the
   whole-clip mask on exactly its frames; ranges may not overlap (server rejects, UI trims/splits). The live preview picks the
   range mask for the previewed frame, matching the render. The brush has "Whole clip / Frame range" scope, a "This frame"
   shortcut and removable range chips. Not part of the undo stack (the whole-clip mask is).
4. **Canvas**: new `Thermal` colormap (default for Diff Map), loupe 3.5x / 4x switch, and a sixth mode "3D Mesh Overlay" that
   turns on the existing 6DoF craniofacial mesh layer over the swapped frame. It is an orientation overlay registered to the
   nose tip, not a per-vertex fit (same limit as the Mesh HUD toggle).
5. **`roop_3d/queue/` package**: the batch engine moved to `queue/batch_matrix.py` (`roop_3d/batch_matrix.py` is an alias to the
   same module object). New `plan_segments`: each distinct target is GOP-split once and shared by every manifest using it,
   failures never abort the batch and are shown on the job as an advisory.
   Bug fixed: `split_video_gop` named chunks with a wall-clock second, so two different targets split in the same second
   overwrote each other's `chunk_0001_<ts>.mp4`; the planner passes a per-file stamp.
6. **10 lifecycle states** (`queue/lifecycle.py`, `QueueJob.lifecycle_state`, `GET /api/v1/queue-lifecycle`). The persisted
   states are not renamed (it would invalidate saved queues); the 10 names are a pure projection of `(state, progress.stage)`.
   `compiling_trt` appears when the stage text names an engine build (`compil…`, `trt_build`, `engine_build`).
   **Nothing in the runner reports such a stage yet**, so today `compiling_trt` is reachable but never produced.
7. **Routes survive reloads**: the last of the nine screens is remembered (`localStorage`, key `roop3d-last-route`) and used only
   when the URL has an empty hash, so typed or clicked links always win.

## Not done / limits

* Two job systems exist: the UI's `/jobs` store (`queued/preparing/running/…`) and `QueueManager` (`/queue`, batch). The new
  `lifecycle_state` is on the latter. The `#/processing` list still reads `/jobs`.
* Browser check of the new canvas controls is limited to unit/markup tests plus the build; see the Stage 24 verification notes.
* `queue` is now also a sub-package name inside `roop_3d`. Never put the `roop_3d` directory itself on `sys.path` (a script run
  from inside it would see `roop_3d/queue` instead of the standard library `queue`); `test_queue_package_does_not_shadow_stdlib_queue…`
  covers the normal import path.

Tests: `roop_3d/tests/test_stage23_workstation_parity.py` (14), `src/stage23.test.ts` (5), `src/stage23Ui.test.tsx` (13).
