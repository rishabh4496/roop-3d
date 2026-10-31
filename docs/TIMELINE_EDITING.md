# Curves, history, snapshots and explicit face roles

The timeline has an expandable parameter curve track beneath its filmstrip and
pose tracks. Select swap weight, restorer opacity, mask feather, mask erosion or
pose correction strength. Add at the playhead or double-click the graph; drag keys,
edit their frame/value, or delete them. Each outgoing segment uses Linear,
Ease-In-Out (cubic Bezier), or Step/Hold. Source frame indices remain absolute when
export trim points change. Preview and rendering share the same bounded curve
contract and easing formula.

**Pose capability:** the pose curve now supports an actual TPS 2.5D / perspective
3D mesh blend. New sessions select the estimated canonical-proxy mesh method;
older requests retain the separate reference retarget mode. Endpoints, visibility,
camera estimation and acceptance limits are described in `POSE_MESH_BLEND.md`.
No personalized reconstruction fidelity or GPU performance acceptance is claimed.

## Non-destructive editing

`Ctrl+Z` and `Ctrl+Y` undo and redo workstation parameter edits, identity
assignments, role changes, splits, source-face selections, mask strokes and curve
edits. The shortcuts can be remapped in Settings. The History Drawer lists applied
and undone commands and supports returning to an earlier edit. Typing retains
native text-input undo behavior. A new edit discards the redo branch. History is
session-local, scoped to the current target, and bounded to 120 commands with a
64 MiB accounting budget for recorded command payloads.

Masks are restored into the brush canvas after undo/redo or reopening the editor.
The mask editor displays the source at full canvas size so brush coordinates match
the upright source dimensions. White mask regions preserve the original target
through the compositor's explicit occluder-mask input. The backend validates PNG
format, encoded size, dimensions and alignment before processing a manual mask.

## Snapshot slots

Click an empty A/B/C/D slot to request a rendered final composite at the current
playhead. The slot retains the returned frame plus a copy of workstation
parameters, curves, mask, face assignments/roles, selection, playhead and trim
state. Source-only images are not substituted when rendering fails. Captures
cancel when their input context changes.

Click a populated slot to view its frozen output. Select any two populated slots
and use Split Comparison Wipe. Shift-click a slot to overwrite it. Restore Slot
Settings applies the captured editing state as one reversible command; Live
Viewport returns to current processing. Images stay in local browser memory;
opening another target clears the target's history and slots.

## Explicit roles and splitting

Right-click a detected bounding box, or use the context-menu key / Shift+F10 on
the overlay, to open the face menu. Roles are Swap Target (mint), Reference Guide
(blue) and Exclude / Never Swap (rose). Matching badges appear on the canvas and
Face Tray. Excluded/reference clusters reject source assignment and are kept out
of automatic embedding merges. Returning to Swap Target requires a new explicit
assignment unless the role edit is undone.

The menu can choose a registered FaceSet or one observed face within it, re-center
the pose camera using the face's derived quaternion, and split the cluster from
the current frame onward. Splits retain their original inspected track identity
but create separate, non-overlapping source/role intervals. They remain distinct
during automatic re-clustering.

Render requests and saved projects retain those intervals and curves. Runtime
track selection removes reference/excluded intervals before swap inference.
Changing a source at a split updates identity conditioning without restarting the
tracker. Project verification includes every referenced FaceSet. Legacy requests
without animation/role fields retain their existing behavior and identity digest.

## Verification

- Full Python suite: 416 tests and 18 subtests passed, including the existing real
  golden render. Temporary data stayed in `.runtime/pytest-curves-full`.
- Frontend: 80 tests passed, including actual DOM commands, Ctrl+Z/Ctrl+Y, curve
  editing/evaluation, exclusion matching gates, splits, snapshot capture and wipe,
  and context-menu actions. TypeScript and the Vite production build passed.
- Restarted only the dedicated Roop 3D launcher through Pinokio after verifying
  its queue was idle. Health and production UI returned successfully; OpenAPI
  exposed the new animation contract and source-view endpoint.
- New inference contract tests use deterministic inference fixtures and the real
  compositor. They do not establish new neural-model fidelity or hardware-specific
  acceptance. Visual/multi-monitor acceptance remains unverified.
- The read-only integrity audit passed at `2026-10-31T18:07:35.448771+00:00`:
  68,390 comprehensive immutable checks and 2,645 historical checks, zero
  violations. The existing policy classified 1,588 runtime files separately;
  preserved source/model baselines were not replaced.

The earlier UI acceptance gaps remain documented in `UI_STAGE07_08.md`.
