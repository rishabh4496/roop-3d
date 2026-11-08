# Roop 3D Workstation UI Bug Audit and Feature Roadmap

* **Author:** rishabh4496
* **Date:** 2026-11-08
* **Review status:** Preliminary working brief. Goldfish review was not run because the available agent interface cannot establish the fresh, isolated reviewer context required by the EGM process. No simulated reviewer findings are included.

## Overview

Roop 3D's React workstation should behave like a focused video editor: selecting a frame should move the playhead and show that frame in the preview; timeline and preview navigation should be direct and predictable; and the layout should keep important controls visible without crowding the work area. The user reports that frame clicks do not show the chosen frame and that preview/timeline controls feel non-clickable, unzoomable, untrackable, or undraggable. The work should first audit and fix these interaction failures, then propose additions to the preview, timeline, FaceSet library, and overall layout.

“Find all bugs” is treated as a systematic audit target, not a promise that every defect can be discovered in one pass.

## Analysis

### What the initial source skim shows

The requested areas already have substantial functionality, so replacing them or adding duplicate controls would risk increasing clutter:

* `app/roop_3d_ui/src/components/timeline/TimelineWorkstation.jsx` includes a filmstrip whose frame cards call `onSeek`, pointer-driven scrubbing, timeline zoom controls, frame stepping, trim handles, markers, and other navigation controls.
* `app/roop_3d_ui/src/VideoViewport.tsx` uses the selected time to request a still preview while paused and includes the preview and face-selection workflow.
* `app/roop_3d_ui/src/components/facemgr/FacesetLibrary.tsx` already supports a searchable list, FaceSet selection, import, rename, delete, refresh, thumbnail repair, saving a track face, and dragging a FaceSet toward a target track.
* The sampled tests cover preview toolbar rendering and basic zoom/grid actions, some FaceSet search behavior, timeline rendering and timecode editing, and zoom/pan math. This skim does not establish whether other tests cover the complete interactions across components.

This makes the reported frame-click problem an integration and behavior question, not proof that a click handler is absent. The audit should trace the complete path from clicking a filmstrip card, through seeking and asynchronous frame decoding, to the visible preview. It should check for stale results after rapid clicks, disabled state, missing media, loading/error feedback, pointer interception, and discrepancies between thumbnail time, decoded frame, playhead, and selection state. These are audit hypotheses, not confirmed defects.

### Bug audit scope

Record each finding with steps to reproduce, expected and actual behavior, severity, affected component, and a regression test. Separate confirmed defects from user reports and unverified hypotheses.

1. **Frame selection and preview sync:** Clicking a filmstrip frame, ruler position, marker, or keyframe should seek to the intended frame and update the visible preview. Verify frame-accurate time conversion, rapid sequential seeks, decoder errors, stale request cancellation, and final-frame behavior.
2. **Pointer interactions:** Test click, drag, pointer capture, release outside the control, pointer cancellation, touch input, and hit-area overlap for scrubbing, trim handles, FaceSet drag-and-drop, and preview panning.
3. **Zoom and navigation:** Test timeline zoom, horizontal navigation, preview zoom and pan, fit/reset, boundaries, keyboard modifiers, and behavior at small window sizes. Verify that zoom does not unexpectedly move the playhead or lose the selected frame.
4. **Face tracking and assignment:** Verify that choosing a detected face, following its track over time, mapping a source FaceSet, and moving between frames preserve the intended identity and visible selection. Make the meaning of “trackable” explicit in the implementation and labels.
5. **State and error handling:** Check empty media, unsupported codecs, delayed or failed thumbnail/preview requests, route changes, loading states, stale selections, and controls that are disabled without explanation.
6. **Usability and accessibility:** Check keyboard focus and shortcuts, accessible names, hit target sizes, contrast, panel crowding, responsive layouts, and whether decorative layers block controls.
7. **Regression coverage:** Add end-to-end tests for the real user path from selecting a timeline frame to seeing that exact frame in the preview. Add interaction tests for drag, zoom, face selection, FaceSet assignment, and failure states. Existing render-only assertions are not enough to establish that these pieces work together.

### Feature ideas, staged to avoid more clutter

**Stage 1: Make current interactions reliable**

* A clear selected-frame highlight and playhead position that stay synchronized with the preview.
* Immediate loading and error feedback while the selected frame is being decoded, without displaying an older frame as if it were current.
* Predictable click-to-seek and drag-to-scrub behavior, with frame snapping and one-frame stepping.
* Clear affordances and accessible alternatives for drag-only actions, including FaceSet assignment.

**Stage 2: Improve preview and timeline navigation**

* Preview zoom centered on the pointer, pan by a familiar gesture, and clear Fit and 100% controls. Keep existing zoom, overlay, comparison, and snapshot features only where they are useful and discoverable.
* Timeline zoom centered on the pointer or playhead, straightforward horizontal panning, and an always-visible indication of the current zoom level.
* Optional visibility controls for timeline tracks so face tracks, keyframes, chapters, and trim controls do not compete for attention.
* A compact transport bar with frame stepping, play/pause, timecode, and playback speed grouped consistently.

**Stage 3: Reduce visual clutter and improve workspace flow**

* Group secondary actions into clearly labeled menus or collapsible sections instead of showing every control at once.
* Resizable and collapsible preview, timeline, inspector, and FaceSet panels, with a small number of saved workspace layouts.
* Consistent control labels, disabled-state explanations, spacing, contrast, and focus states across the workstation.

**Stage 4: Extend the FaceSet library if the audit confirms a need**

* A detail view that shows the views inside a FaceSet before assignment.
* Optional tags, favorites, and sorting for larger local libraries.
* Clear drag targets and an equivalent click-to-assign workflow.
* Visible import validation and duplicate/invalid-file feedback.

These are candidate additions, not a commitment to implement all of them. Keep library operations local. Do not add features that duplicate current controls or add visible complexity without solving a demonstrated workflow problem.

### Success criteria

* Selecting a filmstrip frame changes the playhead and the visible preview to that frame. Rapid selections must not let an older decode replace a newer selection.
* Scrubbing, zooming, panning, trim dragging, and FaceSet assignment have clear feedback and work with supported pointer and keyboard input.
* A user can find the primary preview and timeline actions without navigating a wall of controls. Secondary tools remain available when needed.
* Automated integration coverage exercises the complete frame-selection-to-preview path and key failure cases.
* Every reported defect in the audit is marked confirmed, not reproduced, or unresolved, with evidence and a regression check where applicable.

## Goals vs. Non-Goals

* **In-Scope Goals:** Audit the current preview, timeline, frame selection, face tracking, FaceSet library, and workstation layout; fix confirmed interaction and synchronization defects; create a prioritized feature roadmap based on common video-editor interaction patterns; improve test coverage for cross-component behavior.
* **Out-of-Scope Non-Goals:** Claiming an exhaustive bug guarantee; replacing the React workstation wholesale; adding cloud storage or remote processing; adding speculative controls before existing behavior is tested; changing face-processing algorithms unless the UI audit identifies a direct contract or display problem.

## Alternatives Weighed

* **Add more controls before investigating current behavior:** Not selected for the initial stage because preview zoom, timeline zoom and scrubbing, and FaceSet selection/drag features already exist in the sampled code. More controls could make the clutter worse without fixing the reported frame-selection failure.
* **Replace the current preview, timeline, or FaceSet library:** Not selected at this stage. The current code has reusable components and useful behavior. First establish whether the failures come from wiring, state synchronization, pointer handling, layout, or decoding.
* **Promise to find every bug in one pass:** Not a defensible acceptance criterion. Use a documented, repeatable audit and report what was tested, what was confirmed, and what remains unverified.

## Key Decisions & Staging for Next Steps

1. Use familiar non-linear video editor conventions as the default interaction model rather than asking the user to specify each control.
2. Treat “click a frame and switch to that frame” as a core acceptance behavior. The playhead, selected frame, and visible preview must agree.
3. Repair and simplify existing functionality before adding more visible controls.
4. Keep confirmed bugs separate from reported symptoms and audit hypotheses.
5. Start implementation planning with the frame-selection-to-preview path, then audit pointer/zoom behavior, FaceSet assignment, and visual organization.
6. Before implementation, run a fresh-context multi-lens Goldfish review if an isolated reviewer tool becomes available. This brief has not passed that review.
