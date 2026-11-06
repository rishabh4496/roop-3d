# Roop 3D user manual

Roop 3D replaces the faces in a video you already have, using photos of the person you want to put in. It runs **only on your computer**: your
videos, photos and results are never uploaded. This manual follows the workstation as it appears on screen.

> **Use it responsibly.** The output is synthetic media. Only use videos and photos you have the right to use, and only for people who have
> agreed. Roop 3D keeps a synthetic-media notice on its outputs by default (Settings → Finishing → *synthetic media notice*, which cannot be
> switched off) and can embed a provenance record ([Export guide](EXPORT_GUIDE.md)).

If you would rather be guided, press **Guide** in the header at any time: it shows your progress through these steps and what "done" looks like for each.

---

## 1. The window

**Header** (always visible)

| Control | What it does |
|---|---|
| **How to Use** | A four-step visual tour of the workstation |
| **Guide** | The full guide: workflow checklist, target faces, FaceSets and photos, engines and hardware, render presets, queue and recovery, display and keyboard |
| Project name + **SAVED / UNSAVED** | Name your project here. UNSAVED means something changed since the last save |
| ● *Local service* | Green when the interface is connected to the Roop 3D server on your computer |
| Hardware pill (for example *TRT · 2.3 GB / 12.0 GB · IDLE*) | Which accelerator is in use, video memory used / total, and whether a render or preview is running now. Open it for the active provider, GPU memory use, pipeline throughput and a bottleneck hint |
| **− 100% +** | Interface size (also Ctrl + / Ctrl − / Ctrl 0) |
| Theme | 37 built-in themes (dark, AMOLED black and light). The interface keeps text readable in every one of them |
| **Studio layout / Classic layout** | Four-panel studio, or the older stacked layout |
| **Projects**, **Save project**, **Save copy** | The project library, save, and save as a new project |

Under the header: the **run strip** (*Step 1 · Pre-Scan, Step 2 · Range Lock, Step 3 · Render & Mux*, shown while a render runs), the **screen
navigation** (*Home, Face Swap, Batch Matrix, Processing, Face Manager, Editor, Outputs, History, Settings*) and, on the Face Swap screen, the
**Next-step bar**: one line saying what to do next, with one button that does it.

Messages appear in a bar under the navigation. Errors say what happened and what to try first; open *What happened, and every step* for the
full explanation.

## 2. The workflow in seven steps

1. **Name your project** (recommended). Type a name in the header. Save any time with **Ctrl+S** or **Save project**.
2. **Import the target video.** In the *Target* panel press **Import target video** and choose a file (`.mp4 .mov .mkv .avi .webm .m4v .ts`).
   The file is copied into Roop 3D's own workspace. The panel shows its size, frame rate and length.
3. **Prepare a source FaceSet.** A FaceSet is the identity you want to put into the video: a handful of clear photos of one person from different
   angles. Import a `.fsz` archive, build one in **Face Manager**, or scrub the video, right-click a face and choose **Use Face From Frame**.
   Details: [FaceSet guide](FACESET_GUIDE.md).
4. **Inspect the target faces.** Press **Inspect target faces**. Roop 3D finds every face and follows each person through the video, giving each
   a numbered *track* (Face 0, Face 1, …). A time budget stops very long scans and tells you if the scan was truncated.
5. **Choose who becomes whom.** Drag a FaceSet card onto a track (or onto a face in the video), or use the track's *Source identity* menu. Leave
   *Keep original* for people who must not change. The track's *Role* can be **Swap**, **Reference guide** (keep, used as a guide) or **Exclude**.
6. **Check the live preview.** Move the playhead to a hard moment (a head turn, a hand over the face, strong light). Use **Compare** to wipe
   between original and swap. Adjust the Geometry and Masking panels in the Inspector if edges or angles look wrong.
7. **Choose an export preset and render.** Pick an **Export preset** next to the trim bar ([Export guide](EXPORT_GUIDE.md)) and press **Queue
   render**. The *Pre-Flight Render Linter* checks the set-up first and offers one-click fixes. Watch the queue; when the job reaches
   **COMPLETED**, use *Inspect output* or open **Outputs**.

## 3. The studio panels

### 01 · Target
* **Import target video**, the loaded file's details, and a **frame navigator** (first / previous / frame number / next / last and the timecode).
* **Detected target faces**: one card per track with thumbnail, frame and time range, yaw range, visibility, and tracking confidence. Click a card
  (or a face in the video, or a lane in the timeline) to select that person; the Inspector shows their details.
* **Identity lock** (below the track list): lock the current assignments for a render, so the render refuses to start if the tracks have drifted.
* *Selecting target faces: how does this work?* explains roles and tips in place.

### 02 · Identity
* **Import .fsz**, **Folder**, **Fix Thumbs**, refresh, and a search box for your FaceSets.
* Select a FaceSet to see its **pose coverage** grid (yaw −75° … +75° by up / level / down), a plain-language reading of it, the **source photo
  checklist**, its views, the **3D identity asset** (build and view) and **Identity B (animated Slerp)** to morph between two identities.
* Drag a FaceSet card onto a track to assign it.

### 03 · Preview
* **Snapshots A–D** store parameter sets; **Left / Right** choose what to compare; **Split Comparison Wipe**, **Restore Slot Settings**.
* The **pipeline bar** shows the eight stages (*Input, 3D Pose & TPS Grid, 3D Normal Map, 3D Relighting (SH), Raw Swap, Occlusion & Seam Mask,
  Restorer, Final Output*). **Solo** shows a stage on its own; **Bypass** skips an optional stage.
* The **viewport toolbar**: *Fit / zoom (1:1, 2:1, 4:1)*, *Grid*, *Overlays*, *Landmarks*, *Pose*, *Compare*, *Live swap*, *PNG* (save the frame),
  *Snapshot*, *Float* (floating preview), *PiP* (picture in picture) and full screen.
* Left pane: the source frame with face overlays. Right pane: the swapped frame, or the output of a finished render.
* **Live swap** produces a real swapped frame for the playhead position without queueing a render. While a render or a model build owns the GPU the
  preview says it is busy and retries by itself; that is not an error.
* Other tools in this row: **Use Face From This Frame**, **Use As Source**, **Edit Mask** (paint a manual mask), **3×3 Sweep** (compare nine
  real previews of two parameters), **Undo / Redo / Edit History**.

### 04 · Inspector
* **Target identity**: facts about the selected track and its source/role.
* **Models & engine**: the swap engine (with a recommendation and the reason, and why unavailable engines cannot be chosen), the hardware profile in
  words, and the restoration model (GPEN or CodeFormer). See [Engine comparison](ENGINE_COMPARISON.md).
* **3D settings**: *Render mode* (Standard identity swap · 2.5D Pose-Aware Swap · Full 3D Volumetric Re-enactment) and the 3D viewer, which shows your FaceSet as a textured 3D head you can orbit (see [3D viewport](STAGE04_3D_VIEWPORT.md)).
* **Pose, expression, masking and restoration parameters**: sliders with the recommended range for each, such as *2.5D → 3D pose blend*, *Pose
  frontalization*, *TPS expression weight*, *Swap weight*, *Mask erosion*, *Mask feather*, *Colour match*, *Restorer blend*, *CodeFormer fidelity*.
* **Colour & lighting**: colour transfer (LAB statistics or stabilised histograms) and 3D relighting with a light-probe sphere.
* **Processing**: the same settings as Settings → Workstation Preferences.

### Timeline (bottom)
Transport (play, step, speed), **Snap**, **Loop**, **Chapter** markers, per-track **lanes** (visible ranges, missing ranges, scene cuts, render
progress), and the **trim bar**: **[ In** and **Out ]** set the export range at the playhead; **Reset Range** clears it. The pill shows the length
in seconds and frames. Only the range between In and Out is rendered.

## 4. Rendering, the queue and checkpoints

* **Queue render** saves the project, adds a job to the **persistent queue** and starts it when a slot is free (one job at a time by default).
* Jobs live on disk. Closing the browser tab does not stop a render. If Roop 3D itself stops, the job comes back **Interrupted** with its checkpoint.
* Job actions: **Pause**, **Resume**, **Cancel**, **Retry** (failed), **Duplicate**, **Inspect output**, download the video and its manifest.
* **Resume replays the analysis from the first frame** and skips only re-saving finished frames, so a render that stopped at minute 55 of 60
  needs about 55 minutes to get back. Resume only works on an unchanged job; to change anything, queue a new render.
* Tick *Include selected paused/interrupted checkpoint when saving a project* to keep the checkpoint with the project.
* States: Waiting, Preparing, Rendering, Pausing, Paused, Cancelling, Cancelled, Completed, Failed, Interrupted. The *Guide → Queue & recovery*
  tab explains each and what to do.

## 5. Projects

A project (`.r3dp`) stores references and settings, **never raw media**: the target video, inspection, FaceSet assignments, parameters, curves,
mask, export options, trim range, preview position and (optionally) a checkpoint. Opening one (**Projects → Verify & open**) re-verifies the video,
FaceSet and model hashes first and restores the settings. **Import .r3dp** and **Export** move project files between computers; the videos and
models must exist on the other machine too.

## 6. The other screens

| Screen | Purpose |
|---|---|
| **Home** | Dashboard, your workflow checklist, recent projects |
| **Batch Matrix** | Render many combinations at once: one source to many targets, grouped targets, a per-file matrix grid, or parameter recipes |
| **Processing** | Live per-stage timings, the bottleneck flamegraph, the GPU readout with **Purge VRAM** (unload idle cached models) and the worker monitor |
| **Face Manager** | Build FaceSets from photos (3D Studio) and manage them (FaceSet Library) |
| **Editor** | Post-production: compositing passes, FCPXML / EDL / OTIO timelines, LUT and grain finishing, provenance ([Export guide](EXPORT_GUIDE.md)) |
| **Outputs** | Finished renders with a metadata inspector |
| **History** | Project checkpoints and history, with verification status |
| **Settings** | Workstation preferences (Compute, 3D Spatial, Pipeline, Finishing, UX & Diagnostics, Advanced), themes, the keyboard-shortcut remapper, the diagnostics suite, the Model Hub and the Environment & Offline Manager |

## 7. Keyboard

Everything can be reached with **Tab** / **Shift+Tab**; **Enter** or **Space** activates; **Esc** closes a dialog and returns focus to what opened
it. The first Tab stop is *Skip to main content*. In tab lists (the Guide) the arrow keys, **Home** and **End** move between tabs.

Default shortcuts (all can be changed or switched off in Settings → Workstation Preferences → *Keyboard shortcut remapper*; they pause while you
type in a field or have a dialog open):

| Group | Action | Key |
|---|---|---|
| Transport | Play / pause · previous frame · next frame | Space · ← · → |
| Range | Mark in · mark out · jump to in · jump to out · reset range | I · O · Shift+I · Shift+O · Alt+R |
| Canvas | Toggle split wipe · toggle diff map | W · D |
| Canvas | Solo pipeline stage 1–8 (Input … Final Output) | 1 2 3 4 5 6 7 8 |
| Canvas | Solo selected node · bypass selected node · parameter sweep | S · B · Shift+P |
| Faces | Extract face from frame · capture snapshots A, B, C, D | E · Shift+A, Shift+B, Shift+C, Shift+D |
| Workspace | Floating preview · picture-in-picture | Shift+F · Shift+G |
| Workspace | Save project · undo · redo | Ctrl+S · Ctrl+Z · Ctrl+Y |

## 8. Display and accessibility

* **Interface size**: the header **− 100% +** buttons or Ctrl + / Ctrl − / Ctrl 0 (50% to 160%). Layouts reflow instead of scrolling sideways, including at
  200% zoom.
* **Reduce motion** (Guide → *Display & keyboard*): *Follow my system setting*, *Always reduce* or *Never reduce*. Reduced motion stops pulsing
  indicators, fades, the idle turntable and eased camera in the 3D viewers, and smooth scrolling. Video playback is not affected.
* **Contrast**: after every theme change Roop 3D measures the text colours it uses and adjusts any that fall below 4.5:1 against the surfaces behind them.
* **Screen readers**: controls have names, dialogs are labelled and trap focus, status and error messages are announced, progress bars expose their values.
* **Tablet and small windows**: the preview keeps its full resolution (the interface never asks the server for a smaller picture); only the space it
  occupies changes. On tablets the frame comes first and the tools sit below it.

## 9. Where to look when something is wrong

[Common problems and solutions](COMMON_PROBLEMS.md) · [GPU troubleshooting](GPU_TROUBLESHOOTING.md) · [Performance guide](PERFORMANCE_GUIDE.md) ·
[Offline model installation](OFFLINE_MODELS.md)
