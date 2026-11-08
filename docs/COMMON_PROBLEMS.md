# Common problems and solutions

Start with *what you see*. Every error the app shows has a code (for example `DISK_FULL`); the second half of this page is the full reference,
and it is the same text the app shows under *What happened, and every step*.

## What you see

### "I don't know what to do next"
Press **Guide** in the header (or read the **Next-step bar** under the navigation on the Face Swap screen). It shows which of the six steps are done
and what "done" looks like for the next one. The Home screen has the same checklist.

### The interface loads but buttons are greyed out
The page is still connecting to the local service. The *Local service* dot in the header turns green when it is ready, and the Next-step button
explains why it cannot run yet. Starting the service can take a few seconds (the first hardware check is slow). See
[the service is offline](#service-offline) if it never becomes ready.

### "No faces were detected"
Try a clip or section where the face is clear and large, then **Re-inspect faces**. Faces smaller than a few dozen pixels, or turned far away from
the camera, may not be found.

### Inspecting or rendering stops with `LICENSE_UNVERIFIED`
A model has not completed its licence and safety review, so Roop 3D refuses to run it. Retrying cannot help. Read
[Offline models §4](OFFLINE_MODELS.md#4-model-review-why-a-verified-file-can-still-be-refused).

### The preview shows "busy" or is slow
A render (or a model build) owns the GPU. The preview retries by itself; this is not an error. See the [Performance guide](PERFORMANCE_GUIDE.md).

### The swapped face looks wrong
* When the person turns their head: the FaceSet probably lacks that angle. Open *02 · Identity*, select the FaceSet and read the coverage sentence
  ([FaceSet guide](FACESET_GUIDE.md)). Try the **3D pose** render mode.
* Soft or smeared: turn on a restoration model whose weights are verified ([Engine comparison](ENGINE_COMPARISON.md)).
* Edges or colour: adjust *Mask erosion / feather* and *Colour match* in the Inspector; use **Solo** on *Occlusion & Seam Mask* to see the mask.
* A hand or hair covering the face: use **Edit Mask** to paint a manual mask for that frame.

### The render failed, was paused or says *Interrupted*
Open the queue and read the message under the job (it has a code and a first step). **Paused** or **Interrupted**: press **Resume**. **Failed**: fix
the cause, then **Retry**; if it keeps failing press **Duplicate** for a clean run. Resume replays the analysis from the first frame, so allow about
the time already spent. See [the queue and checkpoints](#queue).

### I closed the window or the computer restarted
Start Roop 3D again. Your projects, FaceSets and queue are on disk. Open **Projects → Verify & open**, then resume the job from the queue.

### The screen looks wrong, a panel is stuck, or text is too small
Reload the page (F5); projects and the queue reload from disk. Use **Classic layout** if a studio panel misbehaves. Ctrl + / Ctrl − change the
interface size and Ctrl 0 resets it. **Guide → Display & keyboard** has the motion setting.

### A file will not import
Videos must be `.mp4 .mov .mkv .avi .webm .m4v` or `.ts`; FaceSets are `.fsz`; photos `.jpg`/`.png`. If a video will not open, re-encode it to H.264 MP4
with a tool such as HandBrake or FFmpeg and import that.

### My output is in an unexpected place or format
Finished videos are in `app/roop_3d/.runtime/outputs` and listed on the **Outputs** screen. The format comes from the **Export preset**
([Export guide](EXPORT_GUIDE.md)).

## Where to look

* The message bar and the queue: every error shows its code and recovery steps.
* **Settings → Diagnostics Suite** and **Settings → Environment & Offline Manager**: live checks.
* The `logs/` folder in the app folder: launcher logs (`logs/api/`, `logs/shell/`); the newest file is `latest`.
* An error's *request* ID (shown next to its code) identifies the matching line in the server's log.

## Error reference

Every row below is generated from the same table the app uses to explain errors.

### The app will not open, or says the service is offline

<a id="service-offline"></a>

| Code | What it means | What to do |
|---|---|---|
| `SERVICE_OFFLINE` | **The local service is not reachable.** This page cannot talk to the Roop 3D server on your computer. It may have been stopped, or it is still starting. | 1. In Pinokio, open Roop 3D and press Start (or wait until the sidebar says “Open Roop 3D”).<br>2. Press Reconnect in this message.<br>3. If it keeps failing, open the terminal tab in Pinokio and look for PORT_IN_USE or a Python error. |
| `PORT_IN_USE` | **The service port is already taken.** Another program is listening on the port Roop 3D wants. Roop 3D never closes other programs. | 1. Close the other program, or set ROOP3D_PORT to a free port and start again. |

### Videos and files

<a id="media"></a>

| Code | What it means | What to do |
|---|---|---|
| `MEDIA_NOT_FOUND` | **The video file was not found.** The target video was moved, renamed or deleted after it was imported. | 1. Import the video again with “Import target video”.<br>2. If you opened a saved project, put the file back where it was or import it again and re-assign the faces. |
| `MEDIA_PATH_DENIED` | **This file location is not allowed.** Roop 3D only reads media that was imported into its own workspace. | 1. Use “Import target video” instead of typing a path. The file is copied into the workspace. |
| `VIDEO_CONTAINER_INVALID` | **The video could not be opened.** The file is damaged or uses a container Roop 3D cannot read. | 1. Try another copy of the video.<br>2. Re-encode it to H.264 MP4 with a tool such as HandBrake or FFmpeg and import that file. |
| `VIDEO_DIMENSIONS_DENIED` | **The video is too large.** The frame size is outside the supported range. | 1. Scale the video down (for example to 1080p) and import it again. |
| `UPLOAD_INCOMPLETE` | **The upload did not finish.** The file transfer was interrupted before every part arrived. | 1. Import the file again. Large files can take a while; keep this tab open. |
| `UPLOAD_FORMAT_INVALID` | **This file type is not supported.** Videos must be .mp4, .mov, .mkv, .avi, .webm, .m4v or .ts. FaceSets are .fsz archives; photos are .jpg/.png. | 1. Convert the file or choose a supported one. |
| `UPLOAD_NAME_INVALID` | **The file name is not accepted.** The name has characters Roop 3D refuses for safety. | 1. Rename the file with letters, numbers, dashes and underscores, then import it again. |
| `PREVIEW_FRAME_MISSING` | **That frame cannot be shown.** The requested frame is past the end of the clip. | 1. Move the playhead inside the clip. |
| `FRAME_OUT_OF_RANGE` | **That frame is outside the video.** The frame number is larger than the clip. | 1. Choose a frame inside the clip. |

### Target faces and scans

<a id="target-faces"></a>

| Code | What it means | What to do |
|---|---|---|
| `INSPECTION_NOT_FOUND` | **The face scan is missing.** The saved scan for this video is gone (cleared cache or moved workspace). | 1. Press “Inspect target faces” again, then re-assign the faces. |
| `INSPECTION_TARGET_MISMATCH` | **The scan belongs to a different video.** The video changed after it was scanned. | 1. Press “Re-inspect faces” for the current video. |
| `TRACK_NOT_FOUND` | **That face track is not in this scan.** The scan was repeated and the tracks were renumbered. | 1. Select a face from the Target panel again. |

### FaceSets

<a id="facesets"></a>

| Code | What it means | What to do |
|---|---|---|
| `FACESET_NOT_FOUND` | **That FaceSet no longer exists.** It was deleted, or this project refers to a FaceSet from another machine. | 1. Open the FaceSet library and import or rebuild the FaceSet, then assign it again. |
| `FACESET_EMPTY` | **The FaceSet has no usable faces.** No face could be found in the images you gave it. | 1. Use sharp, front-facing photos where the whole face is visible. See “Source photo checklist” in the FaceSet panel. |
| `FACESET_ARCHIVE_INVALID` | **The FaceSet archive is not valid.** The .fsz file is damaged or was not made by Roop 3D. | 1. Rebuild it in Face Manager, or export it again from the machine that made it. |
| `FACESET_SECURITY_VIOLATION` | **The FaceSet archive was rejected.** The archive contains paths or files that are not allowed. | 1. Only import .fsz files you made yourself or trust. Rebuild the FaceSet from photos instead. |
| `IDENTITY_GUARD_VIOLATION` | **The identity guard stopped a swap.** The source face and the target pose are too different to swap safely. | 1. Add FaceSet views that cover this pose (see the 3D coverage view in the FaceSet panel). |

### Models and licences

<a id="models"></a>

| Code | What it means | What to do |
|---|---|---|
| `LICENSE_UNVERIFIED` | **A model has not passed its licence and safety review.** Roop 3D will only run a model whose licence, source, security check and compatibility are recorded as reviewed. This model’s record is incomplete, so it is refused. Trying again cannot change that. | 1. Open Settings → Models & Engines Manager and read the model’s status and licence notice.<br>2. The model’s review record has to be completed by whoever maintains this Roop 3D installation. See docs/OFFLINE_MODELS.md, “Model review”.<br>3. There is no setting that skips this check; it exists so results only come from models whose terms are known. |
| `MODEL_INCOMPATIBLE` | **A model does not fit this setup.** The model is installed but its recorded requirements (dependencies, hardware, precision or security review) are not met or not known. | 1. Open Settings → Environment & Offline Manager and press Verify Offline Health.<br>2. Open Models & Engines Manager to see which requirement is listed for this model. See docs/OFFLINE_MODELS.md, “Model review”. |
| `LICENSE_ACCEPTANCE_REQUIRED` | **This model’s licence must be accepted first.** The model carries terms you have to accept before it can be used. | 1. Open Settings → Models & Engines Manager, read the licence notice and accept it there. |
| `MODEL_BLOCKED` | **This model is blocked.** Its licence does not allow it to be used in Roop 3D. | 1. Choose another engine or model. Blocked models cannot be enabled. |
| `COMMERCIAL_USE_NOT_VERIFIED` | **Commercial use is not confirmed for this model.** You asked for a commercial use that the model’s licence does not clearly allow. | 1. Read the licence notice in Model Hub and use the output only for what it permits. |
| `MODEL_NOT_INSTALLED` | **A required model file is missing.** Roop 3D works offline and needs its model files on disk. This one is not installed. | 1. Open Settings → Environment & Offline Manager and press Verify Offline Health to see which files are missing.<br>2. Import the model bundle with the command shown there. See docs/OFFLINE_MODELS.md. |
| `MODEL_DRIFT` | **A model file changed.** The file on disk no longer matches the hash Roop 3D recorded for it. | 1. Restore the original file, or re-import the model bundle. Roop 3D refuses to guess. |
| `MODEL_CHANGED` | **A model file changed since this was saved.** The project or job was made with a different model file. | 1. Re-import the original model, or start a new render with the current model. |
| `RESTORER_UNAVAILABLE` | **Face restoration is switched off.** The restoration model (GPEN or CodeFormer) is not installed, so restoration stays off. | 1. Import the restoration weights (docs/OFFLINE_MODELS.md) or leave restoration off. |

### Engines

<a id="engines"></a>

| Code | What it means | What to do |
|---|---|---|
| `ENGINE_UNAVAILABLE` | **This engine cannot run here.** The selected engine has no installed, verified adapter on this machine. | 1. Choose “Roop 3D Reference ONNX” in Inspector → Models & engine. |

### GPU and memory

<a id="gpu"></a>

| Code | What it means | What to do |
|---|---|---|
| `PROVIDER_UNAVAILABLE` | **The GPU runtime is not available.** The accelerator that was requested (CUDA or TensorRT) cannot be used right now. | 1. Open Settings → Environment & Offline Manager and re-run the preflight.<br>2. In Settings → Workstation Preferences → Compute set Preferred provider to Auto, or Hardware profile to cpu, and try again. See docs/GPU_TROUBLESHOOTING.md. |
| `TRT_UNAVAILABLE` | **TensorRT is not available.** The TensorRT engine could not be built or loaded on this machine. | 1. In Settings → Workstation Preferences → Compute set Preferred provider to Auto or CUDA. Rendering still works without TensorRT, just slower. See docs/GPU_TROUBLESHOOTING.md. |
| `PIPELINE_OOM` | **The graphics card ran out of memory.** The render needed more video memory than is free. Roop 3D already retried once on the CPU. | 1. Close other programs that use the GPU (games, browsers with video).<br>2. In Settings → Workstation Preferences → Compute lower the Hardware profile (12gb → 6gb → cpu) or reduce Stream batch size, then press Retry. |

### The render queue and checkpoints

<a id="queue"></a>

| Code | What it means | What to do |
|---|---|---|
| `JOB_MODELS_CHANGED` | **Models changed since the job paused.** A paused render cannot continue on different model files without changing the result. | 1. Restore the original models and Resume, or press Duplicate to start a new render from frame 0. |
| `OUTPUT_EXISTS` | **An unfinished output is in the way.** A previous render left a file that has no completion record. | 1. Roop 3D quarantines such files on the next start. Restart Roop 3D, then press Retry. |
| `JOB_CHECKPOINT_INVALID` | **The checkpoint cannot be used.** The saved progress for this render is damaged or incomplete. | 1. Press Duplicate to start a new render of the same settings from frame 0. |
| `JOB_CHECKPOINT_MISMATCH` | **The checkpoint does not match this render.** The settings or inputs changed after the checkpoint was saved. | 1. Resume only unchanged jobs. For changed settings, queue a new render. |
| `JOB_SOURCE_CHANGED` | **The source video changed.** The target file is not the one this job started with. | 1. Put the original file back, or queue a new render. |
| `JOB_STALLED` | **The render stopped making progress.** No frames were written for a long time. | 1. Pause and Resume the job. If it stalls again, check GPU troubleshooting and the Processing screen. |
| `JOB_NOT_FOUND` | **That job no longer exists.** It was removed from the queue history. | 1. Refresh the queue; queue a new render if you still need the output. |
| `JOB_REVISION_CONFLICT` | **The project was saved elsewhere.** Another tab changed this project after you opened it. | 1. Reload the page, reopen the project and apply your edits again. |

### Output, disk and export

<a id="export"></a>

| Code | What it means | What to do |
|---|---|---|
| `AUDIO_STREAM_MISSING` | **The audio could not be carried over.** The audio policy asks for the original audio but the file has none or it is unreadable. | 1. Open Settings → Workstation Preferences → Finishing and change Audio policy so it does not require audio, or import a video that has an audio track. |
| `DISK_FULL` | **The disk is full.** There is not enough free space to write the output. | 1. Free space on the drive that holds the output folder (default: roop_3d/.runtime/outputs inside the app folder).<br>2. The job is kept as Interrupted with its checkpoint. Press Resume once there is space. |
| `DISK_SPACE_LOW` | **Low disk space.** Free space is below the safety margin for this render. | 1. Free some space on the drive that holds the app folder (the output location is an offline setting; see docs/EXPORT_GUIDE.md), then press Retry. |
| `FILE_ACCESS_DENIED` | **A file is locked by another program.** Windows refused access to a file Roop 3D was writing, often because a player, antivirus or sync tool has it open. | 1. Close any program that has the output or its folder open, pause cloud sync for the folder, then press Resume. |
| `ENCODER_TERMINATED` | **The video encoder stopped.** The encoding process ended unexpectedly (often antivirus or low memory). | 1. Close other heavy programs, check that your security software is not stopping Roop 3D, then press Resume. |
| `OUTPUT_FIDELITY` | **The finished video failed its quality check.** The encoded file differs from what was rendered by more than the preset allows. | 1. Render again with a higher-quality preset (Export preset → High-quality MP4 or Lossless). |
| `OUTPUT_FRAME_LOSS` | **Frames were lost while writing the video.** The output has fewer frames than the source range. | 1. Press Retry. If it repeats, free disk space and close other heavy programs. |

### Performance

<a id="performance"></a>

| Code | What it means | What to do |
|---|---|---|
| `PREVIEW_BUSY` | **The engine is busy.** The GPU is being used by a render or a model build. This is not a failure. | 1. Wait a few seconds; the live preview retries by itself. |

### Settings

<a id="settings"></a>

| Code | What it means | What to do |
|---|---|---|
| `INVALID_SETTINGS` | **A setting has an invalid value.** Roop 3D refused a configuration value. | 1. Open Settings → Workstation Preferences and set the field back to its default. |
