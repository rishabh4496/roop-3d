/**
 * Plain-language recovery guidance for errors the local service (or a failed render) can report.
 *
 * Every key is a real code the backend raises (`DomainError`, `JobFailure.code`, the middleware envelope) or a
 * well-known browser failure. The wording says what happened and which button or setting to use next; it never
 * names Python modules. `docs/COMMON_PROBLEMS.md` repeats the same entries with more detail.
 */
export interface ErrorGuide {
  title: string;
  cause: string;
  steps: string[];
  /** Section of docs/COMMON_PROBLEMS.md that expands on this. */
  doc?: string;
}

const g = (title: string, cause: string, steps: string[], doc?: string): ErrorGuide => ({ title, cause, steps, doc });

export const ERROR_GUIDES: Record<string, ErrorGuide> = {
  SERVICE_OFFLINE: g('The local service is not reachable', 'This page cannot talk to the Roop 3D server on your computer. It may have been stopped, or it is still starting.',
    ['In Pinokio, open Roop 3D and press Start (or wait until the sidebar says “Open Roop 3D”).', 'Press Reconnect in this message.', 'If it keeps failing, open the terminal tab in Pinokio and look for PORT_IN_USE or a Python error.'], 'service-offline'),
  PORT_IN_USE: g('The service port is already taken', 'Another program is listening on the port Roop 3D wants. Roop 3D never closes other programs.',
    ['Close the other program, or set ROOP3D_PORT to a free port and start again.'], 'service-offline'),
  MEDIA_NOT_FOUND: g('The video file was not found', 'The target video was moved, renamed or deleted after it was imported.',
    ['Import the video again with “Import target video”.', 'If you opened a saved project, put the file back where it was or import it again and re-assign the faces.'], 'media'),
  MEDIA_PATH_DENIED: g('This file location is not allowed', 'Roop 3D only reads media that was imported into its own workspace.',
    ['Use “Import target video” instead of typing a path. The file is copied into the workspace.'], 'media'),
  VIDEO_CONTAINER_INVALID: g('The video could not be opened', 'The file is damaged or uses a container Roop 3D cannot read.',
    ['Try another copy of the video.', 'Re-encode it to H.264 MP4 with a tool such as HandBrake or FFmpeg and import that file.'], 'media'),
  VIDEO_DIMENSIONS_DENIED: g('The video is too large', 'The frame size is outside the supported range.',
    ['Scale the video down (for example to 1080p) and import it again.'], 'media'),
  AUDIO_STREAM_MISSING: g('The audio could not be carried over', 'The audio policy asks for the original audio but the file has none or it is unreadable.',
    ['Open Settings → Workstation Preferences → Finishing and change Audio policy so it does not require audio, or import a video that has an audio track.'], 'export'),
  UPLOAD_INCOMPLETE: g('The upload did not finish', 'The file transfer was interrupted before every part arrived.', ['Import the file again. Large files can take a while; keep this tab open.'], 'media'),
  UPLOAD_FORMAT_INVALID: g('This file type is not supported', 'Videos must be .mp4, .mov, .mkv, .avi, .webm, .m4v or .ts. FaceSets are .fsz archives; photos are .jpg/.png.', ['Convert the file or choose a supported one.'], 'media'),
  UPLOAD_NAME_INVALID: g('The file name is not accepted', 'The name has characters Roop 3D refuses for safety.', ['Rename the file with letters, numbers, dashes and underscores, then import it again.'], 'media'),
  FACESET_NOT_FOUND: g('That FaceSet no longer exists', 'It was deleted, or this project refers to a FaceSet from another machine.', ['Open the FaceSet library and import or rebuild the FaceSet, then assign it again.'], 'facesets'),
  FACESET_EMPTY: g('The FaceSet has no usable faces', 'No face could be found in the images you gave it.', ['Use sharp, front-facing photos where the whole face is visible. See “Source photo checklist” in the FaceSet panel.'], 'facesets'),
  FACESET_ARCHIVE_INVALID: g('The FaceSet archive is not valid', 'The .fsz file is damaged or was not made by Roop 3D.', ['Rebuild it in Face Manager, or export it again from the machine that made it.'], 'facesets'),
  FACESET_SECURITY_VIOLATION: g('The FaceSet archive was rejected', 'The archive contains paths or files that are not allowed.', ['Only import .fsz files you made yourself or trust. Rebuild the FaceSet from photos instead.'], 'facesets'),
  INSPECTION_NOT_FOUND: g('The face scan is missing', 'The saved scan for this video is gone (cleared cache or moved workspace).', ['Press “Inspect target faces” again, then re-assign the faces.'], 'target-faces'),
  INSPECTION_TARGET_MISMATCH: g('The scan belongs to a different video', 'The video changed after it was scanned.', ['Press “Re-inspect faces” for the current video.'], 'target-faces'),
  TRACK_NOT_FOUND: g('That face track is not in this scan', 'The scan was repeated and the tracks were renumbered.', ['Select a face from the Target panel again.'], 'target-faces'),
  LICENSE_UNVERIFIED: g('A model has not passed its licence and safety review', 'Roop 3D will only run a model whose licence, source, security check and compatibility are recorded as reviewed. This model’s record is incomplete, so it is refused. Trying again cannot change that.',
    ['Open Settings → Models & Engines Manager and read the model’s status and licence notice.', 'The model’s review record has to be completed by whoever maintains this Roop 3D installation. See docs/OFFLINE_MODELS.md, “Model review”.', 'There is no setting that skips this check; it exists so results only come from models whose terms are known.'], 'models'),
  MODEL_INCOMPATIBLE: g('A model does not fit this setup', 'The model is installed but its recorded requirements (dependencies, hardware, precision or security review) are not met or not known.',
    ['Open Settings → Environment & Offline Manager and press Verify Offline Health.', 'Open Models & Engines Manager to see which requirement is listed for this model. See docs/OFFLINE_MODELS.md, “Model review”.'], 'models'),
  LICENSE_ACCEPTANCE_REQUIRED: g('This model’s licence must be accepted first', 'The model carries terms you have to accept before it can be used.', ['Open Settings → Models & Engines Manager, read the licence notice and accept it there.'], 'models'),
  MODEL_BLOCKED: g('This model is blocked', 'Its licence does not allow it to be used in Roop 3D.', ['Choose another engine or model. Blocked models cannot be enabled.'], 'models'),
  COMMERCIAL_USE_NOT_VERIFIED: g('Commercial use is not confirmed for this model', 'You asked for a commercial use that the model’s licence does not clearly allow.', ['Read the licence notice in Model Hub and use the output only for what it permits.'], 'models'),
  MODEL_NOT_INSTALLED: g('A required model file is missing', 'Roop 3D works offline and needs its model files on disk. This one is not installed.',
    ['Open Settings → Environment & Offline Manager and press Verify Offline Health to see which files are missing.', 'Import the model bundle with the command shown there. See docs/OFFLINE_MODELS.md.'], 'models'),
  MODEL_DRIFT: g('A model file changed', 'The file on disk no longer matches the hash Roop 3D recorded for it.', ['Restore the original file, or re-import the model bundle. Roop 3D refuses to guess.'], 'models'),
  MODEL_CHANGED: g('A model file changed since this was saved', 'The project or job was made with a different model file.', ['Re-import the original model, or start a new render with the current model.'], 'models'),
  JOB_MODELS_CHANGED: g('Models changed since the job paused', 'A paused render cannot continue on different model files without changing the result.', ['Restore the original models and Resume, or press Duplicate to start a new render from frame 0.'], 'queue'),
  ENGINE_UNAVAILABLE: g('This engine cannot run here', 'The selected engine has no installed, verified adapter on this machine.', ['Choose “Roop 3D Reference ONNX” in Inspector → Models & engine.'], 'engines'),
  PROVIDER_UNAVAILABLE: g('The GPU runtime is not available', 'The accelerator that was requested (CUDA or TensorRT) cannot be used right now.', ['Open Settings → Environment & Offline Manager and re-run the preflight.', 'In Settings → Workstation Preferences → Compute set Preferred provider to Auto, or Hardware profile to cpu, and try again. See docs/GPU_TROUBLESHOOTING.md.'], 'gpu'),
  TRT_UNAVAILABLE: g('TensorRT is not available', 'The TensorRT engine could not be built or loaded on this machine.', ['In Settings → Workstation Preferences → Compute set Preferred provider to Auto or CUDA. Rendering still works without TensorRT, just slower. See docs/GPU_TROUBLESHOOTING.md.'], 'gpu'),
  RESTORER_UNAVAILABLE: g('Face restoration is switched off', 'The restoration model (GPEN or CodeFormer) is not installed, so restoration stays off.', ['Import the restoration weights (docs/OFFLINE_MODELS.md) or leave restoration off.'], 'models'),
  PIPELINE_OOM: g('The graphics card ran out of memory', 'The render needed more video memory than is free. Roop 3D already retried once on the CPU.',
    ['Close other programs that use the GPU (games, browsers with video).', 'In Settings → Workstation Preferences → Compute lower the Hardware profile (12gb → 6gb → cpu) or reduce Stream batch size, then press Retry.'], 'gpu'),
  DISK_FULL: g('The disk is full', 'There is not enough free space to write the output.', ['Free space on the drive that holds the output folder (default: roop_3d/.runtime/outputs inside the app folder).', 'The job is kept as Interrupted with its checkpoint. Press Resume once there is space.'], 'export'),
  DISK_SPACE_LOW: g('Low disk space', 'Free space is below the safety margin for this render.', ['Free some space on the drive that holds the app folder (the output location is an offline setting; see docs/EXPORT_GUIDE.md), then press Retry.'], 'export'),
  FILE_ACCESS_DENIED: g('A file is locked by another program', 'Windows refused access to a file Roop 3D was writing, often because a player, antivirus or sync tool has it open.', ['Close any program that has the output or its folder open, pause cloud sync for the folder, then press Resume.'], 'export'),
  ENCODER_TERMINATED: g('The video encoder stopped', 'The encoding process ended unexpectedly (often antivirus or low memory).', ['Close other heavy programs, check that your security software is not stopping Roop 3D, then press Resume.'], 'export'),
  OUTPUT_EXISTS: g('An unfinished output is in the way', 'A previous render left a file that has no completion record.', ['Roop 3D quarantines such files on the next start. Restart Roop 3D, then press Retry.'], 'queue'),
  JOB_CHECKPOINT_INVALID: g('The checkpoint cannot be used', 'The saved progress for this render is damaged or incomplete.', ['Press Duplicate to start a new render of the same settings from frame 0.'], 'queue'),
  JOB_CHECKPOINT_MISMATCH: g('The checkpoint does not match this render', 'The settings or inputs changed after the checkpoint was saved.', ['Resume only unchanged jobs. For changed settings, queue a new render.'], 'queue'),
  JOB_SOURCE_CHANGED: g('The source video changed', 'The target file is not the one this job started with.', ['Put the original file back, or queue a new render.'], 'queue'),
  JOB_STALLED: g('The render stopped making progress', 'No frames were written for a long time.', ['Pause and Resume the job. If it stalls again, check GPU troubleshooting and the Processing screen.'], 'queue'),
  JOB_NOT_FOUND: g('That job no longer exists', 'It was removed from the queue history.', ['Refresh the queue; queue a new render if you still need the output.'], 'queue'),
  OUTPUT_FIDELITY: g('The finished video failed its quality check', 'The encoded file differs from what was rendered by more than the preset allows.', ['Render again with a higher-quality preset (Export preset → High-quality MP4 or Lossless).'], 'export'),
  OUTPUT_FRAME_LOSS: g('Frames were lost while writing the video', 'The output has fewer frames than the source range.', ['Press Retry. If it repeats, free disk space and close other heavy programs.'], 'export'),
  PREVIEW_BUSY: g('The engine is busy', 'The GPU is being used by a render or a model build. This is not a failure.', ['Wait a few seconds; the live preview retries by itself.'], 'performance'),
  IDENTITY_GUARD_VIOLATION: g('The identity guard stopped a swap', 'The source face and the target pose are too different to swap safely.', ['Add FaceSet views that cover this pose (see the 3D coverage view in the FaceSet panel).'], 'facesets'),
  PREVIEW_FRAME_MISSING: g('That frame cannot be shown', 'The requested frame is past the end of the clip.', ['Move the playhead inside the clip.'], 'media'),
  FRAME_OUT_OF_RANGE: g('That frame is outside the video', 'The frame number is larger than the clip.', ['Choose a frame inside the clip.'], 'media'),
  JOB_REVISION_CONFLICT: g('The project was saved elsewhere', 'Another tab changed this project after you opened it.', ['Reload the page, reopen the project and apply your edits again.'], 'queue'),
  INVALID_SETTINGS: g('A setting has an invalid value', 'Roop 3D refused a configuration value.', ['Open Settings → Workstation Preferences and set the field back to its default.'], 'settings'),
};

const PATTERNS: Array<[RegExp, string]> = [
  [/failed to fetch|networkerror|load failed|connecting|offline|ECONNREFUSED/i, 'SERVICE_OFFLINE'],
  [/local service returned 5\d\d/i, 'SERVICE_OFFLINE'],
  [/out of memory|bad allocation|cuda.*alloc/i, 'PIPELINE_OOM'],
  [/no space left|disk.*full/i, 'DISK_FULL'],
];

const PREFIXES: Array<[string, string]> = [['LICENSE_', 'LICENSE_UNVERIFIED'], ['JOB_CHECKPOINT_', 'JOB_CHECKPOINT_INVALID'], ['FACESET_ARCHIVE_', 'FACESET_ARCHIVE_INVALID'], ['UPLOAD_', 'UPLOAD_INCOMPLETE'],
  ['OUTPUT_', 'OUTPUT_FIDELITY'], ['INSPECTION_', 'INSPECTION_NOT_FOUND'], ['MODEL_', 'MODEL_NOT_INSTALLED']];

/** True when the problem is that the local service cannot be reached (the only case where "Reconnect" helps). */
export const isOfflineError = (input: { code?: string | null; message?: string | null }): boolean => explainError(input) === ERROR_GUIDES.SERVICE_OFFLINE;

/** Best guidance for an error code and/or message, or null when nothing specific is known (the message still shows). */
export function explainError(input: { code?: string | null; message?: string | null }): ErrorGuide | null {
  const code = input.code?.toUpperCase() ?? '';
  if (code && ERROR_GUIDES[code]) return ERROR_GUIDES[code];
  const prefix = PREFIXES.find(([start]) => code.startsWith(start));
  if (prefix) return ERROR_GUIDES[prefix[1]];
  const text = `${input.message ?? ''}`;
  const named = text.match(/\b([A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b/)?.[1];
  if (named && ERROR_GUIDES[named]) return ERROR_GUIDES[named];
  const match = PATTERNS.find(([pattern]) => pattern.test(text));
  return match ? ERROR_GUIDES[match[1]] : null;
}
