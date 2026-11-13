/**
 * Single source of truth for every explanation the workstation shows: the workflow guide, FaceSet and photo
 * guidance, engine / hardware / preset / queue explanations.
 *
 * Every statement mirrors something the application really does (the cited backend module is named next to
 * the table). When a backend rule changes, change it here and in the matching docs/*.md page; the Stage 19 tests
 * pin the numbers that appear in the user-facing text.
 */
import type { JobState } from '../../contracts';

/* ------------------------------------------------------------------ workflow */
export type WorkflowStepId = 'project' | 'target' | 'source' | 'inspect' | 'assign' | 'preview' | 'render';

export interface WorkflowStep {
  id: WorkflowStepId;
  title: string;
  /** One sentence: what this step achieves. */
  goal: string;
  how: string[];
  /** What a good result looks like, so a new user knows they can move on. */
  check: string;
  route: string;
  action: string;
}

export const WORKFLOW_STEPS: WorkflowStep[] = [
  { id: 'project', title: 'Name your project', goal: 'A project remembers your video, faces, settings and render checkpoint so you can close the app and come back.',
    how: ['Type a name in the project box at the top of the window.', 'You can save at any time with Ctrl+S or “Save project”. Reopen it later from Projects.'], check: 'The badge next to the name says SAVED after you save.', route: '#/faceswap', action: 'Go to the workstation' },
  { id: 'target', title: 'Import the target video', goal: 'The target is the video whose faces will change.',
    how: ['Press “Import target video” in the Target panel and choose a file (.mp4, .mov, .mkv, .avi, .webm, .m4v or .ts).', 'The file is copied into this app’s own workspace. It is never uploaded anywhere.'], check: 'The preview shows a frame and the Target panel lists the size, frame rate and length.', route: '#/faceswap', action: 'Import target video' },
  { id: 'source', title: 'Prepare a source FaceSet', goal: 'A FaceSet is the identity you want to put into the video: a handful of clear photos of one person from different angles.',
    how: ['Import a .fsz FaceSet, or build one from photos in Face Manager.', 'Or scrub the video to a good frame, right-click a face and choose “Use Face From Frame”.', 'Select the FaceSet to see its coverage: which head angles it has and which are missing.'], check: 'The FaceSet shows at least a frontal view plus left and right three-quarter views (coverage “partial” or “full”).', route: '#/facemgr', action: 'Open Face Manager' },
  { id: 'inspect', title: 'Inspect the target faces', goal: 'Roop 3D finds every face in the video and follows each person through the shots.',
    how: ['Press “Inspect target faces”. Long clips take longer; a time budget stops very long scans and says so.', 'Each person becomes one numbered face track (Face 0, Face 1, …) with a thumbnail and a timeline lane.'], check: 'Every person you want to change has a track. Tracks you do not want can be set to Reference or Exclude.', route: '#/faceswap', action: 'Inspect target faces' },
  { id: 'assign', title: 'Choose who becomes whom', goal: 'Tell Roop 3D which FaceSet replaces which face track.',
    how: ['Drag a FaceSet card onto a track (or onto a face in the video), or pick it in the track’s “Source identity” menu.', 'Leave a track on “Keep original” to leave that person untouched.'], check: 'The track shows “Swap with <FaceSet>” and the Queue render button becomes enabled.', route: '#/faceswap', action: 'Assign a FaceSet' },
  { id: 'preview', title: 'Check the live preview', goal: 'Look at real frames before you spend time rendering.',
    how: ['Move the playhead to a hard moment: a turned head, a hand in front of the face, strong light.', 'Use Compare to wipe between original and swap. If edges or angles look wrong, adjust the Geometry and Masking panels in the Inspector.'], check: 'The swapped face looks right in several different frames.', route: '#/faceswap', action: 'Open the preview' },
  { id: 'render', title: 'Pick an export preset and render', goal: 'Render the whole range (or just the part between In and Out) to a video file.',
    how: ['Choose an Export preset (see “Render presets”). Standard MP4 plays everywhere.', 'Press “Queue render”. The Pre-Flight Linter checks the setup first and offers one-click fixes.', 'Watch the queue. A render can be paused, resumed after a restart, retried or duplicated.'], check: 'The job reaches COMPLETED. Use “Inspect output” to play it, or open Outputs.', route: '#/faceswap', action: 'Start the render' },
];

/* ------------------------------------------------------------------ target faces */
export const TARGET_FACE_TUTORIAL = {
  title: 'Selecting target faces',
  steps: [
    'Press “Inspect target faces”. Roop 3D scans the video and groups every face it sees into one track per person.',
    'Click a track card, a lane in the timeline, or a face in the video. The Inspector then shows that person’s details: frames, time range, tracking confidence and head-angle range.',
    'Give the track a source: drag a FaceSet onto it, or use its “Source identity” menu.',
    'Use the track Role for people who must not change: Swap (replace), Reference guide (keep original, used as a guide) or Exclude (ignore completely).',
    'If one person was split into two tracks (a cut or a long occlusion), assign the same FaceSet to both.',
  ],
  tips: [
    'Yaw is left-right head turn, Pitch is up-down, Roll is tilt. A track that turns far to the side needs FaceSet views at that angle.',
    'Low tracking confidence means the face was hard to follow. Check those shots in the live preview before you render.',
  ],
};

/* ------------------------------------------------------------------ FaceSet preparation */
export const FACESET_STEPS = [
  'Collect 5–12 photos of one person. More photos only help when they add a new angle or expression.',
  'Make sure the same person is in every photo. A different person in the set blocks the build (“blocked: inconsistent identity”).',
  'Cover the angles: straight ahead, three-quarter left and three-quarter right are required for full 3D coverage. Left and right profile are recommended.',
  'Import the photos in Face Manager (or load a .fsz archive) and read the coverage grid. Add the angles it lists as missing.',
  'Press “Build 3D identity asset” (canonical head) before using the 3D Facial Warp or 3D Expression Transfer modes: they reuse that exact stored asset and never build it themselves. Without a matching asset 3D Facial Warp uses the generic head proxy and 3D Expression Transfer keeps the plain 2D swap.',
];

/** Each rule is a threshold the FaceSet quality analysis (reconstruction/faceset_quality.py) really applies. */
export const SOURCE_PHOTO_RULES: Array<{ id: string; rule: string; detail: string }> = [
  { id: 'sharp', rule: 'Sharp and in focus', detail: 'Photos with almost no fine detail are rejected as too blurry (sharpness below 0.03 on a 0–1 scale). Motion blur and heavy compression both count.' },
  { id: 'size', rule: 'Face at least 40 pixels wide', detail: 'Smaller faces are rejected. Aim for 256 pixels or more across the face; larger photos give the build more to work with.' },
  { id: 'edge', rule: 'Whole face inside the picture', detail: 'A face with more than 35% of its box outside the image is rejected as cut off by the edge.' },
  { id: 'light', rule: 'Even exposure', detail: 'Photos that are mostly black or burnt-out white are rejected for bad exposure. Avoid hard shadows across one side of the face.' },
  { id: 'clear', rule: 'Nothing in front of the face', detail: 'Hands, hair, masks and sunglasses hide the features the swap needs. Glasses are fine only if the target person wears them too.' },
  { id: 'one', rule: 'Only one person', detail: 'Views that do not match the main person (similarity to the group below 0.30) are dropped; if the set is split between people the build is blocked.' },
  { id: 'unique', rule: 'No near-duplicates', detail: 'Two photos that look the same at the same head angle count once; the extra is marked duplicate.' },
  { id: 'detect', rule: 'A detectable face', detail: 'If no face is found, the photo is not used as an observed view. Landmarks are never invented for a 3D build.' },
];

export const VIEW_REASONS: Record<string, string> = {
  TOO_BLURRY: 'Too blurry. Use a sharper photo.',
  BAD_EXPOSURE: 'Too dark or too bright. Use evenly lit photos.',
  FACE_TOO_SMALL: 'The face is too small (under 40 px wide). Use a closer or larger photo.',
  FACE_TRUNCATED_BY_IMAGE_EDGE: 'The face runs off the edge of the picture. Use a photo with the whole face visible.',
  FACE_HEAVILY_OCCLUDED: 'Something covers much of the face.',
  DUPLICATE_IMAGE: 'Looks the same as another photo at the same angle.',
  IDENTITY_SET_INCONSISTENT: 'The set seems to contain more than one person.',
  IDENTITY_OUTLIER: 'Does not look like the rest of the set. It may be a different person.',
  IMAGE_UNDECODABLE: 'The image file could not be read.',
  LANDMARKS_NOT_OBSERVED: 'No face was detected in this photo.',
  LANDMARKS_OUTSIDE_IMAGE: 'Face landmarks fall outside the picture.',
  LANDMARK_TOPOLOGY_INVALID: 'The detected face points are not in a plausible arrangement.',
  POSE_LANDMARK_MISMATCH: 'The head angle and the face points disagree.',
  POSE_OUT_OF_RANGE: 'The head is turned further than 88° (almost fully sideways or behind).',
};

export const READINESS: Record<string, { label: string; meaning: string; next: string }> = {
  insufficient: { label: 'Not enough usable photos', meaning: 'Too few views passed the quality checks to describe the face.', next: 'Add sharp, front-facing photos.' },
  blocked_inconsistent_identity: { label: 'Blocked: more than one person', meaning: 'The photos do not agree on who this is.', next: 'Remove photos of other people, then rebuild.' },
  front_only: { label: 'Front only', meaning: 'Only straight-on views exist. Swaps work while the person faces the camera but degrade when they turn.', next: 'Add three-quarter left and right photos.' },
  partial_coverage: { label: 'Partial coverage', meaning: 'Some turned angles are covered. Moderate head turns should look right.', next: 'Add the angles listed under “suggested additional angles”.' },
  full_coverage: { label: 'Full coverage', meaning: 'Frontal and both three-quarter views are present.', next: 'Nothing required. Profile views (±75°) are an optional extra.' },
};

/* ------------------------------------------------------------------ engines */
export const ENGINE_STATUS: Record<string, { label: string; meaning: string }> = {
  READY_FOR_INFERENCE: { label: 'Ready', meaning: 'Installed, hash-checked and able to run on this computer.' },
  AVAILABLE: { label: 'Available', meaning: 'Listed locally, but not yet verified for inference.' },
  LICENSE_RESTRICTED: { label: 'Licence restricted', meaning: 'Usable here, but its model licence limits how the results may be used. Read the licence before sharing or selling output.' },
  UNVERIFIED: { label: 'Unverified', meaning: 'No verified local adapter exists, so it cannot be selected.' },
  INCOMPATIBLE: { label: 'Incompatible', meaning: 'It does not fit this pipeline (for example it edits still images, not video).' },
  NOT_INSTALLED: { label: 'Not installed', meaning: 'Its model files are not on this computer. Roop 3D does not download them without your explicit action.' },
  BROKEN: { label: 'Broken', meaning: 'Its files are present but failed verification.' },
};

export interface EngineLike { engine_id: string; name: string; status: string; selectable: boolean; actual_inference_available?: boolean; evidence?: string; purpose?: string }

/** The engine to suggest: the first one that can really render a video here. Never an unavailable engine. */
export function recommendEngine<T extends EngineLike>(engines: T[]): { engine: T | null; reason: string } {
  const usable = engines.filter(item => item.selectable);
  if (!usable.length) return { engine: null, reason: 'No engine can render video on this computer yet. Open Settings → Environment & Offline Manager to see what is missing.' };
  const ready = usable.find(item => item.status === 'READY_FOR_INFERENCE' || item.actual_inference_available) ?? usable[0];
  const licence = ready.status === 'LICENSE_RESTRICTED' ? ' Check its model licence before publishing results.' : '';
  return { engine: ready, reason: `It is the one engine here that renders frame-by-frame video with your original audio and timing.${licence}` };
}

/* ------------------------------------------------------------------ hardware */
export type HardwareProfileId = 'auto' | 'cpu' | '6gb' | '12gb';
export const HARDWARE_PROFILES: Record<HardwareProfileId, { label: string; summary: string; behaviour: string[]; needs: string }> = {
  auto: { label: 'Auto', summary: 'Roop 3D looks at your graphics card and picks cpu, 6gb or 12gb itself.', behaviour: ['The profile it picked is shown as “Resolved”.'], needs: 'Nothing. This is the right choice for almost everyone.' },
  '12gb': { label: '12 GB (high capacity)', summary: 'For GPUs with about 12 GB or more of video memory.', behaviour: ['Keeps models resident in GPU memory for the fastest previews.', 'Batches several faces at once (up to 8).', 'Uses up to about 9.5 GB of GPU memory before it tidies its caches.'], needs: 'An NVIDIA GPU with at least 10 GB of video memory (12 GB or more recommended); a card reporting less than 10,240 MiB in total cannot use it.' },
  '6gb': { label: '6 GB (low memory)', summary: 'For GPUs with about 6 GB of video memory, typically laptops.', behaviour: ['Limits itself to about 4.6 GB of GPU memory.', 'Works one frame stage at a time and unloads models that are not in use.', 'Slower than 12gb, but avoids out-of-memory failures.'], needs: 'Any NVIDIA GPU with at least 5 GB of video memory.' },
  cpu: { label: 'CPU only', summary: 'No GPU is used.', behaviour: ['Runs every model on the processor and encodes in software.', 'Previews are limited to a small size to stay responsive.', 'Renders work but are many times slower.'], needs: 'Nothing. Used automatically when no usable NVIDIA GPU is found.' },
};

export const PROVIDER_TERMS: Array<{ term: string; meaning: string }> = [
  { term: 'TRT', meaning: 'TensorRT: NVIDIA’s fastest runtime. The first run builds an engine for your GPU, which can take minutes; it is then cached.' },
  { term: 'CUDA', meaning: 'NVIDIA GPU runtime without the extra build step. Fast, and the usual fallback when TensorRT is unavailable.' },
  { term: 'CPU', meaning: 'The processor. Always available, much slower.' },
  { term: 'VRAM', meaning: 'Video memory on the graphics card. The “2.3 GB / 12.0 GB” figure is used / total right now.' },
  { term: 'IDLE / RUNNING', meaning: 'Whether a render or preview is using the GPU at this moment.' },
];

/* ------------------------------------------------------------------ render presets */
/** `when` is guidance; the encoder facts come live from GET /api/v1/media/export-profiles. */
export const PRESET_GUIDANCE: Record<string, { when: string; size: string }> = {
  standard_mp4: { when: 'Sharing, previews and anything that must play on any device.', size: 'Small' },
  high_quality_mp4: { when: 'A final copy to publish or upload. Slower to encode than Standard.', size: 'Medium' },
  archival_hevc10: { when: 'A master you want to keep. Very high quality; needs a player with HEVC support.', size: 'Large' },
  mov_workflow: { when: 'Editing in Premiere, Final Cut or DaVinci Resolve (ProRes 422 HQ).', size: 'Very large' },
  lossless_intermediate: { when: 'Further processing where nothing may be lost (FFV1 in MKV).', size: 'Very large' },
  image_sequence: { when: 'Compositing programs that want one image per frame (PNG, TIFF or DPX) plus a manifest.', size: 'Very large' },
  color_preserving: { when: 'Footage with HDR or wide-gamut colour tags that an editor needs preserved (ProRes 4444/422 HQ).', size: 'Very large' },
};
export const PRESET_LIMITS = 'Frames are processed as 8-bit SDR RGB. A 10-bit preset reduces encoding loss, but it cannot add precision that the 8-bit pipeline does not have. HDR footage is tone-mapped to SDR unless you choose Reject.';

/* ------------------------------------------------------------------ queue */
export const JOB_STATE_HELP: Record<JobState, { label: string; meaning: string; action: string }> = {
  queued: { label: 'Waiting', meaning: 'In line. Only max-concurrent-jobs run at once (default 1).', action: 'Wait, or Pause to hold it.' },
  preparing: { label: 'Preparing', meaning: 'Checking inputs and models, and scanning the video.', action: 'Nothing to do.' },
  running: { label: 'Rendering', meaning: 'Frames are being swapped and written.', action: 'Pause if you need the GPU for something else.' },
  pausing: { label: 'Pausing', meaning: 'Finishing the current frame and saving a checkpoint.', action: 'Wait a moment.' },
  paused: { label: 'Paused', meaning: 'Stopped on purpose. A checkpoint of the finished frames is saved.', action: 'Press Resume to continue.' },
  cancelling: { label: 'Cancelling', meaning: 'Stopping and removing unfinished output.', action: 'Wait a moment.' },
  cancelled: { label: 'Cancelled', meaning: 'Stopped by you. No video was published.', action: 'Press Duplicate to run it again.' },
  succeeded: { label: 'Completed', meaning: 'The video was written and verified.', action: 'Press Inspect output, or find it in Outputs.' },
  failed: { label: 'Failed', meaning: 'Something went wrong. The message under the job says what.', action: 'Read the message, fix the cause, then press Retry.' },
  interrupted: { label: 'Interrupted', meaning: 'Roop 3D or the computer stopped while this was rendering. The checkpoint is kept.', action: 'Press Resume after Roop 3D is running again.' },
};

export const QUEUE_GUIDE = [
  'Jobs live on disk. Closing the browser tab does not stop a render. If Roop 3D itself stops (restart, crash, antivirus, power loss) the job comes back as Interrupted with its checkpoint, and you can press Resume.',
  'A checkpoint stores the output frames that were finished, not the tracker state. Resume therefore replays the analysis from the first frame and skips only re-saving the finished frames. A render that stopped at minute 55 of 60 needs about 55 minutes to get back to where it was.',
  'Resume only works on an unchanged job. If you changed the faces, settings or models, queue a new render (Duplicate) instead.',
  'To keep a checkpoint with a project, tick “Include selected paused/interrupted checkpoint when saving a project”, then Save project.',
];

/* ------------------------------------------------------------------ recovery */
export const RECOVERY_STEPS: Array<{ problem: string; steps: string[] }> = [
  { problem: 'The page says the local service is offline', steps: ['In Pinokio press Start on Roop 3D.', 'Press Reconnect in the message bar.', 'Your project and queue are on disk; nothing is lost.'] },
  { problem: 'A render stopped, failed or was interrupted', steps: ['Open the queue and read the message under the job.', 'Interrupted or Paused: press Resume. (It replays the analysis from the start, so allow roughly the time already spent.)', 'Failed: fix the cause named in the message, then press Retry. If Retry keeps failing, press Duplicate for a clean run.'] },
  { problem: 'I closed the window while saving or rendering', steps: ['Start Roop 3D again and open Projects.', 'Reopening verifies the video and model hashes, then restores the settings.', 'Resume the job from the queue.'] },
  { problem: 'The screen looks wrong or a panel is stuck', steps: ['Reload the page (F5). Projects and the queue are reloaded from disk.', 'Try the Classic layout button if a studio panel misbehaves.', 'Reset the interface zoom with Ctrl+0.'] },
  { problem: 'Models are missing or changed', steps: ['Open Settings → Environment & Offline Manager and press Verify Offline Health.', 'Import the model bundle as described there. See docs/OFFLINE_MODELS.md.'] },
];
