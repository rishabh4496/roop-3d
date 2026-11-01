export type TourStep = {
  id: 'source' | 'target' | 'preview' | 'render';
  route: string;
  /** CSS selector of the element the coach mark highlights; the card centres itself when it is not on screen. */
  target: string;
  title: string;
  summary: string;
  points: string[];
};

export const TOUR_STORAGE_KEY = 'roop3d-tour-seen-v1';

export const TOUR_STEPS: TourStep[] = [
  {
    id: 'source', route: '#/faceswap', target: '[data-tour="source"]',
    title: 'Select or extract a source face',
    summary: 'The identity you want to put into the video.',
    points: [
      'Upload a photo, or load a .fsz FaceSet built with the 3×3 pose matrix, from the Asset Library.',
      'Or scrub the target video, right-click any face on the canvas and choose “Use Face From Frame”.',
      'The new FaceSet becomes the active source immediately and is saved to your library.',
    ],
  },
  {
    id: 'target', route: '#/faceswap', target: '[data-tour="tracks"]',
    title: 'Target selection & 3D pose tracking',
    summary: 'Decide which faces change.',
    points: [
      'Inspect the target: faces are tracked and grouped into one identity cluster per person.',
      'Give a cluster a source to swap it, or set it to Reference / Exclude to leave it alone. Swap every cluster for “all faces”, or only one for a selected face.',
      'The canvas HUD draws the 5-point ArcFace landmarks and a live Yaw / Pitch / Roll badge on every face.',
    ],
  },
  {
    id: 'preview', route: '#/faceswap', target: '[data-tour="pipeline-bar"]',
    title: 'Live preview & buffer soloing',
    summary: 'Dial the look in before you render.',
    points: [
      'Click a node in the Pipeline Node Flow Bar to solo that buffer, such as the Mask or the Restorer output.',
      'Drag the Split Comparison slider to wipe between the original and the swap.',
      'Run a 3×3 Parameter Sweep to compare nine real previews and pick TPS expression or CodeFormer fidelity values.',
    ],
  },
  {
    id: 'render', route: '#/faceswap', target: '[data-tour="run-pill"]',
    title: 'Hardware-accelerated render queue',
    summary: 'Render once the Pre-Flight Linter is happy.',
    points: [
      'Queue Render runs the Pre-Flight Render Linter first: TensorRT models, pose over-correction, UltraMax and VRAM / thread checks, many with a one-click fix.',
      'Jobs go to the persistent backend queue, so the render keeps going if you close this tab, and projects resume from their checkpoint after a restart.',
      'Watch the 3-stage run pill: Pre-Scan, Range Lock, then Render & Mux. Open #/processing for the per-frame bottleneck flamegraph.',
    ],
  },
];

export function markTourSeen(): void {
  try { localStorage.setItem(TOUR_STORAGE_KEY, '1'); } catch { /* A blocked store only means the hint shows again. */ }
}

export function tourSeen(): boolean {
  try { return localStorage.getItem(TOUR_STORAGE_KEY) === '1'; } catch { return false; }
}
