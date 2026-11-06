import type { WorkstationParameters } from './pipeline';

export type RenderModeId = NonNullable<WorkstationParameters['render_mode']>;
/** The two tabs of the 3D viewer: the single-view estimate and the stitched multi-view head. */
export type PreviewTab = 'estimated_2_5d' | 'reconstruction';

export interface RenderModeInfo {
  id: RenderModeId;
  label: string;
  /** What the pipeline does with the source FaceSet in this mode. */
  hint: string;
  /** Which 3D viewer tab shows the geometry this mode works from. */
  preview: PreviewTab;
  /** Short badge shown on that tab in the viewer. */
  previewNote: string;
}

export const PREVIEW_TAB_LABEL: Record<PreviewTab, string> = { estimated_2_5d: 'Estimated 2.5D', reconstruction: 'Full 3D Reconstructed' };

/** `id` values are the saved `render_mode` parameter values; only the labels and hints are user-facing copy. */
export const RENDER_MODES: readonly RenderModeInfo[] = [
  { id: 'standard', label: 'Standard identity swap',
    hint: 'The proven 2D identity swap path. It does not use 3D geometry, so the viewer is only a preview of what your FaceSet contains.',
    preview: 'estimated_2_5d', previewNote: 'preview only' },
  { id: 'pose_3d', label: '2.5D Pose-Aware Swap',
    hint: "Turns the swap to match each target frame's head pose using the source FaceSet's own 3D geometry (the Estimated 2.5D tab). Falls back to standard per frame when geometry is unreliable.",
    preview: 'estimated_2_5d', previewNote: 'render geometry' },
  { id: 'reenactment_3d', label: 'Full 3D Volumetric Re-enactment',
    hint: "Drives the full multi-view head (the Full 3D Reconstructed tab) with the target's pose and expression. Falls back to standard per frame when geometry is unreliable.",
    preview: 'reconstruction', previewNote: 'render geometry' },
] as const;

export function renderModeInfo(id: string | undefined | null): RenderModeInfo {
  return RENDER_MODES.find(mode => mode.id === id) ?? RENDER_MODES[0];
}
