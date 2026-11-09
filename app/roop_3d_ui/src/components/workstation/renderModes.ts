import type { WorkstationParameters } from './pipeline';

export type RenderModeId = NonNullable<WorkstationParameters['render_mode']>;
/** The two viewer tabs: an estimated proxy and the fitted facial surface. */
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
  { id: 'standard', label: 'Standard 2D Identity Swap',
    hint: 'Uses the selected neural model for frontal frames. Extreme poses require the fitted facial models for 3D correspondence.',
    preview: 'estimated_2_5d', previewNote: 'preview only' },
  { id: 'pose_3d', label: '3D Facial Warp',
    hint: "Fits the target's facial anatomy and pose, warps real source photographs, then applies the selected neural swapper and UltraMax. Requires the facial model bundle.",
    preview: 'reconstruction', previewNote: 'render geometry' },
  { id: 'reenactment_3d', label: '3D Expression Transfer',
    hint: "Uses fitted target pose and asymmetric expressions to warp real source photographs before neural synthesis. Requires the facial model bundle.",
    preview: 'reconstruction', previewNote: 'render geometry' },
] as const;

export function renderModeInfo(id: string | undefined | null): RenderModeInfo {
  return RENDER_MODES.find(mode => mode.id === id) ?? RENDER_MODES[0];
}
