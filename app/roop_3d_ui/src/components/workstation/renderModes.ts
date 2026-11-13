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
    hint: "Projects the swap through the source FaceSet's own 3D identity asset, so it follows the target's head pose; the pixels still come from the selected neural swapper. Build the asset first: without a matching one the generic canonical head proxy is used instead.",
    preview: 'reconstruction', previewNote: 'render geometry' },
  { id: 'reenactment_3d', label: '3D Expression Transfer',
    hint: "Renders the source FaceSet's 3D identity asset (its UV texture) with the target's measured expression and cross-fades it into the swap. Build the asset first: without a matching one the plain 2D swap is kept.",
    preview: 'reconstruction', previewNote: 'render geometry' },
] as const;

export function renderModeInfo(id: string | undefined | null): RenderModeInfo {
  return RENDER_MODES.find(mode => mode.id === id) ?? RENDER_MODES[0];
}
