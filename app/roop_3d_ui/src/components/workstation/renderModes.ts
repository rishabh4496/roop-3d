/**
 * roop-3d has one render mode: the reconstructed full 3D head, drawn through a perspective camera and shaded by the selected
 * 3D shader mode. The former 2D swap and estimated 2.5D modes were removed; saved sessions that still name them load as this mode.
 */
export type RenderModeId = 'full_3d_head';
/** The single viewer tab: the fitted full-head surface. */
export type PreviewTab = 'reconstruction';

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

export const PREVIEW_TAB_LABEL: Record<PreviewTab, string> = { reconstruction: 'Full 3D Head' };

export const RENDER_MODES: readonly RenderModeInfo[] = [
  { id: 'full_3d_head', label: 'Full 3D Head Reconstruction',
    hint: "Reconstructs the source FaceSet as a complete 3D head (topology engine below), poses it with the target's head and camera, and shades it with the selected 3D shader. Build the identity asset first.",
    preview: 'reconstruction', previewNote: 'render geometry' },
] as const;

export function renderModeInfo(_id?: string | undefined | null): RenderModeInfo {
  return RENDER_MODES[0];
}
