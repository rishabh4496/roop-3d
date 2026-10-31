export type WorkstationParameters = { restoration_blend: number; feather_px: number; color_strength: number; pose_strength: number; bypass: ('restorer' | 'color')[] };
export const DEFAULT_PARAMETERS: WorkstationParameters = { restoration_blend: 0, feather_px: 5, color_strength: .65, pose_strength: .35, bypass: [] };
export const SWEEP_PARAMETERS = {
  restoration_blend: { label: 'Restorer blend', min: 0, max: 1, values: [.25, .5, .75] },
  feather_px: { label: 'Mask feather (px)', min: 0, max: 64, values: [0, 5, 10] },
  color_strength: { label: 'Color match strength', min: 0, max: 1, values: [.25, .5, .75] },
};
export type AxisName = keyof typeof SWEEP_PARAMETERS;
export function sweepAxis(parameter: AxisName, text: string) {
  const values = text.split(',').map(item => item.trim() ? Number(item) : NaN), bounds = SWEEP_PARAMETERS[parameter];
  if (values.length !== 3 || new Set(values).size !== 3 || values.some(value => !Number.isFinite(value) || value < bounds.min || value > bounds.max)) throw new Error(`${bounds.label} requires 3 distinct values within ${bounds.min}–${bounds.max}`);
  return { parameter, values };
}
export const PIPELINE_NODES = [
  { id: 'source', label: 'Source FaceSet' }, { id: 'detector', label: 'Target Detector' },
  { id: 'pose', label: '3D Pose & Landmarks', note: 'Estimated 2.5D landmarks; UV projection unavailable' },
  { id: 'swapper', label: 'Face Swapper' }, { id: 'matte', label: 'XSeg / Occluder Matte', note: 'Reference BiSeNet / landmark hull matte; XSeg unavailable' },
  { id: 'restorer', label: 'Face Restorer', optional: true, note: 'Local GPEN adapter; UltraMax / CodeFormer unavailable' },
  { id: 'color', label: 'Color & Grain Match', optional: true, note: 'Reference LAB color match' },
  { id: 'grain', label: 'Grain Closer', optional: true, unavailable: 'No grain adapter is configured' },
  { id: 'final', label: 'Final Composite' },
];
export type PreviewResult = { frame_index: number; timestamp: number; notes: string[]; buffers: Record<string, string>; latency?: Record<string, number>; parameters: WorkstationParameters };
