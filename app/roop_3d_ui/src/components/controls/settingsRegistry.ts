import type { WorkstationParameters } from '../workstation/pipeline';
import { pyramidInactiveReason, xsegInactiveReason } from '../studio/compositing';

export type PanelId = 'masking' | 'color';
export type SettingId = 'swap_weight' | 'erosion_px' | 'feather_px' | 'xseg_occlusion_strength' | 'pyramid_levels' | 'color_strength';

export interface SettingSpec {
  id: SettingId;
  label: string;
  panel: PanelId;
  min: number;
  max: number;
  step: number;
  default: number;
  /** Range that usually looks right, shown in the help and drawn on the slider's help. */
  recommended: [number, number];
  /** What you will SEE change. */
  effect: string;
  /** Contact-sheet values: Low, Medium, High. */
  sweep: [number, number, number];
  unit?: string;
}

export const PANELS: Record<PanelId, { title: string; hint: string }> = {
  masking: { title: 'Masking', hint: 'Where the rendered head is allowed to show' },
  color: { title: 'Colour', hint: 'Colour match of the rendered head to the target frame' },
};

export const SETTINGS: SettingSpec[] = [
  { id: 'swap_weight', label: 'Identity weight', panel: 'masking', min: 0, max: 1, step: .01, default: 1, recommended: [.8, 1], sweep: [.3, .65, 1],
    effect: 'How much of the rendered 3D head replaces the original. 1 is the full head; lower mixes the target\'s own face back in, which hides identity mismatch but looks like a ghost.' },
  { id: 'erosion_px', label: 'Mask erosion', panel: 'masking', min: -32, max: 32, step: 1, default: 0, recommended: [-2, 4], sweep: [-6, 0, 6], unit: 'px',
    effect: 'Shrinks (positive) or grows (negative) the head mask. Positive hides a halo of the old face around the edge; negative covers more skin or hairline.' },
  { id: 'feather_px', label: 'Mask feather', panel: 'masking', min: 0, max: 64, step: .1, default: 5, recommended: [3, 12], sweep: [0, 6, 20], unit: 'px',
    effect: 'Softens the head\'s edge inward from the detected face outline: the matte never grows past it. Low gives a crisp but visible seam; high blends smoothly but lets more of the target\'s own edge pixels show.' },
  { id: 'xseg_occlusion_strength', label: 'XSeg occlusion strength', panel: 'masking', min: 0, max: 1, step: .01, default: 1, recommended: [.7, 1], sweep: [0, .5, 1],
    effect: 'How strongly the XSeg and face-occluder networks cut hands, hair strands, glasses and props out of the render (Robust occlusion only). 0 ignores the networks; the face parser and your manual mask always apply in full.' },
  { id: 'pyramid_levels', label: 'Pyramid bands', panel: 'masking', min: 4, max: 6, step: 1, default: 4, recommended: [4, 5], sweep: [4, 5, 6],
    effect: 'Bands of the multi-band seam blend (not used by Feather or Poisson fusion). More bands widen the low-frequency cross-fade across the face edge; 4 is the engine default.' },
  { id: 'color_strength', label: 'Colour match', panel: 'color', min: 0, max: 1, step: .01, default: .65, recommended: [.4, .8], sweep: [.2, .55, .9],
    effect: 'Pulls the rendered head\'s skin tone and lighting toward the target frame. Too low leaves a pasted-on tint; too high flattens genuine tone differences.' },
];

export const SETTING_BY_ID = Object.fromEntries(SETTINGS.map(spec => [spec.id, spec])) as Record<SettingId, SettingSpec>;
export const settingsOf = (panel: PanelId) => SETTINGS.filter(spec => spec.panel === panel);

/** Current value of a setting, honouring that some parameters are omitted until touched. */
export function settingValue(spec: SettingSpec, parameters: WorkstationParameters): number {
  return (parameters as unknown as Record<string, number | undefined>)[spec.id] ?? spec.default;
}

/**
 * Why a setting changes nothing for these parameters right now, or null when it acts. It follows the occlusion mode and seam blend; the engine accepts the other values but ignores them, so the controls are disabled with the reason instead.
 */
export function inactiveReason(spec: SettingSpec, parameters: WorkstationParameters): string | null {
  switch (spec.id) {
    case 'xseg_occlusion_strength': return xsegInactiveReason(parameters);
    case 'pyramid_levels': return pyramidInactiveReason(parameters);
    default: return null;
  }
}

/** "0.25–0.60" or "-2–4 px" */
export const recommendedText = (spec: SettingSpec) => {
  const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
  return `${fmt(spec.recommended[0])}–${fmt(spec.recommended[1])}${spec.unit ? ` ${spec.unit}` : ''}`;
};
