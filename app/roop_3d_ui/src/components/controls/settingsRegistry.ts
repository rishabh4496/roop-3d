import { fidelityRestorer, type WorkstationParameters } from '../workstation/pipeline';
import { pyramidInactiveReason, xsegInactiveReason } from '../studio/compositing';

export type PanelId = 'geometry' | 'masking' | 'restorer';
export type SettingId = 'pose_strength' | 'pose_frontalization' | 'tps_expression_weight' | 'swap_weight' | 'erosion_px' | 'feather_px'
  | 'xseg_occlusion_strength' | 'pyramid_levels' | 'color_strength' | 'restoration_blend' | 'fidelity_weight';

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
  geometry: { title: '3D Geometry', hint: 'Head pose, frontalization and expression shape' },
  masking: { title: 'Masking', hint: 'Where the swapped face is allowed to show' },
  restorer: { title: 'Restorer', hint: 'Detail recovery and colour match' },
};

export const SETTINGS: SettingSpec[] = [
  { id: 'pose_strength', label: '2.5D → 3D pose blend', panel: 'geometry', min: 0, max: 1, step: .01, default: .35, recommended: [.25, .6], sweep: [0, .5, 1],
    effect: 'Blends the flat thin-plate-spline warp (0) into a perspective 3D head proxy (1). Higher turns the swapped face with the head; too high can warp the cheeks on a three-quarter view.' },
  { id: 'pose_frontalization', label: 'Pose frontalization', panel: 'geometry', min: 0, max: 1, step: .01, default: 0, recommended: [0, .6], sweep: [0, .5, 1],
    effect: 'Straightens a turned head before the swap model sees it, then turns the result back. Helps profile and three-quarter targets look less smeared; leave at 0 for frontal faces.' },
  { id: 'tps_expression_weight', label: 'TPS expression weight', panel: 'geometry', min: 0, max: 1, step: .01, default: 0, recommended: [.3, .8], sweep: [0, .5, 1],
    effect: 'Moves the mouth, eyebrows and eyelids of the swapped face toward the target\'s expression. The face outline is pinned and never moves. 0 keeps the swap model\'s own expression.' },
  { id: 'swap_weight', label: 'Swap weight', panel: 'masking', min: 0, max: 1, step: .01, default: 1, recommended: [.8, 1], sweep: [.3, .65, 1],
    effect: 'How much of the swapped face replaces the original. 1 is a full swap; lower mixes the target\'s own face back in, which hides identity mismatch but looks like a ghost.' },
  { id: 'erosion_px', label: 'Mask erosion', panel: 'masking', min: -32, max: 32, step: 1, default: 0, recommended: [-2, 4], sweep: [-6, 0, 6], unit: 'px',
    effect: 'Shrinks (positive) or grows (negative) the swap mask. Positive hides a halo of the old face around the edge; negative covers more skin or hairline.' },
  { id: 'feather_px', label: 'Mask feather', panel: 'masking', min: 0, max: 64, step: .1, default: 5, recommended: [3, 12], sweep: [0, 6, 20], unit: 'px',
    effect: 'Softens the swap\'s edge inward from the detected face outline: the matte never grows past it. Low gives a crisp but visible seam; high blends smoothly but lets more of the target\'s own edge pixels show.' },
  { id: 'xseg_occlusion_strength', label: 'XSeg occlusion strength', panel: 'masking', min: 0, max: 1, step: .01, default: 1, recommended: [.7, 1], sweep: [0, .5, 1],
    effect: 'How strongly the XSeg and face-occluder networks cut hands, hair strands, glasses and props out of the swap (Robust occlusion only). 0 ignores the networks; the face parser and your manual mask always apply in full.' },
  { id: 'pyramid_levels', label: 'Pyramid bands', panel: 'masking', min: 4, max: 6, step: 1, default: 4, recommended: [4, 5], sweep: [4, 5, 6],
    effect: 'Bands of the multi-band seam blend (not used by Feather or Poisson fusion). More bands widen the low-frequency cross-fade across the face edge; 4 is the engine default.' },
  { id: 'color_strength', label: 'Colour match', panel: 'restorer', min: 0, max: 1, step: .01, default: .65, recommended: [.4, .8], sweep: [.2, .55, .9],
    effect: 'Pulls the swapped face\'s skin tone and lighting toward the target frame. Too low leaves a pasted-on tint; too high flattens genuine tone differences.' },
  { id: 'restoration_blend', label: 'Restorer blend', panel: 'restorer', min: 0, max: 1, step: .01, default: 0, recommended: [.3, .7], sweep: [.25, .6, 1],
    effect: 'Mixes in the selected face restorer (GPEN, CodeFormer, GFPGAN or UltraMax) for sharper skin, eyes and teeth. Beyond about 0.7 skin looks plastic. Needs that restorer\'s weights.' },
  { id: 'fidelity_weight', label: 'CodeFormer fidelity', panel: 'restorer', min: 0, max: 1, step: .01, default: .5, recommended: [.4, .7], sweep: [.2, .5, .8],
    effect: 'The CodeFormer fidelity input, also used by UltraMax (CodeFormer fp16): low recovers more detail, high stays true to the swapped face. GPEN and GFPGAN have no such input.' },
];

export const SETTING_BY_ID = Object.fromEntries(SETTINGS.map(spec => [spec.id, spec])) as Record<SettingId, SettingSpec>;
export const settingsOf = (panel: PanelId) => SETTINGS.filter(spec => spec.panel === panel);

/** Current value of a setting, honouring that some parameters are omitted until touched. */
export function settingValue(spec: SettingSpec, parameters: WorkstationParameters): number {
  return (parameters as unknown as Record<string, number | undefined>)[spec.id] ?? spec.default;
}

/**
 * Why a setting changes nothing for these parameters right now, or null when it acts. It follows the selected restorer, occlusion mode
 * and seam blend; the engine accepts the other values but ignores them, so the controls are disabled with the reason instead.
 */
export function inactiveReason(spec: SettingSpec, parameters: WorkstationParameters): string | null {
  switch (spec.id) {
    case 'fidelity_weight':
      if (!fidelityRestorer(parameters.restorer_model)) return 'Fidelity is the CodeFormer / UltraMax fidelity input: GPEN and GFPGAN have none.';
      return parameters.restoration_blend > 0 && !parameters.bypass.includes('restorer') ? null : 'The restorer is off (blend 0 or bypassed), so there is nothing for fidelity to act on.';
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
