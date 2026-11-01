import type { WorkstationParameters } from '../workstation/pipeline';

export type PanelId = 'geometry' | 'masking' | 'restorer';
export type SettingId = 'pose_strength' | 'pose_frontalization' | 'tps_expression_weight' | 'swap_weight' | 'erosion_px' | 'feather_px'
  | 'color_strength' | 'restoration_blend' | 'fidelity_weight';

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
  /** Present when the engine accepts the control but does nothing with it; the value is the explanation. */
  inactive?: string;
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
    effect: 'Softens the mask edge. Low gives a crisp but visible seam; high blends smoothly but can bleed the swap onto hair or background.' },
  { id: 'color_strength', label: 'Colour match', panel: 'restorer', min: 0, max: 1, step: .01, default: .65, recommended: [.4, .8], sweep: [.2, .55, .9],
    effect: 'Pulls the swapped face\'s skin tone and lighting toward the target frame. Too low leaves a pasted-on tint; too high flattens genuine tone differences.' },
  { id: 'restoration_blend', label: 'Restorer blend', panel: 'restorer', min: 0, max: 1, step: .01, default: 0, recommended: [.3, .7], sweep: [.25, .6, 1],
    effect: 'Mixes in the GPEN face restorer for sharper skin, eyes and teeth. Beyond about 0.7 skin looks plastic. Needs the GPEN weights.' },
  { id: 'fidelity_weight', label: 'CodeFormer fidelity', panel: 'restorer', min: 0, max: 1, step: .01, default: .5, recommended: [.4, .7], sweep: [.2, .5, .8],
    inactive: 'CodeFormer is not wired into the preview compositor yet, so this control currently changes nothing.',
    effect: 'CodeFormer\'s trade-off between restored detail (low) and staying true to the input face (high).' },
];

export const SETTING_BY_ID = Object.fromEntries(SETTINGS.map(spec => [spec.id, spec])) as Record<SettingId, SettingSpec>;
export const settingsOf = (panel: PanelId) => SETTINGS.filter(spec => spec.panel === panel);

/** Current value of a setting, honouring that some parameters are omitted until touched. */
export function settingValue(spec: SettingSpec, parameters: WorkstationParameters, extra: Partial<Record<SettingId, number>> = {}): number {
  if (spec.id === 'fidelity_weight') return extra.fidelity_weight ?? spec.default;
  return (parameters as unknown as Record<string, number | undefined>)[spec.id] ?? spec.default;
}

/** "0.25–0.60" or "-2–4 px" */
export const recommendedText = (spec: SettingSpec) => {
  const fmt = (v: number) => (Number.isInteger(v) ? String(v) : v.toFixed(2));
  return `${fmt(spec.recommended[0])}–${fmt(spec.recommended[1])}${spec.unit ? ` ${spec.unit}` : ''}`;
};
