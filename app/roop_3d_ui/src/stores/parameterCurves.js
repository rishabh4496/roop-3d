/** @type {Record<string, {label: string; min: number; max: number; initial: number; field: string}>} */
export const CURVE_PARAMETERS = {
  fidelity_weight: {label: 'CodeFormer fidelity', min: 0, max: 1, initial: .5, field: 'fidelity_weight'},
  identity_blend_t: {label: 'Identity A to B (Slerp)', min: 0, max: 1, initial: 0, field: 'identity_blend_t'},
  tps_expression_weight: { label: 'TPS expression', min: 0, max: 1, initial: 0, field: 'tps_expression_weight' },
  pose_frontalization: { label: 'Pose frontalization', min: 0, max: 1, initial: 0, field: 'pose_frontalization' },
  swap_weight: { label: 'Swap weight', min: 0, max: 1, initial: 1, field: 'swap_weight' },
  restorer_opacity: { label: 'Restorer opacity', min: 0, max: 1, initial: 0, field: 'restoration_blend' },
  mask_feather: { label: 'Mask feather', min: 0, max: 64, initial: 5, field: 'feather_px' },
  mask_erosion: { label: 'Mask erosion', min: -32, max: 32, initial: 0, field: 'erosion_px' },
  pose_correction_strength: { label: 'Pose correction strength', min: 0, max: 1, initial: .35, field: 'pose_strength' },
};
export function evaluateCurve(keys, frame, fallback) {
  if (!keys?.length) return fallback;
  const ordered = [...keys].sort((a, b) => a.frame - b.frame);
  if (frame <= ordered[0].frame) return ordered[0].value;
  for (let index = 1; index < ordered.length; index++) {
    const left = ordered[index - 1], right = ordered[index];
    if (frame >= right.frame) continue;
    if (['hold','step'].includes(left.easing)) return left.value;
    const t = (frame - left.frame) / (right.frame - left.frame);
    const eased = ['ease-in-out','bezier'].includes(left.easing) ? 3*(1-t)**2*t*(left.control1 ?? 0)+3*(1-t)*t*t*(left.control2 ?? 1)+t**3 : t;
    return left.value + (right.value - left.value) * eased;
  }
  return ordered.at(-1).value;
}
export function evaluateParameters(base, curves, frame) {
  const result = { ...base };
  for (const [name, keys] of Object.entries(curves)) {
    const definition = CURVE_PARAMETERS[name];
    if (definition) result[definition.field] = evaluateCurve(keys, frame, base[definition.field] ?? definition.initial);
  }
  return result;
}
export function upsertKey(keys, point, movingFrame = point.frame) {
  return [...(keys ?? []).filter(key => key.frame !== movingFrame && key.frame !== point.frame), point].sort((a, b) => a.frame - b.frame);
}
