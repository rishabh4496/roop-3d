/** @type {Record<string, {label: string; min: number; max: number; initial: number; field: string}>} */
export const CURVE_PARAMETERS = {
  // The parameters the timeline is built around; every one is a real engine field evaluated per frame by the backend. Curves saved for
// the removed 2D / 2.5D parameters (pose weight, frontalization, TPS, restorer) are ignored by `evaluateParameters` and dropped by `migrateCurves`.
  relight_strength: { label: 'SH relighting intensity', min: 0, max: 1, initial: .8, field: 'relight_strength' },
  mask_feather: { label: 'Mask feather radius', min: 0, max: 64, initial: 5, field: 'feather_px' },
  occlusion_threshold: { label: 'Occlusion threshold (XSeg / BiSeNet)', min: 0, max: 1, initial: .55, field: 'occlusion_threshold' },
  identity_blend_t: {label: 'Identity A to B (Slerp)', min: 0, max: 1, initial: 0, field: 'identity_blend_t'},
  swap_weight: { label: 'Identity weight', min: 0, max: 1, initial: 1, field: 'swap_weight' },
  mask_erosion: { label: 'Mask erosion', min: -32, max: 32, initial: 0, field: 'erosion_px' },
};
/** What each animatable parameter really does, shown under the curve. Only claims the engine keeps. */
export const CURVE_HINTS = {
  occlusion_threshold: 'Confidence the semantic face parser (BiSeNet) must reach before its mask is trusted over the landmark outline. Lower keeps more of the swap behind hands and hair; higher cuts the swap back sooner. This build has no separate XSeg network: the control applies to the installed parser.',
  relight_strength: 'How strongly the Spherical-Harmonics relighting is applied. Takes effect only while 3D Relighting is switched on.',
  mask_feather: 'Softness of the swap mask edge, in pixels.',
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

/** Saved curve tracks: keep only those whose parameter still exists (removed 2D / 2.5D curves are dropped, never an error). */
export function migrateCurves(tracks) {
  if (!tracks || typeof tracks !== 'object') return {};
  return Object.fromEntries(Object.entries(tracks).filter(([name]) => Object.hasOwn(CURVE_PARAMETERS, name)));
}
