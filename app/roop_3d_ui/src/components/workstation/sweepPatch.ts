import type { SettingId } from '../controls/settingsRegistry';
import type { WorkstationParameters } from './pipeline';

/** The three enhancer states the contact sheet compares. They are the Engine & Model Rack's own: "none" is a restoration blend of 0. */
export type EnhancerChoice = 'none' | 'gpen' | 'ultramax';
export const ENHANCER_CHOICES: { id: EnhancerChoice; label: string }[] = [
  { id: 'none', label: 'None' },
  { id: 'gpen', label: 'GPEN Realistic' },
  { id: 'ultramax', label: 'UltraMax' },
];
/** A restorer at blend 0 is indistinguishable from "none", so a swept restorer uses the current blend, or this when it is 0. */
export const SWEEP_ENHANCER_BLEND = 0.7;
export const RELIGHT_LEVELS: [number, number, number] = [0.25, 0.6, 1];

/** Everything a contact-sheet tile can change: the continuous settings plus the enhancer and SH relighting strength. */
export type Patch = Partial<Record<SettingId, number>> & { enhancer?: EnhancerChoice; relight_strength?: number };

/**
 * The workstation parameters a tile stands for. Used for the tile's own preview request and, unchanged, when the tile is
 * chosen, so what the user clicked is exactly what gets applied.
 */
export function applySweepPatch(parameters: WorkstationParameters, patch: Patch): WorkstationParameters {
  const { enhancer, relight_strength, ...continuous } = patch;
  let next: WorkstationParameters = { ...parameters, ...continuous };
  if ('restoration_blend' in continuous) next = { ...next, bypass: next.bypass.filter(item => item !== 'restorer') };
  if (enhancer === 'none') next = { ...next, restoration_blend: 0 };
  else if (enhancer) {
    next = {
      ...next, restorer_model: enhancer,
      restoration_blend: parameters.restoration_blend > 0 ? parameters.restoration_blend : SWEEP_ENHANCER_BLEND,
      bypass: next.bypass.filter(item => item !== 'restorer'),
    };
  }
  // SH relighting only has an effect while it is on, so sweeping its strength switches it on for that tile.
  if (relight_strength !== undefined) next = { ...next, relight: true, relight_strength };
  return next;
}
