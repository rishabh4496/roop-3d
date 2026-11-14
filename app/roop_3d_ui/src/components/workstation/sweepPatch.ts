import type { SettingId } from '../controls/settingsRegistry';
import type { WorkstationParameters } from './pipeline';

export const RELIGHT_LEVELS: [number, number, number] = [0.25, 0.6, 1];
/** Horizontal fields of view the contact sheet compares: telephoto, the solver's starting prior, wide-angle selfie. */
export const CAMERA_FOV_LEVELS: [number, number, number] = [30, 55, 85];

/** Everything a contact-sheet tile can change: the continuous settings plus SH relighting strength and the perspective camera's field of view. */
export type Patch = Partial<Record<SettingId, number>> & { relight_strength?: number; camera_fov_deg?: number };

/**
 * The workstation parameters a tile stands for. Used for the tile's own preview request and, unchanged, when the tile is
 * chosen, so what the user clicked is exactly what gets applied.
 */
export function applySweepPatch(parameters: WorkstationParameters, patch: Patch): WorkstationParameters {
  const { relight_strength, camera_fov_deg, ...continuous } = patch;
  let next: WorkstationParameters = { ...parameters, ...continuous };
  // SH relighting only has an effect while it is on, so sweeping its strength switches it on for that tile.
  if (relight_strength !== undefined) next = { ...next, relight: true, relight_strength };
  // A swept field of view is a manual perspective camera for that tile.
  if (camera_fov_deg !== undefined) next = { ...next, camera: { ...next.camera, mode: 'manual', projection: 'perspective', focal_unit: 'fov_deg', fov_deg: camera_fov_deg } };
  return next;
}
