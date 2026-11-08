import type { WorkstationParameters } from '../workstation/pipeline';

/** What the head viewer needs to preview the relighting controls. `null` while 3D relighting is off (the viewer keeps its neutral rig). */
export type ViewerLighting = { yaw: number; pitch: number; strength: number; ambient: number; specular: number; manual: boolean };
/** What the head viewer needs to preview the geometry morph controls. */
export type ViewerMorph = { blend: number; hairline: number };

export const DEFAULT_AMBIENT_FILL = 0.3;
export const HAIRLINE_RANGE_PX = 32;

export function viewerLighting(parameters: WorkstationParameters | undefined): ViewerLighting | null {
  if (!parameters?.relight) return null;
  const manual = parameters.lighting_mode === 'manual';
  return {
    // Scene (auto) light is only known to the engine, so the viewer shows a neutral three-quarter key until the user takes over.
    yaw: manual ? parameters.manual_light_yaw ?? 35 : 35, pitch: manual ? parameters.manual_light_pitch ?? 25 : 25,
    strength: parameters.relight_strength ?? 0.8, ambient: parameters.ambient_fill ?? DEFAULT_AMBIENT_FILL, specular: parameters.specular_sheen ?? 0.25, manual,
  };
}

/** The morph preview: `blend` is the source-mesh weight the engine really applies (`pose_strength` in mesh-blend mode), `hairline` is -1..1 of the offset range. */
export function viewerMorph(parameters: WorkstationParameters | undefined, enabled: boolean): ViewerMorph | null {
  if (!parameters || !enabled) return null;
  return { blend: parameters.pose_mode === 'retarget' ? 1 : parameters.pose_strength ?? 0.35, hairline: (parameters.hairline_offset_px ?? 0) / HAIRLINE_RANGE_PX };
}
