export type ViewportPreferences = {
  live_swap: boolean;
  view_mode: 'side' | 'compare';
  compare_mode: 'wipe' | 'crossfade' | 'diff' | 'blend' | 'loupe';
  show_landmarks: boolean;
  show_pose: boolean;
  show_overlays: boolean;
  grid_mode: 'off' | 'thirds' | 'crosshair' | 'safe';
  zoom: number;
};
export const DEFAULT_VIEWPORT: ViewportPreferences = {
  live_swap: true, view_mode: 'side', compare_mode: 'wipe', show_landmarks: true,
  show_pose: true, show_overlays: true, grid_mode: 'off', zoom: 1,
};
export type ProjectPreview = { timestamp: number; selected_track: string; source_faceset_id: string; viewport: ViewportPreferences };
