export type WorkstationParameters = {
  color_method?: 'lab' | 'histogram';
  restorer_model?: 'gpen' | 'codeformer'; fidelity_weight?: number; identity_blend_t?: number;
  restoration_blend: number; feather_px: number; color_strength: number; pose_strength: number; pose_mode?: 'retarget' | 'mesh_blend';
  swap_weight?: number; erosion_px?: number; bypass: ('restorer' | 'color')[];
  /** Pre-swap 3D pose frontalization, 0..1 (Stage 5). Omitted until the user touches it, so older requests keep their identity. */
  pose_frontalization?: number;
  /** TPS expression warp toward the target frame, 0..1 (Stage 5). */
  tps_expression_weight?: number;
  /** Compositor occlusion strategy. Omitted until the user touches it, so older requests keep the standard path. */
  occlusion_mode?: 'standard' | 'robust';
  /** Stage 4 render mode. Omitted (= 'standard') until the user picks a 3D mode, so older requests keep their identity. */
  render_mode?: 'standard' | 'pose_3d' | 'reenactment_3d';
  /** Stage 17 3D Relighting: off unless enabled; the strength / light controls below only apply while it is on. */
  relight?: boolean;
  relight_strength?: number;
  specular_sheen?: number;
  shadow_preservation?: number;
  lighting_mode?: 'auto' | 'manual';
  manual_light_yaw?: number;
  manual_light_pitch?: number;
};
export const DEFAULT_PARAMETERS: WorkstationParameters = {
  restoration_blend: 0,
  feather_px: 5,
  color_strength: .65,
  pose_strength: .35,
  pose_mode: 'mesh_blend',
  swap_weight: 1,
  erosion_px: 0,
  bypass: [],
  relight_strength: 0.8,
  specular_sheen: 0.25,
  shadow_preservation: 0.85,
  lighting_mode: 'auto',
  manual_light_yaw: 35,
  manual_light_pitch: 25,
};

/** `buffer` is the `solo_node` that `POST /api/preview` returns for this stage. */
export type PipelineNode = { id: string; label: string; buffer: string; note?: string; optional?: boolean; unavailable?: string };
export const PIPELINE_NODES: PipelineNode[] = [
  { id: 'input', label: 'Input', buffer: 'detector', note: 'Target frame with the detector\'s face boxes' },
  { id: 'pose', label: '3D Pose & TPS Grid', buffer: 'pose', note: 'Landmarks, yaw/pitch/roll and the fitted 3D proxy / TPS warp mesh' },
  { id: 'normal', label: '3D Normal Map', buffer: 'normal_map', note: 'RGB-encoded 3D Surface Normal map (nx, ny, nz)' },
  { id: 'lighting', label: '3D Relighting (SH)', buffer: 'sh_lighting', note: 'Spherical Harmonics scene illumination transfer and shadow mask' },
  { id: 'swap', label: 'Raw Swap', buffer: 'swapper', note: 'The swapped face before masking, restoring or grading' },
  { id: 'matte', label: 'Occlusion & Seam Mask', buffer: 'matte', note: 'Black-and-white blend mask (reference BiSeNet / landmark hull; XSeg unavailable)' },
  { id: 'restorer', label: 'Restorer', buffer: 'restorer', optional: true, note: 'Local GPEN adapter; CodeFormer / UltraMax unavailable' },
  { id: 'final', label: 'Final Output', buffer: 'final', note: 'The composite exactly as a render would write it' },
];
export const nodeBuffer = (id: string | null): string | null => PIPELINE_NODES.find(node => node.id === id)?.buffer ?? null;

export type PreviewResult = { frame_index: number; timestamp: number; notes: string[]; buffers: Record<string, string>; latency?: Record<string, number>; parameters: WorkstationParameters };
