export type RestorerModel = 'gpen' | 'codeformer' | 'ultramax' | 'gfpgan';
/** Weight file each enhancer needs. UltraMax is CodeFormer weights + Laplacian detail fusion, so it shares CodeFormer's file. */
export const restorerWeights = (model: RestorerModel | undefined): string => (model === 'codeformer' || model === 'ultramax' ? 'CodeFormerv0.1.onnx' : model === 'gfpgan' ? 'GFPGANv1.4.onnx' : 'gpen_bfr_512.onnx');
export type WorkstationParameters = {
  expression_restore_factor?: number;
  face_expression_overrides?: Record<string, FaceEditor>;
  color_method?: 'lab' | 'reinhard_lab' | 'histogram';
  restorer_model?: RestorerModel; fidelity_weight?: number; identity_blend_t?: number;
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
  /** Confidence the semantic face parser must reach before its mask is trusted (0..1, engine default 0.55). Omitted until touched. */
  occlusion_threshold?: number;
  specular_sheen?: number;
  shadow_preservation?: number;
  ultramax_small_face_mode?: 'bypass' | 'adaptive' | 'off';
  ultramax_min_native_pixels?: number;
  lighting_mode?: 'auto' | 'manual';
  manual_light_yaw?: number;
  manual_light_pitch?: number;
  /** Ambient fill (the SH band-0 term) of the manual / scene light, 0..1. Omitted until the user touches it. */
  ambient_fill?: number;
  /** Forehead / hairline boundary offset in px (+ grows the swapped region upward, - pulls it down), -32..32. Omitted until touched. */
  hairline_offset_px?: number;
  /** Target camera shutter angle in degrees; 180 is the conventional half-frame exposure. */
  sensor_shutter_angle?: number;
  /** Grain variance match strength, 0..100 percent. */
  grain_match_strength?: number;
};
export type FaceEditor = Partial<Record<'head_pitch' | 'head_yaw' | 'head_roll' | 'gaze_x' | 'gaze_y' | 'blink_left' | 'blink_right' | 'eyebrow_tilt' | 'eyebrow_raise' | 'mouth_smile' | 'mouth_pout' | 'mouth_open' | 'jaw_width' | 'chin_length', number>>;
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

/** `buffer` is the `solo_node` that `POST /api/preview` returns for this stage. `latencyKeys` are the measured fields of the preview's `latency` object that make up this stage's time. */
export type PipelineNode = { id: string; label: string; buffer: string; note?: string; optional?: boolean; unavailable?: string; latencyKeys?: string[]; short?: string };
/**
 * The eight inspectable stages, in execution order. Hotkeys `1`–`8` solo them in this order (`buffer1`…`buffer8`).
 * Every stage maps to a buffer the server really returns; where the engine has no buffer of exactly that name the note says
 * which real buffer stands in for it.
 */
export const PIPELINE_NODES: PipelineNode[] = [
  { id: 'input', label: 'Raw Target', short: 'Raw Target', buffer: 'detector', latencyKeys: ['detection_ms'], note: 'Target frame with the detector\'s face boxes' },
  { id: 'pose', label: '6DoF 3D Mesh & Pose', short: '6DoF Mesh & Pose', buffer: 'pose', latencyKeys: ['retarget_ms'], note: 'Landmarks, yaw/pitch/roll and the fitted 3D proxy / TPS warp mesh' },
  { id: 'uv', label: 'Canonical UV Projection', short: 'UV Projection', buffer: 'source', latencyKeys: ['projection_ms'], note: 'The pose-matched canonical source view the identity is projected from (the engine exports this view, not the raw UV atlas)' },
  { id: 'lighting', label: 'SH 3D Relighting & Normals', short: 'SH Relight & Normals', buffer: 'sh_lighting', note: 'Spherical Harmonics scene illumination transfer and shadow mask. Use the Normals chip for the RGB surface-normal map' },
  { id: 'swap', label: 'Base 3D Swap', short: 'Base Swap', buffer: 'swapper', latencyKeys: ['swap_ms'], note: 'The swapped face before masking, restoring or grading' },
  { id: 'restorer', label: 'UltraMax High-Res', short: 'UltraMax', buffer: 'restorer', optional: true, note: 'The selected enhancer\'s output (GPEN, CodeFormer or UltraMax) at its native resolution' },
  { id: 'matte', label: 'XSeg + Hair Occlusion Matte', short: 'XSeg + Hair Matte', buffer: 'matte', note: 'Black-and-white blend mask. XSeg and the hair/occluder matte are in it when Occlusion handling is set to Robust' },
  { id: 'final', label: 'Final Composite', short: 'Final Composite', buffer: 'final', latencyKeys: ['total_ms'], note: 'The composite exactly as a render would write it' },
];
/** Secondary buffers reached from a chip on their parent stage; they are not stages of their own (and have no hotkey). */
export const PIPELINE_ALT_NODES: (PipelineNode & { parent: string })[] = [
  { id: 'expression', parent: 'swap', label: 'Stage 2.5 Micro-Expression Restorer', short: 'Expression', buffer: 'expression', optional: true, note: 'Target expression residual and per-face 3D edits, before UltraMax' },
  { id: 'normal', parent: 'lighting', label: '3D Normal Map', short: 'Normals', buffer: 'normal_map', note: 'RGB-encoded 3D Surface Normal map (nx, ny, nz)' },
];
export const pipelineNode = (id: string | null): PipelineNode | undefined => PIPELINE_NODES.find(node => node.id === id) ?? PIPELINE_ALT_NODES.find(node => node.id === id);
export const nodeBuffer = (id: string | null): string | null => pipelineNode(id)?.buffer ?? null;

/**
 * Measured milliseconds for one stage from a preview's `latency` object, or null when the engine did not time that stage.
 * Nothing is estimated: the preview times detection, retarget, projection, swap and the total only.
 */
export function stageLatency(node: PipelineNode, latency: Record<string, number> | null | undefined): number | null {
  if (!latency || !node.latencyKeys?.length) return null;
  let total = 0;
  for (const key of node.latencyKeys) {
    const value = latency[key];
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) return null;
    total += value;
  }
  return total;
}
export const formatMs = (value: number | null): string => value === null ? '—' : value < 10 ? `${value.toFixed(1)} ms` : `${Math.round(value)} ms`;

export type PreviewResult = { frame_index: number; timestamp: number; notes: string[]; buffers: Record<string, string>; latency?: Record<string, number>; parameters: WorkstationParameters };
