import type { WorkstationParameters } from './components/workstation/pipeline';
export type Schema = { type?: string; title?: string; description?: string; enum?: (string | number | boolean)[]; const?: unknown; minimum?: number; maximum?: number; properties?: Record<string, Schema> };
export type Config = Record<string, string | number | boolean | string[]>;
export type Feature = { enabled: boolean; reason: string | null };
export interface Capabilities {
  enabled_modes: string[]; rendering_worker_available: boolean; legal_parameter_bounds: Schema;
  inspection_parameter_bounds: Schema; job_parameter_bounds: Schema;
  models: { model_id: string; name: string; state: string; license_state: string; inference_state: string; sha256: string }[];
  workstation: { features: Record<string, Feature>; controls: Record<string, Record<string, string | null>>;
    read_only_parameters: Record<string, string>; disabled_parameters: Record<string, string>; job_states: string[];
    job_controls: Record<string, string[]>; upload_chunk_bytes: number; max_asset_bytes: number };
}
export type JobState = 'queued' | 'preparing' | 'running' | 'pausing' | 'paused' | 'cancelling' | 'cancelled' | 'succeeded' | 'failed' | 'interrupted';
export type CanonicalJobState = 'QUEUED' | 'PREFLIGHT' | 'PRESCANNING' | 'PREPASSING' | 'RENDERING' | 'MUXING' | 'PAUSED' | 'COMPLETED' | 'FAILED' | 'CANCELLED';

export interface QueueSlice {
  slice_id: string;
  slice_index: number;
  chunk_path: string;
  rendered_chunk_path?: string | null;
  start_seconds: number;
  end_seconds?: number | null;
  duration: number;
  status: 'PENDING' | 'RENDERING' | 'COMPLETED' | 'FAILED';
  error?: string | null;
  retry_count: number;
  frame_count?: number;
}

export interface QueueJob {
  job_id: string;
  name: string;
  state: CanonicalJobState;
  queue_order: number;
  request: Record<string, any>;
  progress: {
    frame: number;
    total_frames: number | null;
    fps: number;
    stage: string;
    stage_latency_ms?: number;
    stage_timings_ms?: Record<string, number>;
  };
  slices: QueueSlice[];
  checkpoint_path?: string | null;
  output?: Record<string, any> | null;
  error_message?: string | null;
  revision: number;
  created_at: string;
  updated_at: string;
}
export interface Mapping { track_id: string; faceset_id: string }
export interface ExportOptions { baked_video: boolean; passes: string[]; finishing: { cube: string | null; grain: { size: number; intensity: number; shadows: number; highlights: number; seed: number }; provenance: string } }
export interface JobRequest { identity_blend?: {method: 'slerp' | 'linear'; source_face_ids: string[]; weights?: number[]} | null; target_path: string; inspection_id: string; face_mappings: Mapping[]; processing: 'local'; remote_consent: false; trim_start_seconds: number | null; trim_end_seconds: number | null; mask_base64?: string | null; workstation_parameters?: WorkstationParameters | null; parameter_curves?: { tracks: Record<string, {frame: number; value: number; easing: string}[]> } | null; export?: ExportOptions | null; face_segments?: {cluster_id: string; track_id: string; start_frame: number; end_frame: number | null; role: string; faceset_id: string | null; source_view_index: number | null}[] | null }
export interface Job { job_id: string; state: JobState; request: JobRequest; progress: { frame: number; total_frames: number | null; fps: number; stage: string; stage_latency_ms?: number; stage_timings_ms?: Record<string, number> }; output: { output_path: string; frames: number; passes?: Record<string, string>; pass_manifest_path?: string | null; baked_video?: boolean } | null; failure: { message: string } | null }
export interface Track { track_id: string; first_frame: number; last_frame: number; first_timestamp_seconds: number; last_timestamp_seconds: number; thumbnail_base64: string; thumbnail_mime: string; landmark_count: number; pose_range: { yaw_degrees: [number, number] | null; pitch_degrees: [number, number] | null; roll_degrees: [number, number] | null; method: string }; bounding_box?: { x1: number; y1: number; x2: number; y2: number }; sampled_detections?: number }
export interface DetectedFace {
  id: string;
  frameIndex: number;
  timestamp?: number;
  bbox: [number, number, number, number];
  landmarks5?: [number, number][];
  landmarks68?: [number, number][];
  embedding?: number[];
  clusterId: string;
  /** Legacy frame (pitch + = looking down). Prefer `pose3d` for anything shown to the user. */
  pose: { yaw: number; pitch: number; roll: number };
  /** Standard frame (pitch + = looking up), when the server provided it. */
  pose3d?: { yaw: number; pitch: number; roll: number };
  confidence?: number;
  thumbnailUrl?: string;
}
export interface TargetCluster {
  id: string;
  name?: string;
  faceIds: string[];
  frameCount: number;
  avgPose: { yaw: number; pitch: number; roll: number };
  representativeFaceId: string;
  thumbnailUrl?: string;
  mode: 'swap' | 'ignore' | 'exclude';
  sourceFaceId?: string | null;
  originTrackId?: string;
  startFrame?: number;
  endFrame?: number | null;
  sourceViewIndex?: number | null;
  splitLocked?: boolean;
}
export interface FaceSelectionState {
  detectedFaces: DetectedFace[];
  targetClusters: TargetCluster[];
  assignments: Record<string, string>;
  activeSelection: string | null;
  hoveredFaceId: string | null;
}
export interface Inspection { inspection_id: string; target_sha256: string; tracks: Track[]; warnings: string[]; truncated: boolean }
export interface FaceSet { faceset_id: string; name: string; item_count: number }
export interface Probe { width: number; height: number; duration_seconds: number | null; fps_rational: string | null; declared_frame_count: number | null; audio_tracks: unknown[] }
export interface Coverage { faceset_id: string; views: { yaw: number; pitch: number; roll: number; pose_bin: string }[]; bins: Record<string, number> }
export interface Reconstruction { method: 'flame' | 'tps_2_5d'; coordinate_system: 'opencv' | 'opengl'; vertices: number[][]; faces: number[][]; uvs: number[][] | null; face_uvs?: number[][] | null; texture: string | null; provenance: string }
export interface Project {
  project_id: string;
  name: string;
  request: JobRequest;
  effective_settings: Config;
  last_completed_frame: number;
  checkpoint_job_id: string | null;
  saved_at: string;
  inputs_verified?: boolean;
  status?: string;
  verification_reason?: string | null;
  missing_inputs?: string[];
  checkpoint?: {
    job_id: string;
    completed_frame: number;
    completed_segment?: number;
    total_segments?: number | null;
    execution_provider?: string;
    input_hashes?: Record<string, string>;
    faceset_ids?: string[];
    pose_cache?: Record<string, number[]>;
    keyframe_curves?: Record<string, unknown>;
    manual_mask?: string | null;
    has_binary_checkpoint?: boolean;
  } | null;
  target_path?: string;
  face_mappings?: Mapping[];
  trim_start_seconds?: number | null;
  trim_end_seconds?: number | null;
  has_mask?: boolean;
  has_curves?: boolean;
}
export interface Telemetry { gpus: { name: string; total_vram_mib: number; free_vram_mib: number }[]; logical_cpu_count: number }
export interface Event { type: string; sequence?: number; hardware?: Telemetry }

export interface DetectedFaceItem {
  id: string;
  track_id: string;
  frame_index: number;
  timestamp: number;
  bbox: [number, number, number, number];
  landmarks5: [number, number][];
  confidence: number;
  pose: { yaw: number; pitch: number; roll: number };
  pose3d?: { yaw: number; pitch: number; roll: number };
  pose_cell?: string;
  thumbnail_base64: string;
}

export interface FrameFaceDetectionResponse {
  inspection_id: string;
  faces: DetectedFaceItem[];
  inspection: Inspection;
}
