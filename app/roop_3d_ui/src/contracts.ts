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
export interface Mapping { track_id: string; faceset_id: string }
export interface JobRequest { target_path: string; inspection_id: string; face_mappings: Mapping[]; processing: 'local'; remote_consent: false; trim_start_seconds: number | null; trim_end_seconds: number | null; mask_base64?: string | null }
export interface Job { job_id: string; state: JobState; request: JobRequest; progress: { frame: number; total_frames: number | null; fps: number; stage: string }; output: { output_path: string; frames: number } | null; failure: { message: string } | null }
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
  pose: { yaw: number; pitch: number; roll: number };
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
export interface Project { project_id: string; name: string; request: JobRequest; effective_settings: Config; last_completed_frame: number; checkpoint_job_id: string | null; saved_at: string }
export interface Telemetry { gpus: { name: string; total_vram_mib: number; free_vram_mib: number }[]; logical_cpu_count: number }
export interface Event { type: string; sequence?: number; hardware?: Telemetry }
