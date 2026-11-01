/**
 * Head pose helpers in the STANDARD frame (yaw + = nose toward image right, pitch + = looking up), mirroring
 * `core/pose.py`. The server's raw `pose` tuples (inspection pose ranges, `FaceView`) are the legacy frame where
 * pitch + = looking down; `toStandardPose` converts them.
 */
export type Pose = { yaw: number; pitch: number; roll: number };

export const YAW_EDGE = 15;
export const PITCH_EDGE = 10;
const YAWS = ['left', 'center', 'right'] as const;
const PITCHES = ['down', 'level', 'up'] as const;

export const toStandardPose = (pose: Pose | null | undefined, frame: 'legacy' | 'standard' = 'legacy'): Pose | null =>
  pose && [pose.yaw, pose.pitch, pose.roll].every(Number.isFinite)
    ? { yaw: pose.yaw, pitch: frame === 'legacy' ? -pose.pitch : pose.pitch, roll: pose.roll }
    : null;

/** Cell name such as `right_down`, using the same inclusive centre bands as Stage 0 (|yaw| <= 15, |pitch| <= 10). */
export function poseCellName(pose: Pose): string {
  const yaw = pose.yaw < -YAW_EDGE ? 0 : pose.yaw > YAW_EDGE ? 2 : 1;
  const pitch = pose.pitch < -PITCH_EDGE ? 0 : pose.pitch > PITCH_EDGE ? 2 : 1;
  return `${YAWS[yaw]}_${PITCHES[pitch]}`;
}

/** "Y +12.5°  P -3.0°  R +1.5°": fixed sign and width so the badge doesn't jitter while the pose changes. */
export function formatPose(pose: Pose): string {
  const part = (label: string, value: number) => `${label} ${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(1)}°`;
  return `${part('Y', pose.yaw)}  ${part('P', pose.pitch)}  ${part('R', pose.roll)}`;
}

/** Face number the server uses for `face_index`: position by left edge (normalised box x1), leftmost = 0. */
export function leftToRightRank<T extends { bbox: number[] }>(faces: T[], face: T): number {
  return [...faces].sort((a, b) => a.bbox[0] - b.bbox[0]).indexOf(face);
}
