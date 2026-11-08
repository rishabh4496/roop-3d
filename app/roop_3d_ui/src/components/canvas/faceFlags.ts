import { createAtomicStore } from '../../stores/atomicStore';

export type Pose6 = { yaw: number; pitch: number; roll: number };
type FaceFlags = {
  /** Frozen 6DoF poses of locked faces (standard frame), by face key. The overlay holds the wireframe / gimbal / gaze at this pose. */
  locks: Record<string, Pose6>;
  /** The one face whose mask / mesh is shown at full strength; every other face is dimmed. Null = no solo. */
  soloKey: string | null;
};

/** Per-face canvas state that the right-click menu edits. Overlay-only: nothing here is sent to the render engine. */
export const faceFlags = createAtomicStore<FaceFlags>({ locks: {}, soloKey: null });

/** A face's identity across frames: its track (cluster) id, or its left-to-right rank for a not-yet-locked candidate. */
export const faceKey = (face: { clusterId?: string | null; rank?: number | null; faceRank?: number | null } | null | undefined): string =>
  face?.clusterId || `rank:${face?.rank ?? face?.faceRank ?? 0}`;

export const toggleSoloFace = (key: string) => faceFlags.set(state => ({ soloKey: state.soloKey === key ? null : key }));

export function togglePoseLock(key: string, pose: Pose6 | null) {
  faceFlags.set(state => {
    if (state.locks[key]) { const { [key]: _removed, ...rest } = state.locks; return { locks: rest }; }
    return pose ? { locks: { ...state.locks, [key]: { yaw: pose.yaw, pitch: pose.pitch, roll: pose.roll } } } : {};
  });
}
