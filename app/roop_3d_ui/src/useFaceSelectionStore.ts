import { useCallback, useSyncExternalStore } from 'react';
import type { DetectedFace, FaceSelectionState, Inspection, Probe, TargetCluster } from './contracts';

export interface FaceSelectionStoreState extends FaceSelectionState {
  hoveredFaceId: string | null;
  selectedClusterId: string | null;
  setDetectedFaces: (faces: DetectedFace[]) => void;
  addDetectedFaces: (faces: DetectedFace[]) => void;
  setActiveSelection: (faceId: string | null) => void;
  setHoveredFace: (faceId: string | null) => void;
  assignSource: (targetClusterId: string, sourceFaceId: string | null) => void;
  setClusterMode: (targetClusterId: string, mode: 'swap' | 'ignore' | 'exclude') => void;
  computeClusters: (similarityThreshold?: number) => void;
  populateFromInspection: (
    inspection: Inspection | null,
    probe?: Probe | null,
    existingAssignments?: Record<string, string>
  ) => void;
  getFacesAtFrame: (frameIndex: number) => DetectedFace[];
  getFacesAtTimestamp: (timestamp: number, fps?: number) => DetectedFace[];
  reset: () => void;
}

export function cosineSimilarity(a: number[], b: number[]): number {
  if (!a || !b || a.length !== b.length || a.length === 0) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

export function cosineDistance(a: number[], b: number[]): number {
  return 1 - cosineSimilarity(a, b);
}

export function clusterFaces(
  faces: DetectedFace[],
  similarityThreshold = 0.65,
  existingClusters?: TargetCluster[],
  assignments: Record<string, string> = {}
): TargetCluster[] {
  if (!faces || faces.length === 0) return [];

  const clustersMap = new Map<string, DetectedFace[]>();

  // If faces already have embeddings, group by cosine distance
  const hasEmbeddings = faces.some(f => f.embedding && f.embedding.length > 0);

  if (hasEmbeddings) {
    const visited = new Set<string>();
    let autoClusterIdx = 0;

    for (let i = 0; i < faces.length; i++) {
      const face = faces[i];
      if (visited.has(face.id)) continue;

      const clusterKey = face.clusterId || `cluster_${autoClusterIdx++}`;
      const group: DetectedFace[] = [face];
      visited.add(face.id);

      if (face.embedding) {
        for (let j = i + 1; j < faces.length; j++) {
          const other = faces[j];
          if (visited.has(other.id) || !other.embedding) continue;
          if (cosineDistance(face.embedding, other.embedding) <= 1 - similarityThreshold) {
            group.push(other);
            visited.add(other.id);
            other.clusterId = clusterKey;
          }
        }
      }
      clustersMap.set(clusterKey, group);
    }
  } else {
    // Group by existing clusterId (e.g. track_0000)
    for (const face of faces) {
      const key = face.clusterId || 'cluster_0';
      const list = clustersMap.get(key) ?? [];
      list.push(face);
      clustersMap.set(key, list);
    }
  }

  const existingMap = new Map(existingClusters?.map(c => [c.id, c]) ?? []);

  const results: TargetCluster[] = [];
  let clusterIndex = 0;

  for (const [clusterId, group] of clustersMap.entries()) {
    clusterIndex++;
    const uniqueFrames = new Set(group.map(f => f.frameIndex));

    // Average pose
    let totalYaw = 0;
    let totalPitch = 0;
    let totalRoll = 0;
    for (const f of group) {
      totalYaw += f.pose?.yaw ?? 0;
      totalPitch += f.pose?.pitch ?? 0;
      totalRoll += f.pose?.roll ?? 0;
    }
    const count = group.length || 1;
    const avgPose = {
      yaw: totalYaw / count,
      pitch: totalPitch / count,
      roll: totalRoll / count,
    };

    // Pick representative face: closest to frontal (min |yaw| + |pitch|) or first
    let bestFace = group[0];
    let bestScore = Infinity;
    for (const f of group) {
      const score = Math.abs(f.pose?.yaw ?? 0) + Math.abs(f.pose?.pitch ?? 0) + (1 - (f.confidence ?? 0.8));
      if (score < bestScore) {
        bestScore = score;
        bestFace = f;
      }
    }

    const prev = existingMap.get(clusterId);
    const assignedSource = assignments[clusterId] ?? prev?.sourceFaceId ?? null;

    results.push({
      id: clusterId,
      name: prev?.name ?? `Identity ${clusterIndex}`,
      faceIds: group.map(f => f.id),
      frameCount: uniqueFrames.size,
      avgPose,
      representativeFaceId: bestFace.id,
      thumbnailUrl: bestFace.thumbnailUrl || prev?.thumbnailUrl,
      mode: prev?.mode ?? 'swap',
      sourceFaceId: assignedSource,
    });
  }

  return results;
}

type Listener = () => void;
const listeners = new Set<Listener>();

let state: FaceSelectionStoreState = {
  detectedFaces: [],
  targetClusters: [],
  assignments: {},
  activeSelection: null,
  hoveredFaceId: null,
  selectedClusterId: null,

  setDetectedFaces: (faces: DetectedFace[]) => {
    state = {
      ...state,
      detectedFaces: faces,
      targetClusters: clusterFaces(faces, 0.65, state.targetClusters, state.assignments),
    };
    notify();
  },

  addDetectedFaces: (faces: DetectedFace[]) => {
    const combined = [...state.detectedFaces, ...faces];
    state = {
      ...state,
      detectedFaces: combined,
      targetClusters: clusterFaces(combined, 0.65, state.targetClusters, state.assignments),
    };
    notify();
  },

  setActiveSelection: (faceId: string | null) => {
    const face = faceId ? state.detectedFaces.find(f => f.id === faceId) : null;
    state = {
      ...state,
      activeSelection: faceId,
      selectedClusterId: face?.clusterId ?? null,
    };
    notify();
  },

  setHoveredFace: (faceId: string | null) => {
    if (state.hoveredFaceId === faceId) return;
    state = {
      ...state,
      hoveredFaceId: faceId,
    };
    notify();
  },

  assignSource: (targetClusterId: string, sourceFaceId: string | null) => {
    const nextAssignments = { ...state.assignments };
    if (sourceFaceId) {
      nextAssignments[targetClusterId] = sourceFaceId;
    } else {
      delete nextAssignments[targetClusterId];
    }
    const nextClusters = state.targetClusters.map(c =>
      c.id === targetClusterId ? { ...c, sourceFaceId: sourceFaceId ?? null } : c
    );
    state = {
      ...state,
      assignments: nextAssignments,
      targetClusters: nextClusters,
    };
    notify();
  },

  setClusterMode: (targetClusterId: string, mode: 'swap' | 'ignore' | 'exclude') => {
    const nextClusters = state.targetClusters.map(c =>
      c.id === targetClusterId ? { ...c, mode } : c
    );
    state = {
      ...state,
      targetClusters: nextClusters,
    };
    notify();
  },

  computeClusters: (similarityThreshold = 0.65) => {
    state = {
      ...state,
      targetClusters: clusterFaces(state.detectedFaces, similarityThreshold, state.targetClusters, state.assignments),
    };
    notify();
  },

  populateFromInspection: (
    inspection: Inspection | null,
    probe?: Probe | null,
    existingAssignments: Record<string, string> = {}
  ) => {
    if (!inspection || !inspection.tracks || inspection.tracks.length === 0) {
      state = {
        ...state,
        detectedFaces: [],
        targetClusters: [],
        assignments: {},
        activeSelection: null,
        hoveredFaceId: null,
        selectedClusterId: null,
      };
      notify();
      return;
    }

    const faces: DetectedFace[] = [];
    const fps = probe?.fps_rational ? evalRational(probe.fps_rational) : 30;

    for (const track of inspection.tracks) {
      const box = track.bounding_box;
      // Normalized [0..1] bbox. If bounding_box is in pixels, normalize using probe width/height
      let bboxNorm: [number, number, number, number] = [0.3, 0.2, 0.7, 0.6];
      if (box && probe && probe.width > 0 && probe.height > 0) {
        if (box.x2 <= 1.0 && box.y2 <= 1.0) {
          bboxNorm = [box.x1, box.y1, box.x2, box.y2];
        } else {
          bboxNorm = [
            box.x1 / probe.width,
            box.y1 / probe.height,
            box.x2 / probe.width,
            box.y2 / probe.height,
          ];
        }
      }

      // 5-point ArcFace landmark dots derived inside the bbox
      const [bx1, by1, bx2, by2] = bboxNorm;
      const bw = bx2 - bx1;
      const bh = by2 - by1;
      const landmarks5: [number, number][] = [
        [bx1 + bw * 0.35, by1 + bh * 0.38], // left eye
        [bx1 + bw * 0.65, by1 + bh * 0.38], // right eye
        [bx1 + bw * 0.50, by1 + bh * 0.58], // nose tip
        [bx1 + bw * 0.38, by1 + bh * 0.75], // left mouth
        [bx1 + bw * 0.62, by1 + bh * 0.75], // right mouth
      ];

      const yawAvg = track.pose_range.yaw_degrees ? (track.pose_range.yaw_degrees[0] + track.pose_range.yaw_degrees[1]) / 2 : 0;
      const pitchAvg = track.pose_range.pitch_degrees ? (track.pose_range.pitch_degrees[0] + track.pose_range.pitch_degrees[1]) / 2 : 0;
      const rollAvg = track.pose_range.roll_degrees ? (track.pose_range.roll_degrees[0] + track.pose_range.roll_degrees[1]) / 2 : 0;

      const thumb = track.thumbnail_base64 ? `data:${track.thumbnail_mime || 'image/jpeg'};base64,${track.thumbnail_base64}` : undefined;

      // Sample detections across the track duration
      const sampleCount = Math.max(1, track.sampled_detections ?? Math.min(10, track.last_frame - track.first_frame + 1));
      const frameStep = (track.last_frame - track.first_frame) / Math.max(1, sampleCount - 1);
      const timeStep = (track.last_timestamp_seconds - track.first_timestamp_seconds) / Math.max(1, sampleCount - 1);

      for (let s = 0; s < sampleCount; s++) {
        const frameIndex = Math.round(track.first_frame + s * frameStep);
        const timestamp = track.first_timestamp_seconds + s * timeStep;
        faces.push({
          id: `${track.track_id}_${frameIndex}`,
          frameIndex,
          timestamp,
          bbox: bboxNorm,
          landmarks5,
          clusterId: track.track_id,
          pose: { yaw: yawAvg, pitch: pitchAvg, roll: rollAvg },
          confidence: 0.95,
          thumbnailUrl: thumb,
        });
      }
    }

    const clusters = clusterFaces(faces, 0.65, undefined, existingAssignments);
    state = {
      ...state,
      detectedFaces: faces,
      targetClusters: clusters,
      assignments: { ...existingAssignments },
      activeSelection: faces[0]?.id ?? null,
      selectedClusterId: faces[0]?.clusterId ?? null,
    };
    notify();
  },

  getFacesAtFrame: (frameIndex: number) => {
    return state.detectedFaces.filter(f => f.frameIndex === frameIndex);
  },

  getFacesAtTimestamp: (timestamp: number, fps = 30) => {
    const frameIndex = Math.round(timestamp * fps);
    // Find exact or closest within 0.35 seconds
    const exact = state.detectedFaces.filter(f => f.frameIndex === frameIndex);
    if (exact.length > 0) return exact;

    // Fallback: check timestamp tolerance
    return state.detectedFaces.filter(f => {
      if (f.timestamp !== undefined) {
        return Math.abs(f.timestamp - timestamp) <= 0.25;
      }
      return Math.abs(f.frameIndex - frameIndex) <= 2;
    });
  },

  reset: () => {
    state = {
      ...state,
      detectedFaces: [],
      targetClusters: [],
      assignments: {},
      activeSelection: null,
      hoveredFaceId: null,
      selectedClusterId: null,
    };
    notify();
  },
};

function notify() {
  listeners.forEach(listener => listener());
}

function evalRational(r: string): number {
  if (r.includes('/')) {
    const [num, den] = r.split('/').map(Number);
    return den ? num / den : 30;
  }
  const parsed = parseFloat(r);
  return isNaN(parsed) ? 30 : parsed;
}

export function useFaceSelectionStore<T = FaceSelectionStoreState>(
  selector: (s: FaceSelectionStoreState) => T = (s => s as unknown as T)
): T {
  const getSnapshot = useCallback(() => selector(state), [selector]);
  return useSyncExternalStore(
    cb => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    },
    getSnapshot,
    getSnapshot
  );
}

useFaceSelectionStore.getState = () => state;
useFaceSelectionStore.setState = (
  updater: Partial<FaceSelectionStoreState> | ((prev: FaceSelectionStoreState) => Partial<FaceSelectionStoreState>)
) => {
  const next = typeof updater === 'function' ? updater(state) : updater;
  state = { ...state, ...next };
  notify();
};
useFaceSelectionStore.subscribe = (listener: Listener) => {
  listeners.add(listener);
  return () => listeners.delete(listener);
};
