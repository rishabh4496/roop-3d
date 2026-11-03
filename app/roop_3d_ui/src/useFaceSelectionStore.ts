import { useCallback, useSyncExternalStore } from 'react';
import type { DetectedFace, DetectedFaceItem, FaceSelectionState, Inspection, Probe, TargetCluster } from './contracts';
import { executeCommand } from './stores/useHistoryStore';

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
  splitCluster: (clusterId: string, frame: number) => boolean;
  setSourceView: (clusterId: string, view: number | null) => void;
  replaceAssignments: (assignments: Record<string, string>) => void;
  focusPose: (faceId: string) => void;
  poseFocus: { faceId: string; serial: number } | null;
  addFrameDetectedFaces: (faces: DetectedFaceItem[], autoSelect?: boolean) => string | null;
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
  const protectedCluster = (face: DetectedFace) => existingClusters?.find(cluster => cluster.id === face.clusterId && (cluster.mode !== 'swap' || cluster.splitLocked));

  // If faces already have embeddings, group by cosine distance
  const hasEmbeddings = faces.some(f => f.embedding && f.embedding.length > 0);

  if (hasEmbeddings) {
    const visited = new Set<string>();
    let autoClusterIdx = 0;

    for (let i = 0; i < faces.length; i++) {
      const face = faces[i];
      if (visited.has(face.id)) continue;

      const clusterKey = face.clusterId || `cluster_${autoClusterIdx++}`;
      if (protectedCluster(face)) {
        const group = faces.filter(other => other.clusterId === face.clusterId);
        group.forEach(other => visited.add(other.id));
        clustersMap.set(clusterKey, group);
        continue;
      }
      const group: DetectedFace[] = [face];
      visited.add(face.id);

      if (face.embedding) {
        for (let j = i + 1; j < faces.length; j++) {
          const other = faces[j];
          if (visited.has(other.id) || !other.embedding || protectedCluster(other)) continue;
          if (cosineDistance(face.embedding, other.embedding) <= 1 - similarityThreshold) {
            group.push(other);
            visited.add(other.id);
          }
        }
      }
      clustersMap.set(clusterKey, [...(clustersMap.get(clusterKey) ?? []), ...group]);
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
      ...prev,
      id: clusterId,
      name: prev?.name ?? `Identity ${clusterIndex}`,
      faceIds: group.map(f => f.id),
      frameCount: uniqueFrames.size,
      avgPose,
      representativeFaceId: bestFace.id,
      thumbnailUrl: bestFace.thumbnailUrl || prev?.thumbnailUrl,
      mode: prev?.mode ?? 'swap',
      sourceFaceId: prev?.mode && prev.mode !== 'swap' ? null : assignedSource,
    });
  }

  return results;
}

type Listener = () => void;
const listeners = new Set<Listener>();

function clusteredState(faces: DetectedFace[], threshold: number, clusters: TargetCluster[], assignments: Record<string, string>) {
  const targetClusters = clusterFaces(faces, threshold, clusters, assignments);
  const ids = new Map(targetClusters.flatMap(cluster => cluster.faceIds.map(id => [id, cluster.id] as const)));
  return { detectedFaces: faces.map(face => ({ ...face, clusterId: ids.get(face.id) ?? face.clusterId })), targetClusters };
}

let state: FaceSelectionStoreState = {
  poseFocus: null,
  detectedFaces: [],
  targetClusters: [],
  assignments: {},
  activeSelection: null,
  hoveredFaceId: null,
  selectedClusterId: null,

  setDetectedFaces: (faces: DetectedFace[]) => {
    state = {
      ...state,
      ...clusteredState(faces, 0.65, state.targetClusters, state.assignments),
    };
    notify();
  },

  addDetectedFaces: (faces: DetectedFace[]) => {
    const combined = [...state.detectedFaces, ...faces];
    state = {
      ...state,
      ...clusteredState(combined, 0.65, state.targetClusters, state.assignments),
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
    if (state.targetClusters.find(cluster => cluster.id === targetClusterId)?.mode !== 'swap') return;
    const before = faceEditState();
    const nextAssignments = { ...state.assignments };
    if (sourceFaceId) {
      nextAssignments[targetClusterId] = sourceFaceId;
    } else {
      delete nextAssignments[targetClusterId];
    }
    const nextClusters = state.targetClusters.map(c =>
      c.id === targetClusterId ? { ...c, sourceFaceId: sourceFaceId ?? null, sourceViewIndex: c.sourceFaceId === sourceFaceId ? c.sourceViewIndex : null } : c
    );
    state = {
      ...state,
      assignments: nextAssignments,
      targetClusters: nextClusters,
    };
    recordFaceEdit('Assign source identity', before);
  },

  setClusterMode: (targetClusterId: string, mode: 'swap' | 'ignore' | 'exclude') => {
    const before = faceEditState();
    const nextClusters = state.targetClusters.map(c =>
      c.id === targetClusterId ? { ...c, mode, sourceFaceId: mode === 'swap' ? c.sourceFaceId : null } : c
    );
    state = {
      ...state,
      targetClusters: nextClusters,
      assignments: Object.fromEntries(Object.entries(state.assignments).filter(([id]) => id !== targetClusterId || mode === 'swap')),
    };
    recordFaceEdit(`Set face role: ${mode === 'ignore' ? 'reference guide' : mode}`, before);
  },

  replaceAssignments: assignments => {
    const before = faceEditState();
    const next = Object.fromEntries(Object.entries(assignments).filter(([id, source]) => source && state.targetClusters.some(cluster => cluster.id === id && cluster.mode === 'swap')));
    state = { ...state, assignments: next, targetClusters: state.targetClusters.map(cluster => ({ ...cluster, sourceFaceId: next[cluster.id] ?? null, sourceViewIndex: cluster.sourceFaceId === next[cluster.id] ? cluster.sourceViewIndex : null })) };
    recordFaceEdit('Assign source identities', before);
  },
  setSourceView: (clusterId, view) => {
    const before = faceEditState();
    state = { ...state, targetClusters: state.targetClusters.map(cluster => cluster.id === clusterId && cluster.mode === 'swap' ? { ...cluster, sourceViewIndex: view } : cluster) };
    recordFaceEdit('Choose source face', before);
  },
  focusPose: faceId => {
    state = { ...state, activeSelection: faceId, poseFocus: { faceId, serial: (state.poseFocus?.serial ?? 0) + 1 } }; notify();
  },
  splitCluster: (clusterId, frame) => {
    const cluster = state.targetClusters.find(item => item.id === clusterId);
    if (!cluster || !Number.isInteger(frame) || frame <= (cluster.startFrame ?? 0) || cluster.endFrame != null && frame > cluster.endFrame) return false;
    const before = faceEditState(), id = `${cluster.originTrackId ?? cluster.id}_split_${frame}_${state.targetClusters.length}`;
    const faces = state.detectedFaces.map(face => face.clusterId === clusterId && face.frameIndex >= frame ? { ...face, clusterId: id } : face);
    const headFaces = faces.filter(face => face.clusterId === clusterId), tailFaces = faces.filter(face => face.clusterId === id);
    const head = { ...cluster, endFrame: frame - 1, splitLocked: true, faceIds: headFaces.map(face => face.id), frameCount: new Set(headFaces.map(face => face.frameIndex)).size };
    const tail = { ...cluster, id, name: `${cluster.name ?? cluster.id} from ${frame}`, originTrackId: cluster.originTrackId ?? cluster.id, startFrame: frame, splitLocked: true, faceIds: tailFaces.map(face => face.id), frameCount: new Set(tailFaces.map(face => face.frameIndex)).size };
    state = { ...state, detectedFaces: faces, targetClusters: [...state.targetClusters.filter(item => item.id !== clusterId), head, tail], assignments: { ...state.assignments, ...(state.assignments[clusterId] ? { [id]: state.assignments[clusterId] } : {}) } };
    recordFaceEdit(`Split cluster at frame ${frame}`, before); return true;
  },

  computeClusters: (similarityThreshold = 0.65) => {
    state = {
      ...state,
      ...clusteredState(state.detectedFaces, similarityThreshold, state.targetClusters, state.assignments),
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

    const validTrackIds = new Set(inspection.tracks.map(item => item.track_id));
    const sanitizedAssignments: Record<string, string> = {};
    for (const [trackId, facesetId] of Object.entries(existingAssignments)) {
      if (validTrackIds.has(trackId)) {
        sanitizedAssignments[trackId] = facesetId;
      }
    }

    const clusters = clusterFaces(faces, 0.65, undefined, sanitizedAssignments);
    for (const cluster of clusters) { const track = inspection.tracks.find(item => item.track_id === cluster.id); cluster.originTrackId = cluster.id; cluster.startFrame = track?.first_frame ?? 0; cluster.endFrame = track?.last_frame ?? null; }
    state = {
      ...state,
      detectedFaces: faces,
      targetClusters: clusters,
      assignments: sanitizedAssignments,
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
    // Resolve each track independently so an exact or nearby detection from
    // one face cannot hide another track whose detections are more sparse.
    const nearestByCluster = new Map<string, DetectedFace>();
    for (const cluster of state.targetClusters) {
      if (frameIndex < (cluster.startFrame ?? 0) || (cluster.endFrame != null && frameIndex > cluster.endFrame)) continue;
      const nearest = state.detectedFaces
        .filter(face => face.clusterId === cluster.id)
        .reduce<DetectedFace | null>((best, face) => !best || Math.abs(face.frameIndex - frameIndex) < Math.abs(best.frameIndex - frameIndex) ? face : best, null);
      if (nearest) nearestByCluster.set(cluster.id, nearest);
    }
    if (nearestByCluster.size > 0) return [...nearestByCluster.values()];

    // Keep the low-level fallback for callers that provide detections without
    // track metadata.
    const exact = state.detectedFaces.filter(face => face.frameIndex === frameIndex);
    if (exact.length > 0) return exact;
    return state.detectedFaces.filter(face => face.timestamp !== undefined
      ? Math.abs(face.timestamp - timestamp) <= 0.25
      : Math.abs(face.frameIndex - frameIndex) <= 2);
  },

  addFrameDetectedFaces: (faces: DetectedFaceItem[], autoSelect = true) => {
    if (!faces || faces.length === 0) return null;
    const newFaces: DetectedFace[] = faces.map(f => ({
      id: f.id,
      clusterId: f.track_id,
      frameIndex: f.frame_index,
      timestamp: f.timestamp,
      bbox: f.bbox,
      landmarks5: f.landmarks5,
      confidence: f.confidence,
      pose: f.pose,
      pose3d: f.pose3d,
      thumbnailUrl: f.thumbnail_base64 ? `data:image/jpeg;base64,${f.thumbnail_base64}` : undefined,
    }));
    const existingMap = new Map(state.detectedFaces.map(f => [f.id, f]));
    for (const nf of newFaces) {
      existingMap.set(nf.id, nf);
    }
    const combined = Array.from(existingMap.values());
    const clustered = clusteredState(combined, 0.65, state.targetClusters, state.assignments);
    for (const cluster of clustered.targetClusters) {
      if (cluster.startFrame === undefined) cluster.startFrame = 0;
      if (cluster.endFrame === undefined) cluster.endFrame = null;
    }
    const primaryId = autoSelect ? newFaces[0]?.id ?? null : state.activeSelection;
    const primaryCluster = primaryId
      ? clustered.detectedFaces.find(f => f.id === primaryId)?.clusterId ?? null
      : state.selectedClusterId;
    state = {
      ...state,
      ...clustered,
      activeSelection: primaryId,
      selectedClusterId: primaryCluster,
    };
    notify();
    return primaryId;
  },

  reset: () => {
    state = {
      ...state,
      poseFocus: null,
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

export function faceEditState() {
  return { detectedFaces: state.detectedFaces, targetClusters: state.targetClusters, assignments: state.assignments, activeSelection: state.activeSelection, selectedClusterId: state.selectedClusterId };
}
export function restoreFaceEdit(value: ReturnType<typeof faceEditState>) { state = { ...state, ...value }; notify(); }
function recordFaceEdit(label: string, before: ReturnType<typeof faceEditState>) {
  const after = faceEditState();
  if (JSON.stringify(before) === JSON.stringify(after)) return;
  executeCommand({ label, undo: () => restoreFaceEdit(before), redo: () => restoreFaceEdit(after) });
}
export function buildFaceSegments(clusters: TargetCluster[], assignments: Record<string, string>) {
  return clusters.map(cluster => ({ cluster_id: cluster.id, track_id: cluster.originTrackId ?? cluster.id,
    start_frame: cluster.startFrame ?? 0, end_frame: cluster.endFrame ?? null,
    role: cluster.mode === 'exclude' ? 'exclude' : cluster.mode === 'ignore' || !assignments[cluster.id] ? 'reference' : 'swap',
    faceset_id: cluster.mode === 'swap' ? assignments[cluster.id] ?? null : null, source_view_index: cluster.mode === 'swap' && assignments[cluster.id] ? cluster.sourceViewIndex ?? null : null }));
}
export function hydrateFaceSegments(segments: any[]) {
  const base = faceEditState();
  const clusters: TargetCluster[] = segments.map(segment => {
    const original = base.targetClusters.find(cluster => cluster.id === segment.track_id)!;
    return { ...original, id: segment.cluster_id, originTrackId: segment.track_id, startFrame: segment.start_frame,
      endFrame: segment.end_frame, mode: segment.role === 'reference' ? 'ignore' : segment.role,
      sourceViewIndex: segment.source_view_index, splitLocked: segment.cluster_id !== segment.track_id, sourceFaceId: segment.faceset_id };
  });
  const faces = base.detectedFaces.map(face => {
    const segment = segments.find(item => item.track_id === face.clusterId && face.frameIndex >= item.start_frame && (item.end_frame == null || face.frameIndex <= item.end_frame));
    return segment ? { ...face, clusterId: segment.cluster_id } : face;
  });
  restoreFaceEdit({ ...base, detectedFaces: faces, targetClusters: clusters, assignments: Object.fromEntries(segments.filter(item => item.role === 'swap' && item.faceset_id).map(item => [item.cluster_id, item.faceset_id])) });
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
