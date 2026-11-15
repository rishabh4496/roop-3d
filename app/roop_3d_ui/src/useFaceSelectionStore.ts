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
  /** Keeps a stored face's pose current with the frame it is on screen in (the 3D viewer's Live Pose reads it). */
  updateFacePose: (faceId: string, pose: { yaw: number; pitch: number; roll: number }, pose3d?: { yaw: number; pitch: number; roll: number }) => void;
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
  const selectableFaces = faces.filter(face => face.selectable !== false);
  if (selectableFaces.length === 0) return [];

  const clustersMap = new Map<string, DetectedFace[]>();
  const protectedCluster = (face: DetectedFace) => existingClusters?.find(cluster => cluster.id === face.clusterId && (cluster.mode !== 'swap' || cluster.splitLocked));

  // If faces already have embeddings, group by cosine distance
  const hasEmbeddings = faces.some(f => f.embedding && f.embedding.length > 0);

  if (hasEmbeddings) {
    const visited = new Set<string>();
    let autoClusterIdx = 0;

    for (let i = 0; i < selectableFaces.length; i++) {
      const face = selectableFaces[i];
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
        for (let j = i + 1; j < selectableFaces.length; j++) {
          const other = selectableFaces[j];
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
    for (const face of selectableFaces) {
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
      name: prev?.name ?? `Face ${clusterId.startsWith('track_') ? Number(clusterId.slice(6)) : clusterIndex - 1}`,
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
  return { detectedFaces: faces.map(face => ({ ...face, clusterId: face.selectable === false ? '' : ids.get(face.id) ?? face.clusterId })), targetClusters };
}

/** The scan the workstation currently holds: frame-face requests bind their results to ITS tracks. */
let activeInspectionId: string | null = null;
export const getActiveInspectionId = (): string | null => activeInspectionId;

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
    if (face && (face.selectable === false || !face.clusterId)) return;
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

  updateFacePose: (faceId, pose, pose3d) => {
    const face = state.detectedFaces.find(item => item.id === faceId);
    if (!face) return;
    const same = (a?: { yaw: number; pitch: number; roll: number }, b?: { yaw: number; pitch: number; roll: number }) => !!a === !!b && (!a || (a.yaw === b!.yaw && a.pitch === b!.pitch && a.roll === b!.roll));
    if (same(face.pose, pose) && same(face.pose3d, pose3d)) return;
    state = { ...state, detectedFaces: state.detectedFaces.map(item => (item.id === faceId ? { ...item, pose, ...(pose3d ? { pose3d } : {}) } : item)) };
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
    notify();
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
    notify();
  },

  replaceAssignments: assignments => {
    const before = faceEditState();
    const next = Object.fromEntries(
      Object.entries(assignments).filter(
        ([id, source]) => source && (!state.targetClusters.length || state.targetClusters.some(cluster => (cluster.id === id || cluster.originTrackId === id) && cluster.mode === 'swap'))
      )
    );
    state = {
      ...state,
      assignments: next,
      targetClusters: state.targetClusters.map(cluster => ({
        ...cluster,
        sourceFaceId: next[cluster.id] ?? next[cluster.originTrackId ?? ''] ?? null,
        sourceViewIndex: cluster.sourceFaceId === next[cluster.id] ? cluster.sourceViewIndex : null
      }))
    };
    recordFaceEdit('Assign source identities', before);
    notify();
  },
  setSourceView: (clusterId, view) => {
    const before = faceEditState();
    state = { ...state, targetClusters: state.targetClusters.map(cluster => cluster.id === clusterId && cluster.mode === 'swap' ? { ...cluster, sourceViewIndex: view } : cluster) };
    recordFaceEdit('Choose source face', before);
    notify();
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
    recordFaceEdit(`Split cluster at frame ${frame}`, before);
    notify();
    return true;
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
    activeInspectionId = inspection?.inspection_id ?? null;
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

    for (const track of inspection.tracks) {
      const thumb = track.thumbnail_base64 ? `data:${track.thumbnail_mime || 'image/jpeg'};base64,${track.thumbnail_base64}` : undefined;
      const mean = (range: [number, number] | null | undefined) => range ? (range[0] + range[1]) / 2 : 0;
      const trackPose = { yaw: mean(track.pose_range.yaw_degrees), pitch: mean(track.pose_range.pitch_degrees), roll: mean(track.pose_range.roll_degrees) };

      // Every face is a detection the scan really made in that frame: its own box, landmarks, pose and score. Ids are
      // unique per (track, frame), so a selection can never resolve to another sample of the same track.
      const seen = new Set<number>();
      for (const observation of track.observations ?? []) {
        if (seen.has(observation.frame)) continue;
        seen.add(observation.frame);
        faces.push({
          id: `${track.track_id}_${observation.frame}`,
          frameIndex: observation.frame,
          timestamp: observation.timestamp_seconds,
          bbox: observation.bbox,
          landmarks5: observation.landmarks5.length === 5 ? observation.landmarks5 : undefined,
          clusterId: track.track_id,
          pose: { yaw: observation.yaw_degrees, pitch: observation.pitch_degrees, roll: observation.roll_degrees },
          confidence: observation.confidence,
          thumbnailUrl: thumb,
        });
      }
      if (seen.size > 0) continue;

      // A scan saved before observations existed knows one real box: where the track first appeared. It is shown at that
      // frame only; nothing is interpolated, no landmarks or confidence are invented.
      const box = track.bounding_box;
      if (!box) continue;
      const pixels = box.x2 > 1.0 || box.y2 > 1.0;
      if (pixels && !(probe && probe.width > 0 && probe.height > 0)) continue;
      const bbox: [number, number, number, number] = pixels && probe
        ? [box.x1 / probe.width, box.y1 / probe.height, box.x2 / probe.width, box.y2 / probe.height]
        : [box.x1, box.y1, box.x2, box.y2];
      faces.push({
        id: `${track.track_id}_${track.first_frame}`,
        frameIndex: track.first_frame,
        timestamp: track.first_timestamp_seconds,
        bbox,
        clusterId: track.track_id,
        pose: trackPose,
        thumbnailUrl: thumb,
      });
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
      clusterId: f.track_id ?? '',
      frameIndex: f.frame_index,
      timestamp: f.timestamp,
      bbox: f.bbox,
      landmarks5: f.landmarks5,
      confidence: f.confidence,
      pose: f.pose,
      pose3d: f.pose3d,
      thumbnailUrl: f.thumbnail_base64 ? `data:image/jpeg;base64,${f.thumbnail_base64}` : undefined,
      selectable: f.selectable !== false && !!f.track_id,
      identityState: f.identity_state,
      uncertaintyReason: f.uncertainty_reason,
      clusterMatch: f.cluster_match,
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
    const requestedPrimaryId = autoSelect ? newFaces.find(face => face.selectable !== false && !!face.clusterId)?.id ?? null : state.activeSelection;
    const primaryFace = requestedPrimaryId ? clustered.detectedFaces.find(face => face.id === requestedPrimaryId) : null;
    const primaryId = primaryFace && primaryFace.selectable !== false && primaryFace.clusterId ? requestedPrimaryId : null;
    const primaryCluster = primaryId ? primaryFace?.clusterId ?? null : null;
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
    activeInspectionId = null;
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
