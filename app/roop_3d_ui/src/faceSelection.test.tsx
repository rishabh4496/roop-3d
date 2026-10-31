import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  useFaceSelectionStore,
  cosineSimilarity,
  cosineDistance,
  clusterFaces,
} from './useFaceSelectionStore';
import {
  calculateLetterbox,
  mapBoxToPixels,
  mapPointToPixels,
} from './components/canvas/FaceOverlayCanvas';
import TargetFaceCard from './components/tray/TargetFaceCard';
import FaceSelectionTray from './components/tray/FaceSelectionTray';
import FaceOverlayCanvas from './components/canvas/FaceOverlayCanvas';
import type { DetectedFace, FaceSet, Inspection, Probe, TargetCluster } from './contracts';

describe('useFaceSelectionStore state architecture and clustering', () => {
  it('computes cosine distance and similarity correctly', () => {
    const vecA = [1, 0, 0];
    const vecB = [1, 0, 0];
    const vecC = [0, 1, 0];
    const vecD = [-1, 0, 0];

    // Orthogonal
    expect(cosineSimilarity(vecA, vecC)).toBe(0);
    expect(cosineDistance(vecA, vecC)).toBe(1);

    // Identical
    expect(cosineSimilarity(vecA, vecB)).toBe(1);
    expect(cosineDistance(vecA, vecB)).toBe(0);

    // Opposite
    expect(cosineSimilarity(vecA, vecD)).toBe(-1);
    expect(cosineDistance(vecA, vecD)).toBe(2);
  });

  it('clusters identities by cosine distance similarity threshold', () => {
    const face1: DetectedFace = {
      id: 'f1',
      frameIndex: 0,
      bbox: [0.1, 0.1, 0.3, 0.3],
      embedding: [1, 0, 0],
      clusterId: '',
      pose: { yaw: 2, pitch: 1, roll: 0 },
      confidence: 0.98,
    };
    const face2: DetectedFace = {
      id: 'f2',
      frameIndex: 1,
      bbox: [0.12, 0.1, 0.32, 0.3],
      embedding: [0.99, 0.05, 0], // very close to f1
      clusterId: '',
      pose: { yaw: 4, pitch: -1, roll: 0 },
      confidence: 0.95,
    };
    const face3: DetectedFace = {
      id: 'f3',
      frameIndex: 0,
      bbox: [0.7, 0.2, 0.9, 0.5],
      embedding: [0, 1, 0], // completely different identity
      clusterId: '',
      pose: { yaw: -15, pitch: 5, roll: 2 },
      confidence: 0.92,
    };

    const clusters = clusterFaces([face1, face2, face3], 0.8);
    expect(clusters.length).toBe(2);

    const cluster1 = clusters.find(c => c.faceIds.includes('f1'));
    const cluster2 = clusters.find(c => c.faceIds.includes('f3'));

    expect(cluster1).toBeDefined();
    expect(cluster2).toBeDefined();
    expect(cluster1?.faceIds).toContain('f2');
    expect(cluster1?.frameCount).toBe(2); // frame 0 and frame 1
    expect(cluster1?.avgPose.yaw).toBe(3); // (2 + 4) / 2
    expect(cluster1?.avgPose.pitch).toBe(0); // (1 + -1) / 2
  });

  it('maintains assignments and mode switches', () => {
    useFaceSelectionStore.getState().reset();
    const face: DetectedFace = {
      id: 'face_1',
      frameIndex: 5,
      timestamp: 0.166,
      bbox: [0.2, 0.2, 0.5, 0.5],
      clusterId: 'cluster_alpha',
      pose: { yaw: 0, pitch: 0, roll: 0 },
      confidence: 0.99,
    };

    useFaceSelectionStore.getState().setDetectedFaces([face]);
    expect(useFaceSelectionStore.getState().detectedFaces).toHaveLength(1);
    expect(useFaceSelectionStore.getState().targetClusters).toHaveLength(1);

    // Assignment
    useFaceSelectionStore.getState().assignSource('cluster_alpha', 'source_alice');
    expect(useFaceSelectionStore.getState().assignments['cluster_alpha']).toBe('source_alice');
    expect(useFaceSelectionStore.getState().targetClusters[0].sourceFaceId).toBe('source_alice');

    // Mode switch: "Swap / Ignore / Exclude"
    useFaceSelectionStore.getState().setClusterMode('cluster_alpha', 'exclude');
    expect(useFaceSelectionStore.getState().targetClusters[0].mode).toBe('exclude');

    useFaceSelectionStore.getState().setClusterMode('cluster_alpha', 'ignore');
    expect(useFaceSelectionStore.getState().targetClusters[0].mode).toBe('ignore');

    useFaceSelectionStore.getState().setClusterMode('cluster_alpha', 'swap');
    expect(useFaceSelectionStore.getState().targetClusters[0].mode).toBe('swap');

    // Active selection
    useFaceSelectionStore.getState().setActiveSelection('face_1');
    expect(useFaceSelectionStore.getState().activeSelection).toBe('face_1');

    // Query faces at timestamp
    const atTs = useFaceSelectionStore.getState().getFacesAtTimestamp(0.166);
    expect(atTs).toHaveLength(1);
    expect(atTs[0].id).toBe('face_1');
  });

  it('populates state faithfully from backend inspection tracks', () => {
    const inspection: Inspection = {
      inspection_id: '1234567890abcdef1234567890abcdef',
      target_sha256: 'a'.repeat(64),
      tracks: [
        {
          track_id: 'track_0000',
          first_frame: 0,
          last_frame: 60,
          first_timestamp_seconds: 0.0,
          last_timestamp_seconds: 2.0,
          thumbnail_base64: 'dGVzdA==',
          thumbnail_mime: 'image/jpeg',
          landmark_count: 5,
          pose_range: {
            yaw_degrees: [-5, 5],
            pitch_degrees: [-2, 4],
            roll_degrees: [0, 1],
            method: 'five_point_geometric_estimate',
          },
          bounding_box: { x1: 0.2, y1: 0.2, x2: 0.6, y2: 0.6 },
          sampled_detections: 4,
        },
      ],
      warnings: [],
      truncated: false,
    };

    const probe: Probe = {
      width: 1920,
      height: 1080,
      duration_seconds: 2.0,
      fps_rational: '30/1',
      declared_frame_count: 60,
      audio_tracks: [],
    };

    useFaceSelectionStore.getState().populateFromInspection(inspection, probe, { track_0000: 'alice' });

    const state = useFaceSelectionStore.getState();
    expect(state.detectedFaces.length).toBeGreaterThanOrEqual(4);
    expect(state.targetClusters).toHaveLength(1);
    expect(state.targetClusters[0].id).toBe('track_0000');
    expect(state.assignments['track_0000']).toBe('alice');
    expect(state.targetClusters[0].sourceFaceId).toBe('alice');
    expect(state.targetClusters[0].avgPose.yaw).toBe(0); // (-5 + 5) / 2
    expect(state.targetClusters[0].avgPose.pitch).toBe(1); // (-2 + 4) / 2
  });

  it('keeps target faces selectable between sparse samples, within the inspected track', () => {
    useFaceSelectionStore.getState().reset();
    const inspection: Inspection = {
      inspection_id: 'sparse-samples',
      target_sha256: 'b'.repeat(64),
      tracks: [
        {
          track_id: 'track_sparse',
          first_frame: 0,
          last_frame: 300,
          first_timestamp_seconds: 0,
          last_timestamp_seconds: 10,
          thumbnail_base64: 'dGVzdA==',
          thumbnail_mime: 'image/jpeg',
          landmark_count: 5,
          pose_range: { yaw_degrees: [0, 0], pitch_degrees: [0, 0], roll_degrees: [0, 0], method: 'test' },
          bounding_box: { x1: 0.2, y1: 0.2, x2: 0.6, y2: 0.6 },
          sampled_detections: 4,
        },
        {
          track_id: 'track_nearby',
          first_frame: 90,
          last_frame: 110,
          first_timestamp_seconds: 3,
          last_timestamp_seconds: 11 / 3,
          thumbnail_base64: 'dGVzdA==',
          thumbnail_mime: 'image/jpeg',
          landmark_count: 5,
          pose_range: { yaw_degrees: [0, 0], pitch_degrees: [0, 0], roll_degrees: [0, 0], method: 'test' },
          bounding_box: { x1: 0.4, y1: 0.2, x2: 0.8, y2: 0.6 },
          sampled_detections: 3,
        },
      ],
      warnings: [],
      truncated: false,
    };
    const probe: Probe = { width: 640, height: 360, duration_seconds: 10, fps_rational: '30/1', declared_frame_count: 300, audio_tracks: [] };

    useFaceSelectionStore.getState().populateFromInspection(inspection, probe);

    const betweenSamples = useFaceSelectionStore.getState().getFacesAtTimestamp(5, 30);
    expect(betweenSamples).toHaveLength(1);
    expect(betweenSamples[0].clusterId).toBe('track_sparse');
    const besideNearbyTrack = useFaceSelectionStore.getState().getFacesAtTimestamp(3.6, 30);
    expect(besideNearbyTrack.map(face => face.clusterId).sort()).toEqual(['track_nearby', 'track_sparse']);
    expect(useFaceSelectionStore.getState().getFacesAtTimestamp(11, 30)).toHaveLength(0);
  });
});

describe('Sub-pixel letterbox coordinate mapping for FaceOverlayCanvas', () => {
  it('accurately maps height-constrained pillarbox coordinates', () => {
    // 16:9 media (1920x1080) in ultrawide 21:9 container (2100x900)
    const containerW = 2100;
    const containerH = 900;
    const mediaW = 1920;
    const mediaH = 1080;

    const letterbox = calculateLetterbox(containerW, containerH, mediaW, mediaH);
    expect(letterbox.displayH).toBe(900);
    expect(letterbox.displayW).toBe(900 * (16 / 9)); // 1600
    expect(letterbox.offsetY).toBe(0);
    expect(letterbox.offsetX).toBe((2100 - 1600) / 2); // 250px pillarbox on each side

    // Normalized box [0.25, 0.2, 0.75, 0.8]
    const bbox: [number, number, number, number] = [0.25, 0.2, 0.75, 0.8];
    const pixels = mapBoxToPixels(bbox, letterbox);

    expect(pixels.x).toBeCloseTo(250 + 0.25 * 1600, 5); // 650
    expect(pixels.y).toBeCloseTo(0 + 0.2 * 900, 5); // 180
    expect(pixels.w).toBeCloseTo(0.5 * 1600, 5); // 800
    expect(pixels.h).toBeCloseTo(0.6 * 900, 5); // 540

    // ArcFace landmark sub-pixel mapping
    const landmark: [number, number] = [0.5, 0.5];
    const ptPixels = mapPointToPixels(landmark, letterbox);
    expect(ptPixels.x).toBe(250 + 0.5 * 1600); // 1050
    expect(ptPixels.y).toBe(0.5 * 900); // 450
  });

  it('accurately maps width-constrained letterbox coordinates', () => {
    // 16:9 media (1920x1080) in 4:3 container (800x600)
    const containerW = 800;
    const containerH = 600;
    const mediaW = 1920;
    const mediaH = 1080;

    const letterbox = calculateLetterbox(containerW, containerH, mediaW, mediaH);
    expect(letterbox.displayW).toBe(800);
    expect(letterbox.displayH).toBeCloseTo(800 / (16 / 9), 1); // 450
    expect(letterbox.offsetX).toBe(0);
    expect(letterbox.offsetY).toBe((600 - 450) / 2); // 75px letterbox top/bottom

    const bbox: [number, number, number, number] = [0, 0, 1, 1];
    const pixels = mapBoxToPixels(bbox, letterbox);
    expect(pixels.x).toBe(0);
    expect(pixels.y).toBe(75);
    expect(pixels.w).toBe(800);
    expect(pixels.h).toBe(450);
  });
});

describe('Face selection tray and target cards UI', () => {
  const sampleCluster: TargetCluster = {
    id: 'track_0001',
    name: 'Target Identity 1',
    faceIds: ['f1', 'f2'],
    frameCount: 42,
    avgPose: { yaw: 3.5, pitch: -1.2, roll: 0.1 },
    representativeFaceId: 'f1',
    thumbnailUrl: 'data:image/jpeg;base64,thumb123',
    mode: 'swap',
    sourceFaceId: 'src_alice',
  };

  const sampleSources: FaceSet[] = [
    { faceset_id: 'src_alice', name: 'Alice', item_count: 14 },
    { faceset_id: 'src_bob', name: 'Bob', item_count: 9 },
  ];

  it('renders TargetFaceCard with thumbnail, frame count, yaw/pitch, and mode toggle switch', () => {
    const html = renderToStaticMarkup(
      <TargetFaceCard
        cluster={sampleCluster}
        sources={sampleSources}
        assignedSourceId="src_alice"
        isActive={true}
        isSelected={false}
      />
    );

    // Thumbnail and frame count
    expect(html).toContain('src="data:image/jpeg;base64,thumb123"');
    expect(html).toContain('42f');
    expect(html).toContain('42 detected frames');

    // Yaw and pitch
    expect(html).toContain('+3.5°');
    expect(html).toContain('-1.2°');

    // Source assignment indicator
    expect(html).toContain('src_alice');
    expect(html).toContain('Alice');

    // "Swap / Ignore / Exclude" toggle switch
    expect(html).toContain('mode-btn swap active');
    expect(html).toContain('mode-btn ignore');
    expect(html).toContain('mode-btn exclude');

    // Active in frame highlight indicator
    expect(html).toContain('active-in-frame');
    expect(html).toContain('VISIBLE');
  });

  it('renders FaceSelectionTray and displays active identities', () => {
    useFaceSelectionStore.getState().reset();
    useFaceSelectionStore.getState().setDetectedFaces([
      {
        id: 'f_active',
        frameIndex: 10,
        timestamp: 0.5,
        bbox: [0.2, 0.2, 0.4, 0.4],
        clusterId: 'cluster_hero',
        pose: { yaw: 0, pitch: 0, roll: 0 },
        confidence: 0.99,
        thumbnailUrl: 'data:image/jpeg;base64,hero',
      },
    ]);

    const html = renderToStaticMarkup(
      <FaceSelectionTray
        currentTime={0.5}
        sources={sampleSources}
        fps={30}
      />
    );

    expect(html).toContain('TARGET IDENTITIES');
    expect(html).toContain('1 detected');
    expect(html).toContain('1 visible in current frame');
    expect(html).toContain('cluster_hero');
  });
  it('extracts and selects target faces from frame using addFrameDetectedFaces', () => {
    useFaceSelectionStore.getState().reset();
    const faceItems = [
      {
        id: 'track_0001_15',
        track_id: 'track_0001',
        frame_index: 15,
        timestamp: 0.5,
        bbox: [0.2, 0.2, 0.5, 0.5] as [number, number, number, number],
        landmarks5: [[0.3, 0.3], [0.4, 0.3], [0.35, 0.35], [0.32, 0.4], [0.38, 0.4]] as [number, number][],
        confidence: 0.98,
        pose: { yaw: 5.0, pitch: -2.0, roll: 0.5 },
        thumbnail_base64: 'abc123thumb',
      },
      {
        id: 'track_0002_15',
        track_id: 'track_0002',
        frame_index: 15,
        timestamp: 0.5,
        bbox: [0.6, 0.2, 0.85, 0.5] as [number, number, number, number],
        landmarks5: [[0.68, 0.3], [0.78, 0.3], [0.73, 0.35], [0.7, 0.4], [0.76, 0.4]] as [number, number][],
        confidence: 0.95,
        pose: { yaw: -12.0, pitch: 4.0, roll: -1.0 },
        thumbnail_base64: 'def456thumb',
      },
    ];

    const selectedId = useFaceSelectionStore.getState().addFrameDetectedFaces(faceItems, true);
    expect(selectedId).toBe('track_0001_15');
    const store = useFaceSelectionStore.getState();
    expect(store.activeSelection).toBe('track_0001_15');
    expect(store.selectedClusterId).toBe('track_0001');
    expect(store.detectedFaces.length).toBe(2);
    expect(store.targetClusters.length).toBe(2);

    // Verify clusters have startFrame 0 and endFrame null for frame detection
    const c1 = store.targetClusters.find(c => c.id === 'track_0001');
    expect(c1).toBeDefined();
    expect(c1?.thumbnailUrl).toContain('data:image/jpeg;base64,abc123thumb');

    // Retrieve faces at timestamp
    const visible = store.getFacesAtTimestamp(0.5, 30);
    expect(visible.length).toBe(2);
  });

  it('renders FaceOverlayCanvas preview target selector strip when multiple faces are visible', () => {
    useFaceSelectionStore.getState().reset();
    useFaceSelectionStore.getState().addFrameDetectedFaces([
      {
        id: 'f_p1',
        track_id: 'track_0001',
        frame_index: 10,
        timestamp: 0.33,
        bbox: [0.1, 0.1, 0.4, 0.4] as [number, number, number, number],
        landmarks5: [[0.2, 0.2], [0.3, 0.2], [0.25, 0.25], [0.22, 0.3], [0.28, 0.3]] as [number, number][],
        confidence: 0.99,
        pose: { yaw: 0, pitch: 0, roll: 0 },
        thumbnail_base64: 'p1',
      },
      {
        id: 'f_p2',
        track_id: 'track_0002',
        frame_index: 10,
        timestamp: 0.33,
        bbox: [0.6, 0.1, 0.9, 0.4] as [number, number, number, number],
        landmarks5: [[0.7, 0.2], [0.8, 0.2], [0.75, 0.25], [0.72, 0.3], [0.78, 0.3]] as [number, number][],
        confidence: 0.97,
        pose: { yaw: 0, pitch: 0, roll: 0 },
        thumbnail_base64: 'p2',
      },
    ], true);

    const html = renderToStaticMarkup(
      <FaceOverlayCanvas
        currentTime={0.33}
        fps={30}
        sources={sampleSources}
        mediaDimensions={{ width: 1920, height: 1080 }}
      />
    );

    expect(html).toContain('preview-face-selector-strip');
    expect(html).toContain('TARGET FACE:');
    expect(html).toContain('preview-face-pill');
  });

  it('renders FaceOverlayCanvas detection prompt when no faces detected and onExtractFaces provided', () => {
    useFaceSelectionStore.getState().reset();

    const html = renderToStaticMarkup(
      <FaceOverlayCanvas
        currentTime={2.5}
        fps={30}
        sources={sampleSources}
        onExtractFaces={() => {}}
        mediaDimensions={{ width: 1920, height: 1080 }}
      />
    );

    expect(html).toContain('preview-detect-prompt');
    expect(html).toContain('Use Face from this Frame');
  });
});
