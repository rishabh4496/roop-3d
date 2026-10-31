import { describe, expect, it } from 'vitest';
import { renderToStaticMarkup } from 'react-dom/server';
import {
  crc32,
  buildZip,
  classifyPoseBucket,
  POSE_BUCKETS,
  cosineDistance,
  dbscanClustering,
  buildFaceSetManifest,
  computeMeanEmbedding,
} from './components/facemgr/fszBuilder';
import FaceManager from './components/facemgr/FaceManager';

describe('FaceSet Builder Utilities & Algorithms', () => {
  it('computes CRC-32 checksums accurately', () => {
    const encoder = new TextEncoder();
    const data = encoder.encode('123456789');
    // Standard CRC-32 for "123456789" is 0xcbf43926 (3421760294)
    expect(crc32(data)).toBe(0xcbf43926);
  });

  it('builds a valid ZIP container without external dependencies', async () => {
    const files = [
      { name: 'hello.txt', data: 'Hello World' },
      { name: '0.png', data: new Uint8Array([0x89, 0x50, 0x4e, 0x47]) },
    ];
    const zipBlob = buildZip(files);
    expect(zipBlob).toBeDefined();
    expect(zipBlob.type).toBe('application/zip');
    expect(zipBlob.size).toBeGreaterThan(60);

    const buf = new Uint8Array(await zipBlob.arrayBuffer());
    // Local file header signature 0x04034b50 -> bytes [0x50, 0x4b, 0x03, 0x04]
    expect(buf[0]).toBe(0x50);
    expect(buf[1]).toBe(0x4b);
    expect(buf[2]).toBe(0x03);
    expect(buf[3]).toBe(0x04);
  });

  it('classifies head pose angles into the 5 discrete Pose Angle Buckets', () => {
    // 1. Center Frontal (-10 <= yaw <= 10)
    expect(classifyPoseBucket(0, 0)).toBe(POSE_BUCKETS.FRONTAL);
    expect(classifyPoseBucket(10, 0)).toBe(POSE_BUCKETS.FRONTAL);
    expect(classifyPoseBucket(-10, 5)).toBe(POSE_BUCKETS.FRONTAL);

    // 2. Quarter Left (-35 <= yaw < -10)
    expect(classifyPoseBucket(-15, 0)).toBe(POSE_BUCKETS.QUARTER_LEFT);
    expect(classifyPoseBucket(-35, 2)).toBe(POSE_BUCKETS.QUARTER_LEFT);

    // 3. Quarter Right (10 < yaw <= 35)
    expect(classifyPoseBucket(15, 0)).toBe(POSE_BUCKETS.QUARTER_RIGHT);
    expect(classifyPoseBucket(35, -4)).toBe(POSE_BUCKETS.QUARTER_RIGHT);

    // 4. Full Profile (yaw < -35 or yaw > 35)
    expect(classifyPoseBucket(-45, 0)).toBe(POSE_BUCKETS.PROFILE);
    expect(classifyPoseBucket(60, 5)).toBe(POSE_BUCKETS.PROFILE);

    // 5. Pitch Inclines (pitch > 15 or pitch < -15)
    expect(classifyPoseBucket(0, 25)).toBe(POSE_BUCKETS.PITCH_INCLINE);
    expect(classifyPoseBucket(-15, -22)).toBe(POSE_BUCKETS.PITCH_INCLINE);
    expect(classifyPoseBucket(30, 20)).toBe(POSE_BUCKETS.PITCH_INCLINE);
  });

  it('computes cosine distance for ArcFace 512-d embeddings', () => {
    const v1 = new Array(512).fill(0);
    v1[0] = 1.0;
    const v2 = new Array(512).fill(0);
    v2[0] = 1.0;
    const v3 = new Array(512).fill(0);
    v3[1] = 1.0;

    expect(cosineDistance(v1, v2)).toBeCloseTo(0.0, 5); // Identical
    expect(cosineDistance(v1, v3)).toBeCloseTo(1.0, 5); // Orthogonal
  });

  it('clusters identities using DBSCAN on ArcFace embeddings', () => {
    // Person 1 embeddings (clustered around axis 0)
    const p1_a = new Array(512).fill(0); p1_a[0] = 0.95; p1_a[1] = 0.05;
    const p1_b = new Array(512).fill(0); p1_b[0] = 0.92; p1_b[1] = 0.08;

    // Person 2 embeddings (clustered around axis 10)
    const p2_a = new Array(512).fill(0); p2_a[10] = 0.98;
    const p2_b = new Array(512).fill(0); p2_b[10] = 0.96;

    const faces = [
      { id: 'f1', embedding: p1_a },
      { id: 'f2', embedding: p1_b },
      { id: 'f3', embedding: p2_a },
      { id: 'f4', embedding: p2_b },
    ];

    const clusters = dbscanClustering(faces, 0.35, 1);
    expect(clusters.length).toBe(2);
    expect(clusters[0].faces.map(f => f.id)).toEqual(['f1', 'f2']);
    expect(clusters[1].faces.map(f => f.id)).toEqual(['f3', 'f4']);
  });

  it('builds a compliant FaceSet V2 manifest', () => {
    const emb = new Array(512).fill(0); emb[0] = 1.0;
    const faces = [
      { id: 'f1', yaw: 0, pitch: 0, roll: 0, embedding: emb, confidence: 0.99 },
      { id: 'f2', yaw: 25, pitch: 0, roll: 0, embedding: emb, confidence: 0.95 },
    ];

    const manifest = buildFaceSetManifest('actor_1', 'Lead Actor', faces) as Record<string, any>;
    expect(manifest.schema_version).toBe(2);
    expect(manifest.schema).toBe('roop.fsz.v2');
    expect(manifest.faceset_id).toBe('actor_1');
    expect(manifest.source_name).toBe('Lead Actor');
    expect(manifest.identity.mean_embedding).toHaveLength(512);
    expect(manifest.identity.medoid_embedding).toHaveLength(512);
    expect(manifest.sources).toHaveLength(2);
    expect(manifest.pose_bins).toHaveProperty(POSE_BUCKETS.FRONTAL);
    expect(manifest.pose_bins).toHaveProperty(POSE_BUCKETS.QUARTER_RIGHT);
  });
});

describe('FaceManager Component Rendering', () => {
  it('renders FaceManager workstation with 5 Pose Angle Buckets and Person Clustering', () => {
    const html = renderToStaticMarkup(
      <FaceManager
        target="interview.mp4"
        inspection={{
          tracks: [
            {
              track_id: 'track_0000',
              first_frame: 0,
              last_frame: 100,
              pose_range: { yaw_degrees: [-5, 5], pitch_degrees: [0, 0] },
            },
            {
              track_id: 'track_0001',
              first_frame: 10,
              last_frame: 80,
              pose_range: { yaw_degrees: [25, 30], pitch_degrees: [-2, 0] },
            },
          ],
        }}
      />
    );

    // Header and title
    expect(html).toContain('Face Manager &amp; Multi-Angle Builder');
    expect(html).toContain('3D FACESET STUDIO');
    expect(html).toContain('Import Source Photos');
    expect(html).toContain('Save to FaceSet Library');

    // 5 Pose Angle Buckets columns
    expect(html).toContain('Center Frontal');
    expect(html).toContain('-10° ≤ Yaw ≤ +10°');
    expect(html).toContain('Quarter Left');
    expect(html).toContain('-35° ≤ Yaw &lt; -10°');
    expect(html).toContain('Quarter Right');
    expect(html).toContain('+10° &lt; Yaw ≤ +35°');
    expect(html).toContain('Full Profile');
    expect(html).toContain('Pitch Inclines');

    // Person Grouping & Clustering panel
    expect(html).toContain('IDENTITIES');
    expect(html).toContain('Cluster Tolerance (DBSCAN ε: 0.38)');
    expect(html).toContain('Person A');

    // 3D Geometry Unit Sphere readouts
    expect(html).toContain('GEOMETRY UNIT SPHERE');
    expect(html).toContain('Pose Lattice Coverage');
    expect(html).toContain('Observed View Bins');
    expect(html).toContain('&gt;3 Samples');
    expect(html).toContain('Blind Spot');

    // Batch assignment
    expect(html).toContain('BATCH ASSIGN REPLACEMENT');
    expect(html).toContain('Set as Face 0000');
  });
});
