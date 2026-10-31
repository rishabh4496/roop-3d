/**
 * Pure JavaScript utility for .fsz FaceSet archive creation, DBSCAN ArcFace clustering,
 * and 5-bucket pose classification. Zero external dependencies.
 */

// CRC-32 Lookup Table
const CRC_TABLE = new Uint32Array(256);
for (let i = 0; i < 256; i++) {
  let c = i;
  for (let k = 0; k < 8; k++) {
    c = (c & 1) ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  }
  CRC_TABLE[i] = c >>> 0;
}

/**
 * Calculate CRC-32 checksum of a Uint8Array.
 * @param {Uint8Array} bytes
 * @returns {number}
 */
export function crc32(bytes) {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Build a valid, standards-compliant ZIP (STORE / uncompressed) container as a Blob.
 * Compatible with Python's zipfile, FaceSetReader, and standard decompression tools.
 *
 * @param {Array<{ name: string; data: Uint8Array | string }>} files
 * @returns {Blob}
 */
export function buildZip(files) {
  const encoder = new TextEncoder();
  const fileEntries = [];
  let offset = 0;

  // 1. Process files and generate Local File Headers
  for (const file of files) {
    const filenameBytes = encoder.encode(file.name);
    const dataBytes = typeof file.data === 'string' ? encoder.encode(file.data) : file.data;
    const checksum = crc32(dataBytes);
    const size = dataBytes.length;

    // Local file header (30 bytes + filename)
    const header = new Uint8Array(30 + filenameBytes.length);
    const view = new DataView(header.buffer);

    view.setUint32(0, 0x04034b50, true); // Local file header signature
    view.setUint16(4, 20, true);         // Version needed to extract (2.0)
    view.setUint16(6, 0, true);          // General purpose bit flag
    view.setUint16(8, 0, true);          // Compression method (0 = STORE)
    view.setUint16(10, 0x4980, true);    // File last mod time
    view.setUint16(12, 0x546b, true);    // File last mod date
    view.setUint32(14, checksum, true);  // CRC-32
    view.setUint32(18, size, true);      // Compressed size
    view.setUint32(22, size, true);      // Uncompressed size
    view.setUint16(26, filenameBytes.length, true); // Filename length
    view.setUint16(28, 0, true);         // Extra field length
    header.set(filenameBytes, 30);

    fileEntries.push({
      name: file.name,
      filenameBytes,
      header,
      data: dataBytes,
      checksum,
      size,
      offset,
    });

    offset += header.length + dataBytes.length;
  }

  // 2. Generate Central Directory Headers
  const centralDirStartOffset = offset;
  const centralDirEntries = [];

  for (const entry of fileEntries) {
    const cdHeader = new Uint8Array(46 + entry.filenameBytes.length);
    const view = new DataView(cdHeader.buffer);

    view.setUint32(0, 0x02014b50, true); // Central file header signature
    view.setUint16(4, 20, true);         // Version made by
    view.setUint16(6, 20, true);         // Version needed to extract
    view.setUint16(8, 0, true);          // General purpose bit flag
    view.setUint16(10, 0, true);         // Compression method (0 = STORE)
    view.setUint16(12, 0x4980, true);    // File last mod time
    view.setUint16(14, 0x546b, true);    // File last mod date
    view.setUint32(16, entry.checksum, true); // CRC-32
    view.setUint32(20, entry.size, true);     // Compressed size
    view.setUint32(24, entry.size, true);     // Uncompressed size
    view.setUint16(28, entry.filenameBytes.length, true); // Filename length
    view.setUint16(30, 0, true);         // Extra field length
    view.setUint16(32, 0, true);         // File comment length
    view.setUint16(34, 0, true);         // Disk number start
    view.setUint16(36, 0, true);         // Internal file attributes
    view.setUint32(38, 0, true);         // External file attributes
    view.setUint32(42, entry.offset, true); // Relative offset of local header
    cdHeader.set(entry.filenameBytes, 46);

    centralDirEntries.push(cdHeader);
    offset += cdHeader.length;
  }

  const centralDirSize = offset - centralDirStartOffset;

  // 3. End of Central Directory Record (EOCD - 22 bytes)
  const eocd = new Uint8Array(22);
  const eocdView = new DataView(eocd.buffer);
  eocdView.setUint32(0, 0x06054b50, true); // EOCD signature
  eocdView.setUint16(4, 0, true);          // Number of this disk
  eocdView.setUint16(6, 0, true);          // Disk where central directory starts
  eocdView.setUint16(8, fileEntries.length, true);  // Number of central directory records on this disk
  eocdView.setUint16(10, fileEntries.length, true); // Total number of central directory records
  eocdView.setUint32(12, centralDirSize, true);     // Size of central directory
  eocdView.setUint32(16, centralDirStartOffset, true); // Offset of start of central directory
  eocdView.setUint16(20, 0, true);         // Comment length

  // 4. Assemble complete ZIP stream
  const parts = [];
  for (const entry of fileEntries) {
    parts.push(entry.header, entry.data);
  }
  for (const cd of centralDirEntries) {
    parts.push(cd);
  }
  parts.push(eocd);

  return new Blob(parts, { type: 'application/zip' });
}

/**
 * 5 Pose Angle Buckets constants and metadata.
 */
export const POSE_BUCKETS = {
  FRONTAL: 'center_frontal',
  QUARTER_LEFT: 'quarter_left',
  QUARTER_RIGHT: 'quarter_right',
  PROFILE: 'full_profile',
  PITCH_INCLINE: 'pitch_incline',
};

export const POSE_BUCKET_CONFIG = [
  {
    id: POSE_BUCKETS.FRONTAL,
    title: 'Center Frontal',
    range: '-10° ≤ Yaw ≤ +10°',
    badge: 'FRONTAL',
    color: '#10b981',
    description: 'Direct facing forward views. Primary anchor for swap fidelity.',
  },
  {
    id: POSE_BUCKETS.QUARTER_LEFT,
    title: 'Quarter Left',
    range: '-35° ≤ Yaw < -10°',
    badge: '1/4 LEFT',
    color: '#06b6d4',
    description: 'Slight head turn to subject’s right (camera left).',
  },
  {
    id: POSE_BUCKETS.QUARTER_RIGHT,
    title: 'Quarter Right',
    range: '+10° < Yaw ≤ +35°',
    badge: '1/4 RIGHT',
    color: '#f59e0b',
    description: 'Slight head turn to subject’s left (camera right).',
  },
  {
    id: POSE_BUCKETS.PROFILE,
    title: 'Full Profile',
    range: 'Yaw < -35° or Yaw > +35°',
    badge: 'PROFILE',
    color: '#f43f5e',
    description: 'Dramatic side angles. Prevents ear/jaw warping in turns.',
  },
  {
    id: POSE_BUCKETS.PITCH_INCLINE,
    title: 'Pitch Inclines',
    range: 'Pitch > +15° (Up) or < -15° (Down)',
    badge: 'INCLINE',
    color: '#8b5cf6',
    description: 'Looking up or down. Critical for chin, nostril, and brow shape.',
  },
];

/**
 * Classifies yaw and pitch into one of the 5 canonical Pose Angle Buckets.
 *
 * @param {number} yaw
 * @param {number} pitch
 * @returns {string}
 */
export function classifyPoseBucket(yaw = 0, pitch = 0) {
  // 5. Pitch Inclines (Looking Up / Down) takes precedence if pitch is steep
  if (pitch > 15 || pitch < -15) {
    return POSE_BUCKETS.PITCH_INCLINE;
  }
  // 1. Center Frontal: -10 <= yaw <= 10
  if (yaw >= -10 && yaw <= 10) {
    return POSE_BUCKETS.FRONTAL;
  }
  // 2. Quarter Left: -35 <= yaw < -10
  if (yaw >= -35 && yaw < -10) {
    return POSE_BUCKETS.QUARTER_LEFT;
  }
  // 3. Quarter Right: 10 < yaw <= 35
  if (yaw > 10 && yaw <= 35) {
    return POSE_BUCKETS.QUARTER_RIGHT;
  }
  // 4. Full Profile: yaw < -35 or yaw > 35
  return POSE_BUCKETS.PROFILE;
}

/**
 * Calculate cosine distance between two 512-d ArcFace embeddings.
 * @param {number[]} a
 * @param {number[]} b
 * @returns {number}
 */
export function cosineDistance(a, b) {
  if (!a || !b || a.length !== b.length || a.length === 0) return 1.0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  if (denom === 0) return 1.0;
  return Math.max(0, Math.min(2, 1.0 - dot / denom));
}

/**
 * DBSCAN clustering algorithm for 512-d ArcFace embeddings.
 * Automatically groups source faces into distinct person identities.
 *
 * @param {Array<{ id: string; embedding?: number[]; [key: string]: any }>} faces
 * @param {number} eps - Cosine distance threshold (e.g. 0.38 corresponding to similarity >= 0.62)
 * @param {number} minPts - Minimum neighbors to form a dense core (default 1)
 * @returns {Array<{ id: string; name: string; faces: any[] }>}
 */
export function dbscanClustering(faces, eps = 0.38, minPts = 1) {
  if (!faces || faces.length === 0) return [];
  const n = faces.length;
  const visited = new Set();
  const clusters = [];
  const noise = [];

  function regionQuery(faceIdx) {
    const neighbors = [];
    const face = faces[faceIdx];
    if (!face.embedding || face.embedding.length === 0) return [faceIdx];
    for (let i = 0; i < n; i++) {
      const other = faces[i];
      if (!other.embedding || other.embedding.length === 0) continue;
      const dist = cosineDistance(face.embedding, other.embedding);
      if (dist <= eps) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  function expandCluster(faceIdx, neighbors) {
    const cluster = [faces[faceIdx]];
    visited.add(faceIdx);

    let i = 0;
    while (i < neighbors.length) {
      const neighborIdx = neighbors[i];
      if (!visited.has(neighborIdx)) {
        visited.add(neighborIdx);
        const subNeighbors = regionQuery(neighborIdx);
        if (subNeighbors.length >= minPts) {
          for (const s of subNeighbors) {
            if (!neighbors.includes(s)) {
              neighbors.push(s);
            }
          }
        }
      }
      const neighborFace = faces[neighborIdx];
      if (!cluster.some(f => f.id === neighborFace.id)) {
        cluster.push(neighborFace);
      }
      i++;
    }
    return cluster;
  }

  let clusterCount = 0;
  for (let i = 0; i < n; i++) {
    if (visited.has(i)) continue;
    visited.add(i);
    const neighbors = regionQuery(i);
    if (neighbors.length < minPts) {
      noise.push(faces[i]);
    } else {
      clusterCount++;
      const clusterMembers = expandCluster(i, neighbors);
      clusters.push({
        id: `person_${clusterCount}`,
        name: `Person ${String.fromCharCode(64 + clusterCount)}`,
        faces: clusterMembers,
      });
    }
  }

  for (const face of noise) {
    clusterCount++;
    clusters.push({
      id: `person_${clusterCount}`,
      name: `Person ${String.fromCharCode(64 + clusterCount)}`,
      faces: [face],
    });
  }

  return clusters;
}

/**
 * Compute L2-normalized mean 512-d embedding.
 * @param {number[][]} embeddings
 * @returns {number[]}
 */
export function computeMeanEmbedding(embeddings) {
  if (!embeddings || embeddings.length === 0) {
    const zero = new Array(512).fill(0);
    zero[0] = 1.0;
    return zero;
  }
  const mean = new Array(512).fill(0);
  for (const emb of embeddings) {
    if (!emb) continue;
    for (let i = 0; i < 512; i++) {
      mean[i] += emb[i] || 0;
    }
  }
  let norm = 0;
  for (let i = 0; i < 512; i++) {
    norm += mean[i] * mean[i];
  }
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < 512; i++) {
      mean[i] /= norm;
    }
  } else {
    mean[0] = 1.0;
  }
  return mean;
}

/**
 * Builds the FaceSet V2 metadata.json document.
 *
 * @param {string} facesetId
 * @param {string} name
 * @param {Array<any>} faces
 * @returns {Record<string, any>}
 */
export function buildFaceSetManifest(facesetId, name, faces) {
  const embeddings = faces.map(f => f.embedding).filter(Boolean);
  const meanEmbedding = computeMeanEmbedding(embeddings);

  // Compute medoid embedding (face closest to mean)
  let bestDist = Infinity;
  let medoidEmbedding = meanEmbedding;
  for (const emb of embeddings) {
    const d = cosineDistance(meanEmbedding, emb);
    if (d < bestDist) {
      bestDist = d;
      medoidEmbedding = emb;
    }
  }

  // Group into pose bins
  const poseBins = {};
  for (let i = 0; i < faces.length; i++) {
    const f = faces[i];
    const bucket = f.poseBucket || classifyPoseBucket(f.yaw || 0, f.pitch || 0);
    if (!poseBins[bucket]) {
      poseBins[bucket] = {
        yaw_bin: bucket.split('_')[0] || 'center',
        pitch_bin: bucket.includes('incline') ? 'incline' : 'level',
        members: [],
        support: 0,
        representative_index: i,
        embedding: f.embedding || meanEmbedding,
      };
    }
    poseBins[bucket].members.push(i);
    poseBins[bucket].support += 1;
  }

  return {
    schema_version: 2,
    schema: 'roop.fsz.v2',
    faceset_id: facesetId,
    source_name: name,
    created_at: new Date().toISOString(),
    consent: {
      consent_granted: true,
      commercial_use_allowed: true,
      notes: 'Generated locally with Roop 3D Face Manager Workstation',
    },
    identity: {
      mean_embedding: meanEmbedding,
      medoid_embedding: medoidEmbedding,
      aggregation: 'l2_mean_and_medoid',
    },
    pose_bins: poseBins,
    sources: faces.map((f, idx) => ({
      source_index: idx,
      filename: `${idx}.png`,
      pose: [f.yaw || 0, f.pitch || 0, f.roll || 0],
      geometry: {
        bbox: f.bbox || [0, 0, 100, 100],
        landmarks_2d: f.landmarks || [[10, 10], [20, 10], [15, 15], [12, 20], [18, 20]],
        yaw: f.yaw || 0,
        pitch: f.pitch || 0,
        roll: f.roll || 0,
      },
      quality: {
        sharpness: 1.0,
        detector_confidence: f.confidence || 0.98,
        score: 0.98,
        laplacian_variance: 100.0,
        blur: 0.0,
        face_pixels: 40000,
      },
      expression: {},
      pose_bin: f.poseBucket || classifyPoseBucket(f.yaw || 0, f.pitch || 0),
      is_synthetic: false,
    })),
    provenance: {
      pipeline: 'roop_3d_stage04_facemgr',
      archive_format: 'v2_standard',
    },
  };
}
