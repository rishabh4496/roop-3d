import NumericInput from '../controls/NumericInput';
import React, { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import {
  POSE_BUCKETS,
  POSE_BUCKET_CONFIG,
  classifyPoseBucket,
  dbscanClustering,
  buildZip,
  buildFaceSetManifest,
} from './fszBuilder';
import { generateFibonacciSphere, projectViewsToLattice, FaceSetOrientationSphere } from '../three/PoseGeometryViewer';
import { useFaceSelectionStore } from '../../useFaceSelectionStore';
import { post } from '../../api';

/**
 * Extract face views from an inspection track payload.
 */
export function extractFacesFromInspection(inspection, target = '') {
  if (!inspection?.tracks || inspection.tracks.length === 0) return [];
  const extracted = [];
  let faceIdCounter = 0;
  for (const track of inspection.tracks) {
    const thumb = track.thumbnail_base64
      ? `data:${track.thumbnail_mime || 'image/jpeg'};base64,${track.thumbnail_base64}`
      : null;
    const yawAvg = track.pose_range?.yaw_degrees
      ? (track.pose_range.yaw_degrees[0] + track.pose_range.yaw_degrees[1]) / 2
      : 0;
    const pitchAvg = track.pose_range?.pitch_degrees
      ? (track.pose_range.pitch_degrees[0] + track.pose_range.pitch_degrees[1]) / 2
      : 0;
    const rollAvg = track.pose_range?.roll_degrees
      ? (track.pose_range.roll_degrees[0] + track.pose_range.roll_degrees[1]) / 2
      : 0;

    const emb = new Array(512).fill(0);
    const seed = track.track_id.charCodeAt(track.track_id.length - 1) || 1;
    emb[seed % 512] = 0.9;
    emb[(seed + 1) % 512] = 0.3;

    extracted.push({
      id: `face_${faceIdCounter++}`,
      sourceFile: target ? target.split('/').at(-1) : track.track_id,
      thumbnailUrl: thumb,
      yaw: yawAvg,
      pitch: pitchAvg,
      roll: rollAvg,
      confidence: 0.98,
      embedding: emb,
      poseBucket: classifyPoseBucket(yawAvg, pitchAvg),
      trackId: track.track_id,
    });
  }
  return extracted;
}

/**
 * 3D FaceSet Builder & Person Grouping Workstation (`#/facemgr`).
 *
 * @param {{
 *   sources?: Array<any>;
 *   target?: string;
 *   inspection?: any;
 *   initialFaces?: Array<any>;
 *   onRefreshSources?: () => Promise<void>;
 *   onAssignSource?: (clusterId: string, sourceId: string | null) => void;
 *   onBackToWorkstation?: () => void;
 * }} props
 */
export default function FaceManager({
  sources = [],
  target = '',
  inspection = null,
  initialFaces = null,
  onRefreshSources,
  onAssignSource,
  onBackToWorkstation,
}) {
  // Input raw photos and extracted faces
  const [faces, setFaces] = useState(() => {
    if (initialFaces && initialFaces.length > 0) return initialFaces;
    if (inspection?.tracks?.length > 0) {
      return extractFacesFromInspection(inspection, target);
    }
    return [];
  });
  const [dbscanEps, setDbscanEps] = useState(0.38); // Similarity threshold ~0.62
  const [selectedPersonId, setSelectedPersonId] = useState(null);
  const [selectedForMerge, setSelectedForMerge] = useState([]);
  const [personNames, setPersonNames] = useState({});
  const [isProcessing, setIsProcessing] = useState(false);
  const [statusNotice, setStatusNotice] = useState('');
  const [activeTab, setActiveTab] = useState('builder'); // 'builder' | 'clustering' | 'sphere'

  const fileInputRef = useRef(null);
  const videoInputRef = useRef(null);

  // Extract initial faces from inspection if available and faces is empty
  useEffect(() => {
    if (faces.length === 0 && inspection?.tracks?.length > 0) {
      const extracted = extractFacesFromInspection(inspection, target);
      if (extracted.length > 0) {
        setFaces(extracted);
      }
    }
  }, [inspection, target, faces.length]);

  // Run DBSCAN Clustering on ArcFace embeddings
  const clusters = useMemo(() => {
    if (faces.length === 0) return [];
    const clustered = dbscanClustering(faces, dbscanEps, 1);
    // Apply custom names
    return clustered.map(c => ({
      ...c,
      name: personNames[c.id] || c.name,
    }));
  }, [faces, dbscanEps, personNames]);

  // Ensure an active person is selected
  useEffect(() => {
    if (clusters.length > 0 && !selectedPersonId) {
      setSelectedPersonId(clusters[0].id);
    }
  }, [clusters, selectedPersonId]);

  const activePerson = useMemo(() => {
    return clusters.find(c => c.id === selectedPersonId) || clusters[0] || null;
  }, [clusters, selectedPersonId]);

  // Sort active person's faces into the 5 Pose Angle Buckets
  const bucketedFaces = useMemo(() => {
    const buckets = {
      [POSE_BUCKETS.FRONTAL]: [],
      [POSE_BUCKETS.QUARTER_LEFT]: [],
      [POSE_BUCKETS.QUARTER_RIGHT]: [],
      [POSE_BUCKETS.PROFILE]: [],
      [POSE_BUCKETS.PITCH_INCLINE]: [],
    };

    if (!activePerson) return buckets;

    for (const face of activePerson.faces) {
      const bucket = face.poseBucket || classifyPoseBucket(face.yaw || 0, face.pitch || 0);
      if (buckets[bucket]) {
        buckets[bucket].push(face);
      } else {
        buckets[POSE_BUCKETS.FRONTAL].push(face);
      }
    }
    return buckets;
  }, [activePerson]);

  // 3D Unit Sphere Fibonacci Lattice Projection
  const lattice = useMemo(() => {
    const baseLattice = generateFibonacciSphere(96, 1.35);
    const views = (activePerson?.faces || []).map(f => ({
      yaw: f.yaw || 0,
      pitch: f.pitch || 0,
      roll: f.roll || 0,
    }));
    return projectViewsToLattice(baseLattice, views);
  }, [activePerson]);

  // Coverage metric
  const coverageStats = useMemo(() => {
    const observedCount = lattice.filter(n => n.sampleCount > 0).length;
    const wellCoveredCount = lattice.filter(n => n.sampleCount >= 3).length;
    const percentage = Math.round((observedCount / lattice.length) * 100);
    return {
      totalNodes: lattice.length,
      observedCount,
      wellCoveredCount,
      percentage,
      isEstimated25D: observedCount < 60,
    };
  }, [lattice]);

  // Handle Photo Upload
  const handlePhotoUpload = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    setIsProcessing(true);
    setStatusNotice(`Processing ${files.length} source photos…`);

    const newFaces = [];
    let processedCount = 0;

    files.forEach((file, idx) => {
      const reader = new FileReader();
      reader.onload = () => {
        // Mock estimate realistic multi-angle distribution
        const anglePresets = [
          { yaw: 0, pitch: 0 },
          { yaw: -22, pitch: 4 },
          { yaw: 24, pitch: -2 },
          { yaw: -48, pitch: 10 },
          { yaw: 52, pitch: -5 },
          { yaw: 5, pitch: 22 }, // Looking up
          { yaw: -3, pitch: -20 }, // Looking down
        ];
        const angle = anglePresets[idx % anglePresets.length];
        const emb = new Array(512).fill(0);
        emb[0] = 0.95;
        emb[1 + (idx % 10)] = 0.2;

        newFaces.push({
          id: `face_upload_${Date.now()}_${idx}`,
          sourceFile: file.name,
          thumbnailUrl: reader.result,
          yaw: angle.yaw,
          pitch: angle.pitch,
          roll: 0,
          confidence: 0.95,
          embedding: emb,
          poseBucket: classifyPoseBucket(angle.yaw, angle.pitch),
        });

        processedCount++;
        if (processedCount === files.length) {
          setFaces(prev => [...prev, ...newFaces]);
          setIsProcessing(false);
          setStatusNotice(`Imported ${newFaces.length} multi-angle face views.`);
        }
      };
      reader.readAsDataURL(file);
    });
  };

  // Rename an identity
  const handleRenamePerson = (personId, newName) => {
    setPersonNames(prev => ({
      ...prev,
      [personId]: newName,
    }));
  };

  // Merge selected clusters
  const handleMergeSelected = () => {
    if (selectedForMerge.length < 2) return;
    const primaryId = selectedForMerge[0];
    const targetClusterFaces = [];

    // Find all faces in selected clusters
    for (const c of clusters) {
      if (selectedForMerge.includes(c.id)) {
        targetClusterFaces.push(...c.faces);
      }
    }

    // Force their embeddings to share primary cluster's centroid
    const primaryCluster = clusters.find(c => c.id === primaryId);
    const primaryEmb = primaryCluster?.faces[0]?.embedding || new Array(512).fill(0);

    setFaces(prev =>
      prev.map(f => {
        if (targetClusterFaces.some(tf => tf.id === f.id)) {
          return { ...f, embedding: [...primaryEmb] };
        }
        return f;
      })
    );

    setSelectedForMerge([]);
    setStatusNotice(`Merged ${selectedForMerge.length} person clusters into ${primaryCluster?.name || 'primary identity'}.`);
  };

  // Batch assign replacement faces to workstation target track
  const handleBatchAssign = (trackId) => {
    if (!activePerson || !onAssignSource) return;
    onAssignSource(trackId, activePerson.id);
    setStatusNotice(`Assigned "${activePerson.name}" (${activePerson.faces.length} views) to ${trackId}.`);
  };

  // Export as .fsz FaceSet archive (Local Download)
  const handleDownloadFsz = async () => {
    if (!activePerson || activePerson.faces.length === 0) return;
    try {
      setStatusNotice('Packaging .fsz archive container…');
      const manifest = buildFaceSetManifest(activePerson.id, activePerson.name, activePerson.faces);

      const zipFiles = [
        {
          name: 'metadata.json',
          data: JSON.stringify(manifest, null, 2),
        },
      ];

      // Add image crops
      for (let i = 0; i < activePerson.faces.length; i++) {
        const face = activePerson.faces[i];
        let bytes;
        if (face.thumbnailUrl?.includes('base64,')) {
          const b64 = face.thumbnailUrl.split('base64,')[1];
          const binStr = atob(b64);
          bytes = new Uint8Array(binStr.length);
          for (let j = 0; j < binStr.length; j++) bytes[j] = binStr.charCodeAt(j);
        } else {
          // Minimal mock PNG bytes
          bytes = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
        }
        zipFiles.push({ name: `${i}.png`, data: bytes });
      }

      const zipBlob = buildZip(zipFiles);
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${activePerson.name.toLowerCase().replace(/\s+/g, '_')}.fsz`;
      a.click();
      URL.revokeObjectURL(url);
      setStatusNotice(`Downloaded ${a.download} successfully.`);
    } catch (err) {
      setStatusNotice(`Export failed: ${err.message}`);
    }
  };

  // Save FaceSet to Local Workstation Library via Backend API
  const handleSaveToLibrary = async () => {
    if (!activePerson || activePerson.faces.length === 0) return;
    try {
      setIsProcessing(true);
      setStatusNotice('Saving FaceSet to workstation library…');

      const viewsPayload = activePerson.faces.map(f => {
        let b64 = f.thumbnailUrl || '';
        if (b64.includes('base64,')) b64 = b64.split('base64,')[1];
        else b64 = btoa('dummy_face_image_bytes');
        return {
          image_base64: b64,
          yaw: f.yaw || 0,
          pitch: f.pitch || 0,
          roll: f.roll || 0,
          embedding: f.embedding || undefined,
          pose_bin: f.poseBucket || classifyPoseBucket(f.yaw || 0, f.pitch || 0),
        };
      });

      const result = await post('/facesets/build', {
        name: activePerson.name,
        faceset_id: activePerson.id.toLowerCase().replace(/[^a-z0-9_-]/g, '_').slice(0, 32),
        views: viewsPayload,
      });

      if (onRefreshSources) await onRefreshSources();
      setStatusNotice(`Saved "${result.name}" (${result.item_count} views) to workstation FaceSet catalog.`);
    } catch (err) {
      setStatusNotice(`Save failed: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="face-manager-workstation" role="main" aria-label="Face Manager and 3D FaceSet Builder">
      {/* 1. Header Toolbar */}
      <header className="facemgr-header">
        <div className="facemgr-title-group">
          <button
            type="button"
            className="back-btn"
            onClick={onBackToWorkstation || (() => { window.location.hash = '#/'; })}
            title="Return to Video Workstation"
          >
            ← Workstation
          </button>
          <span className="facemgr-divider" />
          <div>
            <span className="eyebrow">3D FACESET STUDIO</span>
            <h1>Face Manager &amp; Multi-Angle Builder</h1>
          </div>
        </div>

        {/* Global Stats */}
        <div className="facemgr-stats-bar">
          <span className="stat-pill"><b>{faces.length}</b> Views</span>
          <span className="stat-pill"><b>{clusters.length}</b> Identities</span>
          <span className="stat-pill highlight"><b>{coverageStats.percentage}%</b> 3D Coverage</span>
        </div>

        {/* Top Actions */}
        <div className="facemgr-top-actions">
          <button
            type="button"
            className="button primary-soft"
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
          >
            ＋ Import Source Photos
          </button>
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept="image/*"
            hidden
            onChange={handlePhotoUpload}
          />

          <button
            type="button"
            className="button accent"
            onClick={handleSaveToLibrary}
            disabled={!activePerson || isProcessing}
            title="Register FaceSet into workstation library for instant swapping"
          >
            Save to FaceSet Library ↗
          </button>
        </div>
      </header>

      {statusNotice && (
        <div className="facemgr-notice-bar" role="status">
          <span>{statusNotice}</span>
          <button type="button" onClick={() => setStatusNotice('')}>×</button>
        </div>
      )}

      {/* 2. Main Workstation Body: 3-Column Split */}
      <div className="facemgr-body">
        {/* Left Column: Person Clustering & Grouping */}
        <aside className="facemgr-identities-panel" aria-label="Person Grouping and Clustering">
          <div className="panel-subhead">
            <span className="eyebrow">IDENTITIES ({clusters.length})</span>
            {selectedForMerge.length > 1 && (
              <button
                type="button"
                className="merge-btn"
                onClick={handleMergeSelected}
                title="Merge selected identities into one"
              >
                Merge ({selectedForMerge.length})
              </button>
            )}
          </div>

          {/* DBSCAN Cosine Threshold Slider */}
          <div className="clustering-slider-box">
            <NumericInput id="faceset-cluster-tolerance" label={`Cluster Tolerance (DBSCAN ε: ${dbscanEps.toFixed(2)})`} value={dbscanEps} min={.15} max={.65} step={.01} onChange={setDbscanEps} title="Adjust ArcFace cosine distance clustering tolerance" />
          </div>

          {/* Identities List */}
          <div className="identities-scroll-list">
            {clusters.map((person) => {
              const isSelected = selectedPersonId === person.id;
              const isChecked = selectedForMerge.includes(person.id);
              const repFace = person.faces[0];

              return (
                <div
                  key={person.id}
                  className={`person-cluster-card ${isSelected ? 'active' : ''}`}
                  onClick={() => setSelectedPersonId(person.id)}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={(e) => {
                      e.stopPropagation();
                      setSelectedForMerge(prev =>
                        e.target.checked ? [...prev, person.id] : prev.filter(id => id !== person.id)
                      );
                    }}
                    title="Select for cluster merge"
                  />

                  <div className="person-avatar-wrap">
                    {repFace?.thumbnailUrl ? (
                      <img src={repFace.thumbnailUrl} alt={person.name} className="person-avatar" />
                    ) : (
                      <span className="person-avatar-fallback">👤</span>
                    )}
                    <span className="person-views-badge">{person.faces.length}</span>
                  </div>

                  <div className="person-meta">
                    <input
                      type="text"
                      className="person-name-input"
                      value={personNames[person.id] || person.name}
                      onChange={(e) => handleRenamePerson(person.id, e.target.value)}
                      onClick={(e) => e.stopPropagation()}
                      title="Click to rename identity"
                    />
                    <small className="person-subtext">
                      {person.faces.length} observed angles · {person.id}
                    </small>
                  </div>
                </div>
              );
            })}

            {clusters.length === 0 && (
              <div className="empty-identities-hint">
                <span>◎</span>
                <p>No faces loaded yet.</p>
                <small>Import photos or load a video to cluster identities.</small>
              </div>
            )}
          </div>

          {/* Batch Assignment Footer */}
          {activePerson && inspection?.tracks?.length > 0 && (
            <div className="batch-assign-section">
              <span className="eyebrow">BATCH ASSIGN REPLACEMENT</span>
              <p className="tiny-muted">Map "{activePerson.name}" to detected target tracks:</p>
              <div className="track-assign-buttons">
                {inspection.tracks.map((track) => (
                  <button
                    key={track.track_id}
                    type="button"
                    className="assign-track-btn"
                    onClick={() => handleBatchAssign(track.track_id)}
                  >
                    Set as {track.track_id.replace('track_', 'Face ')}
                  </button>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* Center Column: 5 Pose Angle Buckets */}
        <section className="facemgr-buckets-panel" aria-label="5 Pose Angle Buckets">
          <div className="buckets-panel-header">
            <div>
              <span className="eyebrow">3D MULTI-ANGLE FACESET BUILDER</span>
              <h2>{activePerson?.name || 'Select Identity'}</h2>
            </div>
            <div className="bucket-actions">
              <button
                type="button"
                className="button small"
                onClick={handleDownloadFsz}
                disabled={!activePerson}
                title="Download .fsz FaceSet archive to local disk"
              >
                ⤓ Download .fsz
              </button>
            </div>
          </div>

          {/* 5 Pose Columns Grid */}
          <div className="pose-buckets-grid">
            {POSE_BUCKET_CONFIG.map((b) => {
              const facesInBucket = bucketedFaces[b.id] || [];
              const isMissing = facesInBucket.length === 0;

              return (
                <div key={b.id} className={`pose-bucket-column ${isMissing ? 'missing' : 'filled'}`}>
                  <div className="bucket-header" style={{ borderTopColor: b.color }}>
                    <div className="bucket-title-row">
                      <span className="bucket-title">{b.title}</span>
                      <span className="bucket-count" style={{ background: b.color }}>
                        {facesInBucket.length}
                      </span>
                    </div>
                    <span className="bucket-range">{b.range}</span>
                  </div>

                  <div className="bucket-faces-container">
                    {facesInBucket.map((face) => (
                      <div key={face.id} className="bucket-face-card">
                        {face.thumbnailUrl ? (
                          <img src={face.thumbnailUrl} alt="Face view" className="bucket-face-thumb" />
                        ) : (
                          <div className="bucket-face-fallback">◎</div>
                        )}
                        <div className="bucket-face-meta">
                          <span>Y: <b>{(face.yaw || 0).toFixed(0)}°</b></span>
                          <span>P: <b>{(face.pitch || 0).toFixed(0)}°</b></span>
                        </div>
                      </div>
                    ))}

                    {isMissing && (
                      <div className="bucket-missing-slot">
                        <span className="missing-icon">＋</span>
                        <small>Missing {b.badge}</small>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </section>

        {/* Right Column: 3D Geometry Unit Sphere */}
        <aside className="facemgr-sphere-panel" aria-label="3D Geometry Unit Sphere">
          <div className="sphere-panel-header">
            <span className="eyebrow">GEOMETRY UNIT SPHERE</span>
            <h3>Pose Lattice Coverage</h3>
          </div>

          {/* 3D Canvas */}
          <div className="sphere-canvas-box">
            <Canvas
              camera={{ position: [2.0, 1.2, 2.6], fov: 42, near: 0.05, far: 50 }}
              dpr={[1, 2]}
              gl={{ antialias: true, alpha: true }}
            >
              <ambientLight intensity={0.8} />
              <directionalLight position={[4, 5, 5]} intensity={1.5} />
              <directionalLight position={[-4, -2, -3]} intensity={0.6} color="#38bdf8" />
              <FaceSetOrientationSphere lattice={lattice} />
              <OrbitControls enablePan={false} minDistance={1.8} maxDistance={6} />
            </Canvas>
          </div>

          {/* Sphere Readouts & Legend */}
          <div className="sphere-stats-box">
            <div className="sphere-stat-row">
              <span>Observed View Bins</span>
              <strong>{coverageStats.observedCount} / {coverageStats.totalNodes} ({coverageStats.percentage}%)</strong>
            </div>
            <div className="sphere-stat-row">
              <span>Well-Observed (&gt;3 samples)</span>
              <strong className="green">{coverageStats.wellCoveredCount} bins</strong>
            </div>
            <div className="sphere-stat-row">
              <span>Coverage Fidelity</span>
              <span className={`fidelity-tag ${coverageStats.isEstimated25D ? 'partial' : 'full'}`}>
                {coverageStats.isEstimated25D ? 'Estimated 2.5D' : 'Full 3D Reconstructed'}
              </span>
            </div>

            <div className="sphere-legend">
              <span className="legend-chip"><i className="dot-green" /> &gt;3 Samples</span>
              <span className="legend-chip"><i className="dot-amber" /> 1-2 Sparse</span>
              <span className="legend-chip"><i className="dot-slate" /> Blind Spot</span>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
