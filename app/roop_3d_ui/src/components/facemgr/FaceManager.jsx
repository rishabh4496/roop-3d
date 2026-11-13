import { rankClusters, poseGrid } from './faceBank';
import NumericInput from '../controls/NumericInput';
import VirtualList from '../common/VirtualList';
import { renderProbe } from '../../utils/renderProbe';
import { lazy, Suspense, useState, useEffect, useMemo, useRef } from 'react';
import GuardedCanvas from '../three/GuardedCanvas';
import { fileName } from '../../utils/safeData';
import { OrbitControls } from '@react-three/drei';
import {
  POSE_BUCKETS,
  POSE_BUCKET_CONFIG,
  classifyPoseBucket,
} from './fszBuilder';
import { generateFibonacciSphere, projectViewsToLattice } from '../three/poseLattice';
import AngularCoverageRadar from '../three/AngularCoverageRadar';
import FacesetInspector3D from './FacesetInspector3D';
import { api, post } from '../../api';
import FacesetLibrary from './FacesetLibrary';
import PhotoSynthesisAction from './PhotoSynthesisAction';

const FaceSetOrientationSphere = lazy(() => import('../three/PoseGeometryViewer')
  .then(module => ({ default: module.FaceSetOrientationSphere })));

/** The 3x3 pose matrix as ARIA rows of three cells, so `role=grid` has the `row` children it requires. */
const poseRows = cells => Array.from({ length: Math.ceil(cells.length / 3) }, (_, row) => cells.slice(row * 3, row * 3 + 3));

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

    extracted.push({
      id: `face_${faceIdCounter++}`,
      sourceFile: target ? fileName(target) : track.track_id,
      thumbnailUrl: thumb,
      yaw: yawAvg,
      pitch: pitchAvg,
      roll: rollAvg,
      confidence: track.confidence ?? null,
      embedding: track.embedding || track.centroid || null,
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
  renderProbe('FaceManager');
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
  const [assignedSource, setAssignedSource] = useState('');
  const [bank, setBank] = useState(null), [bankTarget, setBankTarget] = useState('');
  const [archiveMatrix, setArchiveMatrix] = useState(null), [matrixRevision, setMatrixRevision] = useState(0), [bankRevision, setBankRevision] = useState(0);
  useEffect(() => {
    const refresh = () => setBankRevision(value => value + 1);
    window.addEventListener('target-angle-scan-completed', refresh);
    return () => window.removeEventListener('target-angle-scan-completed', refresh);
  }, []);
  const bankRef = useRef(null), assignSourceRef = useRef(onAssignSource);
  assignSourceRef.current = onAssignSource;
  bankRef.current = bank;
  const bankReady = bankTarget === target && !!bank;
  const bankEdits = useRef(Promise.resolve());
  const persistBankEdit = (action, clusterIds, extra = {}) => {
    const pending=bankEdits.current.catch(()=>{}).then(async()=>{
      const result = await post('/faces/bank/edit',{target_media_path:target,revision:bankRef.current.revision,action,cluster_ids:clusterIds,...extra});
      bankRef.current=result;setBank(result);
      return result;
    });
    bankEdits.current=pending;
    return pending;
  };
  useEffect(()=>{
    if (!target || !inspection?.inspection_id) {setBank(null);setBankTarget('');return;}
    setBank(null);setBankTarget('');setAssignedSource('');
    const controller=new AbortController();
    void post('/faces/bank/discover',{target_media_path:target,inspection_id:inspection.inspection_id},{signal:controller.signal})
      .then(result=>{setBank(result);setBankTarget(target);setSelectedPersonId(null);setPersonNames({});
        for (const person of result.clusters) if (person.faceset_id) for (const track of person.track_ids) assignSourceRef.current?.(track,person.faceset_id);
        if (result.unresolved_tracks.length) setStatusNotice(`${result.unresolved_tracks.length} tracks need a clearer observed frame before banking.`);
      }).catch(error=>{if(error.name!=='AbortError') setStatusNotice(`Face bank: ${error.message}`);});
    return ()=>controller.abort();
  },[target,inspection?.inspection_id,bankRevision]);
  useEffect(()=>{
    setArchiveMatrix(null);
    if(!assignedSource) return;
    const controller=new AbortController();
    void api(`/facesets/${encodeURIComponent(assignedSource)}/pose_matrix`,{signal:controller.signal})
      .then(setArchiveMatrix).catch(error=>{if(error.name!=='AbortError')setStatusNotice(error.message);});
    return ()=>controller.abort();
  },[assignedSource,matrixRevision]);
  const [activeTab, setActiveTab] = useState('builder'); // 'builder' | 'clustering' | 'sphere'

  const fileInputRef = useRef(null);

  // Extract initial faces from inspection if available and faces is empty
  useEffect(() => {
    if (faces.length === 0 && inspection?.tracks?.length > 0) {
      const extracted = extractFacesFromInspection(inspection, target);
      if (extracted.length > 0) {
        setFaces(extracted);
      }
    }
  }, [inspection, target, faces.length]);

  // Target bank ranks are durable; local imported photos use cosine grouping.
  const clusters = useMemo(() => {
    if (bankReady) return [
      ...bank.clusters.map(person=>({...person,name:personNames[person.id] ?? person.name})),
      ...rankClusters(faces.filter(face=>!face.trackId),dbscanEps).map(person=>({...person,local:true,name:personNames[person.id] ?? person.name})),
    ];
    if (faces.length === 0) return [];
    const clustered = rankClusters(faces, dbscanEps);
    // Apply custom names
    return clustered.map(c => ({
      ...c,
      name: personNames[c.id] || c.name,
    }));
  }, [faces, dbscanEps, personNames, bankReady, bank]);

  // Ensure an active person is selected
  useEffect(() => {
    if (clusters.length > 0 && !selectedPersonId) {
      setSelectedPersonId(clusters[0].id);
    }
  }, [clusters, selectedPersonId]);

  const activePerson = useMemo(() => {
    return clusters.find(c => c.id === selectedPersonId) || clusters[0] || null;
  }, [clusters, selectedPersonId]);

  useEffect(()=>{
    if(bankReady && !activePerson?.local) setAssignedSource(activePerson?.faceset_id ?? '');
  },[bankReady,activePerson?.id,activePerson?.faceset_id,activePerson?.local]);

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

  // The same person's views on the exact 12 x 8 yaw/pitch grid, for the angular coverage radar.
  const radarCoverage = useMemo(() => ({ views: (activePerson?.faces || []).map(f => ({ yaw: f.yaw || 0, pitch: f.pitch || 0 })) }), [activePerson]);

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
  const handlePhotoUpload = async (event) => {
    setIsProcessing(true);
    try {
      for (const file of Array.from(event.target.files || [])) {
        const image = await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});
        const result=await post('/faces/analyze_photo',{image_base64:image});
        setFaces(previous=>[...previous,...result.faces.map(face=>({...face,sourceFile:file.name}))]);
      }
      setStatusNotice('Imported measured face crops and identities.');
    } catch(error) {setStatusNotice(error.message);} finally {setIsProcessing(false);}
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
    if(bankReady && selectedForMerge.some(id=>bank.clusters.some(person=>person.id===id)) && selectedForMerge.some(id=>!bank.clusters.some(person=>person.id===id))) {
      setStatusNotice('Select target people or imported source-photo groups for each merge.');return;
    }
    if (bankReady && selectedForMerge.every(id=>bank.clusters.some(person=>person.id===id))) {
      void persistBankEdit('merge',selectedForMerge).then(()=>{setSelectedForMerge([]);setStatusNotice('Merged person tracks; discovery ranks preserved.');}).catch(error=>setStatusNotice(error.message));
      return;
    }
    const primaryId = selectedForMerge[0];
    const targetClusterFaces = [];

    // Find all faces in selected clusters
    for (const c of clusters) {
      if (selectedForMerge.includes(c.id)) {
        targetClusterFaces.push(...c.faces);
      }
    }

    // Store the explicit merge without replacing original measurements.
    const primaryCluster = clusters.find(c => c.id === primaryId);


    setFaces(prev =>
      prev.map(f => {
        if (targetClusterFaces.some(tf => tf.id === f.id)) {
          return { ...f, manualGroup: primaryId };
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
    if (!assignedSource) return;
    onAssignSource(trackId, assignedSource);
    setStatusNotice(`Assigned "${activePerson.name}" (${activePerson.faces.length} views) to ${trackId}.`);
  };

  // Export as .fsz FaceSet archive (Local Download)
  const buildMeasuredArchive = async () => {
    if (!activePerson?.faces.length) throw new Error('Select measured source faces');
    const views = activePerson.faces.map(face=>{
      if (!face.embedding?.length || !face.landmarks || !face.bbox || !face.thumbnailUrl?.includes('base64,')) throw new Error('Import measured source photos before exporting this identity');
      return {image_base64:face.thumbnailUrl.split('base64,')[1], embedding:face.embedding, landmarks:face.landmarks,bbox:face.bbox,yaw:face.yaw,pitch:face.pitch,roll:face.roll};
    });
    return post('/facesets/build',{name:activePerson.name,faceset_id:activePerson.id.toLowerCase().replace(/[^a-z0-9_-]/g,'_').slice(0,32),views});
  };
  const handleDownloadFsz = async () => {
    setIsProcessing(true);
    try {
      const result=assignedSource ? (sources.find(source=>source.faceset_id===assignedSource) ?? {faceset_id:assignedSource,name:assignedSource}) : await buildMeasuredArchive();
      const anchor=document.createElement('a'); anchor.href=`/api/v1/facesets/${encodeURIComponent(result.faceset_id)}/download`;anchor.download=`${result.name}.fsz`;anchor.click();
      await onRefreshSources?.(); setStatusNotice('Exported measured .fsz archive.');
    } catch(error) {setStatusNotice(error.message);} finally {setIsProcessing(false);}
  };

  // Save FaceSet to Local Workstation Library via Backend API
  const handleSaveToLibrary = async () => {
    if (!activePerson || activePerson.faces.length === 0) return;
    try {
      setIsProcessing(true);
      setStatusNotice('Saving FaceSet to workstation library…');

      const result = await buildMeasuredArchive();

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

        {/* View Mode Switcher */}
        <div className="facemgr-mode-toggle" role="tablist">
          <button
            type="button"
            className={`tab-btn ${activeTab === 'builder' ? 'active' : ''}`}
            role="tab" aria-selected={activeTab === 'builder'}
            onClick={() => setActiveTab('builder')}
          >
            ❖ 3D Studio
          </button>
          <button
            type="button"
            className={`tab-btn ${activeTab === 'library' ? 'active' : ''}`}
            role="tab" aria-selected={activeTab === 'library'}
            onClick={() => setActiveTab('library')}
          >
            📁 FaceSet Library ({sources?.length || 0})
          </button>
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
            accept=".png,.jpg,.jpeg"
            hidden
            onChange={handlePhotoUpload}
          />

          <button
            type="button"
            className="button accent"
            onClick={handleSaveToLibrary}
            disabled={!activePerson || isProcessing || (bankReady && !activePerson.local)}
            title="Register measured source photos as a FaceSet into workstation library for instant swapping"
          >
            Save to FaceSet Library ↗
          </button>
          {activePerson && !(bankReady && !activePerson.local) && <PhotoSynthesisAction
            images={activePerson.faces.filter(face => !face.trackId && face.thumbnailUrl?.includes('base64,')).map(face => face.thumbnailUrl)}
            name={personNames[activePerson.id] || activePerson.name || 'Synthesized 3D Faceset'}
            onSaved={async id => { await onRefreshSources?.(); setAssignedSource(id); setStatusNotice('Synthesized 36-angle faceset saved and activated.'); }} />}
        </div>
      </header>

      {statusNotice && (
        <div className="facemgr-notice-bar" role="status">
          <span>{statusNotice}</span>
          <button type="button" onClick={() => setStatusNotice('')}>×</button>
        </div>
      )}

      {activeTab === 'library' ? (
        <div className="facemgr-library-view" style={{ padding: '24px', maxWidth: '960px', margin: '0 auto' }}>
          <div className="panel-heading" style={{ marginBottom: '16px' }}>
            <div>
              <span className="eyebrow">PERSISTENT .FSZ IDENTITIES</span>
              <h2>FaceSet Library ({sources.length})</h2>
            </div>
          </div>
          <p className="muted" style={{ marginBottom: '20px' }}>
            Browse, search, import, rename, delete and manage your multi-angle .fsz FaceSets on local disk.
          </p>
          <FacesetLibrary
            sources={sources}
            onRefreshSources={onRefreshSources}
            onSelectSource={onAssignSource ? (id) => onAssignSource('primary', id) : undefined}
            notify={msg => setStatusNotice(msg)}
          />
          <FacesetInspector3D sources={sources} />
        </div>
      ) : (
        /* 2. Main Workstation Body: 3-Column Split */
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

          <button className="button small" disabled={!target || !inspection} onClick={()=>setBankRevision(value=>value+1)}>Refresh Person Bank</button>
          {/* Local photo cosine threshold */}
          <div className="clustering-slider-box">
            <NumericInput id="faceset-cluster-tolerance" disabled={bankReady && !activePerson?.local} label={bankReady && !activePerson?.local ? 'Target bank minimum cosine similarity: 0.60' : `Local photo cosine distance: ${dbscanEps.toFixed(2)}`} value={dbscanEps} min={.15} max={.65} step={.01} onChange={setDbscanEps} title="Adjust ArcFace cosine distance for imported photos" />
          </div>

          {/* Identities List */}
          <div className="identities-scroll-list">
            <VirtualList items={clusters} itemKey={person => person.id} estimateHeight={68} gap={8} aria-label="Identities" renderItem={(person) => {
              const isSelected = selectedPersonId === person.id;
              const isChecked = selectedForMerge.includes(person.id);
              const repFace = person.faces[0];

              return (
                <div
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
                      onBlur={() => {if(bankReady && !person.local && personNames[person.id] !== undefined) void persistBankEdit('rename',[person.id],{name:personNames[person.id]}).catch(error=>setStatusNotice(error.message));}}
                      onClick={(e) => e.stopPropagation()}
                      title="Click to rename identity"
                    />
                    <small className="person-subtext">
                      {person.faces.length} observed angles · {person.id}
                    </small>
                  </div>
                </div>
              );
            }} />

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
                disabled={!activePerson && !assignedSource}
                title="Download .fsz FaceSet archive to local disk"
              >
                ⤓ Download .fsz
              </button>
            </div>
          </div>

          <label>Source identity<select aria-label="Bank source identity" value={assignedSource} onChange={event=>{
            const source=event.target.value;setAssignedSource(source);
            if(bankReady && activePerson && !activePerson.local) void persistBankEdit('assign',[activePerson.id],{faceset_id:source || null}).then(()=>{
              for(const track of activePerson.track_ids) onAssignSource?.(track,source || null);
            }).catch(error=>setStatusNotice(error.message));
          }}><option value="">Select .fsz identity</option>{sources.map(source=><option key={source.faceset_id} value={source.faceset_id}>{source.name}</option>)}</select></label>
          <button className="button" disabled={!target || !assignedSource || isProcessing} onClick={async()=>{
            setIsProcessing(true);
            try {
              const result=await post('/faces/fill_missing_angles',{target_media_path:target,faceset_id:assignedSource});
              setMatrixRevision(value=>value+1);
              await onRefreshSources?.(); setStatusNotice(`Added ${result.added.length} measured angles to ${assignedSource}. Scanned ${result.sampled_frames} frames (${result.stop_reason}).`);
            } catch(error) { setStatusNotice(error.message); } finally {setIsProcessing(false);}
          }}>Fill Missing Angles from Current Target Video</button><small>Only observed faces matching the selected source identity are added. Long scans stop at the reported frame or time budget.</small>
          <div role="grid" aria-label="3 by 3 Yaw x Pitch Pose Matrix" style={{display:'grid',gridTemplateColumns:'repeat(3,1fr)',gap:8}}>
            {poseRows(archiveMatrix?.faceset_id === assignedSource ? archiveMatrix.cells : poseGrid(activePerson?.faces || [])).map((row, rowIndex) => <div role="row" key={rowIndex} style={{display:'contents'}}>{row.map(cell=><div role="gridcell" key={cell.cell ?? `${cell.yaw}:${cell.pitch}`} style={{padding:12,border:'1px solid currentColor'}}>
              <b>{cell.cell ?? `Yaw ${cell.yaw} / Pitch ${cell.pitch}`}</b><p>{(cell.support ?? cell.faces?.length ?? 0) ? `${cell.support ?? cell.faces.length} observed` : 'Missing'}</p>
              {(cell.thumbnailUrl || cell.faces?.[0]?.thumbnailUrl) && <img width="64" src={cell.thumbnailUrl || cell.faces?.[0]?.thumbnailUrl} alt="Best measured cell view"/>}
              <small>Sharpness: {cell.best?.sharpness ?? cell.faces?.[0]?.sharpness ?? 'unmeasured'} / Quality: {cell.best?.quality ?? cell.faces?.[0]?.quality ?? 'unmeasured'}</small>
            </div>)}</div>)}
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
                    <VirtualList items={facesInBucket} itemKey={face => face.id} estimateHeight={64} gap={8} aria-label={`${b.title} face crops`}
                      renderItem={(face) => (
                        <div className="bucket-face-card">
                          {face.thumbnailUrl ? (
                            <img src={face.thumbnailUrl} alt="Face view" className="bucket-face-thumb" loading="lazy" decoding="async" />
                          ) : (
                            <div className="bucket-face-fallback">◎</div>
                          )}
                          <div className="bucket-face-meta">
                            <span>Y: <b>{(face.yaw || 0).toFixed(0)}°</b></span>
                            <span>P: <b>{(face.pitch || 0).toFixed(0)}°</b></span>
                          </div>
                        </div>
                      )} />

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
            <GuardedCanvas
              label="Geometry unit sphere"
              fallback={({ reason, retry }) => <p className="muted" role="status">The 3D sphere is not drawing. {reason}{retry && <> <button type="button" className="button small" onClick={retry}>Restart 3D view</button></>} The angular coverage radar below still shows the same data.</p>}
              camera={{ position: [2.0, 1.2, 2.6], fov: 42, near: 0.05, far: 50 }}
              dpr={[1, 2]}
              gl={{ antialias: true, alpha: true }}
            >
              <ambientLight intensity={0.8} />
              <directionalLight position={[4, 5, 5]} intensity={1.5} />
              <directionalLight position={[-4, -2, -3]} intensity={0.6} color="#38bdf8" />
              <Suspense fallback={null}>
                <FaceSetOrientationSphere lattice={lattice} />
              </Suspense>
              <OrbitControls enablePan={false} minDistance={1.8} maxDistance={6} />
            </GuardedCanvas>
          </div>

          <AngularCoverageRadar coverage={radarCoverage} name={activePerson?.name} />

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
      )}
    </div>
  );
}
