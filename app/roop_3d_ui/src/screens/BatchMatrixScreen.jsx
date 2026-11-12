import React, { useState, useMemo, useEffect } from 'react';
import { api, post } from '../api';
import { engineSelection } from '../stores/engineSelection';
import { rackStore, setRack, useRack } from '../stores/rackSelection';
import EngineModelRack from '../components/modelhub/EngineModelRack';
import { featureReason } from '../state';
import NumericInput from '../components/controls/NumericInput';
import { SKIP_KEY, describeLint, lintBatch } from '../components/diagnostics/batchLint';
import { DragItems } from '../components/ui-layouts/DragItems';

export const BUILTIN_RECIPES = [
  {
    id: 'photoreal_natural',
    name: 'Natural Photoreal',
    description: 'GPEN 512 enhancement (blend 0.35), retarget pose, balanced sharpness and edge blend.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.35, eye_open_ratio: 1.0, mouth_open_ratio: 1.0, edge_feather_px: 2, erosion_px: 2, swap_weight: 1.0 }
  },
  {
    id: 'high_fidelity',
    name: 'High-Fidelity Ultra',
    description: 'Strong GPEN restoration (0.80), retarget pose, high-resolution edge blending.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.80, eye_open_ratio: 1.0, mouth_open_ratio: 1.0, edge_feather_px: 3, erosion_px: 3, swap_weight: 1.0 }
  },
  {
    id: 'mesh_geometry_3d',
    name: '3D Mesh Blend & Geometry',
    description: 'Full 3D mesh blend pose mode, expressive LivePortrait landmarks, geometric warping.',
    parameters: { pose_mode: 'mesh_blend', restoration_blend: 0.40, eye_open_ratio: 1.05, mouth_open_ratio: 1.05, edge_feather_px: 4, erosion_px: 2, swap_weight: 0.9 }
  },
  {
    id: 'cinematic_soft',
    name: 'Cinematic Soft Swap',
    description: 'Subtle restoration (0.20), wide 5px edge feathering, soft skin tone preservation.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.20, eye_open_ratio: 1.0, mouth_open_ratio: 1.0, edge_feather_px: 5, erosion_px: 1, swap_weight: 0.85 }
  },
  {
    id: 'action_smoothing',
    name: 'Action / Motion Smoothing',
    description: 'Fast retarget, strict occlusion threshold, motion blur adaptation.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.30, eye_open_ratio: 1.0, mouth_open_ratio: 1.0, edge_feather_px: 3, erosion_px: 2, swap_weight: 0.95 }
  },
  {
    id: 'codeformer_fidelity',
    name: 'CodeFormer Fidelity Sweep',
    description: 'CodeFormer scalar fidelity feed (w=0.85) mapped to CodeFormer training template.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.70, codeformer_fidelity: 0.85, edge_feather_px: 3, erosion_px: 2, swap_weight: 1.0 }
  },
  {
    id: 'ultramax_restoration',
    name: 'UltraMax Deep Dynamic',
    description: 'UltraMax registered model with dynamic preprocessing contract.',
    parameters: { pose_mode: 'retarget', restoration_blend: 0.65, edge_feather_px: 2, erosion_px: 2, swap_weight: 1.0 }
  },
  {
    id: 'frontalization_tps_sweep',
    name: 'TPS Frontalization Sweep',
    description: 'Canonical Stage 0 frontalization (0.75) and TPS expression transfer (0.80).',
    parameters: { pose_mode: 'retarget', pose_frontalization: 0.75, tps_expression_weight: 0.80, restoration_blend: 0.40, swap_weight: 1.0 }
  }
];

export default function BatchMatrixScreen({
  sources = [],
  projects = [],
  caps = null,
  target = '',
  probe = null,
  inspection = null,
  refreshJobs = async () => {},
}) {
  const [strategy, setStrategy] = useState('one-to-many'); // 'one-to-many' | 'grouped' | 'matrix' | 'recipe'
  const batchParameters = useRack(state => state.batchParameters);
  const [engineConfig, setEngineConfig] = useState(null);
  // The batch detector is the saved global choice (the rack persists it on change); sent with every queued job.
  const batchDetector = ['scrfd', 'retinaface', 'yoloface'].includes(engineConfig?.detector_model_id) ? engineConfig.detector_model_id : null;
  useEffect(() => { let alive = true; api('/config').then(value => { if (alive) setEngineConfig(value); }).catch(() => undefined); return () => { alive = false; }; }, []);
  const [batchBusy, setBatchBusy] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  // Segment Splitting State
  const [autoSplit, setAutoSplit] = useState(false);
  const [maxSegmentDuration, setMaxSegmentDuration] = useState(15.0);
  const [autoConcat, setAutoConcat] = useState(true);
  const [splitProgress, setSplitProgress] = useState(null);

  // Strategy 1: One-to-Many
  const [oneSourceId, setOneSourceId] = useState(sources[0]?.faceset_id || '');
  const [selectedTargetIds, setSelectedTargetIds] = useState([]);

  // Strategy 2: Grouped
  const [groups, setGroups] = useState([
    { id: 'group_1', name: 'Primary Targets', facesetId: sources[0]?.faceset_id || '', targetIds: [] },
  ]);

  // Strategy 3: Per-File Matrix (M Sources × N Targets)
  const [matrixSelection, setMatrixSelection] = useState([]);
  const [outputNamingTemplate, setOutputNamingTemplate] = useState('{target}_{source}_{restorer}_{timestamp}.mp4');

  // Strategy 4: Recipe Matrix
  const [recipes, setRecipes] = useState(BUILTIN_RECIPES);
  const [recipeSelection, setRecipeSelection] = useState([]); // Array of `targetId:recipeId`
  const [recipeSourceId, setRecipeSourceId] = useState(sources[0]?.faceset_id || '');
  const [showNewRecipeModal, setShowNewRecipeModal] = useState(false);
  const [newRecipeName, setNewRecipeName] = useState('');
  const [newRecipeRestoration, setNewRecipeRestoration] = useState(0.4);
  const [newRecipePoseMode, setNewRecipePoseMode] = useState('retarget');

  // Unified available targets pool
  const targetPool = useMemo(() => {
    const pool = [];
    if (target) {
      pool.push({
        id: `current_target`,
        name: `Current Workstation Target (${target.split(/[\\/]/).at(-1)})`,
        target_path: target,
        inspection_id: inspection?.inspection_id || '',
        duration: probe?.duration_seconds || null,
        isCurrent: true,
      });
    }
    for (const p of projects) {
      pool.push({
        id: p.project_id,
        name: p.name,
        target_path: p.target_path || (p.request && p.request.target_path) || '',
        inspection_id: p.request?.inspection_id || '',
        duration: null,
        lastCompletedFrame: p.last_completed_frame,
        isProject: true,
      });
    }
    return pool;
  }, [target, inspection, probe, projects]);

  // Auto-select first source if none selected
  React.useEffect(() => {
    if (sources.length > 0) {
      if (!oneSourceId) setOneSourceId(sources[0].faceset_id);
      if (!recipeSourceId) setRecipeSourceId(sources[0].faceset_id);
    }
  }, [sources, oneSourceId, recipeSourceId]);

  // Handle Video Segment Splitting helper
  const prepareJobSegments = async (targetPath, baseRequest) => {
    if (!autoSplit) {
      return [baseRequest];
    }
    try {
      const boundaries = await post('/media/split-boundaries', {
        target_path: targetPath,
        max_segment_duration: Number(maxSegmentDuration) || 15.0,
      });
      if (!boundaries || boundaries.length <= 1) {
        return [baseRequest];
      }
      return boundaries.map((seg, idx) => ({
        ...baseRequest,
        trim_start_seconds: seg.start_seconds,
        trim_end_seconds: seg.end_seconds,
        _segment_index: idx,
        _total_segments: boundaries.length,
      }));
    } catch {
      return [baseRequest];
    }
  };

  // Helper: Dispatch queued jobs with optional segment concat tracking
  const dispatchBatch = async (jobPayloads, label) => {
    setBatchBusy(true);
    setStatusMessage('');
    setErrorMessage('');
    let queuedCount = 0;
    const trackedSegmentBatches = {};
    let lintNote = '';

    try {
      // Pre-flight the distinct targets once: a FAIL (e.g. not enough disk for the estimated render) stops the whole batch before any job
      // is created; warnings are reported with the result. The same session opt-out as the single-job linter applies.
      const rack0 = rackStore.get();
      const lint = await lintBatch(post, jobPayloads, { rackParameters: rack0.batchParameters, swapperId: rack0.swapper });
      const verdict = describeLint(lint);
      if (lint.failures.length && sessionStorage.getItem(SKIP_KEY) !== 'true') { setErrorMessage(verdict.block); return; }
      lintNote = verdict.note;
      for (const payload of jobPayloads) {
        const targetPath = payload.target_path;
        const subRequests = await prepareJobSegments(targetPath, payload);
        const batchTag = `batch_${crypto.randomUUID().slice(0, 8)}`;

        for (const sub of subRequests) {
          const idemKey = `matrix-${crypto.randomUUID()}`;
          const cleanRequest = { engine_id: engineSelection.get().engineId, ...sub };
          // Engine & Model Rack: the chosen swapper and any touched enhancer / 3D / mask fields apply to every queued job.
          const rack = rackStore.get();
          if (rack.swapper) cleanRequest.swapper_model_id = rack.swapper;
          if (batchDetector) cleanRequest.detector_model_id = batchDetector;
          if (Object.keys(rack.batchParameters).length) cleanRequest.workstation_parameters = { ...(cleanRequest.workstation_parameters || {}), ...rack.batchParameters };
          const segIdx = cleanRequest._segment_index;
          const totalSegs = cleanRequest._total_segments;
          delete cleanRequest._segment_index;
          delete cleanRequest._total_segments;

          const created = await post('/jobs', cleanRequest, {
            headers: { 'Idempotency-Key': idemKey }
          });
          queuedCount++;

          if (segIdx !== undefined && totalSegs !== undefined) {
            if (!trackedSegmentBatches[batchTag]) {
              trackedSegmentBatches[batchTag] = {
                targetPath,
                total: totalSegs,
                jobs: [],
              };
            }
            trackedSegmentBatches[batchTag].jobs.push(created.job_id);
          }
        }
      }

      setStatusMessage(`Successfully queued ${queuedCount} ${label} render jobs to the local engine.${lintNote ? ` ${lintNote}` : ''}`);
      if (Object.keys(trackedSegmentBatches).length > 0 && autoConcat) {
        setSplitProgress(`${Object.keys(trackedSegmentBatches).length} multi-segment video render(s) active. FFmpeg stream-copy concatenation will merge segments when completed.`);
      }
    } catch (err) {
      setErrorMessage(`Error queueing batch (${queuedCount} queued): ${err.message}`);
    } finally {
      setBatchBusy(false);
      await refreshJobs();
    }
  };

  // 1. One-to-Many Execution
  const handleQueueOneToMany = async () => {
    if (!oneSourceId) {
      setErrorMessage('Please select a Source FaceSet.');
      return;
    }
    if (selectedTargetIds.length === 0) {
      setErrorMessage('Please select at least one target video/project.');
      return;
    }

    const payloads = [];
    for (const targetId of selectedTargetIds) {
      const item = targetPool.find(t => t.id === targetId);
      if (!item) continue;
      let request;
      if (item.isProject) {
        const full = await api(`/projects/${item.id}`);
        request = {
          ...full.request,
          processing: 'local',
          remote_consent: false,
          face_mappings: full.request.face_mappings.map(m => ({ ...m, faceset_id: oneSourceId })),
        };
      } else {
        request = {
          target_path: item.target_path,
          inspection_id: item.inspection_id,
          face_mappings: (inspection?.tracks || []).map(t => ({ track_id: t.track_id, faceset_id: oneSourceId })),
          processing: 'local',
          remote_consent: false,
          trim_start_seconds: null,
          trim_end_seconds: null,
        };
      }
      payloads.push(request);
    }

    await dispatchBatch(payloads, 'One-to-Many');
  };

  // 2. Grouped Execution
  const handleQueueGrouped = async () => {
    const payloads = [];
    for (const group of groups) {
      if (!group.facesetId || group.targetIds.length === 0) continue;
      for (const targetId of group.targetIds) {
        const item = targetPool.find(t => t.id === targetId);
        if (!item) continue;
        let request;
        if (item.isProject) {
          const full = await api(`/projects/${item.id}`);
          request = {
            ...full.request,
            processing: 'local',
            remote_consent: false,
            face_mappings: full.request.face_mappings.map(m => ({ ...m, faceset_id: group.facesetId })),
          };
        } else {
          request = {
            target_path: item.target_path,
            inspection_id: item.inspection_id,
            face_mappings: (inspection?.tracks || []).map(t => ({ track_id: t.track_id, faceset_id: group.facesetId })),
            processing: 'local',
            remote_consent: false,
            trim_start_seconds: null,
            trim_end_seconds: null,
          };
        }
        payloads.push(request);
      }
    }

    if (payloads.length === 0) {
      setErrorMessage('No valid targets mapped in groups.');
      return;
    }
    await dispatchBatch(payloads, 'Grouped');
  };

  // 3. Per-File Matrix Execution
  const handleQueueMatrix = async () => {
    if (matrixSelection.length === 0) {
      setErrorMessage('Please select at least one matrix intersection cell.');
      return;
    }

    const payloads = [];
    for (const cell of matrixSelection) {
      const [targetId, srcId] = cell.split(':');
      const item = targetPool.find(t => t.id === targetId);
      if (!item) continue;
      let request;
      if (item.isProject) {
        const full = await api(`/projects/${item.id}`);
        request = {
          ...full.request,
          processing: 'local',
          remote_consent: false,
          face_mappings: full.request.face_mappings.map(m => ({ ...m, faceset_id: srcId })),
        };
      } else {
        request = {
          target_path: item.target_path,
          inspection_id: item.inspection_id,
          face_mappings: (inspection?.tracks || []).map(t => ({ track_id: t.track_id, faceset_id: srcId })),
          processing: 'local',
          remote_consent: false,
          trim_start_seconds: null,
          trim_end_seconds: null,
        };
      }
      payloads.push(request);
    }

    await dispatchBatch(payloads, 'Matrix');
  };

  // 4. Recipe Matrix Execution
  const handleQueueRecipe = async () => {
    if (recipeSelection.length === 0) {
      setErrorMessage('Please select at least one Target × Recipe cell.');
      return;
    }
    if (!recipeSourceId) {
      setErrorMessage('Please pick a Source FaceSet to use with the selected recipes.');
      return;
    }

    const payloads = [];
    // Jobs are queued in the order of the dragged recipe chips; within one recipe, in the order the targets were ticked.
    const recipeRank = new Map(recipes.map((entry, position) => [entry.id, position]));
    const orderedCells = [...recipeSelection].sort((a, b) => (recipeRank.get(a.split(':')[1]) ?? 0) - (recipeRank.get(b.split(':')[1]) ?? 0));
    for (const cell of orderedCells) {
      const [targetId, recipeId] = cell.split(':');
      const item = targetPool.find(t => t.id === targetId);
      const recipe = recipes.find(r => r.id === recipeId);
      if (!item || !recipe) continue;

      let baseRequest;
      if (item.isProject) {
        const full = await api(`/projects/${item.id}`);
        baseRequest = {
          ...full.request,
          processing: 'local',
          remote_consent: false,
          face_mappings: full.request.face_mappings.map(m => ({ ...m, faceset_id: recipeSourceId })),
          workstation_parameters: {
            ...(full.request.workstation_parameters || {}),
            ...recipe.parameters,
          }
        };
      } else {
        baseRequest = {
          target_path: item.target_path,
          inspection_id: item.inspection_id,
          face_mappings: (inspection?.tracks || []).map(t => ({ track_id: t.track_id, faceset_id: recipeSourceId })),
          processing: 'local',
          remote_consent: false,
          trim_start_seconds: null,
          trim_end_seconds: null,
          workstation_parameters: recipe.parameters,
        };
      }
      payloads.push(baseRequest);
    }

    await dispatchBatch(payloads, 'Recipe Matrix');
  };

  // Matrix bulk helpers
  const toggleMatrixCell = (targetId, srcId) => {
    const key = `${targetId}:${srcId}`;
    setMatrixSelection(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]);
  };
  const selectAllMatrix = () => {
    const all = [];
    for (const t of targetPool) {
      for (const s of sources) {
        all.push(`${t.id}:${s.faceset_id}`);
      }
    }
    setMatrixSelection(all);
  };
  const clearMatrix = () => setMatrixSelection([]);
  const invertMatrix = () => {
    const allKeys = [];
    for (const t of targetPool) {
      for (const s of sources) {
        allKeys.push(`${t.id}:${s.faceset_id}`);
      }
    }
    setMatrixSelection(allKeys.filter(k => !matrixSelection.includes(k)));
  };
  const selectRow = (targetId) => {
    const rowKeys = sources.map(s => `${targetId}:${s.faceset_id}`);
    const allIn = rowKeys.every(k => matrixSelection.includes(k));
    if (allIn) {
      setMatrixSelection(prev => prev.filter(k => !rowKeys.includes(k)));
    } else {
      setMatrixSelection(prev => Array.from(new Set([...prev, ...rowKeys])));
    }
  };
  const selectCol = (srcId) => {
    const colKeys = targetPool.map(t => `${t.id}:${srcId}`);
    const allIn = colKeys.every(k => matrixSelection.includes(k));
    if (allIn) {
      setMatrixSelection(prev => prev.filter(k => !colKeys.includes(k)));
    } else {
      setMatrixSelection(prev => Array.from(new Set([...prev, ...colKeys])));
    }
  };

  // Recipe helpers
  const toggleRecipeCell = (targetId, recipeId) => {
    const key = `${targetId}:${recipeId}`;
    setRecipeSelection(prev => prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]);
  };

  const handleAddCustomRecipe = () => {
    if (!newRecipeName.trim()) return;
    const custom = {
      id: `custom_${Date.now()}`,
      name: newRecipeName.trim(),
      description: `Custom preset: GPEN ${newRecipeRestoration}, ${newRecipePoseMode} pose.`,
      parameters: {
        pose_mode: newRecipePoseMode,
        restoration_blend: Number(newRecipeRestoration) || 0.35,
        eye_open_ratio: 1.0,
        mouth_open_ratio: 1.0,
        edge_feather_px: 2,
        erosion_px: 2,
        swap_weight: 1.0,
      }
    };
    setRecipes(prev => [...prev, custom]);
    setNewRecipeName('');
    setShowNewRecipeModal(false);
  };

  return (
    <main className="screen-page batch-matrix-container" data-testid="batch-matrix-screen">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">BATCH PROCESSING & AUTOMATION</span>
          <h1>Batch Matrix Engine</h1>
        </div>
        <div className="segmented" role="tablist" aria-label="Batch Strategies">
          <button
            role="tab"
            aria-selected={strategy === 'one-to-many'}
            className={strategy === 'one-to-many' ? 'active' : ''}
            onClick={() => setStrategy('one-to-many')}
          >
            One-to-Many
          </button>
          <button
            role="tab"
            aria-selected={strategy === 'grouped'}
            className={strategy === 'grouped' ? 'active' : ''}
            onClick={() => setStrategy('grouped')}
          >
            Grouped
          </button>
          <button
            role="tab"
            aria-selected={strategy === 'matrix'}
            className={strategy === 'matrix' ? 'active' : ''}
            onClick={() => setStrategy('matrix')}
          >
            Per-File Matrix
          </button>
          <button
            role="tab"
            aria-selected={strategy === 'recipe'}
            className={strategy === 'recipe' ? 'active' : ''}
            onClick={() => setStrategy('recipe')}
          >
            Recipe Matrix
          </button>
        </div>
      </div>

      <details className="batch-rack" open data-testid="batch-model-rack">
        <summary><b>Engine &amp; Model Rack</b> <span className="muted">applies to every job in this batch</span></summary>
        <p className="muted">Fields you change here override the matching recipe values in every queued job; untouched fields keep the recipe's own.</p>
        <EngineModelRack variant="batch" params={batchParameters} onParams={patch => setRack({ batchParameters: { ...rackStore.get().batchParameters, ...patch } })}
          config={engineConfig} caps={caps}
          onSaveConfig={value => { void api('/config', { method: 'PUT', body: JSON.stringify(value) }).then(setEngineConfig).catch(reason => setErrorMessage(reason instanceof Error ? reason.message : 'Could not save the engine setting.')); }} />
      </details>

      {/* Segment Splitting Controls Banner */}
      <section className="batch-split-banner">
        <div className="split-header">
          <label className="split-toggle-label">
            <input
              type="checkbox"
              checked={autoSplit}
              onChange={e => setAutoSplit(e.target.checked)}
            />
            <strong>Automatic Video Segment Splitting</strong>
          </label>
          <span className="tiny-badge">LOSSLESS GOP CONCAT</span>
        </div>
        <p className="muted">
          Automatically slices long target videos at GOP/scene keyframe boundaries into independent batch sub-jobs, rendering them in parallel or sequence, and losslessly concatenating them via FFmpeg stream-copy upon completion.
        </p>
        {autoSplit && (
          <div className="split-options-grid">
            <div className="setting-field">
              <NumericInput label="Max Segment Duration (seconds)" min={5} max={120} step={1} value={maxSegmentDuration}
                onChange={value => setMaxSegmentDuration(value ?? 15)} />
            </div>
            <label className="setting-checkbox">
              <input
                type="checkbox"
                checked={autoConcat}
                onChange={e => setAutoConcat(e.target.checked)}
              />
              <span>Lossless FFmpeg stream-copy concatenation on finish</span>
            </label>
          </div>
        )}
        {splitProgress && <p className="split-progress-note">✦ {splitProgress}</p>}
      </section>

      {/* Status Notifications */}
      {statusMessage && <div className="notification" role="status"><span>{statusMessage}</span> <a href="#/processing" className="button small">View Live Queue →</a></div>}
      {errorMessage && <div className="notification error" role="alert"><span>{errorMessage}</span></div>}

      {/* STRATEGY 1: ONE-TO-MANY */}
      {strategy === 'one-to-many' && (
        <section className="batch-strategy-panel">
          <div className="panel-subhead">
            <h2>Strategy A: One-to-Many</h2>
            <p className="muted">Apply 1 selected Source FaceSet identity across N target videos/images simultaneously.</p>
          </div>

          <div className="strategy-grid-split">
            {/* Source Identity Selection */}
            <div className="batch-card-block">
              <h3>1. Select Source FaceSet</h3>
              <div className="source-picker-list">
                {sources.map(src => (
                  <button
                    key={src.faceset_id}
                    className={`source-picker-item ${oneSourceId === src.faceset_id ? 'selected' : ''}`}
                    onClick={() => setOneSourceId(src.faceset_id)}
                  >
                    <span className="source-avatar">FS</span>
                    <div>
                      <strong>{src.name}</strong>
                      <small>{src.item_count} observed views</small>
                    </div>
                    <span className="check-indicator">{oneSourceId === src.faceset_id ? '●' : '○'}</span>
                  </button>
                ))}
                {!sources.length && <p className="muted empty-note">No FaceSets available. Import a .fsz file in the Workstation or Face Manager.</p>}
              </div>
            </div>

            {/* Target Media Selection */}
            <div className="batch-card-block">
              <div className="flex-between">
                <h3>2. Select Target Media ({selectedTargetIds.length}/{targetPool.length})</h3>
                <div className="mini-actions">
                  <button className="button small" onClick={() => setSelectedTargetIds(targetPool.map(t => t.id))}>Select All</button>
                  <button className="button small" onClick={() => setSelectedTargetIds([])}>Clear</button>
                </div>
              </div>
              <div className="target-picker-list">
                {targetPool.map(tgt => (
                  <label key={tgt.id} className="target-picker-item">
                    <input
                      type="checkbox"
                      checked={selectedTargetIds.includes(tgt.id)}
                      onChange={e => setSelectedTargetIds(prev => e.target.checked ? [...prev, tgt.id] : prev.filter(id => id !== tgt.id))}
                    />
                    <div className="target-item-meta">
                      <strong>{tgt.name}</strong>
                      <small>{tgt.target_path}</small>
                    </div>
                  </label>
                ))}
                {!targetPool.length && <p className="muted empty-note">No targets found. Load a target or save a project first.</p>}
              </div>
            </div>
          </div>

          <div className="batch-footer-action">
            <button
              className="button accent"
              disabled={batchBusy || !oneSourceId || selectedTargetIds.length === 0 || !!featureReason(caps, 'queue')}
              onClick={handleQueueOneToMany}
            >
              {batchBusy ? 'Queueing jobs…' : `Queue One-to-Many Batch (${selectedTargetIds.length} Targets)`}
            </button>
          </div>
        </section>
      )}

      {/* STRATEGY 2: GROUPED */}
      {strategy === 'grouped' && (
        <section className="batch-strategy-panel">
          <div className="panel-subhead">
            <h2>Strategy B: Grouped Target Batch</h2>
            <p className="muted">Map M Source FaceSets to grouped target collections or scene folders.</p>
          </div>

          <div className="groups-list">
            {groups.map(group => (
              <div key={group.id} className="group-builder-card">
                <div className="group-header">
                  <input
                    className="group-name-input"
                    value={group.name}
                    onChange={e => {
                      const val = e.target.value;
                      setGroups(prev => prev.map(g => g.id === group.id ? { ...g, name: val } : g));
                    }}
                    placeholder="Group Name"
                  />
                  <div className="group-right">
                    <label className="inline-select">
                      <span>Mapped FaceSet:</span>
                      <select
                        value={group.facesetId}
                        onChange={e => {
                          const val = e.target.value;
                          setGroups(prev => prev.map(g => g.id === group.id ? { ...g, facesetId: val } : g));
                        }}
                      >
                        <option value="">Select FaceSet</option>
                        {sources.map(s => <option key={s.faceset_id} value={s.faceset_id}>{s.name}</option>)}
                      </select>
                    </label>
                    {groups.length > 1 && (
                      <button
                        className="button small"
                        onClick={() => setGroups(prev => prev.filter(g => g.id !== group.id))}
                      >
                        Remove
                      </button>
                    )}
                  </div>
                </div>

                <div className="group-targets-selection">
                  <span className="eyebrow">ASSIGNED TARGETS ({group.targetIds.length})</span>
                  <div className="target-chips-container">
                    {targetPool.map(t => {
                      const assigned = group.targetIds.includes(t.id);
                      return (
                        <button
                          key={t.id}
                          className={`target-chip ${assigned ? 'active' : ''}`}
                          onClick={() => {
                            setGroups(prev => prev.map(g => {
                              if (g.id !== group.id) return g;
                              const nextIds = assigned ? g.targetIds.filter(id => id !== t.id) : [...g.targetIds, t.id];
                              return { ...g, targetIds: nextIds };
                            }));
                          }}
                        >
                          {assigned ? '✓ ' : '+ '} {t.name}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="flex-actions-bar">
            <button
              className="button"
              onClick={() => setGroups(prev => [...prev, {
                id: `group_${Date.now()}`,
                name: `Group ${prev.length + 1}`,
                facesetId: sources[0]?.faceset_id || '',
                targetIds: [],
              }])}
            >
              ＋ Add Target Group
            </button>
            <button
              className="button accent"
              disabled={batchBusy || groups.every(g => g.targetIds.length === 0 || !g.facesetId) || !!featureReason(caps, 'queue')}
              onClick={handleQueueGrouped}
            >
              {batchBusy ? 'Queueing…' : `Queue Grouped Batch (${groups.reduce((acc, g) => acc + (g.facesetId ? g.targetIds.length : 0), 0)} Jobs)`}
            </button>
          </div>
        </section>
      )}

      {/* STRATEGY 3: PER-FILE MATRIX */}
      {strategy === 'matrix' && (
        <section className="batch-strategy-panel">
          <div className="panel-subhead">
            <h2>Strategy C: Per-File Cartesian Matrix Grid</h2>
            <p className="muted">Cartesian product grid (`M Sources × N Targets`) with toggleable intersection cells.</p>
          </div>

          <div className="matrix-toolbar">
            <div className="matrix-counters">
              <span className="matrix-badge">
                <strong>{matrixSelection.length}</strong> / {targetPool.length * sources.length} Cells Active
              </span>
            </div>
            <div className="matrix-controls">
              <button className="button small" onClick={selectAllMatrix}>Select All</button>
              <button className="button small" onClick={clearMatrix}>Clear All</button>
              <button className="button small" onClick={invertMatrix}>Invert</button>
            </div>
          </div>

          <div className="naming-template-bar" style={{ margin: '12px 0' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span className="muted">Output Naming Template:</span>
              <input
                type="text"
                className="input text"
                value={outputNamingTemplate}
                onChange={(e) => setOutputNamingTemplate(e.target.value)}
                style={{ flex: 1, fontFamily: 'monospace', fontSize: '12px' }}
                placeholder="{target}_{source}_{restorer}_{timestamp}.mp4"
              />
            </label>
            <p className="muted" style={{ fontSize: '11px', marginTop: '4px' }}>
              Supported tokens: <code>{'{target}'}</code>, <code>{'{source}'}</code>, <code>{'{restorer}'}</code>, <code>{'{timestamp}'}</code>
            </p>
          </div>

          <div className="matrix-scroll-wrapper">
            <table className="batch-matrix-grid">
              <thead>
                <tr>
                  <th className="corner-header">Target Clips \ Source FaceSets</th>
                  {sources.map(src => (
                    <th key={src.faceset_id} className="source-col-header">
                      <div className="col-header-inner">
                        <strong>{src.name}</strong>
                        <button className="mini-toggle-btn" onClick={() => selectCol(src.faceset_id)}>Col All</button>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {targetPool.map(tgt => (
                  <tr key={tgt.id}>
                    <th className="target-row-header">
                      <div className="row-header-inner">
                        <span className="row-title" title={tgt.target_path}>{tgt.name}</span>
                        <button className="mini-toggle-btn" onClick={() => selectRow(tgt.id)}>Row All</button>
                      </div>
                    </th>
                    {sources.map(src => {
                      const cellKey = `${tgt.id}:${src.faceset_id}`;
                      const isChecked = matrixSelection.includes(cellKey);
                      return (
                        <td
                          key={cellKey}
                          className={`matrix-cell ${isChecked ? 'active' : ''}`}
                          onClick={() => toggleMatrixCell(tgt.id, src.faceset_id)}
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => {}} // Controlled by cell click
                            aria-label={`Pair ${tgt.name} with ${src.name}`}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {(!sources.length || !targetPool.length) && (
            <p className="empty-note muted">Load targets and add at least one FaceSet to populate the grid.</p>
          )}

          <div className="batch-footer-action">
            <button
              className="button accent"
              disabled={batchBusy || matrixSelection.length === 0 || !!featureReason(caps, 'queue')}
              onClick={handleQueueMatrix}
            >
              {batchBusy ? 'Queueing…' : `Queue ${matrixSelection.length} Matrix Cell Jobs`}
            </button>
          </div>
        </section>
      )}

      {/* STRATEGY 4: RECIPE MATRIX */}
      {strategy === 'recipe' && (
        <section className="batch-strategy-panel">
          <div className="panel-subhead">
            <h2>Strategy D: Recipe Matrix (Parameter & Enhancer Presets)</h2>
            <p className="muted">Apply multiple saved parameter/enhancer presets across target clips with a selected identity.</p>
          </div>

          <div className="recipe-source-bar">
            <label className="inline-select">
              <span>Source Identity:</span>
              <select value={recipeSourceId} onChange={e => setRecipeSourceId(e.target.value)}>
                {sources.map(s => <option key={s.faceset_id} value={s.faceset_id}>{s.name}</option>)}
              </select>
            </label>
            <button className="button small" onClick={() => setShowNewRecipeModal(true)}>＋ Create Custom Preset</button>
          </div>

          <div className="recipe-order">
            <span className="muted">Recipe order: drag a recipe (or Alt+arrow keys on its grip) to reorder the columns and the order jobs are queued.</span>
            <DragItems axis="x" items={recipes} getKey={entry => entry.id} getLabel={entry => entry.name} label="Batch recipes" onReorder={setRecipes}
              renderItem={(entry, { controls }) => <span className="recipe-order-chip">{controls}{entry.name}</span>} />
          </div>

          <div className="matrix-scroll-wrapper">
            <table className="batch-matrix-grid">
              <thead>
                <tr>
                  <th className="corner-header">Target Clips \ Recipes</th>
                  {recipes.map(recipe => (
                    <th key={recipe.id} className="source-col-header recipe-header">
                      <div className="recipe-header-card">
                        <strong>{recipe.name}</strong>
                        <small>{recipe.description}</small>
                      </div>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {targetPool.map(tgt => (
                  <tr key={tgt.id}>
                    <th className="target-row-header">
                      <span className="row-title" title={tgt.target_path}>{tgt.name}</span>
                    </th>
                    {recipes.map(recipe => {
                      const cellKey = `${tgt.id}:${recipe.id}`;
                      const isChecked = recipeSelection.includes(cellKey);
                      return (
                        <td
                          key={cellKey}
                          className={`matrix-cell ${isChecked ? 'active' : ''}`}
                          onClick={() => toggleRecipeCell(tgt.id, recipe.id)}
                        >
                          <input
                            type="checkbox"
                            checked={isChecked}
                            onChange={() => {}}
                            aria-label={`Apply ${recipe.name} to ${tgt.name}`}
                          />
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="batch-footer-action">
            <button
              className="button accent"
              disabled={batchBusy || recipeSelection.length === 0 || !recipeSourceId || !!featureReason(caps, 'queue')}
              onClick={handleQueueRecipe}
            >
              {batchBusy ? 'Queueing…' : `Queue ${recipeSelection.length} Recipe Jobs`}
            </button>
          </div>

          {/* New Custom Recipe Modal */}
          {showNewRecipeModal && (
            <div className="modal-backdrop" onClick={() => setShowNewRecipeModal(false)}>
              <div className="project-modal" onClick={e => e.stopPropagation()}>
                <div className="panel-heading">
                  <h2>Create Custom Parameter Recipe</h2>
                  <button className="icon-button" onClick={() => setShowNewRecipeModal(false)}>×</button>
                </div>
                <div className="form-group" style={{ margin: '16px 0', display: 'flex', flexDirection: 'column', gap: '10px' }}>
                  <label>
                    <span>Recipe Name:</span>
                    <input
                      style={{ width: '100%', marginTop: '4px' }}
                      value={newRecipeName}
                      onChange={e => setNewRecipeName(e.target.value)}
                      placeholder="e.g. Extreme Sharpening"
                    />
                  </label>
                  <NumericInput label="Restoration Blend (0.0 to 1.0)" step={0.05} min={0} max={1} value={newRecipeRestoration}
                    onChange={value => setNewRecipeRestoration(value ?? 0)} />
                  <label>
                    <span>Pose Mode:</span>
                    <select
                      style={{ width: '100%', marginTop: '4px' }}
                      value={newRecipePoseMode}
                      onChange={e => setNewRecipePoseMode(e.target.value)}
                    >
                      <option value="retarget">Retarget (3D Pose Tracking)</option>
                      <option value="mesh_blend">Mesh Blend (Full 3D Geometry)</option>
                    </select>
                  </label>
                </div>
                <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                  <button className="button" onClick={() => setShowNewRecipeModal(false)}>Cancel</button>
                  <button className="button accent" onClick={handleAddCustomRecipe}>Add Recipe</button>
                </div>
              </div>
            </div>
          )}
        </section>
      )}
    </main>
  );
}
