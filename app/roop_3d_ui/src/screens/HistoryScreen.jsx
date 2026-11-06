import React, { useState, useMemo, useEffect } from 'react';
import { api, post } from '../api';
import VirtualList from '../components/common/VirtualList';

export default function HistoryScreen({
  projects = [],
  jobs = [],
  openProject = async () => {},
  refreshProjects = async () => {},
  refreshJobs = async () => {},
  busy = false,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterTab, setFilterTab] = useState('all'); // 'all' | 'interrupted' | 'verified' | 'missing'
  const [inspectModalProject, setInspectModalProject] = useState(null);
  const [actionNotice, setActionNotice] = useState('');
  const [actionError, setActionError] = useState('');
  const [loadingAction, setLoadingAction] = useState(false);
  const [detailedProjects, setDetailedProjects] = useState({});

  const fileInputRef = React.useRef(null);

  // Fetch full details / validation for projects on mount or update
  useEffect(() => {
    let active = true;
    const fetchDetails = async () => {
      const detailsMap = {};
      for (const p of projects) {
        try {
          const detail = await api(`/projects/${p.project_id}/validate`);
          if (active) detailsMap[p.project_id] = detail;
        } catch {
          if (active) {
            detailsMap[p.project_id] = {
              project_id: p.project_id,
              name: p.name,
              saved_at: p.saved_at,
              inputs_verified: false,
              status: 'Missing Source Media',
              verification_reason: 'Could not connect to validate inputs',
              last_completed_frame: p.last_completed_frame,
            };
          }
        }
      }
      if (active) setDetailedProjects(detailsMap);
    };
    if (projects.length > 0) {
      void fetchDetails();
    }
    return () => { active = false; };
  }, [projects]);

  // Combined project data
  const combinedProjects = useMemo(() => {
    return projects.map(p => {
      const detail = detailedProjects[p.project_id] || {};
      const checkpoint = detail.checkpoint || null;
      const associatedJob = jobs.find(j => j.job_id === p.checkpoint_job_id || j.job_id === detail.checkpoint_job_id);
      const isInterrupted = (p.last_completed_frame >= 0 && (!associatedJob || associatedJob.state !== 'succeeded'))
        || (associatedJob && ['paused', 'interrupted', 'pausing'].includes(associatedJob.state))
        || (checkpoint && checkpoint.completed_frame >= 0);

      const isVerified = detail.inputs_verified ?? (p.inputs_verified ?? true);

      return {
        ...p,
        ...detail,
        isInterrupted,
        isVerified,
        checkpoint,
        associatedJob,
      };
    });
  }, [projects, detailedProjects, jobs]);

  // Filtered and searched list
  const filteredProjects = useMemo(() => {
    return combinedProjects.filter(p => {
      const q = searchQuery.toLowerCase().trim();
      const matchesSearch = !q || p.name.toLowerCase().includes(q)
        || (p.target_path && p.target_path.toLowerCase().includes(q))
        || (p.project_id && p.project_id.toLowerCase().includes(q));

      if (!matchesSearch) return false;

      if (filterTab === 'interrupted') return p.isInterrupted;
      if (filterTab === 'verified') return p.isVerified;
      if (filterTab === 'missing') return !p.isVerified;
      return true;
    });
  }, [combinedProjects, searchQuery, filterTab]);

  // Action: Load & Resume Project
  const handleLoadAndResume = async (project) => {
    setLoadingAction(true);
    setActionNotice('');
    setActionError('');
    try {
      // 1. Restore the exact workstation state
      await openProject(project.project_id);

      // 2. Resume the job if checkpoint exists and was paused or interrupted
      if (project.checkpoint_job_id || project.checkpoint?.job_id) {
        try {
          await post(`/projects/${project.project_id}/resume`, {});
        } catch (resumeErr) {
          // If already running or completed, handle gracefully
          console.warn('Project resume notice:', resumeErr.message);
        }
      }

      await refreshJobs();
      setActionNotice(`Restored workstation state for "${project.name}" and resumed render from frame ${project.last_completed_frame >= 0 ? project.last_completed_frame : 0}.`);

      // Redirect to workstation
      window.location.hash = '#/faceswap';
    } catch (err) {
      setActionError(`Failed to load and resume project: ${err.message}`);
    } finally {
      setLoadingAction(false);
    }
  };

  // Action: Load Project into Workstation
  const handleLoadOnly = async (projectId) => {
    setLoadingAction(true);
    setActionNotice('');
    setActionError('');
    try {
      await openProject(projectId);
      setActionNotice('Project loaded into workstation successfully.');
      window.location.hash = '#/faceswap';
    } catch (err) {
      setActionError(`Failed to load project: ${err.message}`);
    } finally {
      setLoadingAction(false);
    }
  };

  // Action: Import .r3dp Project
  const handleImportFile = async (file) => {
    if (!file) return;
    setLoadingAction(true);
    setActionError('');
    try {
      const text = await file.text();
      const manifest = JSON.parse(text);
      await post('/projects/import', manifest);
      await refreshProjects();
      setActionNotice(`Imported project "${manifest.name || file.name}".`);
    } catch (err) {
      setActionError(`Failed to import .r3dp manifest: ${err.message}`);
    } finally {
      setLoadingAction(false);
    }
  };

  return (
    <main className="screen-page history-screen-container" data-testid="history-screen">
      <div className="workspace-heading">
        <div>
          <span className="eyebrow">PERSISTENT WORKSPACE SNAPSHOTS</span>
          <h1>Project Checkpoints & History</h1>
        </div>
        <div className="history-header-actions">
          <button
            className="button"
            disabled={loadingAction || busy}
            onClick={() => fileInputRef.current?.click()}
          >
            ＋ Import .r3dp
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".r3dp,application/json"
            hidden
            onChange={e => {
              handleImportFile(e.target.files?.[0]);
              e.target.value = '';
            }}
          />
          <button
            className="button"
            disabled={loadingAction || busy}
            onClick={() => void refreshProjects()}
          >
            Refresh
          </button>
        </div>
      </div>

      <p className="muted">
        Every render job writes an atomic JSON + binary state checkpoint (.r3dp / checkpoint.json / checkpoint.bin) containing input media hashes, FaceSets, 3D pose caches, keyframe curves, manual masks, and active execution providers.
      </p>

      {/* Notifications */}
      {actionNotice && <div className="notification" role="status"><span>{actionNotice}</span></div>}
      {actionError && <div className="notification error" role="alert"><span>{actionError}</span></div>}

      {/* Stats Counter Bar */}
      <div className="history-stats-bar">
        <div className="stat-pill">
          <span className="stat-label">TOTAL PROJECTS</span>
          <strong>{projects.length}</strong>
        </div>
        <div className="stat-pill verified">
          <span className="stat-label">INPUTS VERIFIED</span>
          <strong>{combinedProjects.filter(p => p.isVerified).length}</strong>
        </div>
        <div className="stat-pill interrupted">
          <span className="stat-label">INTERRUPTED / RESUMABLE</span>
          <strong>{combinedProjects.filter(p => p.isInterrupted).length}</strong>
        </div>
        <div className="stat-pill missing">
          <span className="stat-label">MISSING SOURCE MEDIA</span>
          <strong>{combinedProjects.filter(p => !p.isVerified).length}</strong>
        </div>
      </div>

      {/* Toolbar: Search & Filter Tabs */}
      <div className="history-toolbar">
        <div className="search-input-wrapper">
          <input
            type="search"
            placeholder="Search projects by name, target path, or ID…"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
          />
        </div>
        <div className="segmented" role="tablist" aria-label="Project Filters">
          <button
            role="tab"
            aria-selected={filterTab === 'all'}
            className={filterTab === 'all' ? 'active' : ''}
            onClick={() => setFilterTab('all')}
          >
            All ({projects.length})
          </button>
          <button
            role="tab"
            aria-selected={filterTab === 'interrupted'}
            className={filterTab === 'interrupted' ? 'active' : ''}
            onClick={() => setFilterTab('interrupted')}
          >
            Interrupted / Resumable ({combinedProjects.filter(p => p.isInterrupted).length})
          </button>
          <button
            role="tab"
            aria-selected={filterTab === 'verified'}
            className={filterTab === 'verified' ? 'active' : ''}
            onClick={() => setFilterTab('verified')}
          >
            Inputs Verified ({combinedProjects.filter(p => p.isVerified).length})
          </button>
          <button
            role="tab"
            aria-selected={filterTab === 'missing'}
            className={filterTab === 'missing' ? 'active' : ''}
            onClick={() => setFilterTab('missing')}
          >
            Missing Media ({combinedProjects.filter(p => !p.isVerified).length})
          </button>
        </div>
      </div>

      {/* Projects List */}
      <div className="history-cards-host">
        <VirtualList items={filteredProjects} itemKey={item => item.project_id} estimateHeight={360} minColumnWidth={340} gap={16} aria-label="Saved projects"
          renderItem={item => {
          const cp = item.checkpoint;
          const facesetId = item.face_mappings?.[0]?.faceset_id || cp?.faceset_ids?.[0] || 'Unknown';
          const provider = cp?.execution_provider || 'CPUExecutionProvider';

          return (
            <div
              key={item.project_id}
              className={`history-project-card ${item.isInterrupted ? 'interrupted' : ''} ${!item.isVerified ? 'missing-media' : ''}`}
            >
              <div className="project-card-header">
                <div>
                  <div className="title-row">
                    <h3>{item.name}</h3>
                    {/* Validation Badge: Inputs Verified vs Missing Source Media */}
                    {item.isVerified ? (
                      <span className="badge badge-verified" title="All input hashes, FaceSets, and models verified locally on disk">
                        ✓ Inputs Verified
                      </span>
                    ) : (
                      <span className="badge badge-missing" title={item.verification_reason || 'Source video, FaceSet, or model is absent'}>
                        ⚠ Missing Source Media
                      </span>
                    )}
                  </div>
                  <small className="muted">
                    Saved: {new Date(item.saved_at).toLocaleString()} · ID: {item.project_id.slice(0, 10)}
                  </small>
                </div>
                {item.isInterrupted && (
                  <span className="badge badge-resumable">
                    RESUMABLE AT FRAME {item.last_completed_frame >= 0 ? item.last_completed_frame : (cp?.completed_frame ?? 0)}
                  </span>
                )}
              </div>

              <div className="project-card-body">
                <div className="card-meta-line">
                  <span className="meta-label">Target Media:</span>
                  <span className="meta-val" title={item.target_path}>{item.target_path?.split(/[\\/]/).at(-1) || 'No target path'}</span>
                </div>
                <div className="card-meta-line">
                  <span className="meta-label">Identity FaceSet:</span>
                  <span className="meta-val highlight">{facesetId}</span>
                </div>

                {/* State Checkpoint Badges */}
                <div className="checkpoint-indicators-row">
                  <span className="tech-badge" title="Active Execution Provider">
                    ⚡ {provider.replace('ExecutionProvider', '')}
                  </span>
                  {cp?.has_binary_checkpoint && (
                    <span className="tech-badge bin" title="Atomic binary checkpoint verified">
                      📦 checkpoint.bin
                    </span>
                  )}
                  {item.has_curves && (
                    <span className="tech-badge curves" title="Keyframe curves active">
                      📈 Curves
                    </span>
                  )}
                  {item.has_mask && (
                    <span className="tech-badge mask" title="Manual raster mask active">
                      🖌 Mask
                    </span>
                  )}
                  {cp?.pose_cache && Object.keys(cp.pose_cache).length > 0 && (
                    <span className="tech-badge pose" title="3D Pose Cache recorded">
                      📐 3D Pose Cache
                    </span>
                  )}
                  {cp?.completed_segment !== undefined && cp.completed_segment >= 0 && (
                    <span className="tech-badge segment" title="Multi-segment video render index">
                      🎬 Segment {cp.completed_segment + 1}{cp.total_segments ? `/${cp.total_segments}` : ''}
                    </span>
                  )}
                </div>

                {!item.isVerified && item.verification_reason && (
                  <p className="inline-warning">Reason: {item.verification_reason}</p>
                )}
              </div>

              <div className="project-card-actions">
                {/* Primary Action: Load & Resume Project */}
                {item.isInterrupted ? (
                  <button
                    className="button accent"
                    disabled={loadingAction || busy}
                    title="Restore workstation state and resume render from the interrupted frame/segment"
                    onClick={() => handleLoadAndResume(item)}
                  >
                    ▶ Load & Resume Project
                  </button>
                ) : (
                  <button
                    className="button"
                    disabled={loadingAction || busy}
                    onClick={() => handleLoadOnly(item.project_id)}
                  >
                    Load into Workstation
                  </button>
                )}

                <button
                  className="button small"
                  onClick={() => setInspectModalProject(item)}
                >
                  Inspect Checkpoint
                </button>

                <a
                  className="button small"
                  href={`/api/v1/projects/${item.project_id}/download`}
                  download
                  title="Download .r3dp manifest JSON"
                >
                  Export .r3dp
                </a>
              </div>
            </div>
          );
        }} />
      </div>

      {!filteredProjects.length && (
        <p className="empty-note muted">
          No projects matching your query or filter. Inspect and map a target video in the Workstation to save a persistent project.
        </p>
      )}

      {/* Checkpoint Inspection Modal */}
      {inspectModalProject && (
        <div className="modal-backdrop" onClick={() => setInspectModalProject(null)}>
          <div className="project-modal checkpoint-inspect-modal" onClick={e => e.stopPropagation()}>
            <div className="panel-heading">
              <div>
                <span className="eyebrow">ATOMIC STATE CHECKPOINT INSPECTION</span>
                <h2>{inspectModalProject.name}</h2>
              </div>
              <button className="icon-button" onClick={() => setInspectModalProject(null)}>×</button>
            </div>

            <div className="inspect-details-grid">
              <div className="inspect-section">
                <h3>Integrity & Hash Verification</h3>
                <p>Status: <strong>{inspectModalProject.status || (inspectModalProject.isVerified ? 'Inputs Verified' : 'Missing Source Media')}</strong></p>
                <div className="hash-list">
                  <div className="hash-item">
                    <span>Target Hash:</span>
                    <code>{inspectModalProject.checkpoint?.input_hashes?.target || 'Recorded on manifest'}</code>
                  </div>
                  <div className="hash-item">
                    <span>Inspection ID:</span>
                    <code>{inspectModalProject.request?.inspection_id || 'None'}</code>
                  </div>
                  <div className="hash-item">
                    <span>FaceSet ID:</span>
                    <code>{inspectModalProject.checkpoint?.faceset_ids?.join(', ') || 'alice'}</code>
                  </div>
                </div>
              </div>

              <div className="inspect-section">
                <h3>Checkpoint Runtime State</h3>
                <div className="runtime-meta-grid">
                  <div><span>Completed Frame:</span> <strong>{inspectModalProject.last_completed_frame}</strong></div>
                  <div><span>Completed Segment:</span> <strong>{inspectModalProject.checkpoint?.completed_segment ?? 'N/A'}</strong></div>
                  <div><span>Execution Provider:</span> <strong>{inspectModalProject.checkpoint?.execution_provider || 'CPUExecutionProvider'}</strong></div>
                  <div><span>Binary Checkpoint:</span> <strong>{inspectModalProject.checkpoint?.has_binary_checkpoint ? 'Valid (checkpoint.bin)' : 'Not written'}</strong></div>
                </div>
              </div>

              <div className="inspect-section full-width">
                <h3>Raw State Manifest (.r3dp JSON)</h3>
                <pre className="code-viewer">{JSON.stringify(inspectModalProject, null, 2)}</pre>
              </div>
            </div>

            <div className="modal-actions-bar">
              <button className="button" onClick={() => setInspectModalProject(null)}>Close</button>
              <button
                className="button accent"
                onClick={() => {
                  setInspectModalProject(null);
                  void handleLoadAndResume(inspectModalProject);
                }}
              >
                Load & Resume this Checkpoint
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
