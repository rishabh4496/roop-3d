import React, { useState, useEffect, useMemo } from 'react';

// Default 9-stage pipeline definition
export const PIPELINE_STAGES = [
  { id: 'frame_decode', label: 'Frame Decode', hw: 'NVDEC / CPU', defaultMs: 2.1, color: '#38bdf8' },
  { id: 'smart_skip_roi', label: 'Smart-Skip / ROI Prepass', hw: 'CPU Motion Flow', defaultMs: 1.4, color: '#2dd4bf' },
  { id: 'insightface_det', label: 'InsightFace Detection', hw: 'TensorRT / ONNX (YOLOv8-Face)', defaultMs: 6.8, color: '#818cf8' },
  { id: 'mesh_pose_solve', label: '3D Mesh & Pose Solve', hw: 'LivePortrait Keypoints', defaultMs: 5.2, color: '#a78bfa' },
  { id: 'swapper_inference', label: 'Swapper Inference', hw: 'inswapper_128 (FP32 Graph)', defaultMs: 9.6, color: '#f43f5e' },
  { id: 'mask_xseg', label: 'Mask & XSeg', hw: 'BiSeNet / Occluder Mask', defaultMs: 4.1, color: '#fb923c' },
  { id: 'restorer_inference', label: 'Restorer (TRT/ONNX)', hw: 'GPEN-512 / CodeFormer', defaultMs: 14.5, color: '#f59e0b' },
  { id: 'grain_color_match', label: 'Grain & Color Match', hw: 'Reinhard / Alpha Blending', defaultMs: 2.8, color: '#10b981' },
  { id: 'nvenc_encode', label: 'NVENC Encode', hw: 'FFmpeg NVENC Hardware', defaultMs: 3.2, color: '#06b6d4' },
];

export const DEFAULT_TEMPORAL_SETTINGS = {
  filter: 'one_euro', // 'one_euro' | 'kalman' | 'none'
  minCutoff: 1.0, // 0.1 to 5.0 Hz
  beta: 0.05, // 0.001 to 1.0
  sceneCutSensitivity: 0.35, // 0.1 to 0.9
  roiReuse: true,
  antiFlickerNormalization: true,
};

export default function BottleneckFlamegraph({
  latency = null,
  activeJob = null,
  compact = false,
  onTemporalChange = null,
}) {
  const [selectedStageId, setSelectedStageId] = useState(null);
  const [activeTab, setActiveTab] = useState('flamegraph'); // 'flamegraph' | 'temporal'
  const [temporalSettings, setTemporalSettings] = useState(() => {
    try {
      const saved = localStorage.getItem('roop3d_temporal_settings');
      return saved ? { ...DEFAULT_TEMPORAL_SETTINGS, ...JSON.parse(saved) } : DEFAULT_TEMPORAL_SETTINGS;
    } catch {
      return DEFAULT_TEMPORAL_SETTINGS;
    }
  });
  const [temporalSavedNotice, setTemporalSavedNotice] = useState(false);

  // Compute timings from latency props or defaults
  const stageData = useMemo(() => {
    return PIPELINE_STAGES.map(stage => {
      let ms = stage.defaultMs;
      if (latency && typeof latency === 'object') {
        // Try exact match or case-insensitive partial match
        if (typeof latency[stage.id] === 'number') {
          ms = latency[stage.id];
        } else {
          const matchKey = Object.keys(latency).find(
            k => k.toLowerCase().replace(/[^a-z0-9]/g, '') === stage.id.toLowerCase().replace(/[^a-z0-9]/g, '')
          );
          if (matchKey && typeof latency[matchKey] === 'number') {
            ms = latency[matchKey];
          }
        }
      }
      return {
        ...stage,
        ms: Math.max(0.1, Number(ms.toFixed(2))),
      };
    });
  }, [latency]);

  const totalMs = useMemo(() => {
    return Number(stageData.reduce((acc, s) => acc + s.ms, 0).toFixed(2));
  }, [stageData]);

  const fps = useMemo(() => {
    return totalMs > 0 ? (1000 / totalMs).toFixed(1) : '0';
  }, [totalMs]);

  // Identify bottleneck
  const primaryBottleneck = useMemo(() => {
    if (!stageData.length) return null;
    return stageData.reduce((max, s) => (s.ms > max.ms ? s : max), stageData[0]);
  }, [stageData]);

  const bottleneckAdvice = useMemo(() => {
    if (!primaryBottleneck) return '';
    switch (primaryBottleneck.id) {
      case 'restorer_inference':
        return 'Restorer consumes the largest portion of frame budget. Enabling TensorRT FP16 execution engine can reduce restorer latency by up to 55%.';
      case 'swapper_inference':
        return 'Swapper graph is the primary constraint. Ensure GPU execution provider is selected and input face resolution matches 128x128.';
      case 'insightface_det':
        return 'InsightFace detector is taking significant time. Enable Motion-Compensated ROI Frame Reuse in Temporal Controls to skip full-frame scans.';
      case 'mesh_pose_solve':
        return '3D Pose Solve is active on all frames. Adjusting landmark smoothing or switching from volumetric blend to retarget can yield speedups.';
      case 'nvenc_encode':
        return 'NVENC encoding latency elevated. Check FFmpeg preset (p1-p4) or rate control mode in Settings.';
      default:
        return `${primaryBottleneck.label} is currently the primary stage bottleneck (${primaryBottleneck.ms} ms, ${((primaryBottleneck.ms / totalMs) * 100).toFixed(1)}% of frame budget).`;
    }
  }, [primaryBottleneck, totalMs]);

  // Update temporal settings and persist
  const handleUpdateTemporal = (key, value) => {
    const next = { ...temporalSettings, [key]: value };
    setTemporalSettings(next);
    try {
      localStorage.setItem('roop3d_temporal_settings', JSON.stringify(next));
    } catch {
      // Ignore quota
    }
    if (onTemporalChange) {
      onTemporalChange(next);
    }
    setTemporalSavedNotice(true);
    setTimeout(() => setTemporalSavedNotice(false), 2000);
  };

  const handleResetTemporal = () => {
    setTemporalSettings(DEFAULT_TEMPORAL_SETTINGS);
    try {
      localStorage.setItem('roop3d_temporal_settings', JSON.stringify(DEFAULT_TEMPORAL_SETTINGS));
    } catch {}
    if (onTemporalChange) {
      onTemporalChange(DEFAULT_TEMPORAL_SETTINGS);
    }
  };

  const selectedStage = useMemo(() => {
    return stageData.find(s => s.id === selectedStageId) || null;
  }, [stageData, selectedStageId]);

  return (
    <div className={`bottleneck-flamegraph-container ${compact ? 'compact' : ''}`} data-testid="bottleneck-flamegraph">
      <div className="flamegraph-header">
        <div className="header-titles">
          <span className="eyebrow">PER-FRAME EXECUTION PROFILER</span>
          <h3>Pipeline Bottleneck Flamegraph</h3>
        </div>
        <div className="flamegraph-tabs">
          <button
            className={`tab-btn ${activeTab === 'flamegraph' ? 'active' : ''}`}
            onClick={() => setActiveTab('flamegraph')}
          >
            Flamegraph & Stacked Timeline
          </button>
          <button
            className={`tab-btn ${activeTab === 'temporal' ? 'active' : ''}`}
            onClick={() => setActiveTab('temporal')}
          >
            Temporal Tracking & Flicker Control
          </button>
        </div>
      </div>

      {activeTab === 'flamegraph' && (
        <div className="flamegraph-body">
          {/* Top Metrics Row */}
          <div className="flamegraph-metrics-row">
            <div className="metric-pill">
              <small>Total Frame Latency</small>
              <strong>{totalMs} ms</strong>
            </div>
            <div className="metric-pill">
              <small>Effective Throughput</small>
              <strong className="mint">{fps} FPS</strong>
            </div>
            <div className="metric-pill highlight">
              <small>Primary Bottleneck</small>
              <strong className="amber">{primaryBottleneck?.label || 'None'}</strong>
            </div>
            {activeJob && (
              <div className="metric-pill">
                <small>Active Job</small>
                <strong>{activeJob.job_id.slice(0, 8)} ({activeJob.state})</strong>
              </div>
            )}
          </div>

          {/* Primary Bottleneck Advisory */}
          {primaryBottleneck && (
            <div className="flamegraph-advisory-banner">
              <span className="advisory-icon">⚡</span>
              <div className="advisory-text">
                <strong>Bottleneck Optimization: {primaryBottleneck.label} ({primaryBottleneck.ms} ms · {((primaryBottleneck.ms / totalMs) * 100).toFixed(1)}%)</strong>
                <p>{bottleneckAdvice}</p>
              </div>
            </div>
          )}

          {/* Stacked Timeline Bar */}
          <div className="stacked-timeline-wrapper">
            <div className="timeline-title-row">
              <span className="label">100% Normalized Frame Budget</span>
              <span className="muted">{totalMs} ms total span</span>
            </div>
            <div className="stacked-timeline-bar" role="progressbar" aria-label="Stacked frame stage timeline">
              {stageData.map(stage => {
                const pct = totalMs > 0 ? (stage.ms / totalMs) * 100 : 0;
                return (
                  <div
                    key={stage.id}
                    className={`timeline-segment ${selectedStageId === stage.id ? 'active' : ''}`}
                    style={{
                      width: `${pct}%`,
                      backgroundColor: stage.color,
                    }}
                    title={`${stage.label}: ${stage.ms} ms (${pct.toFixed(1)}%)`}
                    onClick={() => setSelectedStageId(stage.id === selectedStageId ? null : stage.id)}
                  />
                );
              })}
            </div>
          </div>

          {/* Horizontal Flamegraph Stage Rows */}
          <div className="flamegraph-stages-list">
            {stageData.map((stage, idx) => {
              const pct = totalMs > 0 ? (stage.ms / totalMs) * 100 : 0;
              const isBottleneck = primaryBottleneck?.id === stage.id;
              const isSelected = selectedStageId === stage.id;
              const maxMs = Math.max(...stageData.map(s => s.ms), 1);
              const barWidthPct = (stage.ms / maxMs) * 100;

              return (
                <div
                  key={stage.id}
                  className={`flamegraph-row ${isBottleneck ? 'bottleneck' : ''} ${isSelected ? 'selected' : ''}`}
                  onClick={() => setSelectedStageId(isSelected ? null : stage.id)}
                  data-testid={`flamegraph-stage-${stage.id}`}
                >
                  <div className="stage-meta">
                    <span className="stage-index">{idx + 1}.</span>
                    <span className="stage-name">{stage.label}</span>
                    <span className="stage-hw muted">{stage.hw}</span>
                  </div>

                  <div className="stage-bar-track">
                    <div
                      className="stage-bar-fill"
                      style={{
                        width: `${barWidthPct}%`,
                        backgroundColor: stage.color,
                      }}
                    />
                  </div>

                  <div className="stage-timing">
                    <span className="stage-ms"><strong>{stage.ms}</strong> ms</span>
                    <span className="stage-pct muted">({pct.toFixed(1)}%)</span>
                    {isBottleneck && <span className="badge-warning">BOTTLENECK</span>}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Selected Stage Detail Drawer */}
          {selectedStage && (
            <div className="stage-inspector-card">
              <div className="inspector-header">
                <div>
                  <span className="eyebrow">STAGE PROFILE INSPECTION</span>
                  <h4>{selectedStage.label}</h4>
                </div>
                <button className="icon-button" onClick={() => setSelectedStageId(null)}>×</button>
              </div>
              <div className="inspector-grid">
                <div>
                  <span className="muted">Stage Execution Time:</span>
                  <strong> {selectedStage.ms} ms</strong>
                </div>
                <div>
                  <span className="muted">Frame Budget Share:</span>
                  <strong> {((selectedStage.ms / totalMs) * 100).toFixed(1)}%</strong>
                </div>
                <div>
                  <span className="muted">Hardware Accelerator:</span>
                  <strong> {selectedStage.hw}</strong>
                </div>
                <div>
                  <span className="muted">Pipeline Status:</span>
                  <strong className={primaryBottleneck?.id === selectedStage.id ? 'amber' : 'mint'}>
                    {primaryBottleneck?.id === selectedStage.id ? 'Primary Bottleneck' : 'Nominal'}
                  </strong>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'temporal' && (
        <div className="temporal-controls-body" data-testid="temporal-controls-panel">
          <div className="temporal-intro">
            <h4>Temporal Face Tracking & Anti-Flicker Architecture</h4>
            <p className="muted">
              Configure multi-frame landmark stabilization, scene-cut threshold triggers, and motion-compensated ROI reuse to eliminate micro-jitter and temporal luminance flicker across synthesized sequences.
            </p>
          </div>

          <div className="temporal-form-grid">
            {/* Smoothing Filter Algorithm */}
            <div className="temporal-field">
              <label htmlFor="filter-select">
                <strong>Landmark Smoothing Filter</strong>
                <span className="field-desc">Select the temporal stabilization algorithm applied to facial landmarks and 3D pose vertices.</span>
              </label>
              <select
                id="filter-select"
                value={temporalSettings.filter}
                onChange={e => handleUpdateTemporal('filter', e.target.value)}
              >
                <option value="one_euro">One-Euro Filter (Adaptive low-latency jitter suppression)</option>
                <option value="kalman">Kalman Filter (Linear quadratic state estimation)</option>
                <option value="none">Disabled (Raw per-frame detection)</option>
              </select>
            </div>

            {/* Min Cutoff Frequency */}
            <div className="temporal-field">
              <div className="label-with-val">
                <label htmlFor="min-cutoff-slider">
                  <strong>Min Cutoff Frequency (fc_min): {temporalSettings.minCutoff} Hz</strong>
                </label>
                <span className="field-desc">Lower values reduce high-frequency landmark jitter during stationary pauses. Range: 0.1 – 5.0 Hz.</span>
              </div>
              <input
                id="min-cutoff-slider"
                type="range"
                min="0.1"
                max="5.0"
                step="0.05"
                value={temporalSettings.minCutoff}
                onChange={e => handleUpdateTemporal('minCutoff', parseFloat(e.target.value))}
              />
            </div>

            {/* Speed Coefficient (Beta) */}
            <div className="temporal-field">
              <div className="label-with-val">
                <label htmlFor="beta-slider">
                  <strong>Speed Coefficient (Beta): {temporalSettings.beta}</strong>
                </label>
                <span className="field-desc">Controls responsiveness to sudden head velocity. Higher values eliminate lag during fast head turns. Range: 0.001 – 1.0.</span>
              </div>
              <input
                id="beta-slider"
                type="range"
                min="0.001"
                max="1.0"
                step="0.005"
                value={temporalSettings.beta}
                onChange={e => handleUpdateTemporal('beta', parseFloat(e.target.value))}
              />
            </div>

            {/* Scene-Cut Threshold */}
            <div className="temporal-field">
              <div className="label-with-val">
                <label htmlFor="scenecut-slider">
                  <strong>Scene-Cut Sensitivity Threshold: {temporalSettings.sceneCutSensitivity}</strong>
                </label>
                <span className="field-desc">ArcFace feature cosine shift trigger that flushes temporal history to prevent ghosting across edit cuts. Range: 0.1 – 0.9.</span>
              </div>
              <input
                id="scenecut-slider"
                type="range"
                min="0.1"
                max="0.9"
                step="0.02"
                value={temporalSettings.sceneCutSensitivity}
                onChange={e => handleUpdateTemporal('sceneCutSensitivity', parseFloat(e.target.value))}
              />
            </div>

            {/* Motion-compensated ROI Frame Reuse Toggle */}
            <div className="temporal-toggle-field">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={temporalSettings.roiReuse}
                  onChange={e => handleUpdateTemporal('roiReuse', e.target.checked)}
                />
                <div>
                  <strong>Motion-Compensated ROI Frame Reuse</strong>
                  <span className="field-desc">
                    Reuses detected face bounding boxes with optical flow offset across adjacent frames, bypassing full-frame InsightFace detection for up to 35% speedup on static camera shots.
                  </span>
                </div>
              </label>
            </div>

            {/* Temporal Color Anti-Flicker Normalization Toggle */}
            <div className="temporal-toggle-field">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={temporalSettings.antiFlickerNormalization}
                  onChange={e => handleUpdateTemporal('antiFlickerNormalization', e.target.checked)}
                />
                <div>
                  <strong>Temporal Color / Exposure Anti-Flicker Normalization</strong>
                  <span className="field-desc">
                    Harmonizes color temperature, gain, and gamma over a sliding temporal window of 5 frames to eliminate lighting pulsation in restored regions.
                  </span>
                </div>
              </label>
            </div>
          </div>

          <div className="temporal-actions">
            <button className="button" onClick={handleResetTemporal}>
              Reset to Recommended Defaults
            </button>
            {temporalSavedNotice && (
              <span className="save-notice mint">✓ Temporal parameters saved and active</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
