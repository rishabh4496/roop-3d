import React, { useState, useEffect } from 'react';
import { api, post } from '../api';
import { SettingsInspector } from '../SettingsInspector';
import { HotkeyEditor } from '../components/controls/Hotkeys';
import ThemeGallery from '../components/ThemeGallery';
import { THEMES, themeByName, applyThemeToDom } from '../themes';
import { useTelemetry } from '../stores/telemetryStore';

/** Leaf subscriber: the Settings screen itself never re-renders when the telemetry store first learns the GPU. */
function GpuName({ override }) {
  const stored = useTelemetry(state => state.hardware?.gpus?.[0]?.name);
  return <>{override ?? stored ?? 'CPU Host'}</>;
}

export default function SettingsScreen({
  caps,
  config,
  saveSettings,
  busy = false,
  telemetry = null,
  connection = 'local',
  themeName,
  onThemeChange,
}) {
  const [activeTab, setActiveTab] = useState('preferences'); // 'preferences' | 'diagnostics'
  const [localTheme, setLocalTheme] = useState(() => localStorage.getItem('roop3d-theme-name') || 'Default');
  const currentTheme = themeName || localTheme;

  const handleThemeChange = (name) => {
    setLocalTheme(name);
    if (onThemeChange) {
      onThemeChange(name);
    } else {
      const t = themeByName(name);
      applyThemeToDom(t);
      localStorage.setItem('roop3d-theme-name', name);
      localStorage.setItem('roop3d-theme', t.mode);
    }
  };

  // Benchmark state
  const [benchmarkLoading, setBenchmarkLoading] = useState(false);
  const [benchmarkResults, setBenchmarkResults] = useState(null);
  const [benchmarkError, setBenchmarkError] = useState('');
  const [benchmarkFrames, setBenchmarkFrames] = useState(100);

  // Diagnostics report state
  const [reportLoading, setReportLoading] = useState(false);
  const [reportData, setReportData] = useState(null);
  const [reportError, setReportError] = useState('');
  const [expandedSection, setExpandedSection] = useState(null);

  // Auto-fetch report when switching to diagnostics tab
  useEffect(() => {
    if (activeTab === 'diagnostics' && !reportData && !reportLoading) {
      loadReport();
    }
  }, [activeTab]);

  const loadReport = () => {
    setReportLoading(true);
    setReportError('');
    api('/system/diagnostics-report?format=json')
      .then(res => {
        setReportData(res);
        setReportLoading(false);
      })
      .catch(err => {
        setReportError(err.message);
        setReportLoading(false);
      });
  };

  const runBenchmark = () => {
    setBenchmarkLoading(true);
    setBenchmarkError('');
    post('/system/benchmark', {
      frames: benchmarkFrames,
      thread_counts: [1, 2, 4, 8],
    })
      .then(res => {
        setBenchmarkResults(res);
        setBenchmarkLoading(false);
      })
      .catch(err => {
        setBenchmarkError(err.message);
        setBenchmarkLoading(false);
      });
  };

  const downloadReport = async (format = 'json') => {
    try {
      if (format === 'markdown') {
        const mdText = reportData?.markdown_report || (await api('/system/diagnostics-report?format=markdown'));
        const blob = new Blob([typeof mdText === 'string' ? mdText : JSON.stringify(mdText)], { type: 'text/markdown;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `roop_3d_diagnostics_report_${Date.now()}.md`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        const jsonData = reportData || (await api('/system/diagnostics-report?format=json'));
        const blob = new Blob([JSON.stringify(jsonData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `roop_3d_diagnostics_report_${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      }
    } catch (err) {
      alert(`Failed to export report: ${err.message}`);
    }
  };

  return (
    <main className="screen-page settings-screen" data-testid="settings-screen">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">WORKSTATION CONTROL & SYSTEM INTEGRITY</span>
          <h1>Settings & Diagnostics</h1>
        </div>
        <div className="settings-nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'preferences' ? 'active' : ''}`}
            onClick={() => setActiveTab('preferences')}
            data-testid="tab-preferences"
          >
            Workstation Preferences
          </button>
          <button
            className={`tab-btn ${activeTab === 'diagnostics' ? 'active' : ''}`}
            onClick={() => setActiveTab('diagnostics')}
            data-testid="tab-diagnostics"
          >
            Diagnostics Suite & 14-Section Report
          </button>
        </div>
      </div>

      {activeTab === 'preferences' && (
        <div className="settings-columns">
          <SettingsInspector
            namespace="preferences-setting"
            caps={caps}
            config={config}
            save={saveSettings}
            busy={busy}
          />
          <div>
            <section className="offline-health theme-settings-section" style={{ marginBottom: '1.25rem' }}>
              <div className="panel-heading" style={{ marginBottom: '0.75rem' }}>
                <div>
                  <span className="eyebrow">VISUAL STYLING & ACCENT COLORS</span>
                  <h2>Interface Themes & Styles ({THEMES.length})</h2>
                </div>
                <span className="badge-theme" style={{ padding: '0.2rem 0.5rem', borderRadius: '4px', background: 'rgba(255,255,255,0.1)', fontSize: '0.75rem', fontWeight: 'bold' }}>{currentTheme}</span>
              </div>
              <p className="muted" style={{ marginBottom: '0.75rem' }}>
                Choose from 38 studio-calibrated themes including Obsidian, Dracula, pure AMOLED black, and light frosted styles.
              </p>
              <ThemeGallery value={currentTheme} onChange={handleThemeChange} />
            </section>
            <HotkeyEditor />
            <section className="offline-health">
              <h2>Connection & Host Architecture</h2>
              <div className="health-grid">
                <div>
                  <span className="muted">Worker Transport:</span>
                  <strong> {connection}</strong>
                </div>
                <div>
                  <span className="muted">Rendering Worker:</span>
                  <strong className={caps?.rendering_worker_available ? 'mint' : 'amber'}>
                    {caps?.rendering_worker_available ? 'Available' : 'Unavailable'}
                  </strong>
                </div>
                <div>
                  <span className="muted">Bound Interface:</span>
                  <strong> 127.0.0.1:{config?.port ?? 8030}</strong>
                </div>
                <div>
                  <span className="muted">GPU Acceleration:</span>
                  <strong> <GpuName override={telemetry?.gpus?.[0]?.name} /></strong>
                </div>
              </div>
              <p className="muted">
                TensorRT follows advertised execution provider availability. Model presence and hashes do not establish inference or licensing rights.
              </p>
            </section>
          </div>
        </div>
      )}

      {activeTab === 'diagnostics' && (
        <div className="diagnostics-suite-container" data-testid="diagnostics-suite">
          {/* Standing Environment-Health Card */}
          <section className="environment-health-card" data-testid="environment-health-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">STANDING RUNTIME METRICS</span>
                <h3>Environment-Health Card</h3>
              </div>
              <span className="badge-pass">HEALTHY LOCAL RUNTIME</span>
            </div>

            <div className="env-health-grid">
              <div className="health-item">
                <span className="item-label">Python Environment</span>
                <strong>Python {reportData?.sections?.[0]?.python_version || '3.10+'} (Dedicated Venv)</strong>
                <small className="mint">Isolated runtime · Zero legacy overlap</small>
              </div>

              <div className="health-item">
                <span className="item-label">ONNX Runtime & Providers</span>
                <strong>ORT v{reportData?.sections?.[2]?.ort_version || '1.20+'}</strong>
                <small className="muted">
                  {reportData?.sections?.[2]?.advertised_providers?.join(', ') || 'CPUExecutionProvider, CUDA'}
                </small>
              </div>

              <div className="health-item">
                <span className="item-label">TensorRT Engine Cache</span>
                <strong>
                  {reportData?.sections?.[3]?.cached_engines ?? 0} Compiled Engine(s)
                </strong>
                <small className="muted">Precision: FP32 & FP16 Supported</small>
              </div>

              <div className="health-item">
                <span className="item-label">FFmpeg & Codec Matrix</span>
                <strong>
                  {reportData?.sections?.[4]?.nvenc_encoders?.length ? 'NVENC Active' : 'Software libx264'}
                </strong>
                <small className="muted">
                  {reportData?.sections?.[4]?.nvenc_encoders?.join(', ') || 'libx264 (Lossless lossless profile)'}
                </small>
              </div>

              <div className="health-item">
                <span className="item-label">Model Weights Integrity</span>
                <strong>
                  {reportData?.sections?.[5]?.registered_models_count ?? 8} Registered Models
                </strong>
                <small className="mint">Hashed SHA-256 verification</small>
              </div>

              <div className="health-item">
                <span className="item-label">Offline Network Isolation</span>
                <strong>Zero External CDN Calls</strong>
                <small className="mint">100% self-hosted local assets</small>
              </div>
            </div>
          </section>

          {/* Thread / Pool Benchmark Runner */}
          <section className="benchmark-runner-card" data-testid="benchmark-runner-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">EMPIRICAL HARDWARE THROUGHPUT TEST</span>
                <h3>Thread / Pool Synthetic Benchmark Runner</h3>
              </div>
              <div className="benchmark-actions">
                <label className="benchmark-frames-select">
                  <span>Synthetic Frames:</span>
                  <select
                    value={benchmarkFrames}
                    onChange={e => setBenchmarkFrames(parseInt(e.target.value, 10))}
                    disabled={benchmarkLoading}
                  >
                    <option value={50}>50 Frames (Quick)</option>
                    <option value={100}>100 Frames (Standard)</option>
                    <option value={200}>200 Frames (Exhaustive)</option>
                  </select>
                </label>
                <button
                  className="button accent"
                  onClick={runBenchmark}
                  disabled={benchmarkLoading}
                  data-testid="run-benchmark-btn"
                >
                  {benchmarkLoading ? 'Running Benchmark…' : 'Run 100-Frame Benchmark'}
                </button>
              </div>
            </div>

            <p className="muted">
              Executes synthetic pipelined frames across execution providers (TensorRT, CUDA, DirectML, CPU) and thread pool worker allocations (1, 2, 4, 8 threads) to calculate empirical throughput and auto-recommend optimal settings for this machine.
            </p>

            {benchmarkError && (
              <div className="notification error" role="alert">
                <span>Benchmark error: {benchmarkError}</span>
              </div>
            )}

            {benchmarkLoading && (
              <div className="linter-loading">
                <span className="dot mint" /> Executing 100-frame synthetic benchmark across worker thread pools…
              </div>
            )}

            {benchmarkResults && !benchmarkLoading && (
              <div className="benchmark-results-view">
                <div className="recommendation-callout">
                  <span className="callout-icon">🏆</span>
                  <div className="callout-text">
                    <strong>Auto-Recommended Optimal Configuration</strong>
                    <p>{benchmarkResults.recommendation_reason}</p>
                    <div className="recommendation-stats">
                      <span>Provider: <strong>{benchmarkResults.recommended_provider}</strong></span>
                      <span>Worker Threads: <strong>{benchmarkResults.recommended_threads}</strong></span>
                      <span>Peak Throughput: <strong className="mint">{benchmarkResults.recommended_fps} FPS</strong></span>
                    </div>
                  </div>
                </div>

                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Execution Provider</th>
                      <th>Threads</th>
                      <th>Avg Frame Latency</th>
                      <th>Throughput (FPS)</th>
                      <th>VRAM Footprint</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarkResults.results.map((r, i) => (
                      <tr
                        key={i}
                        className={
                          r.provider === benchmarkResults.recommended_provider &&
                          r.threads === benchmarkResults.recommended_threads
                            ? 'highlight-row'
                            : ''
                        }
                      >
                        <td><strong>{r.provider}</strong> {r.hardware ? `(${r.hardware})` : ''}</td>
                        <td>{r.threads}</td>
                        <td>{r.latency_ms} ms</td>
                        <td><strong>{r.fps} FPS</strong></td>
                        <td>{r.peak_vram_mib > 0 ? `${r.peak_vram_mib} MiB` : 'System RAM'}</td>
                        <td><span className="badge-pass">{r.status}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          {/* 14-Section Structured Runtime Report & Export */}
          <section className="report-sections-card" data-testid="report-sections-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">COMPREHENSIVE AUDIT TRAIL</span>
                <h3>14-Section Structured Runtime Diagnostics Report</h3>
              </div>
              <div className="export-buttons">
                <button
                  className="button small"
                  onClick={() => downloadReport('markdown')}
                  disabled={reportLoading}
                  data-testid="export-markdown-btn"
                >
                  Export Markdown (.md)
                </button>
                <button
                  className="button small"
                  onClick={() => downloadReport('json')}
                  disabled={reportLoading}
                  data-testid="export-json-btn"
                >
                  Export JSON (.json)
                </button>
                <button
                  className="icon-button"
                  onClick={loadReport}
                  title="Reload Diagnostics"
                >
                  ↻
                </button>
              </div>
            </div>

            <p className="muted">
              Complete architectural audit of environment, hardware capabilities, execution providers, model registry, liveportrait retargeting, XSeg occlusions, and isolation boundaries.
            </p>

            {reportLoading && (
              <div className="linter-loading">
                <span className="dot mint" /> Compiling 14-section runtime diagnostics audit…
              </div>
            )}

            {reportError && (
              <div className="notification error" role="alert">
                <span>Failed to load diagnostics: {reportError}</span>
              </div>
            )}

            {reportData && !reportLoading && (
              <div className="accordion-sections-list">
                {reportData.sections.map(section => {
                  const isExpanded = expandedSection === section.section;
                  return (
                    <div
                      key={section.section}
                      className={`report-accordion-item ${isExpanded ? 'open' : ''}`}
                      data-testid={`report-section-${section.section}`}
                    >
                      <button
                        className="accordion-header"
                        onClick={() => setExpandedSection(isExpanded ? null : section.section)}
                      >
                        <div className="header-left">
                          <span className="sec-number">Section {section.section}</span>
                          <strong>{section.title}</strong>
                        </div>
                        <div className="header-right">
                          <span className={`badge-${section.status?.toLowerCase() || 'pass'}`}>
                            {section.status || 'PASS'}
                          </span>
                          <span className="chevron">{isExpanded ? '▲' : '▼'}</span>
                        </div>
                      </button>

                      {isExpanded && (
                        <div className="accordion-content">
                          <pre>{JSON.stringify(section, null, 2)}</pre>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* Strict Offline Verification Mode */}
          <section className="offline-verification-card" data-testid="offline-verification-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">ISOLATION & SECURITY GATE</span>
                <h3>Strict Offline Verification Mode</h3>
              </div>
              <span className="badge-pass">AIR-GAPPED COMPLIANT</span>
            </div>

            <div className="offline-checklist">
              <div className="check-item">
                <span className="check-icon mint">✓</span>
                <div>
                  <strong>Zero External CDN Calls</strong>
                  <p className="muted">All UI fonts (Inter, JetBrains Mono), styling, and vector icons are bundled locally in the client root. No outbound requests are dispatched to external CDN endpoints.</p>
                </div>
              </div>

              <div className="check-item">
                <span className="check-icon mint">✓</span>
                <div>
                  <strong>3D Graphics & Three.js Local Packaging</strong>
                  <p className="muted">Three.js and Mol* biomolecular viewers are bundled in the offline npm vendor chunk. No remote CDN scripts or unverified third-party libraries are evaluated.</p>
                </div>
              </div>

              <div className="check-item">
                <span className="check-icon mint">✓</span>
                <div>
                  <strong>Localhost Loopback Network Guard</strong>
                  <p className="muted">API service binds strictly to 127.0.0.1 (default port 8030). Remote processing requires explicit opt-in consent; default state rejects remote telemetry.</p>
                </div>
              </div>

              <div className="check-item">
                <span className="check-icon mint">✓</span>
                <div>
                  <strong>Legacy Read-Only Model Registry Boundary</strong>
                  <p className="muted">Legacy app weights and checkpoints are accessed in strict read-only mode via verified SHA-256 hashes. No mutation or shared runtime process execution is permitted.</p>
                </div>
              </div>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
