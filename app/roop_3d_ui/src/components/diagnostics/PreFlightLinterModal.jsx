import React, { useState, useEffect } from 'react';
import { post } from '../../api';

export default function PreFlightLinterModal({
  isOpen = false,
  onClose = () => {},
  onConfirmQueue = () => {},
  targetPath = '',
  inspectionId = '',
  faceMappings = [],
  workstationParameters = null,
  maskBase64 = null,
  hardware = null,
}) {
  const [loading, setLoading] = useState(true);
  const [linterData, setLinterData] = useState(null);
  const [lintError, setLintError] = useState('');
  const [skipLinterNextTime, setSkipLinterNextTime] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    let active = true;
    setLoading(true);
    setLintError('');

    post('/system/preflight-lint', {
      target_path: targetPath,
      inspection_id: inspectionId || null,
      face_mappings: faceMappings,
      workstation_parameters: workstationParameters,
      mask_base64: maskBase64,
    })
      .then(res => {
        if (active) {
          setLinterData(res);
          setLoading(false);
        }
      })
      .catch(err => {
        if (active) {
          setLintError(err.message);
          setLoading(false);
        }
      });

    return () => { active = false; };
  }, [isOpen, targetPath, inspectionId, faceMappings, workstationParameters, maskBase64]);

  if (!isOpen) return null;

  const checks = linterData?.checks || [];
  const hasFailures = checks.some(c => c.status === 'FAIL');
  const hasWarnings = checks.some(c => c.status === 'WARN');

  const handleProceed = () => {
    if (skipLinterNextTime) {
      sessionStorage.setItem('roop3d_skip_linter', 'true');
    }
    onConfirmQueue();
    onClose();
  };

  return (
    <div className="modal-backdrop" onClick={onClose} data-testid="preflight-linter-modal">
      <div className="project-modal preflight-linter-modal" onClick={e => e.stopPropagation()}>
        <div className="panel-heading">
          <div>
            <span className="eyebrow">AUTOMATED PIPELINE SAFETY & AUDIT</span>
            <h2>Pre-Flight Render Linter</h2>
          </div>
          <button className="icon-button" onClick={onClose} aria-label="Close Pre-Flight Linter">×</button>
        </div>

        <p className="muted">
          Pre-flight linting validates 3D orientation coverage, peak VRAM allocation, out-of-frame boundaries, occluder seams, and output encoder write permissions before dispatching render jobs.
        </p>

        {loading && (
          <div className="linter-loading">
            <span className="dot mint" /> Running pre-flight diagnostic passes…
          </div>
        )}

        {lintError && (
          <div className="notification error" role="alert">
            <span>Pre-flight check failed to query engine: {lintError}</span>
          </div>
        )}

        {!loading && linterData && (
          <div className="linter-body">
            <div className={`linter-summary-banner ${linterData.status.toLowerCase()}`}>
              <div className="summary-left">
                <strong>Pre-Flight Status: {linterData.status}</strong>
                <small>
                  {hasFailures
                    ? 'Critical issues detected. Resolve errors before rendering.'
                    : hasWarnings
                    ? 'Warnings flagged. Pipeline can proceed with automated guardrails.'
                    : 'All 4 pre-flight validation gates passed cleanly.'}
                </small>
              </div>
              <span className={`status-badge-pill ${linterData.status.toLowerCase()}`}>
                {linterData.status}
              </span>
            </div>

            <div className="linter-checks-list">
              {checks.map(check => (
                <div key={check.id} className={`linter-check-card ${check.status.toLowerCase()}`}>
                  <div className="check-card-header">
                    <span className={`check-status-dot ${check.status.toLowerCase()}`}>
                      {check.status === 'PASS' ? '✓' : check.status === 'WARN' ? '⚠' : '✕'}
                    </span>
                    <strong>{check.name}</strong>
                    <span className={`check-pill ${check.status.toLowerCase()}`}>{check.status}</span>
                  </div>
                  <p className="check-message">{check.message}</p>
                  {check.details && check.details.length > 0 && (
                    <ul className="check-details-list">
                      {check.details.map((item, idx) => (
                        <li key={idx}><small>{item}</small></li>
                      ))}
                    </ul>
                  )}
                  {check.id === 'vram_headroom' && check.estimated_vram_mib && (
                    <div className="vram-meter-box">
                      <div className="meter-labels">
                        <span>Estimated Peak VRAM: <strong>{check.estimated_vram_mib} MiB</strong></span>
                        <span>Available: <strong>{check.available_vram_mib} MiB</strong></span>
                      </div>
                      <progress
                        value={check.estimated_vram_mib}
                        max={Math.max(check.estimated_vram_mib, check.available_vram_mib || 4096)}
                      />
                    </div>
                  )}
                </div>
              ))}
            </div>

            <div className="linter-options-row">
              <label className="skip-linter-label">
                <input
                  type="checkbox"
                  checked={skipLinterNextTime}
                  onChange={e => setSkipLinterNextTime(e.target.checked)}
                />
                <span>Skip pre-flight linter for remainder of this session</span>
              </label>
            </div>
          </div>
        )}

        <div className="modal-actions-bar">
          <button className="button" onClick={onClose}>
            Cancel & Inspect Workstation
          </button>
          <button
            className={`button ${hasFailures ? '' : 'accent'}`}
            disabled={loading || (hasFailures && !skipLinterNextTime)}
            onClick={handleProceed}
          >
            {hasFailures ? 'Blocked by Errors' : 'Proceed & Queue Render ↗'}
          </button>
        </div>
      </div>
    </div>
  );
}
