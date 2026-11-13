import React, { useState, useEffect, useMemo, useRef } from 'react';
import BottleneckFlamegraph from '../diagnostics/BottleneckFlamegraph';
import PipelineSaturationPanel from '../diagnostics/PipelineSaturationPanel';
import TrackingTelemetryPanel from './TrackingTelemetryPanel';
import { publishHardware, publishProgress, useTelemetry } from '../../stores/telemetryStore';

/**
 * Standard 5-stage facial transformation pipeline stages.
 */
export const PIPELINE_STAGES = [
  { id: 'detection', label: 'Detection', abbr: 'DET' },
  { id: 'landmark', label: 'Landmark', abbr: 'LMK' },
  { id: 'warp', label: 'Warp', abbr: 'WRP' },
  { id: 'restoration', label: 'Restoration', abbr: 'RST' },
  { id: 'blend', label: 'Blend', abbr: 'BLN' },
];

/**
 * Resolve primary active hardware execution provider name.
 * @param {any} telemetry
 * @returns {{ name: string; type: string; color: string; badge: string }}
 */
export function resolveProvider(telemetry) {
  if (!telemetry) {
    return { name: 'CPU Reference', type: 'cpu', color: '#8d9c9f', badge: 'CPU' };
  }
  const providers = telemetry.advertised_providers || [];
  if (providers.some(p => p.toLowerCase().includes('tensorrt'))) {
    return { name: 'NVIDIA TensorRT', type: 'tensorrt', color: '#10b981', badge: 'TRT' };
  }
  if (providers.some(p => p.toLowerCase().includes('cuda')) || (telemetry.gpus && telemetry.gpus.length > 0)) {
    return { name: 'NVIDIA CUDA', type: 'cuda', color: '#00f5d4', badge: 'CUDA' };
  }
  if (providers.some(p => p.toLowerCase().includes('directml') || p.toLowerCase().includes('dml'))) {
    return { name: 'DirectML (Windows DX12)', type: 'directml', color: '#38bdf8', badge: 'DML' };
  }
  if (providers.some(p => p.toLowerCase().includes('rocm'))) {
    return { name: 'AMD ROCm (HIP)', type: 'rocm', color: '#f43f5e', badge: 'ROCM' };
  }
  if (providers.some(p => p.toLowerCase().includes('coreml'))) {
    return { name: 'Apple Silicon (Metal/ANE)', type: 'coreml', color: '#a855f7', badge: 'APPLE' };
  }
  return { name: 'CPU Reference', type: 'cpu', color: '#8d9c9f', badge: 'CPU' };
}

/**
 * Identify real-time pipeline bottleneck.
 * @param {any} telemetry
 * @param {number} fps
 * @param {string} stage
 * @returns {{ label: string; severity: 'optimal' | 'warning' | 'critical' }}
 */
export function computeBottleneck(telemetry, fps = 0, stage = '') {
  if (!telemetry) {
    return { label: 'Optimal / Offline', severity: 'optimal' };
  }
  const gpu = telemetry.gpus?.[0];
  if (gpu) {
    const freeMib = gpu.free_vram_mib || 0;
    const totalMib = gpu.total_vram_mib || 1;
    const usedPct = ((totalMib - freeMib) / totalMib) * 100;
    if (usedPct > 90) {
      return { label: 'VRAM Bound (>90% memory allocation)', severity: 'critical' };
    }
    if (usedPct > 80) {
      return { label: 'VRAM Pressure (>80% allocation)', severity: 'warning' };
    }
  }
  if (fps > 0 && fps < 8) {
    if (stage === 'restoration') {
      return { label: 'Face Restoration Model Bound', severity: 'warning' };
    }
    if (stage === 'warp') {
      return { label: 'Affine Warp / Tensor Flow Bound', severity: 'warning' };
    }
    return { label: 'Compute Bound (Low FPS)', severity: 'warning' };
  }
  return { label: 'Optimal Pipeline Throughput', severity: 'optimal' };
}

/**
 * System HUD & Diagnostic Pill (`PipelinePill.jsx`).
 * Connects to `/ws/telemetry` WebSocket endpoint and displays a floating, collapsible top-right HUD.
 *
 * @param {{
 *   telemetry?: any;
 *   activeJob?: any;
 *   defaultExpanded?: boolean;
 * }} props
 */
export default function PipelinePill({
  telemetry = null,
  activeJob = null,
  defaultExpanded = false,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);
  // Stage 11: hardware + progress come from the atomic telemetry store, so this pill (and only the components that
  // subscribe to a slice) re-render per tick. `telemetry` stays as an explicit override for tests and storybook use.
  const storedTelemetry = useTelemetry(state => state.hardware);
  const liveProgress = useTelemetry(state => state.progress);
  const liveTelemetry = telemetry ?? storedTelemetry;
  const [wsConnected, setWsConnected] = useState(false);
  const wsRef = useRef(null);

  // Connect to /ws/telemetry WebSocket
  useEffect(() => {
    if (typeof window === 'undefined') return;

    let active = true;
    let retryTimer = null;

    const connectWs = () => {
      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Try /ws/telemetry directly or /api/v1/ws/telemetry
        const wsUrl = `${protocol}//${window.location.host}/ws/telemetry`;
        const socket = new WebSocket(wsUrl);
        wsRef.current = socket;

        socket.onopen = () => {
          if (active) setWsConnected(true);
        };

        socket.onmessage = (event) => {
          if (!active) return;
          try {
            const data = JSON.parse(event.data);
            if (data.type === 'system.telemetry' && data.hardware) {
              publishHardware(data.hardware);
            } else if (data.type === 'job.progress') {
              publishProgress(data);
            }
          } catch {
            // Ignore malformed payloads
          }
        };

        socket.onclose = () => {
          if (active) {
            setWsConnected(false);
            retryTimer = setTimeout(connectWs, 3000);
          }
        };

        socket.onerror = () => {
          socket.close();
        };
      } catch {
        // Fallback
      }
    };

    connectWs();

    return () => {
      active = false;
      if (retryTimer) clearTimeout(retryTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const provider = useMemo(() => resolveProvider(liveTelemetry), [liveTelemetry]);

  // VRAM metrics
  const gpu = liveTelemetry?.gpus?.[0] || null;
  const vramStats = useMemo(() => {
    if (!gpu || gpu.total_vram_mib <= 0) {
      return {
        usedMib: 0,
        totalMib: 0,
        freeMib: 0,
        usedPct: 0,
        formattedUsed: '0 GB',
        formattedTotal: '0 GB',
      };
    }
    const total = gpu.total_vram_mib;
    const free = gpu.free_vram_mib;
    const used = Math.max(0, total - free);
    const pct = Math.min(100, Math.round((used / total) * 100));
    return {
      usedMib: used,
      totalMib: total,
      freeMib: free,
      usedPct: pct,
      formattedUsed: (used / 1024).toFixed(1) + ' GB',
      formattedTotal: (total / 1024).toFixed(1) + ' GB',
    };
  }, [gpu]);

  // Current FPS and Stage
  const progress = liveProgress || activeJob?.progress || null;
  const currentStage = progress?.stage || (activeJob?.state === 'running' ? 'warp' : 'idle');
  const currentFps = progress?.fps || 0;
  const gpuUtil = liveTelemetry?.gpu_utilization_pct ?? gpu?.gpu_utilization_pct ?? (gpu ? vramStats.usedPct : 0);
  const bottleneck = useMemo(
    () => computeBottleneck(liveTelemetry, currentFps, currentStage),
    [liveTelemetry, currentFps, currentStage]
  );

  return (
    <div
      className={`pipeline-hud-container ${expanded ? 'expanded' : 'collapsed'}`}
      role="region"
      aria-label="Hardware Diagnostics and Pipeline HUD"
    >
      {/* 1. Diagnostic Pill (Always Visible Top-Right) */}
      <button
        type="button"
        className="pipeline-pill-btn"
        onClick={() => setExpanded(prev => !prev)}
        title={expanded ? 'Collapse Diagnostics HUD' : 'Expand Diagnostics HUD'}
        aria-expanded={expanded}
      >
        <span className="pill-dot-pulse" style={{ background: wsConnected ? '#00f5d4' : '#dfb977' }} />
        <span className="pill-badge" style={{ color: provider.color }}>{provider.type === 'tensorrt' ? 'TensorRT FP16' : provider.badge}</span>
        <span className="pill-sep">|</span>
        <span className="pill-gpu-util">GPU {gpuUtil}%</span>
        <span className="pill-sep">|</span>
        <span className="pill-metric">
          {vramStats.totalMib > 0 ? `${vramStats.formattedUsed} / ${vramStats.formattedTotal}` : `${liveTelemetry?.logical_cpu_count || 4} Cores`}
        </span>
        <span className="pill-sep">|</span>
        <span className="pill-fps">{currentFps > 0 ? `${currentFps.toFixed(1)} FPS` : '0.0 FPS'}</span>
        <span className="pill-sep">|</span>
        <span className="pill-stage">{currentStage.toUpperCase()}</span>
        <span className="pill-chevron">{expanded ? '▲' : '▼'}</span>
      </button>

      {/* 2. Expanded Floating HUD Modal Panel */}
      {expanded && (
        <div className="pipeline-hud-panel" aria-label="Expanded Hardware Diagnostics">
          <div className="hud-header">
            <div className="hud-title-wrap">
              <span className="eyebrow">LIVE TELEMETRY HUD</span>
              <h4>Hardware &amp; Engine Diagnostics</h4>
            </div>
            <button
              type="button"
              className="hud-close-btn"
              onClick={() => setExpanded(false)}
              aria-label="Close HUD"
            >
              ×
            </button>
          </div>

          {/* Active Provider Card */}
          <div className="hud-card provider-card">
            <div className="hud-card-row">
              <span className="hud-label">ACTIVE PROVIDER:</span>
              <strong className="hud-val" style={{ color: provider.color }}>
                {provider.name}
              </strong>
            </div>
            <div className="hud-sub-row">
              <small>Driver: {gpu?.driver_version || 'System Native'}</small>
              <span className={`status-tag ${wsConnected ? 'online' : 'offline'}`}>
                {wsConnected ? 'WebSocket Online' : 'Polling Sync'}
              </span>
            </div>
          </div>

          {/* GPU VRAM Usage Bar */}
          <div className="hud-card vram-card">
            <div className="hud-card-row">
              <span className="hud-label">GPU VRAM USAGE:</span>
              <strong className="hud-val">
                {vramStats.formattedUsed} / {vramStats.formattedTotal} ({vramStats.usedPct}%)
              </strong>
            </div>
            <div className="vram-meter-bar">
              <div
                className={`vram-meter-fill ${vramStats.usedPct > 85 ? 'high' : ''}`}
                style={{ width: `${vramStats.usedPct}%` }}
              />
            </div>
            <div className="hud-sub-row">
              <small>{(vramStats.freeMib / 1024).toFixed(1)} GB Available</small>
              {vramStats.usedPct > 85 && <span className="warning-text">⚠️ High Allocation</span>}
            </div>
          </div>

          {/* Processing FPS and Current Stage Breadcrumbs */}
          <div className="hud-card pipeline-stage-card">
            <div className="hud-card-row">
              <span className="hud-label">PIPELINE THROUGHPUT:</span>
              <strong className="hud-val highlight">
                {currentFps > 0 ? `${currentFps.toFixed(1)} FPS` : 'Idle / Standby'}
              </strong>
            </div>

            {/* 5 Pipeline Stages Breadcrumb */}
            <div className="pipeline-stages-track">
              {PIPELINE_STAGES.map((s, idx) => {
                const isActive = currentStage.toLowerCase().includes(s.id);
                return (
                  <div key={s.id} className={`pipeline-step ${isActive ? 'active' : ''}`}>
                    <span className="step-num">{idx + 1}</span>
                    <span className="step-name">{s.label}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Measured per-frame stage breakdown (Decode … Encode) */}
          <div className="hud-card hud-flamegraph-card" data-testid="hud-flamegraph">
            <BottleneckFlamegraph compact timings={liveProgress?.stage_timings_ms} activeJob={activeJob} />
          </div>

          {/* Queue depths, CUDA-stream latency, GPU/CPU/PCIe utilisation of the asynchronous pipeline */}
          <div className="hud-card hud-saturation-card" data-testid="hud-saturation">
            <PipelineSaturationPanel enabled={expanded} />
          </div>

          {/* Live 6DoF head pose and temporal stability (tracking.telemetry frames) */}
          <div className="hud-card hud-tracking-card" data-testid="hud-tracking">
            <TrackingTelemetryPanel />
          </div>

          {/* Real-Time Bottleneck Indicator */}
          <div className={`hud-card bottleneck-card ${bottleneck.severity}`}>
            <span className="hud-label">BOTTLENECK DIAGNOSTIC:</span>
            <div className="bottleneck-detail">
              <span className={`bottleneck-dot ${bottleneck.severity}`} />
              <strong>{bottleneck.label}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
