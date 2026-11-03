import React, { useRef, useState, useCallback, useMemo } from 'react';
import type { WorkstationParameters } from '../workstation/pipeline';
import { kelvinToRgb } from '../../utils/kelvinToRgb';

interface LightProbeSphereProps {
  parameters?: WorkstationParameters;
  setParameters?: (params: WorkstationParameters) => void;
  detectedLightDirection?: [number, number, number];
  colorTemperatureK?: number;
}

export default function LightProbeSphere({
  parameters,
  setParameters,
  detectedLightDirection = [0.85, 0.0, 0.52],
  colorTemperatureK = 5500,
}: LightProbeSphereProps) {
  const sphereRef = useRef<SVGSVGElement | null>(null);
  const [isDragging, setIsDragging] = useState(false);

  const mode = parameters?.lighting_mode ?? 'auto';
  const relightStrength = parameters?.relight_strength ?? 0.8;
  const specularSheen = parameters?.specular_sheen ?? 0.25;
  const shadowPreservation = parameters?.shadow_preservation ?? 0.85;
  const manualYaw = parameters?.manual_light_yaw ?? 35.0;
  const manualPitch = parameters?.manual_light_pitch ?? 25.0;

  // Active light direction based on mode
  const currentYaw = mode === 'manual' ? manualYaw : (Math.atan2(detectedLightDirection[0], detectedLightDirection[2]) * 180) / Math.PI;
  const currentPitch = mode === 'manual' ? manualPitch : (Math.asin(Math.max(-1, Math.min(1, -detectedLightDirection[1]))) * 180) / Math.PI;

  const updateParam = useCallback((key: keyof WorkstationParameters, value: any) => {
    if (!setParameters) return;
    const current = parameters ?? {
      restoration_blend: 0, feather_px: 5, color_strength: 0.65, pose_strength: 0.35, bypass: []
    };
    setParameters({ ...current, [key]: value });
  }, [parameters, setParameters]);

  // Convert (yaw, pitch) in degrees to normalized 2D sphere circle coordinate [-1, 1]
  const lightMarkerPos = useMemo(() => {
    const yawRad = (currentYaw * Math.PI) / 180;
    const pitchRad = (currentPitch * Math.PI) / 180;
    // Standard orthographic sphere projection: x = sin(yaw)*cos(pitch), y = -sin(pitch)
    const px = Math.sin(yawRad) * Math.cos(pitchRad);
    const py = -Math.sin(pitchRad);
    return { x: px, y: py };
  }, [currentYaw, currentPitch]);

  const handlePointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    setIsDragging(true);
    (event.target as HTMLElement).setPointerCapture?.(event.pointerId);
    handlePointerMove(event);
  };

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!isDragging && event.buttons !== 1) return;
    if (!sphereRef.current) return;
    const rect = sphereRef.current.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const radius = rect.width / 2;

    const dx = (event.clientX - cx) / radius;
    const dy = (event.clientY - cy) / radius;
    const r = Math.sqrt(dx * dx + dy * dy);
    const clampedR = Math.min(0.95, r);
    const scale = r > 0.95 ? clampedR / r : 1.0;
    const nx = dx * scale;
    const ny = dy * scale;
    const nz = Math.sqrt(Math.max(0, 1.0 - nx * nx - ny * ny));

    // Convert to yaw and pitch
    const newYaw = (Math.atan2(nx, nz) * 180) / Math.PI;
    const newPitch = (Math.asin(-ny) * 180) / Math.PI;

    if (setParameters) {
      const current = parameters ?? {
        restoration_blend: 0, feather_px: 5, color_strength: 0.65, pose_strength: 0.35, bypass: []
      };
      setParameters({
        ...current,
        lighting_mode: 'manual',
        manual_light_yaw: Math.round(newYaw * 10) / 10,
        manual_light_pitch: Math.round(newPitch * 10) / 10,
      });
    }
  };

  const handlePointerUp = (event: React.PointerEvent<SVGSVGElement>) => {
    setIsDragging(false);
    try {
      (event.target as HTMLElement).releasePointerCapture?.(event.pointerId);
    } catch {
      // ignore
    }
  };

  // Color temperature indicator color (Tanner Helland algorithm)
  const tempColor = useMemo(() => {
    return kelvinToRgb(colorTemperatureK).hex;
  }, [colorTemperatureK]);

  return (
    <div className="light-probe-widget" role="region" aria-label="3D Light Probe & Neural Relighting">
      <div className="probe-header">
        <span className="probe-title">3D LIGHT PROBE & RELIGHTING</span>
        <span className="probe-temp-badge" style={{ borderColor: tempColor, color: tempColor }}>
          {Math.round(colorTemperatureK)} K
        </span>
      </div>

      <div className="probe-mode-tabs" role="tablist" aria-label="Lighting Mode">
        <button
          type="button"
          role="tab"
          className={`probe-mode-btn ${mode === 'auto' ? 'active' : ''}`}
          aria-label="Auto-Match Target Scene 3D Lighting"
          aria-selected={mode === 'auto'}
          onClick={() => updateParam('lighting_mode', 'auto')}
        >
          Auto-Match Target Scene 3D Lighting
        </button>
        <button
          type="button"
          role="tab"
          className={`probe-mode-btn ${mode === 'manual' ? 'active' : ''}`}
          aria-label="Manual 3D Virtual Key Light"
          aria-selected={mode === 'manual'}
          onClick={() => updateParam('lighting_mode', 'manual')}
        >
          Manual 3D Virtual Key Light
        </button>
      </div>

      <div className="probe-viewport-wrapper">
        <svg
          ref={sphereRef}
          className="probe-sphere-svg"
          viewBox="-100 -100 200 200"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          aria-label="Interactive 3D Light Direction Sphere"
        >
          <defs>
            <radialGradient id="probeGradient" cx="35%" cy="35%" r="65%">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity="0.8" />
              <stop offset="40%" stopColor="#1e293b" stopOpacity="0.9" />
              <stop offset="100%" stopColor="#0f172a" stopOpacity="1" />
            </radialGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Base sphere */}
          <circle cx="0" cy="0" r="85" fill="url(#probeGradient)" stroke="#334155" strokeWidth="2" />

          {/* Latitude & longitude guide rings */}
          <ellipse cx="0" cy="0" rx="85" ry="32" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
          <ellipse cx="0" cy="0" rx="32" ry="85" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
          <line x1="-85" y1="0" x2="85" y2="0" stroke="#334155" strokeWidth="1" opacity="0.5" />
          <line x1="0" y1="-85" x2="0" y2="85" stroke="#334155" strokeWidth="1" opacity="0.5" />

          {/* Light marker position */}
          <g transform={`translate(${lightMarkerPos.x * 85}, ${lightMarkerPos.y * 85})`} filter="url(#glow)">
            <circle cx="0" cy="0" r="9" fill={mode === 'manual' ? '#f59e0b' : '#38bdf8'} stroke="#ffffff" strokeWidth="2" />
            <circle cx="0" cy="0" r="16" fill="none" stroke={mode === 'manual' ? '#f59e0b' : '#38bdf8'} strokeWidth="1.5" opacity="0.5" />
          </g>
        </svg>

        <div className="probe-readout">
          <div className="probe-stat">
            <span className="probe-stat-label">Direction</span>
            <strong>
              {currentYaw >= 0 ? '+' : ''}{currentYaw.toFixed(1)}° Y ·{' '}
              {currentPitch >= 0 ? '+' : ''}{currentPitch.toFixed(1)}° P
            </strong>
          </div>
          <div className="probe-stat">
            <span className="probe-stat-label">Key Light Status</span>
            <strong>{mode === 'manual' ? 'Virtual Manual Key' : 'Scene Auto-Detected'}</strong>
          </div>
        </div>
      </div>

      <div className="probe-controls">
        <label className="probe-slider-label">
          <span>Relight Strength ({relightStrength.toFixed(2)})</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            aria-label="Relight Strength (0.0–1.0)"
            value={relightStrength}
            onChange={e => updateParam('relight_strength', parseFloat(e.target.value))}
          />
        </label>

        <label className="probe-slider-label">
          <span>Specular Skin Sheen ({specularSheen.toFixed(2)})</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            aria-label="Specular Skin Sheen (0.0–1.0)"
            value={specularSheen}
            onChange={e => updateParam('specular_sheen', parseFloat(e.target.value))}
          />
        </label>

        <label className="probe-slider-label">
          <span>Shadow Preservation ({shadowPreservation.toFixed(2)})</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            aria-label="Shadow Preservation (0.0–1.0)"
            value={shadowPreservation}
            onChange={e => updateParam('shadow_preservation', parseFloat(e.target.value))}
          />
        </label>
      </div>
    </div>
  );
}
