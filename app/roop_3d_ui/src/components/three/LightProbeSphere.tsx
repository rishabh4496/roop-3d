import React, { useId, useRef, useState, useCallback, useMemo } from 'react';
import type { WorkstationParameters } from '../workstation/pipeline';
import { kelvinToRgb } from '../../utils/kelvinToRgb';
import { DEFAULT_AMBIENT_FILL } from './viewerRigs';
import { PrecisionRangeSlider } from '../ui-layouts';

// Defaults shown when a render parameter is unset; also the double-click reset targets of the sliders below.
const DEFAULT_RELIGHT = 0.8, DEFAULT_SHEEN = 0.25, DEFAULT_SHADOW = 0.85;

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
  const relightStrength = parameters?.relight_strength ?? DEFAULT_RELIGHT;
  const specularSheen = parameters?.specular_sheen ?? DEFAULT_SHEEN;
  const shadowPreservation = parameters?.shadow_preservation ?? DEFAULT_SHADOW;
  const ambientFill = parameters?.ambient_fill ?? DEFAULT_AMBIENT_FILL;
  const gradientId = useId().replace(/[^a-zA-Z0-9_-]/g, '');
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
            {/* The sphere is shaded by the controls themselves: the highlight sits where the key light is, its strength follows Relight
                Strength, and Ambient Fill lifts the unlit side. */}
            <radialGradient id={`${gradientId}-sphere`} cx="50%" cy="50%" r="78%" fx={`${50 + lightMarkerPos.x * 38}%`} fy={`${50 + lightMarkerPos.y * 38}%`}>
              <stop offset="0%" stopColor={tempColor} stopOpacity={0.3 + 0.6 * relightStrength} />
              <stop offset="45%" stopColor={`rgb(${30 + Math.round(ambientFill * 70)}, ${41 + Math.round(ambientFill * 70)}, ${59 + Math.round(ambientFill * 70)})`} stopOpacity="0.95" />
              <stop offset="100%" stopColor={`rgb(${15 + Math.round(ambientFill * 55)}, ${23 + Math.round(ambientFill * 55)}, ${42 + Math.round(ambientFill * 55)})`} stopOpacity="1" />
            </radialGradient>
            <filter id={`${gradientId}-glow`}>
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Base sphere */}
          <circle cx="0" cy="0" r="85" fill={`url(#${gradientId}-sphere)`} stroke="#334155" strokeWidth="2" />

          {/* Latitude & longitude guide rings */}
          <ellipse cx="0" cy="0" rx="85" ry="32" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
          <ellipse cx="0" cy="0" rx="32" ry="85" fill="none" stroke="#475569" strokeWidth="1" strokeDasharray="3 3" opacity="0.6" />
          <line x1="-85" y1="0" x2="85" y2="0" stroke="#334155" strokeWidth="1" opacity="0.5" />
          <line x1="0" y1="-85" x2="0" y2="85" stroke="#334155" strokeWidth="1" opacity="0.5" />

          {/* Specular skin sheen: a small hot spot under the key light, as bright as the Specular Skin Sheen slider says. */}
          <ellipse cx={lightMarkerPos.x * 62} cy={lightMarkerPos.y * 62} rx={9 + 12 * specularSheen} ry={6 + 8 * specularSheen} fill="#ffffff"
            opacity={0.08 + 0.55 * specularSheen} data-testid="probe-sheen" pointerEvents="none" />

          {/* Light marker position */}
          <g transform={`translate(${lightMarkerPos.x * 85}, ${lightMarkerPos.y * 85})`} filter={`url(#${gradientId}-glow)`}>
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
        <PrecisionRangeSlider className="probe-slider-label" label="Relight Strength" min={0} max={1} step={0.01} value={relightStrength} defaultValue={DEFAULT_RELIGHT} onChange={v => updateParam('relight_strength', v)} />
        <PrecisionRangeSlider className="probe-slider-label" label="Specular Skin Sheen" min={0} max={1} step={0.01} value={specularSheen} defaultValue={DEFAULT_SHEEN} onChange={v => updateParam('specular_sheen', v)} />
        <PrecisionRangeSlider
          className="probe-slider-label" label="Ambient Fill" min={0} max={1} step={0.01} value={ambientFill} defaultValue={DEFAULT_AMBIENT_FILL} disabled={mode !== 'manual'}
          title={mode === 'manual' ? 'Ambient (SH band-0) fill of the virtual key light' : 'Applies to the manual virtual key light; the auto light is estimated from the target frame'}
          onChange={v => updateParam('ambient_fill', v)}
        />
        <PrecisionRangeSlider className="probe-slider-label" label="Shadow Preservation" min={0} max={1} step={0.01} value={shadowPreservation} defaultValue={DEFAULT_SHADOW} onChange={v => updateParam('shadow_preservation', v)} />
      </div>
    </div>
  );
}
