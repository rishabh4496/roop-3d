import React from 'react';

/**
 * Composition and Framing Guide Overlays.
 * Modes:
 * - 'thirds': Rule of thirds with golden section power points.
 * - 'crosshair': Precision center crosshair with concentric alignment circle.
 * - 'safe': SMPTE / EBU Action Safe (90%) and Title Safe (80%) margins.
 *
 * @param {{ mode?: 'off' | 'thirds' | 'crosshair' | 'safe' }} props
 */
export default function CompositionGridOverlay({ mode = 'off' }) {
  if (!mode || mode === 'off') return null;

  return (
    <div
      className={`composition-grid-overlay mode-${mode}`}
      aria-label={`Framing Grid Guide: ${mode}`}
    >
      <svg
        className="composition-grid-svg"
        width="100%"
        height="100%"
        xmlns="http://www.w3.org/2000/svg"
      >
        {mode === 'thirds' && (
          <g className="grid-thirds">
            {/* Vertical grid lines */}
            <line x1="33.333%" y1="0%" x2="33.333%" y2="100%" className="guide-line thirds-line" />
            <line x1="66.667%" y1="0%" x2="66.667%" y2="100%" className="guide-line thirds-line" />

            {/* Horizontal grid lines */}
            <line x1="0%" y1="33.333%" x2="100%" y2="33.333%" className="guide-line thirds-line" />
            <line x1="0%" y1="66.667%" x2="100%" y2="66.667%" className="guide-line thirds-line" />

            {/* 4 Golden ratio power points */}
            <circle cx="33.333%" cy="33.333%" r="4" className="guide-point" />
            <circle cx="66.667%" cy="33.333%" r="4" className="guide-point" />
            <circle cx="33.333%" cy="66.667%" r="4" className="guide-point" />
            <circle cx="66.667%" cy="66.667%" r="4" className="guide-point" />

            <text x="12" y="22" className="guide-label">RULE OF THIRDS</text>
          </g>
        )}

        {mode === 'crosshair' && (
          <g className="grid-crosshair">
            {/* Center axes */}
            <line x1="50%" y1="0%" x2="50%" y2="100%" className="guide-line crosshair-axis" />
            <line x1="0%" y1="50%" x2="100%" y2="50%" className="guide-line crosshair-axis" />

            {/* Center alignment ring and dot */}
            <circle cx="50%" cy="50%" r="28" className="crosshair-ring" />
            <circle cx="50%" cy="50%" r="56" className="crosshair-ring outer" />
            <circle cx="50%" cy="50%" r="2.5" className="crosshair-center-dot" />

            {/* Axis tick marks */}
            <line x1="49%" y1="25%" x2="51%" y2="25%" className="guide-tick" />
            <line x1="49%" y1="75%" x2="51%" y2="75%" className="guide-tick" />
            <line x1="25%" y1="49%" x2="25%" y2="51%" className="guide-tick" />
            <line x1="75%" y1="49%" x2="75%" y2="51%" className="guide-tick" />

            <text x="12" y="22" className="guide-label">CENTER CROSSHAIR · 3D ALIGNMENT</text>
          </g>
        )}

        {mode === 'safe' && (
          <g className="grid-safe">
            {/* 90% Action Safe box */}
            <rect x="5%" y="5%" width="90%" height="90%" className="safe-box action-safe" />
            <text x="5.5%" y="7.5%" className="guide-label safe-label action">90% ACTION SAFE</text>

            {/* 80% Title Safe box */}
            <rect x="10%" y="10%" width="80%" height="80%" className="safe-box title-safe" />
            <text x="10.5%" y="12.5%" className="guide-label safe-label title">80% TITLE SAFE</text>

            {/* Center cross marker */}
            <line x1="48%" y1="50%" x2="52%" y2="50%" className="guide-tick" />
            <line x1="50%" y1="48%" x2="50%" y2="52%" className="guide-tick" />
          </g>
        )}
      </svg>
    </div>
  );
}
