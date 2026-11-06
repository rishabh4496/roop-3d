import React from 'react';
import { THEMES, type ThemeDefinition } from '../themes';
import { normalizeRecipe, type ThemeRecipe } from '../themeVars';

export interface ThemeCardProps {
  theme: ThemeDefinition | ThemeRecipe;
  active: boolean;
  onChange: (name: string) => void;
  onEdit?: (theme: ThemeRecipe) => void;
}

/** One swatch card. Colours come from the theme itself (inline), everything else from shell.css `.theme-card*`. */
export function ThemeCard({ theme, active, onChange, onEdit }: ThemeCardProps) {
  const t = theme;
  return (
    <div className="theme-card-wrapper">
      <button
        type="button"
        onClick={() => onChange(t.name)}
        title={t.label || t.name}
        aria-pressed={active}
        className={`theme-card ${active ? 'active' : ''}`}
        style={active ? { borderColor: t.accent, boxShadow: `0 0 0 1px ${t.accent}, 0 10px 28px ${t.accent}40` } : undefined}
      >
        <span className="theme-swatch-box" style={{ background: t.bg }}>
          <span className="theme-swatch-glow" style={{ background: `radial-gradient(circle at 25% 20%, ${t.accent}55, transparent 62%)` }} />
          <span className="theme-swatch-dots">
            <span className="theme-dot" style={{ background: t.accent }} />
            <span className="theme-bar" style={{ background: `${t.accent}66` }} />
            <span className="theme-bar short" />
          </span>
          {active && <span className="theme-check" style={{ background: t.accent, color: t.mode === 'light' ? '#ffffff' : '#000000' }}>✓</span>}
        </span>
        <span className="theme-label">
          <span className="theme-label-row">
            <span className="theme-name">{t.name}</span>
            <span className={`theme-mode ${t.mode}`}>{t.mode}</span>
          </span>
          <span className="theme-sub">{t.custom ? 'Your theme' : t.label}</span>
        </span>
      </button>
      {t.custom && onEdit && (
        <button type="button" onClick={() => onEdit(t as ThemeRecipe)} title={`Edit ${t.name}`} aria-label={`Edit theme ${t.name}`} className="theme-edit">
          ⚙
        </button>
      )}
    </div>
  );
}

export interface ThemeGalleryProps {
  value?: string | null;
  onChange: (name: string) => void;
  customThemes?: ThemeRecipe[];
  onEdit?: (theme: ThemeRecipe) => void;
  onCreate?: () => void;
}

export default function ThemeGallery({
  value,
  onChange,
  customThemes = [],
  onEdit,
  onCreate,
}: ThemeGalleryProps) {
  const current = value || 'Default';
  const customs = (Array.isArray(customThemes) ? customThemes : []).map(normalizeRecipe);

  return (
    <div className="theme-gallery-grid">
      {/* Custom themes */}
      {customs.map((t) => (
        <ThemeCard key={`custom-${t.name}`} theme={t} active={t.name === current} onChange={onChange} onEdit={onEdit} />
      ))}

      {onCreate && (
        <button type="button" onClick={onCreate} className="theme-new">
          <span className="theme-new-plus">＋</span>
          <span>New theme</span>
        </button>
      )}

      {/* Preset themes */}
      {THEMES.map((t) => (
        <ThemeCard key={t.name} theme={t} active={t.name === current} onChange={onChange} onEdit={onEdit} />
      ))}
    </div>
  );
}
