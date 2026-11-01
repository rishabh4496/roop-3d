import React from 'react';
import { THEMES, type ThemeDefinition } from '../themes';
import { normalizeRecipe, type ThemeRecipe } from '../themeVars';

export interface ThemeCardProps {
  theme: ThemeDefinition | ThemeRecipe;
  active: boolean;
  onChange: (name: string) => void;
  onEdit?: (theme: ThemeRecipe) => void;
}

export function ThemeCard({ theme, active, onChange, onEdit }: ThemeCardProps) {
  const t = theme;
  return (
    <div className="relative theme-card-wrapper">
      <button
        type="button"
        onClick={() => onChange(t.name)}
        title={t.label || t.name}
        aria-pressed={active}
        className={`theme-card tap group relative w-full text-left rounded-xl overflow-hidden border-2 focus:outline-none transition-all ${
          active ? 'active shadow-lg ring-2' : 'hover:border-white/20'
        }`}
        style={{
          borderColor: active ? t.accent : 'rgba(255,255,255,0.08)',
          boxShadow: active ? `0 0 0 1px ${t.accent}, 0 8px 24px ${t.accent}33` : undefined,
        }}
      >
        {/* Swatch preview */}
        <div className="h-12 w-full relative theme-swatch-box" style={{ background: t.bg }}>
          <div
            className="absolute inset-0 opacity-90"
            style={{ background: `radial-gradient(circle at 25% 20%, ${t.accent}44, transparent 60%)` }}
          />
          <div className="absolute bottom-1.5 left-2 flex items-center gap-1.5">
            <span className="h-4 w-4 rounded-full shadow inline-block" style={{ background: t.accent }} />
            <span className="h-2 w-8 rounded-full inline-block" style={{ background: `${t.accent}66` }} />
            <span className="h-2 w-4 rounded-full bg-white/20 inline-block" />
          </div>
          {active && (
            <span
              className="absolute top-1.5 right-1.5 h-5 w-5 rounded-full flex items-center justify-center text-xs font-black shadow"
              style={{ background: t.accent, color: t.mode === 'light' ? '#ffffff' : '#000000' }}
            >
              ✓
            </span>
          )}
        </div>
        {/* Label */}
        <div className="px-2.5 py-2 bg-black/40 backdrop-blur-sm">
          <div className="flex items-center justify-between gap-1">
            <span className="text-mini font-bold text-white/90 truncate">{t.name}</span>
            <span
              className={`text-nano font-semibold uppercase tracking-wider px-1 py-0.5 rounded shrink-0 ${
                t.mode === 'light' ? 'bg-amber-400/20 text-amber-300' : 'bg-white/10 text-muted'
              }`}
            >
              {t.mode}
            </span>
          </div>
          <div className="text-nano text-muted truncate">{t.custom ? 'Your theme' : t.label}</div>
        </div>
      </button>
      {t.custom && onEdit && (
        <button
          type="button"
          onClick={() => onEdit(t as ThemeRecipe)}
          title={`Edit ${t.name}`}
          aria-label={`Edit theme ${t.name}`}
          className="absolute top-1.5 left-1.5 h-6 w-6 grid place-items-center rounded-md bg-black/60 backdrop-blur text-white/70 hover:text-white hover:bg-black/80 border border-white/15 transition-colors"
        >
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
    <div className="theme-gallery-grid grid grid-cols-2 sm:grid-cols-3 gap-2.5 max-h-[480px] overflow-y-auto custom-scrollbar p-1">
      {/* Custom themes */}
      {customs.map((t) => (
        <ThemeCard
          key={`custom-${t.name}`}
          theme={t}
          active={t.name === current}
          onChange={onChange}
          onEdit={onEdit}
        />
      ))}

      {onCreate && (
        <button
          type="button"
          onClick={onCreate}
          className="tap rounded-xl border-2 border-dashed border-white/15 hover:border-[var(--accent)]/60 hover:bg-white/[0.03] transition-colors flex flex-col items-center justify-center gap-1 py-4 min-h-[86px] text-muted hover:text-white/85 focus:outline-none"
        >
          <span className="text-lg">＋</span>
          <span className="text-mini font-semibold">New theme</span>
        </button>
      )}

      {/* Preset themes */}
      {THEMES.map((t) => (
        <ThemeCard
          key={t.name}
          theme={t}
          active={t.name === current}
          onChange={onChange}
          onEdit={onEdit}
        />
      ))}
    </div>
  );
}
