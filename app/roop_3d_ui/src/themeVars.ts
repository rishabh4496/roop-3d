// ── Custom theme derivation matching roop-ultimate ─────────────────────────────
export function hexToRgb(hex: string): [number, number, number] | null {
  if (typeof hex !== 'string') return null;
  let h = hex.trim().replace(/^#/, '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return null;
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

const clamp255 = (n: number) => Math.max(0, Math.min(255, Math.round(n)));

export function rgbToHex([r, g, b]: [number, number, number]): string {
  return `#${[r, g, b].map((c) => clamp255(c).toString(16).padStart(2, '0')).join('')}`;
}

export function mix(a: string, b: string, t: number): string {
  const A = hexToRgb(a) || [0, 0, 0];
  const B = hexToRgb(b) || [0, 0, 0];
  return rgbToHex([
    A[0] + (B[0] - A[0]) * t,
    A[1] + (B[1] - A[1]) * t,
    A[2] + (B[2] - A[2]) * t,
  ]);
}

export const lighten = (hex: string, t: number) => mix(hex, '#ffffff', t);
export const darken = (hex: string, t: number) => mix(hex, '#000000', t);

export function alpha(hex: string, a: number): string {
  const c = hexToRgb(hex) || [0, 0, 0];
  const av = Math.round(Math.max(0, Math.min(1, a)) * 1000) / 1000;
  return `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${av})`;
}

export function luminance(hex: string): number {
  const c = hexToRgb(hex) || [0, 0, 0];
  const f = c.map((v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * f[0] + 0.7152 * f[1] + 0.0722 * f[2];
}

export function contrastRatio(a: string, b: string): number {
  const la = luminance(a);
  const lb = luminance(b);
  const hi = Math.max(la, lb);
  const lo = Math.min(la, lb);
  return (hi + 0.05) / (lo + 0.05);
}

export interface ThemeRecipe {
  name: string;
  custom?: boolean;
  mode: 'dark' | 'light';
  accent: string;
  bg: string;
  surface?: number;
  border?: number;
  radius?: number;
  label?: string;
  className?: string;
}

export const DEFAULT_RECIPE: ThemeRecipe = {
  name: 'My Theme',
  custom: true,
  mode: 'dark',
  accent: '#E94560',
  bg: '#0B0C12',
  surface: 0.55,
  border: 0.07,
  radius: 16,
};

export const normalizeRecipe = (t: Partial<ThemeRecipe> | null | undefined): ThemeRecipe => ({
  ...DEFAULT_RECIPE,
  ...(t || {}),
  custom: true,
});

export const STATUS_COLORS = {
  dark: { '--ok': '#34D399', '--warn': '#FBBF24', '--danger': '#F87171', '--info': '#60A5FA' },
  light: { '--ok': '#047857', '--warn': '#B45309', '--danger': '#B91C1C', '--info': '#1D4ED8' },
};

export function deriveThemeVars(recipe: Partial<ThemeRecipe>): Record<string, string> {
  const t = normalizeRecipe(recipe);
  const light = t.mode === 'light';
  const accent = hexToRgb(t.accent) ? t.accent : DEFAULT_RECIPE.accent;
  const bg = hexToRgb(t.bg) ? t.bg : (light ? '#e5e9f0' : DEFAULT_RECIPE.bg);

  const ink = light ? darken(bg, 0.86) : lighten(bg, 0.94);
  const inkHex = ink;
  const cardTint = light ? lighten(bg, 0.75) : lighten(bg, 0.06);
  const inputTint = light ? lighten(bg, 0.55) : darken(bg, 0.45);

  const surface = Number(t.surface) || DEFAULT_RECIPE.surface || 0.55;
  const border = Number(t.border) || DEFAULT_RECIPE.border || 0.07;
  const radius = Number.isFinite(Number(t.radius)) ? Number(t.radius) : DEFAULT_RECIPE.radius || 16;
  const accentHover = light ? darken(accent, 0.12) : lighten(accent, 0.14);

  const surfaceGlass = light
    ? 'rgba(255, 255, 255, 0.75)'
    : alpha(cardTint, 0.72);
  const surfaceGlassRaised = light
    ? 'rgba(245, 247, 248, 0.85)'
    : alpha(lighten(cardTint, 0.08), 0.78);
  const glassBorder = light
    ? '1px solid rgba(0, 0, 0, 0.08)'
    : `1px solid ${alpha(inkHex, 0.08)}`;
  const glassSpecular = light
    ? 'inset 0 1px 0 rgba(255, 255, 255, 0.6)'
    : `inset 0 1px 0 ${alpha(lighten(bg, 1), 0.06)}`;
  const glassGlow = light
    ? '0 8px 32px 0 rgba(0, 0, 0, 0.08)'
    : '0 8px 32px 0 rgba(0, 0, 0, 0.37)';

  return {
    '--bg-gradient': [
      `radial-gradient(at 18% -12%, ${alpha(accent, light ? 0.1 : 0.06)} 0px, transparent 42%)`,
      `radial-gradient(at 100% 108%, ${alpha(accentHover, light ? 0.08 : 0.04)} 0px, transparent 45%)`,
      `linear-gradient(162deg, ${light ? lighten(bg, 0.5) : darken(bg, 0.35)}, ${bg} 55%, ${light ? bg : darken(bg, 0.2)})`,
    ].join(', '),
    '--card-bg': alpha(cardTint, surface),
    '--surface-2': alpha(inkHex, light ? 0.05 : 0.035),
    '--surface-glass': surfaceGlass,
    '--surface-glass-raised': surfaceGlassRaised,
    '--glass-bg': surfaceGlass,
    '--glass-border': glassBorder,
    '--glass-specular': glassSpecular,
    '--glass-highlight': light ? 'rgba(255, 255, 255, 0.65)' : alpha(lighten(bg, 1), 0.06),
    '--glass-glow': glassGlow,
    '--glass-shadow': glassGlow,
    '--glass-blur': 'blur(12px) saturate(180%)',
    '--text-main': ink,
    '--text-muted': alpha(inkHex, 0.55),
    '--accent': accent,
    '--accent-hover': accentHover,
    '--accent-glow': alpha(accent, light ? 0.18 : 0.28),
    '--border-color': alpha(inkHex, border),
    '--border-strong': alpha(inkHex, Math.min(0.4, Math.max(border * 1.9, border + 0.05))),
    '--input-bg': alpha(inputTint, light ? 0.5 : 0.55),
    '--input-focus': alpha(accent, light ? 0.12 : 0.16),
    '--scrollbar-thumb': alpha(inkHex, light ? 0.14 : 0.13),
    '--scrollbar-thumb-hover': alpha(inkHex, light ? 0.3 : 0.3),
    '--shadow-card': light
      ? '0 1px 2px rgba(0, 0, 0, 0.06), 0 10px 30px rgba(0, 0, 0, 0.07)'
      : '0 1px 2px rgba(0, 0, 0, 0.28), 0 10px 30px rgba(0, 0, 0, 0.22)',
    '--radius-lg': `${radius}px`,
    '--specular': light ? 'rgba(0, 0, 0, 0.035)' : alpha(lighten(bg, 1), 0.05),
    ...STATUS_COLORS[light ? 'light' : 'dark'],
    ...(light ? { '--input-bg-focus': alpha(lighten(bg, 0.85), 0.92) } : {}),
  };
}
