/**
 * Keeps text readable in every theme, including themes a user builds themselves.
 *
 * Themes define colours; they cannot all be audited by hand. After a theme is applied we measure the pairs the
 * interface really uses (muted/accent/danger text on the four surface tokens, button text on the accent fill) and
 * nudge only the failing foreground tokens toward the theme's own text colour (or, for button text, to the better of
 * the theme's dark/light extremes) until WCAG 2.x AA (4.5:1) is met. Passing themes are not touched.
 */
export interface RGB { r: number; g: number; b: number }
export interface RGBA extends RGB { a: number }

export const AA = 4.5;

export function parseColor(input: string): RGBA | null {
  const text = input.trim().toLowerCase();
  let match = text.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/);
  if (match) {
    const hex = match[1].length === 3 ? match[1].split('').map(c => c + c).join('') : match[1];
    return { r: parseInt(hex.slice(0, 2), 16), g: parseInt(hex.slice(2, 4), 16), b: parseInt(hex.slice(4, 6), 16), a: 1 };
  }
  match = text.match(/^rgba?\(\s*([\d.]+)[ ,]+([\d.]+)[ ,]+([\d.]+)(?:\s*[,/]\s*([\d.]+%?))?\s*\)$/);
  if (match) return { r: +match[1], g: +match[2], b: +match[3], a: match[4] === undefined ? 1 : match[4].endsWith('%') ? parseFloat(match[4]) / 100 : +match[4] };
  match = text.match(/^color\(srgb\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)(?:\s*\/\s*([\d.]+%?))?\s*\)$/);
  if (match) return { r: +match[1] * 255, g: +match[2] * 255, b: +match[3] * 255, a: match[4] === undefined ? 1 : match[4].endsWith('%') ? parseFloat(match[4]) / 100 : +match[4] };
  return null;
}

const channel = (value: number) => { const s = value / 255; return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4; };
export const luminance = (c: RGB) => 0.2126 * channel(c.r) + 0.7152 * channel(c.g) + 0.0722 * channel(c.b);
export function contrast(a: RGB, b: RGB): number { const x = luminance(a), y = luminance(b); return (Math.max(x, y) + 0.05) / (Math.min(x, y) + 0.05); }
/** Colour `over` painted on opaque `under`. */
export const flatten = (over: RGBA, under: RGB): RGB => ({ r: over.r * over.a + under.r * (1 - over.a), g: over.g * over.a + under.g * (1 - over.a), b: over.b * over.a + under.b * (1 - over.a) });
export const mix = (a: RGB, b: RGB, t: number): RGB => ({ r: a.r + (b.r - a.r) * t, g: a.g + (b.g - a.g) * t, b: a.b + (b.b - a.b) * t });
export const toCss = (c: RGB) => `rgb(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)})`;
const worst = (fg: RGB, backgrounds: RGB[]) => Math.min(...backgrounds.map(bg => contrast(fg, bg)));

/** `fg`, moved toward `toward` in 4% steps until it reaches `min` against every background; `fg` itself when it already does. */
export function readable(fg: RGB, backgrounds: RGB[], toward: RGB, min = AA): RGB {
  if (worst(fg, backgrounds) >= min) return fg;
  for (let step = 1; step <= 25; step++) {
    const candidate = mix(fg, toward, step * 0.04);
    if (worst(candidate, backgrounds) >= min) return candidate;
  }
  return toward;
}

/** The text colour for a button painted with `fill`: whichever of `dark` / `light` reads better on it. */
export function onFill(fill: RGB, dark: RGB, light: RGB): RGB { return contrast(dark, fill) >= contrast(light, fill) ? dark : light; }

export interface ThemeColors { bg: RGB; surface: RGB; raised: RGB; soft: RGB; text: RGB; muted: RGB; accent: RGB; ink: RGB; danger: RGB }

/** Pure core of the enforcement, separated from the DOM so it can be tested with plain colours. */
export function enforce(colors: ThemeColors): Partial<Record<'--muted' | '--accent' | '--ink' | '--danger', string>> {
  const out: Partial<Record<'--muted' | '--accent' | '--ink' | '--danger', string>> = {};
  const grounds = [colors.bg, colors.surface, colors.raised, colors.soft];
  const dark = luminance(colors.text) < luminance(colors.bg) ? colors.text : colors.bg, light = dark === colors.text ? colors.bg : colors.text;
  const muted = readable(colors.muted, grounds, colors.text);
  if (muted !== colors.muted) out['--muted'] = toCss(muted);
  const accent = readable(colors.accent, grounds, colors.text);
  if (accent !== colors.accent) out['--accent'] = toCss(accent);
  const danger = readable(colors.danger, grounds, colors.text);
  if (danger !== colors.danger) out['--danger'] = toCss(danger);
  const fill = accent;
  if (contrast(colors.ink, fill) < AA) out['--ink'] = toCss(onFill(fill, dark, light));
  return out;
}

const TOKENS = ['--muted', '--accent', '--ink', '--danger'] as const;

/** Resolve a CSS custom property (or any colour expression) to a colour, through a probe element so `var()` and `color-mix()` work. */
function resolve(root: HTMLElement, token: string, property: 'color' | 'background-color' = 'color'): RGBA | null {
  const probe = document.createElement('i');
  probe.style.cssText = `position:fixed;left:-99px;${property}:var(${token})`;
  root.append(probe);
  const value = getComputedStyle(probe).getPropertyValue(property);
  probe.remove();
  return parseColor(value);
}

/**
 * Measure the applied theme and override the failing tokens inline on `root`. Returns what was overridden.
 * `root` is `<body>` on purpose: named themes declare their tokens on `body.theme-*` as well as on `<html>`, and a
 * declaration on the body itself beats anything inherited from `<html>`.
 */
export function enforceReadableTokens(root: HTMLElement = document.body): Record<string, string> {
  for (const token of TOKENS) root.style.removeProperty(token);   // measure the theme itself, not our previous correction
  const base = resolve(root, '--bg', 'background-color');
  if (!base) return {};
  const opaque = flatten(base, { r: 0, g: 0, b: 0 });
  const surface = (token: string) => { const c = resolve(root, token, 'background-color'); return c ? flatten(c, opaque) : null; };
  const text = resolve(root, '--text'), muted = resolve(root, '--muted'), accent = resolve(root, '--accent'), ink = resolve(root, '--ink'), danger = resolve(root, '--danger');
  const surfaceC = surface('--surface'), raised = surface('--raised'), soft = surface('--soft');
  if (!text || !muted || !accent || !ink || !danger || !surfaceC || !raised || !soft) return {};
  const overrides = enforce({ bg: opaque, surface: surfaceC, raised, soft, text: flatten(text, opaque), muted: flatten(muted, opaque), accent: flatten(accent, opaque), ink: flatten(ink, opaque), danger: flatten(danger, opaque) });
  for (const [token, value] of Object.entries(overrides)) root.style.setProperty(token, value);
  return overrides;
}
