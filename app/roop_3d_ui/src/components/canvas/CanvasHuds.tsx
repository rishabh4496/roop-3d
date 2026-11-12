import { useEffect, useRef } from 'react';
import { LiquidGlassCard } from '../ui-layouts/LiquidGlassCard';
import { insideStage, loupeCrop, type Pointer, type Stage } from './loupeCrop';
import { facePose } from './poseHud';

/** Floating Stage 24 HUDs for the full-bleed canvas, drawn on liquid glass. All of them are optional extras over the picture. */

interface PoseFace { id?: string; pose?: unknown; pose3d?: unknown }
const signed = (value: number) => `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(1)}°`;

export interface ViewportTelemetryPillProps {
  faces: PoseFace[];
  activeFaceId?: string | null;
  mode: string;
  frame: number;
  zoom?: number;
}

/** Live yaw / pitch / roll of the selected (else first) face at the playhead, plus the frame and canvas mode. Drag it out of the way. */
export function ViewportTelemetryPill({ faces, activeFaceId = null, mode, frame, zoom }: ViewportTelemetryPillProps) {
  const face = faces.find(item => item.id === activeFaceId) ?? faces[0];
  let pose: { yaw: number; pitch: number; roll: number } | null = null;
  try { pose = face ? (facePose(face) as { yaw: number; pitch: number; roll: number } | null) : null; } catch { pose = null; }
  const ok = pose && Number.isFinite(pose.yaw) && Number.isFinite(pose.pitch) && Number.isFinite(pose.roll);
  return (
    <LiquidGlassCard draggable className="canvas-hud canvas-hud--telemetry" borderRadius="999px" blurIntensity="md" glowIntensity="xs" shadowIntensity="sm"
      role="group" aria-label="3D viewport telemetry" data-testid="viewport-telemetry-pill">
      <span className="hud-cell"><i>YAW</i><b className="telemetry">{ok ? signed(pose!.yaw) : '—'}</b></span>
      <span className="hud-cell"><i>PITCH</i><b className="telemetry">{ok ? signed(pose!.pitch) : '—'}</b></span>
      <span className="hud-cell"><i>ROLL</i><b className="telemetry">{ok ? signed(pose!.roll) : '—'}</b></span>
      <span className="hud-cell hud-cell--meta"><i>{faces.length} {faces.length === 1 ? 'FACE' : 'FACES'}</i><b className="telemetry">F{frame}</b></span>
      <span className="hud-cell hud-cell--meta"><i>VIEW</i><b className="telemetry">{mode === 'loupe' && zoom ? `loupe ${zoom}x` : mode}</b></span>
    </LiquidGlassCard>
  );
}

export interface BeforeAfterLoupeHudProps {
  imageA: HTMLImageElement | null;
  imageB: HTMLImageElement | null;
  stage: Stage;
  /** Pointer in container pixels (the same space as `stage`). */
  pointer: Pointer & { active: boolean };
  zoom?: number;
  size?: number;
}

const LENS = 112;

/** Two small lenses (original / swapped) at the same spot under the cursor, so a pore-level change reads without switching modes. */
export function BeforeAfterLoupeHud({ imageA, imageB, stage, pointer, zoom = 4, size = LENS }: BeforeAfterLoupeHudProps) {
  const before = useRef<HTMLCanvasElement>(null), after = useRef<HTMLCanvasElement>(null);
  const last = useRef<Pointer>({ x: stage.left + stage.width / 2, y: stage.top + stage.height / 2 });
  if (pointer.active && insideStage(pointer, stage)) last.current = { x: pointer.x, y: pointer.y };
  const { x, y } = last.current;
  const { left, top, width, height } = stage;

  useEffect(() => {
    for (const [canvas, image] of [[before.current, imageA], [after.current, imageB ?? imageA]] as const) {
      const ctx = canvas?.getContext('2d');
      if (!canvas || !ctx) continue;
      canvas.width = size; canvas.height = size;
      ctx.clearRect(0, 0, size, size);
      if (!image) continue;
      const crop = loupeCrop({ x, y }, { left, top, width, height }, { width: image.naturalWidth, height: image.naturalHeight }, size, zoom);
      if (!crop) continue;
      ctx.imageSmoothingEnabled = false;                      // show real pixels, not an interpolated guess
      ctx.drawImage(image, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, size, size);
    }
  }, [imageA, imageB, left, top, width, height, x, y, zoom, size]);

  return (
    <LiquidGlassCard draggable className="canvas-hud canvas-hud--loupe" borderRadius="20px" blurIntensity="lg" glowIntensity="xs" shadowIntensity="sm"
      role="group" aria-label={`Before and after loupe at ${zoom}x`} data-testid="before-after-loupe">
      <figure><canvas ref={before} width={size} height={size} aria-label="Before (original)" role="img" /><figcaption>BEFORE</figcaption></figure>
      <figure><canvas ref={after} width={size} height={size} aria-label="After (swapped)" role="img" /><figcaption>AFTER · {zoom}x</figcaption></figure>
    </LiquidGlassCard>
  );
}
