/** Pure geometry for the before / after loupe HUD. */
export interface Stage { left: number; top: number; width: number; height: number }
export interface Pointer { x: number; y: number }
export interface CropRect { sx: number; sy: number; sw: number; sh: number }

/**
 * The part of an image (natural size `natural`) that a `size`-px lens shows at `zoom`x around `pointer`, where `pointer` is in the
 * same container pixels as `stage` (the on-screen rectangle the image is drawn into). The rect is clamped inside the image so
 * the lens never samples outside it; null when the stage has no area yet.
 */
export function loupeCrop(pointer: Pointer, stage: Stage, natural: { width: number; height: number }, size: number, zoom: number): CropRect | null {
  if (!(stage.width > 0) || !(stage.height > 0) || !(natural.width > 0) || !(natural.height > 0) || !(zoom > 0) || !(size > 0)) return null;
  const scaleX = natural.width / stage.width, scaleY = natural.height / stage.height;
  const sw = Math.min(natural.width, (size / zoom) * scaleX), sh = Math.min(natural.height, (size / zoom) * scaleY);
  const cx = (pointer.x - stage.left) * scaleX, cy = (pointer.y - stage.top) * scaleY;
  const sx = Math.min(Math.max(0, cx - sw / 2), natural.width - sw), sy = Math.min(Math.max(0, cy - sh / 2), natural.height - sh);
  return { sx, sy, sw, sh };
}

/** Is the pointer over the picture (not the letterbox bars around it)? */
export const insideStage = (pointer: Pointer, stage: Stage) =>
  pointer.x >= stage.left && pointer.x <= stage.left + stage.width && pointer.y >= stage.top && pointer.y <= stage.top + stage.height;
