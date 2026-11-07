/** Sub-pixel letterbox geometry shared by the face overlay, the comparison canvas and hit-testing. */
/**
 * Calculates letterbox / contain coordinates for sub-pixel accuracy.
 */
export function calculateLetterbox(containerW, containerH, mediaW, mediaH) {
  if (!containerW || !containerH || !mediaW || !mediaH) {
    return { displayW: containerW || 0, displayH: containerH || 0, offsetX: 0, offsetY: 0 };
  }
  const mediaAspect = mediaW / mediaH;
  const containerAspect = containerW / containerH;

  let displayW, displayH, offsetX, offsetY;
  if (containerAspect > mediaAspect) {
    displayH = containerH;
    displayW = containerH * mediaAspect;
    offsetX = (containerW - displayW) / 2;
    offsetY = 0;
  } else {
    displayW = containerW;
    displayH = containerW / mediaAspect;
    offsetX = 0;
    offsetY = (containerH - displayH) / 2;
  }

  return { displayW, displayH, offsetX, offsetY };
}

/**
 * Sub-pixel letterbox coordinate mapping for normalized bounding box [x1, y1, x2, y2].
 */
export function mapBoxToPixels(bbox, letterbox) {
  const [x1, y1, x2, y2] = bbox;
  const { displayW, displayH, offsetX, offsetY } = letterbox;
  return {
    x: offsetX + x1 * displayW,
    y: offsetY + y1 * displayH,
    w: Math.max(2, (x2 - x1) * displayW),
    h: Math.max(2, (y2 - y1) * displayH),
  };
}

/**
 * Maps a normalized point [px, py] to visual pixels.
 */
export function mapPointToPixels(point, letterbox) {
  const [px, py] = point;
  const { displayW, displayH, offsetX, offsetY } = letterbox;
  return {
    x: offsetX + px * displayW,
    y: offsetY + py * displayH,
  };
}

/**
 * Exact mapping between the native video frame and the on-screen stage, for an element drawn with `object-fit: contain`
 * inside a container that is then scaled by `zoom` about its centre and translated by `pan` (CSS
 * `translate(pan) scale(zoom)`, `transform-origin: center`, which is what the preview stage applies).
 *
 * `view = { containerW, containerH, mediaW, mediaH, zoom = 1, pan = { x: 0, y: 0 } }`; all lengths in CSS pixels of the
 * UNTRANSFORMED container. Points are in container-relative pixels (the stage's own box), NOT page coordinates: subtract the
 * stage's `getBoundingClientRect()` origin first when starting from a mouse event on a transformed element.
 * Video coordinates are native frame pixels with sub-pixel precision (no rounding anywhere).
 */
export function videoToCanvasCoords(point, view) {
  const { containerW, containerH, mediaW, mediaH, zoom = 1, pan = { x: 0, y: 0 } } = view;
  const { displayW, displayH, offsetX, offsetY } = calculateLetterbox(containerW, containerH, mediaW, mediaH);
  const [vx, vy] = point;
  const lx = offsetX + (vx / mediaW) * displayW, ly = offsetY + (vy / mediaH) * displayH;     // untransformed container space
  const cx = containerW / 2, cy = containerH / 2;
  return { x: cx + (lx - cx) * zoom + pan.x, y: cy + (ly - cy) * zoom + pan.y };
}

/** Inverse of `videoToCanvasCoords`. `inside` is false for a point on the letterbox bars (the coordinates are still returned). */
export function canvasToVideoCoords(point, view) {
  const { containerW, containerH, mediaW, mediaH, zoom = 1, pan = { x: 0, y: 0 } } = view;
  const { displayW, displayH, offsetX, offsetY } = calculateLetterbox(containerW, containerH, mediaW, mediaH);
  const [sx, sy] = point;
  const cx = containerW / 2, cy = containerH / 2;
  const lx = cx + (sx - pan.x - cx) / zoom, ly = cy + (sy - pan.y - cy) / zoom;
  const x = ((lx - offsetX) / displayW) * mediaW, y = ((ly - offsetY) / displayH) * mediaH;
  return { x, y, inside: x >= 0 && y >= 0 && x <= mediaW && y <= mediaH };
}

/** A native-pixel box [x1, y1, x2, y2] on screen. */
export function videoBoxToCanvas(box, view) {
  const a = videoToCanvasCoords([box[0], box[1]], view), b = videoToCanvasCoords([box[2], box[3]], view);
  return { x: a.x, y: a.y, w: b.x - a.x, h: b.y - a.y };
}
