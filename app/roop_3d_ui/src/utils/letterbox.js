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
