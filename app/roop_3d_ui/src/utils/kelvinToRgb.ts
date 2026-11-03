/**
 * Tanner Helland's algorithm to convert color temperature in Kelvin (1000K to 12000K)
 * to standard sRGB color values.
 */
export function kelvinToRgb(kelvin: number): { r: number; g: number; b: number; hex: string } {
  const temp = Math.max(1000, Math.min(12000, kelvin)) / 100;
  let r: number, g: number, b: number;

  // Calculate Red
  if (temp <= 66) {
    r = 255;
  } else {
    r = temp - 60;
    r = 329.698727446 * Math.pow(r, -0.1332047592);
    r = Math.min(255, Math.max(0, r));
  }

  // Calculate Green
  if (temp <= 66) {
    g = temp;
    g = 99.4708025861 * Math.log(g) - 161.1195681661;
    g = Math.min(255, Math.max(0, g));
  } else {
    g = temp - 60;
    g = 288.1221695283 * Math.pow(g, -0.0755148492);
    g = Math.min(255, Math.max(0, g));
  }

  // Calculate Blue
  if (temp >= 66) {
    b = 255;
  } else if (temp <= 19) {
    b = 0;
  } else {
    b = temp - 10;
    b = 138.5177312231 * Math.log(b) - 305.0447927307;
    b = Math.min(255, Math.max(0, b));
  }

  const ri = Math.round(r);
  const gi = Math.round(g);
  const bi = Math.round(b);
  const hex = `#${((1 << 24) + (ri << 16) + (gi << 8) + bi).toString(16).slice(1)}`;

  return { r: ri, g: gi, b: bi, hex };
}
