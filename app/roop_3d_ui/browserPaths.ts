/** Keep browser profiles, downloads and test evidence inside this checkout; reject linked ancestors. */
import { lstatSync, mkdirSync } from 'node:fs';
import { dirname, isAbsolute, relative, resolve } from 'node:path';

export const appRoot = resolve(import.meta.dirname, '..');
export function ownedBrowserPath(value: string): string {
  const path = resolve(value), rel = relative(appRoot, path);
  if (isAbsolute(rel) || rel === '..' || rel.startsWith('..\\') || rel.startsWith('../')) throw new Error('Browser artifact escapes Roop 3D');
  for (let current = path; ; current = dirname(current)) {
    const info = (() => { try { return lstatSync(current); } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error;
      return null;
    } })();
    if (info) {
      if (info.isSymbolicLink() || info.isFile() && info.nlink !== 1) throw new Error('Linked browser artifact path refused');
    }
    if (dirname(current) === current) break;
  }
  return path;
}

export function localBrowserTemp(): void {
  const temporary = ownedBrowserPath(resolve(appRoot, 'roop_3d/.runtime/browser-tests-temp'));
  mkdirSync(temporary, { recursive: true });
  process.env.TEMP = temporary;
  process.env.TMP = temporary;
  process.env.TMPDIR = temporary;
}
