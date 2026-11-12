import type { LivePreviewResult } from './api';

/** Short-lived, CPU-only PNG cache. Never retains object URLs or decoded GPU textures. */
export class PreviewFrameCache {
  private entries = new Map<string, { result: LivePreviewResult; expires: number; bytes: number }>();
  private bytes = 0;

  constructor(private maxBytes = 24 * 1024 * 1024, private maxEntries = 12, private ttlMs = 30_000) {}

  get(key: string, now = Date.now()): LivePreviewResult | undefined {
    const entry = this.entries.get(key);
    if (!entry) return;
    this.entries.delete(key);
    if (entry.expires <= now) { this.bytes -= entry.bytes; return; }
    this.entries.set(key, entry);
    return { ...entry.result, cached: true };
  }

  put(key: string, result: LivePreviewResult, now = Date.now()): void {
    if (result.cancelled || !result.blob.size) return;
    const previous = this.entries.get(key);
    if (previous) { this.entries.delete(key); this.bytes -= previous.bytes; }
    const bytes = result.blob.size + key.length * 2;
    if (bytes > this.maxBytes) return;
    this.entries.set(key, { result, expires: now + this.ttlMs, bytes });
    this.bytes += bytes;
    while (this.bytes > this.maxBytes || this.entries.size > this.maxEntries) {
      const oldest = this.entries.keys().next().value!;
      this.bytes -= this.entries.get(oldest)!.bytes;
      this.entries.delete(oldest);
    }
  }

  clear(): void { this.entries.clear(); this.bytes = 0; }
}
