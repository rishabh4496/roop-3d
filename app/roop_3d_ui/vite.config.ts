import { defineConfig } from 'vite';
export default defineConfig({
  server: { host: '127.0.0.1', port: 5174, strictPort: true, proxy: { '/api': { target: 'http://127.0.0.1:8030', ws: true, changeOrigin: true } } },
  build: { sourcemap: false, chunkSizeWarningLimit: 1200, rollupOptions: {
    // framer-motion ships `"use client"` (a React Server Components hint). This is a client-only SPA, so the directive is a
    // no-op and the bundler's "may not be preserved" notice is noise. Every other warning still surfaces.
    onwarn(warning, warn) { if (warning.code === 'MODULE_LEVEL_DIRECTIVE') return; warn(warning); },
    output: { manualChunks: (id: string) => (id.includes('/three') || id.includes('@react-three')) ? 'three' : undefined } } },
});
