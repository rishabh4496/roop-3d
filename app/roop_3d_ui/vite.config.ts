import { defineConfig } from 'vite';
export default defineConfig({
  server: { host: '127.0.0.1', port: 5174, strictPort: true, proxy: { '/api': { target: 'http://127.0.0.1:8030', ws: true, changeOrigin: true } } },
  build: { sourcemap: false, chunkSizeWarningLimit: 1200, rollupOptions: { output: { manualChunks: (id: string) => (id.includes('/three') || id.includes('@react-three')) ? 'three' : undefined } } },
});
