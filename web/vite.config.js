import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The Python API (src/server.py) owns the model and Binance access; this app
// only renders what it returns. In dev, Vite serves the UI on :5173 and
// proxies /api to uvicorn on :8000, so there is no CORS dance and no second
// copy of the prediction logic in JavaScript.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // Server-Sent Events must not be buffered by the proxy, or the
        // "realtime" push arrives in clumps.
        configure: (proxy) => {
          proxy.on('proxyRes', (proxyRes) => {
            if (proxyRes.headers['content-type']?.includes('text/event-stream')) {
              proxyRes.headers['cache-control'] = 'no-cache'
            }
          })
        },
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1200,
  },
})
