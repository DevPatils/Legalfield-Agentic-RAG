import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The SSE endpoint streams; proxying keeps the browser on one origin so
    // EventSource works without CORS preflight surprises.
    proxy: { '/api': { target: 'http://localhost:8000', changeOrigin: true } },
  },
})
