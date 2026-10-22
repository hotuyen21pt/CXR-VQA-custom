import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// `base: './'` makes the built bundle load from relative URLs, so `dist/` can
// be served straight out of `vindr_cxr_vqa/serve.py` -- or from a Kaggle
// tunnel that mounts it under a path prefix -- without a rebuild.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: { port: 5173, open: true },
  build: { outDir: 'dist', emptyOutDir: true },
})
