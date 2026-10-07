/**
 * `npm run check` — bundle web/ssr-check.jsx and run it.
 *
 * The bundling lives here rather than in a package.json script because the
 * command needs a `--banner` shim (react-dom/server is CJS and does a dynamic
 * `require`), and quoting that through npm on Windows is a portability trap.
 * esbuild's JS API takes it as a plain string.
 *
 * The bundle is written next to this file and left on disk so a failure can be
 * inspected; it is a build artefact, not source.
 */
import { build } from 'esbuild'
import { fileURLToPath, pathToFileURL } from 'node:url'

const entry = fileURLToPath(new URL('./ssr-check.jsx', import.meta.url))
const outfile = fileURLToPath(new URL('./_ssr-check.mjs', import.meta.url))

await build({
  entryPoints: [entry],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile,
  loader: { '.jsx': 'jsx' },
  jsx: 'automatic',
  define: { 'process.env.NODE_ENV': '"production"' },
  // react-dom/server requires 'stream' at runtime; ESM output cannot express
  // that without a real `require` in scope.
  banner: {
    js: 'import { createRequire } from "module"; const require = createRequire(import.meta.url);',
  },
  logLevel: 'warning',
})

await import(pathToFileURL(outfile).href)
