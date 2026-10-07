// Presentation helpers. No model logic lives here - every number shown is the
// one the API returned, only rounded for display.

export const pct = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : `${(v * 100).toFixed(digits)}%`

export const signed = (v, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v)
    ? '—'
    : `${v >= 0 ? '+' : ''}${v.toFixed(digits)}`

export const money = (v, digits = 2) =>
  v === null || v === undefined || Number.isNaN(v)
    ? '—'
    : v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const num = (v, digits = 0) =>
  v === null || v === undefined || Number.isNaN(v)
    ? '—'
    : v.toLocaleString('en-US', { minimumFractionDigits: digits, maximumFractionDigits: digits })

export const hhmm = (unixSeconds) =>
  new Date(unixSeconds * 1000).toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'UTC',
  })

export const hhmmss = (unixSeconds) =>
  new Date(unixSeconds * 1000).toLocaleTimeString('en-US', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
    hour12: false, timeZone: 'UTC',
  })

export const stamp = (iso) => (iso ? iso.replace('T', ' ').replace('+00:00', 'Z') : '—')

export const untilNext = (ms, now = Date.now()) => {
  const d = Math.max(0, ms - now)
  const m = Math.floor(d / 60000)
  const s = Math.floor((d % 60000) / 1000)
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// Colour tokens, matched to the CSS variables so charts and DOM agree.
export const C = {
  up: '#26a69a',
  down: '#ef5350',
  neutral: '#d9a441',
  active: '#f0a13a',
  quiet: '#5b7fb5',
  model: '#7c5cff',
  naive: '#6b7688',
  // The recorded projections. Deliberately the same violet as the live model
  // line - it is the same head's output, only recorded earlier - and the dash
  // pattern is what separates them, not the hue.
  history: 'rgba(160,140,255,0.85)',
  grid: '#262d3d',
  axis: '#6b7688',
  panel: '#131722',
}

export const dirColor = (d) =>
  d === 'UP' ? C.up : d === 'DOWN' ? C.down : C.neutral

export const actColor = (a) => (a === 'ACTIVE' ? C.active : C.quiet)

export const dirClass = (d) =>
  d === 'UP' ? 'up' : d === 'DOWN' ? 'down' : 'neutral'

export const actClass = (a) => (a === 'ACTIVE' ? 'active' : 'quiet')
