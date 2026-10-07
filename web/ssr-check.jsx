/**
 * Render every panel against a REAL API snapshot, in Node, with no browser.
 *
 *   node ssr-check.mjs
 *
 * Why: the components are hand-written against a JSON contract, and the
 * failure mode of a mismatch (an undefined property, an off-by-one in a
 * horizon lookup, a `.map` over something that can be null) is a blank panel
 * that only shows up when a human opens the page. renderToString executes all
 * of the render logic - the useMemo bodies, the early returns, the table
 * builders - against the payload the server actually produced, so a crash
 * surfaces here instead.
 *
 * It does NOT run effects, so the two lightweight-charts panels only render
 * their container here; the chart wiring is checked by the browser itself.
 * Equally, ResponsiveContainer renders no children during SSR, so recharts
 * internals are not exercised - but every component body that decides WHAT to
 * chart is.
 */
import React from 'react'
import { renderToString } from 'react-dom/server'
import { readFileSync } from 'node:fs'

import App from './src/App.jsx'
import PriceChart from './src/components/PriceChart.jsx'
import VolumeChart from './src/components/VolumeChart.jsx'
import ReturnDist from './src/components/ReturnDist.jsx'
import SimpleView from './src/components/SimpleView.jsx'
import TrackRecord from './src/components/TrackRecord.jsx'
import MetricsTable from './src/components/MetricsTable.jsx'
import CallsTable from './src/components/CallsTable.jsx'
import { ActivityBars, DirectionBars } from './src/components/ProbabilityBars.jsx'

const snapshot = JSON.parse(readFileSync(new URL('./_snapshot.json', import.meta.url)))

// The fixture is a CAPTURE, and a stale one silently turns this whole suite
// into a check of the payload of a month ago - which is how `prediction_history`
// came to be absent here while the components were being written for it, and
// the history cases passed only against the synthetic fixture below. Re-capture
// with: python src/server.py --port 8099, then GET /api/snapshot into
// web/_snapshot.json. Not a hard failure: a fixture older than the last time
// the machine was on is a normal way to run a test suite, and failing the build
// for it would just teach everyone to ignore this output. Loud, though.
const ageH = snapshot.generated_at
  ? (Date.now() - Date.parse(snapshot.generated_at)) / 36e5
  : null
if (ageH !== null && ageH > 6) {
  console.log(`  note  the snapshot fixture is ${ageH.toFixed(1)}h old ` +
              `(anchor ${new Date(snapshot.anchor.timestamp_ms).toISOString()}) ` +
              `- re-capture it if a payload field has been added since\n`)
}

const cases = [
  ['PriceChart', <PriceChart snapshot={snapshot} />],
  ['PriceChart(simple)', <PriceChart snapshot={snapshot} simple />],
  ['SimpleView', <SimpleView data={snapshot} />],
  ['VolumeChart', <VolumeChart candles={snapshot.candles} />],
  ['DirectionBars', <DirectionBars horizons={snapshot.prediction.horizons} />],
  ['ActivityBars', <ActivityBars horizons={snapshot.prediction.horizons}
     tau={snapshot.regime.tau} reference={snapshot.regime.reference_absmove}
     refWindow={snapshot.regime.reference_window_candles} />],
  ['TrackRecord(regime)', <TrackRecord snapshot={snapshot} head="regime" />],
  ['TrackRecord(direction)', <TrackRecord snapshot={snapshot} head="direction" />],
  ['ReturnDist', <ReturnDist snapshot={snapshot} />],
  ['MetricsTable', <MetricsTable snapshot={snapshot} />],
  ['CallsTable', <CallsTable snapshot={snapshot} />],
]

// Degenerate payloads: the panels must degrade, not throw.
const emptyTrack = { ...snapshot, track_record: { recent: [], n_calls: 0, n_resolved: 0 } }
const noBands = { ...snapshot, bands: null }
const noMetrics = { ...snapshot, metrics: null }
const errPayload = { ok: false, error: 'boom', symbol: 'BTCUSDT', interval: '15m' }
const noActivity = {
  ...snapshot,
  prediction: {
    ...snapshot.prediction,
    horizons: snapshot.prediction.horizons.map((h) => ({ ...h, activity: null, activity_probs: null })),
  },
}

// The recorded-projection block: the empty form is what the panel shows when
// nothing has been stored yet, and the gapped form is the one that matters -
// a whitespace point has a time and no value, so a component that assumes
// every point has one throws here rather than on a chart with an outage in it.
const noHistory = { ...snapshot, prediction_history: null }
const gappedHistory = {
  ...snapshot,
  prediction_history: {
    ...(snapshot.prediction_history || {}),
    points: [
      { time: 1790600000, value: 109000.5, n: 3, actual: 109050.0 },
      { time: 1790600900, value: 109010.25, n: 2, actual: null },
      { time: 1790601800 },                       // the gap: no value
      { time: 1790700000, value: 108700.0, n: 1, actual: 108650.0 },
    ],
    n_points: 4, n_windows: 1, n_scored: 2, gap_candles: 4,
    note: 'test fixture',
  },
}

const degenerate = [
  ['PriceChart(no history)', <PriceChart snapshot={noHistory} />],
  ['PriceChart(gapped history)', <PriceChart snapshot={gappedHistory} />],
  // The simple view is the default, so it is the one a first-time reader gets
  // against a half-built or failed payload. It must degrade like the rest.
  ['SimpleView(no calls)', <SimpleView data={emptyTrack} />],
  ['SimpleView(no activity)', <SimpleView data={noActivity} />],
  ['SimpleView(err)', <SimpleView data={errPayload} />],
  ['CallsTable(empty)', <CallsTable snapshot={emptyTrack} />],
  ['TrackRecord(empty)', <TrackRecord snapshot={emptyTrack} head="regime" />],
  ['ReturnDist(no bands)', <ReturnDist snapshot={noBands} />],
  ['MetricsTable(none)', <MetricsTable snapshot={noMetrics} />],
  ['ActivityBars(none)', <ActivityBars horizons={noActivity.prediction.horizons} />],
  ['App(err)', <App initial={errPayload} />],
]

// The two lightweight-charts panels build their chart in an effect, and
// renderToString does not run effects - so all they emit is the container div
// they hand to createChart. That is the correct SSR output, not a failure, so
// they are held to a lower bar than the panels that render real content.
const CONTAINER_ONLY = new Set(['PriceChart', 'PriceChart(simple)', 'VolumeChart'])

let failed = 0
const run = (label, el) => {
  const floor = CONTAINER_ONLY.has(label) ? 20 : 50
  try {
    const html = renderToString(el)
    const n = html.length
    if (n < floor) throw new Error(`rendered only ${n} chars`)
    const note = CONTAINER_ONLY.has(label) ? '  (container only)' : ''
    console.log(`  ok    ${label.padEnd(24)} ${String(n).padStart(6)} chars${note}`)
  } catch (e) {
    failed += 1
    console.log(`  FAIL  ${label.padEnd(24)} ${e.message}`)
  }
}

console.log('real snapshot:')
cases.forEach(([l, el]) => run(l, el))
console.log('\ndegenerate payloads (must degrade, not throw):')
degenerate.forEach(([l, el]) => run(l, el))

// Which dashboard opens first is a product decision, not an implementation
// detail, so it is asserted instead of assumed. Node has no `window`, so this
// exercises the same path a first-time visitor with no stored preference takes.
console.log('\ndefault view (must be the simple one):')
const appHtml = renderToString(<App initial={snapshot} />)
const viewChecks = [
  ['opens on the simple view', appHtml.includes('Will the price move more than usual?')],
  ['the advanced panels are not mounted', !appHtml.includes('held-out performance')],
  ['the toggle is offered', appHtml.includes('Advanced view')],
  // The chart itself is container-only under SSR, but its legend is real DOM,
  // so the legend is what proves which layers the default view claims to draw.
  // The recorded history is asserted by name because it was briefly dropped
  // from the simple view and had to be put back: a reader who is told the model
  // is guessing should be able to see how its earlier guesses turned out.
  ['the price chart is present', appHtml.includes('what the model expects next')],
  ['the recorded history curve is claimed', appHtml.includes('what the model predicted earlier')],
  ['the advanced-only layers are not claimed', !appHtml.includes('no-change baseline')],
]
viewChecks.forEach(([label, ok]) => {
  if (!ok) failed += 1
  console.log(`  ${ok ? 'ok  ' : 'FAIL'}  ${label}`)
})

console.log(failed ? `\n${failed} component(s) failed` : '\nall components rendered')
process.exit(failed ? 1 : 0)
