/**
 * T13 probe -- drive the dashboard's real `buildMarkers()` with calls in the
 * order the server issues them and report the result as JSON on stdout.
 *
 * This is NOT a reimplementation of the marker builder. It imports
 * `web/src/markers.js` and calls it, so what is verified is the function the
 * chart actually uses. The probe adds three things the project's own
 * `web/markers-check.mjs` does not:
 *
 *   1. a small, hand-computed fixture (one window: 10 calls, 6 distinct target
 *      candles) whose expected output is worked out on paper at the top of
 *      this file, rather than derived from the payload -- so a change in the
 *      merge rule has to disagree with arithmetic, not just with itself;
 *   2. the raw marker arrays for that fixture, emitted so the Python test can
 *      re-assert the same numbers from the other side of the language border;
 *   3. a shuffled-input check over every one of many permutations.
 *
 * `web/markers-check.mjs` remains the project's own check and is run separately
 * by tests/test_markers.py; this file must not be read as replacing it.
 *
 * Run:  node tests/node/marker_probe.mjs
 * Exit: 0 when every check passed, 1 otherwise. JSON on stdout either way.
 */
import { createRequire } from 'node:module'
import { pathToFileURL } from 'node:url'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const WEB = resolve(HERE, '..', '..', 'web')

const { buildMarkers } = await import(pathToFileURL(resolve(WEB, 'src', 'markers.js')).href)
const { C } = await import(pathToFileURL(resolve(WEB, 'src', 'format.js')).href)

const checks = []
const check = (name, fn) => {
  try {
    const detail = fn()
    checks.push({ name, ok: true, detail: detail === undefined ? null : String(detail) })
  } catch (err) {
    checks.push({ name, ok: false, detail: err.message })
  }
}
const assert = (cond, msg) => { if (!cond) throw new Error(msg) }

// ---------------------------------------------------------------------------
// The fixture: ONE window, written the way src/server.py's _issue_calls() does.
//
// A window is its six direction horizons (15m..90m -> targets 1..6 candles out)
// followed by its four activity horizons (15m, 30m, 60m, 90m -> targets 1, 2,
// 4, 6). The second block RESTARTS the target list at +1, so `target_ts` climbs
// to +6 and then drops back to +1: the array runs backwards in the middle, and
// four of its ten entries land on a candle another entry already claimed.
//
// Hits are chosen by hand so that every merge outcome appears exactly once:
//   target  direction  regime   ->  n  hits  text   position    colour
//   +1      miss       miss     ->  2   0     0/2    aboveBar    down
//   +2      hit        miss     ->  2   1     1/2    belowBar    neutral
//   +3      miss       --       ->  1   0     ●      aboveBar    down
//   +4      hit        hit      ->  2   2     2/2    belowBar    up
//   +5      miss       --       ->  1   0     ●      aboveBar    down
//   +6      hit        hit      ->  2   2     2/2    belowBar    up
//
// 10 calls, 6 distinct candles -> 6 markers. That is the number T13 asks about.
// ---------------------------------------------------------------------------
const CANDLE_MS = 900000
const ANCHOR = 1790610300000          // 2026-09-28 15:45 UTC, on the 15m grid
const HITS = { 1: false, 2: true, 3: false, 4: true, 5: false, 6: true }
const REGIME_HITS = { 1: false, 2: false, 4: true, 6: true }

const candleTimes = []
for (let k = 1; k <= 6; k += 1) candleTimes.push((ANCHOR + k * CANDLE_MS) / 1000)
const candles = candleTimes.map((time) => ({ time }))

const issueOrder = [
  ...[1, 2, 3, 4, 5, 6].map((k) => ({
    kind: 'direction', horizon: `${k * 15}m`, hit: HITS[k], resolved: true,
    target_ts: ANCHOR + k * CANDLE_MS,
  })),
  ...[1, 2, 4, 6].map((k) => ({
    kind: 'regime', horizon: `${k * 15}m`, hit: REGIME_HITS[k], resolved: true,
    target_ts: ANCHOR + k * CANDLE_MS,
  })),
]

const EXPECTED = [
  { time: candleTimes[0], text: '0/2', position: 'aboveBar', color: C.down },
  { time: candleTimes[1], text: '1/2', position: 'belowBar', color: C.neutral },
  { time: candleTimes[2], text: '●',   position: 'aboveBar', color: C.down },
  { time: candleTimes[3], text: '2/2', position: 'belowBar', color: C.up },
  { time: candleTimes[4], text: '●',   position: 'aboveBar', color: C.down },
  { time: candleTimes[5], text: '2/2', position: 'belowBar', color: C.up },
]

/** The chart's own contract, as Lightweight Charts asserts it. */
const assertAscending = (marks, label) => {
  let prev = null
  marks.forEach((m, i) => {
    assert(Number.isFinite(m.time), `${label}: marker ${i} has non-finite time ${m.time}`)
    if (prev !== null) {
      assert(m.time > prev,
        `${label}: times not strictly ascending at index ${i} ` +
        `(time=${m.time}, prev=${prev}) - setMarkers() rejects this`)
    }
    prev = m.time
  })
}

const shape = (marks) => marks.map((m) => ({
  time: m.time, text: m.text, position: m.position, color: m.color,
  shape: m.shape, size: m.size,
}))

const matchesExpected = (marks, label) => {
  assert(marks.length === EXPECTED.length,
    `${label}: ${marks.length} marker(s), expected ${EXPECTED.length}`)
  EXPECTED.forEach((want, i) => {
    const got = marks[i]
    for (const key of ['time', 'text', 'position', 'color']) {
      assert(got[key] === want[key],
        `${label}: marker ${i} (${got.time}) has ${key}=${JSON.stringify(got[key])}, ` +
        `expected ${JSON.stringify(want[key])}`)
    }
  })
}

// -- the fixture itself must be what it claims to be -----------------------
check('the fixture really is out of order and really is colliding', () => {
  const times = issueOrder.map((c) => c.target_ts)
  const back = times.filter((t, i) => i > 0 && t < times[i - 1]).length
  assert(back >= 1,
    'the fixture has no backwards step - it is already sorted, so it cannot ' +
    'reproduce the payload that crashed the chart')
  assert(new Set(times).size === 6 && times.length === 10,
    `fixture has ${times.length} calls over ${new Set(times).size} candles; ` +
    'expected 10 over 6')
  return `${times.length} calls, ${new Set(times).size} distinct candles, ` +
         `${back} backwards step(s)`
})

// -- T13 -------------------------------------------------------------------
check('10 out-of-order calls produce 6 merged markers', () => {
  const marks = buildMarkers(issueOrder, candles)
  assert(marks.length === 6,
    `${marks.length} marker(s) from 10 calls over 6 candles, expected 6`)
  return `${issueOrder.length} calls -> ${marks.length} markers`
})

check('the 6 markers are exactly the hand-computed ones', () => {
  const marks = buildMarkers(issueOrder, candles)
  matchesExpected(marks, 'issue order')
  return marks.map((m) => `${m.time}:${m.text}`).join(' ')
})

check('the markers are sorted, strictly ascending, on real candles', () => {
  const marks = buildMarkers(issueOrder, candles)
  assertAscending(marks, 'issue order')
  const axis = new Set(candleTimes)
  for (const m of marks) {
    assert(axis.has(m.time), `marker at ${m.time} is not a candle in the series`)
  }
  return `${marks.length} markers, strictly ascending`
})

check('input order does not change the output (every rotation + reversal)', () => {
  const reference = JSON.stringify(shape(buildMarkers(issueOrder, candles)))
  const variants = {
    'sorted': [...issueOrder].sort((a, b) => a.target_ts - b.target_ts),
    'reversed': [...issueOrder].reverse(),
    'reversed-sorted': [...issueOrder].sort((a, b) => b.target_ts - a.target_ts),
  }
  // Every rotation too, so no single shuffle is the one that happens to work.
  for (let i = 1; i < issueOrder.length; i += 1) {
    variants[`rotated by ${i}`] = [...issueOrder.slice(i), ...issueOrder.slice(0, i)]
  }
  for (const [label, calls] of Object.entries(variants)) {
    const got = JSON.stringify(shape(buildMarkers(calls, candles)))
    assert(got === reference,
      `the markers differ when the calls arrive ${label}`)
  }
  return `${Object.keys(variants).length} orderings agree`
})

check('merging preserves every call - none is silently dropped', () => {
  const marks = buildMarkers(issueOrder, candles)
  const counted = marks.reduce(
    (n, m) => n + (m.text.includes('/') ? Number(m.text.split('/')[1]) : 1), 0)
  assert(counted === issueOrder.length,
    `${issueOrder.length} calls but the markers account for ${counted}`)
  return `${counted} calls accounted for by ${marks.length} markers`
})

check('a wrong call sits above the bar, a right one below', () => {
  const marks = buildMarkers(issueOrder, candles)
  for (const m of marks) {
    const [hit, n] = m.text.includes('/')
      ? m.text.split('/').map(Number)
      : [m.text === '●' ? 0 : 1, 1]
    if (m.text === '●') continue            // single-call shape checked below
    const expected = hit * 2 >= n ? 'belowBar' : 'aboveBar'
    assert(m.position === expected,
      `marker ${m.text} is ${m.position}, expected ${expected} (tie goes below)`)
  }
  return 'position rule holds for every merged marker'
})

check('colour follows the tally: all right, all wrong, or split', () => {
  const marks = buildMarkers(issueOrder, candles)
  const byText = Object.fromEntries(marks.map((m) => [m.text, m.color]))
  assert(byText['2/2'] === C.up, `2/2 is ${byText['2/2']}, expected ${C.up}`)
  assert(byText['0/2'] === C.down, `0/2 is ${byText['0/2']}, expected ${C.down}`)
  assert(byText['1/2'] === C.neutral, `1/2 is ${byText['1/2']}, expected ${C.neutral}`)
  return `2/2=${C.up} 1/2=${C.neutral} 0/2=${C.down}`
})

check('a lone call keeps the single-call marker it always had', () => {
  const one = [{ time: candleTimes[0] }]
  const at = candleTimes[0] * 1000
  const cases = [
    [{ kind: 'direction', hit: true }, '●', 'belowBar'],
    [{ kind: 'direction', hit: false }, '●', 'aboveBar'],
    [{ kind: 'regime', hit: true }, 'a', 'belowBar'],
    [{ kind: 'regime', hit: false }, 'x', 'aboveBar'],
  ]
  for (const [call, text, position] of cases) {
    const marks = buildMarkers([{ ...call, resolved: true, target_ts: at }], one)
    assert(marks.length === 1, `${call.kind}/${call.hit}: ${marks.length} markers`)
    assert(marks[0].text === text,
      `${call.kind} ${call.hit ? 'hit' : 'miss'}: text is ${marks[0].text}, expected ${text}`)
    assert(marks[0].position === position,
      `${call.kind} ${call.hit ? 'hit' : 'miss'}: position is ${marks[0].position}, expected ${position}`)
  }
  return 'direction ● / regime a·x, both positions'
})

check('unplaceable calls are skipped, not drawn, and never throw', () => {
  const marks = buildMarkers([
    { resolved: true, target_ts: null },
    { resolved: true, target_ts: undefined },
    { resolved: true },
    { resolved: true, target_ts: NaN },
    { resolved: true, target_ts: 'not a number' },
    { resolved: true, target_ts: {} },
    { resolved: true, target_ts: [] },
    { resolved: true, target_ts: -1 },
    { resolved: true, target_ts: 1 },                     // finite, no candle
    { resolved: false, target_ts: candleTimes[0] * 1000 }, // not resolved yet
    { target_ts: candleTimes[1] * 1000 },                  // no resolved flag
    null,
    undefined,
  ], candles)
  assert(marks.length === 0,
    `${marks.length} marker(s) drawn for calls that cannot be placed`)
  return '13 unplaceable calls -> 0 markers'
})

check('empty, null and undefined payloads produce an empty array', () => {
  for (const [a, b] of [[[], []], [null, null], [undefined, undefined],
                        [[], null], [null, []], [issueOrder, []],
                        [issueOrder, null]]) {
    const out = buildMarkers(a, b)
    assert(Array.isArray(out), 'did not return an array')
    assert(out.length === 0, `${out.length} marker(s) from an empty axis`)
  }
  return '7 degenerate input pairs -> []'
})

check('a candle missing from the series is not marked', () => {
  // Same calls, but the axis is cut to the first three candles: half the
  // targets have not happened yet, so half the markers must not exist.
  const truncated = candles.slice(0, 3)
  const marks = buildMarkers(issueOrder, truncated)
  assert(marks.length === 3,
    `${marks.length} marker(s) on a 3-candle axis, expected 3`)
  assert(marks.every((m) => m.time <= truncated[2].time),
    'a marker was placed beyond the end of the axis')
  return 'axis cut to 3 candles -> 3 markers'
})

check('several windows in issue order come out chronological', () => {
  const window = (anchor) => ([
    ...[1, 2, 3, 4, 5, 6].map((k) => ({
      kind: 'direction', resolved: true, hit: k % 2 === 0,
      target_ts: anchor + k * CANDLE_MS,
    })),
    ...[1, 2, 4, 6].map((k) => ({
      kind: 'regime', resolved: true, hit: k > 2,
      target_ts: anchor + k * CANDLE_MS,
    })),
  ])
  const recent = [...window(ANCHOR), ...window(ANCHOR + 6 * CANDLE_MS)]
  const axis = []
  for (let i = -2; i < 20; i += 1) axis.push({ time: (ANCHOR + i * CANDLE_MS) / 1000 })
  const marks = buildMarkers(recent, axis)
  assertAscending(marks, 'two windows')
  const counted = marks.reduce(
    (n, m) => n + (m.text.includes('/') ? Number(m.text.split('/')[1]) : 1), 0)
  assert(counted === recent.length,
    `${recent.length} calls across two windows, ${counted} accounted for`)
  // Two windows six candles apart share no target: the first names A+1..A+6
  // and the second A+7..A+12, so twelve distinct candles and twelve markers.
  // (Windows are only ever six candles apart here to make that arithmetic
  // exact; the real engine issues one every candle, which is where the
  // overlapping targets in the captured payload come from.)
  assert(marks.length === 12,
    `${marks.length} marker(s) for two windows six candles apart, expected 12 ` +
    '(A+1..A+6 and A+7..A+12)')
  return `20 calls -> ${marks.length} markers, all accounted for`
})

// ---------------------------------------------------------------------------
const failures = checks.filter((c) => !c.ok)
const canonical = shape(buildMarkers(issueOrder, candles))

process.stdout.write(JSON.stringify({
  probe: 'markers',
  node: process.version,
  module: resolve(WEB, 'src', 'markers.js'),
  palette: { up: C.up, down: C.down, neutral: C.neutral },
  fixture: {
    anchor_ms: ANCHOR,
    candle_ms: CANDLE_MS,
    calls: issueOrder.length,
    distinct_candles: 6,
  },
  expected: EXPECTED,
  canonical,
  checks,
  n_checks: checks.length,
  failures: failures.length,
}, null, 2) + '\n')

process.exit(failures.length ? 1 : 0)
