/**
 * `npm run check:markers` — prove buildMarkers() output satisfies the contract
 * Lightweight Charts asserts on, starting from the payload that broke it.
 *
 * Why this exists: the crash was `setMarkers()` rejecting its argument with
 * "data must be asc ordered by time, index=2, time=1790610300, prev time=1790611200",
 * and that assertion only fires in a browser, against whatever the server
 * happened to be serving. The payload is reproduced here in Node so a
 * regression fails in a second instead of in the browser.
 *
 * The pathological order is CONSTRUCTED from the captured calls rather than
 * hoped for in them. It used to be read straight out of web/_snapshot.json,
 * back when the server emitted `track_record.recent` in issue order - but
 * src/server.py now sorts that array, so a re-captured fixture is already
 * ascending and the assertion that "the fixture contains a backwards step"
 * fails on every re-capture. Worse, it fails for the *right* reason: the bug
 * it is guarding is fixed, so the guard reads as a false alarm. Issue order is
 * instead rebuilt here from the same real calls, using the rule that produced
 * it (six direction horizons, then four activity ones restarting at +1), which
 * is stable across re-captures and still fails if buildMarkers() regresses.
 *
 * The assertion in assertChartAccepts() is a reimplementation of the chart's
 * own check (strictly increasing, so equal times fail too), not a rule of our
 * own invention.
 */
import { readFileSync } from 'node:fs'
import { buildMarkers } from './src/markers.js'

let failures = 0
const check = (name, fn) => {
  try {
    fn()
    console.log(`  ok    ${name}`)
  } catch (err) {
    failures += 1
    console.log(`  FAIL  ${name}\n        ${err.message}`)
  }
}
const assert = (cond, msg) => { if (!cond) throw new Error(msg) }

/** The chart's contract: strictly ascending, so equal times fail as well. */
const assertChartAccepts = (marks, label) => {
  let prev = null
  marks.forEach((m, i) => {
    assert(Number.isFinite(m.time), `${label}: marker ${i} has non-finite time ${m.time}`)
    if (prev !== null) {
      assert(m.time > prev,
        `${label}: data must be asc ordered by time, index=${i}, ` +
        `time=${m.time}, prev time=${prev}`)
    }
    prev = m.time
  })
}

/** How many calls a marker stands for: `2/2` is two, `●` is one. */
const callsIn = (m) => (m.text.includes('/') ? Number(m.text.split('/')[1]) : 1)

const CANDLE_MS = 900000
// horizon label -> candles ahead, mirroring CANDLES_BY_LABEL in src/server.py.
const CANDLES_BY_LABEL = { '15m': 1, '30m': 2, '45m': 3, '60m': 4, '75m': 5, '90m': 6 }

const snapshot = JSON.parse(readFileSync(new URL('./_snapshot.json', import.meta.url)))
const realRecent = snapshot.track_record.recent
const realCandles = snapshot.candles

/**
 * Rebuild the order `_issue_calls()` writes: per window, every direction
 * horizon by distance, then every activity horizon by distance - the second
 * block restarting the target list at +1 candle. That restart is the backwards
 * step, and the four shared targets (+1, +2, +4, +6) are the duplicates.
 *
 * Derived from the calls themselves so it survives a re-capture: the anchor is
 * recovered from the target and the horizon's distance rather than read from a
 * field the payload does not carry.
 */
const toIssueOrder = (calls) => {
  const byAnchor = new Map()
  for (const c of calls) {
    const k = CANDLES_BY_LABEL[c.horizon]
    if (!k) continue
    const anchor = c.target_ts - k * CANDLE_MS
    if (!byAnchor.has(anchor)) byAnchor.set(anchor, [])
    byAnchor.get(anchor).push(c)
  }
  const out = []
  for (const anchor of [...byAnchor.keys()].sort((a, b) => a - b)) {
    const win = byAnchor.get(anchor)
    const byTarget = (a, b) => a.target_ts - b.target_ts
    out.push(...win.filter((c) => c.kind === 'direction').sort(byTarget))
    out.push(...win.filter((c) => c.kind === 'regime').sort(byTarget))
  }
  return out
}

const issueRecent = toIssueOrder(realRecent)

// A call's target is 1-6 candles AHEAD of the anchor, so a capture taken the
// moment the window was issued has no target candle yet and nothing resolves -
// which is why it draws nothing and does not crash. The crash needs both:
// resolved calls, and an axis to put them on. Both are simulated below; the
// calls themselves are the fixture's, untouched.
const resolvedRecent = realRecent.map((c) => ({
  ...c,
  resolved: true,
  // Test input, not a claim: which calls the model got right is irrelevant to
  // whether the marker layer can draw them. Alternating gives each shared
  // candle a mix of hits and misses, which is the case the merge must report.
  hit: Math.round(c.target_ts / CANDLE_MS) % 2 === 0,
}))
const issueResolved = toIssueOrder(resolvedRecent)

const axis = [...realCandles,
  ...[...new Set(realRecent.map((c) => Math.floor(c.target_ts / 1000)))]
    .map((t) => ({ time: t }))]
  .sort((a, b) => a.time - b.time)

console.log('markers.js — contract check')

// -- 1. the payload that caused the crash ---------------------------------
check('the rebuilt issue order really is pathological', () => {
  const times = issueRecent.map((c) => c.target_ts)
  assert(times.some((t, i) => i > 0 && t < times[i - 1]),
    'no backwards step - toIssueOrder() no longer reproduces the crash payload')
  assert(new Set(times).size !== times.length,
    'no duplicate target_ts - toIssueOrder() no longer reproduces the crash payload')
})

check('the fix does not depend on how the calls arrive', () => {
  // Three inputs, one output: the raw payload, the raw payload reversed, and
  // the order _issue_calls() actually writes. If the merge depended on input
  // order at all, these would differ.
  const sorted = [...resolvedRecent].sort((a, b) => a.target_ts - b.target_ts)
  const reversed = [...resolvedRecent].reverse()
  const a = JSON.stringify(buildMarkers(sorted, axis))
  const b = JSON.stringify(buildMarkers(reversed, axis))
  const c = JSON.stringify(buildMarkers(issueResolved, axis))
  assert(a === b, 'the markers differ when the calls arrive reversed')
  assert(a === c, 'the markers differ between the server order and issue order')
  console.log(`        (${issueResolved.length - new Set(issueResolved.map((x) => x.target_ts)).size} ` +
              'duplicate(s) and the backwards step absorbed either way)')
})

check('a call whose target candle has not arrived draws no marker', () => {
  // Cut the axis to before the earliest target, which is the state of a
  // capture taken the moment a window was issued - reconstructed here rather
  // than assumed of the fixture, whose candles now extend past many targets.
  const earliest = Math.min(...realRecent.map((c) => Math.floor(c.target_ts / 1000)))
  const before = realCandles.filter((c) => c.time < earliest)
  assert(before.length > 0, 'the fixture has no candles before its earliest target')
  const marks = buildMarkers(realRecent, before)
  assert(marks.length === 0,
    `${marks.length} marker(s) drawn for candles that do not exist`)
})

check('the chart accepts the markers built from that payload', () => {
  const marks = buildMarkers(resolvedRecent, axis)
  assertChartAccepts(marks, 'captured, resolved')
  assert(marks.length > 0, 'no markers produced at all')
  for (const m of marks) {
    assert(axis.some((c) => c.time === m.time), `marker at ${m.time} is not on a candle`)
  }
  // The first marker must be the earliest call, not the first one made.
  const earliest = Math.min(...resolvedRecent.map((c) => Math.floor(c.target_ts / 1000)))
  assert(marks[0].time === earliest,
    `first marker is at ${marks[0].time}, earliest call is at ${earliest}`)
})

check('every resolved call is accounted for (merged, not silently dropped)', () => {
  const marks = buildMarkers(resolvedRecent, axis)
  const counted = marks.reduce((n, m) => n + callsIn(m), 0)
  assert(counted === resolvedRecent.length,
    `${resolvedRecent.length} calls resolved but the markers account for ${counted}`)
})

check('same-candle calls from both heads collapse into one marker', () => {
  const marks = buildMarkers(resolvedRecent, axis)
  const times = resolvedRecent.map((c) => Math.floor(c.target_ts / 1000))
  const distinct = [...new Set(times)]
  // Derived, not hardcoded: how many collisions this fixture has depends on how
  // many windows were captured, so a literal would break on every re-capture.
  const collisions = times.length - distinct.length
  assert(collisions > 0, 'the fixture has no shared candles - nothing to merge')
  assert(marks.length === distinct.length,
    `${marks.length} markers for ${distinct.length} distinct candles`)
  for (const m of marks) {
    const group = resolvedRecent.filter((c) => Math.floor(c.target_ts / 1000) === m.time)
    if (group.length === 1) continue          // single-call text is checked below
    const hits = group.filter((c) => c.hit).length
    assert(m.text === `${hits}/${group.length}`,
      `marker at ${m.time} says ${m.text}, its ${group.length} calls say ${hits}/${group.length}`)
  }
})

// -- 3. rendering is unchanged for the single-call case --------------------
check('a lone call keeps the marker it always had', () => {
  const one = [{ time: axis[0].time }]
  const dir = buildMarkers([{ resolved: true, kind: 'direction', hit: true, target_ts: one[0].time * 1000 }], one)
  assert(dir.length === 1 && dir[0].text === '●' && dir[0].position === 'belowBar',
    `direction hit: ${JSON.stringify(dir)}`)
  const miss = buildMarkers([{ resolved: true, kind: 'direction', hit: false, target_ts: one[0].time * 1000 }], one)
  assert(miss[0].position === 'aboveBar', 'a missed call must sit above the bar')
  const reg = buildMarkers([{ resolved: true, kind: 'regime', hit: true, target_ts: one[0].time * 1000 }], one)
  assert(reg[0].text === 'a', `regime hit: ${JSON.stringify(reg)}`)
  const regX = buildMarkers([{ resolved: true, kind: 'regime', hit: false, target_ts: one[0].time * 1000 }], one)
  assert(regX[0].text === 'x', `regime miss: ${JSON.stringify(regX)}`)
})

// -- 4. hostile input ------------------------------------------------------
check('missing / null / NaN / string / unknown-candle times are skipped', () => {
  const marks = buildMarkers([
    { resolved: true, target_ts: null },
    { resolved: true, target_ts: undefined },
    { resolved: true },
    { resolved: true, target_ts: NaN },
    { resolved: true, target_ts: 'not a number' },
    { resolved: true, target_ts: 1 },                            // finite, no candle
    { resolved: true, target_ts: realCandles[0].time * 1000 },   // the one good row
    null,
    undefined,
    { resolved: false, target_ts: realCandles[1].time * 1000 },  // not resolved yet
  ], realCandles)
  assertChartAccepts(marks, 'hostile')
  assert(marks.length === 1, `expected the 1 placeable call, got ${marks.length}`)
})

check('several windows in issue order come out chronological', () => {
  // Exactly as _issue_calls() writes them: six direction horizons then four
  // activity ones, the second block restarting the target list at +1 candle.
  const A = 1790610300000
  const win = (anchor) => ([
    ...[1, 2, 3, 4, 5, 6].map((k) => ({ resolved: true, kind: 'direction', hit: k % 2 === 0, target_ts: anchor + k * CANDLE_MS })),
    ...[1, 2, 4, 6].map((k) => ({ resolved: true, kind: 'regime', hit: k > 2, target_ts: anchor + k * CANDLE_MS })),
  ])
  const recent = [...win(A), ...win(A + 6 * CANDLE_MS)]
  const candles = []
  for (let i = -5; i < 30; i += 1) candles.push({ time: (A + i * CANDLE_MS) / 1000 })
  const marks = buildMarkers(recent, candles)
  assertChartAccepts(marks, 'two windows')
  assert(marks.reduce((n, m) => n + callsIn(m), 0) === recent.length,
    'a call lost its marker across the window boundary')
})

check('empty and null payloads produce no markers rather than throwing', () => {
  for (const [a, b] of [[[], []], [null, null], [undefined, undefined], [[], null], [null, []]]) {
    assert(Array.isArray(buildMarkers(a, b)), 'did not return an array')
    assert(buildMarkers(a, b).length === 0, 'produced markers from nothing')
  }
})

// -- what actually reaches the chart --------------------------------------
const marks = buildMarkers(resolvedRecent, axis)
console.log(`\nfixture, once its candles arrive: ${resolvedRecent.length} calls -> ${marks.length} markers`)
for (const m of marks) {
  console.log(`  ${String(m.time).padEnd(12)} ${m.text.padEnd(4)} ` +
              `${m.position === 'belowBar' ? 'below' : 'above'}  ${m.color}`)
}

console.log(failures ? `\n${failures} check(s) FAILED` : '\nall marker checks passed')
process.exit(failures ? 1 : 0)
