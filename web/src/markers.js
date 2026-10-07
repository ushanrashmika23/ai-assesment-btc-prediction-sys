// Chart markers for the model's recorded calls. No model logic lives here -
// every marker is derived from a call the API returned, nothing is inferred.
import { C } from './format.js'

/**
 * Turn `track_record.recent` into markers a candlestick series will accept.
 *
 * Lightweight Charts asserts that marker times are STRICTLY ascending, and the
 * payload is neither ascending nor unique - by construction, not by accident:
 *
 *   * `recent` is in the order the calls were MADE, and a window is written
 *     as its six direction horizons (15m..90m, targets 1..6 candles out) and
 *     then its four activity horizons (15m, 30m, 60m, 90m, targets 1, 2, 4, 6).
 *     So target_ts climbs, then drops back to +1 candle: the array runs
 *     backwards mid-window.
 *   * Both heads answer the same 15m/30m/60m/90m candles, so four of a
 *     window's ten calls resolve at a timestamp another call already claimed.
 *
 * Sorting alone would leave the duplicates, which the same assertion rejects.
 * Calls sharing a candle are therefore merged into ONE marker whose text is a
 * tally (`2/2`, `1/2`) - dropping three of four markers would hide the fact
 * that two heads disagreed at that candle, which is the interesting part.
 *
 * Anything that cannot be placed on the axis - a missing, non-numeric, or
 * not-an-actual-candle timestamp - is skipped rather than passed through.
 *
 * @param {Array} recent  snapshot.track_record.recent
 * @param {Array} candles snapshot.candles, for the set of placeable times
 * @returns {Array} markers, time strictly ascending and unique
 */
export function buildMarkers(recent, candles) {
  const candleTimes = new Set()
  for (const c of candles || []) {
    if (Number.isFinite(c?.time)) candleTimes.add(c.time)
  }

  const byTime = new Map()
  for (const c of recent || []) {
    if (!c || !c.resolved || c.target_ts == null) continue
    const t = Math.floor(Number(c.target_ts) / 1000)
    if (!Number.isFinite(t) || !candleTimes.has(t)) continue

    const g = byTime.get(t) || { time: t, n: 0, hits: 0, regime: 0 }
    g.n += 1
    if (c.hit) g.hits += 1
    if (c.kind === 'regime') g.regime += 1
    byTime.set(t, g)
  }

  return [...byTime.values()]
    .sort((a, b) => a.time - b.time)
    .map((g) => ({
      time: g.time,
      // A tie goes below the bar; a call the model got wrong goes above it.
      position: g.hits * 2 >= g.n ? 'belowBar' : 'aboveBar',
      color: g.hits === g.n ? C.up : g.hits === 0 ? C.down : C.neutral,
      shape: 'circle',
      // One call, one verdict - the shape a single call always had. Two or
      // more, and the marker has to say how many of them landed.
      text: g.n === 1
        ? (g.regime ? (g.hits ? 'a' : 'x') : '●')
        : `${g.hits}/${g.n}`,
      size: 1,
    }))
}
