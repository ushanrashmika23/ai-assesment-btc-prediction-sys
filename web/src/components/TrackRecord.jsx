import React, { useMemo } from 'react'
import {
  CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer,
  Tooltip, XAxis, YAxis,
} from 'recharts'
import { C, num, pct } from '../format.js'

/**
 * The model's own live scorecard, cumulative.
 *
 * This is the only place in the UI where the accuracy figure is measured on
 * data the model had genuinely never seen AND that was not used to select it -
 * because the calls were made live, before the outcome existed. It is also a
 * very small sample, so it is drawn as a running line rather than printed as a
 * headline number: the early points swing wildly and the line only becomes
 * informative after a few dozen resolutions.
 *
 * The two heads are charted separately, never pooled: their baselines differ
 * (0.50 for activity, the majority-class rate for direction), so a single
 * blended "accuracy" would describe neither.
 */
function running(calls) {
  let hits = 0
  return calls.map((c, i) => {
    hits += c.hit ? 1 : 0
    return { i: i + 1, acc: hits / (i + 1), hit: c.hit }
  })
}

export default function TrackRecord({ snapshot, head = 'regime' }) {
  const tr = snapshot?.track_record
  const series = useMemo(() => {
    const rows = (tr?.recent || []).filter(
      (c) => c.resolved && c.kind === head,
    )
    return running(rows)
  }, [tr?.recent, head])

  const tally = head === 'regime' ? tr?.regime : tr?.direction
  const baseline = head === 'regime' ? 0.5 : null

  if (!series.length) {
    return (
      <div className="muted" style={{ padding: '10px 0' }}>
        No resolved {head === 'regime' ? 'activity' : 'direction'} calls yet.
        Calls resolve once their target candle prints, so this fills in as the
        server keeps running — {num(tr?.n_resolved || 0)} resolved so far.
        <p className="note">
          A live scorecard is a thin sample and will be noisy. The held-out
          test figures in the table below are the ones with statistical weight;
          this is here so the model is judged on calls made before the outcome
          was known, not only on a test set it was tuned against.
        </p>
      </div>
    )
  }

  return (
    <div>
      <ResponsiveContainer width="100%" height={210}>
        <LineChart data={series} margin={{ top: 8, right: 14, bottom: 4, left: -20 }}>
          <CartesianGrid stroke={C.grid} />
          <XAxis dataKey="i" stroke={C.axis} tick={{ fontSize: 10 }}
                 tickLine={false} label={{
                   value: 'resolved calls (oldest → newest)', position: 'insideBottom',
                   offset: -2, fill: C.axis, fontSize: 10,
                 }} />
          <YAxis domain={[0, 1]} stroke={C.axis} tick={{ fontSize: 10 }}
                 tickLine={false} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ background: '#1a1f2e', border: '1px solid #333c52',
                            borderRadius: 8, fontSize: 12 }}
            formatter={(v) => [pct(v, 1), 'cumulative accuracy']}
            labelFormatter={(i) => `after ${i} calls`}
          />
          {baseline !== null ? (
            <ReferenceLine y={baseline} stroke={C.naive} strokeDasharray="4 4"
              label={{ value: 'baseline 50%', position: 'right',
                       fill: C.naive, fontSize: 10 }} />
          ) : null}
          <Line type="monotone" dataKey="acc" stroke={C.active} strokeWidth={2}
                dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>

      <p className="note">
        Cumulative hit rate over the last {num(series.length)} resolved{' '}
        {head === 'regime' ? 'activity' : 'direction'} calls
        {tally?.n ? ` (${num(tally.n)} total, ${pct(tally.accuracy, 1)})` : ''}.
        {head === 'regime'
          ? ' The activity classes are balanced by construction, so 0.50 is the number to beat.'
          : ' Direction is 3-class and dominated by NEUTRAL, so its baseline is the majority-class rate, not 0.50 — a raw accuracy here is not comparable with the activity line.'}
      </p>
    </div>
  )
}
