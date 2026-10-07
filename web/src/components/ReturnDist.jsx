import React, { useMemo, useState } from 'react'
import {
  Bar, BarChart, CartesianGrid, Cell, ReferenceArea, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts'
import { C, num, pct } from '../format.js'

/**
 * What the model is predicting against what actually happens.
 *
 * The bars are the distribution of REALISED h-candle returns on the TRAIN
 * split. The shaded band is the frozen NEUTRAL region: returns inside it are
 * labelled NEUTRAL, which is why ~40% of the training set is NEUTRAL by
 * construction. The white line is today's prediction for this horizon.
 *
 * The chart answers the question a bare accuracy number does not: how far
 * outside the middle the model is actually willing to commit, and whether the
 * move it expects is large compared with the noise it is sitting in.
 */
export default function ReturnDist({ snapshot }) {
  const horizons = snapshot?.prediction?.horizons || []
  const [label, setLabel] = useState('60m')
  const active = horizons.find((h) => h.label === label) || horizons[0]

  const data = useMemo(() => {
    const bands = snapshot?.bands
    if (!bands || !active) return []
    const entry = bands.overall?.[active.label]
    if (!entry?.hist) return []
    const edges = bands.hist_edges
    return entry.hist.map((count, i) => ({
      centre: (edges[i] + edges[i + 1]) / 2,
      lo: edges[i],
      hi: edges[i + 1],
      count,
    }))
  }, [snapshot?.generated_at, active?.label])

  if (!active) return <div className="muted">no prediction yet</div>
  if (!data.length) {
    return (
      <div className="muted">
        no train distribution on disk — run{' '}
        <code>python src/server.py --rebuild-bands</code>
      </div>
    )
  }

  const thr = snapshot.thresholds?.[active.label]
  const pred = active.expected_return_pct
  // Colour each bar by the class its returns would be labelled with.
  const colour = (v) => {
    if (!thr) return C.quiet
    if (v > thr.upper_pct) return C.up
    if (v < thr.lower_pct) return C.down
    return C.neutral
  }

  return (
    <div>
      <div style={{ display: 'flex', gap: 6, marginBottom: 10, flexWrap: 'wrap' }}>
        {horizons.map((h) => (
          <button
            key={h.label}
            onClick={() => setLabel(h.label)}
            className={`chip ${h.label === active.label ? 'warn' : ''}`}
            style={{ cursor: 'pointer', background: 'transparent' }}
          >
            {h.label}
          </button>
        ))}
      </div>

      <ResponsiveContainer width="100%" height={230}>
        <BarChart data={data} margin={{ top: 6, right: 12, bottom: 4, left: -18 }}>
          <CartesianGrid stroke={C.grid} vertical={false} />
          <XAxis
            dataKey="centre" type="number" domain={['dataMin', 'dataMax']}
            tickFormatter={(v) => `${v.toFixed(1)}%`}
            stroke={C.axis} tick={{ fontSize: 10 }} tickLine={false}
          />
          <YAxis stroke={C.axis} tick={{ fontSize: 10 }} tickLine={false}
                 tickFormatter={(v) => num(v)} />
          <Tooltip
            contentStyle={{
              background: '#1a1f2e', border: '1px solid #333c52',
              borderRadius: 8, fontSize: 12,
            }}
            formatter={(v, _n, p) => [
              `${num(v)} candles`,
              `${p.payload.lo.toFixed(2)}% … ${p.payload.hi.toFixed(2)}%`,
            ]}
          />
          {thr ? (
            <ReferenceArea
              x1={thr.lower_pct} x2={thr.upper_pct}
              fill="#d9a441" fillOpacity={0.10} strokeOpacity={0}
            />
          ) : null}
          <ReferenceLine x={0} stroke="#4a5468" strokeDasharray="3 3" />
          <ReferenceLine
            x={pred} stroke="#ffffff" strokeWidth={2}
            label={{ value: `model ${pred >= 0 ? '+' : ''}${pred.toFixed(3)}%`,
                     position: 'top', fill: '#e6e9ef', fontSize: 11 }}
          />
          <Bar dataKey="count" isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={colour(d.centre)} fillOpacity={0.72} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="note">
        {num(data.reduce((a, d) => a + d.count, 0))} training candles,{' '}
        {active.label} horizon
        {(() => {
          const o = snapshot.bands?.overall?.[active.label]
          const tail = (o?.hist_under || 0) + (o?.hist_over || 0)
          return tail
            ? `, plus ${num(tail)} outside the ±4% window this axis shows`
            : ''
        })()}
        . The shaded band is the NEUTRAL region
        {thr ? ` (${thr.lower_pct.toFixed(2)}% … ${thr.upper_pct.toFixed(2)}%)` : ''}
        {' '}fixed from the training split. NEUTRAL is{' '}
        {pct((data.filter((d) => thr && d.centre >= thr.lower_pct
             && d.centre <= thr.upper_pct).reduce((a, d) => a + d.count, 0))
             / data.reduce((a, d) => a + d.count, 0), 1)}
        {' '}of it <em>by construction</em> — which is why the direction head's
        accuracy has to be read against the majority-class rate, not against
        50%.
      </p>
    </div>
  )
}
