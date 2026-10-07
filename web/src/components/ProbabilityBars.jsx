import React from 'react'
import { C, pct } from '../format.js'

/**
 * Per-horizon class probabilities, as a stacked bar.
 *
 * Used for both heads. They are deliberately separate components-in-use
 * rather than one pooled chart: direction is 3-class with NEUTRAL carrying
 * ~40% of the mass, activity is 2-class and balanced by construction, so
 * putting them on one axis would invite comparing numbers that are not
 * comparable.
 */
export function DirectionBars({ horizons }) {
  return (
    <div>
      {horizons.map((h) => {
        const p = h.direction_probs
        const segs = [
          { k: 'DOWN', v: p.DOWN, c: C.down },
          { k: 'NEUTRAL', v: p.NEUTRAL, c: C.neutral },
          { k: 'UP', v: p.UP, c: C.up },
        ]
        return (
          <div className="bar-row" key={h.label}>
            <span className="lbl">{h.label}</span>
            <div className="bar-track">
              {segs.map((s) => (
                <div
                  key={s.k}
                  className="bar-seg"
                  style={{ width: `${s.v * 100}%`, background: s.c }}
                  title={`${s.k} ${pct(s.v, 1)}`}
                >
                  {s.v > 0.12 ? pct(s.v, 0) : ''}
                </div>
              ))}
            </div>
            <span className="bar-val">
              {h.direction} {pct(h.direction_confidence, 0)}
            </span>
          </div>
        )
      })}
    </div>
  )
}

export function ActivityBars({ horizons, tau, reference, refWindow }) {
  const withActivity = horizons.filter((h) => h.activity_probs)
  if (!withActivity.length) {
    return <div className="muted">this model has no activity head</div>
  }
  return (
    <div>
      {withActivity.map((h) => {
        const p = h.activity_probs
        const segs = [
          { k: 'QUIET', v: p.QUIET, c: C.quiet },
          { k: 'ACTIVE', v: p.ACTIVE, c: C.active },
        ]
        const t = tau?.[h.label]
        return (
          <div className="bar-row" key={h.label}>
            <span className="lbl">{h.label}</span>
            <div className="bar-track">
              {segs.map((s) => (
                <div
                  key={s.k}
                  className="bar-seg"
                  style={{ width: `${s.v * 100}%`, background: s.c }}
                  title={`${s.k} ${pct(s.v, 1)}`}
                >
                  {s.v > 0.14 ? pct(s.v, 0) : ''}
                </div>
              ))}
            </div>
            <span className="bar-val">
              {h.activity} {pct(h.activity_confidence, 0)}
              {t && reference
                ? ` · active if >${(t * reference).toFixed(3)}%`
                : ''}
            </span>
          </div>
        )
      })}
      <p className="note">
        ACTIVE means the model expects the next {withActivity[0]?.candles ?? 1}
        –{withActivity[withActivity.length - 1]?.candles ?? 6} candles to move
        more than this market&apos;s own trailing {refWindow}-candle average.
        The threshold is the frozen TRAIN median of that ratio, which is why the
        two classes are 50/50 and why <strong>0.50 is the number to beat</strong>
        — unlike the direction head, whose majority class alone scores ~0.48.
        {reference
          ? ` Current trailing average: ${reference.toFixed(3)}% per 15m candle.`
          : ''}
      </p>
    </div>
  )
}
