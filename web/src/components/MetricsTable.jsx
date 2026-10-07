import React from 'react'
import { num, pct, signed } from '../format.js'

/**
 * The held-out numbers, and the baselines they have to beat.
 *
 * Shown next to the live scorecard on purpose. The live numbers are the more
 * honest kind of evidence (the calls were made before the outcomes existed)
 * but there are far too few of them to be precise; these are the opposite -
 * statistically solid, but on a split that the model's design was chosen
 * against, so they are an optimistic read. Neither alone is the truth, so both
 * are on screen and each says which it is.
 */
export default function MetricsTable({ snapshot }) {
  const m = snapshot?.metrics
  if (!m) return <div className="muted">no evaluation summary on disk</div>
  if (m.note) {
    return (
      <div>
        <div className="muted">{m.note}</div>
        <p className="note">
          Available keys: {(m.available_keys || []).join(', ') || '—'}. Run{' '}
          <code>python src/evaluate.py</code> to produce this model&apos;s row.
        </p>
      </div>
    )
  }

  const d = m.direction || {}
  const r = m.regime || {}
  const isTest = m.split === 'test'

  return (
    <div>
      <table className="tbl">
        <thead>
          <tr>
            <th>metric</th>
            <th>model</th>
            <th>baseline</th>
            <th>lift</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td className="txt">activity accuracy (all horizons)</td>
            <td>{pct(r.accuracy, 2)}</td>
            <td>{pct(r.baseline_accuracy, 2)}</td>
            <td style={{ color: (r.lift ?? 0) > 0 ? '#26a69a' : '#ef5350' }}>
              {r.lift === undefined ? '—' : `+${(r.lift * 100).toFixed(1)}pp`}
            </td>
          </tr>
          <tr>
            <td className="txt">activity macro-F1</td>
            <td>{pct(r.macro_F1, 3)}</td>
            <td className="muted">—</td>
            <td className="muted">—</td>
          </tr>
          <tr>
            <td className="txt">direction accuracy</td>
            <td>{pct(d.accuracy, 2)}</td>
            <td>{pct(d.baseline_accuracy, 2)}</td>
            <td style={{
              color: (d.accuracy ?? 0) > (d.baseline_accuracy ?? 0)
                ? '#26a69a' : '#ef5350',
            }}>
              {d.accuracy === undefined || d.baseline_accuracy === undefined
                ? '—'
                : `${signed((d.accuracy - d.baseline_accuracy) * 100, 1)}pp`}
            </td>
          </tr>
          <tr>
            <td className="txt">direction macro-F1</td>
            <td>{pct(d.macro_f1, 3)}</td>
            <td className="muted">—</td>
            <td className="muted">—</td>
          </tr>
        </tbody>
      </table>

      {r.per_horizon ? (
        <table className="tbl" style={{ marginTop: 12 }}>
          <thead>
            <tr>
              <th>activity, per horizon</th>
              <th>accuracy</th>
              <th>baseline</th>
              <th>lift</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(r.per_horizon).map(([h, v]) => (
              <tr key={h}>
                <td className="txt">{h}</td>
                <td>{pct(v.accuracy, 2)}</td>
                <td>{pct(v.baseline, 2)}</td>
                <td style={{ color: v.lift > 0 ? '#26a69a' : '#ef5350' }}>
                  +{(v.lift * 100).toFixed(1)}pp
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      <p className={`note ${isTest ? 'good' : 'warn'}`}>
        {isTest
          ? `Measured on the held-out TEST split (${num(m.n_samples)} samples) — the 15% of history no threshold, scaler or early-stopping decision ever touched. It is the most honest number here, and still an average over nine years of a market that changed a great deal.`
          : `This is the VALIDATION split, not test — early stopping and the class-weight settings were chosen against it, so treat it as optimistic.`}
        {m.generated_utc ? ` Generated ${m.generated_utc}.` : ''}
      </p>
      <p className="note">
        The activity baseline is ~0.50 because the two classes are balanced by
        construction; the direction baseline is the majority-class rate (~0.52),
        which is what always predicting NEUTRAL scores. Read the two headlines
        against their own baselines — the direction lift is a couple of
        percentage points, the activity lift is much larger. That difference is
        the honest summary of what this model can and cannot do.
      </p>
    </div>
  )
}
