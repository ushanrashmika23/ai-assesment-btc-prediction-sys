import React from 'react'
import { money, num, pct, signed, stamp } from '../format.js'

/**
 * The raw call log, newest first.
 *
 * Nothing is filtered out: a wrong call sits next to a right one with the same
 * typography. A log that only showed the hits would be decoration.
 */
export default function CallsTable({ snapshot }) {
  const tr = snapshot?.track_record
  const rows = [...(tr?.recent || [])].reverse()

  if (!rows.length) {
    return (
      <div className="muted" style={{ padding: '8px 0' }}>
        No calls recorded yet. One set is issued per new 15m candle, per head
        (six direction calls, four activity calls), and each is resolved once
        its target candle prints.
        {tr?.log_file ? (
          <p className="note">
            Track record file: <code>{tr.log_file}</code> — one file per model,
            so two models&apos; calls are never blended into one hit rate.
          </p>
        ) : null}
      </div>
    )
  }

  return (
    <div className="scroll">
      <table className="tbl">
        <thead>
          <tr>
            <th>made</th>
            <th>head</th>
            <th>hz</th>
            <th>called</th>
            <th>conf</th>
            <th>price</th>
            <th>target</th>
            <th>outcome</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((c, i) => {
            const tone = c.resolved ? (c.hit ? '#26a69a' : '#ef5350') : '#9aa4b8'
            return (
              <tr key={`${c.target_ts}-${c.kind}-${c.horizon}-${i}`}>
                <td>{stamp(c.made_at).slice(11, 19)}</td>
                <td className="txt">{c.kind === 'regime' ? 'activity' : 'direction'}</td>
                <td>{c.horizon}</td>
                <td className="txt" style={{ color: tone }}>{c.predicted_label}</td>
                <td>{pct(c.confidence, 0)}</td>
                <td>{money(c.price_at_call)}</td>
                <td>{stamp(c.target_ts_iso).slice(11, 16)}</td>
                <td className="txt">
                  {!c.resolved
                    ? <span className="muted">pending</span>
                    : c.kind === 'regime'
                      ? `${c.realized_label} (${(c.realized_pct ?? 0).toFixed(3)}% vs ref)`
                      : `${c.realized_label} ${signed(c.realized_pct, 3)}%`}
                </td>
                <td style={{ color: tone }}>
                  {!c.resolved ? '' : c.hit ? 'hit' : 'miss'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <p className="note" style={{ padding: '8px 10px 0' }}>
        {num(tr?.n_calls)} calls recorded, {num(tr?.n_resolved)} resolved.{' '}
        {tr?.regime?.baseline_note
          ? `Activity baseline: ${tr.regime.baseline_note}.`
          : ''}
      </p>
    </div>
  )
}
