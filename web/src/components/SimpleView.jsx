import React from 'react'
import PriceChart from './PriceChart.jsx'
import { Panel } from './Panel.jsx'
import { money, num, pct, signed } from '../format.js'

/**
 * The dashboard for someone who has not read the rest of the project.
 *
 * It answers the two questions the model is actually asked, in the words a
 * person would use, and it says out loud which of the two is worth trusting.
 * That last part is the whole design constraint: simplifying the presentation
 * must never simplify the claim. The direction answer really is close to a
 * coin flip and the simple view says so on the card, in the same size type as
 * the answer itself - a view that hid that would be a better-looking lie than
 * the advanced one.
 *
 * Nothing here computes anything. Every number is one the API already
 * returned; this file only chooses which ones a newcomer sees first.
 */

const MOVE_PLAIN = {
  ACTIVE: 'Yes — a bigger move than usual',
  QUIET: 'No — a calmer stretch than usual',
}

const WAY_PLAIN = {
  UP: 'The model leans up',
  DOWN: 'The model leans down',
  NEUTRAL: 'The model sees no clear direction',
}

/** Two plain sentences per answer, so the card explains itself without a legend. */
function moveWords(activity, refWindow) {
  const win = refWindow ? `the last ${refWindow} candles (about ${Math.round(refWindow / 4)} hours)` : 'recent candles'
  return activity === 'ACTIVE'
    ? `The model expects the next 15 minutes to move more than this market's own average over ${win}.`
    : `The model expects the next 15 minutes to move less than this market's own average over ${win}.`
}

export default function SimpleView({ data }) {
  const a = data.anchor || {}
  const hs = data.prediction?.horizons || []
  const candles = data.candles || []
  const tr = data.track_record || {}
  const refWindow = data.regime?.reference_window_candles

  const short = hs.find((h) => h.candles === 1) || hs[0]
  const long = [...hs].reverse().find((h) => h.activity) || null
  const prev = candles.length > 1 ? candles[candles.length - 2].close : null
  const chg = prev ? ((a.close / prev - 1) * 100) : 0

  // The two scorecards, in the only form a newcomer needs: right out of how
  // many, and what guessing would have scored instead.
  const dir = tr.direction || {}
  const reg = tr.regime || {}

  return (
    <div className="simple">
      {/* ------------------------------------------------- the two answers */}
      <div className="grid grid-2" style={{ marginBottom: 14 }}>
        <div className="answer answer-move">
          <div className="answer-q">Will the price move more than usual?</div>
          <div className={`answer-a ${short?.activity === 'ACTIVE' ? 'active' : 'quiet'}`}>
            {short?.activity || 'n/a'}
          </div>
          <p className="answer-plain">
            {short?.activity
              ? moveWords(short.activity, refWindow)
              : 'This model has no activity head, so it cannot answer this question.'}
          </p>
          <div className="answer-track">
            <span className="answer-track-n">{pct(short?.activity_confidence, 0)}</span>
            confident, and{' '}
            {reg.n
              ? <>right <strong>{reg.hits} of {reg.n}</strong> times ({pct(reg.accuracy, 0)}) since the dashboard started</>
              : 'no calls have finished yet, so there is no track record to show'}
            .
          </div>
          <div className="answer-base">
            Guessing would get you 50%. This is the model&apos;s strong question.
          </div>
        </div>

        <div className="answer answer-way">
          <div className="answer-q">Which way will the price go?</div>
          <div className={`answer-a ${short?.direction === 'UP' ? 'up' : short?.direction === 'DOWN' ? 'down' : 'neutral'}`}>
            {short?.direction || '—'}
          </div>
          <p className="answer-plain">
            {short ? WAY_PLAIN[short.direction] : 'No prediction yet.'} over the next
            15 minutes, moving to about{' '}
            <strong>{money(short?.projected_close)}</strong>
            {short ? ` (${signed(short.expected_return_pct, 3)}%)` : ''}.
          </p>
          <div className="answer-track">
            <span className="answer-track-n">{pct(short?.direction_confidence, 0)}</span>
            confident
            {dir.n
              ? <>, and right <strong>{dir.hits} of {dir.n}</strong> times ({pct(dir.accuracy, 0)})</>
              : <>, with no finished calls to check it against</>}
            .
          </div>
          <div className="answer-base warn">
            This question is close to a coin flip. Always answering
            &ldquo;no clear direction&rdquo; already scores about 52%. Treat the
            activity answer as the model&apos;s real output and this one as a
            weak second opinion.
          </div>
        </div>
      </div>

      {/* --------------------------------------------------------- chart */}
      <Panel
        title="the price, the model's guess, and every guess before it"
        sub={`last ${candles.length} candles (15 minutes each) + the next 90 minutes`}
        bodyClass="tight"
        style={{ marginBottom: 14 }}
      >
        <PriceChart snapshot={data} height={380} simple />
      </Panel>

      {/* ---------------------------------------------------- further out */}
      <Panel title="further ahead" style={{ marginBottom: 14 }}>
        <p className="dim" style={{ marginTop: 0 }}>
          Everything above is the next 15 minutes. The model also looks up to 90
          minutes ahead — a 15-minute candle move is hard to call, and the
          longer stretches are the ones it reads better.
        </p>
        <table className="tbl">
          <thead>
            <tr>
              <th>how far ahead</th>
              <th>move?</th>
              <th>which way?</th>
              <th>expected price</th>
            </tr>
          </thead>
          <tbody>
            {hs.map((h) => (
              <tr key={h.label}>
                <td className="txt">{h.label}</td>
                <td className="txt" style={{ color: h.activity === 'ACTIVE' ? 'var(--active)' : 'var(--quiet)' }}>
                  {h.activity || <span className="muted">not asked</span>}
                </td>
                <td className="txt" style={{ color: h.direction === 'UP' ? 'var(--up)' : h.direction === 'DOWN' ? 'var(--down)' : 'var(--neutral)' }}>
                  {h.direction}
                </td>
                <td>{money(h.projected_close)} <span className="muted">({signed(h.expected_return_pct, 3)}%)</span></td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="note">
          &ldquo;Not asked&rdquo; means the model was never trained on the move
          question at that distance. It is not a missing number.
        </p>
      </Panel>

      {/* ------------------------------------------------------- the price */}
      <Panel title="today" style={{ marginBottom: 14 }}>
        <div className="simple-facts">
          <div>
            <div className="k">price now</div>
            <div className="v">{money(a.close)}</div>
          </div>
          <div>
            <div className="k">since the last candle</div>
            <div className={`v ${chg >= 0 ? 'up' : 'down'}`}>{signed(chg, 2)}%</div>
          </div>
          <div>
            <div className="k">model expects in 90 minutes</div>
            <div className="v">{long ? money(long.projected_close) : '—'}</div>
          </div>
          <div>
            <div className="k">candles the model read</div>
            <div className="v">{num(a.candles_in_window)}</div>
          </div>
        </div>
      </Panel>

      <div className="panel">
        <div className="panel-body">
          <p className="note warn" style={{ margin: 0 }}>
            <strong>This is a school project, not a trading service.</strong> It
            places no orders and predicts no profit. The move question is right
            roughly six times out of ten and the direction question is barely
            better than guessing, so neither is a reason to buy or sell
            anything. A wrong call is shown next to a right one everywhere in
            this dashboard, on purpose.
          </p>
        </div>
      </div>
    </div>
  )
}
