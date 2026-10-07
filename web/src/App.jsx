import React, { useEffect, useState } from 'react'
import { useLiveSnapshot } from './api.js'
import { ActivityBars, DirectionBars } from './components/ProbabilityBars.jsx'
import CallsTable from './components/CallsTable.jsx'
import MetricsTable from './components/MetricsTable.jsx'
import PriceChart from './components/PriceChart.jsx'
import ReturnDist from './components/ReturnDist.jsx'
import SimpleView from './components/SimpleView.jsx'
import TrackRecord from './components/TrackRecord.jsx'
import VolumeChart from './components/VolumeChart.jsx'
import { Panel, Stat } from './components/Panel.jsx'
import {
  actClass, dirClass, hhmm, hhmmss, money, num, pct, signed, untilNext,
} from './format.js'

/**
 * Which of the two dashboards is showing.
 *
 * Simplified is the default because it is the one that can be read without a
 * briefing: two plain answers and a price chart. The advanced view is the
 * project's real instrument panel and everything in it is still one click
 * away - it is a different depth of the same numbers, not a different site.
 *
 * The choice is remembered per browser so switching to advanced survives a
 * refresh. Storage can throw (private windows, blocked site data) and there is
 * no `window` during the SSR check, so both directions are guarded and a
 * failure just means the default.
 */
const VIEW_KEY = 'btc-dashboard-view'

function readStoredView() {
  try {
    return window.localStorage.getItem(VIEW_KEY) === 'advanced' ? 'advanced' : 'simple'
  } catch {
    return 'simple'
  }
}

function storeView(v) {
  try {
    window.localStorage.setItem(VIEW_KEY, v)
  } catch {
    /* the toggle still works for this page view */
  }
}

function useClock() {
  const [now, setNow] = useState(Date.now())
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(id)
  }, [])
  return now
}

/**
 * `initial` exists so the whole tree can be rendered against a fixed payload
 * in a test (web/ssr-check.jsx) — it is never passed by main.jsx, which is
 * what the live hook and the browser use. Everything else comes from the API.
 */
export default function App({ initial = null }) {
  const conn = useLiveSnapshot()
  const now = useClock()
  const data = initial ?? conn.data
  const { error, lastUpdate } = conn
  const mode = initial ? 'test' : conn.mode
  const [view, setView] = useState(readStoredView)
  const advanced = view === 'advanced'

  const toggleView = () => {
    const next = advanced ? 'simple' : 'advanced'
    setView(next)
    storeView(next)
  }

  if (!data) {
    return (
      <div className="boot" style={{ height: '100vh' }}>
        <div className="spinner" />
        <div>connecting to the API on :8000 …</div>
        <div className="muted" style={{ fontSize: 12, maxWidth: 520, textAlign: 'center' }}>
          Start it with <code>python src/server.py</code>. TensorFlow takes about
          ten seconds to import and the first Binance fetch a few more, so the
          first frame can take a little while.
        </div>
        {error ? <div className="note bad">{error}</div> : null}
      </div>
    )
  }

  const a = data.anchor || {}
  const hs = data.prediction?.horizons || []
  const short = hs.find((h) => h.candles === 1) || hs[0]
  const long = hs.find((h) => h.candles === 6) || hs[hs.length - 1]
  const regimeCall = hs.find((h) => h.activity) || null

  const candles = data.candles || []
  const prev = candles.length > 1 ? candles[candles.length - 2].close : null
  const chg = prev ? ((a.close / prev - 1) * 100) : 0

  const tr = data.track_record || {}
  const live = data.ok !== false

  return (
    <div className="app">
      {/* ------------------------------------------------------- header */}
      <div className="topbar">
        <div className="brand">
          <span className="sym">{data.symbol}</span>
          <span className="tf">{data.interval}</span>
        </div>

        <div>
          <div className="price-big">{money(a.close)}</div>
          <div style={{ fontSize: 12 }} className={chg >= 0 ? 'v up' : 'v down'}>
            {signed(chg, 2)}% on the last candle
          </div>
        </div>

        <div className="chip">
          <span className={`dot ${live ? 'live' : 'err'}`} />
          {mode === 'sse' ? 'live (pushed)' : mode === 'poll' ? 'live (polling)' : 'connecting'}
        </div>

        <div className="chip" title={a.timestamp}>
          candle {a.timestamp ? a.timestamp.slice(11, 19) : '—'} UTC
        </div>

        <div className="chip">
          next candle in {untilNext(data.next_candle_ms, now)}
        </div>

        <div className="spacer" />

        <div className="chip" title={data.model}>
          model {data.model || '—'}
        </div>
        {lastUpdate ? (
          <div className="chip">updated {hhmmss(lastUpdate.getTime() / 1000)}</div>
        ) : null}

        <button
          type="button"
          className="viewtoggle"
          onClick={toggleView}
          aria-pressed={advanced}
          title={advanced
            ? 'Hide the raw probabilities, scorecards and tables'
            : 'Show every chart, probability and table the project produces'}
        >
          {advanced ? '← Simple view' : 'Advanced view →'}
        </button>
      </div>

      {data.ok === false ? (
        <div className="error-box">
          <strong>The last refresh failed.</strong> {data.error}
          <div style={{ marginTop: 8, fontSize: 12 }} className="dim">
            The panel is showing the last good reading. Binance rate limits and
            network drops are the usual causes; the server retries on the next
            tick. Check <code>/api/health</code> or the server console.
          </div>
        </div>
      ) : null}

      {/* The two dashboards. Everything between here and the footer is the
          advanced one; the simple one is a single component that reads the
          same payload. Only one of them is ever mounted. */}
      {!advanced ? <SimpleView data={data} /> : <>
      {/* -------------------------------------------------------- stats */}
      <div className="stats" style={{ marginBottom: 14 }}>
        <Stat
          k="activity call · 15m"
          v={regimeCall ? regimeCall.activity : 'n/a'}
          tone={regimeCall ? actClass(regimeCall.activity) : ''}
          s={regimeCall
            ? `${pct(regimeCall.activity_confidence, 1)} confident — measured right ~60–68% of the time vs a 50% baseline`
            : 'this model has no activity head'}
        />
        <Stat
          k="direction call · 15m"
          v={short ? short.direction : '—'}
          tone={short ? dirClass(short.direction) : ''}
          s={short
            ? `${pct(short.direction_confidence, 1)} confident — close to a coin flip; read the activity line instead`
            : ''}
        />
        <Stat
          k="activity call · 90m"
          v={long && long.activity ? long.activity : 'n/a'}
          tone={long && long.activity ? actClass(long.activity) : ''}
          s={long && long.activity ? `${pct(long.activity_confidence, 1)} confident` : ''}
        />
        <Stat
          k="live activity hit rate"
          v={tr.regime?.accuracy != null ? pct(tr.regime.accuracy, 1) : '—'}
          s={tr.regime?.n
            ? `${tr.regime.hits}/${tr.regime.n} resolved calls · baseline 50%`
            : 'no calls resolved yet'}
        />
        <Stat
          k="tracked candles"
          v={num(a.candles_in_window)}
          s={`model input ${JSON.stringify(a.input_shape)}`}
        />
      </div>

      {/* -------------------------------------------------- the main chart */}
      <Panel
        title="actual price vs predicted curve — same chart, same time axis"
        sub={`${candles.length} candles of history + 6-candle projection`}
        bodyClass="tight"
        style={{ marginBottom: 14 }}
      >
        <PriceChart snapshot={data} height={470} />
      </Panel>

      <Panel title="volume" bodyClass="flush" style={{ marginBottom: 14 }}>
        <VolumeChart candles={candles} height={120} />
      </Panel>

      {/* ------------------------------------------------------- heads */}
      <div className="grid grid-2" style={{ marginBottom: 14 }}>
        <Panel title="direction probabilities" sub="DOWN / NEUTRAL / UP per horizon">
          <DirectionBars horizons={hs} />
          <p className="note">
            The three classes are fixed by thresholds taken from the training
            split, which leaves ~40% of training candles NEUTRAL by
            construction. So always answering NEUTRAL already scores about
            0.52 — any accuracy figure here has to be read against that, not
            against 0.33.
          </p>
        </Panel>

        <Panel title="activity probabilities" sub="QUIET / ACTIVE per horizon">
          <ActivityBars
            horizons={hs}
            tau={data.regime?.tau}
            reference={data.regime?.reference_absmove}
            refWindow={data.regime?.reference_window_candles}
          />
        </Panel>
      </div>

      {/* -------------------------------------------------- scorecards */}
      <div className="grid grid-2" style={{ marginBottom: 14 }}>
        <Panel title="live activity scorecard" sub="calls made before the outcome existed">
          <TrackRecord snapshot={data} head="regime" />
        </Panel>
        <Panel title="live direction scorecard" sub="noisy — small sample">
          <TrackRecord snapshot={data} head="direction" />
        </Panel>
      </div>

      {/* ------------------------------------------------ distribution */}
      <Panel
        title="realised returns vs what the model expects"
        sub="training split distribution, frozen NEUTRAL band, today's prediction"
        style={{ marginBottom: 14 }}
      >
        <ReturnDist snapshot={data} />
      </Panel>

      {/* --------------------------------------------- held-out metrics */}
      <div className="grid grid-2" style={{ marginBottom: 14 }}>
        <Panel title="held-out performance" sub="models/eval_summary.json">
          <MetricsTable snapshot={data} />
        </Panel>
        <Panel title="call log" sub="every call, hits and misses alike">
          <CallsTable snapshot={data} />
        </Panel>
      </div>

      {/* ------------------------------------------------------- footer */}
      <div className="panel">
        <div className="panel-body">
          <div className="legend" style={{ marginBottom: 8 }}>
            <span>generated {data.generated_at}</span>
            <span>status {data.status}</span>
            {data.prediction?.band_note
              ? <span title={data.prediction.band_note}>band = realised train outcomes</span>
              : null}
          </div>
          <p className="note warn" style={{ margin: 0 }}>
            <strong>Research viewer, not a trading system.</strong> It places no
            orders, sizes no positions and models no fees, funding or slippage.
            A directional call is not a profitability claim, and fees alone can
            exceed a 15-minute edge. The projected curve is the regression
            head&apos;s raw output, and that head is measurably worse than
            predicting no change at every horizon — which is exactly why the
            flat baseline is drawn next to it. The live scorecard is a small,
            noisy sample; the held-out test figures carry statistical weight and
            a live edge, if any, would show up in the scorecard over weeks.
            Nothing here is investment advice.
          </p>
        </div>
      </div>
      </>}
    </div>
  )
}
