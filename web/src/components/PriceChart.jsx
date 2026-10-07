import React, { useEffect, useMemo, useRef } from 'react'
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts'
import { C, money, num } from '../format.js'
import { buildMarkers } from '../markers.js'

/**
 * ACTUAL PRICE + PREDICTED CURVE, ON ONE CHART, SHARING ONE TIME AXIS.
 *
 * Four things are drawn together, and each is a different claim:
 *
 *   candles      what the market did.
 *   model curve  what the regression head says the close will be over the
 *                next six candles. A projected path, which is literally what
 *                the model outputs - NOT a forecast you should trade.
 *   naive curve  the same projection with zero return. This is the baseline
 *                the regression head has to beat, and on held-out test data
 *                it does not beat it. Drawing it is the point: seeing the two
 *                nearly coincide is more informative than any caption.
 *   band         the 10/25/50/75/90th percentiles of REALISED returns on the
 *                training split, conditional on the class the model predicts.
 *                A description of the training distribution, not a confidence
 *                interval for this path.
 *   history      every projection this dashboard has ever recorded, one point
 *                per candle, read back from models/predictions.db. The live
 *                curve above is one window; this is all of them, so the two
 *                together show whether the projections have been tracking the
 *                price or only look plausible once. It BREAKS where the
 *                dashboard was not running, because across an outage the model
 *                made no call and a line drawn through it would say otherwise.
 *                Drawn SOLID and 2px in the model's own violet: it is the same
 *                head's output recorded earlier, and the dash pattern is what
 *                separates it from the live projection, not the hue.
 *
 * History and projection live on the same axis so the eye can compare them
 * directly - which is the whole reason to put them in one chart rather than
 * two stacked ones.
 *
 * `simple` drops the two layers that need a briefing to read (the percentile
 * rails and the no-change baseline) and leaves the price, the projection, the
 * recorded history and the markers. The history stays in both views: it is the
 * model's actual track rather than a decoration, and "has its projection been
 * following the price" is a question a newcomer has too. So do the markers - a
 * green dot for a call that landed and a red one for a call that did not is
 * the one part of this chart that explains itself.
 */
export default function PriceChart({ snapshot, height = 460, simple = false }) {
  const boxRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef({})

  // -- create once ---------------------------------------------------------
  useEffect(() => {
    if (!boxRef.current) return undefined

    const chart = createChart(boxRef.current, {
      autoSize: true,
      layout: {
        background: { type: 'solid', color: C.panel },
        textColor: '#9aa4b8',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: C.grid },
        horzLines: { color: C.grid },
      },
      rightPriceScale: {
        borderColor: C.grid,
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      timeScale: {
        borderColor: C.grid,
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 8,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#4c8dff', width: 1, style: LineStyle.Dotted, labelBackgroundColor: '#1f2536' },
        horzLine: { color: '#4c8dff', width: 1, style: LineStyle.Dotted, labelBackgroundColor: '#1f2536' },
      },
      localization: {
        priceFormatter: (p) => p.toLocaleString('en-US', { maximumFractionDigits: 2 }),
      },
    })

    seriesRef.current.candles = chart.addCandlestickSeries({
      upColor: C.up, downColor: C.down,
      borderUpColor: C.up, borderDownColor: C.down,
      wickUpColor: C.up, wickDownColor: C.down,
      priceLineVisible: false,
    })

    // The recorded history is drawn in BOTH views - it is the model's actual
    // track on the price, not a decoration, and "has its projection been
    // tracking the market or did it only look plausible once" is a question a
    // newcomer has too. Solid, because a dotted 1px line is not a curve: the
    // old styling made it nearly invisible against the candles.
    //
    // Same violet as the live projection on purpose - it is the same head's
    // output, recorded earlier - and it is the dash pattern that separates
    // them, not the hue. They also barely share an x-range: history is mostly
    // behind "now" and the projection is entirely ahead of it.
    seriesRef.current.history = chart.addLineSeries({
      color: C.history, lineWidth: 2,
      lineStyle: LineStyle.Solid, priceLineVisible: false,
      lastValueVisible: false, crosshairMarkerVisible: false,
      title: 'recorded projections',
    })

    // Advanced-only layers. In the simple view these are never created - not
    // created and left empty, because an empty series still takes a colour in
    // the legend and a reader would be told the chart is drawing more lines
    // than it is.
    if (!simple) {
      // The rails go in before the model line, because lightweight-charts has
      // no z-order API in v4: a series draws over the ones added before it, so
      // insertion order here is the only thing keeping the projection legible
      // against its own error bars.
      seriesRef.current.bandHi = chart.addLineSeries({
        color: 'rgba(124,92,255,0.55)', lineWidth: 1,
        lineStyle: LineStyle.Dotted, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      })
      seriesRef.current.bandLo = chart.addLineSeries({
        color: 'rgba(124,92,255,0.55)', lineWidth: 1,
        lineStyle: LineStyle.Dotted, priceLineVisible: false,
        lastValueVisible: false, crosshairMarkerVisible: false,
      })
      seriesRef.current.naive = chart.addLineSeries({
        color: C.naive, lineWidth: 2, lineStyle: LineStyle.Dotted,
        priceLineVisible: false, lastValueVisible: false,
        crosshairMarkerVisible: false, title: 'no-change baseline',
      })
    }
    seriesRef.current.model = chart.addLineSeries({
      color: C.model, lineWidth: 3, lineStyle: LineStyle.Dashed,
      priceLineVisible: false, lastValueVisible: true,
      crosshairMarkerRadius: 4, title: 'model projection',
    })

    chartRef.current = chart
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = {}
    }
    // Rebuilt when the view changes: a different set of series has to be
    // created, so the chart is torn down rather than patched. The feed effect
    // below carries the same dep, or the new chart would be left blank.
  }, [simple])

  // -- feed it -------------------------------------------------------------
  const model = useMemo(() => {
    const p = snapshot?.prediction
    if (!p) return null
    const anchorT = snapshot.anchor.timestamp_ms / 1000
    const band = snapshot.prediction.horizons.map((h, i) => {
      const b = h.band
      if (!b) return null
      const q = Object.fromEntries(b.percentiles.map((x) => [String(x.q), x.close]))
      return {
        time: anchorT + h.candles * 900,
        hi: q['0.75'] ?? null,
        lo: q['0.25'] ?? null,
      }
    })
    return {
      model: p.curve.map((pt) => ({ time: pt.time, value: pt.close })),
      naive: p.naive_curve.map((pt) => ({ time: pt.time, value: pt.close })),
      band: band.filter(Boolean),
      nReal: p.curve.length,
    }
  }, [snapshot?.generated_at])

  useEffect(() => {
    const s = seriesRef.current
    if (!chartRef.current || !s.candles || !model || !snapshot) return

    s.candles.setData(snapshot.candles || [])
    s.model.setData(model.model)

    // Points carrying only a time are the gaps: passing `value: undefined`
    // would be a different thing to the chart than a whitespace point, and the
    // line would bridge an outage instead of breaking at it.
    s.history.setData((snapshot.prediction_history?.points || []).map(
      (p) => (p.value === undefined || p.value === null
        ? { time: p.time }
        : { time: p.time, value: p.value })))

    // Absent in the simple view, so the advanced series are fed behind a guard
    // rather than assuming the create effect built them.
    if (s.naive) s.naive.setData(model.naive)
    if (s.bandHi && s.bandLo) {
      const anchorT = snapshot.anchor.timestamp_ms / 1000
      s.bandHi.setData([{ time: anchorT, value: snapshot.anchor.close },
                        ...model.band.map((b) => ({ time: b.time, value: b.hi }))])
      s.bandLo.setData([{ time: anchorT, value: snapshot.anchor.close },
                        ...model.band.map((b) => ({ time: b.time, value: b.lo }))])
    }

    // Mark where the model's own recorded calls resolved. Green = the class it
    // named is the class that happened, red = it was not.
    //
    // buildMarkers() sorts, merges same-candle calls and drops unplaceable
    // times, because setMarkers() asserts on anything else - see
    // web/src/markers.js for why the payload cannot be fed in as it arrives.
    s.candles.setMarkers(buildMarkers(snapshot.track_record?.recent,
                                      snapshot.candles))

    // Show the recent past AND the whole projection: the comparison between
    // them is the point of the chart.
    const total = (snapshot.candles || []).length
    if (total > 0) {
      chartRef.current.timeScale().setVisibleLogicalRange({
        from: Math.max(0, total - 110),
        to: total + model.nReal + 6,
      })
    }
  }, [model, snapshot, simple])

  return (
    <div>
      <div ref={boxRef} style={{ height }} />
      {simple ? (
        <div className="chart-note">
          <span className="legend" style={{ marginRight: 12 }}>
            <span><span className="swatch" style={{ borderTopColor: C.up, borderTopStyle: 'solid' }} />what the price actually did</span>
            <span><span className="swatch" style={{ borderTopColor: C.model, borderTopStyle: 'dashed' }} />what the model expects next</span>
            <span><span className="swatch" style={{ borderTopColor: C.history, borderTopStyle: 'solid' }} />what the model predicted earlier</span>
          </span>
          <div style={{ marginTop: 6 }}>
            The dots are calls the model already made. <strong>Green</strong> means
            it was right, <strong>red</strong> means it was wrong.
          </div>
          <div style={{ marginTop: 6 }}>
            The solid purple line is every prediction the model made while this
            dashboard was running, one point per 15 minutes, so you can see
            whether they have been following the price or only looked good once.
            {snapshot?.prediction_history?.n_points
              ? ` ${num(snapshot.prediction_history.n_points)} points so far, from ${num(snapshot.prediction_history.n_windows)} prediction${snapshot.prediction_history.n_windows === 1 ? '' : 's'}.`
              : ' It is empty until the dashboard has been running for a while.'}
            {' '}Where the line breaks, the dashboard was not running — no
            prediction was made for that stretch, so none is drawn.
          </div>
          <div style={{ marginTop: 6 }}>
            The candlesticks are real prices. The dashed purple line is the
            model&apos;s guess for the next 90 minutes, and a guess is all it is
            — it places no orders and predicts no profit.
          </div>
        </div>
      ) : (
      <div className="chart-note">
        <span className="legend" style={{ marginRight: 12 }}>
          <span><span className="swatch" style={{ borderTopColor: C.up, borderTopStyle: 'solid' }} />actual price</span>
          <span><span className="swatch" style={{ borderTopColor: C.model, borderTopStyle: 'dashed' }} />model projection</span>
          <span><span className="swatch" style={{ borderTopColor: C.naive, borderTopStyle: 'dotted' }} />no-change baseline</span>
          <span><span className="swatch" style={{ borderTopColor: 'rgba(124,92,255,0.55)', borderTopStyle: 'dotted' }} />realised-outcome range (train)</span>
          <span><span className="swatch" style={{ borderTopColor: C.history, borderTopStyle: 'solid' }} />recorded projections</span>
        </span>
        {snapshot?.prediction_history?.n_points ? (
          <div style={{ marginTop: 6, color: '#8b93a7' }} title={snapshot.prediction_history.note}>
            Recorded projections: {num(snapshot.prediction_history.n_points)} points
            across {num(snapshot.prediction_history.n_windows)} windows
            {snapshot.prediction_history.n_scored
              ? `, ${num(snapshot.prediction_history.n_scored)} with a realised close` : ''}
            {' '}— the line breaks where the dashboard was not running.
          </div>
        ) : null}
        {snapshot?.prediction?.caveat ? (
          <div style={{ marginTop: 6, color: '#d8c69a' }}>{snapshot.prediction.caveat}</div>
        ) : null}
        <div style={{ marginTop: 4 }}>
          Projection points:&nbsp;
          {(snapshot?.prediction?.horizons || []).map((h) => (
            <span key={h.label} style={{ marginRight: 10 }}>
              {h.label} {money(h.projected_close)} ({h.expected_return_pct >= 0 ? '+' : ''}
              {h.expected_return_pct.toFixed(3)}%)
            </span>
          ))}
        </div>
      </div>
      )}
    </div>
  )
}
