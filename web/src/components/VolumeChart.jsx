import React, { useEffect, useRef } from 'react'
import { createChart, CrosshairMode, LineStyle } from 'lightweight-charts'
import { C } from '../format.js'

/**
 * Volume, on the same time axis as the price chart above it.
 *
 * A histogram series rather than recharts, so that the two charts can be read
 * down a shared vertical line - a bar chart with its own x-axis would break
 * the alignment that makes a volume pane worth having.
 */
export default function VolumeChart({ candles, height = 110 }) {
  const boxRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)

  useEffect(() => {
    if (!boxRef.current) return undefined
    const chart = createChart(boxRef.current, {
      autoSize: true,
      layout: { background: { type: 'solid', color: C.panel }, textColor: '#9aa4b8', fontSize: 10 },
      grid: { vertLines: { color: C.grid }, horzLines: { visible: false } },
      rightPriceScale: { borderColor: C.grid, scaleMargins: { top: 0.15, bottom: 0 } },
      timeScale: { borderColor: C.grid, timeVisible: true, secondsVisible: false, rightOffset: 8 },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#4c8dff', width: 1, style: LineStyle.Dotted, labelBackgroundColor: '#1f2536' },
        horzLine: { visible: false },
      },
      handleScale: false,
      handleScroll: false,
    })
    seriesRef.current = chart.addHistogramSeries({
      priceFormat: { type: 'volume' },
      priceLineVisible: false,
      lastValueVisible: false,
    })
    chartRef.current = chart
    return () => {
      chart.remove()
      chartRef.current = null
      seriesRef.current = null
    }
  }, [])

  useEffect(() => {
    if (!seriesRef.current || !chartRef.current || !candles?.length) return
    seriesRef.current.setData(
      candles.map((c) => ({
        time: c.time,
        value: c.volume,
        // Colouring by the candle's own direction ties the pane to the price
        // chart above without needing a second axis.
        color: c.close >= c.open ? 'rgba(38,166,154,0.45)' : 'rgba(239,83,80,0.45)',
      })),
    )
    chartRef.current.timeScale().setVisibleLogicalRange({
      from: Math.max(0, candles.length - 110),
      to: candles.length + 12,
    })
  }, [candles])

  return <div ref={boxRef} style={{ height }} />
}
