import { useCallback, useEffect, useRef, useState } from 'react'

// In dev this hits the Vite proxy; once FastAPI serves web/dist it is the same
// origin. Either way the path is relative, so nothing here hard-codes a host.
const API = ''

export async function getJSON(path, signal) {
  const res = await fetch(`${API}${path}`, { signal })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${path}`)
  return res.json()
}

/**
 * Live snapshot, pushed over Server-Sent Events with a polling fallback.
 *
 * SSE is the preferred path: the server re-reads on its own schedule and
 * pushes only when the payload actually changed. But proxies and some
 * browsers drop long-lived connections silently, and a dashboard that stops
 * updating without saying so is worse than one that polls - so if no frame
 * arrives within SILENCE_MS the hook falls back to `GET /api/snapshot`,
 * and the badge in the header says which mode is live.
 *
 * What this hook is NOT responsible for is the record. The server runs the
 * model and stores every projection on its own clock whether or not anything
 * is connected, so a tab that is closed, hidden or frozen behind a throttled
 * timer loses nothing - the curve is unbroken when the user comes back. All
 * the page owes them is not to present the stale frame it was left holding as
 * if it were current, which is what the visibility wake below is for.
 */
const SILENCE_MS = 25000
const POLL_MS = 5000
// Two ways of noticing the same return to a tab (`visibilitychange` and
// `focus`) collapse into one catch-up within this window.
const WAKE_MS = 2000

export function useLiveSnapshot() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('connecting')   // sse | poll | connecting
  const [lastUpdate, setLastUpdate] = useState(null)

  const srcRef = useRef(null)
  const pollRef = useRef(null)
  const timerRef = useRef(null)
  const mounted = useRef(true)
  const wokeAt = useRef(0)

  const accept = useCallback((payload) => {
    if (!mounted.current) return
    setData(payload)
    setLastUpdate(new Date())
    setError(payload && payload.ok === false ? (payload.error || 'no data') : null)
  }, [])

  const startPolling = useCallback(() => {
    if (pollRef.current) return
    setMode('poll')

    const tick = async () => {
      try {
        accept(await getJSON('/api/snapshot'))
      } catch (e) {
        if (mounted.current) setError(String(e.message || e))
      }
    }
    tick()
    pollRef.current = setInterval(tick, POLL_MS)
  }, [accept])

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  // Arm a watchdog: if SSE goes quiet, switch to polling for good.
  const armWatchdog = useCallback(() => {
    if (timerRef.current) clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => {
      if (!mounted.current) return
      if (srcRef.current) {
        srcRef.current.close()
        srcRef.current = null
      }
      startPolling()
    }, SILENCE_MS)
  }, [startPolling])

  useEffect(() => {
    mounted.current = true
    let es
    try {
      es = new EventSource(`${API}/api/stream`)
      srcRef.current = es

      es.addEventListener('snapshot', (ev) => {
        try {
          accept(JSON.parse(ev.data))
          setMode('sse')
          armWatchdog()
        } catch (e) {
          setError(`bad frame: ${e.message}`)
        }
      })
      es.addEventListener('error', () => {
        // EventSource retries on its own; the watchdog decides when to give up.
        armWatchdog()
      })
      es.onopen = () => {
        setMode('sse')
        armWatchdog()
      }
    } catch (e) {
      startPolling()
    }
    armWatchdog()

    return () => {
      mounted.current = false
      if (timerRef.current) clearTimeout(timerRef.current)
      if (srcRef.current) srcRef.current.close()
      stopPolling()
    }
  }, [accept, armWatchdog, startPolling, stopPolling])

  // Being hidden is not a reason to stop, and the connection is deliberately
  // left alone while it happens: timers and stream delivery can be throttled
  // or frozen in a background tab, but the server keeps recording regardless,
  // so the only cost of inactivity is a view that is behind. Coming back is
  // exactly the moment to stop being behind - so ask now, rather than waiting
  // out a timer the browser may have stretched to a minute.
  useEffect(() => {
    const onVisible = () => {
      if (typeof document === 'undefined') return
      if (document.visibilityState !== 'visible') return
      // Coming back to a tab fires `focus` and `visibilitychange` together, and
      // two catch-ups for one return is one more than the server needs.
      if (Date.now() - wokeAt.current < WAKE_MS) return
      wokeAt.current = Date.now()
      getJSON('/api/snapshot').then(accept).catch(() => {
        // A failed catch-up is not worth an error banner: the stream is still
        // armed and its own watchdog will report a genuinely dead server.
      })
      armWatchdog()
    }
    document.addEventListener('visibilitychange', onVisible)
    window.addEventListener('focus', onVisible)
    return () => {
      document.removeEventListener('visibilitychange', onVisible)
      window.removeEventListener('focus', onVisible)
    }
  }, [accept, armWatchdog])

  return { data, error, mode, lastUpdate }
}
