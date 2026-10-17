# Marker contract check (React chart)

| | |
|---|---|
| command | `node web/markers-check.mjs` |
| exit code | `0` |
| started (UTC) | 2026-09-28 16:21:12 |
| finished (UTC) | 2026-09-28 16:21:12 |
| wall clock | 0.3s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

buildMarkers() output against the payload that crashed setMarkers()

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
markers.js — contract check
  ok    the captured payload is in issue order, not time order
        (4 duplicate(s) and the backwards step absorbed either way)
  ok    the fix does not depend on the server having pre-sorted it
  ok    a call whose target candle has not arrived draws no marker
  ok    the chart accepts the markers built from that payload
  ok    every resolved call is accounted for (merged, not silently dropped)
  ok    same-candle calls from both heads collapse into one marker
  ok    a lone call keeps the marker it always had
  ok    missing / null / NaN / string / unknown-candle times are skipped
  ok    several windows in issue order come out chronological
  ok    empty and null payloads produce no markers rather than throwing

fixture, once its candles arrive: 10 calls -> 6 markers
  1790610300   2/2  below  #26a69a
  1790611200   2/2  below  #26a69a
  1790612100   ●    below  #26a69a
  1790613000   0/2  above  #ef5350
  1790613900   ●    below  #26a69a
  1790614800   2/2  below  #26a69a

all marker checks passed
```
