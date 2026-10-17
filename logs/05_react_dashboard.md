# Step 5 - React dashboard SSR check

| | |
|---|---|
| command | `node web/check.mjs` |
| exit code | `0` |
| started (UTC) | 2026-09-28 16:21:56 |
| finished (UTC) | 2026-09-28 16:22:00 |
| wall clock | 4.1s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

Bundles web/ssr-check.jsx with esbuild and renders every panel in Node against web/_snapshot.json (a real payload from the shipped ensemble), plus seven degenerate payloads. Catches a panel that would render blank or throw in the browser.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
real snapshot:
  ok    PriceChart                 1647 chars  (container only)
  ok    VolumeChart                  32 chars  (container only)
  ok    DirectionBars              2665 chars
  ok    ActivityBars               1971 chars
  ok    TrackRecord(regime)         504 chars
  ok    TrackRecord(direction)      505 chars
  ok    ReturnDist                 1124 chars
  ok    MetricsTable               1988 chars
  ok    CallsTable                 2775 chars

degenerate payloads (must degrade, not throw):
  ok    PriceChart(no history)     1647 chars
  ok    PriceChart(gapped history)   1899 chars
  ok    CallsTable(empty)           212 chars
  ok    TrackRecord(empty)          504 chars
  ok    ReturnDist(no bands)        118 chars
  ok    MetricsTable(none)           54 chars
  ok    ActivityBars(none)           56 chars
  ok    App(err)                   6990 chars

all components rendered
```
