# Simple view, view toggle, recorded-projection curve

| | |
|---|---|
| command | `node web/check.mjs` |
| exit code | `0` |
| started (UTC) | 2026-09-28 17:36:03 |
| finished (UTC) | 2026-09-28 17:36:04 |
| wall clock | 1.1s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

Default simple view + advanced toggle. Recorded projections drawn as a solid 2px violet curve on the price chart in BOTH views (breaks at outages); asserted by name in the default-view check. Snapshot fixture re-captured from the live server so the suite exercises a real history, with a staleness warning. markers-check no longer depends on the fixture happening to be pathological: it rebuilds issue order from the captured calls, derives the collision count, and constructs its own axis.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
real snapshot:
  ok    PriceChart                 2071 chars  (container only)
  ok    PriceChart(simple)         1300 chars  (container only)
  ok    SimpleView                 6045 chars
  ok    VolumeChart                  32 chars  (container only)
  ok    DirectionBars              2664 chars
  ok    ActivityBars               1976 chars
  ok    TrackRecord(regime)         349 chars
  ok    TrackRecord(direction)      432 chars
  ok    ReturnDist                 1124 chars
  ok    MetricsTable               1988 chars
  ok    CallsTable                19702 chars

degenerate payloads (must degrade, not throw):
  ok    PriceChart(no history)     1647 chars
  ok    PriceChart(gapped history)   1899 chars
  ok    SimpleView(no calls)       5969 chars
  ok    SimpleView(no activity)    6087 chars
  ok    SimpleView(err)            4486 chars
  ok    CallsTable(empty)           212 chars
  ok    TrackRecord(empty)          504 chars
  ok    ReturnDist(no bands)        118 chars
  ok    MetricsTable(none)           54 chars
  ok    ActivityBars(none)           56 chars
  ok    App(err)                   5467 chars

default view (must be the simple one):
  ok    opens on the simple view
  ok    the advanced panels are not mounted
  ok    the toggle is offered
  ok    the price chart is present
  ok    the recorded history curve is claimed
  ok    the advanced-only layers are not claimed

all components rendered
```
