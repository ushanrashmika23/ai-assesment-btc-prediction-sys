# Projection store check (SQLite)

| | |
|---|---|
| command | `python src/_check_store.py` |
| exit code | `0` |
| started (UTC) | 2026-09-28 16:21:18 |
| finished (UTC) | 2026-09-28 16:21:20 |
| wall clock | 1.5s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

idempotence, resolution, gap breaking and the read-only guarantee

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
temp store: C:\Users\ushan\AppData\Local\Temp\tmp6sctgmt8\check.db

  ok    a window writes 10 rows                              
  ok    re-recording the same window adds nothing            INSERT OR IGNORE
  ok    a later refresh cannot overwrite it                  the first write wins
  ok    a new window adds its own rows                       
  ok    calls per candle are merged, not double-counted      7 candle(s) from 20 calls
  ok    history counts the windows behind it                 n_windows=2
  ok    history reads oldest first                           
  ok    the direction head is what is charted                regime rows store expected_pct=0 and would drag the mean down
  ok    regime rows are excluded from the mean               mean=100.1
  ok    settling fills the realised close                    
  ok    settling again is a no-op                            only outstanding targets are written
  ok    every settled row carries a close                    
  ok    the store invents no points inside an outage         largest step 282 candles, vs 7 within a window
  ok    the chart's copy breaks the line at the outage       1 whitespace point(s) - a straight line across an outage would be a forecast nobody made
  ok    the break sits between two real points               1790615700 < 1790616600 < 1790869500
  ok    the break spans the outage, not one candle           spans 282 candles
  ok    charted times stay strictly ascending                14 point(s), 14 distinct
  ok    --no-record against a missing file creates nothing   
  ok    --no-record against a real file still reads it       
  ok    --no-record cannot write to it                       30 row(s) before, 30 after
  ok    and the read-only mode is SQLite's, not just ours    opened mode=ro; a write would raise, not silently pass
  ok    stats agree with the table                           30 rows, 3 windows, 20 settled

all store checks passed
```
