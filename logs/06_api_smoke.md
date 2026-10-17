# API smoke test

| | |
|---|---|
| command | `python src/_smoke_api.py` |
| exit code | `0` |
| started (UTC) | 2026-09-28 16:21:20 |
| finished (UTC) | 2026-09-28 16:21:46 |
| wall clock | 25.7s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

every endpoint, plus the recorded-projection block

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
$ C:\Users\ushan\miniconda3\python.exe E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\src\server.py --port 10235 --no-record
  expecting: ENSEMBLE (inference_bundle.json)

endpoints:
  ok    health.model                           'inference_bundle.json'
  ok    GET /api/snapshot                      200  57 KB
  ok    snapshot.regime.horizons               ['15m', '30m', '60m', '90m']  tau={'15m': 0.739017686756132, '30m': 0.8089737353105876, '60m': 0.8528035829192344, '90m': 0.8723865147803103}
  ok    snapshot.prediction.horizons           6 horizons
  ok    every horizon has direction + regression 6/6
  ok    activity exactly where the regime head exists at ['15m', '30m', '60m', '90m']
  ok    snapshot.prediction.curve              7 points
  ok    snapshot.prediction.naive_curve        7 points
  ok    curve starts at the anchor, not the first horizon 
  ok    curve closes are finite and positive   
  ok    snapshot.prediction_history present    dict
  ok    history.points is a list               6 point(s)
  ok    every history point has an integer time 6 point(s)
  ok    history times ascend                   
  ok    history values are finite where present 6 priced, 0 whitespace
  ok    history points sit on 15m boundaries   6 point(s)
  ok    history may run past the last candle   6 projected point(s) beyond the price history
  ok    history reports its gap rule           gap_candles=4
  ok    snapshot.metrics                       key='ensemble_x3' split='test' n=46686
  ok    metrics keyed to the ensemble          'ensemble_x3'
  ok    snapshot.candles                       240 candles
  ok    model_summary says ENSEMBLE            'ENSEMBLE of 3' members='gru_s42, gru_s43, gru_s44'
  ok    GET /api/candles?limit=20              200  candles=20
  ok    GET /api/prediction                    200  prediction=6
  ok    GET /api/track-record                  200  16829 bytes
  ok    GET /api/metrics                       200  2501 bytes
  ok    GET /api/candles?limit=5 -> 422        (below the endpoint's ge=10 floor)

stream:
  ok    GET /api/stream                        2 frames, ok
  ok    stream frame parses                    57 KB
  ok    stream frame has ok + prediction       

server stopped (code 1)

all checks passed
```
