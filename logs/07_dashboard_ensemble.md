# Step 7 - Terminal dashboard on the ensemble

| | |
|---|---|
| command | `python src/dashboard.py --once` |
| exit code | `0` |
| started (UTC) | 2026-09-28 15:44:01 |
| finished (UTC) | 2026-09-28 15:44:13 |
| wall clock | 12.0s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `e7cf297` |
| provenance | captured live by this invocation |

python src/dashboard.py --once with no --model, so it takes the shipped bundle and builds Ensemble([3 members]). This is the default path for both the terminal UI and the API; before this run only the single-.keras path had ever been executed.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```
  [ok]   REST top-up  664 candles
╭─────────────────────── BTCUSDT 15m - live trend read ────────────────────────╮
│  last candle  2026-09-28 15:30 UTC              close  83,194.00             │
│       change  +0.15%                candles in window  568                   │
│ next refresh  30s                    confidence floor  0.00                  │
╰──────────────────────── model inference_bundle.json ─────────────────────────╯
╭─────────────────────────────────── PRICE ────────────────────────────────────╮
│ last 16h  ▇▇█▆▆▅▆▄▅▄▄▃▄▃▃▄▄▄▄▃▂▃▃▂▃▃▂▂▂▂▂▂▂▂▂▁▁▁▂▂▂▂▂▃▃▃▄▃▄▄▃▃▂▂▂▃           │
│    range  82,626.01 - 84,792.47   (2.62% wide)   net -1.51%                  │
╰──────────────────────────────────────────────────────────────────────────────╯
╭────────────────────────── PREDICTION - 6 horizons ───────────────────────────╮
│ horizon   call         conf                       activity         expected  │
│ 15m       v DOWN      38.7%  ███████░░░░░░░░░░░   ACTIVE 71%        -0.112%  │
│ 30m       v DOWN      39.8%  ███████░░░░░░░░░░░   ACTIVE 78%        -0.183%  │
│ 45m       v DOWN      41.8%  ████████░░░░░░░░░░   -                 -0.183%  │
│ 60m       v DOWN      41.4%  ███████░░░░░░░░░░░   ACTIVE 84%        -0.196%  │
│ 75m       v DOWN      42.0%  ████████░░░░░░░░░░   -                 -0.204%  │
│ 90m       v DOWN      42.9%  ████████░░░░░░░░░░   ACTIVE 87%        -0.219%  │
│ trend label is argmax of the direction head; expected is the regression      │
│ head's percent return; activity is the regime head                           │
│ activity = will the next window be busier than this sample's own trailing    │
│ 24h?  (reference now 0.130% per 15m candle; classes are 50/50 by             │
│ construction, so 0.50 is the baseline)                                       │
│ MEASURED: the regression head is WORSE than predicting the mean at every     │
│ horizon - read 'expected' as an output, not a forecast. The direction head   │
│ is the part that carries the signal.                                         │
╰──────────────────────────────────────────────────────────────────────────────╯
╭─────────────────────────────── LIVE SCORECARD ───────────────────────────────╮
│ calls issued  10 (0 window(s))                                               │
│     resolved  0 - the first calls resolve once their horizon elapses         │
╰─────────────────────────── measured, not asserted ───────────────────────────╯
```
