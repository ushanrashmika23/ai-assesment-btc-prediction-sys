# Pipeline: build the dataset

| | |
|---|---|
| command | `python src/pipeline.py` |
| exit code | `0` |
| started (UTC) | 2026-09-28 14:36:05 |
| finished (UTC) | 2026-09-28 14:37:27 |
| wall clock | 82.5s |
| python | 3.13.12 |
| platform | Windows-11-10.0.26200-SP0 |
| cwd | `E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction` |
| git | `not a git repository` |
| provenance | captured live by this invocation |

Rebuilt after the regime-target fix: ratios are now computed once on the clean candle series (step 2b) instead of iterating to a fixpoint that never converged.

Output below is verbatim stdout+stderr from the command above, with only the
TensorFlow startup banner removed. Reproduce it by running the command shown.

## Output

```

========================================================================
STEP 1/8 - DATA ACQUISITION
========================================================================
Reusing cached raw data: E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\data\raw\btcusdt_15m_raw.csv
(pass --refresh to re-download from Binance)

========================================================================
STEP 2/8 - CLEANING, VALIDATION AND GAP AUDIT
========================================================================
Loading raw candles ...
Cleaning ...
  [clean] dropped 81 rows with unparseable timestamps
  [clean] 317,838 clean candles, 32 contiguous segments
========================================================================
CONTINUITY / GAP AUDIT
========================================================================
candles present            : 317,838
candles expected on grid   : 318,481
missing candles            : 643  (0.2019%)
discontinuities (gaps)     : 31
  of which large (>=3 candles) : 30
largest gap (candles)      : 215
contiguous segments        : 32
first candle (UTC)         : 2017-08-17 04:00:00+00:00
last candle  (UTC)         : 2026-09-16 16:00:00+00:00
------------------------------------------------------------------------
gap detail (all discontinuities):
  2017-09-06 15:45:00+00:00 -> 2017-09-06 23:00:00+00:00   missing=28  [LARGE]
  2017-12-18 12:15:00+00:00 -> 2017-12-18 13:30:00+00:00   missing=4  [LARGE]
  2018-01-04 03:00:00+00:00 -> 2018-01-04 05:00:00+00:00   missing=7  [LARGE]
  2018-02-08 00:15:00+00:00 -> 2018-02-10 06:15:00+00:00   missing=215  [LARGE]
  2018-02-11 04:00:00+00:00 -> 2018-02-11 04:30:00+00:00   missing=1
  2018-06-26 01:45:00+00:00 -> 2018-06-26 12:00:00+00:00   missing=40  [LARGE]
  2018-06-27 12:45:00+00:00 -> 2018-06-27 14:45:00+00:00   missing=7  [LARGE]
  2018-07-04 00:15:00+00:00 -> 2018-07-04 08:00:00+00:00   missing=30  [LARGE]
  2018-10-19 05:45:00+00:00 -> 2018-10-19 09:30:00+00:00   missing=14  [LARGE]
  2018-11-14 01:45:00+00:00 -> 2018-11-14 09:00:00+00:00   missing=28  [LARGE]
  2019-03-12 01:45:00+00:00 -> 2019-03-12 08:00:00+00:00   missing=24  [LARGE]
  2019-05-15 02:45:00+00:00 -> 2019-05-15 13:00:00+00:00   missing=40  [LARGE]
  2019-06-07 21:00:00+00:00 -> 2019-06-07 22:15:00+00:00   missing=4  [LARGE]
  2019-08-15 01:45:00+00:00 -> 2019-08-15 10:00:00+00:00   missing=32  [LARGE]
  2019-11-13 01:45:00+00:00 -> 2019-11-13 04:15:00+00:00   missing=9  [LARGE]
  ... and 16 more
========================================================================
Policy: NO candle is fabricated. Every discontinuity starts a new
        segment; sequences and targets may not cross a segment edge.
========================================================================

Wrote E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\data\processed\btcusdt_15m_clean.csv  (317,838 rows)

NOTE: discontinuities exist. They are handled by segment-aware
      sequence generation and target invalidation - no prices are
      interpolated or forward-filled.

========================================================================
STEP 3/8 - FEATURE ENGINEERING
========================================================================
  [features] 49 features across 11 groups

========================================================================
STEP 2b/8 - MARKET-REGIME (ACTIVITY) TARGETS
========================================================================
  [regime] 4 horizons (15m, 30m, 60m, 90m), reference = trailing 96 candles (24h)
  [regime] forward window = mean |15m return| over the next 1-6 candles; ratio = forward / reference

========================================================================
STEP 4/8 - TARGET ENGINEERING (future returns + activity regime)
========================================================================
  [targets] 6 horizons (15m, 30m, 45m, 60m, 75m, 90m)
  [targets] last 6 rows have no future close by construction
  [targets] 636 additional targets invalidated because they would cross a data gap

========================================================================
STEP 5/8 - WARMUP AND MISSING-VALUE HANDLING
========================================================================
  dropped first 500 warmup rows (indicator convergence)
  4,911 residual feature NaNs across 7 columns
      volume_change_1: 49
      trade_count_change_1: 49
      avg_trade_size_ratio_96: 2,088
      taker_buy_ratio: 49
      taker_buy_ratio_dev_16: 486
      taker_buy_ratio_zscore_96: 2,088
      body_ratio: 102
  rows after feature NaN drop : 315,238
  rows after target NaN drop  : 315,063
  total rows removed          : 2,775
  segment ids recomputed after row drops: 32 -> 56 contiguous runs (holes created by dropped rows are now boundaries too)

========================================================================
STEP 6/8 - SEQUENCE WINDOWS AND CHRONOLOGICAL SPLIT
========================================================================
  candidate windows           : 315,000
  valid windows (no gap, target present): 312,165
  purge/embargo between splits: 70 samples (= lookback 64 + max horizon 6, i.e. no train label or window touches a validation candle)
  train 218,515 | val 46,824 | test 46,686 | purged 140

========================================================================
STEP 7/8 - SCALING AND THRESHOLDS (fitted on TRAIN only)
========================================================================
  scaler fitted on rows [0, 221,413) - every row of every TRAIN window and nothing beyond
  last train label touches   : row 221,418 (target candle + 6 forward candles)
  first validation feature   : row 221,420 (1 row of clearance, purge=70 samples)
  scaler kind                 : standard
  scaled blocks               : X (221413, 49) (46887, 49) (46749, 49), clip=+/-5.0 sigma
  window offsets              : 218,515 / 46,824 / 46,686 samples
  threshold mode              : train_quantile
       15m: [-0.1033%, +0.1087%]  = [-10.33, +10.87] bps
       30m: [-0.1389%, +0.1491%]  = [-13.89, +14.91] bps
       45m: [-0.1634%, +0.1796%]  = [-16.34, +17.96] bps
       60m: [-0.1841%, +0.2034%]  = [-18.41, +20.34] bps
       75m: [-0.2045%, +0.2288%]  = [-20.45, +22.88] bps
       90m: [-0.2220%, +0.2505%]  = [-22.20, +25.05] bps
  regime threshold mode       : median of TRAIN ratio (quantile 0.5)
       15m: ratio > 0.7390 => ACTIVE
       30m: ratio > 0.8090 => ACTIVE
       60m: ratio > 0.8528 => ACTIVE
       90m: ratio > 0.8724 => ACTIVE

========================================================================
STEP 8/8 - WRITING FINAL DELIVERABLES
========================================================================
  btcusdt_15m_dataset.csv          315,063 rows x 72 cols
  X_{train,val,test}.npy        scaled 2D blocks (float32)
  starts_{train,val,test}.npy  window offsets into each block
  y_*.npy, y_dir_*.npy, y_regime_*.npy, timestamps_*.npy
  timestamps_all.npy, segment_ids.npy  (audit row index)
  scaler.pkl                       (StandardScaler)
  target_scaler.pkl                (StandardScaler)
  dataset_metadata.json            (all reported statistics)

========================================================================
FINAL DATASET SUMMARY
========================================================================
rows               : 315,063
features           : 49
return targets     : 6
direction targets  : 6
lookback           : 64 candles (16h)
horizons           : 15m, 30m, 45m, 60m, 75m, 90m
train / val / test : 218,515 / 46,824 / 46,686  (+140 purged)
train period       : 2017-08-23 21:45:00+00:00  ->  2024-01-15 02:00:00+00:00
val period         : 2024-01-15 19:45:00+00:00  ->  2025-05-17 13:30:00+00:00
test period        : 2025-05-18 07:15:00+00:00  ->  2026-09-16 14:30:00+00:00

return distribution (full sample, percent):
                         count  mean_%  std_%   p05_%  p50_%  p95_%  pct_positive
future_return_15m 315,063.0000  0.0016 0.3915 -0.4904 0.0011 0.4943       50.3071
future_return_30m 315,063.0000  0.0032 0.5495 -0.6948 0.0035 0.7013       50.5921
future_return_45m 315,063.0000  0.0047 0.6649 -0.8526 0.0051 0.8587       50.7127
future_return_60m 315,063.0000  0.0062 0.7638 -0.9859 0.0066 0.9988       50.8628
future_return_75m 315,063.0000  0.0077 0.8486 -1.1164 0.0080 1.1271       50.9285
future_return_90m 315,063.0000  0.0092 0.9228 -1.2298 0.0097 1.2394       51.0136

class balance per split (train thresholds applied):
  train:
       15m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
       30m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
       45m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
       60m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
       75m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
       90m  DOWN  30.00%   NEUTRAL  40.00%   UP  30.00%
  val:
       15m  DOWN  28.15%   NEUTRAL  43.53%   UP  28.32%
       30m  DOWN  28.13%   NEUTRAL  43.75%   UP  28.12%
       45m  DOWN  28.20%   NEUTRAL  43.62%   UP  28.18%
       60m  DOWN  28.37%   NEUTRAL  43.14%   UP  28.49%
       75m  DOWN  28.32%   NEUTRAL  43.23%   UP  28.45%
       90m  DOWN  28.48%   NEUTRAL  42.99%   UP  28.53%
  test:
       15m  DOWN  23.69%   NEUTRAL  53.83%   UP  22.49%
       30m  DOWN  24.69%   NEUTRAL  52.21%   UP  23.10%
       45m  DOWN  24.90%   NEUTRAL  51.90%   UP  23.20%
       60m  DOWN  25.29%   NEUTRAL  51.25%   UP  23.46%
       75m  DOWN  25.33%   NEUTRAL  51.48%   UP  23.19%
       90m  DOWN  25.30%   NEUTRAL  51.44%   UP  23.26%

regime balance per split (train median threshold applied) - note how little it moves:
  train:
       15m  QUIET  50.00%   ACTIVE  50.00%
       30m  QUIET  50.00%   ACTIVE  50.00%
       60m  QUIET  50.00%   ACTIVE  50.00%
       90m  QUIET  50.00%   ACTIVE  50.00%
  val:
       15m  QUIET  49.36%   ACTIVE  50.64%
       30m  QUIET  49.31%   ACTIVE  50.69%
       60m  QUIET  49.96%   ACTIVE  50.04%
       90m  QUIET  49.76%   ACTIVE  50.24%
  test:
       15m  QUIET  50.39%   ACTIVE  49.61%
       30m  QUIET  49.75%   ACTIVE  50.25%
       60m  QUIET  49.51%   ACTIVE  50.49%
       90m  QUIET  49.42%   ACTIVE  50.58%
  (the direction balance above slides from 30/40/30 on train to
   ~24/54/22 on test; the regime balance does not, because the
   ratio is normalised by the sample's own trailing activity)

All deliverables in: E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\data\final
E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\src\pipeline.py:454: UserWarning: obj.round has no effect with datetime, timedelta, or period dtypes. Use obj.dt.round(...) instead.
  feature_only.round(8).to_csv(FEATURES_CSV, index=False)
E:\_KDU\Academic\Semester 4\ai\GROUP AS\btc-trend-prediction\src\pipeline.py:455: UserWarning: obj.round has no effect with datetime, timedelta, or period dtypes. Use obj.dt.round(...) instead.
  final_frame.round(8).to_csv(LABELLED_CSV, index=False)
```
