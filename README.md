# BTCUSDT 15m Short-Term Trend Prediction

A leakage-audited data pipeline that turns the full public Binance BTCUSDT
15-minute history into a Colab-ready dataset for predicting **future percentage
return** (15/30/45/60/75/90 minutes ahead) with a **DOWN / NEUTRAL / UP**
direction label as an auxiliary target.

The primary model is a **GRU** with a shared recurrent trunk and **three
heads** — the return, the direction label, and a **market-regime head** that
answers a question with an honest baseline (see [Trained
results](#trained-results--including-the-parts-that-did-not-work)). It is
shipped as a **3-seed ensemble**, averaged rather than picked. An **LSTM** and
**naive baselines** are trained alongside it for comparison. Training runs
headless (`python src/train.py`), in Google Colab
(`notebooks/04_colab_training.ipynb`), or locally
(`notebooks/05_local_training.ipynb`).

A trained model can then be watched live: `python src/dashboard.py` opens an
interactive terminal dashboard that pulls the newest candles, runs the same
frozen scaler and network, and shows the six-horizon read refreshing on a timer
— alongside a scorecard of how its own past calls actually turned out.

> ### What this project does *not* claim
>
> **No profitability or trading-edge claim is made anywhere in this
> repository.** Every number in this README is a *measurement* on one historical
> test period, reported next to the naive baseline it has to beat, and none of
> them is a promise. Nothing here is investment advice, and there is no backtest
> — fees, slippage, funding and regime change all sit between a directional call
> and a P&L. Directional accuracy above 33% on a three-class problem is not
> evidence of a profitable strategy; read the macro-F1 and the per-class
> recalls, not the headline accuracy.
>
> The trained numbers are in [Trained results](#trained-results--including-the-part-that-did-not-work)
> below — **including the regression head, which measurably fails to beat
> predicting the mean.**

---

## Results so far: the dataset, not the model

The pipeline has been run end to end on real Binance data. These are measured
facts about the **data**, not about any model:

| | |
|---|---|
| Raw candles downloaded | 317,919 (2017-08-17 → 2026-09-16) |
| Clean candles after validation | 317,838 |
| Missing candles on the 15m grid | 643 (0.2019%) across 31 gaps, 32 contiguous segments |
| Dataset rows | 315,063 × 63 columns |
| Model features | 49, in 11 groups |
| Lookback | 64 candles (16 hours) |
| Valid sequence windows | 312,165 (312,025 after the 140-sample purge) |
| Train / Val / Test samples | 218,515 / 46,824 / 46,686 (+140 purged) |
| Train period | 2017-08-23 → 2024-01-15 |
| Validation period | 2024-01-15 → 2025-05-17 |
| Test period | 2025-05-18 → 2026-09-16 |
| Automated leakage checks | **24 / 24 passing** |

Every figure above is reproducible: `python src/pipeline.py` prints them, and
`python src/audit_leakage.py` re-derives them from the files on disk.

---

## Trained results — including the parts that did not work

Produced by `python src/train.py --arch gru --seeds 42,43,44 --epochs 25
--patience 5 --batch 1024 --tag gru`, on the full 218,515-sample training
split, on a CPU with no GPU (TensorFlow 2.21.0, Keras 3.14.0). 80–85 s per
epoch, 53 minutes wall clock, three seeds. Full transcripts:
`logs/02_train.md` and `logs/03_evaluate.md`.

The three seeds are **averaged** (mean of the per-member softmax outputs)
rather than one being picked, so no single winning `.keras` is written. The
shipped artefact is the three-member `models/inference_bundle.json`, and the
seed-to-seed spread is the reason: the differences between seeds are within
noise, so choosing one would be choosing noise.

Everything below is the 46,686-sample held-out test set (2025-05-18 →
2026-09-16), read **once** by `src/evaluate.py`. Nothing in these tables was
used to select anything.

### The regime head is the headline

The question it answers: **will the mean absolute move over the next h candles
be larger than this sample's own trailing 24-hour average?** The threshold is
the training median, so QUIET and ACTIVE are ~50% of the set by construction
and **0.50 is the baseline** — there is no majority class to hide behind.

| Horizon | Baseline | Model | Lift |
|---|---|---|---|
| 15m | 0.5039 | **0.6004** | +0.0964 |
| 30m | 0.4975 | **0.6400** | +0.1425 |
| 60m | 0.4951 | **0.6708** | +0.1757 |
| 90m | 0.4942 | **0.6799** | +0.1856 |
| **mean** | 0.4977 | **0.6477** | **+0.1501** |

Both classes are predicted competently rather than one being defaulted to:
at 90m, QUIET precision/recall is 0.71/0.60 and ACTIVE is 0.66/0.76. This is
the number this project is willing to put its name to.

It is **not a trading signal**. "This window will be busier than the last 24
hours" says nothing about direction. There is no position, no backtest, no
fees and no slippage here.

### The direction head has measurable skill — but it is weak

Macro-F1 on the test set (higher is better):

| Horizon | Majority class | Ensemble | |
|---|---|---|---|
| 15m | 0.2333 | **0.4279** | ×1.83 |
| 30m | 0.2287 | **0.4276** | ×1.87 |
| 45m | 0.2278 | **0.4250** | ×1.87 |
| 60m | 0.2259 | **0.4194** | ×1.86 |
| 75m | 0.2266 | **0.4211** | ×1.86 |
| 90m | 0.2264 | **0.4212** | ×1.86 |

It beats the majority-class baseline at every horizon, by a wide margin.

**But the model is heavily NEUTRAL-biased.** Per-class recall at 15m: NEUTRAL
0.805, DOWN 0.242, UP 0.241. It catches a directional move roughly one time in
four. DOWN and UP precision sit near 0.35–0.36. Plain accuracy is barely better
than the always-NEUTRAL baseline (0.5450 vs 0.5383 at 15m) — which is exactly
why macro-F1, not accuracy, is the metric reported here. Accuracy on a
three-class problem with one dominant class flatters the model; macro-F1 does
not let it.

### The regression head does not work

MAE in percent return (lower is better). `naive_mean` is "always predict the
mean":

| Horizon | naive mean | Ensemble | |
|---|---|---|---|
| 15m | **0.145945** | 0.146489 | worse |
| 30m | **0.205102** | 0.207241 | worse |
| 45m | **0.249220** | 0.252593 | worse |
| 60m | **0.287304** | 0.292507 | worse |
| 75m | **0.321510** | 0.327982 | worse |
| 90m | **0.352292** | 0.361520 | worse |

**It is worse than predicting the mean at every single horizon.** The
correlation between predicted and actual return is 0.012 at 15m and goes
*negative* (−0.005) at 90m. This is reported rather than hidden: the regression
head was trained as an auxiliary task to shape the shared trunk, it does not
produce a usable forecast, and both the terminal dashboard and the React one
label its `expected` column accordingly — and draw it next to the flat
zero-return baseline so the two can be compared directly.

### What this does and does not establish

It establishes that 16 hours of 15m candles contain *some* information about
both the direction and the volatility of the next 15–90 minutes that a
recurrent network can extract, on one test period, under a leakage-audited
split. The volatility question is answered well; the direction question is
answered poorly but better than chance.

It does not establish profitability. The test period is a single regime, the
directional recall is low, and no fees, slippage or funding are modelled
anywhere. A macro-F1 of 0.42 on a three-class problem, and a volatility
classifier that is right 65% of the time, are a long way from a trading edge.

---

## Evaluating it yourself

```bash
python src/evaluate.py                       # the whole report, val + test
python src/evaluate.py --split val           # validation only
python src/evaluate.py --model models/gru_s42_best.keras
python src/evaluate.py --all                 # every model, compared
python src/evaluate.py --at 2025-08-01       # test ONE moment you choose
python src/evaluate.py --sample 20 --split test --seed 42
```

`evaluate.py` loads a saved `.keras` file and the frozen artefacts and
re-measures from scratch — nothing is refitted, nothing is retrained, and it
**selects nothing**. That separation is deliberate: model choice belongs to
validation macro-F1 inside `train.py`. Run it, change the model because of what
you saw, run it again, and the test split has quietly become a second
validation split. The leak would be in that loop, not in the code.

It reports, per split:

- **Regression** — MAE / RMSE / bias / correlation per horizon, each beside the
  naive "predict the mean" baseline and the delta.
- **Direction** — accuracy and macro-F1 per horizon beside the majority-class
  baseline, then per-class precision / recall / F1.
- **Confusion matrices** — 3×3 counts with row percentages, one per horizon, so
  recall reads straight off the diagonal.
- **Calibration** — every prediction bucketed by the model's own top
  probability, with the realised accuracy in that bucket.
- **Confidence-floor sweep** — what the dashboard's `c` key actually buys you.

### Two results worth reading

**The confidence floor is a straight loss.** The dashboard lets you raise a
floor: calls below it display as NEUTRAL. It sounds prudent. Measured, mean
over the six horizons:

| Floor | Test macro-F1 | Test accuracy | Commits to a direction |
|---|---|---|---|
| 0.00 | **0.4237** | 0.5282 | 33.2% |
| 0.38 | 0.3960 | 0.5337 | 24.0% |
| 0.50 | 0.2469 | 0.5234 | 1.6% |
| 0.70 | 0.2283 | 0.5202 | 0.0% |
| *always-NEUTRAL baseline* | *0.2281* | *0.5202* | *0%* |

Macro-F1 falls monotonically on **both** splits and accuracy barely moves. The
reason: downgrading a weak DOWN or UP call removes exactly the scarce thing —
DOWN and UP recall are already the bottleneck (0.22–0.29 on test), while
NEUTRAL is already over-predicted. So the `c` key is documented as a control
you can test, not a feature that helps, and the dashboard prints that measured
caveat whenever the floor is non-zero.

This does *not* contradict the calibration table. Calibration says the
probability is informative *within* the model's own argmax calls (test accuracy
rises 0.37 → 0.83 across confidence buckets). The floor *overrides* those
argmax calls, and the override is what costs. Both are measurements of
different things.

### The chart

```bash
python src/plot_trend.py                        # test split, 15m, last 400 candles
python src/plot_trend.py --horizon 90m --last 800
python src/plot_trend.py --from 2026-01-01 --to 2026-03-01
```

Writes `charts/actual_vs_predicted.png`. The top block is the actual-vs-predicted
trend view: the price, then **the ACTUAL direction class and the PREDICTED one as
colour bands on the same time axis**. Reading down a column tells you what the
market did and what the model said at that instant:

```
   price   ╱╲___╱╲___
   ACTUAL  ██▓██▓▓███▓█▓██▓▓█     ← mostly ▲/▼ stripes
   PREDICT  ▓▓▓▓▓▓█▓▓▓▓▓▓▓▓▓▓▓     ← mostly grey NEUTRAL
```

That visual gap *is* the NEUTRAL bias — NEUTRAL recall 0.86 against DOWN 0.19
and UP 0.21 — which a table makes you squint at and a band makes obvious.

Three panels underneath quantify the match instead of asserting it: predicted
vs actual return with the identity line (the regression head collapses to a
vertical smear at x≈0, exactly as its r = 0.023 implies), a reliability curve
(does "70%" mean 70%?), and rolling accuracy over time against the
always-NEUTRAL baseline — which shows the skill is persistent rather than one
lucky regime.

The bands are plotted from the recorded labels, which use the **frozen
train-only thresholds**, so class boundaries do not move between splits. The
price line is broken at data gaps rather than drawn across them.

Two honest limits: the bottom panels always use the whole split, while the
bands show a window — past ~1,200 candles the bands degrade into noise and the
script says so. And the chart is a measurement on one period, not a result.

**`--at` is the honest way to watch it work.** Give it a date and it finds the
nearest window, prints the six-horizon call with per-horizon probabilities, and
shows what actually happened next:

```
 window ends  2025-08-01 11:45 UTC   (last of 64 candles = 16h of history)
 split        TEST   (sample #7,219)
  horizon  model says          conf    predicted     actual   actual class
      15m  NEUTRAL  ok        43.4%      -0.037%    -0.044%   NEUTRAL
           NEUTRAL 0.43  DOWN 0.37  UP 0.20
      30m  NEUTRAL  MISS      42.9%      -0.060%    +0.331%   UP
 direction: 2/6 horizons correct (chance-level would be about 3/6 by
            always saying NEUTRAL)
```

It searches all splits and tells you which one the date landed in — if it lands
in **train**, it says so and warns that a correct call there is memorisation,
not evidence. The probabilities are printed per horizon so you can see whether
a call was close or a coin-flip; the softmax bug above was caught exactly this
way, by noticing the three numbers did not sum to 1.

---

## Quick start

```bash
pip install -r requirements.txt

# 1. download ~9 years of 15m candles (~2 minutes, no API key needed)
python src/data_download.py

# 2. build the full dataset: clean -> features -> targets -> split -> scale
python src/pipeline.py

# 3. prove no future information leaked in
python src/audit_leakage.py          # must print 24/24 and exit 0

# 4. train (needs TensorFlow; step 4a tells you what it will cost)
python src/train.py --probe          # time one epoch on this machine, save nothing
python src/train.py --arch gru --seeds 42,43,44   # the reported run (~53 min on CPU)

# 5. evaluate the trained model: validation + test metrics, confusion matrices
python src/evaluate.py               # the whole report
python src/evaluate.py --at 2025-08-01        # test ONE moment you choose

# 6. chart actual vs predicted trend  -> charts/actual_vs_predicted.png
python src/plot_trend.py             # needs matplotlib

# 7. watch the trained model live (interactive TUI, needs a model in models/)
python src/dashboard.py

# 8. the same thing in a browser, with charts (needs Node 18+)
pip install fastapi uvicorn          # optional - only this needs them
cd web && npm install && npm run build && cd ..
python src/server.py --serve-web     # http://127.0.0.1:8000
```

Step 4 can instead be `notebooks/04_colab_training.ipynb` (upload `data/final/`
first) or `notebooks/05_local_training.ipynb`.

Any step can be recorded verbatim by prefixing it with
`python src/runlog.py --name <slug> --`. The transcripts in `logs/` are exactly
that, and the numbers quoted in this README come from them rather than from a
transcription of a terminal.

**The data pipeline never needs TensorFlow or PyTorch.** Everything under `src/`
is pure `numpy` + `pandas` + `scikit-learn`. Only `src/train.py`,
`src/dashboard.py`, `src/server.py`, `src/inference.py` and the training
notebooks import a deep-learning framework, and they do it lazily so the
pipeline stays clean.

---

## Repository layout

```
src/
  config.py            single source of truth for every tunable knob
  data_download.py     Binance Vision bulk archives + REST fallback
  preprocessing.py     cleaning, validation, gap detection, segmentation
  features.py          49 backward-looking features in 11 documented groups
  targets.py           future returns, gap-safe invalidation, thresholds
  sequences.py         zero-copy windowing, chronological split, batcher
  pipeline.py          the 8-step orchestrator (run this)
  audit_leakage.py     24 automated leakage checks (run this)
  inference.py         latest-candle trend read (TF-free input path)
  regime.py            the QUIET/ACTIVE target and its train-only threshold
  train.py             headless training: --probe, --arch gru|lstm|both, --seeds
  evaluate.py          validation + test metrics, confusion, calibration, --at
  plot_trend.py        actual-vs-predicted charts -> PNG (matplotlib)
  dashboard.py         interactive live terminal dashboard (rich)
  server.py            the React dashboard's API (FastAPI + SSE)
  predstore.py         every projection the dashboard makes, in SQLite
  runlog.py            run a command, save its verbatim output to logs/*.md

  _check_names.py      internal: flag undefined globals (no pyflakes needed)
  _smoke_api.py        internal: start the API, hit every endpoint, assert

notebooks/
  01_download_data.ipynb        acquire and sanity-check the raw data
  02_feature_engineering.ipynb  gap audit + the 49 features
  03_targets_and_sequences.ipynb targets, thresholds, window geometry
  04_colab_training.ipynb       COLAB ONLY - the reference training run
  05_local_training.ipynb       same models, runnable locally (CPU-friendly)

web/                   the React dashboard: React 18 + Vite + lightweight-charts
  ssr-check.jsx          every panel rendered against a real payload (npm run check)
  markers-check.mjs      the marker ordering contract (npm run check:markers)
  src/App.jsx            the two views and the toggle between them
  src/markers.js         call markers -> what setMarkers() will accept
  src/components/SimpleView.jsx  the default view: plain answers, plain words
data/                  raw -> processed -> final (see PROJECT_USAGE.txt)
models/                trained .keras files + metrics + per-model track records
                       + predictions.db (local, gitignored: the projection store)
charts/                rendered PNGs from src/plot_trend.py
logs/                  verbatim terminal transcripts of the reported runs
```

---

## How leakage is prevented

Six mechanisms, all verified automatically by `src/audit_leakage.py`:

1. **Every feature is strictly backward-looking.** Verified by corrupting all
   rows after a cut point and asserting the features before it are bit-for-bit
   unchanged (measured deviation: exactly `0.000e+00`).
2. **Targets are the only place future data is touched**, and a target that
   would span a data gap is set to `NaN` rather than measured across missing
   candles.
3. **Missing candles are never fabricated.** No interpolation, no forward-fill.
   Every discontinuity starts a new segment, and neither a window nor a target
   may cross one. Segment ids are recomputed *after* row drops, so a hole
   created by dropping a bad row becomes a boundary too.
4. **Splits are strictly chronological** — 70/15/15 in time order, never
   shuffled.
5. **A purge/embargo of 70 samples** (`lookback + max horizon`) separates the
   splits, so no training label is resolved from any candle inside the
   validation or test feature windows.
6. **The scaler and the direction thresholds are fitted on the TRAIN split
   only** and then frozen.

Two subtleties that are easy to get wrong and are handled here explicitly:

- **`X_*.npy` are contiguous blocks, not one row per sample.** Gap-excluded
  windows leave *holes* in the window-start index, so a window is
  `X[starts[k] : starts[k] + LOOKBACK]`. Slicing `X[k : k + LOOKBACK]` would
  silently splice across a hole and fabricate a window spanning missing market
  data. (On this dataset the training split contains 43 such holes.)
- **The purge must exceed `lookback - 1 + max horizon`**, not merely
  `max horizon`: a sample's *label* reaches far past its own window.

Full detail, including what each of the 24 checks proves, is in
`PROJECT_USAGE.txt`.

---

## Configuration

Everything tunable lives in `src/config.py`: symbol, interval, feature windows,
`LOOKBACK`, `HORIZONS`, split fractions, `PURGE_SAMPLES`, `SCALER_KIND`,
`CLIP_SIGMA`, `THRESHOLD_MODE`, `NEUTRAL_QUANTILE` and `RANDOM_SEED`. Nothing
else hard-codes a constant.

---

## Training

Three ways to run the same models, with the same metrics and the same honest
reporting:

| | `src/train.py` | `04_colab_training` | `05_local_training` |
|---|---|---|---|
| Runs | any terminal | Colab only (hard guard) | anywhere TensorFlow imports |
| Data | the repo's `data/final/` | `data/final/` uploaded to Drive | the repo's `data/final/` |
| Output | `models/` | Drive `trained/` | the repo's `models/` |
| Defaults | full 218,515 samples, GRU | full 218,515 samples, GPU | tail-subset, 25 epochs |
| Status | **the reference configuration** | Colab equivalent | iteration and inspection |

`src/train.py` is the reference path now: it is scriptable, repeatable, and it
is what writes the `.keras` files the dashboard loads.

```bash
python src/train.py --probe              # time ONE epoch here, save nothing
python src/train.py --arch gru           # single GRU, full split, early stopping
python src/train.py --arch gru --seeds 42,43,44   # the reported run: 3 seeds, averaged
python src/train.py --arch both          # GRU and LSTM, then pick on validation
python src/train.py --subset 60000 --epochs 15   # fast iteration
```

Three things it deliberately does:

- **`--subset` takes the most recent N training samples**, never a random N. A
  random subset would break the chronological contract the whole project rests
  on. A subset run prints a warning and is not a reportable configuration.
- **Architecture selection uses validation macro-F1 only.** The test split is
  loaded once, at the very end.
- **Several seeds are averaged, not compared.** `--seeds 42,43,44` writes one
  `.keras` per seed and a bundle naming all three; at inference the members'
  softmax outputs are averaged. Picking the best seed on validation would be
  selecting on noise — the seed-to-seed spread is larger than the gap that
  would be used to pick. `--no-ensemble` forces the old single-model behaviour.

Notebook 05 defaults to a tail subset so a first run finishes while you watch
it. That proves the pipeline works; it is **not** a result to report. Set
`SUBSET_TRAIN = None` for the full split.

Results across the three will not match exactly even with identical seeds —
different device, different kernel selection, different reduction order.

### A bug worth knowing about

An earlier revision of these models applied `softmax` to the flat 18-vector and
reshaped to `(6, 3)` afterwards. That shares one normaliser across all six
horizons, so each horizon's three probabilities summed to ≈1/6 rather than 1,
and the head could not express "UP at 15m, DOWN at 90m" independently. Argmax
survived it (softmax is monotonic within a group), which is exactly what made it
easy to miss — it was caught only by noticing that displayed probabilities did
not sum to 1.

The order is now reshape-then-softmax, and `build_multi_task_model()` asserts
the normalisation after building, as does notebook 05. Shapes alone would not
have caught it; the assertion is on the *values*.

---

## The live dashboard

```bash
python src/dashboard.py                      # the bundle's 3-seed ensemble
python src/dashboard.py --interval 60        # slower refresh
python src/dashboard.py --model models/gru_s42_best.keras   # one seed instead
python src/dashboard.py --once               # one frame, no UI, writes nothing
```

It fetches the newest BTCUSDT 15m candles, pushes them through the *same*
cleaning → features → frozen-scaler path the model was trained on, and renders
the six-horizon read. Every 15m candle it issues a fresh set of calls, so the
view refreshes with the market rather than re-predicting on a stale window.

```
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

Every panel above is verbatim output - `logs/07_dashboard_ensemble.md`
is this exact frame, captured by `src/runlog.py`.

| Key | Action |
|---|---|
| `r` | refresh now |
| `i` | cycle the refresh interval (15/30/60/120/300s) |
| `c` | cycle the confidence floor — calls below it display as NEUTRAL (measured to *hurt*; see [Evaluating it yourself](#evaluating-it-yourself)) |
| `m` | cycle the models in `models/` |
| `l` | toggle the log panel |
| `x` | clear this model's track record |
| `q` | quit |

### It keeps score on itself

The panel that matters is the scorecard. Every call is written to
`models/dashboard_log_<model>.json` with the wall-clock time at which it can be
checked; once that candle prints, the dashboard looks up what the price
*actually* did, classifies it with the **frozen train-only thresholds**, and
marks the call right or wrong.

So the dashboard reports its own measured live hit rate next to the
majority-class baseline from the frozen metadata — rather than asserting the
model works. One track record per model, because blending two models' calls
into one accuracy figure would describe neither.

It is a small, noisy sample and it is evidence, not proof. Predictions are only
issued once per new candle, so a 30-second refresh does not inflate the count.

The dashboard places no orders, sizes no positions, and models no fees or
slippage. It is a viewer.

---

## The React dashboard

The terminal view is one user at one terminal with no charts. `web/` is the
same model in a browser, with the charts the terminal cannot draw.

```bash
pip install fastapi uvicorn          # optional - only this needs them

# single URL: FastAPI serves the built app AND the API
cd web && npm install && npm run build && cd ..
python src/server.py --serve-web      # http://127.0.0.1:8000

# or, for front-end work: Vite on :5173 proxying /api to uvicorn on :8000
python src/server.py                  # terminal 1
cd web && npm run dev                 # terminal 2
```

### Two views, one payload

The app opens on a **simplified view**, and a button in the top bar switches to
the advanced one. Both read the same `/api/snapshot`; neither computes
anything the other does not already have.

| view | what it is | who it is for |
|---|---|---|
| **simple** (default) | two plain-language answers, the chart (price, projection, and the model's recorded projections as a curve), and a six-row table of what the model expects at each distance | someone who has not read this README |
| **advanced** | everything below on this page — the five-layer chart, both probability panels, two scorecards, the return distribution, held-out metrics, the call log | reading the model rather than the answer |

The simple view is where the project's honesty rule gets tested hardest,
because simplifying the *presentation* must not simplify the *claim*. So the
direction card says **on the card, in the same size type as the answer** that
the question is close to a coin flip and that always answering NEUTRAL already
scores ~52%. A view that dropped that line would be a better-looking lie than
the advanced one, which is the failure mode a "simple mode" invites.

The choice is remembered in `localStorage` per browser, guarded on both read
and write — a private window loses the preference, it does not break the page.

### Actual price and the predicted curve, on one chart

The advanced view's main panel draws five things against a single shared time
axis — the whole reason to overlay them rather than stack two charts:

| drawn | what it is |
|---|---|
| candlesticks | what the market did |
| dashed violet line | the regression head's projected close over the next six candles — a projected path, which is literally what the model outputs |
| dotted grey line | the same projection with **zero** return — the baseline the regression head has to beat |
| dotted violet rails | the 10/25/50/75/90th percentiles of *realised* returns on the **train** split, conditional on the class the model predicts |
| solid violet line | **every** projection the dashboard has ever recorded, read back from `models/predictions.db` — one point per candle, averaged over the windows that named it, and **broken** only where the server itself was not running |

The projection and the no-change baseline are drawn nearly on top of each
other, and that is the point: the regression head is worse than predicting no
change at every horizon on held-out data, so showing the projection alone would
flatter it. The rails are a description of the training distribution, **not** a
confidence interval for this path, and the panel says so.

The recorded history shares the live projection's violet deliberately — it is
the same head's output, only recorded earlier — so the **dash pattern** is what
separates them, not the hue. They also barely share an x-range: history sits
behind "now" and the projection entirely ahead of it. It is drawn solid and 2px
rather than the 1px dotted it started as, because a dashed line at that weight
reads as a decoration rather than as the model's track record.

Below it: volume on the same axis, direction and activity probabilities per
horizon, two live scorecards (running hit rate against each head's own
baseline), the realised-return distribution with the frozen NEUTRAL band and
today's prediction marked on it, the held-out test metrics, and the raw call
log — hits and misses set in the same typography.

The simple view draws the same chart with `simple` set, which drops the two
layers that need a briefing to read (the percentile rails and the no-change
baseline) and leaves the price, the projection, the recorded history and the
markers. Those two series are never created rather than created and left empty,
because an empty series still takes a colour in the legend and would tell a
reader there are five lines when the chart draws three.

The recorded history stays in **both** views. It is the model's actual track
rather than a decoration, and *"has its projection been following the price or
did it only look plausible once"* is a question a newcomer has too — arguably
more so. The markers stay for the same reason: a green dot for a call that
landed and a red one for a call that did not is the one part of this chart that
explains itself. What the simple view drops is the two layers that are
statements about the *training distribution* rather than about the market in
front of you; what it keeps is everything that is a statement about the model's
own behaviour.

### The projection store

The dashed curve is one window: what the model is saying *now*. On its own
that says nothing about whether the projections track the price or merely look
plausible once, so every projection the dashboard makes is written to SQLite
(`models/predictions.db`, stdlib `sqlite3`, no new dependency) and read back
into the same chart.

- One row per horizon per head per window, keyed `(anchor_ts, horizon, kind)`.
  Re-refreshing a window is `INSERT OR IGNORE`, so it cannot double-count.
- `actual_close` is filled in once the target candle prints, so the store is
  self-contained: prediction and outcome in the same row.
- The chart line uses the **direction head only**. The activity head stores
  `expected_pct = 0` by construction — it predicts a class, not a return — so
  averaging its rows in would drag the curve towards "no change" and make the
  model look worse than it is. That would be a bug in the chart, not a finding.
- **The server records on its own clock, not the browser's.** For as long as
  `src/server.py` is up, a background thread re-reads and stores every window
  even with no page open (`LiveEngine.start_recorder()`, default every 30s,
  `--record-interval` to change it). Recording used to be a side effect of a
  request arriving, so closing or merely backgrounding the tab — where the
  browser throttles the stream and its timers — quietly left holes in the
  curve. The browser is a viewer of the record, never the reason it exists.
- A gap longer than 4 candles is still drawn as a **break**, and now means one
  thing only: the server itself was down. Across that stretch the model made
  no call, and a line spanning it would imply otherwise.
- `--no-record` opens the file read-only if it exists and creates nothing if
  it does not, and starts no recorder — there is nothing to record into.
  Verified against a populated store, not assumed: 10 rows before the run,
  10 after.

It is local state — gitignored, not a result to read out of the repository.

### The API is a thin layer, on purpose

`src/server.py` contains **no** prediction, scoring or labelling logic. It
drives the same `Dashboard` object `src/dashboard.py` drives — `refresh()`
fetches candles, builds features through the training path, runs the ensemble,
issues calls and resolves them — and serialises the result to JSON:

```
GET /api/snapshot        everything the UI needs, one round trip
GET /api/candles         OHLCV only
GET /api/prediction      the model read + the projected curve
GET /api/track-record    resolved calls and the tallies
GET /api/metrics         held-out metrics from models/eval_summary.json
GET /api/stream          the snapshot again, as Server-Sent Events
GET /api/health          liveness, model name, last fetch
```

A browser panel that computed its own accuracy would be describing a different
model from the one the scorecard describes, so there is exactly one
implementation of every rule and it is the Python one. `web/` only renders what
it is given.

The UI updates by SSE and **falls back to polling** if no frame arrives for 25
seconds; the badge in the header says which mode is live, because a dashboard
that quietly stops updating is worse than one that admits it is polling.

```bash
cd web && npm run check            # every panel renders against a real payload
cd web && npm run check:markers    # marker times satisfy the chart's contract
```

`web/ssr-check.jsx` renders every panel against a real captured snapshot in
Node, with no browser, so a prop/shape mismatch fails a command instead of
showing up as a blank panel. It also renders ten degenerate payloads — empty
track record, no bands, no metrics, no activity head, no history, a history
with a gap in it, an error frame, and three the simple view has to survive —
which must degrade rather than throw. It ends by asserting which view opens by
default, because that is a product decision and not an implementation detail:
a first-time reader must land on the plain-language dashboard, the advanced
panels must not be mounted alongside it, and the toggle must be offered. The
`lightweight-charts` panels (including the simple chart) build in an effect, so
the check sees only their container and says so; the chart wiring itself is
verified in a browser.

The fixture is a **capture**, not a live read, and a stale one quietly turns the
suite into a check of a payload from weeks ago — which is exactly how
`prediction_history` came to be missing from it while the components were being
written against it, so the history cases were passing only against the synthetic
fixture. The check now prints the fixture's age and warns past six hours. It is
a warning and not a failure on purpose: a fixture older than the last time the
machine was on is a normal way to run a test suite, and a build that breaks for
it only teaches everyone to ignore the output. Re-capture with:

```
python src/server.py --port 8099        # terminal 1
curl -s http://127.0.0.1:8099/api/snapshot -o web/_snapshot.json
```

`web/markers-check.mjs` covers the one part of the wiring that is pure
arithmetic and can be checked without a browser: the call markers.

The crash it guards was `setMarkers()` rejecting its argument with `data must
be asc ordered by time`. The order that caused it is neither ascending nor
unique, because a window writes its six direction horizons (targets +1..+6
candles) and then its four activity horizons, whose list restarts at +1.
`src/server.py` now sorts `recent` by `target_ts`, but the chart does not
depend on that: `buildMarkers()` sorts, merges calls that share a candle into
one tally marker, and drops unplaceable times.

The check rebuilds that pathological order **from** the captured calls rather
than expecting to find it in them. It used to assert the fixture still
contained a backwards step — which was true only while the server emitted
issue order, so the moment the server was fixed the guard began failing on
every re-capture, and for the right reason: the bug was gone, so a genuine
re-capture read as a regression. Rebuilding it from the same real calls, using
the rule that produced it, is stable across captures and still fails if the
merge regresses. Three further properties are asserted rather than assumed:
identical output for sorted, reversed and issue-ordered input; no marker for a
call whose target candle has not arrived; and every resolved call accounted
for across the merge. The collision count is derived rather than written down,
because how many windows a capture contains is not a stable fact about the
code.

---

## Limitations

- **One asset, one interval, one exchange.** BTCUSDT 15m on Binance spot only.
- **The test period is a different market regime.** The train thresholds place
  30/40/30 on train by construction; the same frozen thresholds give
  ≈24/54/22 on test. That drift is real and is reported rather than hidden.
- **Fat tails.** 15m return kurtosis is ≈84. Features are clipped at ±5σ
  (0.33% of training values) for numerical stability, which is a modelling
  choice, not a claim that extremes do not matter.
- **No backtest is implemented.** Notebooks 04 and 05 contain an intentionally
  empty scaffold, and the dashboard keeps a live scorecard but does not model
  fees, slippage or position sizing. Directional accuracy is not profitability,
  and no profitability result is claimed.
- **Inference is a research helper**, not a service. There *is* an API
  (`src/server.py`) and it deliberately holds no prediction logic, but there is
  no scheduler, no authentication and no order execution. Both dashboards are
  viewers - neither is an order entry system, and neither has risk controls.
  The one thing that does persist is the projections themselves, in
  `models/predictions.db`; that is local state on whichever machine runs the
  dashboard, not a service, and a reader has no way to audit a number quoted
  from someone else's store.
- **The dashboard's scorecard is a small live sample.** It is measured, not
  asserted, but the first few dozen calls are noise. Do not read a hit rate off
  twenty calls.
- **The confidence floor does not help.** Measured on both splits: raising it
  costs macro-F1 monotonically and buys essentially no accuracy. It is exposed
  as a control you can test, not as a tuning knob that improves the output.
- **The direction head is NEUTRAL-biased, and that is the main weakness.**
  Directional recall is 0.22–0.29 on test — better than the 0.18–0.22 measured
  before the class-weighted loss was introduced, but still missing most
  directional moves. The class weighting was the direct attack on this and it
  helped without solving it; a different neutral band is the obvious next
  lever. Do not spend that effort on the regression head, which has no signal
  to recover.
- **The regime head answers a question about volatility, not direction.** It is
  the strongest head (mean lift +0.15 over a 0.50 baseline on test) and it is
  still not a trade. Do not let the better number migrate into a directional
  claim.
