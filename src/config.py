"""
Central configuration for the BTCUSDT 15m short-term trend prediction project.

Every tunable knob lives here so that the pipeline, the notebooks and the
inference helper all agree on one single source of truth.

Nothing in this module imports TensorFlow / PyTorch. The local pipeline is a
pure pandas + numpy + scikit-learn data pipeline.
"""

from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FINAL_DIR = DATA_DIR / "final"
MODELS_DIR = PROJECT_ROOT / "models"

# train.py writes this next to the weights. It names the ensemble members and
# carries the frozen regime thresholds, so it is what inference should read
# rather than hard-coding a filename.
INFERENCE_BUNDLE = MODELS_DIR / "inference_bundle.json"

# Raw artefacts
RAW_ZIP_DIR = RAW_DIR / "zips"            # untouched Binance archives
RAW_CANDLES_CSV = RAW_DIR / "btcusdt_15m_raw.csv"   # merged raw candles

# Processed artefacts
CLEAN_CANDLES_CSV = PROCESSED_DIR / "btcusdt_15m_clean.csv"        # gaps audited
FEATURES_CSV = PROCESSED_DIR / "btcusdt_15m_features.csv"          # + features
LABELLED_CSV = PROCESSED_DIR / "btcusdt_15m_features_targets.csv"  # + targets

# Final deliverable (upload THESE to Google Colab)
FINAL_DATASET_CSV = FINAL_DIR / "btcusdt_15m_dataset.csv"
FINAL_X_TRAIN = FINAL_DIR / "X_train.npy"
FINAL_X_VAL = FINAL_DIR / "X_val.npy"
FINAL_X_TEST = FINAL_DIR / "X_test.npy"
FINAL_Y_TRAIN = FINAL_DIR / "y_train.npy"
FINAL_Y_VAL = FINAL_DIR / "y_val.npy"
FINAL_Y_TEST = FINAL_DIR / "y_test.npy"
FINAL_YDIR_TRAIN = FINAL_DIR / "y_dir_train.npy"
FINAL_YDIR_VAL = FINAL_DIR / "y_dir_val.npy"
FINAL_YDIR_TEST = FINAL_DIR / "y_dir_test.npy"
FINAL_YREGIME_TRAIN = FINAL_DIR / "y_regime_train.npy"
FINAL_YREGIME_VAL = FINAL_DIR / "y_regime_val.npy"
FINAL_YREGIME_TEST = FINAL_DIR / "y_regime_test.npy"
FINAL_TS_TRAIN = FINAL_DIR / "timestamps_train.npy"
FINAL_TS_VAL = FINAL_DIR / "timestamps_val.npy"
FINAL_TS_TEST = FINAL_DIR / "timestamps_test.npy"
# Target timestamp of EVERY row of the labelled frame, in order. Lets the audit
# map a window's target timestamp back to its absolute row index.
FINAL_TS_ALL = FINAL_DIR / "timestamps_all.npy"
FINAL_SEGMENT_IDS = FINAL_DIR / "segment_ids.npy"

# Window-start offsets into the corresponding X_*.npy block. Required because
# gap-excluded windows leave holes in the start index - see pipeline.py.
FINAL_STARTS_TRAIN = FINAL_DIR / "starts_train.npy"
FINAL_STARTS_VAL = FINAL_DIR / "starts_val.npy"
FINAL_STARTS_TEST = FINAL_DIR / "starts_test.npy"
FINAL_SCALER = FINAL_DIR / "scaler.pkl"
FINAL_TARGET_SCALER = FINAL_DIR / "target_scaler.pkl"
FINAL_METADATA = FINAL_DIR / "dataset_metadata.json"
FINAL_FEATURE_NAMES = FINAL_DIR / "feature_names.json"

# ----------------------------------------------------------------------------
# MARKET DATA SOURCE
# ----------------------------------------------------------------------------

SYMBOL = "BTCUSDT"
INTERVAL = "15m"
INTERVAL_MINUTES = 15

# Binance Vision bulk archive (efficient historical backfill)
VISION_BASE = "https://data.binance.vision/data/spot"
# Binance REST API (fallback / most recent candles)
REST_BASE = "https://api.binance.com"

# First month in which Binance publishes BTCUSDT 15m klines (checked: 2017-08).
FIRST_DATA_MONTH = "2017-08"

# Binance switched kline timestamp units from milliseconds to MICROSECONDS
# starting with the 2025-01 archive. The loader normalises both to UTC.
MICROSECOND_EPOCH_FROM = "2025-01"

# Politeness / robustness for the REST fallback
REST_LIMIT = 1000                 # Binance max klines per request
REST_MAX_RETRIES = 5
REST_BACKOFF_SECONDS = 2.0
HTTP_TIMEOUT = 60

# ----------------------------------------------------------------------------
# RAW COLUMN SCHEMA (Binance spot klines - 12 positional columns, no header)
# ----------------------------------------------------------------------------

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

# Columns we actually keep (drop `ignore` and `close_time`, which is derivable).
RAW_KEEP_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]

# ----------------------------------------------------------------------------
# CLEANING / GAP POLICY
# ----------------------------------------------------------------------------

# A "gap" is any discontinuity in the 15m grid.
# max_gap_tolerance_candles = number of *missing* candles we tolerate before the
# discontinuity is reported as a "large gap" in the audit report.  A hard
# sequence boundary is created at EVERY discontinuity regardless of size, so a
# rolling window can never straddle missing market data.
LARGE_GAP_MISSING_CANDLES = 3      # >= 3 missing candles (>=45 min) => "large"

# Rows dropped from the head of the series so that every lookback feature
# (EMA-100, rolling-96 windows, ...) is fully warmed up and no recursive
# indicator is still converging from its seed value.
WARMUP_ROWS = 500                  # ~5.2 days of 15m candles

# ----------------------------------------------------------------------------
# FEATURES
# ----------------------------------------------------------------------------

RETURN_LAGS = [1, 2, 4, 8, 16, 32]           # in 15m candles

VOL_WINDOWS = [16, 32, 96]                   # rolling std of 1-bar returns
ATR_WINDOW = 14
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
BB_WINDOW, BB_STD = 20, 2.0
EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 100
EMA_SLOPE_LAG = 4                            # 1 hour
VOLUME_SHORT, VOLUME_LONG = 16, 96
RANGE_POSITION_WINDOW = 64                   # 16 hours (== lookback)
HIGH_LOW_LOOKBACK = 96                       # 24 hours

# ----------------------------------------------------------------------------
# SEQUENCES  (model input geometry)
# ----------------------------------------------------------------------------

# 64 x 15m == 16 hours of historical context per sample.
LOOKBACK = 64

# Step between consecutive sequence start indices.  1 == fully rolling
# ("rows 0..63, then 1..64, ...").  Raising it thins the sample count.
SEQUENCE_STRIDE = 1

# ----------------------------------------------------------------------------
# TARGETS
# ----------------------------------------------------------------------------

# Prediction horizons expressed in 15m candles (and their wall-clock label).
HORIZONS = [1, 2, 3, 4, 5, 6]
HORIZON_LABELS = {1: "15m", 2: "30m", 3: "45m", 4: "60m", 5: "75m", 6: "90m"}
HORIZON_MAX = max(HORIZONS)                  # 6 candles == 90 minutes

FUTURE_RETURN_COLUMNS = [f"future_return_{HORIZON_LABELS[h]}" for h in HORIZONS]
DIRECTION_COLUMNS = [f"direction_{HORIZON_LABELS[h]}" for h in HORIZONS]

# Directional class encoding.  NEUTRAL sits in the middle.
DIRECTION_CLASSES = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}

# --- Threshold strategy -----------------------------------------------------
# "train_quantile" : thresholds = quantiles of the TRAIN target distribution.
#                    Chosen so NEUTRAL occupies a fixed central mass, which
#                    keeps the three classes usable at every horizon (a single
#                    fixed bps threshold would leave 15m almost entirely
#                    NEUTRAL and 90m almost entirely directional).
# "fixed_bps"      : a constant symmetric threshold in basis points.
#
# Thresholds are ALWAYS derived from the training split only and then applied
# unchanged to validation / test.  The test set never influences the threshold.
THRESHOLD_MODE = "train_quantile"

# Fraction of the TRAIN distribution assigned to NEUTRAL (middle band).
# 0.40 => ~30% DOWN / ~40% NEUTRAL / ~30% UP on train by construction.
NEUTRAL_QUANTILE = 0.40

# Used only when THRESHOLD_MODE == "fixed_bps"; one value per horizon (bps).
FIXED_THRESHOLD_BPS = {1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0, 6: 70.0}

# ----------------------------------------------------------------------------
# MARKET-REGIME (ACTIVITY) TARGETS
# ----------------------------------------------------------------------------
# The direction head answers "which way?". Its accuracy ceiling on 15m BTC is
# ~53-56% and it is dominated by the NEUTRAL class, because the sign of a
# 15-minute move is close to unpredictable.
#
# The regime head answers a DIFFERENT and much more predictable question:
# "is the market about to be active or quiet?". Concretely, for a horizon of h
# candles,
#
#     forward_absmove_h[t] = mean( |15m return| over candles t+1 .. t+h )
#     reference_absmove[t] = mean( |15m return| over candles t-95 .. t )
#     regime_ratio_h[t]    = forward_absmove_h[t] / reference_absmove[t]
#     class                = ACTIVE if regime_ratio_h[t] > tau_h else QUIET
#
# where tau_h is the MEDIAN of regime_ratio_h over the TRAIN split only, so the
# two classes are 50/50 by construction on train.
#
# Why this is a legitimate target and not a relabelled cheat:
#   * It is self-normalising. Both sides of the ratio are measured in the same
#     units over the same asset, so the label does not drift when BTC's price
#     or its overall volatility regime changes. Train, validation and test all
#     land near 50/50 - the class balance IS stationary, which is exactly what
#     the direction thresholds fail to achieve.
#   * It is a genuinely different question from direction. It predicts the
#     MAGNITUDE of upcoming movement, not its sign, and it is the one question
#     in this dataset where the model beats its baseline by a wide margin.
#   * Volatility clustering - the fact that turbulent periods follow turbulent
#     periods - is the single most robust stylised fact in financial time
#     series. The model is exploiting that, and the exploit is real.
#
# It does NOT replace the direction head. Both are trained jointly off one
# shared trunk; the regime head simply gets to be the honest headline, because
# it is the one whose accuracy means something.
REGIME_ENABLED = True

# Horizons the regime head predicts, in 15m candles. Deliberately a SUBSET of
# HORIZONS: short windows carry the most signal, and every extra head costs
# parameters and training time.
REGIME_HORIZONS = [1, 2, 4, 6]                 # 15m, 30m, 1h, 90m

# Trailing window (candles) used as each sample's own activity reference.
REGIME_REF_WINDOW = 96                         # 24 hours

# Class encoding for the regime head.
REGIME_CLASSES = {0: "QUIET", 1: "ACTIVE"}

# How the per-horizon threshold is chosen from the TRAIN ratio distribution.
# "median" gives an exactly balanced 50/50 split on train.
REGIME_THRESHOLD_QUANTILE = 0.50

REGIME_COLUMNS = [f"regime_ratio_{HORIZON_LABELS[h]}" for h in REGIME_HORIZONS]
REGIME_TARGET_COLUMNS = [f"regime_{HORIZON_LABELS[h]}" for h in REGIME_HORIZONS]

# ----------------------------------------------------------------------------
# DIRECTION CLASS WEIGHTING
# ----------------------------------------------------------------------------
# The direction head, trained with a plain cross-entropy, collapses onto
# NEUTRAL: measured per-class recall on the test split was DOWN 0.19 /
# NEUTRAL 0.85 / UP 0.21, i.e. it only commits to a direction on 26% of calls.
# Weighting the loss by inverse class frequency is the standard correction.
#
# `strength` interpolates between no weighting (0.0) and full inverse-frequency
# weighting (1.0), as w_c  proportional to  (1 / freq_c) ** strength.  Weights
# are then normalised to mean 1 so the overall loss scale - and therefore the
# effective learning rate - does not change with the setting.
#
# This TRADES accuracy for macro-F1: pushing recall into the minority classes
# necessarily costs some majority-class hits. Accuracy is barely affected
# either way, and the regime head is the accuracy headline, so the default
# takes the macro-F1.
DIRECTION_CLASS_WEIGHT_STRENGTH = 0.5

# ----------------------------------------------------------------------------
# CHECKPOINT / EARLY-STOPPING METRIC
# ----------------------------------------------------------------------------
# The original configuration early-stopped on `val_loss`, which is
# MSE(return) + CE(direction) + CE(regime). The regression head has NO skill
# (its test correlation with the truth is ~0.02, and it is worse than
# predicting the mean at every horizon), so most of the early-stopping signal
# was a head that cannot predict anything drowning out the two that can.
#
# Instead we monitor a score computed on the validation split by a callback:
#
#     selection_score = 0.5 * direction_macro_F1 + 0.5 * regime_accuracy
#
# Both terms live in [0, 1] and both are things we actually report.
SELECTION_DIRECTION_WEIGHT = 0.5
SELECTION_REGIME_WEIGHT = 0.5

# ----------------------------------------------------------------------------
# SEED ENSEMBLING
# ----------------------------------------------------------------------------
# Averaging the softmax outputs of several seeds trained from the same data is
# the cheapest reliable accuracy gain available here - it costs only time, it
# cannot leak, and it reduces the variance that makes a single-seed score hard
# to trust. `python src/train.py --seeds 42,43,44` trains three models and
# writes an ensemble manifest; `--ensemble 0,1,2` picks which of them to mix.
ENSEMBLE_SEEDS = (42, 43, 44)

# ----------------------------------------------------------------------------
# SPLITTING  (chronological, never shuffled)
# ----------------------------------------------------------------------------

TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# Purge/embargo: number of SAMPLES removed at each split boundary (train->val
# and val->test).  Because samples roll forward one candle at a time, a purge of
# P samples guarantees at least P candles of separation between the two splits.
#
# Why P must be LOOKBACK + HORIZON_MAX and not just HORIZON_MAX:
#   the LAST train sample's label is resolved from candles
#       [start + LOOKBACK - 1,  start + LOOKBACK - 1 + HORIZON_MAX]
#   i.e. it reads up to LOOKBACK - 1 + HORIZON_MAX rows past its own start.
#   The FIRST validation sample's feature window begins at its own start.
#   For no training label to be resolved from candles inside the validation
#   feature window, the separation must exceed LOOKBACK - 1 + HORIZON_MAX.
# Purging a full LOOKBACK + HORIZON_MAX also means the train and validation
# windows share no raw candle at all - the strictest available reading of
# "no overlap".  The cost is tiny: 2 x 70 rows out of ~313,000 samples.
PURGE_SAMPLES = LOOKBACK + HORIZON_MAX

# ----------------------------------------------------------------------------
# SCALING
# ----------------------------------------------------------------------------

# "standard" -> StandardScaler, "robust" -> RobustScaler (median / IQR).
SCALER_KIND = "standard"

# Scaled features are clipped to +/- this many sigmas.  Bounds are derived from
# the TRAIN split only.  Crypto features have fat tails; clipping keeps the
# recurrent network numerically stable.  Set to None to disable.
CLIP_SIGMA = 5.0

# Targets are percentage returns; a target scaler makes NN regression far
# better conditioned. Raw (unscaled) returns remain the source of truth in the
# CSV / y_*.npy files - the scaler is an optional training convenience.
SCALE_TARGETS = True

# ----------------------------------------------------------------------------
# REPRODUCIBILITY
# ----------------------------------------------------------------------------

RANDOM_SEED = 42

# Save the materialised 3D (samples, LOOKBACK, features) arrays to disk.
# DEFAULT IS FALSE ON PURPOSE: with ~9 years of 15m candles the 3D array is
# several GB, while the equivalent 2D matrix is ~60 MB.  The 3D tensors are
# reconstructed as zero-copy sliding windows at training time (see
# src/sequences.py::make_sequences and the Colab notebook).
SAVE_3D_ARRAYS = False


def horizon_label(candles: int) -> str:
    """'1' -> '15m'."""
    return HORIZON_LABELS[candles]


def future_return_col(candles: int) -> str:
    return f"future_return_{horizon_label(candles)}"


def direction_col(candles: int) -> str:
    return f"direction_{horizon_label(candles)}"
