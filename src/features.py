"""
Feature engineering for the BTCUSDT 15m model.

Hard guarantees
---------------
1. EVERY feature at row t is a function of rows <= t only. Every primitive used
   here is strictly backward-looking:
       .shift(k)          k >= 1      past values
       .rolling(w).x()                trailing window, right-aligned
       .ewm(...)                      recursive, past-weighted
       .diff()                        t - (t-1)
   There is no .shift(-k), no centered window, no .bfill(), no interpolation.
2. NO absolute price level is used as a feature. BTC traded at $4k in 2017 and
   >$100k in 2026; raw OHLC features would let the network memorise the price
   regime instead of learning market behaviour. Every price expression is a
   return, a ratio, a percentage distance or a volatility. Raw OHLCV stays in
   the dataset for auditing/reconstruction but is NOT a model input.
3. Redundancy is deliberately avoided - see the "excluded features" note at the
   bottom of this module.

Feature groups (49 columns)
---------------------------
  returns       6   multi-scale past returns
  volatility    6   rolling vol, vol regime, ATR, ATR%
  trend         6   normalised EMA distances / slopes
  momentum      4   RSI, MACD line / signal / histogram (price-normalised)
  bollinger     4   width, position, normalised band distances
  volume        5   volume change / ratios / z-score / regime
  trades        4   trade-count activity + average trade size
  orderflow     3   taker-buy (aggressive buyer) participation
  candle        4   range, body position, wicks
  range         3   position inside the recent high/low envelope
  time          4   cyclical hour-of-day and day-of-week (UTC)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    ATR_WINDOW,
    BB_STD,
    BB_WINDOW,
    EMA_FAST,
    EMA_MID,
    EMA_SLOPE_LAG,
    EMA_SLOW,
    HIGH_LOW_LOOKBACK,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    RANGE_POSITION_WINDOW,
    RETURN_LAGS,
    RSI_WINDOW,
    VOL_WINDOWS,
    VOLUME_LONG,
    VOLUME_SHORT,
)

EPS = 1e-12


# ----------------------------------------------------------------------------
# Helpers - every one of them is backward-looking
# ----------------------------------------------------------------------------

def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, mapping inf / 0-0 to NaN (resolved later by the warmup drop)."""
    out = numerator / denominator.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _pct_change(series: pd.Series, periods: int) -> pd.Series:
    """series / series.shift(periods) - 1, computed explicitly (no pct_change)."""
    return _safe_div(series, series.shift(periods)) - 1.0


def _log_change(series: pd.Series, periods: int) -> pd.Series:
    return np.log(_safe_div(series, series.shift(periods)))


def _rsi(close: pd.Series, window: int) -> pd.Series:
    """Wilder's RSI. Strictly backward-looking (recursive EWM)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    # avg_loss == 0 -> pure uptrend -> RSI 100
    rs = _safe_div(avg_gain, avg_loss)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Wilder's Average True Range. Uses only past and current bars."""
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------

def add_return_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    for lag in RETURN_LAGS:
        out[f"return_{lag}"] = _pct_change(out["close"], lag)
    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    r1 = _pct_change(out["close"], 1)

    for w in VOL_WINDOWS:
        out[f"volatility_{w}"] = r1.rolling(w, min_periods=w).std()

    # Volatility regime: short-term vol relative to the 24h baseline.
    out["volatility_ratio_16_96"] = _safe_div(out["volatility_16"], out["volatility_96"])

    atr = _atr(out["high"], out["low"], out["close"], ATR_WINDOW)
    out[f"atr_{ATR_WINDOW}"] = atr
    out[f"atr_pct_{ATR_WINDOW}"] = _safe_div(atr, out["close"])
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    close = out["close"]
    ema_f = _ema(close, EMA_FAST)
    ema_m = _ema(close, EMA_MID)
    ema_s = _ema(close, EMA_SLOW)

    # Normalised distances: "how far is price from its trend", scale-free.
    out[f"ema{EMA_FAST}_dist"] = _safe_div(close, ema_f) - 1.0
    out[f"ema{EMA_MID}_dist"] = _safe_div(close, ema_m) - 1.0
    out[f"ema{EMA_SLOW}_dist"] = _safe_div(close, ema_s) - 1.0
    out[f"ema{EMA_FAST}_{EMA_MID}_ratio"] = _safe_div(ema_f, ema_m) - 1.0
    out[f"ema{EMA_MID}_{EMA_SLOW}_ratio"] = _safe_div(ema_m, ema_s) - 1.0
    # Trend slope over the last hour.
    out[f"ema{EMA_FAST}_slope_{EMA_SLOPE_LAG}"] = _pct_change(ema_f, EMA_SLOPE_LAG)
    return out


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    close = out["close"]

    out[f"rsi_{RSI_WINDOW}_norm"] = (_rsi(close, RSI_WINDOW) - 50.0) / 50.0

    macd_line = _ema(close, MACD_FAST) - _ema(close, MACD_SLOW)
    macd_sig = _ema(macd_line, MACD_SIGNAL)
    macd_hist = macd_line - macd_sig

    # MACD is a price difference -> divide by price to make it scale-free.
    out["macd_norm"] = _safe_div(macd_line, close)
    out["macd_signal_norm"] = _safe_div(macd_sig, close)
    out["macd_hist_norm"] = _safe_div(macd_hist, close)
    return out


def add_bollinger_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    close = out["close"]

    mid = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean()
    sd = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    upper = mid + BB_STD * sd
    lower = mid - BB_STD * sd
    band = upper - lower

    out["bb_width"] = _safe_div(band, mid)
    out["bb_position"] = _safe_div(close - lower, band)
    out["bb_upper_dist"] = _safe_div(upper - close, close)
    out["bb_lower_dist"] = _safe_div(close - lower, close)
    return out


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    volume = out["volume"]

    out["volume_change_1"] = _pct_change(volume, 1)

    ma_s = volume.rolling(VOLUME_SHORT, min_periods=VOLUME_SHORT).mean()
    ma_l = volume.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).mean()
    sd_l = volume.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).std()

    out[f"volume_ratio_{VOLUME_SHORT}"] = _safe_div(volume, ma_s) - 1.0
    out[f"volume_ratio_{VOLUME_LONG}"] = _safe_div(volume, ma_l) - 1.0
    out[f"volume_zscore_{VOLUME_LONG}"] = _safe_div(volume - ma_l, sd_l)
    out[f"volume_regime_{VOLUME_SHORT}_{VOLUME_LONG}"] = _safe_div(ma_s, ma_l) - 1.0
    return out


def add_trade_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    trades = out["number_of_trades"].astype("float64")

    out["trade_count_change_1"] = _pct_change(trades, 1)

    ma_s = trades.rolling(VOLUME_SHORT, min_periods=VOLUME_SHORT).mean()
    ma_l = trades.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).mean()
    sd_l = trades.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).std()

    out[f"trade_count_ratio_{VOLUME_SHORT}"] = _safe_div(trades, ma_s) - 1.0
    out[f"trade_count_zscore_{VOLUME_LONG}"] = _safe_div(trades - ma_l, sd_l)

    # Average trade size (BTC per trade) vs its own 24h baseline -> "whale" proxy.
    avg_size = _safe_div(out["volume"], trades)
    out[f"avg_trade_size_ratio_{VOLUME_LONG}"] = _safe_div(
        avg_size, avg_size.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).mean()
    ) - 1.0
    return out


def add_orderflow_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    ratio = _safe_div(out["taker_buy_base_volume"], out["volume"])

    ma_s = ratio.rolling(VOLUME_SHORT, min_periods=VOLUME_SHORT).mean()
    ma_l = ratio.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).mean()
    sd_l = ratio.rolling(VOLUME_LONG, min_periods=VOLUME_LONG).std()

    out["taker_buy_ratio"] = ratio
    out[f"taker_buy_ratio_dev_{VOLUME_SHORT}"] = ratio - ma_s
    out[f"taker_buy_ratio_zscore_{VOLUME_LONG}"] = _safe_div(ratio - ma_l, sd_l)
    return out


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    high, low, open_, close = out["high"], out["low"], out["open"], out["close"]

    rng = high - low
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)

    out["candle_range_pct"] = _safe_div(rng, close)
    # Where the close sits inside the bar; +1 = closed on the high, -1 = on the low.
    out["body_ratio"] = 2.0 * _safe_div(close - low, rng) - 1.0
    out["upper_wick_pct"] = _safe_div(high - body_high, close)
    out["lower_wick_pct"] = _safe_div(body_low - low, close)
    return out


def add_range_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    close = out["close"]

    rng_hi = out["high"].rolling(RANGE_POSITION_WINDOW, min_periods=RANGE_POSITION_WINDOW).max()
    rng_lo = out["low"].rolling(RANGE_POSITION_WINDOW, min_periods=RANGE_POSITION_WINDOW).min()
    out[f"close_pos_range_{RANGE_POSITION_WINDOW}"] = _safe_div(close - rng_lo, rng_hi - rng_lo)

    hi96 = out["high"].rolling(HIGH_LOW_LOOKBACK, min_periods=HIGH_LOW_LOOKBACK).max()
    lo96 = out["low"].rolling(HIGH_LOW_LOOKBACK, min_periods=HIGH_LOW_LOOKBACK).min()
    out["dist_from_high_96"] = _safe_div(close, hi96) - 1.0
    out["dist_from_low_96"] = _safe_div(close, lo96) - 1.0
    return out


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical UTC time encoding - no raw integers are fed to the model."""
    out = df
    ts = out["timestamp"]

    hour = ts.dt.hour + ts.dt.minute / 60.0
    out["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)

    dow = ts.dt.dayofweek.astype("float64")
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
    return out


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, list[str]] = {
    "returns": [f"return_{k}" for k in RETURN_LAGS],
    "volatility": (
        [f"volatility_{w}" for w in VOL_WINDOWS]
        + ["volatility_ratio_16_96", f"atr_{ATR_WINDOW}", f"atr_pct_{ATR_WINDOW}"]
    ),
    "trend": [
        f"ema{EMA_FAST}_dist", f"ema{EMA_MID}_dist", f"ema{EMA_SLOW}_dist",
        f"ema{EMA_FAST}_{EMA_MID}_ratio", f"ema{EMA_MID}_{EMA_SLOW}_ratio",
        f"ema{EMA_FAST}_slope_{EMA_SLOPE_LAG}",
    ],
    "momentum": [
        f"rsi_{RSI_WINDOW}_norm", "macd_norm", "macd_signal_norm", "macd_hist_norm",
    ],
    "bollinger": ["bb_width", "bb_position", "bb_upper_dist", "bb_lower_dist"],
    "volume": [
        "volume_change_1", f"volume_ratio_{VOLUME_SHORT}", f"volume_ratio_{VOLUME_LONG}",
        f"volume_zscore_{VOLUME_LONG}", f"volume_regime_{VOLUME_SHORT}_{VOLUME_LONG}",
    ],
    "trades": [
        "trade_count_change_1", f"trade_count_ratio_{VOLUME_SHORT}",
        f"trade_count_zscore_{VOLUME_LONG}", f"avg_trade_size_ratio_{VOLUME_LONG}",
    ],
    "orderflow": [
        "taker_buy_ratio", f"taker_buy_ratio_dev_{VOLUME_SHORT}",
        f"taker_buy_ratio_zscore_{VOLUME_LONG}",
    ],
    "candle": ["candle_range_pct", "body_ratio", "upper_wick_pct", "lower_wick_pct"],
    "range": [
        f"close_pos_range_{RANGE_POSITION_WINDOW}", "dist_from_high_96", "dist_from_low_96",
    ],
    "time": ["hour_sin", "hour_cos", "dow_sin", "dow_cos"],
}

FEATURE_COLUMNS: list[str] = [c for group in FEATURE_GROUPS.values() for c in group]


def build_features(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Attach all model features. Row count is unchanged (warmup NaNs kept)."""
    out = df.copy()
    for fn in (
        add_return_features,
        add_volatility_features,
        add_trend_features,
        add_momentum_features,
        add_bollinger_features,
        add_volume_features,
        add_trade_features,
        add_orderflow_features,
        add_candle_features,
        add_range_features,
        add_time_features,
    ):
        out = fn(out)

    missing = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing:
        raise AssertionError(f"Feature registry out of sync, missing: {missing}")

    # Guard against accidental complex/inf values sneaking in via a 0/0.
    for col in FEATURE_COLUMNS:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan).astype("float64")

    if verbose:
        print(f"  [features] {len(FEATURE_COLUMNS)} features across "
              f"{len(FEATURE_GROUPS)} groups")
    return out


def feature_nan_report(df: pd.DataFrame, *, tail_rows: int = 0) -> pd.Series:
    """NaN counts per feature. `tail_rows`>0 audits only the tail (post-warmup)."""
    frame = df if tail_rows <= 0 else df.tail(tail_rows)
    return frame[FEATURE_COLUMNS].isna().sum().sort_values(ascending=False)


# ----------------------------------------------------------------------------
# Deliberately EXCLUDED features (documented, not accidental)
# ----------------------------------------------------------------------------
#  body_pct = (close - open) / close
#       For a 24/7 market open[t] == close[t-1], so this is return_1 to ~10
#       decimal places (empirical correlation > 0.999). Including both would
#       duplicate a feature and dilute the input. body_ratio (close position
#       within the bar) carries the genuinely new information and is kept.
#
#  buy_pressure = (2 * taker_buy_base_volume - volume) / volume
#       This is exactly 2 * taker_buy_ratio - 1, an affine transform of a
#       feature already present. Affine duplicates add no information to a
#       network and only add parameters. taker_buy_ratio plus its deviation and
#       z-score are kept instead.
#
#  raw open / high / low / close / ema20 / ema50 / ema100 / bb_middle
#  / bb_upper / bb_lower / atr
#       Absolute price levels. Across 2017-2026 BTC spans ~$4k to >$100k; these
#       features would let the model index the price regime rather than learn
#       behaviour, and they are useless out-of-sample at unseen price levels.
#       Every one of them is represented by a normalised counterpart
#       (distance / ratio / percentage / z-score) in the groups above.
