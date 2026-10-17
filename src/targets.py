"""
Target engineering: future returns + directional classes.

Target definition
-----------------
For a candle at index t and horizon h (in 15m candles):

    future_return_h[t] = 100 * (close[t+h] / close[t] - 1)      [percent]

  h = 1 -> 15m     h = 4 -> 60m
  h = 2 -> 30m     h = 5 -> 75m
  h = 3 -> 45m     h = 6 -> 90m

These are the ONLY places future data is touched, and they are never fed back
into a feature.

Gap safety
----------
If close[t+h] lies in a different segment than close[t] (i.e. the horizon spans
a data gap), the target is invalidated to NaN rather than silently measuring a
return across missing market data. Because a segment is a contiguous run of
candles, comparing the two endpoints is sufficient.

Directional classes
-------------------
    0 = DOWN     return <  lower_threshold
    1 = NEUTRAL  lower_threshold <= return <= upper_threshold
    2 = UP       return >  upper_threshold

Thresholds are selected from the TRAINING split only and then frozen. The
validation and test sets never influence them (see `compute_thresholds`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    DIRECTION_CLASSES,
    DIRECTION_COLUMNS,
    FIXED_THRESHOLD_BPS,
    FUTURE_RETURN_COLUMNS,
    HORIZON_LABELS,
    HORIZONS,
    NEUTRAL_QUANTILE,
    THRESHOLD_MODE,
    future_return_col,
)

DOWN, NEUTRAL, UP = 0, 1, 2


# ----------------------------------------------------------------------------
# Future returns
# ----------------------------------------------------------------------------

def add_future_returns(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """Attach the six forward percentage returns. Gap-crossing targets -> NaN."""
    out = df.copy()
    close = out["close"]
    seg = out["segment_id"]

    for h in HORIZONS:
        col = future_return_col(h)
        forward = close.shift(-h)                       # FUTURE - target only
        ret = (forward / close - 1.0) * 100.0           # percent

        # Endpoint segment check == "did this horizon span a gap?".
        same_segment = seg.shift(-h) == seg
        out[col] = ret.where(same_segment)

    if verbose:
        bad = out[FUTURE_RETURN_COLUMNS].isna().sum()
        total = len(out)
        tail = max(HORIZONS)                            # unavoidable tail NaNs
        print(f"  [targets] {len(FUTURE_RETURN_COLUMNS)} horizons "
              f"({', '.join(HORIZON_LABELS[h] for h in HORIZONS)})")
        print(f"  [targets] last {tail} rows have no future close by construction")
        gap_invalidated = int((bad - tail).clip(lower=0).sum())
        if gap_invalidated:
            print(f"  [targets] {gap_invalidated:,} additional targets invalidated "
                  f"because they would cross a data gap")
        _ = total
    return out


# ----------------------------------------------------------------------------
# Distribution analysis
# ----------------------------------------------------------------------------

def return_distribution_report(
    returns: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Descriptive statistics used to justify the thresholding strategy."""
    columns = columns or FUTURE_RETURN_COLUMNS
    rows = {}
    for col in columns:
        s = returns[col].dropna()
        if s.empty:
            continue
        rows[col] = {
            "count": int(s.size),
            "mean_%": s.mean(),
            "std_%": s.std(),
            "skew": s.skew(),
            "kurtosis": s.kurtosis(),
            "min_%": s.min(),
            "p01_%": s.quantile(0.01),
            "p05_%": s.quantile(0.05),
            "p25_%": s.quantile(0.25),
            "p50_%": s.quantile(0.50),
            "p75_%": s.quantile(0.75),
            "p95_%": s.quantile(0.95),
            "p99_%": s.quantile(0.99),
            "max_%": s.max(),
            "pct_positive": 100.0 * float((s > 0).mean()),
            "pct_exactly_zero": 100.0 * float((s == 0).mean()),
        }
    return pd.DataFrame(rows).T


def describe_threshold_effect(returns: pd.DataFrame, thresholds: dict[int, tuple[float, float]]) -> pd.DataFrame:
    """Class balance implied by a threshold set (used for train/val/test tables)."""
    rows = {}
    for h in HORIZONS:
        lo, hi = thresholds[h]
        s = returns[future_return_col(h)].dropna()
        if s.empty:
            continue
        n = s.size
        n_down = int((s < lo).sum())
        n_up = int((s > hi).sum())
        n_neutral = n - n_down - n_up
        rows[HORIZON_LABELS[h]] = {
            "lower_%": lo,
            "upper_%": hi,
            "n": n,
            "DOWN_n": n_down,
            "DOWN_%": 100.0 * n_down / n,
            "NEUTRAL_n": n_neutral,
            "NEUTRAL_%": 100.0 * n_neutral / n,
            "UP_n": n_up,
            "UP_%": 100.0 * n_up / n,
        }
    return pd.DataFrame(rows).T


# ----------------------------------------------------------------------------
# Threshold selection  (TRAIN SPLIT ONLY)
# ----------------------------------------------------------------------------

def compute_thresholds(
    y_train_percent: np.ndarray,
    *,
    mode: str = THRESHOLD_MODE,
    neutral_quantile: float = NEUTRAL_QUANTILE,
    fixed_bps: dict[int, float] | None = None,
) -> dict[int, tuple[float, float]]:
    """
    Derive one (lower, upper) threshold pair per horizon, in PERCENT units.

    `y_train_percent` has shape (n_train, n_horizons) and MUST come from the
    training split only. Nothing here may ever see validation or test data.
    """
    fixed_bps = fixed_bps or FIXED_THRESHOLD_BPS
    thresholds: dict[int, tuple[float, float]] = {}

    if mode == "fixed_bps":
        for h in HORIZONS:
            half = fixed_bps[h] / 100.0          # bps -> percent
            thresholds[h] = (-half, half)
        return thresholds

    if mode != "train_quantile":
        raise ValueError(f"Unknown THRESHOLD_MODE: {mode!r}")

    tail = (1.0 - neutral_quantile) / 2.0
    for i, h in enumerate(HORIZONS):
        col = y_train_percent[:, i]
        col = col[~np.isnan(col)]
        if col.size == 0:
            raise ValueError(f"No training targets available for horizon {h}")
        lo = float(np.quantile(col, tail))
        hi = float(np.quantile(col, 1.0 - tail))
        if lo > hi:                              # degenerate; keep it sane
            lo, hi = min(lo, hi), max(lo, hi)
        thresholds[h] = (lo, hi)
    return thresholds


def classify_returns(
    y_percent: np.ndarray,
    thresholds: dict[int, tuple[float, float]],
) -> np.ndarray:
    """
    (n, n_horizons) percent returns -> (n, n_horizons) int8 class labels.
    NaN targets map to -1 so they stay trivially maskable.
    """
    n, k = y_percent.shape
    out = np.full((n, k), -1, dtype=np.int8)
    for i, h in enumerate(HORIZONS):
        lo, hi = thresholds[h]
        col = y_percent[:, i]
        valid = ~np.isnan(col)
        labels = np.full(col.shape, NEUTRAL, dtype=np.int8)
        labels[col < lo] = DOWN
        labels[col > hi] = UP
        out[valid, i] = labels[valid]
    return out


# ----------------------------------------------------------------------------
# Convenience for the CSV deliverable
# ----------------------------------------------------------------------------

def add_direction_columns(
    df: pd.DataFrame,
    thresholds: dict[int, tuple[float, float]],
) -> pd.DataFrame:
    out = df.copy()
    for h in HORIZONS:
        lo, hi = thresholds[h]
        ret = out[future_return_col(h)]
        label = pd.Series(NEUTRAL, index=out.index, dtype="float64")
        label = label.mask(ret < lo, DOWN).mask(ret > hi, UP)
        label = label.where(ret.notna())
        out[f"direction_{HORIZON_LABELS[h]}"] = label.astype("Int8")
    missing = [c for c in DIRECTION_COLUMNS if c not in out.columns]
    if missing:
        raise AssertionError(f"Direction registry out of sync: {missing}")
    return out


def thresholds_to_jsonable(thresholds: dict[int, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {
        HORIZON_LABELS[h]: {
            "lower_percent": thresholds[h][0],
            "upper_percent": thresholds[h][1],
            "lower_bps": thresholds[h][0] * 100.0,
            "upper_bps": thresholds[h][1] * 100.0,
        }
        for h in HORIZONS
    }


def class_names() -> list[str]:
    return [DIRECTION_CLASSES[i] for i in sorted(DIRECTION_CLASSES)]
