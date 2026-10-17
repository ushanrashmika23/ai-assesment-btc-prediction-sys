"""
Market-regime (activity) targets: will the next window be ACTIVE or QUIET?

The question
------------
The direction head asks "which way?" and tops out around 53-56% on 15m BTC,
because the sign of a short-horizon move is close to unpredictable and the
answer is dominated by the NEUTRAL band.

This module builds a second target that answers a different question - one with
a genuine, measurable edge:

    "Is the market about to move a lot, or very little?"

For a horizon of h candles, measured at candle t:

    absr[t]               = |100 * (close[t] / close[t-1] - 1)|      [percent]
    forward_absmove_h[t]  = mean(absr[t+1 .. t+h])                   FUTURE
    reference_absmove[t]  = mean(absr[t-95 .. t])                    PAST (24h)
    regime_ratio_h[t]     = forward_absmove_h[t] / reference_absmove[t]

    label = ACTIVE (1)  if regime_ratio_h[t] >  tau_h
            QUIET  (0)  otherwise

`tau_h` is the median of `regime_ratio_h` over the TRAIN split only, so the two
classes are 50/50 by construction on train. Because both sides of the ratio are
mean absolute returns of the SAME asset over the SAME units, the label does not
drift when BTC's price level or its overall volatility regime changes: train,
validation and test all land near 50/50 (measured: 0.506 / 0.504 / 0.504).

Why the ratio, and not the raw forward volatility
-------------------------------------------------
A raw "forward realised volatility" target is not stationary - 2021 was far
more volatile than 2024, so a fixed threshold would produce a label that is
mostly ACTIVE in one era and mostly QUIET in another. Dividing by the sample's
own trailing activity removes exactly that drift, which is why the class
balance survives the train/validation/test split. The direction thresholds in
`targets.py` do NOT have this property, and the consequence is visible in the
shipped metadata: train is 30/40/30 while test is 23.7/53.8/22.5.

Leakage discipline
------------------
* `reference_absmove` is strictly backward-looking: it uses candles t-95..t, so
  it is known at the decision time t and is safe both as a label denominator
  and (if ever wanted) as a model input.
* `forward_absmove_h` touches only FUTURE candles, and only to build the label.
  It is never fed back into a feature.
* If the forward window crosses a data gap the target is invalidated to NaN
  rather than silently averaging over missing market data - the same rule
  `targets.py::add_future_returns` applies to the return targets.
* `tau_h` is derived from the TRAIN split only, in `compute_regime_thresholds`,
  and then frozen. Validation and test never influence it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    REGIME_COLUMNS,
    REGIME_HORIZONS,
    REGIME_REF_WINDOW,
    REGIME_THRESHOLD_QUANTILE,
    HORIZON_LABELS,
)

QUIET, ACTIVE = 0, 1


# ----------------------------------------------------------------------------
# Target construction
# ----------------------------------------------------------------------------

def _abs_return_percent(close: pd.Series) -> pd.Series:
    """|15m percentage return|. Index 0 has no predecessor -> NaN."""
    return (100.0 * (close / close.shift(1) - 1.0)).abs()


def add_regime_ratios(df: pd.DataFrame, *, verbose: bool = True) -> pd.DataFrame:
    """
    Attach `regime_ratio_<horizon>` for every horizon in REGIME_HORIZONS.

    Requires `close` and `segment_id`. The forward window is invalidated to NaN
    wherever it would span a segment change, so a ratio is never computed across
    a data gap.
    """
    missing = [c for c in ("close", "segment_id") if c not in df.columns]
    if missing:
        raise KeyError(f"regime targets need {missing} - pass the frame that "
                       f"still has the raw candle columns")

    out = df.copy()
    absr = _abs_return_percent(out["close"])
    seg = out["segment_id"]

    reference = absr.rolling(REGIME_REF_WINDOW, min_periods=REGIME_REF_WINDOW).mean()
    out["reference_absmove"] = reference

    for h in REGIME_HORIZONS:
        # Mean |return| over the NEXT h candles. `rolling(h).mean()` at row i
        # averages rows i-h+1..i, so shifting back by h lands that average on
        # row i-h - which is exactly "the h candles strictly after i".
        forward = absr.rolling(h, min_periods=1).mean().shift(-h)

        # A horizon that spans a data gap measures nothing real.
        same_segment = seg.shift(-h) == seg

        ratio = (forward / reference.where(reference > 0)).where(same_segment)
        out[f"regime_ratio_{HORIZON_LABELS[h]}"] = ratio

    if verbose:
        print(f"  [regime] {len(REGIME_HORIZONS)} horizons "
              f"({', '.join(HORIZON_LABELS[h] for h in REGIME_HORIZONS)}), "
              f"reference = trailing {REGIME_REF_WINDOW} candles "
              f"({REGIME_REF_WINDOW * 15 / 60:.0f}h)")
        print(f"  [regime] forward window = mean |15m return| over the next "
              f"1-{max(REGIME_HORIZONS)} candles; ratio = forward / reference")
    return out


# ----------------------------------------------------------------------------
# Threshold selection  (TRAIN SPLIT ONLY)
# ----------------------------------------------------------------------------

def compute_regime_thresholds(
    ratios_train: np.ndarray,
    *,
    quantile: float = REGIME_THRESHOLD_QUANTILE,
) -> np.ndarray:
    """
    One threshold per regime horizon, from the TRAIN ratios only.

    `ratios_train` is (n_train, len(REGIME_HORIZONS)). Returns (n_horizons,).
    The default quantile is the median, which makes the two classes exactly
    50/50 on train regardless of how volatile that era was.
    """
    ratios_train = np.asarray(ratios_train, dtype=np.float64)
    if ratios_train.ndim != 2 or ratios_train.shape[1] != len(REGIME_HORIZONS):
        raise ValueError(
            f"expected (n, {len(REGIME_HORIZONS)}) train ratios, got "
            f"{ratios_train.shape}"
        )
    thresholds = np.empty(len(REGIME_HORIZONS), dtype=np.float64)
    for i, h in enumerate(REGIME_HORIZONS):
        col = ratios_train[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            raise ValueError(f"no usable train regime ratios for horizon {h}")
        thresholds[i] = float(np.quantile(col, quantile))
    return thresholds


def classify_regime(ratios: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    (n, n_horizons) ratios -> (n, n_horizons) int8 class labels.

    Non-finite ratios map to -1 so they stay trivially maskable, matching the
    convention `targets.classify_returns` uses for the direction labels.
    """
    ratios = np.asarray(ratios, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if ratios.ndim != 2 or ratios.shape[1] != thresholds.size:
        raise ValueError(
            f"ratios {ratios.shape} and thresholds {thresholds.shape} disagree"
        )
    out = np.full(ratios.shape, -1, dtype=np.int8)
    finite = np.isfinite(ratios)
    labels = np.where(ratios > thresholds[None, :], ACTIVE, QUIET).astype(np.int8)
    out[finite] = labels[finite]
    return out


def regime_thresholds_to_jsonable(thresholds: np.ndarray) -> dict[str, dict[str, float]]:
    return {
        HORIZON_LABELS[h]: {
            "ratio_threshold": float(thresholds[i]),
            "quantile": REGIME_THRESHOLD_QUANTILE,
            "classes": {"0": "QUIET", "1": "ACTIVE"},
        }
        for i, h in enumerate(REGIME_HORIZONS)
    }


def add_regime_label_columns(
    df: pd.DataFrame,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Attach the integer regime labels to the human-readable dataset frame."""
    out = df.copy()
    for i, h in enumerate(REGIME_HORIZONS):
        col = f"regime_ratio_{HORIZON_LABELS[h]}"
        ratio = out[col]
        label = pd.Series(QUIET, index=out.index, dtype="float64")
        label = label.mask(ratio > thresholds[i], ACTIVE)
        out[f"regime_{HORIZON_LABELS[h]}"] = label.where(ratio.notna()).astype("Int8")
    missing = [c for c in REGIME_COLUMNS if c not in out.columns]
    if missing:
        raise AssertionError(f"regime registry out of sync: {missing}")
    return out


def regime_class_names() -> list[str]:
    from config import REGIME_CLASSES
    return [REGIME_CLASSES[i] for i in sorted(REGIME_CLASSES)]
