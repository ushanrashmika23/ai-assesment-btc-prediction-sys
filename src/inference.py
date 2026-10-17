"""
Inference: turn the newest Binance candles into a six-horizon trend read.

    Binance latest 15m candles
        -> same cleaning + feature code as training
        -> last 64 candles
        -> scaler.pkl (loaded, NEVER refitted)
        -> trained GRU/LSTM
        -> 15m/30m/45m/60m/75m/90m trend + confidence

Import layout
-------------
The first half of this module (fetching, cleaning, features, windowing,
scaling) is plain pandas/numpy/sklearn and has NO TensorFlow dependency - it is
the same code path the training pipeline used, which is exactly what makes the
inference consistent with training.

Only `load_model()` and `predict()` need the trained network, and TensorFlow is
imported *inside* `load_model()` so that importing this module on the local
machine stays TensorFlow-free.

    # local, TF-free: inspect the input the model would receive
    python -c "from inference import build_live_input; build_live_input()"

    # with a trained model present (Colab or a TF-enabled environment)
    python src/inference.py                      # the shipped ensemble
    python src/inference.py --model models/gru_s42_best.keras

This is a research helper, not a production service - there is no API, no
scheduler and no order execution here, by design.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    DIRECTION_CLASSES,
    FINAL_METADATA,
    FINAL_SCALER,
    HORIZON_LABELS,
    HORIZONS,
    INTERVAL,
    LOOKBACK,
    MODELS_DIR,
    REGIME_CLASSES,
    SYMBOL,
    WARMUP_ROWS,
)
from config import INFERENCE_BUNDLE
from data_download import download_rest_range, DownloadReport
from features import FEATURE_COLUMNS, build_features
from preprocessing import build_segment_ids, clean_candles, detect_gaps
from targets import add_future_returns

# Candles to pull so that every warmup-dependent feature is valid.
FETCH_CANDLES = WARMUP_ROWS + LOOKBACK + 100


# ----------------------------------------------------------------------------
# 1. Acquire the newest candles  (TF-free)
# ----------------------------------------------------------------------------

def fetch_latest_candles(n: int = FETCH_CANDLES) -> pd.DataFrame:
    """Most recent `n` 15m candles from the Binance REST API."""
    import time

    now_ms = int(time.time() * 1000)
    start_ms = now_ms - n * 15 * 60 * 1000

    report = DownloadReport()
    raw = download_rest_range(start_ms, now_ms, report)
    if raw.empty:
        raise RuntimeError(
            "Could not fetch live candles from Binance. Check network access "
            "to api.binance.com."
        )
    raw = raw.drop(columns=["_source"], errors="ignore")
    raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return raw.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 2. Prepare the model input  (TF-free, identical to the training path)
# ----------------------------------------------------------------------------

@dataclass
class LiveInput:
    X: np.ndarray              # (1, LOOKBACK, n_features) float32, scaled
    X_raw: np.ndarray          # same window, unscaled - for inspection
    last_timestamp: pd.Timestamp
    last_close: float
    n_candles: int


def build_live_input(candles: pd.DataFrame | None = None) -> LiveInput:
    """
    Run the newest candles through the SAME cleaning, feature and scaling code
    used for training, and return the single (1, 64, n_features) model input.

    The scaler is loaded from data/final/scaler.pkl and applied as-is. Fitting a
    new scaler here would silently corrupt every prediction.
    """
    if candles is None:
        candles = fetch_latest_candles()

    clean = clean_candles(candles, verbose=False)
    report = detect_gaps(clean)
    if report.n_gaps:
        print(f"[warn] {report.n_gaps} gap(s) in the fetched window; "
              f"the last contiguous segment is used.")

    featured = build_features(clean, verbose=False)
    featured = add_future_returns(featured, verbose=False)   # keeps columns aligned

    # Every feature must be non-NaN -> drop the warmup head.
    featured = featured.dropna(subset=FEATURE_COLUMNS)
    if len(featured) < LOOKBACK:
        raise RuntimeError(
            f"Only {len(featured)} usable candles after warmup; need {LOOKBACK}. "
            f"Fetch more history (increase FETCH_CANDLES)."
        )

    # Only use the newest contiguous segment.
    last_segment = featured["segment_id"].iloc[-1]
    segment = featured[featured["segment_id"] == last_segment]
    if len(segment) < LOOKBACK:
        raise RuntimeError(
            f"The newest contiguous segment has {len(segment)} candles; "
            f"need {LOOKBACK}. Fetch more history."
        )

    window = segment.tail(LOOKBACK)
    X_raw = window[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    with open(FINAL_SCALER, "rb") as fh:
        scaler = pickle.load(fh)

    meta = load_metadata()
    clip = meta["scaling"].get("clip_sigma") if meta else None

    X = scaler.transform(X_raw)
    if clip is not None:
        np.clip(X, -clip, clip, out=X)
    X = X.astype(np.float32)[None, :, :]          # (1, 64, n_features)

    return LiveInput(
        X=X,
        X_raw=X_raw.astype(np.float32),
        last_timestamp=window["timestamp"].iloc[-1],
        last_close=float(window["close"].iloc[-1]),
        n_candles=len(segment),
    )


def load_metadata() -> dict | None:
    import json
    if FINAL_METADATA.exists():
        with open(FINAL_METADATA, encoding="utf-8") as fh:
            return json.load(fh)
    return None


# ----------------------------------------------------------------------------
# 3. The trained model  (REQUIRES TENSORFLOW - not importable locally)
# ----------------------------------------------------------------------------

class Ensemble:
    """
    Several trained members, averaged as the mean of their softmax outputs.

    Deliberately the same rule `src/evaluate.py` and `src/train.py` apply. If
    inference averaged logits while evaluation averaged probabilities, the
    reported accuracy would not describe the thing this class predicts - the
    numbers would be about a model that is never actually used.
    """

    def __init__(self, members: list):
        self.members = members

    def predict(self, X, verbose: int = 0):
        acc: dict[str, np.ndarray] = {}
        for m in self.members:
            raw = m.predict(X, verbose=verbose)
            for k, v in raw.items():
                acc[k] = np.asarray(v) if k not in acc else acc[k] + np.asarray(v)
        return {k: v / len(self.members) for k, v in acc.items()}

    def count_params(self) -> int:
        """Per-member parameter count. A method, mirroring the Keras API."""
        return int(self.members[0].count_params()) if self.members else 0

    def __repr__(self) -> str:
        return f"Ensemble({len(self.members)} members)"


def resolve_model_paths(bundle_path: Path) -> list[Path]:
    """
    The member .keras files the bundle names, if they are all still present.

    A member that has gone missing means the bundle describes a model that no
    longer exists. That is worth refusing over rather than silently predicting
    from whatever subset happens to be on disk.
    """
    import json

    if not bundle_path.exists():
        return []
    with open(bundle_path, encoding="utf-8") as fh:
        bundle = json.load(fh)
    members = (bundle.get("ensemble") or {}).get("members") or []
    paths = [MODELS_DIR / m["path"] for m in members if m.get("path")]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"inference_bundle.json lists ensemble members that are gone: "
            f"{', '.join(missing)}. Re-run `python src/train.py` to rebuild "
            f"them, or pass --model to name a single .keras file."
        )
    return paths


def load_model(model_path: str | Path | None = None):
    """
    Load the shipped model: the ensemble the bundle names, or one .keras file.

    TensorFlow is imported HERE and nowhere else in the project. On a machine
    without TensorFlow this function raises a clear message instead of breaking
    the import of the rest of the module.
    """
    try:
        from tensorflow import keras          # noqa: PLC0415 - deliberate lazy import
    except ImportError as exc:                # pragma: no cover
        raise ImportError(
            "TensorFlow is required to load a trained model. It is intentionally "
            "not installed in the local data-pipeline environment - run inference "
            "in Google Colab, or install TensorFlow in a separate environment."
        ) from exc

    def _load(path: Path):
        # compile=False: the direction head uses a custom weighted loss, which
        # Keras cannot deserialise without the class being importable at load
        # time. Nothing here trains or calls .fit, so the compiled loss is
        # dead weight - and leaving it out is what makes the model portable.
        if not path.exists():
            raise FileNotFoundError(
                f"No trained model at {path}. Train one with "
                f"`python src/train.py`, or pass --model."
            )
        return keras.models.load_model(path, compile=False)

    if model_path is not None:
        return _load(Path(model_path))

    paths = resolve_model_paths(INFERENCE_BUNDLE)
    if not paths:
        raise FileNotFoundError(
            f"No single model was named and {INFERENCE_BUNDLE} names no "
            f"ensemble. Pass --model with a .keras file."
        )
    if len(paths) == 1:
        return _load(paths[0])
    return Ensemble([_load(p) for p in paths])


# ----------------------------------------------------------------------------
# 4. Prediction  (REQUIRES TENSORFLOW)
# ----------------------------------------------------------------------------

@dataclass
class HorizonPrediction:
    horizon: str
    trend: str
    confidence: float
    expected_return_pct: float
    activity: str = ""            # ACTIVE/QUIET, empty for non-regime horizons
    activity_confidence: float = 0.0


def predict_proba(model, live: LiveInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Raw multi-task output for one window.

    Returns `(returns_percent, direction_probs, regime_probs)` with shapes
    (n_horizons,), (n_horizons, 3) and (n_regime_horizons, 2). Kept as the
    single place the model is called, so the CLI, the dashboard and anything
    else cannot drift apart.
    """
    raw = model.predict(live.X, verbose=0)
    regime = raw.get("regime_head")
    return (np.asarray(raw["return_head"])[0],
            np.asarray(raw["direction_head"])[0],
            np.asarray(regime)[0] if regime is not None else np.empty((0, 2)))


def predict(model, live: LiveInput, *, min_confidence: float = 0.0) -> list[HorizonPrediction]:
    """
    Multi-task output -> one trend per horizon.

    `return_head` predicts the percentage return, `direction_head` gives the
    3-class probabilities, and `regime_head` answers the separate question the
    regime horizons were trained on: is the market about to be more active than
    its own recent average? A NEUTRAL call below `min_confidence` stays
    NEUTRAL - that is the "no-trade zone" the future backtest must honour.
    """
    returns, probs, regime_probs = predict_proba(model, live)

    # Which regime horizons the loaded model actually has. Read from the model
    # rather than assumed, so a model trained without the regime head still
    # renders instead of raising on a missing row.
    regime_meta = (load_metadata() or {}).get("regime_targets", {})
    regime_horizons = list(regime_meta.get("horizons", []))
    if regime_probs.shape[0] != len(regime_horizons):
        regime_horizons = []

    out: list[HorizonPrediction] = []
    for i, h in enumerate(HORIZONS):
        p = probs[i]
        cls = int(np.argmax(p))
        if float(p[cls]) < min_confidence:
            cls = 1                                     # NEUTRAL
        j = (regime_horizons.index(HORIZON_LABELS[h])
             if HORIZON_LABELS[h] in regime_horizons else None)
        out.append(
            HorizonPrediction(
                horizon=HORIZON_LABELS[h],
                trend=DIRECTION_CLASSES[cls],
                confidence=float(p[cls]),
                expected_return_pct=float(returns[i]),
                activity=(REGIME_CLASSES[int(np.argmax(regime_probs[j]))]
                          if j is not None else ""),
                activity_confidence=(float(np.max(regime_probs[j]))
                                     if j is not None else 0.0),
            )
        )
    return out


def render(predictions: list[HorizonPrediction], live: LiveInput) -> str:
    lines = [
        f"{SYMBOL} {INTERVAL} - trend read",
        f"last candle : {live.last_timestamp}   close {live.last_close:,.2f}",
        f"input shape : {live.X.shape}  (1, {LOOKBACK} candles, "
        f"{len(FEATURE_COLUMNS)} features)",
        "-" * 62,
        f"  {'horizon':>7}  {'direction':<9} {'conf':>6}   "
        f"{'activity':<8} {'conf':>6}",
        "-" * 62,
    ]
    for p in predictions:
        activity = (f"{p.activity:<8} {p.activity_confidence:6.1%}"
                    if p.activity else f"{'-':<8} {'':>6}")
        lines.append(
            f"  {p.horizon:>7}  {p.trend:<9} {p.confidence:6.1%}   {activity}"
            f"   (expected {p.expected_return_pct:+.3f}%)"
        )
    lines.append("-" * 62)
    lines.append(
        "The activity call is the one to read first. Measured on held-out data\n"
        "it is right 60-68% of the time against a ~50% baseline, because the two\n"
        "classes are balanced by construction. The direction call is close to a\n"
        "coin flip and beats always-NEUTRAL by only ~3pp - it is reported for\n"
        "completeness, not because it is a signal."
    )
    lines.append(
        "Not investment advice. A directional call is not a profitability "
        "claim;\nfees, slippage and regime change all sit between this output "
        "and any P&L."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BTCUSDT 15m trend inference")
    parser.add_argument("--model", default=None,
                        help="path to a single .keras file. Default: use the "
                             "ensemble named in models/inference_bundle.json, "
                             "which is the model the reported accuracy is about")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="below this, a directional call is downgraded to NEUTRAL")
    args = parser.parse_args()

    print("Fetching the newest candles ...")
    live = build_live_input()
    print(f"model input ready: {live.X.shape}\n")

    model = load_model(args.model)
    print(f"model: {model!r}\n")
    preds = predict(model, live, min_confidence=args.min_confidence)
    print(render(preds, live))


if __name__ == "__main__":
    main()
