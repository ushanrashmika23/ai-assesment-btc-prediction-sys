"""
Model evaluation: validation and test metrics for a trained model.

    python src/evaluate.py                       # the canonical GRU, val + test
    python src/evaluate.py --split val           # validation only
    python src/evaluate.py --split test          # test only
    python src/evaluate.py --model models/gru_s42_best.keras
    python src/evaluate.py --all                 # every .keras in models/, compared
    python src/evaluate.py --csv models/eval.csv # also write the tables

    # test it on ONE moment you choose, and see what actually happened next
    python src/evaluate.py --at 2025-08-01
    python src/evaluate.py --at "2026-01-15 14:30"

    # test it on N random windows from the test split (seeded, reproducible)
    python src/evaluate.py --sample 20 --split test --seed 42

WHAT THIS REPORTS
    For each split: regression metrics against the naive zero-return baseline,
    direction metrics against the majority-class baseline, the per-class
    precision/recall/F1, a confusion matrix per horizon, and a calibration
    table that answers "is the confidence the model reports meaningful?".

WHAT THIS IS **NOT** FOR
    Choosing a model. Model selection is done on VALIDATION macro-F1, inside
    src/train.py. Reading test results and then going back to change something
    turns the test split into a second validation split, and the numbers stop
    meaning what they claim to mean. This script is read-only: it never picks
    anything, it only measures. If you run it, change the model because of what
    you saw, and run it again, that is on you - the leak is in the loop, not in
    the code.

    Re-running this on an unchanged model is harmless. The test split has been
    read once already by src/train.py, and nothing here feeds back.

TensorFlow is imported inside main(), never at module level, so the rest of
src/ stays importable on a machine without it (audit check 17).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from config import MODELS_DIR                                   # noqa: E402
from train import (                                             # noqa: E402
    CLASS_NAMES,
    REGIME_NAMES,
    Artefacts,
    bar,
    direction_metrics,
    make_sequence,
    naive_direction,
    naive_regime,
    regime_metrics,
    regression_metrics,
)

SPLIT_LABELS = {"train": "TRAIN", "val": "VALIDATION", "test": "TEST"}


# ---------------------------------------------------------------------------
# Rendering helpers  (ASCII only - a legacy Windows code page cannot encode
# box-drawing characters, which is a real bug this project already hit once)
# ---------------------------------------------------------------------------

def rule(char: str = "=", width: int = 80) -> str:
    return char * width


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray,
                     n_cls: int = 3) -> np.ndarray:
    """n_cls x n_cls count matrix, rows = actual class, cols = predicted."""
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for a in range(n_cls):
        for p in range(n_cls):
            cm[a, p] = int(np.sum((y_true == a) & (y_pred == p)))
    return cm


def render_confusion(cm: np.ndarray, title: str,
                     names: list[str] | None = None) -> list[str]:
    """Counts with the row percentage in brackets, so recall reads off directly."""
    names = names or CLASS_NAMES
    n = len(names)
    lines = [f"  {title}"]
    lines.append(f"    {'actual \\ pred':<14}"
                 + "".join(f"{nm:>12}" for nm in names) + f"{'recall':>10}")
    for a in range(n):
        row = cm[a]
        total = row.sum()
        cells = "".join(
            f"{row[p]:>7,} ({100.0 * row[p] / total:>3.0f}%)" if total else
            f"{0:>7,} (  0%)"
            for p in range(n)
        )
        rec = f"{100.0 * row[a] / total:>9.1f}%" if total else "        n/a"
        lines.append(f"    {names[a]:<14}{cells}{rec}")
    prec = []
    for p in range(n):
        col = cm[:, p].sum()
        prec.append(f"{names[p]} {100.0 * cm[p, p] / col:.1f}%" if col
                    else f"{names[p]} n/a")
    lines.append(f"    precision: {'   '.join(prec)}")
    return lines


def calibration_table(probs: np.ndarray, y_true: np.ndarray,
                      edges: tuple | None = None) -> pd.DataFrame:
    """
    Does a confident call actually tend to be a correct one?

    Buckets every (sample, horizon) prediction by the model's own top
    probability and reports the realised accuracy in each bucket. If accuracy
    rises with confidence, the probability output is informative; if it is flat,
    the confidence is decoration and `--min-confidence` is meaningless.

    Default edges depend on the number of classes: a 3-class head can never be
    less confident than 1/3, a 2-class head never less than 1/2, so bucketing a
    binary head from 1/3 would put every single prediction in the first bucket.
    """
    if edges is None:
        edges = ((1 / 3, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 1.01)
                 if probs.shape[-1] >= 3
                 else (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.01))
    conf = probs.max(axis=-1).ravel()
    pred = probs.argmax(axis=-1).ravel()
    truth = np.asarray(y_true).ravel()
    correct = (pred == truth)

    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        n = int(m.sum())
        if n == 0:
            continue
        rows.append({
            "bucket": f"[{lo:.2f}, {hi:.2f})",
            "n": n,
            "share": n / len(conf),
            "mean_conf": float(conf[m].mean()),
            "accuracy": float(correct[m].mean()),
            "gap": float(correct[m].mean() - conf[m].mean()),
        })
    return pd.DataFrame(rows).set_index("bucket")


def confidence_sweep(probs: np.ndarray, y_true: np.ndarray,
                     floors: tuple | None = None) -> pd.DataFrame:
    """
    What does a confidence floor actually buy you?

    THREE-CLASS HEADS get the dashboard's `c` key semantics: a call below the
    floor is downgraded to NEUTRAL, exactly as
    `inference.predict(min_confidence=...)` does. Reports accuracy, macro-F1 and
    how often the model still commits to a direction.

    TWO-CLASS HEADS have no abstain class, so a floor instead means SELECTIVE
    PREDICTION - answer only where the model is confident enough, and stay
    silent elsewhere. Reports accuracy and coverage on the answered subset,
    which is the question that actually matters for a binary call.

    The floor is a TUNABLE, so the number that should decide it is the
    VALIDATION column. Reading the test column to pick a floor would be tuning
    on the test set, which this project forbids.
    """
    binary = probs.shape[-1] < 3
    if floors is None:
        floors = ((0.0, 0.34, 0.38, 0.42, 0.46, 0.50, 0.55, 0.60, 0.70)
                  if not binary else
                  (0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 0.95))
    conf = probs.max(axis=-1)
    pred = probs.argmax(axis=-1)
    rows = {}
    for f in floors:
        if binary:
            # Selective prediction: score only the calls above the floor.
            accs, covs = [], []
            for i in range(y_true.shape[1]):
                keep = conf[:, i] >= f
                covs.append(float(keep.mean()))
                if not keep.any():
                    accs.append(float("nan"))
                    continue
                accs.append(float((pred[keep, i] == y_true[keep, i]).mean()))
            rows[f"{f:.2f}"] = {
                "accuracy": float(np.nanmean(accs)),
                "coverage": float(np.mean(covs)),
                "n_answered": int(round(np.mean(covs) * len(y_true))),
            }
            continue
        acted = np.where(conf >= f, pred, 1).astype(np.int64)
        f1s, accs = [], []
        for i in range(y_true.shape[1]):
            m = direction_metrics(y_true[:, [i]], acted[:, [i]], ["h"])
            f1s.append(float(m.loc["h", "macro_F1"]))
            accs.append(float(m.loc["h", "accuracy"]))
        rows[f"{f:.2f}"] = {
            "macro_F1": float(np.mean(f1s)),
            "accuracy": float(np.mean(accs)),
            "directional_share": float(np.mean(acted != 1)),
        }
    return pd.DataFrame(rows).T


def window_span(ts: np.ndarray) -> str:
    a = pd.to_datetime(int(ts[0]), unit="ms", utc=True)
    b = pd.to_datetime(int(ts[-1]), unit="ms", utc=True)
    return f"{a:%Y-%m-%d} -> {b:%Y-%m-%d}"


def utc(when: str | pd.Timestamp) -> pd.Timestamp:
    """Parse a user-supplied date to a tz-aware UTC timestamp."""
    t = pd.Timestamp(when)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


# ---------------------------------------------------------------------------
# Single-window spot check: "what did the model say here, and was it right?"
#
# This is the part that answers "I need to test the model". The aggregate
# tables above say how it does on average; this says what it does on one
# specific moment you choose, next to what actually happened.
#
# X_*.npy already holds SCALED features (the pipeline scaled them), so a single
# window is just X[starts[k] : starts[k] + lookback] - no scaler involved, and
# no way to accidentally refit one.
# ---------------------------------------------------------------------------

def one_window(art: Artefacts, split: str, k: int) -> tuple[np.ndarray, float]:
    """The scaled (1, lookback, n_f) input for sample k, plus its window-end close."""
    X = getattr(art, f"X_{split}")
    starts = getattr(art, f"s_{split}")
    lo = int(starts[k])
    win = X[lo: lo + art.lookback]
    assert win.shape[0] == art.lookback, "window runs past the block end"
    return win[None, :, :].astype(np.float32), float(win[-1, -1])


def locate(art: Artefacts, when: str) -> tuple[str, int, pd.Timestamp]:
    """
    The sample whose DECISION time is nearest to `when`, and which split holds it.

    `ts_*.npy` holds the timestamp of the last candle in the window - the
    moment the decision is made - NOT the time being predicted. Verified
    against the raw closes: for every horizon h, close[ts + h] / close[ts] - 1
    reproduces y[:, i] exactly. So a 15m call made at 12:00 is settled by the
    12:15 candle.

    Searching all three splits is not leakage: this is a read-only lookup of
    where a date lives, and it is how the caller finds out the date is in
    training data rather than the test set - which they need to know to read
    the result correctly.
    """
    want = utc(when)
    best = None
    for split in ("train", "val", "test"):
        ts = getattr(art, f"ts_{split}")
        idx = int(np.clip(np.searchsorted(ts, want.value // 10**6), 0, len(ts) - 1))
        for cand in (idx - 1, idx):
            if 0 <= cand < len(ts):
                delta = abs(pd.to_datetime(int(ts[cand]), unit="ms", utc=True)
                            - want)
                if best is None or delta < best[0]:
                    best = (delta, split, cand,
                            pd.to_datetime(int(ts[cand]), unit="ms", utc=True))
    _, split, k, actual_ts = best
    return split, k, actual_ts


def describe_window(art: Artefacts, split: str, k: int, model, keras,
                    show_all: bool = True) -> bool:
    """
    Predict one historical window and score it against the recorded outcome.
    Returns True if the 15m direction call was correct.
    """
    X, _ = one_window(art, split, k)
    raw = model.predict(X, verbose=0)
    pred_ret = art.from_scaled(np.asarray(raw["return_head"]))[0]
    probs = np.asarray(raw["direction_head"])[0]

    y = getattr(art, f"y_{split}")[k]
    d = getattr(art, f"d_{split}")[k]
    ts = getattr(art, f"ts_{split}")[k]
    decided = pd.to_datetime(int(ts), unit="ms", utc=True)

    print()
    print(f" decided at   {decided:%Y-%m-%d %H:%M} UTC   "
          f"(last of {art.lookback} candles = "
          f"{art.lookback * 15 / 60:.0f}h of history)")
    settled_by = ", ".join(
        f"{hz} {decided + pd.Timedelta(minutes=15 * c):%H:%M}"
        for hz, c in ((art.horizons[0], art.horizon_candles[0]),
                      (art.horizons[-1], art.horizon_candles[-1])))
    print(f" settled      {settled_by} UTC")
    print(f" split        {SPLIT_LABELS[split]}   (sample #{k:,})")
    print(f" {'horizon':>8}  {'model says':<18}{'conf':>6}   "
          f"{'predicted':>10} {'actual':>10}   {'actual class':<14}")
    print(f" {'-' * 8}  {'-' * 18}{'-' * 6}   {'-' * 10} {'-' * 10}   "
          f"{'-' * 14}")

    hits = 0
    for i, hz in enumerate(art.horizons):
        p = probs[i]
        cls = int(np.argmax(p))
        correct = cls == int(d[i])
        hits += correct
        mark = "ok " if correct else "MISS"
        print(f" {hz:>8}  {CLASS_NAMES[cls]:<8} {mark:<9}"
              f"{p[cls]:>6.1%}   {pred_ret[i]:>+9.3f}% {y[i]:>+9.3f}%   "
              f"{CLASS_NAMES[int(d[i])]:<14}")
        if show_all:
            ordered = "  ".join(f"{CLASS_NAMES[c]} {p[c]:.2f}"
                                for c in np.argsort(-p))
            print(f" {'':>8}  {ordered}")
    print(f" {'-' * 8}")
    print(f" direction: {hits}/{len(art.horizons)} horizons correct "
          f"(chance-level would be about {int(round(0.43 * len(art.horizons)))}"
          f"/{len(art.horizons)} by always saying NEUTRAL)")
    return bool(probs[0].argmax() == d[0])


def spot_check(art: Artefacts, model, keras, split: str, n: int, seed: int,
               when: str | None) -> None:
    """Either one chosen moment, or a seeded random sample of them."""
    ts = getattr(art, f"ts_{split}")

    if when is not None:
        found_split, k, actual_ts = locate(art, when)
        print()
        print(rule())
        print(f" SPOT CHECK  -  nearest sample to {utc(when):%Y-%m-%d %H:%M} UTC")
        print(rule())
        if found_split != split:
            print(f" note: that date falls in the {SPLIT_LABELS[found_split]} "
                  f"split, not {SPLIT_LABELS[split]} - using it.")
        if found_split == "train":
            print(" note: this window is IN THE TRAINING DATA. The model has "
                  "seen it.\n"
                  "       A correct call here is memorisation, not evidence.")
        print(f" nearest decision candle: {actual_ts:%Y-%m-%d %H:%M} UTC")
        print(f"                          (that is the last candle IN the "
              f"window, not the time being predicted)")
        describe_window(art, found_split, k, model, keras)
        return

    print()
    print(rule())
    print(f" SPOT CHECK  -  {n} random windows from {SPLIT_LABELS[split]}")
    print(rule())
    print(f" seed {seed} - rerun with --seed {seed} to get these exact windows.")
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ts), size=min(n, len(ts)), replace=False))
    hits = [describe_window(art, split, int(k), model, keras, show_all=False)
            for k in idx]
    print()
    print(rule("-"))
    print(f" {sum(hits)}/{len(hits)} of these windows called 15m direction "
          f"correctly.")
    base = float((getattr(art, f"d_{split}")[:, 0] ==
                  naive_direction(getattr(art, f"d_{split}").shape,
                                  art.d_train)[:, 0]).mean())
    print(f" always-NEUTRAL baseline on all of {SPLIT_LABELS[split]}: {base:.1%}")
    if len(hits) < 20:
        print(f" {len(hits)} windows is far too few to conclude anything - this "
              f"is a sanity check, not a measurement.")
    print(rule("-"))


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def predict_split(model, art: Artefacts, keras, split: str, batch: int):
    """
    Run the model over one split.

    Returns (returns_pct, probs, y, y_dir, regime_probs) - regime_probs is None
    unless the model has a regime head.
    """
    X = getattr(art, f"X_{split}")
    s = getattr(art, f"s_{split}")
    y = getattr(art, f"y_{split}")
    d = getattr(art, f"d_{split}")

    seq = make_sequence(keras, X, art.to_scaled(y), d, s, art.lookback, batch)
    raw = model.predict(seq, verbose=0)

    returns = art.from_scaled(np.asarray(raw["return_head"]))
    probs = np.asarray(raw["direction_head"])
    r_probs = (np.asarray(raw["regime_head"]) if "regime_head" in raw else None)
    return returns, probs, y, d, r_probs


def evaluate_split(art: Artefacts, model, keras, split: str, batch: int,
                   verbose: bool = True) -> dict:
    returns, probs, y, d, r_probs = predict_split(model, art, keras, split,
                                                  batch)
    pred_dir = probs.argmax(axis=-1).astype(np.int64)

    naive_reg = regression_metrics(y, np.zeros_like(y), art.horizons)
    naive_dir = direction_metrics(
        d, naive_direction(d.shape, art.d_train), art.horizons)
    m_reg = regression_metrics(y, returns, art.horizons)
    m_dir = direction_metrics(d, pred_dir, art.horizons)

    calib = calibration_table(probs, d)

    result = {
        "split": split,
        "n": int(len(y)),
        "span": window_span(getattr(art, f"ts_{split}")),
        "regression": m_reg,
        "regression_naive": naive_reg,
        "direction": m_dir,
        "direction_naive": naive_dir,
        "calibration": calib,
        "sweep": confidence_sweep(probs, d),
        "confusion": {h: confusion_counts(d[:, i], pred_dir[:, i]).tolist()
                      for i, h in enumerate(art.horizons)},
    }

    # The regime head, when present. Its accuracy is the honest headline: the
    # two classes are ~50/50 by construction, so there is no majority class to
    # hide behind the way NEUTRAL hides the direction head.
    r_true = getattr(art, f"r_{split}", None) if art.n_regime else None
    if r_probs is not None and r_true is not None:
        r_pred = r_probs.argmax(axis=-1).astype(np.int64)
        result["regime"] = regime_metrics(r_true, r_pred,
                                          art.regime_horizons)
        result["regime_naive"] = regime_metrics(
            r_true, naive_regime(r_true.shape, art.r_train),
            art.regime_horizons)
        result["regime_sweep"] = confidence_sweep(r_probs, r_true)
        result["regime_confusion"] = {
            h: confusion_counts(r_true[:, i], r_pred[:, i],
                                n_cls=2).tolist()
            for i, h in enumerate(art.regime_horizons)}

    if verbose:
        print_split_report(art, result)
    return result


def print_split_report(art: Artefacts, r: dict) -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    h, n = SPLIT_LABELS[r["split"]], r["n"]
    print()
    print(rule())
    print(f" {h}  -  {n:,} samples  -  {r['span']}")
    print(rule())

    m_reg, n_reg = r["regression"], r["regression_naive"]
    m_dir, n_dir = r["direction"], r["direction_naive"]

    # -- regime first: this is the head whose accuracy means something --------
    if "regime" in r:
        m_rg, n_rg = r["regime"], r["regime_naive"]
        print("\n REGIME  (QUIET vs ACTIVE - THE ACCURACY HEADLINE)")
        print("   'will the mean |15m return| over the next h candles exceed its")
        print("    own trailing 24h average?'  Both classes are ~50% by")
        print("    construction, so 0.50 IS the baseline and the lift is real.")
        tbl = pd.DataFrame({
            "majority_acc": n_rg["accuracy"],
            "model_acc": m_rg["accuracy"],
            "lift": m_rg["accuracy"] - n_rg["accuracy"],
            "model_macroF1": m_rg["macro_F1"],
            "base_rate": m_rg["base_rate"],
        })
        print(tbl.to_string(float_format=lambda v: f"{v:,.4f}"))
        for hz in art.regime_horizons:
            a = m_rg.loc[hz, "accuracy"]
            print(f"   {hz:>4}  accuracy {a:.4f}  {bar(a)}  "
                  f"{a - n_rg.loc[hz, 'accuracy']:+.4f} vs baseline")

        print("\n REGIME PER-CLASS DETAIL")
        cols = ["QUIET_prec", "QUIET_rec", "QUIET_f1",
                "ACTIVE_prec", "ACTIVE_rec", "ACTIVE_f1"]
        print(m_rg[cols].to_string(float_format=lambda v: f"{v:,.4f}"))

        print("\n REGIME CONFUSION  (rows = actual, cols = predicted)")
        for i, hz in enumerate(art.regime_horizons):
            print()
            for line in render_confusion(np.array(r["regime_confusion"][hz]),
                                         f"{hz} horizon", REGIME_NAMES):
                print(line)

        print("\n REGIME SELECTIVE ACCURACY  (answer only above a confidence "
              "floor)")
        sw = r["regime_sweep"]
        print(sw.to_string(float_format=lambda v: f"{v:,.4f}"))
        if r["split"] == "val":
            print("   -> choose a floor from THIS column (it is a tunable).")
        else:
            print("   -> NOT a column to choose a floor from - that would be "
                  "tuning on test.")
        print("   -> coverage is the share of calls you would still make; the "
              "trade is")
        print("      explicit: higher floor, higher accuracy, fewer answers.")

    print("\n DIRECTION  (the harder question - reported on macro-F1, not "
          "accuracy)")
    m_dir, n_dir = r["direction"], r["direction_naive"]

    print("\n REGRESSION  (percent return; lower MAE is better)")
    tbl = pd.DataFrame({
        "naive_mean_MAE": n_reg["MAE_%"],
        "model_MAE": m_reg["MAE_%"],
        "delta": m_reg["MAE_%"] - n_reg["MAE_%"],
        "model_RMSE": m_reg["RMSE_%"],
        "bias": m_reg["bias_%"],
        "corr": m_reg["corr"],
    })
    print(tbl.to_string(float_format=lambda v: f"{v:,.6f}"))
    worse = int((m_reg["MAE_%"] > n_reg["MAE_%"]).sum())
    if worse == len(art.horizons):
        print("   -> WORSE than predicting the mean at every horizon.")
        print("      The regression head has no usable skill. Do not read the")
        print("      'expected' number in the dashboard as a forecast.")
    elif worse:
        print(f"   -> worse than the mean at {worse}/{len(art.horizons)} "
              f"horizons.")
    else:
        print("   -> beats predicting the mean at every horizon.")

    tbl = pd.DataFrame({
        "majority_acc": n_dir["accuracy"],
        "model_acc": m_dir["accuracy"],
        "majority_macroF1": n_dir["macro_F1"],
        "model_macroF1": m_dir["macro_F1"],
        "lift": m_dir["macro_F1"] / n_dir["macro_F1"],
        "n": m_dir["n"].astype(int),
    })
    print(tbl.to_string(float_format=lambda v: f"{v:,.4f}"))
    for hz in art.horizons:
        f1 = m_dir.loc[hz, "macro_F1"]
        base = n_dir.loc[hz, "macro_F1"]
        mark = "beats" if f1 > base else "LOSES TO"
        print(f"   {hz:>4}  macro-F1 {f1:.4f}  {bar(f1)}  {mark} baseline "
              f"{base:.4f}")

    print("\n PER-CLASS DETAIL  (precision / recall / F1)")
    cols = ["DOWN_prec", "DOWN_rec", "DOWN_f1",
            "NEUTRAL_prec", "NEUTRAL_rec", "NEUTRAL_f1",
            "UP_prec", "UP_rec", "UP_f1"]
    print(m_dir[cols].to_string(float_format=lambda v: f"{v:,.4f}"))
    collapsed = [c for c in ("DOWN_rec", "NEUTRAL_rec", "UP_rec")
                 if m_dir[c].max() < 0.05]
    if collapsed:
        print(f"   -> WARNING: near-collapsed class(es): "
              f"{', '.join(c.split('_')[0] for c in collapsed)}")
    else:
        print("   -> every class is predicted at least sometimes (recall > 5% "
              "somewhere).")
    print("   -> low DOWN/UP recall with high NEUTRAL recall means the model "
          "mostly says NEUTRAL.")

    print("\n CONFUSION  (rows = actual, cols = predicted; counts and row %)")
    for i, hz in enumerate(art.horizons):
        print()
        for line in render_confusion(np.array(r["confusion"][hz]),
                                     f"{hz} horizon"):
            print(line)

    print("\n CALIBRATION  (is the confidence meaningful?)")
    cal = r["calibration"]
    print(cal.to_string(float_format=lambda v: f"{v:,.4f}"))
    if len(cal) >= 2 and cal["accuracy"].iloc[-1] > cal["accuracy"].iloc[0]:
        print("   -> accuracy rises with confidence: the probability output "
              "carries information.")
    else:
        print("   -> accuracy does NOT clearly rise with confidence. Treat the "
              "confidence as unreliable.")
    print("   -> 'gap' = accuracy - mean confidence. Negative means "
          "overconfident.")

    print("\n CONFIDENCE FLOOR SWEEP  (what the dashboard's 'c' key buys you)")
    sw = r["sweep"]
    base_acc = r["direction_naive"]["accuracy"].mean()
    print(sw.to_string(float_format=lambda v: f"{v:,.4f}"))
    print(f"   always-NEUTRAL accuracy on this split: {base_acc:.4f}")
    if r["split"] == "val":
        print("   -> THIS is the column to choose a floor from. It is a "
              "tunable.")
    else:
        print("   -> Do NOT choose a floor from this. Choosing on test would "
              "be tuning on test.")
    better = sw[sw["accuracy"] > base_acc].index.tolist()
    if better:
        print(f"   -> floors that beat always-NEUTRAL on accuracy: "
              f"{', '.join(better)}")
    else:
        print("   -> no floor beats always-NEUTRAL on accuracy here. Acting "
              "on this model's")
        print("      direction costs accuracy at every confidence level; "
              "macro-F1 is where it wins.")
    d0 = sw["directional_share"].iloc[0]
    dN = sw["directional_share"].iloc[-1]
    print(f"   -> commitment falls from {d0:.1%} of calls to {dN:.1%} across "
          f"the sweep.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_keras_model(path: Path):
    from tensorflow import keras                                # noqa: PLC0415
    # compile=False: the direction loss is a locally-defined WeightedSCC that
    # Keras cannot resolve by name. Nothing here trains, so it is not wanted.
    return keras.models.load_model(path, compile=False)


class EnsembleModel:
    """
    Several trained members presented as one model.

    Averaging the members' softmax probabilities is the recipe train.py
    declares in inference_bundle.json ("combine": "mean of per-member softmax
    probabilities"). This class exists so every caller - predict_split,
    one_window, the dashboard - keeps calling `.predict()` on one object and
    gets the ensemble's answer without knowing there is more than one file.

    Members are loaded lazily and released after each predict, because a full
    ensemble on CPU is several hundred MB of weights.
    """

    def __init__(self, paths: list[Path], architecture: str = "ENSEMBLE"):
        if not paths:
            raise ValueError("an ensemble needs at least one member")
        self.paths = list(paths)
        self.architecture = architecture
        self.name = f"{architecture.lower()}_x{len(paths)}"

    def _member(self, path):
        from tensorflow import keras                            # noqa: PLC0415
        return keras.models.load_model(path, compile=False)

    def predict(self, x, verbose: int = 0):
        total: dict[str, np.ndarray] = {}
        for p in self.paths:
            m = self._member(p)
            out = m.predict(x, verbose=0)
            for k, v in out.items():
                v = np.asarray(v, dtype=np.float64)
                total[k] = v if k not in total else total[k] + v
            del m
        n = len(self.paths)
        return {k: (v / n).astype(np.float32) for k, v in total.items()}

    def summary(self, **kwargs):                                # noqa: D102
        from tensorflow import keras                            # noqa: PLC0415
        self._member(self.paths[0]).summary(**kwargs)

    def count_params(self) -> int:
        """Method, not a property - mirrors the Keras model API deliberately."""
        return sum(self._member(p).count_params() for p in self.paths)

    @property
    def output_shape(self):
        return self._member(self.paths[0]).output_shape


def resolve_bundle_members(bundle_path: Path) -> list[Path] | None:
    """
    The member .keras files train.py declared, if all of them are still there.

    A missing member means the bundle describes a model that no longer exists,
    which is worth saying out loud rather than silently evaluating a subset.
    """
    if not bundle_path.exists():
        return None
    with open(bundle_path, encoding="utf-8") as fh:
        b = json.load(fh)
    members = (b.get("ensemble") or {}).get("members") or []
    if not members:
        return None
    paths = [MODELS_DIR / m["path"] for m in members]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        print(f" WARNING: inference_bundle.json lists ensemble members that are "
              f"gone: {', '.join(missing)}\n"
              f"          re-run `python src/train.py` to rebuild them.")
        return None
    return paths


def describe_model(path: Path, model) -> None:
    print(rule())
    print(" MODEL EVALUATION")
    print(rule())
    if isinstance(model, EnsembleModel):
        print(f" model            ENSEMBLE of {len(model.paths)} members")
        for p in model.paths:
            print(f"                    - {p.name}  "
                  f"({p.stat().st_size / 1024:,.0f} KB)")
        print(f" combine          mean of per-member softmax probabilities")
    else:
        print(f" model            {path.name}  "
              f"({path.stat().st_size / 1024:,.0f} KB)")
    n_members = len(model.paths) if isinstance(model, EnsembleModel) else 1
    extra = " (per member)" if n_members > 1 else ""
    print(f" parameters       {model.count_params():,}{extra}")
    if isinstance(model.output_shape, dict):
        outs = ", ".join(f"{k} {tuple(v.shape[1:])}"
                         for k, v in model.output_shape.items())
        print(f" outputs          {outs}")
    bundle = MODELS_DIR / "inference_bundle.json"
    if bundle.exists():
        with open(bundle, encoding="utf-8") as fh:
            b = json.load(fh)
        for k, label in (("architecture", "architecture"),
                         ("trained_utc", "trained (UTC)"),
                         ("trained_by", "trained by"),
                         ("seed", "seed"),
                         ("lookback", "lookback"),
                         ("train_samples_used", "train samples"),
                         ("subset_used", "subset used")):
            if k in b:
                print(f" {label:<16} {b[k]}")
        sel = b.get("selection") or {}
        if sel.get("validation_scores"):
            print(f" {'chosen on':<16} {sel.get('metric')}  "
                  f"{sel['validation_scores']}")


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Validation and test metrics for a trained model.")
    p.add_argument("--model", default=None,
                   help="path to one .keras file (default: the canonical GRU, "
                        "chosen by the same ranking the dashboard uses)")
    p.add_argument("--all", action="store_true",
                   help="evaluate every .keras in models/ and compare them")
    p.add_argument("--no-ensemble", action="store_true",
                   help="ignore the ensemble in inference_bundle.json and "
                        "evaluate a single member")
    p.add_argument("--split", default="both",
                   choices=["val", "test", "both", "train"])
    p.add_argument("--at", default=None, metavar="WHEN",
                   help="spot-check the window nearest this date, e.g. "
                        "--at 2025-08-01 or --at '2025-08-01 14:30'. Also "
                        "reports what actually happened next.")
    p.add_argument("--sample", type=int, default=None, metavar="N",
                   help="spot-check N random windows from --split (use with "
                        "--split test; pair with --seed to reproduce)")
    p.add_argument("--seed", type=int, default=42,
                   help="seed for --sample (default 42)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--csv", default=None,
                   help="also write the per-horizon tables to this CSV")
    p.add_argument("--json-out", default=None,
                   help="write the full result dict to this JSON file")
    p.add_argument("--no-save", action="store_true",
                   help="do not refresh models/eval_summary.json (the dashboard "
                        "reads it for its measured caveats)")
    args = p.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    available = sorted(p.resolve() for p in MODELS_DIR.glob("*.keras"))
    if not available:
        raise SystemExit(
            f"No trained model in {MODELS_DIR}.\n"
            f"Train one first:  python src/train.py --arch gru")

    # An ensemble is the default when train.py left one, because that is the
    # artefact whose numbers were reported. --all compares the individual
    # members on equal footing instead.
    bundle_members = (None if (args.model or args.all or args.no_ensemble)
                      else resolve_bundle_members(MODELS_DIR /
                                                  "inference_bundle.json"))

    # Everything downstream works on a list of (name, model) pairs. A single
    # .keras is just a one-element list, so the ensemble and the single-model
    # paths share every line of the report.
    if bundle_members:
        print(f"inference_bundle.json declares an ensemble of "
              f"{len(bundle_members)} members - evaluating it as ONE model.")
        print("use --no-ensemble to evaluate a single member instead.")
        todo: list[tuple[str, object, Path]] = [
            (f"ensemble_x{len(bundle_members)}",
             EnsembleModel(bundle_members), bundle_members[0])]
    else:
        if args.model:
            want = Path(args.model).resolve()
            if not want.exists():
                raise SystemExit(f"No model at {want}")
            paths = [want]
        elif args.all:
            paths = available
        else:
            # The dashboard's default rule: prefer a multitask file, prefer
            # GRU, then alphabetical. Not a claim that this is the best model -
            # train.py already chose on validation.
            def rank(m: Path) -> tuple:
                return ("multitask" not in m.name, "gru" not in m.name, m.name)

            paths = [min(available, key=rank)]
            if len(available) > 1:
                others = ", ".join(p.name for p in available if p != paths[0])
                print(f"evaluating {paths[0].name} (the dashboard's default). "
                      f"Also present: {others}")
                print(f"use --all to evaluate every model, or --model to pick "
                      f"one.")
        todo = [(p.name, None, p) for p in paths]   # None = load lazily below

    art = Artefacts()
    print(f"tensorflow {tf.__version__}  |  keras {keras.__version__}  |  "
          f"device: {'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'}")
    print(f"artefacts: {len(art.y_train):,} train / {len(art.y_val):,} val / "
          f"{len(art.y_test):,} test   |   purge clearance "
          f"{art.purge_clearance()} candle(s)")
    print(f"scaler and thresholds are LOADED from data/final/, never refitted.")

    # -- spot check mode: no aggregate tables, just chosen windows ---------
    if args.at or args.sample:
        if len(todo) > 1:
            raise SystemExit("--at/--sample need one model; drop --all or "
                             "pass --model.")
        name, model, path = todo[0]
        model = model if model is not None else load_keras_model(path)
        describe_model(path, model)
        split = "test" if args.split == "both" else args.split
        spot_check(art, model, keras, split, args.sample or 0, args.seed,
                   args.at)
        return 0

    splits = ["val", "test"] if args.split == "both" else [args.split]

    if "test" in splits:
        print()
        print(" NOTE ON THE TEST SPLIT")
        print("   This report is read-only; it selects nothing. Model choice")
        print("   belongs to validation macro-F1 (src/train.py). If you change")
        print("   the model because of a number below and re-run, the test")
        print("   split has become a second validation split.")

    all_results: dict = {}
    for name, model, path in todo:
        model = model if model is not None else load_keras_model(path)
        if len(todo) > 1:
            print()
            print(rule())
            print(f" {name}")
            print(rule())
        else:
            describe_model(path, model)

        per_split = {}
        for sp in splits:
            per_split[sp] = evaluate_split(art, model, keras, sp, args.batch)
        all_results[name] = per_split
        del model

    # -- comparison table when more than one model ------------------------
    if len(todo) > 1:
        print()
        print(rule())
        print(" COMPARISON  (choose on VALIDATION, never on test)")
        print(rule())
        rows = {}
        for name, per_split in all_results.items():
            rows[name] = {}
            for sp, r in per_split.items():
                rows[name][f"{sp}_macroF1"] = r["direction"]["macro_F1"].mean()
                rows[name][f"{sp}_acc"] = r["direction"]["accuracy"].mean()
                rows[name][f"{sp}_MAE"] = r["regression"]["MAE_%"].mean()
        cmp_df = pd.DataFrame(rows).T
        print(cmp_df.to_string(float_format=lambda v: f"{v:,.4f}"))
        if "val" in splits:
            best = cmp_df["val_macroF1"].idxmax()
            print(f"\n   best validation macro-F1: {best}")
            print("   A small gap between models is a hypothesis, not a "
                  "conclusion - re-run with other seeds before believing it.")

    # -- final verdict -----------------------------------------------------
    print()
    print(rule())
    print(" VERDICT")
    print(rule())
    for name, per_split in all_results.items():
        for sp, r in per_split.items():
            m_reg, n_reg = r["regression"], r["regression_naive"]
            m_dir, n_dir = r["direction"], r["direction_naive"]
            worse = int((m_reg["MAE_%"] > n_reg["MAE_%"]).sum())
            beat = int((m_dir["macro_F1"] > n_dir["macro_F1"]).sum())
            k = len(art.horizons)
            print(f" {name} [{SPLIT_LABELS[sp]}]")
            if "regime" in r:
                m_rg, n_rg = r["regime"], r["regime_naive"]
                rk = len(art.regime_horizons)
                rb = int((m_rg["accuracy"] > n_rg["accuracy"]).sum())
                lift = (m_rg["accuracy"] - n_rg["accuracy"]).mean()
                print(f"   regime head    : beats the majority-class accuracy "
                      f"at {rb}/{rk} horizons, mean lift {lift:+.4f} "
                      f"- THE HEADLINE")
            print(f"   direction head : beats the majority-class macro-F1 at "
                  f"{beat}/{k} horizons")
            if worse == k:
                print(f"   regression head: WORSE than predicting the mean at "
                      f"{k}/{k} horizons - no usable skill")
            elif worse:
                print(f"   regression head: worse than the mean at {worse}/{k} "
                      f"horizons")
            else:
                print(f"   regression head: beats the mean at {k}/{k} horizons")
    print("""
 Regime accuracy is not profitability either. 'This window will be more volatile
 than the last 24 hours' is not a trade: there is no position, no backtest, no
 fees and no slippage here, and the test period is one regime. Nothing above is
 a promise about the future.""")

    # -- summary file, so the dashboard can cite measured numbers ----------
    # Small and deliberately narrow: only what src/dashboard.py needs to warn
    # about the confidence floor. The full tables go to --json-out/--csv.
    if not args.no_save:
        summary = {"generated_utc": pd.Timestamp.now("UTC").isoformat()}
        for name, per_split in all_results.items():
            entry = {}
            for sp, r in per_split.items():
                sw = r["sweep"]
                entry[sp] = {
                    # Persisted because a reader of this file cannot recover
                    # it: the split sizes live in scripts that recompute them,
                    # and a dashboard showing held-out accuracy without saying
                    # how many samples it rests on overstates it.
                    "n_samples": int(r["n"]),
                    "macro_F1": float(r["direction"]["macro_F1"].mean()),
                    "accuracy": float(r["direction"]["accuracy"].mean()),
                    "baseline_accuracy": float(
                        r["direction_naive"]["accuracy"].mean()),
                    "sweep": {k: {kk: float(vv) for kk, vv in v.items()}
                              for k, v in sw.to_dict("index").items()},
                    "best_floor_by_macro_f1": str(sw["macro_F1"].idxmax()),
                }
                if "regime" in r:
                    m_rg, n_rg = r["regime"], r["regime_naive"]
                    rsw = r["regime_sweep"]
                    entry[sp]["regime"] = {
                        "accuracy": float(m_rg["accuracy"].mean()),
                        "baseline_accuracy": float(n_rg["accuracy"].mean()),
                        "lift": float((m_rg["accuracy"]
                                       - n_rg["accuracy"]).mean()),
                        "macro_F1": float(m_rg["macro_F1"].mean()),
                        "per_horizon": {
                            hz: {
                                "accuracy": float(m_rg.loc[hz, "accuracy"]),
                                "baseline": float(n_rg.loc[hz, "accuracy"]),
                                "lift": float(m_rg.loc[hz, "accuracy"]
                                              - n_rg.loc[hz, "accuracy"]),
                            } for hz in art.regime_horizons},
                        "sweep": {k: {kk: float(vv) for kk, vv in v.items()}
                                  for k, v in rsw.to_dict("index").items()},
                    }
            summary[name] = entry
        out = MODELS_DIR / "eval_summary.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f" wrote {out}")

    # -- optional outputs --------------------------------------------------
    if args.json_out:
        def enc(o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            raise TypeError(str(type(o)))
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(all_results, fh, indent=2, default=enc)
        print(f"\n wrote {args.json_out}")

    if args.csv:
        frames = []
        for name, per_split in all_results.items():
            for sp, r in per_split.items():
                for label, key in (("regression", "regression"),
                                   ("direction", "direction"),
                                   ("calibration", "calibration")):
                    df = r[key].copy()
                    df.insert(0, "split", sp)
                    df.insert(0, "model", name)
                    df.insert(0, "table", label)
                    frames.append(df)
        pd.concat(frames).to_csv(args.csv)
        print(f" wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
