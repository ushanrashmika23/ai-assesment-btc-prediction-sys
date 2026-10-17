"""
Actual vs predicted trend charts.

    python src/plot_trend.py                       # test split, 15m, last 400 candles
    python src/plot_trend.py --horizon 90m --last 800
    python src/plot_trend.py --split val --horizon 30m
    python src/plot_trend.py --from 2026-01-01 --to 2026-03-01
    python src/plot_trend.py --out charts/trend.png --dpi 200

WHAT IT DRAWS
    A single figure with two blocks, sharing the same honesty rules as the rest
    of the project:

    TOP     the price over the chosen window, then two colour bands underneath:
            the ACTUAL direction class per candle and the PREDICTED one. This
            is the "actual trend vs predicted trend" view - and because the
            bands sit on the same axis you can see the NEUTRAL bias directly,
            rather than reading it off a table.

    BOTTOM  three panels that quantify the match rather than assert it:
              * predicted vs actual return, with the identity line. A vertical
                smear at x=0 means the regression head is not predicting.
              * reliability: when the model says 70%, is it right 70% of the
                time? Deviation from the diagonal is miscalibration.
              * rolling directional accuracy over time, against the
                majority-class baseline - is the skill stable, or is it one
                good regime?

READ THIS BEFORE QUOTING THE CHART
    Every panel is a measurement on one historical period. Directional accuracy
    is not profitability: no fees, no slippage, no funding, no position sizing.
    The regression panel is EXPECTED to look bad - the measured correlation is
    0.01-0.03 - and it is drawn that way deliberately rather than omitted.

    The bands are plotted from the recorded labels, which use the FROZEN
    train-only thresholds. Class boundaries therefore do not move between
    splits, which is why the actual-band looks more balanced on train than on
    test. That drift is real and is the reason the labels are not re-thresholded
    per split.

TIMESTAMPS
    `ts_*.npy` is the DECISION time - the last candle inside the 64-candle
    window - not the time being predicted. Verified against the raw closes: for
    every horizon h, close[ts + h] / close[ts] - 1 reproduces y[:, i] exactly
    (312/312 sampled pairs). So a 15m call stamped 12:00 is settled by the
    12:15 candle.

matplotlib is imported here only. Nothing else in src/ needs it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib                                                # noqa: E402
matplotlib.use("Agg")                                            # headless-safe
import matplotlib.dates as mdates                                # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap       # noqa: E402
from matplotlib.gridspec import GridSpec                         # noqa: E402

from config import CLEAN_CANDLES_CSV, INFERENCE_BUNDLE, MODELS_DIR  # noqa: E402
from inference import Ensemble, resolve_model_paths              # noqa: E402
from train import Artefacts                                      # noqa: E402
from evaluate import predict_split                               # noqa: E402

CANDLE_MS = 15 * 60 * 1000

# Class colours. Deliberately NOT red/green alone: ~8% of men cannot separate
# them, so the bands also differ in lightness and the panels are labelled.
CLASS_COLORS = ["#c0392b", "#b8b8b8", "#1e8449"]     # DOWN, NEUTRAL, UP
CLASS_CMAP = ListedColormap(CLASS_COLORS)
CLASS_NORM = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)


# ---------------------------------------------------------------------------
# Price context
# ---------------------------------------------------------------------------

def load_closes() -> dict[int, float]:
    """
    Decision-time close lookup, keyed by epoch milliseconds.

    Keys are built through `as_unit("ms")` rather than `astype("int64")`: with
    pandas 3.x the latter returns the column's OWN unit, and this project's
    clean CSV is datetime64[us], so the naive version produced keys in seconds,
    matched nothing, and silently yielded an empty price line.
    """
    path = CLEAN_CANDLES_CSV
    if not path.exists():
        return {}
    clean = pd.read_csv(path, usecols=["timestamp", "close"])
    t = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")
    ms = pd.DatetimeIndex(t).as_unit("ms").astype("int64").to_numpy()
    ok = ~np.isnan(ms.astype(float))
    return dict(zip(ms[ok].tolist(), clean["close"].to_numpy()[ok].tolist()))


def break_at_gaps(t_ms: np.ndarray, values: np.ndarray):
    """
    Insert NaN wherever consecutive samples are more than one candle apart.

    The split has holes where windows crossed a data gap. Matplotlib would
    otherwise draw a straight line across a hole, which is exactly the kind of
    fabricated continuity this project refuses to produce.
    """
    t = np.asarray(t_ms, dtype=np.int64)
    v = np.asarray(values, dtype=float)
    out_t, out_v = [], []
    for i in range(len(t)):
        if i and (t[i] - t[i - 1]) > CANDLE_MS:
            out_t.append(t[i - 1] + CANDLE_MS)      # blank cell across the hole
            out_v.append(np.nan)
        out_t.append(t[i])
        out_v.append(v[i])
    return pd.to_datetime(np.array(out_t), unit="ms", utc=True), np.array(out_v)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def draw_bands(ax, t_ms, labels, title):
    """One colour band per sample, at its real position in time."""
    t = np.asarray(t_ms, dtype=np.int64)
    edges = np.append(t - CANDLE_MS / 2, t[-1] + CANDLE_MS / 2)
    x = pd.to_datetime(edges, unit="ms", utc=True)
    ax.pcolormesh(x, [0, 1], np.asarray(labels)[None, :],
                  cmap=CLASS_CMAP, norm=CLASS_NORM, shading="flat")
    ax.set_yticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=9)


def panel_price(ax, t_ms, closes, horizon_label):
    t, c = break_at_gaps(t_ms, closes)
    ax.plot(t, c, lw=1.0, color="#1f4e79")
    ax.set_ylabel("close (USDT)", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.tick_params(labelbottom=False)
    ax.set_title(f"BTCUSDT 15m  -  price at each decision candle, and the "
                 f"{horizon_label} direction it was called", fontsize=11,
                 loc="left")
    lo, hi = float(np.nanmin(c)), float(np.nanmax(c))
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0:
        ax.text(0.005, 0.94, f"range {lo:,.0f} - {hi:,.0f}",
                transform=ax.transAxes, fontsize=8, va="top", color="#444")


def panel_scatter(ax, pred, actual, horizon_label):
    ax.hexbin(pred, actual, gridsize=44, bins="log", cmap="Blues",
              mincnt=1, linewidths=0)
    lim = float(np.nanpercentile(np.abs(np.concatenate([pred, actual])), 99.5))
    lim = max(lim, 1e-6)
    ax.plot([-lim, lim], [-lim, lim], color="#c0392b", lw=1.0, ls="--",
            label="perfect prediction")
    ax.axhline(0, color="#666", lw=0.8, ls=":", label="predict-the-mean")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    r = (float(np.corrcoef(pred, actual)[0, 1])
         if np.std(pred) > 0 and np.std(actual) > 0 else float("nan"))
    ax.set_xlabel(f"predicted {horizon_label} return  (%)", fontsize=9)
    ax.set_ylabel("actual return  (%)", fontsize=9)
    ax.set_title(f"Regression head:  r = {r:+.3f}", fontsize=10, loc="left")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)


def panel_reliability(ax, probs, actual_dir, horizon_label):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == actual_dir).astype(float)
    edges = np.linspace(1 / 3, 1.0, 13)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.sum() < 30:
            continue
        xs.append(float(conf[m].mean()))
        ys.append(float(correct[m].mean()))
        ns.append(int(m.sum()))
    ax.plot([1 / 3, 1], [1 / 3, 1], color="#c0392b", lw=1.0, ls="--",
            label="perfectly calibrated")
    ax.plot(xs, ys, "o-", color="#1f4e79", ms=4, lw=1.2, label="this model")
    ax.axhline(float(correct.mean()), color="#666", lw=0.8, ls=":",
               label=f"overall {correct.mean():.3f}")
    ax.set_xlabel("stated confidence", fontsize=9)
    ax.set_ylabel("actual accuracy", fontsize=9)
    ax.set_title("Reliability: does 70% mean 70%?", fontsize=10, loc="left")
    ax.set_xlim(1 / 3, 1.0)
    ax.set_ylim(1 / 3, 1.0)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)
    if ns:
        ax.text(0.98, 0.04, f"largest bin n={max(ns):,}", transform=ax.transAxes,
                fontsize=7, ha="right", color="#444")


def panel_rolling(ax, t_ms, pred_dir, actual_dir, baseline, horizon_label,
                  window):
    correct = (pred_dir == actual_dir).astype(float)
    roll = pd.Series(correct).rolling(window, min_periods=window // 2).mean()
    t, v = break_at_gaps(t_ms, roll.to_numpy(dtype=float))
    ax.plot(t, v, lw=1.2, color="#1f4e79",
            label=f"rolling accuracy ({window} candles)")
    ax.axhline(baseline, color="#c0392b", lw=1.0, ls="--",
               label=f"always-NEUTRAL {baseline:.3f}")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.set_xlabel("decision time (UTC)", fontsize=9)
    ax.set_title(f"{horizon_label} direction, over time", fontsize=10, loc="left")
    # This panel spans the whole split inside a third of the figure width, so
    # the default locator packs in ~10 "YYYY-MM" labels that overprint each
    # other. Cap the count instead of guessing an interval.
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    for lbl in ax.get_xticklabels():
        lbl.set_fontsize(8)
    ax.legend(fontsize=7, loc="lower left", framealpha=0.9)
    ax.grid(alpha=0.2, lw=0.5)


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot actual vs predicted trend for a trained model.")
    p.add_argument("--model", default=None,
                   help="path to a .keras file (default: the same canonical "
                        "GRU the dashboard picks)")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--horizon", default="15m",
                   help="which horizon to analyse (default 15m)")
    p.add_argument("--last", type=int, default=400,
                   help="how many candles of the window to show (default 400 "
                        "= ~4 days). The bottom panels always use the whole "
                        "split.")
    p.add_argument("--from", dest="start", default=None, metavar="DATE",
                   help="window start, e.g. 2026-01-01 (overrides --last)")
    p.add_argument("--to", dest="end", default=None, metavar="DATE",
                   help="window end, e.g. 2026-03-01")
    p.add_argument("--rolling", type=int, default=1000,
                   help="rolling-accuracy window in candles (default 1000)")
    p.add_argument("--out", default="charts/actual_vs_predicted.png",
                   help="output PNG path")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--batch", type=int, default=1024)
    args = p.parse_args()

    import tensorflow as tf
    from tensorflow import keras

    available = sorted(q.resolve() for q in MODELS_DIR.glob("*.keras"))
    if not available:
        raise SystemExit(f"No trained model in {MODELS_DIR}. Train one first:\n"
                         f"  python src/train.py --arch gru")
    member_paths = None
    if args.model:
        model_path = Path(args.model).resolve()
        if not model_path.exists():
            raise SystemExit(f"No model at {model_path}")
    else:
        # The ensemble the bundle names, for the same reason evaluate.py uses
        # it: the reported numbers are about that model, so a chart of one
        # member would be a chart of something else.
        member_paths = resolve_model_paths(INFERENCE_BUNDLE) or None
        if member_paths:
            model_path = member_paths[0]
        else:
            def rank(m: Path) -> tuple:
                return ("multitask" not in m.name, "gru" not in m.name, m.name)
            model_path = min(available, key=rank)

    art = Artefacts()

    if args.horizon not in art.horizons:
        raise SystemExit(f"--horizon must be one of {', '.join(art.horizons)}")
    hi = art.horizons.index(args.horizon)

    print(f"model    {('ENSEMBLE of ' + str(len(member_paths)) + ' - '
                        + ', '.join(p.stem for p in member_paths))
                       if member_paths else model_path.name}")
    print(f"split    {args.split}   horizon {args.horizon}   "
          f"rolling {args.rolling}")

    # compile=False: the direction head uses a loss class defined inside
    # train.py, which Keras cannot rebuild at load time. Nothing here fits.
    if member_paths:
        model = Ensemble([keras.models.load_model(p, compile=False)
                          for p in member_paths])
    else:
        model = keras.models.load_model(model_path, compile=False)
    # The regime probabilities are not plotted here, but they are part of the
    # model's output and are unpacked so a future panel can use them.
    returns, probs, y, d, _ = predict_split(model, art, keras, args.split,
                                            args.batch)
    ts = getattr(art, f"ts_{args.split}")
    print(f"samples  {len(ts):,}  ({tf.__version__}, "
          f"{'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'})")

    pred_dir = probs.argmax(axis=2)[:, hi]
    actual_dir = d[:, hi].astype(int)
    pred_ret = returns[:, hi]
    actual_ret = y[:, hi]
    conf = probs[:, hi, :].max(axis=1)

    # Majority-class baseline, from the TRAIN split as always.
    vals, counts = np.unique(art.d_train[:, hi], return_counts=True)
    majority = int(vals[np.argmax(counts)])
    baseline = float((actual_dir == majority).mean())

    # -- which slice of time to show in the top block ----------------------
    t_all = pd.to_datetime(ts, unit="ms", utc=True)
    if args.start or args.end:
        lo = pd.Timestamp(args.start, tz="UTC") if args.start else t_all[0]
        hi_t = pd.Timestamp(args.end, tz="UTC") if args.end else t_all[-1]
        sel = np.where((t_all >= lo) & (t_all <= hi_t))[0]
        if len(sel) == 0:
            raise SystemExit(f"No samples between {lo} and {hi_t} in "
                             f"{args.split}.")
        sel = sel[:20000]
    else:
        sel = np.arange(max(0, len(ts) - args.last), len(ts))
    print(f"window   {t_all[sel[0]]:%Y-%m-%d %H:%M} -> "
          f"{t_all[sel[-1]]:%Y-%m-%d %H:%M} UTC  ({len(sel):,} candles)")
    if len(sel) > 1200:
        print(f"  [warn] {len(sel):,} candles in the window is past the point "
              f"where the colour bands\n"
              f"         are readable - they degrade to noise. The bottom "
              f"panels are unaffected\n"
              f"         (they use the whole split). Try --last 400 for a "
              f"legible band view.")

    closes_map = load_closes()
    if closes_map:
        closes = np.array([closes_map.get(int(x), np.nan) for x in ts[sel]])
        if np.isnan(closes).all():
            print("  [warn] no prices matched; the price panel will be empty.")
    else:
        closes = np.full(len(sel), np.nan)
        print("  [warn] data/processed/btcusdt_15m_clean.csv not found - "
              "price panel omitted.")

    # -- figure -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 11.5))
    gs = GridSpec(4, 3, figure=fig, height_ratios=[2.6, 0.40, 0.40, 2.6],
                  hspace=0.62, wspace=0.28)

    ax_price = fig.add_subplot(gs[0, :])
    ax_act = fig.add_subplot(gs[1, :], sharex=ax_price)
    ax_pred = fig.add_subplot(gs[2, :], sharex=ax_price)
    ax_sc = fig.add_subplot(gs[3, 0])
    ax_rel = fig.add_subplot(gs[3, 1])
    ax_roll = fig.add_subplot(gs[3, 2])

    panel_price(ax_price, ts[sel], closes, args.horizon)
    draw_bands(ax_act, ts[sel], actual_dir[sel], "ACTUAL")
    draw_bands(ax_pred, ts[sel], pred_dir[sel], "PREDICTED")

    hits = int((pred_dir[sel] == actual_dir[sel]).sum())
    # Short label here: the per-window hit rate goes in the subtitle, where it
    # cannot collide with the bottom row's titles.
    ax_pred.set_xlabel("decision time (UTC)", fontsize=9)
    for ax in (ax_act, ax_pred):
        ax.tick_params(labelbottom=False)
    ax_pred.tick_params(labelbottom=True)
    ax_pred.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_pred.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    for lbl in ax_pred.get_xticklabels():
        lbl.set_rotation(0)
        lbl.set_fontsize(8)

    panel_scatter(ax_sc, pred_ret, actual_ret, args.horizon)
    panel_reliability(ax_rel, probs[:, hi, :], actual_dir, args.horizon)
    panel_rolling(ax_roll, ts, pred_dir, actual_dir, baseline, args.horizon,
                  args.rolling)

    acc = float((pred_dir == actual_dir).mean())
    fig.suptitle(
        f"Actual vs predicted trend  -  {model_path.name}  -  "
        f"{args.split} split  -  {args.horizon} horizon",
        fontsize=13, y=0.975)
    fig.text(0.5, 0.945,
             f"Window: {len(sel):,} candles, {hits} called correctly "
             f"({100 * hits / len(sel):.1f}%)   |   "
             f"Whole split: {len(ts):,} candles, accuracy {acc:.4f} vs "
             f"always-NEUTRAL {baseline:.4f}",
             ha="center", fontsize=9, color="#333")
    fig.text(0.5, 0.925,
             "NOT a profitability claim - no fees, slippage, funding or "
             "position sizing are modelled.",
             ha="center", fontsize=8.5, color="#555")
    fig.text(0.5, 0.012,
             "Colour bands are the recorded DOWN/NEUTRAL/UP labels, built with "
             "the FROZEN train-only thresholds - the same ones on every split. "
             "The regression panel is expected to show no skill (measured "
             "r = 0.01-0.03); it is plotted rather than hidden.",
             ha="center", fontsize=7.5, color="#777")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"\nwrote {out}  ({out.stat().st_size / 1024:,.0f} KB)")

    # -- the same numbers, for the terminal ---------------------------------
    from sklearn.metrics import f1_score
    f1 = f1_score(actual_dir, pred_dir, average="macro", zero_division=0)
    f1_base = f1_score(actual_dir, np.full_like(actual_dir, majority),
                       average="macro", zero_division=0)
    print(f"\n{args.horizon} on the {args.split} split ({len(ts):,} candles)")
    print(f"  accuracy    {acc:.4f}   always-NEUTRAL {baseline:.4f}")
    print(f"  macro-F1    {f1:.4f}   always-NEUTRAL {f1_base:.4f}")
    print(f"  regression  r = {np.corrcoef(pred_ret, actual_ret)[0, 1]:+.4f}")
    print(f"              MAE {(np.abs(pred_ret - actual_ret)).mean():.6f}%  "
          f"vs predicting the mean {(np.abs(actual_ret)).mean():.6f}%")
    print(f"  confidence  mean {conf.mean():.4f}  max {conf.max():.4f}  "
          f"share above 0.50 {float((conf > 0.5).mean()):.1%}")
    print("\n  One test period, one asset. Not investment advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
