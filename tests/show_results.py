#!/usr/bin/env python
"""
The ten report figures, drawn from this project's OWN recorded results.

    python tests/show_results.py                 # instant - no TensorFlow
    python tests/show_results.py --infer         # + the two panels that need a model run
    python tests/show_results.py --show          # open them on screen as well
    python tests/show_results.py --out DIR --dpi 200

WHAT IT DRAWS
     1  Actual vs predicted price / trend
     2  Confusion matrix, DOWN / NEUTRAL / UP
     3  Precision, recall and F1 per class
     4  Model vs baseline, macro-F1
     5  Macro-F1 across every prediction horizon, 15m..90m
     6  Market activity (QUIET vs ACTIVE) against its baseline
     7  Return regression, predicted vs actual
     8  Regression error against the predict-the-mean baseline
     9  Training and validation loss curves
    10  Feature importance / ablation

WHAT THIS IS, AND WHAT IT IS NOT
    A VIEWER. It re-measures nothing. Every number it draws is read back out of
    a file this project already wrote, and each panel records which file it came
    from. If a source is missing the panel says so instead of drawing something
    plausible - a fabricated chart is worse than a missing one, because it looks
    exactly like a real result.

    The results it reads are the ones this project reports:
      models/train_metrics.json      the 3-seed GRU ensemble, TEST split
      models/eval_summary.json       the same ensemble, val + test headline
      logs/03_evaluate.md            `src/evaluate.py --split both`, verbatim -
                                     the only recorded source of CONFUSION
                                     MATRICES, and of the val-split tables
      logs/02_train.md               the per-epoch fit log for all three seeds
      models/legacy_pre_regime/      the retired pre-regime GRU, kept as a
                                     baseline
      charts/actual_vs_predicted.png the canonical chart from src/plot_trend.py

    Two of the ten panels (1 and 7) need per-sample model outputs, which no file
    on disk holds. Those are drawn only with --infer, which runs the ensemble
    over the test split the same way src/evaluate.py does (~1 minute on CPU).
    Without it, panel 1 falls back to the existing chart if there is one, and
    panel 7 reports itself as not computed in this run.

    Panel 10 is different again: NO feature-importance or ablation experiment
    exists anywhere in this repository. The panel states that rather than
    inventing a ranking. See its docstring.

WHY THE LOG IS PARSED AS WELL AS THE JSON
    The confusion matrices are the one headline result that was never written to
    JSON, so the log is the only source for them. Parsing a human-readable log is
    fragile, so every parsed table is CROSS-CHECKED against the structured JSON
    before it is drawn, and a disagreement is reported as a WARNING rather than
    quietly plotted. If the log format ever changes, this script says so.

HOW TO READ ANY OF IT
    Every panel is a measurement on one historical period of one asset.
    Directional accuracy is not profitability: no fees, no slippage, no funding
    and no position sizing are modelled anywhere in this project. The direction
    head loses to always-NEUTRAL on accuracy at every horizon and wins on
    macro-F1; the regression head has no usable skill at all. Both are drawn
    deliberately rather than omitted, and no panel below should be quoted
    without the sentence next to it.

matplotlib and pandas are imported at module level. TensorFlow is imported
inside the one function that needs it, so the default run stays instant and
works on a machine without it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths - derived from this file, so the script runs from any directory
# ---------------------------------------------------------------------------

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_FINAL = PROJECT_ROOT / "data" / "final"
CHARTS_DIR = PROJECT_ROOT / "charts"
FIGURES_DIR = TESTS_DIR / "figures"

TRAIN_METRICS = MODELS_DIR / "train_metrics.json"
EVAL_SUMMARY = MODELS_DIR / "eval_summary.json"
EVALUATE_LOG = LOGS_DIR / "03_evaluate.md"
TRAIN_LOG = LOGS_DIR / "02_train.md"
LEGACY_DIR = MODELS_DIR / "legacy_pre_regime"
EXISTING_TREND_CHART = CHARTS_DIR / "actual_vs_predicted.png"

# The cache holds the per-sample outputs --infer produces, so a second run does
# not re-run the model. It is keyed on the weight files, and a changed model
# invalidates it rather than serving stale predictions.
INFER_CACHE = FIGURES_DIR / "_infer_cache"

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib                                                # noqa: E402
import matplotlib.pyplot as plt                                  # noqa: E402

# Class colours, deliberately matching src/plot_trend.py so the same class is
# the same colour in every figure in this project. They are NOT red/green only:
# ~8% of men cannot separate those, so lightness differs too and every panel is
# labelled.
C_DOWN, C_NEUTRAL, C_UP = "#c0392b", "#b8b8b8", "#1e8449"
C_MODEL, C_BASE, C_LEGACY = "#1f4e79", "#c0392b", "#7d3c98"
C_QUIET, C_ACTIVE = "#5b8db8", "#d98836"

DIRECTION_HORIZONS = ["15m", "30m", "45m", "60m", "75m", "90m"]
REGIME_HORIZONS = ["15m", "30m", "60m", "90m"]
CLASS_NAMES = ["DOWN", "NEUTRAL", "UP"]
REGIME_NAMES = ["QUIET", "ACTIVE"]

# Every figure carries this, because a crop of any one panel is otherwise very
# easy to read as a trading result.
HONESTY = ("One historical period, one asset. Directional accuracy is not "
           "profitability: no fees, slippage,\nfunding or position sizing are "
           "modelled. The regression head is expected to show no skill and is "
           "drawn that way on purpose.")


# ---------------------------------------------------------------------------
# Reading the recorded results
# ---------------------------------------------------------------------------

def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


CELL = re.compile(r"(\d[\d,]*)\s*\(\s*\d+%\)")
SPLIT_LINE = re.compile(r"^\s*(VALIDATION|TEST|TRAIN)\s+-\s+([\d,]+) samples\s+-")
# The log prints "VALIDATION"; the split is called "val" everywhere else in this
# project. Mapping it explicitly (rather than lower()-ing the matched word) keeps
# these keys identical to the ones train_metrics.json and eval_summary.json use.
SPLIT_KEYS = {"VALIDATION": "val", "TEST": "test", "TRAIN": "train"}
HZ_LINE = re.compile(r"^\s*(\d+m) horizon\s*$")
ROW_LINE = re.compile(r"^\s*(DOWN|NEUTRAL|UP|QUIET|ACTIVE)\s+(.*)$")
DATA_ROW = re.compile(r"^\s*(\d+m)\s+([-\d.eE,\s]+)$")


def _floats(text: str) -> list[float] | None:
    """All tokens on a table row as floats, or None if any token is not one."""
    out = []
    for tok in text.replace(",", "").split():
        try:
            out.append(float(tok))
        except ValueError:
            return None
    return out or None


def parse_evaluate_log(path: Path) -> dict:
    """
    Pull the recorded tables and confusion matrices out of logs/03_evaluate.md.

    Returns {split: {name: DataFrame}} plus {split: {"direction_confusion":
    {horizon: ndarray}, "regime_confusion": {...}, "n": int}}.

    The log is verbatim stdout from src/evaluate.py, so its shapes are fixed by
    `render_confusion` and `print_split_report` in that file. Anything this
    parser cannot read is simply absent from the result - the callers treat a
    missing key as "not available", never as zero.
    """
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    out: dict = {}
    split = None
    block = None            # which confusion section we are inside, if any
    horizon = None
    rows: list[list[int]] = []

    def flush() -> None:
        """Store the rows collected for the current horizon, if complete."""
        if not (split and block and horizon and rows):
            return
        want = 2 if block == "regime" else 3
        if len(rows) == want:
            key = f"{block}_confusion"
            out.setdefault(split, {}).setdefault(key, {})[horizon] = \
                np.array(rows, dtype=np.int64)

    for i, line in enumerate(lines):
        m = SPLIT_LINE.match(line)
        if m:
            flush()
            rows, horizon, block = [], None, None
            split = SPLIT_KEYS[m.group(1)]
            out.setdefault(split, {})
            out[split]["n"] = int(m.group(2).replace(",", ""))
            continue

        if line.strip().startswith("REGIME CONFUSION"):
            flush()
            block, horizon, rows = "regime", None, []
            continue
        if line.strip().startswith("CONFUSION") and "rows = actual" in line:
            flush()
            block, horizon, rows = "direction", None, []
            continue

        if block is not None:
            m = HZ_LINE.match(line)
            if m:
                flush()
                horizon, rows = m.group(1), []
                continue
            m = ROW_LINE.match(line)
            if m and horizon:
                counts = [int(c.replace(",", "")) for c in CELL.findall(m.group(2))]
                if counts:
                    rows.append(counts)
                    continue

        # A table: a header naming the columns, then one row per horizon.
        toks = line.split()
        if split and len(toks) >= 3 and not any(c.isdigit() for c in toks[0]):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            if DATA_ROW.match(nxt):
                names = toks
                data, j = {}, i + 1
                while j < len(lines):
                    m = DATA_ROW.match(lines[j])
                    if not m:
                        break
                    vals = _floats(m.group(2))
                    if vals is None or len(vals) != len(names):
                        break
                    data[m.group(1)] = vals
                    j += 1
                if len(data) >= 2:
                    frame = pd.DataFrame(data, index=names).T
                    frame.index.name = "horizon"
                    out[split][_table_key(names)] = frame
    flush()
    return out


def _table_key(names: list[str]) -> str:
    """A stable name for a parsed table, from its header columns."""
    if "naive_mean_MAE" in names:
        return "regression"
    if "majority_macroF1" in names:
        return "direction_vs_baseline"
    if "base_rate" in names:
        return "regime_vs_baseline"
    if any(n.startswith("DOWN_") for n in names):
        return "direction_per_class"
    if any(n.startswith("QUIET_") for n in names):
        return "regime_per_class"
    return "table_" + "_".join(names[:3])


EPOCH_HEAD = re.compile(r"^\s*Epoch (\d+)/(\d+)\s*$")
EPOCH_METRICS = re.compile(r"^\s*\d+/\d+\s+-\s+")
# Keys may contain digits - `val_direction_macro_f1`, `volatility_16` - so the
# class allows them after the first character. `[A-Za-z_]+` silently dropped
# every *_f1 column, which is the one the checkpoint selection is built on.
KV = re.compile(r"([A-Za-z_][A-Za-z_0-9]*):\s*(-?[\d.]+(?:e-?\d+)?)")
SEED_HEAD = re.compile(r"TRAINING\s+(\w+)\s+\((gru_s\d+)\)\s+seed\s+(\d+)")
BEST_EPOCH = re.compile(r"best epoch: (\d+)")
DURATION = re.compile(r"^(\S+): (\d+) epoch\(s\) in ([\d.]+) (min|s)")


def parse_train_log(path: Path) -> dict:
    """
    Per-epoch metrics for every seed in logs/02_train.md.

    Returns {seed_label: {"epochs": DataFrame, "tag": str, "best_restored":
    int|None, "seconds": float|None}}.

    Keras writes one long `key: value` line per epoch; the ReduceLROnPlateau and
    early-stopping notices are separate lines and are matched deliberately, so
    the restored best epoch can be checked against argmax(val_selection_score)
    rather than asserted.
    """
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    runs: dict = {}
    current = None
    epoch = None

    for line in lines:
        m = SEED_HEAD.search(line)
        if m:
            current = m.group(2)
            runs[current] = {"epochs": [], "tag": m.group(1),
                             "best_restored": None, "minutes": None}
            epoch = None
            continue
        if current is None:
            continue

        m = EPOCH_HEAD.match(line)
        if m:
            epoch = int(m.group(1))
            continue

        if EPOCH_METRICS.match(line):
            vals = {k: float(v) for k, v in KV.findall(line)}
            if vals:
                vals["epoch"] = epoch
                runs[current]["epochs"].append(vals)
            continue

        m = BEST_EPOCH.search(line)
        if m:
            runs[current]["best_restored"] = int(m.group(1))
            continue

        m = DURATION.match(line)
        if m and m.group(1) == current:
            runs[current]["minutes"] = (float(m.group(3)) if m.group(4) == "min"
                                        else float(m.group(3)) / 60.0)

    for name, run in runs.items():
        if run["epochs"]:
            run["epochs"] = pd.DataFrame(run["epochs"]).set_index("epoch")
    return {k: v for k, v in runs.items() if len(v["epochs"])}


def load_legacy_baseline() -> pd.DataFrame | None:
    """
    The retired pre-regime GRU, from models/legacy_pre_regime/eval.csv.

    It was evaluated on the SAME val/test splits (46,824 / 46,686 samples), so
    the split is comparable - but it is an older architecture without the regime
    head and without the selection-metric checkpointing. It is drawn as a
    labelled reference point, never as the current model.
    """
    path = LEGACY_DIR / "eval.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "macro_F1" not in df.columns or "table" not in df.columns:
        return None
    d = df[(df["table"] == "direction") &
           (df["model"] == "gru_multitask.keras")].copy()
    if d.empty:
        return None
    d = d.rename(columns={d.columns[0]: "horizon"})
    return d[["horizon", "split", "macro_F1"]]


# ---------------------------------------------------------------------------
# Cross-checks - a parsed number that disagrees with the JSON is a bug, and it
# gets said out loud rather than plotted
# ---------------------------------------------------------------------------

def check_confusion_against_metrics(parsed: dict, tm: dict | None) -> list[str]:
    """
    Recompute precision/recall from each parsed TEST confusion matrix and
    compare with the per-class metrics recorded in train_metrics.json.

    This is what makes it safe to draw a confusion matrix parsed out of a text
    log: if the parse were off by a column or a row, the marginals would not
    reproduce the JSON, and the difference would show up here.
    """
    notes: list[str] = []
    if not tm:
        return ["train_metrics.json missing - confusion matrices are NOT "
                "cross-checked"]
    test = parsed.get("test", {})
    for kind, key, names, metrics in (
            ("direction", "direction_confusion", CLASS_NAMES,
             tm.get("test_direction")),
            ("regime", "regime_confusion", REGIME_NAMES, tm.get("test_regime"))):
        cms = test.get(key)
        if not cms or not metrics:
            notes.append(f"{kind}: no parsed confusion to check")
            continue
        worst = 0.0
        for hz, cm in cms.items():
            for c, name in enumerate(names):
                row, col = cm[c].sum(), cm[:, c].sum()
                rec = cm[c, c] / row if row else 0.0
                prec = cm[c, c] / col if col else 0.0
                for label, got in ((f"{name}_rec", rec), (f"{name}_prec", prec)):
                    want = metrics.get(label, {}).get(hz)
                    if want is not None:
                        worst = max(worst, abs(got - want))
        verdict = "agrees" if worst < 5e-4 else "DISAGREES"
        notes.append(f"{kind} confusion vs train_metrics.json: {verdict} "
                     f"(max |diff| {worst:.5f})")
    return notes


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def figure(title: str, sub: str, source: str, size=(14, 8)) -> plt.Figure:
    fig = plt.figure(figsize=size)
    fig.suptitle(title, fontsize=13, y=0.975)
    # The subtitle must clear the title by a real distance, not by a fixed
    # FRACTION of the figure: 0.037 of an 11-inch figure is 29 pt, but the same
    # fraction of a 5.6-inch figure is only 15 pt -- less than one 13 pt line --
    # so the two overlap. Use whichever gap is larger.
    gap = max(0.037, 19.0 / (72.0 * size[1]))
    fig.text(0.5, 0.975 - gap, sub, ha="center", fontsize=9.5, color="#333")
    fig.text(0.5, 0.012, source + "\n" + HONESTY, ha="center", fontsize=7.5,
             color="#666")
    return fig


def unavailable(title: str, lines: list[str], source: str, *,
                status: str = "NOT DRAWN IN THIS RUN") -> tuple[plt.Figure, str]:
    """
    A panel that says why it has nothing to draw. Never a guess.

    Returns (figure, provenance) like every other panel, so a caller can return
    it from anywhere without special-casing the shape.

    ``status`` is the red line under the title -- "not drawn in this run" for a
    panel that a run of this script could produce, and something stronger where
    no run ever could.

    The card is laid out in INCHES from the top edge and the figure is sized
    from the line count, because a fixed fraction per line cannot be right for
    both a four-line card and a twenty-five-line one: the short one floats in
    white space and the tall one runs off the bottom.
    """
    n = len(lines)
    step = 0.215                            # inches per line at fontsize 10
    height = max(4.6, 1.55 + step * n + 0.80)
    fig = plt.figure(figsize=(14, height))
    fig.text(0.5, 1 - 0.55 / height, title, ha="center", fontsize=16,
             color="#333")
    fig.text(0.5, 1 - 1.08 / height, status, ha="center",
             fontsize=11, color="#c0392b", weight="bold")
    body_top, body_bot = 1.45, height - 0.80
    start = body_top + max(0.0, (body_bot - body_top - step * (n - 1)) / 2)
    for i, line in enumerate(lines):
        fig.text(0.5, 1 - (start + step * i) / height, line, ha="center",
                 va="top", fontsize=10, color="#444", family="monospace")
    fig.text(0.5, 0.45 / height, source, ha="center", va="bottom", fontsize=8,
             color="#777")
    return fig, f"not available ({source})"


def style(ax, grid: bool = True) -> None:
    if grid:
        ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _lift_axes_clear_of_footer(fig: plt.Figure, pad_px: float = 9.0) -> float:
    """
    Push the axes up until nothing of them touches the source/honesty block.

    The footer is anchored at the BOTTOM of the figure and grows UPWARD, so how
    far it reaches depends on how many lines it has and how tall the figure is --
    neither of which a panel's hard-coded ``bottom=`` can know. A fixed bottom
    that looks right on a 14x11 figure overlaps on a 14x5.6 one, which is
    exactly what kept happening. So measure the overlap after layout and take
    exactly that much off the bottom of every axes, instead of guessing again.

    Returns the total fraction of figure height that was taken.
    """
    footer = next((t for t in fig.texts if t.get_text().startswith("source: ")),
                  None)
    if footer is None:
        return 0.0
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    taken = 0.0
    for _ in range(5):
        footer_top = footer.get_window_extent(r).y1 + pad_px
        boxes = [ax.get_tightbbox(r) for ax in fig.axes
                 if ax.get_visible() and ax.get_position().height > 0]
        if not boxes:
            return taken
        lowest = min(b.y0 for b in boxes)
        if lowest >= footer_top:
            return taken
        deficit = (footer_top - lowest) / fig.bbox.height
        for ax in fig.axes:
            p = ax.get_position()
            if p.height > 0:
                ax.set_position([p.x0, p.y0 + deficit, p.width,
                                 p.height - deficit])
        taken += deficit
        fig.canvas.draw()
    return taken


def save(fig: plt.Figure, path: Path, dpi: int) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    _lift_axes_clear_of_footer(fig)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path.stat().st_size


def show_path(path: Path) -> str:
    """
    A path for the console: project-relative when it is inside the project,
    absolute when it is not. Plain ``.relative_to`` raised on any --out that
    pointed outside the tree, which took the whole run down with it.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


# ---------------------------------------------------------------------------
# 1. Actual vs predicted price / trend
# ---------------------------------------------------------------------------

def panel_actual_vs_predicted(infer: bool, args) -> tuple[plt.Figure, str]:
    """
    Price with the ACTUAL and PREDICTED direction bands underneath it.

    Needs per-sample outputs, so --infer computes them; otherwise the chart
    src/plot_trend.py already produced is embedded, if it is there. The bands
    are what make the NEUTRAL bias visible directly instead of having to be read
    off a table - the predicted band stays grey far more often than the actual
    one.
    """
    if infer:
        preds = infer_test_split(args)
        if preds is None:
            return unavailable(
                "Actual vs predicted price / trend",
                ["--infer was given but the model run did not produce outputs."],
                "source: none")
        sys.path.insert(0, str(SRC_DIR))
        from plot_trend import (break_at_gaps, draw_bands,  # noqa: PLC0415
                               load_closes)

        ts = preds["ts"]
        hi = DIRECTION_HORIZONS.index(args.horizon)
        actual = preds["d"][:, hi].astype(int)
        pred = preds["probs"][:, hi, :].argmax(axis=1)
        sel = np.arange(max(0, len(ts) - args.last), len(ts))

        closes_map = load_closes()
        closes = (np.array([closes_map.get(int(x), np.nan) for x in ts[sel]])
                  if closes_map else np.full(len(sel), np.nan))

        hits = int((pred[sel] == actual[sel]).sum())
        share = float((pred[sel] == 1).mean())
        fig = figure(
            f"Actual vs predicted trend  -  test split  -  {args.horizon}",
            f"last {len(sel):,} decision candles   |   "
            f"whole test split accuracy "
            f"{np.mean(pred == actual):.4f}",
            f"source: live model run in this script (--infer), "
            f"{', '.join(preds['members'])}",
            size=(15, 7.8))
        # Header band: the hit-rate line on the left, the colour key on the
        # right. Both used to be drawn INTO the plot area -- the key hung 1.6
        # axes-heights below the bottom band and the hit-rate line landed
        # between the two bands -- so they now sit in the empty strip under the
        # figure subtitle instead.
        gs = fig.add_gridspec(3, 1, height_ratios=[3.2, 0.55, 0.55], hspace=0.34,
                              top=0.855, bottom=0.135, left=0.075, right=0.97)
        ax_p = fig.add_subplot(gs[0])
        ax_a = fig.add_subplot(gs[1], sharex=ax_p)
        ax_d = fig.add_subplot(gs[2], sharex=ax_p)

        t, c = break_at_gaps(ts[sel], closes)
        ax_p.plot(t, c, lw=1.0, color="#1f4e79")
        ax_p.set_ylabel("close (USDT)", fontsize=9)
        ax_p.grid(alpha=0.25, lw=0.5)
        ax_p.tick_params(labelbottom=False)
        ax_p.set_title("price at each decision candle", fontsize=10, loc="left")

        draw_bands(ax_a, ts[sel], actual[sel], "ACTUAL")
        draw_bands(ax_d, ts[sel], pred[sel], "PREDICTED")
        # Only the bottom band may print dates: the middle one shares this x
        # axis, so leaving its labels on painted a second set of dates straight
        # through its own colour band.
        ax_a.tick_params(labelbottom=False)
        ax_d.set_xlabel("decision time (UTC)", fontsize=9)

        handles = [plt.Rectangle((0, 0), 1, 1, color=c)
                   for c in (C_DOWN, C_NEUTRAL, C_UP)]
        fig.legend(handles, ["DOWN", "NEUTRAL", "UP"], fontsize=8, ncol=3,
                   loc="lower right", bbox_to_anchor=(0.97, 0.875),
                   frameon=False)
        fig.text(0.075, 0.893,
                 f"{hits:,}/{len(sel):,} called correctly "
                 f"({100 * hits / len(sel):.1f}%)   |   the model said NEUTRAL "
                 f"on {100 * share:.1f}% of them",
                 fontsize=9, color="#444", ha="left", va="center")
        return fig, (f"live model run (--infer) over the test split, "
                     f"{', '.join(preds['members'])}")

    if EXISTING_TREND_CHART.exists():
        img = plt.imread(EXISTING_TREND_CHART)
        fig = plt.figure(figsize=(14, 9))
        ax = fig.add_subplot(111)
        ax.imshow(img)
        ax.axis("off")
        fig.suptitle("Actual vs predicted price / trend", fontsize=13, y=0.98)
        fig.text(0.5, 0.945,
                 "the chart src/plot_trend.py already produced for this "
                 "project - embedded, not regenerated", ha="center",
                 fontsize=9.5, color="#333")
        fig.text(0.5, 0.02,
                 f"source: {EXISTING_TREND_CHART.relative_to(PROJECT_ROOT)}  "
                 f"({EXISTING_TREND_CHART.stat().st_size / 1024:,.0f} KB)\n"
                 "run `python tests/show_results.py --infer` to redraw it here, "
                 "or `python src/plot_trend.py` for the full multi-panel "
                 "version.\n" + HONESTY,
                 ha="center", fontsize=7.5, color="#666")
        fig.subplots_adjust(top=0.94, bottom=0.13)
        return fig, (f"embedded from "
                     f"{EXISTING_TREND_CHART.relative_to(PROJECT_ROOT)} "
                     f"(produced by src/plot_trend.py)")

    return unavailable(
        "Actual vs predicted price / trend",
        ["No per-sample predictions on disk, and no chart at",
         "charts/actual_vs_predicted.png to fall back on.",
         "",
         "Run:  python tests/show_results.py --infer",
         "  or: python src/plot_trend.py"],
        "source: none - needs a model run")


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def panel_confusion(parsed: dict, split: str) -> tuple[plt.Figure | None, str]:
    """Counts and row percentages for all six horizons, test split by default."""
    cms = parsed.get(split, {}).get("direction_confusion")
    if not cms:
        return None, "logs/03_evaluate.md has no direction confusion matrix"
    horizons = [h for h in DIRECTION_HORIZONS if h in cms]
    fig = figure(
        f"Confusion matrix, direction head  -  {split} split",
        "rows = what actually happened, columns = what the model said   |   "
        "counts with the row percentage in brackets",
        f"source: logs/03_evaluate.md ({split}), the recorded stdout of "
        f"`python src/evaluate.py --split both`",
        size=(14, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.22, top=0.87,
                          bottom=0.10, left=0.07, right=0.97)
    for i, hz in enumerate(horizons):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        cm = cms[hz]
        row = cm.sum(axis=1, keepdims=True)
        pct = np.divide(cm, row, out=np.zeros_like(cm, dtype=float),
                        where=row > 0)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for p in range(3):
                ax.text(p, a, f"{cm[a, p]:,}\n{100 * pct[a, p]:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if pct[a, p] > 0.55 else "#222")
        ax.set_xticks(range(3), CLASS_NAMES, fontsize=8)
        ax.set_yticks(range(3), CLASS_NAMES, fontsize=8)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("actual", fontsize=8)
        rec = np.diag(pct)
        ax.set_title(f"{hz}   recall  " + "  ".join(
            f"{n[:1]}{100 * r:.0f}%" for n, r in zip(CLASS_NAMES, rec)),
            fontsize=9.5)
    return fig, (f"logs/03_evaluate.md ({split} section)")


# ---------------------------------------------------------------------------
# 3. Precision / recall / F1
# ---------------------------------------------------------------------------

def panel_prf(tm: dict | None, src_label: str) -> tuple[plt.Figure | None, str]:
    d = (tm or {}).get("test_direction")
    if not d:
        return None, "train_metrics.json has no test_direction block"
    metrics = ["prec", "rec", "f1"]
    fig = figure(
        "Precision, recall and F1 per class  -  direction head, test split",
        "DOWN and UP are the classes the model has to commit to; NEUTRAL is the "
        "one it can always fall back on, which is why its recall is high and "
        "theirs is not",
        "source: models/train_metrics.json (test_direction), the 3-seed GRU "
        "ensemble",
        size=(14, 6.2))
    gs = fig.add_gridspec(1, 3, wspace=0.22, top=0.775, bottom=0.15,
                          left=0.06, right=0.98)
    x = np.arange(len(DIRECTION_HORIZONS))
    width = 0.26
    for c, name in enumerate(CLASS_NAMES):
        ax = fig.add_subplot(gs[0, c])
        for k, metric in enumerate(metrics):
            vals = [d.get(f"{name}_{metric}", {}).get(hz, np.nan)
                    for hz in DIRECTION_HORIZONS]
            ax.bar(x + (k - 1) * width, vals, width, label=metric.upper(),
                   color=["#1f4e79", "#5b8db8", "#a8c4dc"][k],
                   edgecolor="white", lw=0.5)
            for xi, v in zip(x + (k - 1) * width, vals):
                if np.isfinite(v):
                    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center",
                            fontsize=6.5, rotation=90, color="#333")
        ax.set_xticks(x, DIRECTION_HORIZONS, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{name}", fontsize=11,
                     color=[C_DOWN, "#555", C_UP][c])
        if c == 0:
            ax.set_ylabel("score", fontsize=9)
        style(ax)
    fig.legend(*fig.axes[0].get_legend_handles_labels(), fontsize=9, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, 0.905), frameon=False)
    return fig, src_label


# ---------------------------------------------------------------------------
# 4. Model vs baseline
# ---------------------------------------------------------------------------

def panel_model_vs_baseline(parsed: dict, tm: dict | None,
                            legacy: pd.DataFrame | None,
                            split: str) -> tuple[plt.Figure | None, str]:
    """
    Macro-F1 against the always-NEUTRAL majority-class baseline, and against the
    retired pre-regime GRU where its CSV records one.

    Accuracy is deliberately NOT the metric here: always-NEUTRAL scores higher
    accuracy than the model at every horizon, and only macro-F1 exposes that the
    model is doing something the baseline is not.
    """
    table = parsed.get(split, {}).get("direction_vs_baseline")
    model = base = None
    if table is not None and {"model_macroF1", "majority_macroF1"} <= set(table.columns):
        model = [table.loc[h, "model_macroF1"] for h in DIRECTION_HORIZONS
                 if h in table.index]
        base = [table.loc[h, "majority_macroF1"] for h in DIRECTION_HORIZONS
                if h in table.index]
    elif tm and split == "test":
        model = [tm["test_direction"]["macro_F1"][h] for h in DIRECTION_HORIZONS]
        base = [tm["test_macro_f1_majority"][h] for h in DIRECTION_HORIZONS]
    if model is None:
        return None, "no recorded model-vs-baseline table for this split"

    x = np.arange(len(DIRECTION_HORIZONS))
    width = 0.27 if legacy is not None else 0.34
    fig = figure(
        f"Model vs baseline, macro-F1  -  {split} split",
        "macro-F1 is the fair comparison: always-NEUTRAL wins on ACCURACY at "
        "every horizon and loses on macro-F1, because it never calls a direction",
        f"source: logs/03_evaluate.md + models/train_metrics.json"
        + (", models/legacy_pre_regime/eval.csv" if legacy is not None else ""),
        size=(14, 6.2))
    ax = fig.add_subplot(111)
    ax.bar(x - width, model, width, label="GRU ensemble (this project)",
           color=C_MODEL, edgecolor="white", lw=0.5)
    ax.bar(x, base, width, label="always-NEUTRAL majority baseline",
           color=C_BASE, edgecolor="white", lw=0.5)
    if legacy is not None:
        sub = legacy[legacy["split"] == split]
        vals = [float(sub[sub["horizon"] == hz]["macro_F1"].iloc[0])
                if not sub[sub["horizon"] == hz].empty else np.nan
                for hz in DIRECTION_HORIZONS]
        ax.bar(x + width, vals, width,
               label="pre-regime GRU (retired, no regime head)",
               color=C_LEGACY, edgecolor="white", lw=0.5)
    for xi, (m, b) in enumerate(zip(model, base)):
        ax.text(xi - width if legacy is not None else xi - width, m + 0.008,
                f"{m:.3f}", ha="center", fontsize=7.5, color="#123")
        ax.text(xi, b + 0.008, f"{b:.3f}", ha="center", fontsize=7.5,
                color="#123")
    ax.set_xticks(x, DIRECTION_HORIZONS)
    ax.set_ylabel("macro-F1", fontsize=9)
    ax.set_ylim(0, max(model) * 1.32)
    ax.legend(fontsize=9, framealpha=0.9)
    style(ax)
    lift = np.mean([m - b for m, b in zip(model, base)])
    ax.set_title(f"mean macro-F1 lift over the baseline: {lift:+.4f}   "
                 f"(the model beats the baseline at "
                 f"{sum(m > b for m, b in zip(model, base))}/{len(model)} "
                 f"horizons)", fontsize=10, loc="left")
    return fig, (f"logs/03_evaluate.md + models/train_metrics.json ({split})")


# ---------------------------------------------------------------------------
# 5. Performance across prediction horizons
# ---------------------------------------------------------------------------

def panel_horizons(parsed: dict, tm: dict | None) -> tuple[plt.Figure | None, str]:
    fig = figure(
        "Macro-F1 across every prediction horizon",
        "the direction head is almost flat from 15m to 90m - there is no horizon "
        "at which it suddenly works - while the regime head rises with it",
        "source: models/train_metrics.json (test) + logs/03_evaluate.md (val)",
        size=(14, 6.2))
    ax = fig.add_subplot(111)
    x = np.arange(len(DIRECTION_HORIZONS))

    val = parsed.get("val", {}).get("direction_vs_baseline")
    if val is not None and "model_macroF1" in val.columns:
        ax.plot(x, [val.loc[h, "model_macroF1"] for h in DIRECTION_HORIZONS],
                "o-", color="#5b8db8", lw=1.4, ms=5, label="direction, val")
    if tm:
        ax.plot(x, [tm["test_direction"]["macro_F1"][h]
                    for h in DIRECTION_HORIZONS],
                "o-", color=C_MODEL, lw=1.8, ms=5.5, label="direction, test")
        ax.plot(x, [tm["test_macro_f1_majority"][h]
                    for h in DIRECTION_HORIZONS],
                "s--", color=C_BASE, lw=1.2, ms=4.5,
                label="always-NEUTRAL baseline, test")
        # The regime head predicts a SUBSET of the horizons (15m/30m/60m/90m), so
        # it is plotted at its own x positions rather than stretched onto all six.
        ax.plot([DIRECTION_HORIZONS.index(h) for h in REGIME_HORIZONS],
                [tm["test_regime"]["macro_F1"][h] for h in REGIME_HORIZONS],
                "^:", color=C_UP, lw=1.6, ms=5.5,
                label="regime QUIET/ACTIVE, test (its own horizons)")
    ax.set_xticks(x, DIRECTION_HORIZONS)
    ax.set_xlabel("prediction horizon", fontsize=9)
    ax.set_ylabel("macro-F1", fontsize=9)
    ax.set_ylim(0, 0.8)
    ax.axhline(1 / 3, color="#999", lw=0.8, ls=":",
               label="chance level for 3 balanced classes (0.333)")
    ax.legend(fontsize=9, framealpha=0.9, loc="center right")
    style(ax)
    return fig, ("models/train_metrics.json (test), logs/03_evaluate.md (val)")


# ---------------------------------------------------------------------------
# 6. Market activity (QUIET vs ACTIVE)
# ---------------------------------------------------------------------------

def panel_regime(tm: dict | None, parsed: dict, split: str) -> tuple[plt.Figure | None, str]:
    """Accuracy against the 50% baseline, plus the regime confusion matrices."""
    if not tm:
        return None, "train_metrics.json missing"
    acc = tm["test_regime"]["accuracy"]
    base = tm["test_regime_majority"]["accuracy"]
    per_class = tm["test_regime"]

    fig = figure(
        "Market activity classification: QUIET vs ACTIVE",
        "'will the next h candles be more volatile than the last 24 hours?'  "
        "Both classes are ~50/50 by construction, so 0.50 IS the baseline and "
        "the lift is real",
        "source: models/train_metrics.json (test_regime, test_regime_majority) "
        "+ logs/03_evaluate.md (regime confusion)",
        size=(14, 8.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1.0], hspace=0.48,
                          wspace=0.30, top=0.86, bottom=0.11, left=0.07,
                          right=0.97)

    ax = fig.add_subplot(gs[0, :2])
    x = np.arange(len(REGIME_HORIZONS))
    ax.bar(x - 0.19, [acc[h] for h in REGIME_HORIZONS], 0.38,
           label="model accuracy", color=C_MODEL, edgecolor="white", lw=0.5)
    ax.bar(x + 0.19, [base[h] for h in REGIME_HORIZONS], 0.38,
           label="majority / 50% baseline", color=C_BASE, edgecolor="white",
           lw=0.5)
    for xi, h in enumerate(REGIME_HORIZONS):
        ax.text(xi - 0.19, acc[h] + 0.012, f"{acc[h]:.3f}", ha="center",
                fontsize=8)
        ax.text(xi, max(acc[h], base[h]) + 0.055,
                f"{acc[h] - base[h]:+.3f}", ha="center", fontsize=8,
                color="#1e8449", weight="bold")
    ax.set_xticks(x, REGIME_HORIZONS)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.set_title("accuracy vs baseline, test split", fontsize=10, loc="left")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper left")
    style(ax)

    ax = fig.add_subplot(gs[0, 2:])
    width = 0.13
    for k, (metric, shade) in enumerate((("prec", "#1f4e79"), ("rec", "#5b8db8"),
                                         ("f1", "#a8c4dc"))):
        vals = []
        for hz in REGIME_HORIZONS:
            for name in REGIME_NAMES:
                vals.append(per_class[f"{name}_{metric}"][hz])
        offs = [i + (k - 1) * width for i in range(len(vals))]
        ax.bar(offs, vals, width, color=shade, edgecolor="white", lw=0.4,
               label=metric.upper())
    ax.set_xticks([i + 0.15 for i in range(0, len(REGIME_HORIZONS) * 2, 2)],
                  REGIME_HORIZONS, fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score", fontsize=9)
    ax.set_title("per-class precision / recall / F1   "
                 "(pairs are QUIET then ACTIVE)", fontsize=10, loc="left")
    ax.legend(fontsize=8, ncol=3, framealpha=0.9)
    style(ax)

    cms = parsed.get(split, {}).get("regime_confusion")
    for i, hz in enumerate(REGIME_HORIZONS):
        ax = fig.add_subplot(gs[1, i])
        if not cms or hz not in cms:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{hz}\nconfusion not in the log", ha="center",
                    va="center", fontsize=9, color="#888")
            continue
        cm = cms[hz]
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for a in range(2):
            for p in range(2):
                ax.text(p, a, f"{cm[a, p]:,}\n{100 * pct[a, p]:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if pct[a, p] > 0.55 else "#222")
        ax.set_xticks(range(2), REGIME_NAMES, fontsize=8)
        ax.set_yticks(range(2), REGIME_NAMES, fontsize=8)
        ax.set_title(f"{hz}", fontsize=10)
    return fig, "models/train_metrics.json + logs/03_evaluate.md (regime)"


# ---------------------------------------------------------------------------
# 7. Return regression, predicted vs actual
# ---------------------------------------------------------------------------

def panel_regression_scatter(infer: bool, args) -> tuple[plt.Figure, str]:
    """
    Predicted against actual return, one panel per horizon.

    A vertical smear around x=0 means the head is not predicting - it is
    emitting something close to the mean and letting the spread come from the
    data. That is exactly what the recorded correlation of 0.001-0.012 says, and
    it is drawn rather than omitted.
    """
    if not infer:
        return unavailable(
            "Return regression: predicted vs actual",
            ["A scatter needs per-sample predictions, which no file on disk "
             "holds.",
             "",
             "Run:  python tests/show_results.py --infer",
             "",
             "The recorded summary of the same thing is in the other panels:",
             "correlation 0.012 (15m) down to -0.005 (90m), and worse MAE than",
             "predicting the mean at every horizon."],
            "source: none - needs a model run")

    preds = infer_test_split(args)
    if preds is None:
        return unavailable("Return regression: predicted vs actual",
                           ["--infer ran but produced no outputs."],
                           "source: none")
    y, ret = preds["y"], preds["returns"]
    fig = figure(
        "Return regression: predicted vs actual  -  test split",
        "a vertical smear at x=0 means the head is not predicting at all - it "
        "emits something near the mean and the spread is all data",
        f"source: live model run in this script (--infer), "
        f"{', '.join(preds['members'])}",
        size=(14, 8.6))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.24, top=0.87,
                          bottom=0.10, left=0.06, right=0.98)
    for i, hz in enumerate(DIRECTION_HORIZONS):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        p, a = ret[:, i], y[:, i]
        lim = float(np.nanpercentile(np.abs(np.concatenate([p, a])), 99.5))
        lim = max(lim, 1e-6)
        ax.hexbin(p, a, gridsize=40, bins="log", cmap="Blues", mincnt=1,
                  linewidths=0, extent=(-lim, lim, -lim, lim))
        ax.plot([-lim, lim], [-lim, lim], color=C_BASE, lw=1.2, ls="--",
                label="perfect prediction")
        ax.axhline(0, color="#666", lw=0.8, ls=":", label="predict-the-mean")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        r = (float(np.corrcoef(p, a)[0, 1])
             if np.std(p) > 0 and np.std(a) > 0 else float("nan"))
        mae = float(np.abs(p - a).mean())
        naive = float(np.abs(a).mean())
        ax.set_title(f"{hz}   r = {r:+.3f}   MAE {mae:.4f}% vs mean "
                     f"{naive:.4f}%", fontsize=9.5)
        if i == 0:
            ax.set_ylabel("actual return (%)", fontsize=9)
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        if i >= 3:
            ax.set_xlabel("predicted return (%)", fontsize=9)
        ax.grid(alpha=0.2, lw=0.5)
    return fig, (f"live model run (--infer) over the test split, "
                 f"{', '.join(preds['members'])}")


# ---------------------------------------------------------------------------
# 8. Regression against the mean-prediction baseline
# ---------------------------------------------------------------------------

def panel_regression_baseline(tm: dict | None, parsed: dict,
                              split: str) -> tuple[plt.Figure | None, str]:
    if not tm:
        return None, "train_metrics.json missing"
    reg, naive = tm["test_regression"], tm["test_mae_naive_zero"]
    fig = figure(
        "Regression error vs the predict-the-mean baseline",
        "MAE and RMSE in PERCENT RETURN. Predicting the mean is the bar a "
        "regression head has to clear to be worth anything, and this one clears "
        "it nowhere - which is the honest reading of the regression head",
        "source: models/train_metrics.json (test_regression, "
        "test_mae_naive_zero)",
        size=(14, 6.0))
    gs = fig.add_gridspec(1, 3, wspace=0.28, top=0.82, bottom=0.16,
                          left=0.06, right=0.98)
    x = np.arange(len(DIRECTION_HORIZONS))
    width = 0.34

    ax = fig.add_subplot(gs[0, 0])
    ax.bar(x - width / 2, [reg["MAE_%"][h] for h in DIRECTION_HORIZONS], width,
           label="model MAE", color=C_MODEL, edgecolor="white", lw=0.5)
    ax.bar(x + width / 2, [naive[h] for h in DIRECTION_HORIZONS], width,
           label="predict the mean", color=C_BASE, edgecolor="white", lw=0.5)
    ax.set_xticks(x, DIRECTION_HORIZONS, fontsize=8)
    ax.set_ylabel("MAE (% return)", fontsize=9)
    ax.set_title("MAE: lower is better", fontsize=10, loc="left")
    ax.legend(fontsize=8, framealpha=0.9)
    style(ax)

    ax = fig.add_subplot(gs[0, 1])
    ax.bar(x, [reg["RMSE_%"][h] for h in DIRECTION_HORIZONS], 0.5,
           color="#5b8db8", edgecolor="white", lw=0.5)
    ax.set_xticks(x, DIRECTION_HORIZONS, fontsize=8)
    ax.set_ylabel("RMSE (% return)", fontsize=9)
    ax.set_title("RMSE", fontsize=10, loc="left")
    style(ax)

    ax = fig.add_subplot(gs[0, 2])
    delta = [reg["MAE_%"][h] - naive[h] for h in DIRECTION_HORIZONS]
    ax.bar(x, delta, 0.5, color=[C_BASE if d > 0 else "#1e8449" for d in delta],
           edgecolor="white", lw=0.5)
    ax.axhline(0, color="#333", lw=1.0)
    for xi, d in enumerate(delta):
        ax.text(xi, d + 0.0004, f"{d:+.4f}", ha="center", fontsize=7,
                rotation=90, color="#333")
    ax.set_xticks(x, DIRECTION_HORIZONS, fontsize=8)
    ax.set_ylabel("model MAE - baseline MAE", fontsize=9)
    ax.set_title("above zero = worse than predicting the mean", fontsize=10,
                 loc="left")
    style(ax)
    worse = sum(d > 0 for d in delta)
    fig.text(0.5, 0.885,
             f"VERDICT: worse than predicting the mean at {worse}/"
             f"{len(delta)} horizons. The regression head has no usable skill, "
             f"and no number it emits should be read as a forecast.",
             ha="center", fontsize=10, color="#c0392b", weight="bold")
    return fig, "models/train_metrics.json (test)"


# ---------------------------------------------------------------------------
# 9. Training and validation loss curves
# ---------------------------------------------------------------------------

def panel_loss_curves(runs: dict) -> tuple[plt.Figure | None, str]:
    """
    Train vs validation loss for every seed, plus the validation selection score
    the checkpoint was actually chosen on.

    The gap between the two curves is the overfitting question: the train loss
    keeps falling while val_loss flattens almost immediately and then drifts up,
    which is why early stopping fires at epoch 10/11/16 and the best weights are
    restored from much earlier.
    """
    if not runs:
        return None, "logs/02_train.md has no per-epoch lines"
    seeds = sorted(runs)
    fig = figure(
        "Training and validation loss curves",
        "one row per seed   |   the train curve keeps falling while val_loss "
        "never meaningfully improves after the first epoch - early stopping is "
        "doing real work here",
        "source: logs/02_train.md (the per-epoch lines written by "
        "src/train.py)",
        size=(14, 11.2))
    gs = fig.add_gridspec(len(seeds), 2, hspace=0.42, wspace=0.20, top=0.895,
                          bottom=0.125, left=0.07, right=0.97)
    for r, seed in enumerate(seeds):
        df = runs[seed]["epochs"]
        restored = runs[seed].get("best_restored")
        best = int(df["val_selection_score"].idxmax())

        ax = fig.add_subplot(gs[r, 0])
        ax.plot(df.index, df["loss"], "o-", color=C_MODEL, ms=3.5, lw=1.4,
                label="train loss (total)")
        ax.plot(df.index, df["val_loss"], "s-", color=C_BASE, ms=3.5, lw=1.4,
                label="validation loss (total)")
        ax.axvline(best, color="#1e8449", lw=1.0, ls="--",
                   label=f"best epoch by val selection score ({best})")
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.set_ylabel("loss", fontsize=8.5)
        ax.set_title(f"{seed}   -   {len(df)} epochs, restored from epoch "
                     f"{restored}", fontsize=10, loc="left")
        ax.legend(fontsize=8, framealpha=0.9)
        style(ax)

        ax = fig.add_subplot(gs[r, 1])
        ax.plot(df.index, df["val_selection_score"], "o-", color="#111",
                ms=3.5, lw=1.6,
                label="val selection score (0.5*dir F1 + 0.5*regime acc)")
        ax.plot(df.index, df["val_direction_macro_f1"], "s-", color=C_MODEL,
                ms=3.0, lw=1.2, label="val direction macro-F1")
        ax.plot(df.index, df["val_regime_accuracy"], "^-", color=C_UP, ms=3.0,
                lw=1.2, label="val regime accuracy")
        ax.axvline(best, color="#1e8449", lw=1.0, ls="--")
        ax.set_xlabel("epoch", fontsize=8.5)
        ax.set_ylabel("score", fontsize=8.5)
        ax.set_ylim(0.35, 0.75)
        ax.set_title("what the checkpoint was actually chosen on", fontsize=10,
                     loc="left")
        ax.legend(fontsize=8, framealpha=0.9)
        style(ax)

    note = []
    for seed in seeds:
        df, restored = runs[seed]["epochs"], runs[seed].get("best_restored")
        best = int(df["val_selection_score"].idxmax())
        note.append(f"{seed}: argmax(val_selection_score) = epoch {best}, "
                    f"log says restored from {restored}"
                    + ("" if best == restored else "   <-- MISMATCH"))
    # The footer block (source line + the honesty note) is several lines tall,
    # so its TOP edge sits well above the y it is anchored at. Measure where it
    # actually landed and stack the note on top of it, rather than guessing a y
    # and colliding with it -- which is what a fixed y does here.
    fig.canvas.draw()
    footer = next(t for t in fig.texts if t.get_text().startswith("source: "))
    top = footer.get_window_extent(fig.canvas.get_renderer()).y1
    fig.text(0.5, (top + 11) / fig.bbox.height,
             "checkpoint self-check (the best epoch recomputed from the "
             "logged curve, against the epoch the log says was restored):\n"
             + "   |   ".join(note),
             ha="center", va="bottom", fontsize=8, color="#444",
             linespacing=1.6)
    return fig, "logs/02_train.md (per-epoch lines, all seeds)"


# ---------------------------------------------------------------------------
# 10. Feature importance / ablation
# ---------------------------------------------------------------------------

def panel_feature_importance() -> tuple[plt.Figure, str]:
    """
    NOT CONDUCTED. There is no feature-importance or ablation experiment
    anywhere in this repository.

    This panel does not invent a ranking. Nothing in the pipeline permutes a
    feature, drops a group, or reads a learned importance out of the GRU - a
    `grep -rn "feature_importances_|ablation"` over the whole tree returns
    nothing - so any bar chart here would be fiction. What it shows instead is
    the feature inventory an ablation would be run over, and the exact shape
    such an experiment would take.

    The only feature-level probing that WAS done lives in .scratch/exp_*.py: a
    set of XGBoost label-design probes on the VALIDATION split, used to decide
    which target was learnable at all. Those are experiment-selection probes,
    not importance measurements on the shipped model, and they are not drawn
    here as if they were.
    """
    groups = None
    path = DATA_FINAL / "feature_names.json"
    if path.exists():
        try:
            with open(path, encoding="utf-8") as fh:
                groups = json.load(fh).get("feature_groups")
        except Exception:
            groups = None

    body = [
        "A bar chart of 'which features matter most' would have to be invented:",
        "the GRU exposes no learned importance, and no feature was ever dropped,",
        "permuted or ablated anywhere in this repository.",
        "",
        "The feature inventory an ablation would run over "
        f"({sum(len(v) for v in groups.values()) if groups else 49} features):"
        if groups else
        "The feature inventory an ablation would run over (49 features):",
    ]
    if groups:
        for name, cols in groups.items():
            body.append(f"    {name:<12} {len(cols):>2}   "
                        f"{', '.join(cols[:4])}"
                        + (" ..." if len(cols) > 4 else ""))
    body += [
        "",
        "WHAT WOULD MAKE THIS A REAL PANEL",
        "  * leave-one-group-out: retrain once per feature group, report the",
        "    drop in val direction macro-F1 and val regime accuracy (the",
        "    selection score src/train.py already optimises), and",
        "  * permutation importance on the trained ensemble: shuffle one column",
        "    of the test windows and measure the fall in macro-F1 - cheap,",
        "    needs no retraining, and is the honest first cut.",
        "",
        "Each retrain costs ~15 min on CPU (logs/02_train.md), so a full",
        "leave-one-group-out over 11 groups is roughly 3 hours of compute.",
    ]
    return unavailable(
        "Feature importance / ablation",
        body,
        "source: (absence verified by grep over the whole tree - no "
        "'feature_importances_', no 'ablation', no experiment output).\n"
        "This project's rule is that a missing result is reported as missing "
        "rather than filled in with something plausible.",
        status="NOT CONDUCTED - nothing is drawn here on purpose")


# ---------------------------------------------------------------------------
# The one function that runs the model (only under --infer)
# ---------------------------------------------------------------------------

def _member_fingerprint(paths: list[Path]) -> str:
    return ";".join(f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}"
                    for p in paths)


_INFER_MEMO: dict | None = None


def infer_test_split(args, need_price: bool = False) -> dict | None:
    """
    Run the ensemble over the test split once, and cache the result.

    This is the ONLY place that touches TensorFlow, and it mirrors what
    src/evaluate.py does: the same Artefacts, the same predict_split, the same
    bundle-declared ensemble. Nothing here re-implements the model plumbing, so
    a change to how evaluation runs cannot silently diverge from this.

    The cache is keyed on the weight files' size and mtime, so a retrained model
    invalidates it instead of being drawn with stale predictions.

    The result is also memoised for the process, so the cross-check at the end
    of the run reuses the same predictions instead of loading them a third time.
    """
    global _INFER_MEMO
    if _INFER_MEMO is not None and not args.refresh:
        return _INFER_MEMO
    sys.path.insert(0, str(SRC_DIR))
    print("  loading the ensemble and running the test split (about a "
          "minute on CPU)...")
    import tensorflow as tf                                          # noqa: PLC0415
    from tensorflow import keras                                     # noqa: PLC0415

    from config import INFERENCE_BUNDLE                              # noqa: PLC0415
    from evaluate import EnsembleModel, load_keras_model, predict_split  # noqa: PLC0415
    from train import Artefacts                                      # noqa: PLC0415

    art = Artefacts()
    paths = sorted(MODELS_DIR.glob("*.keras"))
    members = [p for p in paths if not p.name.startswith("legacy")]
    if not members:
        print("  no .keras files in models/ - cannot run inference.")
        return None

    bundle = load_json(INFERENCE_BUNDLE) or {}
    declared = [MODELS_DIR / m["path"]
                for m in (bundle.get("ensemble") or {}).get("members") or []]
    declared = [p for p in declared if p.exists()]
    members = declared or members[:1]

    fingerprint = _member_fingerprint(members)
    cache = INFER_CACHE / f"test_{args.batch}.npz"
    if cache.exists() and not args.refresh:
        try:
            with np.load(cache, allow_pickle=False) as z:
                if str(z["fingerprint"]) == fingerprint:
                    print(f"  using cached predictions ({cache.name})")
                    _INFER_MEMO = {"returns": z["returns"], "probs": z["probs"],
                                   "y": z["y"], "d": z["d"], "ts": z["ts"],
                                   "members": [str(m) for m in z["members"]]}
                    return _INFER_MEMO
                print("  cache is stale (the weights changed) - re-running")
        except Exception as exc:
            print(f"  cache unreadable ({type(exc).__name__}) - re-running")

    model = (EnsembleModel(members) if len(members) > 1
             else load_keras_model(members[0]))
    returns, probs, y, d, _ = predict_split(model, art, keras, "test",
                                            args.batch)
    members = [p.name for p in members]
    INFER_CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, returns=returns, probs=probs, y=y, d=d,
                        ts=getattr(art, "ts_test"),
                        members=np.array(members), fingerprint=fingerprint)
    print(f"  {tf.__version__} on "
          f"{'GPU' if tf.config.list_physical_devices('GPU') else 'CPU'} - "
          f"cached to {cache.relative_to(PROJECT_ROOT)}")
    _INFER_MEMO = {"returns": returns, "probs": probs, "y": y, "d": d,
                   "ts": getattr(art, "ts_test"), "members": members}
    return _INFER_MEMO


def check_inference_against_metrics(tm: dict | None,
                                    preds: dict | None) -> list[str]:
    """
    Recompute the regression head's correlation and MAE from the live
    predictions and compare them with what models/train_metrics.json recorded.

    The JSON was written by src/evaluate.py; these numbers come from this script
    running the ensemble itself, through different code. Agreement to four
    decimals is therefore real evidence that the recorded metrics describe the
    weights that are actually in models/ -- rather than a run that has since
    been retrained, leaving the JSON describing a model nobody has any more.
    """
    if preds is None:
        return ["live inference vs train_metrics.json: not checked "
                "(no --infer this run)"]
    rec = (tm or {}).get("test_regression") or {}
    if not rec.get("corr"):
        return ["live inference vs train_metrics.json: the JSON has no "
                "test_regression block to compare against"]
    worst_r = worst_m = 0.0
    for i, hz in enumerate(DIRECTION_HORIZONS):
        p, a = preds["returns"][:, i], preds["y"][:, i]
        r = (float(np.corrcoef(p, a)[0, 1])
             if np.std(p) > 0 and np.std(a) > 0 else float("nan"))
        worst_r = max(worst_r, abs(r - float(rec["corr"][hz])))
        worst_m = max(worst_m, abs(float(np.abs(p - a).mean())
                                   - float(rec["MAE_%"][hz])))
    verdict = "agrees" if max(worst_r, worst_m) < 1e-6 else "DISAGREES"
    return [f"live inference vs train_metrics.json (test regression, 6 "
            f"horizons): {verdict}",
            f"    max |diff| in r: {worst_r:.2e}   "
            f"max |diff| in MAE %: {worst_m:.2e}"]


# ---------------------------------------------------------------------------
# Contact sheet
# ---------------------------------------------------------------------------

def contact_sheet(pngs: list[tuple[str, Path]], out: Path, dpi: int) -> None:
    cols, rows = 3, int(np.ceil(len(pngs) / 3))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.0, rows * 3.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, (title, path) in zip(np.ravel(axes), pngs):
        ax.imshow(plt.imread(path))
        ax.set_title(title, fontsize=9)
    fig.suptitle("All ten report figures", fontsize=14, y=0.995)
    fig.text(0.5, 0.005,
             "Each panel is also written as its own PNG next to this one. "
             "Sources and caveats are printed in the run's output.\n" + HONESTY,
             ha="center", fontsize=7.5, color="#666")
    fig.tight_layout(rect=(0, 0.035, 1, 0.975))
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="Draw the ten report figures from this project's recorded "
                    "results. Re-measures nothing.")
    p.add_argument("--infer", action="store_true",
                   help="also run the ensemble over the test split so panels 1 "
                        "and 7 can be drawn (~1 minute on CPU; cached)")
    p.add_argument("--refresh", action="store_true",
                   help="ignore the cached predictions and re-run")
    p.add_argument("--split", default="test", choices=["val", "test"],
                   help="split for the confusion and comparison panels")
    p.add_argument("--horizon", default="15m", choices=DIRECTION_HORIZONS,
                   help="horizon for panel 1 (default 15m)")
    p.add_argument("--last", type=int, default=500,
                   help="candles shown in panel 1 (default 500)")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--out", default=str(FIGURES_DIR),
                   help=f"output directory (default {FIGURES_DIR})")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--show", action="store_true",
                   help="also open the figures in a window")
    args = p.parse_args()

    # Resolve --out before anything prints a path: a relative --out made every
    # later `path.relative_to(PROJECT_ROOT)` raise, which lost the whole run.
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("REPORT FIGURES  -  drawn from this project's recorded results")
    print("=" * 78)
    print(f"project   {PROJECT_ROOT}")
    print(f"output    {out_dir}")
    print(f"inference {'ON  (running the ensemble)' if args.infer else 'OFF (recorded results only)'}")
    print()

    tm = load_json(TRAIN_METRICS)
    summary = load_json(EVAL_SUMMARY)
    parsed = parse_evaluate_log(EVALUATE_LOG)
    runs = parse_train_log(TRAIN_LOG)
    legacy = load_legacy_baseline()

    for label, ok, path in (
            ("train_metrics.json", tm is not None, TRAIN_METRICS),
            ("eval_summary.json", summary is not None, EVAL_SUMMARY),
            ("03_evaluate.md", bool(parsed), EVALUATE_LOG),
            ("02_train.md", bool(runs), TRAIN_LOG),
            ("legacy eval.csv", legacy is not None, LEGACY_DIR / "eval.csv")):
        print(f"  [{'ok ' if ok else 'MISSING'}] {label:<20} "
              f"{path.relative_to(PROJECT_ROOT)}")

    if parsed:
        for split in ("val", "test"):
            sec = parsed.get(split, {})
            print(f"  [ok ] 03_evaluate.md {split:<6} "
                  f"tables: {', '.join(sorted(k for k in sec if isinstance(sec[k], pd.DataFrame))) or 'none'}")
            print(f"        confusion: direction "
                  f"{len(sec.get('direction_confusion', {}))} horizons, "
                  f"regime {len(sec.get('regime_confusion', {}))} horizons")
    if runs:
        print(f"  [ok ] 02_train.md seeds: "
              + ", ".join(f"{s} ({len(r['epochs'])} epochs)" for s, r in sorted(runs.items())))
    print()

    print("CROSS-CHECKS  (a parsed number that disagrees with the JSON is a bug)")
    for note in check_confusion_against_metrics(parsed, tm):
        print(f"  {note}")
    if not args.infer:
        print("  live inference vs train_metrics.json: not checked "
              "(no --infer this run)")
    print()

    # -- every panel, in the order the report asks for --------------------
    panels: list[tuple[str, str, callable]] = [
        ("01_actual_vs_predicted",
         "1. Actual vs predicted price / trend", None),
        ("02_confusion_matrix",
         "2. Confusion matrix (DOWN / NEUTRAL / UP)", None),
        ("03_precision_recall_f1",
         "3. Precision, recall and F1 per class", None),
        ("04_model_vs_baseline",
         "4. Model vs baseline (macro-F1)", None),
        ("05_performance_across_horizons",
         "5. Macro-F1 across prediction horizons", None),
        ("06_market_activity",
         "6. Market activity (QUIET vs ACTIVE)", None),
        ("07_return_regression_scatter",
         "7. Return regression: predicted vs actual", None),
        ("08_regression_baseline",
         "8. Regression vs mean-prediction baseline", None),
        ("09_loss_curves",
         "9. Training and validation loss curves", None),
        ("10_feature_importance",
         "10. Feature importance / ablation", None),
    ]

    builders = {
        "01_actual_vs_predicted":
            lambda: panel_actual_vs_predicted(args.infer, args),
        "02_confusion_matrix":
            lambda: panel_confusion(parsed, args.split),
        "03_precision_recall_f1":
            lambda: panel_prf(
                tm, "models/train_metrics.json (test_direction)"),
        "04_model_vs_baseline":
            lambda: panel_model_vs_baseline(parsed, tm, legacy, args.split),
        "05_performance_across_horizons":
            lambda: panel_horizons(parsed, tm),
        "06_market_activity":
            lambda: panel_regime(tm, parsed, args.split),
        "07_return_regression_scatter":
            lambda: panel_regression_scatter(args.infer, args),
        "08_regression_baseline":
            lambda: panel_regression_baseline(tm, parsed, args.split),
        "09_loss_curves":
            lambda: panel_loss_curves(runs),
        "10_feature_importance":
            lambda: panel_feature_importance(),
    }

    written: list[tuple[str, Path]] = []
    report: list[tuple[str, str, str]] = []

    for key, title, _ in panels:
        print(f"drawing  {title}")
        try:
            fig, source = builders[key]()
        except Exception as exc:                      # noqa: BLE001
            print(f"  [error] {type(exc).__name__}: {exc}")
            fig, source = unavailable(title, [
                f"this panel raised {type(exc).__name__}: {exc}",
                "the other panels are unaffected."], "source: none")
        if fig is None:
            fig, note = unavailable(title, [f"no source available: {source}"],
                                    "source: none")
            source = f"{note} - {source}"
        path = out_dir / f"{key}.png"
        size = save(fig, path, args.dpi)
        print(f"  wrote {show_path(path)}  ({size / 1024:,.0f} KB)")
        written.append((title, path))
        report.append((title, source, f"{size / 1024:,.0f} KB"))
        if args.show:
            plt.show()
        plt.close(fig)

    sheet = out_dir / "00_all_results.png"
    contact_sheet(written, sheet, args.dpi)
    print(f"\nwrote {show_path(sheet)}  "
          f"({sheet.stat().st_size / 1024:,.0f} KB)")

    if args.infer:
        # Printed here rather than with the parse checks above, because the
        # predictions only exist once a panel has asked for them -- and running
        # the ensemble in the middle of that block printed the model load inside
        # the cross-check list. The memoised call below costs nothing.
        print()
        print("CROSS-CHECK  (this script's own inference vs the recorded metrics)")
        for note in check_inference_against_metrics(tm, infer_test_split(args)):
            print(f"  {note}" if not note.startswith("    ") else note)

    print()
    print("=" * 78)
    print("WHAT EACH PANEL CAME FROM")
    print("=" * 78)
    for title, source, size in report:
        print(f"  {title}")
        print(f"      {size:>10}   {source}")

    if not args.infer:
        print()
        print("=" * 78)
        print("NOT DRAWN - these need a model run")
        print("=" * 78)
        print("  panel 1 (unless a chart already exists in charts/) and panel 7")
        print("  need per-sample predictions, which no file on disk holds.")
        print("  Get them with:  python tests/show_results.py --infer")
        print("  (~1 minute on CPU; the result is cached for the next run)")

    print()
    print("Panel 10 is not an omission: no feature-importance or ablation")
    print("experiment exists in this repository, so there is nothing to draw.")
    print()
    print("Reminder: one historical period, one asset, and directional accuracy")
    print("is not profitability. Nothing above is a promise about the future.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
